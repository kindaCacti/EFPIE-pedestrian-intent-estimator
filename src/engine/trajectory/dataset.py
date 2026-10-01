"""Sliding track windows from completed COCO temporal artifacts only."""

from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..data_prep.detection_writer import validate_coco, validate_split_isolation
from ..data_prep.manifest import sha256_file
from ..temporal.sequence_builder import box_to_point, box_to_normalized_cxcywh


class TrajectoryDataset(Dataset):
    def __init__(self, prepared_dir, split="train", observation_frames=8, prediction_frames=12,
                 allow_observation_gaps=False, allow_padded_history=False,
                 require_visible_future=True, window_stride=1, ann_file=None,
                 evaluation_mode="detected_history", detected_history_dir=None,
                 history_match_iou=.3, history_only=False, **kwargs):
        if split not in {"train", "val", "test"} or min(observation_frames, prediction_frames, window_stride) <= 0:
            raise ValueError("Invalid split or non-positive trajectory horizon/window stride")
        if evaluation_mode not in {"ground_truth_history", "detected_history"}:
            raise ValueError("evaluation_mode must be ground_truth_history or detected_history")
        self.root = Path(prepared_dir)
        manifest = self.root / "manifests" / "prepared_dataset.json"
        if not manifest.is_file():
            raise ValueError("TrajectoryDataset requires a completed prepared_dataset.json; run the preparation tool first")
        with manifest.open() as stream:
            self.provenance = json.load(stream)
        if self.provenance.get("status") != "complete" or self.provenance.get("format_version") != 1:
            raise ValueError("Prepared dataset is incomplete or has an unsupported version")
        if self.provenance.get("coordinate_system") != "image_normalized":
            raise ValueError("This reader requires image_normalized coordinates")
        if evaluation_mode == "ground_truth_history" and self.provenance.get("label_provenance") == "detector_tracker_pseudo_labels":
            raise ValueError("ground_truth_history requires verified human annotations, not detector pseudo-labels")
        self.evaluation_mode = evaluation_mode
        self.observation_frames, self.prediction_frames = observation_frames, prediction_frames
        self.allow_gaps, self.allow_padding = allow_observation_gaps, allow_padded_history
        self.history_only = history_only
        self.frame_stride = int(self.provenance.get("frame_stride", 1))
        exports = {}
        for name in ("train", "val", "test"):
            path = Path(ann_file) if ann_file and name == split else self.root / "annotations" / f"instances_{name}.json"
            if not path.is_file():
                raise ValueError(f"Missing prepared split {path}")
            expected = self.provenance.get("annotation_fingerprints", {}).get(name)
            if expected and sha256_file(path) != expected:
                raise ValueError(f"Prepared annotations changed after publication: {path}")
            with path.open() as stream:
                exports[name] = json.load(stream)
            validate_coco(exports[name])
        validate_split_isolation(list(exports.values()))
        data = exports[split]
        images = {i["id"]: i for i in data["images"]}
        timelines = defaultdict(dict)
        for image in data["images"]:
            timelines[image["sequence_id"]][image["frame_index"]] = image
        tracks = defaultdict(dict)
        self.exclusion_counts = {"untracked_annotations": 0, "short_tracks": 0, "invalid_windows": 0}
        for annotation in data["annotations"]:
            track_id = annotation["track_id"]
            if track_id is None:
                self.exclusion_counts["untracked_annotations"] += 1
                continue
            image = images[annotation["image_id"]]
            key = (image["sequence_id"], track_id)
            x, y, w, h = annotation["bbox"]
            tracks[key][image["frame_index"]] = {
                "image": image, "box": (x, y, x + w, y + h),
                "visible": bool(annotation.get("visible", True)),
            }
        self.windows = []
        total = observation_frames + prediction_frames
        for (sid, tid), records in sorted(tracks.items()):
            first, last = min(records), max(records)
            if len(records) < (observation_frames if history_only else total) and not allow_padded_history:
                self.exclusion_counts["short_tracks"] += 1
                continue
            endpoints = range(first + (0 if allow_padded_history else (observation_frames - 1) * self.frame_stride), last - (0 if history_only else prediction_frames * self.frame_stride) + 1, window_stride * self.frame_stride)
            for end in endpoints:
                indices = [end + (i - observation_frames + 1) * self.frame_stride for i in range(total)]
                observed = indices[:observation_frames]
                future = indices[observation_frames:]
                pad = [i < first for i in observed]
                valid_obs = [i in records and records[i]["visible"] for i in observed]
                valid_future = [i in records and records[i]["visible"] for i in future]
                accepted = bool(valid_obs[-1]) and any(valid_obs)
                accepted &= allow_observation_gaps or all(v or (p and allow_padded_history) for v, p in zip(valid_obs, pad))
                if not history_only:
                    accepted &= all(valid_future) if require_visible_future else any(valid_future)
                accepted &= all(i in timelines[sid] for i in (observed if history_only else indices) if i >= first)
                if not accepted:
                    self.exclusion_counts["invalid_windows"] += 1
                    continue
                self.windows.append((sid, tid, records, timelines[sid], indices))
        self._detected_dataset, self._history_matches = None, {}
        if detected_history_dir is not None:
            if evaluation_mode != "detected_history" or self.provenance.get("label_provenance") != "human_ground_truth":
                raise ValueError("detected_history_dir requires human target annotations and detected_history evaluation mode")
            if not 0 < history_match_iou <= 1:
                raise ValueError("history_match_iou must lie in (0, 1]")
            self._detected_dataset = TrajectoryDataset(detected_history_dir, split=split,
                observation_frames=observation_frames, prediction_frames=prediction_frames,
                allow_observation_gaps=allow_observation_gaps, allow_padded_history=allow_padded_history,
                history_only=True)
            if self._detected_dataset.frame_stride != self.frame_stride:
                raise ValueError("Detected and labelled artifacts use different frame sampling strides")
            self._match_detected_histories(history_match_iou)

    def _match_detected_histories(self, threshold):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        from ..temporal.tracker import box_iou
        groups, detected_groups = defaultdict(list), defaultdict(list)
        for i, (sid, _, _, _, indices) in enumerate(self.windows):
            groups[(sid, indices[self.observation_frames - 1])].append(i)
        for j, (sid, _, _, _, indices) in enumerate(self._detected_dataset.windows):
            detected_groups[(sid, indices[self.observation_frames - 1])].append(j)
        matches = {}
        def normalized_box(window):
            _, _, records, timeline, indices = window
            frame = indices[self.observation_frames - 1]
            box = records[frame]["box"]
            image = timeline[frame]
            return (box[0] / image["width"], box[1] / image["height"], box[2] / image["width"], box[3] / image["height"])
        for key, targets in groups.items():
            candidates = detected_groups.get(key, [])
            if not candidates:
                continue
            cost = np.full((len(targets), len(candidates)), 1e6)
            for row, i in enumerate(targets):
                for col, j in enumerate(candidates):
                    gt = self.windows[i]
                    det = self._detected_dataset.windows[j]
                    if any(abs(gt[3][f]["timestamp_s"] - det[3][f]["timestamp_s"]) > 1e-5 for f in gt[4] if f in gt[3] and f in det[3]):
                        raise ValueError("Detected and labelled frame timestamps are not aligned")
                    iou = box_iou(normalized_box(gt), normalized_box(det))
                    if iou >= threshold:
                        cost[row, col] = 1 - iou
            rows, cols = linear_sum_assignment(np.concatenate([cost, np.full((len(targets), len(targets)), 2.)], axis=1))
            for row, col in zip(rows, cols):
                if col < len(candidates) and cost[row, col] < 2:
                    matches[targets[row]] = candidates[col]
        self.exclusion_counts["unmatched_ground_truth_windows"] = len(self.windows) - len(matches)
        old_windows = self.windows
        self.windows = [old_windows[i] for i in sorted(matches)]
        self._history_matches = {new: matches[old] for new, old in enumerate(sorted(matches))}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sid, tid, records, timeline, indices = self.windows[index]
        points, boxes, mask, sizes, timestamps = [], [], [], [], []
        fallback = timeline[min(timeline)]
        fps = self.provenance.get("source_fps", {}).get(sid, 30)
        for frame_index in indices:
            image = timeline.get(frame_index, fallback)
            record = records.get(frame_index)
            visible = record is not None and record["visible"]
            size = (image["width"], image["height"])
            points.append(box_to_point(record["box"], size) if visible else (0, 0))
            boxes.append(box_to_normalized_cxcywh(record["box"], size) if visible else (0, 0, 0, 0))
            mask.append(visible)
            sizes.append(size)
            timestamps.append(image["timestamp_s"] if frame_index in timeline else fallback["timestamp_s"] + (frame_index - fallback["frame_index"]) / fps)
        obs = self.observation_frames
        times = torch.tensor(timestamps, dtype=torch.float64)
        deltas = torch.diff(times[:obs], prepend=times[:1]).float()
        future_deltas = (times[obs:] - times[obs - 1]).float()
        sample = {"sequence_id": sid, "track_id": tid,
                "observed_points": torch.tensor(points[:obs], dtype=torch.float32),
                "observed_boxes": torch.tensor(boxes[:obs], dtype=torch.float32),
                "observed_time_deltas": deltas, "observed_mask": torch.tensor(mask[:obs]),
                "future_points": torch.tensor(points[obs:], dtype=torch.float32),
                "future_boxes": torch.tensor(boxes[obs:], dtype=torch.float32),
                "future_mask": torch.tensor(mask[obs:]), "future_time_deltas": future_deltas,
                "metadata": {"coordinate_system": "image_normalized", "box_format": "normalized_cxcywh",
                             "image_sizes": sizes, "source_frame_index": indices[obs - 1],
                             "timestamp_s": timestamps[obs - 1], "prediction_dt_s": self.frame_stride / fps,
                             "label_provenance": self.provenance.get("label_provenance"),
                             "evaluation_mode": self.evaluation_mode}}
        if self._detected_dataset is not None:
            detected = self._detected_dataset[self._history_matches[index]]
            for key in ("observed_points", "observed_boxes", "observed_mask", "observed_time_deltas"):
                sample[key] = detected[key]
            sample["metadata"]["detected_track_id"] = detected["track_id"]
            sample["metadata"]["image_sizes"][:obs] = detected["metadata"]["image_sizes"][:obs]
        return sample
