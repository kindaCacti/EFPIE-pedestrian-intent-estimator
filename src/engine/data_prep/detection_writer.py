"""COCO-plus-temporal writer and strict schema/split validation."""

from collections import Counter
import math
from pathlib import Path
import re

from .manifest import write_json


def validate_coco(data, require_temporal=True):
    for field in ("images", "annotations", "categories"):
        if field not in data:
            raise ValueError(f"COCO export is missing {field}")
    categories = {c["id"] for c in data["categories"]}
    if len(categories) != len(data["categories"]):
        raise ValueError("Duplicate COCO category IDs")
    images, sequence_frames, timelines = {}, set(), {}
    for image in data["images"]:
        if image["id"] in images:
            raise ValueError("Duplicate COCO image IDs")
        if image["width"] <= 0 or image["height"] <= 0:
            raise ValueError("COCO image dimensions must be positive")
        file_name = Path(image["file_name"])
        if file_name.is_absolute() or ".." in file_name.parts:
            raise ValueError("COCO file_name must be a safe relative image path")
        if require_temporal:
            for field in ("sequence_id", "frame_index", "timestamp_s"):
                if field not in image:
                    raise ValueError(f"Trajectory COCO requires image extension field {field}")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(image["sequence_id"])):
                raise ValueError("COCO sequence_id must be a safe directory name")
            if not isinstance(image["frame_index"], int):
                raise ValueError("COCO frame_index must be an integer")
            key = (image["sequence_id"], image["frame_index"])
            if key in sequence_frames or image["frame_index"] < 0 or not math.isfinite(image["timestamp_s"]):
                raise ValueError("Duplicate/invalid sequence frame or timestamp")
            sequence_frames.add(key)
            timelines.setdefault(image["sequence_id"], []).append((image["frame_index"], image["timestamp_s"]))
        images[image["id"]] = image
    for timeline in timelines.values():
        timeline.sort()
        if any(b[1] <= a[1] for a, b in zip(timeline, timeline[1:])):
            raise ValueError("Nonmonotonic COCO timestamps")
    annotations, track_frames = set(), set()
    for annotation in data["annotations"]:
        if annotation["id"] in annotations:
            raise ValueError("Duplicate annotation ID")
        annotations.add(annotation["id"])
        if annotation["image_id"] not in images or annotation["category_id"] not in categories:
            raise ValueError("Annotation references unknown image/category")
        image = images[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or not (x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= image["width"] + 1e-5 and y + h <= image["height"] + 1e-5):
            raise ValueError("Invalid COCO pixel xywh box")
        if annotation.get("iscrowd") != 0 or not math.isclose(annotation.get("area", -1), w * h, rel_tol=1e-5):
            raise ValueError("Invalid COCO area/iscrowd")
        if "score" in annotation and (not math.isfinite(annotation["score"]) or not 0 <= annotation["score"] <= 1):
            raise ValueError("COCO detection score must lie in [0, 1]")
        if require_temporal:
            if "track_id" not in annotation:
                raise ValueError("Trajectory COCO requires annotation extension field track_id")
            track = annotation["track_id"]
            if track is not None:
                if not str(track).startswith(image["sequence_id"] + ":"):
                    raise ValueError("track_id must be namespaced by sequence_id")
                key = (annotation["image_id"], track)
                if key in track_frames:
                    raise ValueError("Duplicate track identity in a frame")
                track_frames.add(key)
    return images


def validate_split_isolation(exports):
    seen = set()
    for data in exports:
        sequences = {i["sequence_id"] for i in data["images"]}
        if seen.intersection(sequences):
            raise ValueError("Sequence-level split leakage: sequence_id appears in multiple splits")
        seen.update(sequences)


class DetectionWriter:
    def __init__(self, root, category_mapping):
        self.root = Path(root)
        categories = [{"id": coco_id, "name": name, "supercategory": "pedestrian"} for _, (coco_id, name) in sorted(category_mapping.items())]
        self.mapping = category_mapping
        self.splits = {split: {"images": [], "annotations": [], "categories": categories} for split in ("train", "val", "test")}
        self.image_id = self.annotation_id = 0

    def add_frame(self, sequence, index, timestamp, size, file_name, detections):
        data = self.splits[sequence.split]
        self.image_id += 1
        data["images"].append({"id": self.image_id, "file_name": file_name,
                               "width": size[0], "height": size[1],
                               "sequence_id": sequence.sequence_id, "frame_index": index,
                               "timestamp_s": timestamp})
        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            self.annotation_id += 1
            data["annotations"].append({"id": self.annotation_id, "image_id": self.image_id,
                "category_id": self.mapping[det.class_id][0], "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                "score": det.score, "track_id": det.track_id})

    def finish(self, min_track_length=20):
        validate_split_isolation(list(self.splits.values()))
        counts = {}
        for split, data in self.splits.items():
            validate_coco(data)
            for image in data["images"]:
                if not (self.root / "images" / image["file_name"]).is_file():
                    raise ValueError(f"Missing prepared image {image['file_name']}")
            write_json(self.root / "annotations" / f"instances_{split}.json", data)
            tracks = Counter(a["track_id"] for a in data["annotations"] if a["track_id"] is not None)
            counts[split] = {"images": len(data["images"]), "annotations": len(data["annotations"]),
                             "tracks": len(tracks), "short_tracks": sum(n < min_track_length for n in tracks.values())}
        return counts
