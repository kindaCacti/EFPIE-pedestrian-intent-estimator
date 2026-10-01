"""Resolution-independent histories with explicit missing observations."""

import numpy as np


def box_to_point(box, image_size):
    x1, _, x2, y2 = box
    w, h = image_size
    return ((x1 + x2) / (2 * w), y2 / h)


def box_to_normalized_cxcywh(box, image_size):
    x1, y1, x2, y2 = box
    w, h = image_size
    return ((x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h)


class SequenceBuilder:
    def __init__(self, observation_frames=8, prediction_frames=12,
                 allow_padded_history=False, allow_observation_gaps=False,
                 point="bottom_center_normalized", frame_stride=1, **kwargs):
        if observation_frames <= 0 or prediction_frames <= 0 or frame_stride <= 0:
            raise ValueError("Observation/prediction horizons and frame_stride must be positive")
        if point != "bottom_center_normalized":
            raise ValueError("Only bottom_center_normalized coordinates are supported")
        self.observation_frames = observation_frames
        self.prediction_frames = prediction_frames
        self.allow_padding = allow_padded_history
        self.allow_gaps = allow_observation_gaps
        self.frame_stride = frame_stride

    def history_for(self, snapshot, track_id):
        frames = snapshot.frames[-self.observation_frames:]
        if not frames or (len(frames) < self.observation_frames and not self.allow_padding):
            return None
        if any(b.frame_index - a.frame_index != self.frame_stride for a, b in zip(frames, frames[1:])):
            return None  # a dropped frame is not silently compressed in time
        points, boxes, mask, sizes = [], [], [], []
        for frame in frames:
            detections = [d for d in frame.detections if d.track_id == track_id]
            if len(detections) > 1:
                raise ValueError("A track has multiple detections in one frame")
            det = detections[0] if detections else None
            points.append(box_to_point(det.box_xyxy, frame.image_size) if det else (0, 0))
            boxes.append(box_to_normalized_cxcywh(det.box_xyxy, frame.image_size) if det else (0, 0, 0, 0))
            mask.append(det is not None)
            sizes.append(frame.image_size)
        if not mask[-1] or not any(mask) or (not all(mask) and not self.allow_gaps):
            return None
        times = np.array([f.timestamp_s for f in frames], dtype=np.float64)
        deltas = np.diff(times, prepend=times[0]).astype(np.float32)
        pad = self.observation_frames - len(frames)
        return {
            "sequence_id": snapshot.sequence_id, "track_id": track_id,
            "observed_points": np.pad(np.asarray(points, np.float32), ((pad, 0), (0, 0))),
            "observed_boxes": np.pad(np.asarray(boxes, np.float32), ((pad, 0), (0, 0))),
            "observed_time_deltas": np.pad(deltas, (pad, 0)),
            "observed_mask": np.pad(np.asarray(mask, bool), (pad, 0)),
            "metadata": {"coordinate_system": "image_normalized", "box_format": "normalized_cxcywh",
                         "source_frame_index": frames[-1].frame_index,
                         "timestamp_s": frames[-1].timestamp_s, "image_sizes": sizes,
                         "prediction_dt_s": float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30},
        }
