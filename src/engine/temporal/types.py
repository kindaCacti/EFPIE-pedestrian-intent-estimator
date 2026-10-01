"""Canonical image-space temporal records; this package does not import torch."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str | None = None
    track_id: str | int | None = None

    def __post_init__(self):
        for value in (self.box_xyxy, self.score, self.class_id, self.track_id):
            if type(value).__module__.startswith("torch"):
                raise TypeError("Normalize detector tensors before creating cache records")
        box = tuple(float(x) for x in self.box_xyxy)
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            raise ValueError("Detection requires four finite absolute xyxy coordinates")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Detection box must have positive width and height")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Detection score must be in [0, 1]")
        object.__setattr__(self, "box_xyxy", box)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "class_id", int(self.class_id))
        if self.class_name is not None and not isinstance(self.class_name, str):
            raise TypeError("class_name must be a string or None")
        if self.track_id is not None and not isinstance(self.track_id, (str, int)):
            raise TypeError("track_id must be an immutable sequence-namespaced string or integer")


@dataclass(frozen=True)
class FrameRecord:
    sequence_id: str
    frame_index: int
    timestamp_s: float
    image: np.ndarray | None
    image_size: tuple[int, int]
    detections: tuple[Detection, ...]

    def __post_init__(self):
        if not isinstance(self.sequence_id, str) or not isinstance(self.frame_index, int):
            raise TypeError("Frame sequence_id/index must be a string and integer")
        size = tuple(self.image_size)
        if not self.sequence_id or self.frame_index < 0 or not math.isfinite(self.timestamp_s):
            raise ValueError("Frame requires a sequence ID, non-negative index and finite timestamp")
        if len(size) != 2 or any(not isinstance(x, (int, np.integer)) or x <= 0 for x in size):
            raise ValueError("image_size must be positive integer (width, height)")
        detections = tuple(self.detections)
        if any(not isinstance(d, Detection) for d in detections):
            raise TypeError("Frame detections must be canonical Detection records")
        for d in detections:
            x1, y1, x2, y2 = d.box_xyxy
            if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
                raise ValueError("Detection box lies outside original image; clip it before caching")
        if self.image is not None:
            if not isinstance(self.image, np.ndarray) or self.image.dtype.hasobject:
                raise TypeError("Cache images must be numpy arrays, never tensors or autograd graphs")
            if self.image.shape[:2] != (size[1], size[0]):
                raise ValueError("Image shape does not agree with image_size")
            # bytes-backed arrays cannot be made writable, even via setflags().
            array = np.frombuffer(self.image.tobytes(), dtype=self.image.dtype).reshape(self.image.shape)
            object.__setattr__(self, "image", array)
        object.__setattr__(self, "image_size", size)
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "detections", detections)


@dataclass(frozen=True)
class TemporalSnapshot:
    sequence_id: str
    frames: tuple[FrameRecord, ...]

    def __post_init__(self):
        frames = tuple(self.frames)
        if any(f.sequence_id != self.sequence_id for f in frames):
            raise ValueError("Snapshot cannot mix source sequences")
        object.__setattr__(self, "frames", frames)


def normalize_detections(result, image_size, class_ids, threshold=0.0, rejected=None):
    """Convert canonical detector dictionaries (including CPU/CUDA tensors).

    Detach outside the cache, clip boxes, and report each discarded annotation.
    """
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    boxes = array(result.get("boxes", [])).reshape(-1, 4)
    scores = array(result.get("scores", []))
    labels = array(result.get("labels", []))
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError("Detector boxes, scores and labels have inconsistent lengths")
    width, height = image_size
    detections = []
    names = result.get("class_names", [])
    for box, score, label in zip(boxes, scores, labels):
        if int(label) not in class_ids or float(score) < threshold:
            continue
        try:
            if not np.isfinite(box).all():
                raise ValueError("non-finite box")
            clipped = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            name = names.get(int(label)) if isinstance(names, dict) else names[int(label)] if 0 <= int(label) < len(names) else None
            detections.append(Detection(tuple(clipped), float(score), int(label), class_name=name))
        except ValueError as exc:
            if rejected is not None:
                rejected.append(str(exc))
    return tuple(detections)
