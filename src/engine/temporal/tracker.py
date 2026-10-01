"""Class-aware ByteTrack association with a constant-velocity Kalman filter.

High-score association includes lost tracks; low-score recovery is restricted
to unmatched tracks active in the previous frame. No appearance model is used.
Reference: https://github.com/ifzhang/ByteTrack (ECCV 2022).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Detection


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    return intersection / max(float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - intersection, 1e-12)


class PedestrianTracker(ABC):
    @abstractmethod
    def update(self, detections, timestamp_s): ...

    @abstractmethod
    def reset(self, sequence_id=""): ...


@dataclass
class _Track:
    id: int
    class_id: int
    mean: np.ndarray
    covariance: np.ndarray
    last_seen: int

    @property
    def box(self):
        cx, cy, w, h = self.mean[:4]
        w, h = max(w, 1e-6), max(h, 1e-6)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class ByteTrackPedestrianTracker(PedestrianTracker):
    def __init__(self, pedestrian_class_ids=(0,), high_confidence_threshold=0.5,
                 low_confidence_threshold=0.1, match_iou_threshold=0.3,
                 track_buffer_frames=30, **kwargs):
        if not 0 <= low_confidence_threshold <= high_confidence_threshold <= 1:
            raise ValueError("Tracker requires 0 <= low threshold <= high threshold <= 1")
        if not 0 < match_iou_threshold <= 1 or track_buffer_frames <= 0:
            raise ValueError("Tracker requires positive buffer and IoU threshold in (0, 1]")
        if not pedestrian_class_ids:
            raise ValueError("Tracker requires configured pedestrian_class_ids")
        self.class_ids = set(map(int, pedestrian_class_ids))
        self.high = high_confidence_threshold
        self.low = low_confidence_threshold
        self.match_iou = match_iou_threshold
        self.buffer = track_buffer_frames
        self.reset()

    def reset(self, sequence_id=""):
        self.sequence_id = sequence_id
        self._tracks = {}
        self._counter = 0
        self._frame = 0
        self._timestamp = None

    @property
    def active_tracks(self):
        return sum(t.last_seen == self._frame for t in self._tracks.values())

    @staticmethod
    def _measurement(d):
        x1, y1, x2, y2 = d.box_xyxy
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

    def _predict(self, track, dt):
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4) * dt
        scale = max(track.mean[3], 1) / 20
        noise = np.diag([scale**2] * 4 + [(scale / 8)**2] * 4)
        track.mean = transition @ track.mean
        track.covariance = transition @ track.covariance @ transition.T + noise

    def _correct(self, track, detection):
        measurement = self._measurement(detection)
        noise = np.eye(4) * (max(track.mean[3], 1) / 20)**2
        innovation = track.covariance[:4, :4] + noise
        gain = np.linalg.solve(innovation, track.covariance[:4, :]).T
        track.mean += gain @ (measurement - track.mean[:4])
        track.covariance -= gain @ track.covariance[:4, :]
        track.last_seen = self._frame

    def _associate(self, tracks, detections):
        if not tracks or not detections:
            return [], tracks, list(range(len(detections)))
        cost = np.full((len(tracks), len(detections)), 1e6)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = box_iou(track.box, det.box_xyxy)
                if track.class_id == det.class_id and iou >= self.match_iou:
                    cost[i, j] = 1 - iou
        # Dummy assignments ensure the optimum never sacrifices a valid match
        # merely to fill an invalid edge in a rectangular assignment matrix.
        augmented = np.concatenate([cost, np.full((len(tracks), len(tracks)), 2.0)], axis=1)
        rows, cols = linear_sum_assignment(augmented)
        matches = [(int(i), int(j)) for i, j in zip(rows, cols) if j < len(detections) and cost[i, j] < 2]
        used_t, used_d = {i for i, _ in matches}, {j for _, j in matches}
        return matches, [t for i, t in enumerate(tracks) if i not in used_t], [j for j in range(len(detections)) if j not in used_d]

    def update(self, detections: tuple[Detection, ...], timestamp_s):
        if not math.isfinite(timestamp_s) or (self._timestamp is not None and timestamp_s <= self._timestamp):
            raise ValueError("Tracker timestamps must be finite and strictly increasing")
        dt = timestamp_s - self._timestamp if self._timestamp is not None else 1 / 30
        self._timestamp = timestamp_s
        self._frame += 1
        self._tracks = {k: t for k, t in self._tracks.items() if self._frame - t.last_seen <= self.buffer}
        tracks = list(self._tracks.values())
        for track in tracks:
            self._predict(track, dt)
        selected = [d for d in detections if d.class_id in self.class_ids and d.score >= self.low]
        high = [d for d in selected if d.score >= self.high]
        low = [d for d in selected if d.score < self.high]
        assigned = {}
        matches, unmatched, new = self._associate(tracks, high)
        for i, j in matches:
            self._correct(tracks[i], high[j])
            assigned[id(high[j])] = tracks[i].id
        recoverable = [t for t in unmatched if t.last_seen == self._frame - 1]
        matches, _, _ = self._associate(recoverable, low)
        for i, j in matches:
            self._correct(recoverable[i], low[j])
            assigned[id(low[j])] = recoverable[i].id
        for j in new:
            self._counter += 1
            measurement = self._measurement(high[j])
            mean = np.concatenate([measurement, np.zeros(4)])
            scale = max(measurement[3], 1)
            covariance = np.diag([(scale / 10)**2] * 4 + [scale**2] * 4)
            self._tracks[self._counter] = _Track(self._counter, high[j].class_id, mean, covariance, self._frame)
            assigned[id(high[j])] = self._counter
        # Low-score unmatched boxes remain in COCO for audit, with no identity.
        return tuple(replace(d, track_id=f"{self.sequence_id}:{assigned[id(d)]}" if id(d) in assigned else None) for d in selected)


class IoUTracker(ByteTrackPedestrianTracker):
    """Selectable simpler association baseline, with no motion prediction."""
    def _predict(self, track, dt):
        pass


def create_tracker(config):
    config = dict(config)
    kind = config.pop("type", "bytetrack")
    classes = {"bytetrack": ByteTrackPedestrianTracker, "iou": IoUTracker}
    if kind not in classes:
        raise ValueError(f"Unknown tracker {kind!r}; choose bytetrack or iou")
    return classes[kind](**config)
