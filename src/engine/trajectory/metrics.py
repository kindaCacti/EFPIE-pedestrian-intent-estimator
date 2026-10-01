"""Masked path and box metrics, with explicitly reported sample counts."""

import math

import torch


class TrajectoryMetrics:
    def __init__(self, coordinate_system="image_normalized", kde_bandwidth=None):
        if coordinate_system != "image_normalized":
            raise ValueError("Pixel/metre metrics require an explicit coordinate transformer")
        if kde_bandwidth is not None and kde_bandwidth <= 0:
            raise ValueError("KDE bandwidth must be positive")
        self.coordinate_system = coordinate_system
        self.kde_bandwidth = kde_bandwidth
        self.sums = {}
        self.counts = {}
        self.sample_counts = set()
        self.valid_samples = self.valid_points = self.dropped_samples = 0

    def _add(self, key, value):
        self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.counts[key] = self.counts.get(key, 0) + 1

    def update(self, predictions, batch):
        if len(predictions) != len(batch["track_id"]):
            raise ValueError("Prediction count must agree with canonical batch")
        for i, pred in enumerate(predictions):
            if pred.track_id != batch["track_id"][i]:
                raise ValueError("Prediction track identity does not match evaluation target")
            if pred.coordinate_system != self.coordinate_system:
                raise ValueError("Prediction and evaluation coordinate systems disagree")
            mask = torch.as_tensor(batch["future_mask"][i], dtype=torch.bool, device="cpu")
            if not mask.any():
                self.dropped_samples += 1
                continue
            target = batch["future_points"][i].detach().cpu()
            paths = torch.as_tensor(pred.paths).detach().cpu()
            if paths.ndim == 2:
                paths = paths.unsqueeze(0)
            if paths.shape[1:] != target.shape or not torch.isfinite(paths).all():
                raise ValueError("Prediction horizon/shape or finite values do not match target")
            k = len(paths)
            self.sample_counts.add(k)
            distance = torch.linalg.vector_norm(paths[:, mask] - target[mask], dim=-1)
            ade, fde = distance.mean(-1), distance[:, -1]
            # ADE/FDE use highest-probability path, or the first sample when
            # probabilities are unavailable. min metrics always use all K.
            chosen = int(torch.as_tensor(pred.probabilities).argmax()) if pred.probabilities is not None else 0
            self._add("ADE", ade[chosen])
            self._add("FDE", fde[chosen])
            if k > 1:
                self._add("minADE", ade.min())
                self._add("minFDE", fde.min())
            self.valid_samples += 1
            self.valid_points += int(mask.sum())
            if self.kde_bandwidth is not None and k > 1:
                variance = self.kde_bandwidth**2
                log_kernel = -distance.square() / (2 * variance) - math.log(2 * math.pi * variance)
                self._add("KDE_NLL", -(torch.logsumexp(log_kernel, dim=0) - math.log(k)).mean())
            raw_boxes = pred.metadata.get("box_paths_normalized_cxcywh")
            if raw_boxes is not None and "future_boxes" in batch:
                boxes = torch.as_tensor(raw_boxes).detach().cpu()
                if boxes.ndim == 2:
                    boxes = boxes.unsqueeze(0)
                target_boxes = batch["future_boxes"][i].detach().cpu()
                if boxes.shape[1:] != target_boxes.shape or not torch.isfinite(boxes).all():
                    raise ValueError("Predicted box horizon/values disagree with target")
                box_distance = torch.linalg.vector_norm(boxes[:, mask] - target_boxes[mask], dim=-1)
                center_distance = torch.linalg.vector_norm(boxes[:, mask, :2] - target_boxes[mask, :2], dim=-1)
                for key, values in (("box_ADE", box_distance.mean(-1)), ("CADE", center_distance.mean(-1)), ("CFDE", center_distance[:, -1])):
                    self._add(key, values[chosen])
                    if k > 1:
                        self._add("min" + key, values.min())

    def compute(self):
        return {**{k: v / self.counts[k] for k, v in self.sums.items()},
                "coordinate_system": self.coordinate_system, "units": "dimensionless",
                "valid_samples": self.valid_samples, "valid_points": self.valid_points,
                "dropped_samples": self.dropped_samples,
                "sample_count": next(iter(self.sample_counts)) if len(self.sample_counts) == 1 else sorted(self.sample_counts),
                "ADE_aggregation": "mean_per_track"}


def evaluate_trajectories(predictions, batch, **kwargs):
    metrics = TrajectoryMetrics(**kwargs)
    metrics.update(predictions, batch)
    return metrics.compute()
