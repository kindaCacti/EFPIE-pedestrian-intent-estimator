"""Model-independent trajectory lifecycle and portable checkpoint envelope."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .collate import trajectory_collate


@dataclass(frozen=True)
class TrajectoryPrediction:
    track_id: str | int
    paths: torch.Tensor  # [T, 2] deterministic or [K, T, 2] multi-modal
    probabilities: torch.Tensor | None = None
    confidence: float | None = None
    uncertainty: torch.Tensor | None = None
    coordinate_system: str = "image_normalized"
    sequence_id: str = ""
    source_frame_index: int | None = None
    metadata: dict = field(default_factory=dict)
    component_paths: torch.Tensor | None = None  # optional GMM [C,T,2] means
    component_probabilities: torch.Tensor | None = None

    def __post_init__(self):
        if self.paths.ndim not in {2, 3} or self.paths.shape[-1] != 2 or not torch.isfinite(self.paths).all():
            raise ValueError("Trajectory paths must be finite [T,2] or [K,T,2] tensors")
        if self.paths.shape[-2] == 0 or (self.paths.ndim == 3 and self.paths.shape[0] == 0):
            raise ValueError("Trajectory prediction must contain a nonempty future path")
        if self.probabilities is not None:
            k = self.paths.shape[0] if self.paths.ndim == 3 else 1
            p = self.probabilities
            if p.shape != (k,) or not torch.isfinite(p).all() or (p < 0).any() or not torch.isclose(p.sum(), p.new_tensor(1.), atol=1e-4):
                raise ValueError("Path probabilities must be non-negative, sum to one and align with K")
        if self.component_paths is not None:
            paths, probabilities = self.component_paths, self.component_probabilities
            if paths.ndim != 3 or paths.shape[1:] != self.paths.shape[-2:] or not torch.isfinite(paths).all():
                raise ValueError("Component paths must be finite [C,T,2] aligned with sampled paths")
            if probabilities is None or probabilities.shape != (len(paths),) or not torch.isfinite(probabilities).all() or (probabilities < 0).any() or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.), atol=1e-4):
                raise ValueError("Component probabilities must align with C and sum to one")

    @property
    def path(self):
        if self.paths.ndim == 2:
            return self.paths
        index = int(self.probabilities.argmax()) if self.probabilities is not None else 0
        return self.paths[index]


class TrajectoryAdapter(ABC):
    name = ""
    uses_external_trainer = False

    def __init__(self, cfg):
        self.cfg = cfg
        raw_config = cfg.yaml_cfg if hasattr(cfg, "yaml_cfg") else cfg
        self.config = {k: v for k, v in raw_config.items() if k != "__include__"}
        self.sequence = self.config.get("temporal", {}).get("sequence", {})
        self.observation_frames = self.sequence.get("observation_frames", 8)
        self.prediction_frames = self.sequence.get("prediction_frames", 12)
        if min(self.observation_frames, self.prediction_frames) <= 0:
            raise ValueError("Trajectory horizons must be positive")
        self._model = None

    @property
    def device(self):
        return torch.device(self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    @property
    def input_schema(self):
        return {"coordinate_system": "image_normalized", "observation_frames": self.observation_frames,
                "prediction_frames": self.prediction_frames, "point": "bottom_center_normalized"}

    def build(self):
        if self._model is None:
            self._model = self.build_model(self.device)
            checkpoint = self.config.get("resume") or self.config.get("tuning")
            if checkpoint:
                self.load(checkpoint)
        return self._model

    @abstractmethod
    def build_model(self, device, checkpoint=None): ...

    def prepare_batch(self, batch, training=False):
        if isinstance(batch, list):
            batch = trajectory_collate(batch)
        required = ("observed_points", "observed_time_deltas", "observed_mask", "track_id", "sequence_id", "metadata")
        if any(k not in batch for k in required):
            raise ValueError(f"Canonical trajectory batch requires {required}")
        prepared = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        points, mask, dt = (prepared[k] for k in ("observed_points", "observed_mask", "observed_time_deltas"))
        if points.ndim != 3 or points.shape[-1] != 2 or mask.shape != points.shape[:2] or dt.shape != mask.shape:
            raise ValueError("Expected observed_points[B,T,2], mask[B,T] and deltas[B,T]")
        if points.shape[1] > self.observation_frames or not mask.any(dim=1).all():
            raise ValueError("Observation horizon exceeded or history has no visible points")
        if not torch.isfinite(points).all() or not torch.isfinite(dt).all() or (dt < 0).any():
            raise ValueError("Observed points/time deltas must be finite, with non-negative deltas")
        if mask.dtype != torch.bool:
            raise ValueError("observed_mask must be a boolean tensor")
        if any(m.get("coordinate_system") != "image_normalized" for m in batch["metadata"]):
            raise ValueError("Adapter requires image_normalized observation coordinates")
        if len(batch["track_id"]) != len(points) or len(batch["metadata"]) != len(points):
            raise ValueError("Batch identity/metadata lengths do not agree")
        if training:
            if "future_points" not in prepared or "future_mask" not in prepared:
                raise ValueError("Training requires future_points and future_mask")
            future, valid = prepared["future_points"], prepared["future_mask"]
            if future.shape != (len(points), self.prediction_frames, 2) or valid.shape != future.shape[:2] or not valid.any(dim=1).all() or not torch.isfinite(future).all():
                raise ValueError("Future targets/masks must agree with the configured horizon and contain valid points")
        prepared["training"] = training
        if "future_time_deltas" in prepared:
            future_times = prepared["future_time_deltas"]
            if future_times.shape != (len(points), self.prediction_frames) or not torch.isfinite(future_times).all() or (future_times <= 0).any() or (torch.diff(future_times, dim=1) <= 0).any():
                raise ValueError("Future times must be finite, positive, increasing and align with the prediction horizon")
        return prepared

    @abstractmethod
    def forward(self, prepared_batch): ...

    def compute_loss(self, outputs, prepared_batch):
        target, mask = prepared_batch["future_points"], prepared_batch["future_mask"]
        error = (outputs - target).square().sum(-1)
        return error[mask].mean()

    def decode(self, outputs, prepared_batch):
        predictions = []
        for i, path in enumerate(outputs):
            metadata = prepared_batch["metadata"][i]
            predictions.append(TrajectoryPrediction(
                track_id=prepared_batch["track_id"][i], paths=path.detach().cpu(),
                sequence_id=prepared_batch["sequence_id"][i], source_frame_index=metadata.get("source_frame_index"),
                metadata={"prediction_dt_s": metadata.get("prediction_dt_s")}))
        return predictions

    def predict_paths(self, batch):
        model = self.build()
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                prepared = self.prepare_batch(batch, training=False)
                return self.decode(self.forward(prepared), prepared)
        finally:
            model.train(was_training)

    def train_external(self):
        raise NotImplementedError(f"{self.name} uses the native TrajectorySolver loop")

    def validate_external(self):
        raise NotImplementedError(f"{self.name} uses native trajectory validation")

    def checkpoint(self, optimizer=None, epoch=-1):
        return {"format_version": 1, "task": "trajectory", "adapter": self.name,
                "model": self.build().state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "epoch": epoch, "config": self.config, "coordinate_system": "image_normalized",
                "input_schema": self.input_schema}

    def save(self, path, optimizer=None, epoch=-1):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(optimizer, epoch), path)
        return path

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        return self._restore_checkpoint(checkpoint)

    def _restore_checkpoint(self, checkpoint):
        if checkpoint.get("task") != "trajectory" or checkpoint.get("format_version") != 1 or checkpoint.get("adapter") != self.name:
            raise ValueError("Incompatible checkpoint: expected a version-1 trajectory envelope for " + self.name)
        if checkpoint.get("coordinate_system") != "image_normalized" or checkpoint.get("input_schema") != self.input_schema:
            raise ValueError("Checkpoint normalization, input representation or horizons differ from the configured schema")
        if self._model is None:
            self._model = self.build_model(self.device)
        self._model.load_state_dict(checkpoint["model"])
        return checkpoint


def prediction_times(batch, horizon):
    if "future_time_deltas" in batch:
        return batch["future_time_deltas"]
    points = batch["observed_points"]
    step = points.new_tensor([m.get("prediction_dt_s", 1 / 30) for m in batch["metadata"]])
    return step[:, None] * torch.arange(1, horizon + 1, device=points.device)
