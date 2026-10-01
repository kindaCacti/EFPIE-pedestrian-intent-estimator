"""Optional upstream goal-conditioned BiTraP D/NP/GMM box adapter."""

from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from .base_adapter import TrajectoryAdapter, TrajectoryPrediction


class BiTraPAdapter(TrajectoryAdapter):
    name = "bitrap"
    source_repo = "https://github.com/umautobots/bidirection-trajectory-predicter"
    license = "CC-BY-NC-SA-4.0"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.options = self.config.get("bitrap", {})
        self.pretrained_profile = self.options.get("pretrained_profile")
        self.decoder_with_z = self.options.get("decoder_with_z", True)
        if not isinstance(self.decoder_with_z, bool):
            raise ValueError("bitrap.decoder_with_z must be boolean")
        if self.pretrained_profile not in {None, "pie_np"}:
            raise ValueError("Unknown BiTraP pretrained_profile")
        self.variant = self.options.get("variant", "deterministic")
        self.representation = self.options.get("coordinate_representation", "normalized_cxcywh")
        if self.variant not in {"deterministic", "np", "gmm"}:
            raise ValueError("bitrap.variant must be deterministic, np or gmm")
        if self.representation not in {"normalized_cxcywh", "xywh"}:
            raise ValueError("BiTraP requires normalized_cxcywh or pixel xywh box coordinates")
        self.num_samples = 1 if self.variant == "deterministic" else int(self.options.get("num_samples", 20))
        if self.num_samples <= 0:
            raise ValueError("bitrap.num_samples must be positive")
        if self.options.get("sampling_fps", 30) <= 0:
            raise ValueError("bitrap.sampling_fps must be positive")
        self.upstream_root = self.options.get("upstream_root", str(Path(__file__).resolve().parents[2] / "third_party" / "bitrap"))
        for key, value in (("observation_frames", self.observation_frames), ("prediction_frames", self.prediction_frames)):
            if key in self.options and self.options[key] != value:
                raise ValueError(f"bitrap.{key} must match temporal.sequence.{key}")
        if self.pretrained_profile:
            expected = ("np", "normalized_cxcywh", 15, 45, 30, 256, 32, False, 1, [1920, 1080])
            actual = (self.variant, self.representation, self.observation_frames, self.prediction_frames,
                      self.options.get("sampling_fps", 30), self.options.get("hidden_size", 256),
                      self.options.get("latent_dim", 32), self.decoder_with_z,
                      self.sequence.get("frame_stride", 1), self.options.get("reference_image_size"))
            if actual != expected:
                raise ValueError("Pretrained PIE NP requires normalized cxcywh, 15/45 frames at 30 Hz, "
                                 "256/32 hidden/latent, decoder_with_z=false and 1920x1080 input")

    @property
    def input_schema(self):
        from ._bitrap_runtime import source_digest
        schema = {**super().input_schema, "box_representation": self.representation,
                "variant": self.variant, "sampling_fps": self.options.get("sampling_fps", 30),
                "normalization": "per_frame_image_dimensions", "upstream_source_sha256": source_digest(self.upstream_root),
                "compatibility_profile": "sova_bitrap_v1"}
        if self.pretrained_profile or not self.decoder_with_z:
            schema["decoder_with_z"] = self.decoder_with_z
        if self.pretrained_profile:
            schema.update(normalization="fixed_reference_image_dimensions",
                          reference_image_size=[1920, 1080], compatibility_profile="sova_bitrap_pie_np_v1")
        return schema

    def load(self, path):
        if not self.pretrained_profile:
            return super().load(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("task") != "trajectory":
            raise ValueError("Raw research checkpoint: use tools/import_bitrap_checkpoint.py to create an envelope")
        restored = self._restore_checkpoint(checkpoint)
        self._pretrained_provenance = checkpoint.get("pretrained_provenance")
        return restored

    def checkpoint(self, optimizer=None, epoch=-1):
        checkpoint = super().checkpoint(optimizer, epoch)
        if getattr(self, "_pretrained_provenance", None):
            checkpoint["pretrained_provenance"] = self._pretrained_provenance
        return checkpoint

    def build_model(self, device, checkpoint=None):
        from ._bitrap_runtime import load_bitrap
        np_class, gmm_class = load_bitrap(self.upstream_root, device)
        hidden = self.options.get("hidden_size", 256)
        values = dict(GLOBAL_INPUT_DIM=4, DEC_OUTPUT_DIM=4, INPUT_EMBED_SIZE=hidden,
                      ENC_HIDDEN_SIZE=hidden, DEC_INPUT_SIZE=hidden, DEC_HIDDEN_SIZE=hidden,
                      GOAL_HIDDEN_SIZE=64, LATENT_DIM=self.options.get("latent_dim", 32),
                      DEC_WITH_Z=self.decoder_with_z, PRIOR_DROPOUT=0., DROPOUT=0., Z_CLIP=False,
                      KL_MIN=.07 if self.pretrained_profile else .001, BEST_OF_MANY=True, PRED_LEN=self.prediction_frames,
                      K=self.num_samples, dt=1 / self.options.get("sampling_fps", 30))
        model = (gmm_class if self.variant == "gmm" else np_class)(SimpleNamespace(**values), dataset_name="PIE").to(device)
        if self.variant == "deterministic":
            # BiTraP-D: no latent sampling or posterior branch.
            def no_latent(module, enc_h, cur_state, target=None, z_mode=None):
                return enc_h.new_zeros((len(enc_h), 1, module.cfg.LATENT_DIM)), enc_h.new_zeros(())
            model.gaussian_latent_net = MethodType(no_latent, model)
        return model

    def prepare_batch(self, batch, training=False):
        prepared = super().prepare_batch(batch, training)
        mask = prepared["observed_mask"]
        if mask.shape[1] != self.observation_frames or not mask.all():
            raise ValueError("BiTraP requires a full continuous visible box history; padding/gaps are unsupported")
        if "observed_boxes" not in prepared or prepared["observed_boxes"].shape != (*mask.shape, 4):
            raise ValueError("BiTraP requires observed_boxes[B,T,4] in normalized cxcywh")
        fps = self.options.get("sampling_fps", 30)
        if fps <= 0:
            raise ValueError("bitrap.sampling_fps must be positive")
        for metadata in prepared["metadata"]:
            if metadata.get("box_format") != "normalized_cxcywh" or not metadata.get("image_sizes"):
                raise ValueError("BiTraP requires box_format and original image dimensions in metadata")
            if self.pretrained_profile and any(list(size) != [1920, 1080] for size in metadata["image_sizes"]):
                raise ValueError("Pretrained PIE normalization requires original 1920x1080 images")
            if abs(metadata.get("prediction_dt_s", 1 / fps) - 1 / fps) > 1e-5:
                raise ValueError("BiTraP sampling_fps differs from prepared data; change horizons/rate together")
        deltas = prepared["observed_time_deltas"][:, 1:]
        if not torch.allclose(deltas, torch.full_like(deltas, 1 / fps), atol=1e-5):
            raise ValueError("BiTraP requires consecutive regularly sampled timesteps")
        if training and ("future_boxes" not in prepared or not prepared["future_mask"].all()):
            raise ValueError("BiTraP training requires continuous visible future_boxes")
        prepared["bitrap_observed"] = self._convert_boxes(prepared["observed_boxes"], prepared, False)
        if training:
            prepared["bitrap_future"] = self._convert_boxes(prepared["future_boxes"], prepared, True)
        return prepared

    def _convert_boxes(self, boxes, batch, future):
        if not torch.isfinite(boxes).all() or (boxes[..., 2:] <= 0).any():
            raise ValueError("BiTraP input boxes must be finite with positive dimensions")
        if self.representation == "normalized_cxcywh":
            return boxes
        converted = boxes.clone()
        converted[..., :2] -= converted[..., 2:] / 2
        sizes = []
        for metadata in batch["metadata"]:
            all_sizes = metadata["image_sizes"]
            chosen = all_sizes[self.observation_frames:] if future else all_sizes[:self.observation_frames]
            if len(chosen) != boxes.shape[1]:
                raise ValueError("Box sequence lacks aligned original image dimensions")
            sizes.append(chosen)
        scale = boxes.new_tensor(sizes).repeat(1, 1, 2)
        return converted * scale

    def forward(self, batch):
        return self.build()(batch["bitrap_observed"], target_y=batch.get("bitrap_future") if batch["training"] else None)

    def compute_loss(self, outputs, batch):
        losses = outputs[2]
        return losses["loss_goal"] + losses["loss_traj"] + self.options.get("kl_weight", .01) * losses["loss_kld"]

    def decode(self, outputs, batch):
        boxes = outputs[1]  # upstream [B,T,K,4]
        if boxes.ndim == 3:
            boxes = boxes.unsqueeze(2)
        boxes = boxes.permute(0, 2, 1, 3)
        predictions = []
        for i, paths in enumerate(boxes):
            metadata = batch["metadata"][i]
            normalized = paths.clone()
            if self.representation == "xywh":
                # Live inference uses current resolution for future rendering.
                sizes = metadata["image_sizes"][self.observation_frames:]
                if not sizes:
                    sizes = [metadata["image_sizes"][-1]] * self.prediction_frames
                scale = paths.new_tensor(sizes).repeat(1, 2)
                normalized /= scale[None]
                normalized[..., :2] += normalized[..., 2:] / 2
            points = torch.stack([normalized[..., 0], normalized[..., 1] + normalized[..., 3] / 2], dim=-1)
            extra = {"box_paths_normalized_cxcywh": normalized.detach().cpu(), "variant": self.variant,
                     "implementation": "upstream_bitrap_with_sova_compatibility", "num_samples": len(paths)}
            component_paths, component_probabilities = None, None
            # Full GMM components have genuine mixture probabilities. Sampled
            # paths have no per-path likelihood, so do not invent probabilities.
            if self.variant == "gmm" and outputs[4] is not None:
                mixture = outputs[4]
                extra["component_probabilities"] = mixture.log_pis[i].exp().detach().cpu()
                extra["component_box_means"] = mixture.mus[i].detach().cpu()
                means = mixture.mus[i, 0].permute(1, 0, 2).clone()
                if self.representation == "xywh":
                    means /= scale[None]
                    means[..., :2] += means[..., 2:] / 2
                component_paths = torch.stack([means[..., 0], means[..., 1] + means[..., 3] / 2], dim=-1).detach().cpu()
                component_probabilities = mixture.log_pis[i, 0, 0].exp().detach().cpu()
            predictions.append(TrajectoryPrediction(track_id=batch["track_id"][i],
                paths=(points[0] if self.variant == "deterministic" else points).detach().cpu(),
                sequence_id=batch["sequence_id"][i], source_frame_index=metadata.get("source_frame_index"), metadata=extra,
                component_paths=component_paths, component_probabilities=component_probabilities))
        return predictions
