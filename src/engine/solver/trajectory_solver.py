"""Trajectory task orchestration: prepared data, generic adapters and metrics."""

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ..data_prep.manifest import write_json
from ..temporal.config import validate_temporal_config
from ..trajectory import TRAJECTORY_ADAPTERS
from ..trajectory.collate import trajectory_collate
from ..trajectory.dataset import TrajectoryDataset
from ..trajectory.metrics import TrajectoryMetrics


class TrajectorySolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.config = {k: v for k, v in cfg.yaml_cfg.items() if k != "__include__"}
        validate_temporal_config(self.config)
        self.output_dir = Path(self.config.get("output_dir", "outputs/trajectory"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._adapter, self._loaders = None, {}
        self.last_epoch = -1
        seed = self.config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with (self.output_dir / "resolved_config.yml").open("w") as stream:
            yaml.safe_dump(self.config, stream, sort_keys=False)

    def adapter(self):
        if self._adapter is None:
            name = self.config["model"]["adapter"]
            if name not in TRAJECTORY_ADAPTERS:
                raise ValueError(f"Unknown trajectory adapter {name!r}; choose {sorted(TRAJECTORY_ADAPTERS)}")
            self._adapter = TRAJECTORY_ADAPTERS[name](self.cfg)
        return self._adapter

    @property
    def device(self):
        return self.adapter().device

    def _loader(self, split):
        if split not in self._loaders:
            options = self.config.get(f"{split}_dataloader", {})
            dataset_options = dict(options.get("dataset", {}))
            dataset_options.pop("type", None)
            sequence_options = dict(self.config.get("temporal", {}).get("sequence", {}))
            sequence_options.pop("point", None)
            dataset_options = {**sequence_options, **dataset_options, "split": split,
                               "evaluation_mode": self.config.get("evaluation", {}).get("mode", "detected_history")}
            dataset = TrajectoryDataset(**dataset_options)
            if not len(dataset):
                raise ValueError(f"No valid {split} trajectory windows: {dataset.exclusion_counts}; check horizons, visibility and sampling rate")
            # Preserve provenance for each used split, even if roots differ.
            write_json(self.output_dir / f"dataset_provenance_{split}.json", dataset.provenance)
            write_json(self.output_dir / f"dataset_exclusions_{split}.json", dataset.exclusion_counts)
            loader = DataLoader(dataset, batch_size=options.get("batch_size", 32),
                                shuffle=split == "train", num_workers=options.get("num_workers", 0),
                                collate_fn=trajectory_collate,
                                generator=torch.Generator().manual_seed(self.config.get("seed", 42)))
            self._loaders[split] = loader
            if "train" in self._loaders and "val" in self._loaders:
                train_sequences = {w[0] for w in self._loaders["train"].dataset.windows}
                val_sequences = {w[0] for w in self._loaders["val"].dataset.windows}
                if train_sequences.intersection(val_sequences):
                    raise ValueError("Sequence-level split leakage between training and validation roots")
        return self._loaders[split]

    def fit(self):
        adapter = self.adapter()
        if adapter.uses_external_trainer:
            return adapter.train_external()
        model = adapter.build()
        loader = self._loader("train")
        self._loader("val")
        params = [p for p in model.parameters() if p.requires_grad]
        settings = self.config.get("trajectory", {})
        optimizer = torch.optim.AdamW(params, lr=settings.get("learning_rate", .001),
                                      weight_decay=settings.get("weight_decay", 0)) if params else None
        scheduler_config = settings.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config.get("step_size", 20),
                    gamma=scheduler_config.get("gamma", .5)) if optimizer and scheduler_config else None
        resume = self.config.get("resume")
        if resume:
            checkpoint = adapter.load(resume)
            self.last_epoch = checkpoint.get("epoch", -1)
            if optimizer and checkpoint.get("optimizer"):
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler and checkpoint.get("scheduler"):
                scheduler.load_state_dict(checkpoint["scheduler"])
        enabled_amp = self.config.get("use_amp", False) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=enabled_amp)
        if resume and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        epochs = self.config.get("epochs", 1)
        frequency = self.config.get("checkpoint_freq", 1)
        if epochs <= 0 or frequency <= 0:
            raise ValueError("epochs and checkpoint_freq must be positive")
        for epoch in range(self.last_epoch + 1, epochs):
            model.train()
            total_loss, batches = 0., 0
            for batch in loader:
                prepared = adapter.prepare_batch(batch, training=True)
                if optimizer:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=enabled_amp):
                    loss = adapter.compute_loss(adapter.forward(prepared), prepared)
                    if isinstance(loss, dict):
                        loss = sum(loss.values())
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite trajectory loss; check input coordinates and model configuration")
                if optimizer:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip = self.config.get("clip_max_norm", 0)
                    if clip:
                        torch.nn.utils.clip_grad_norm_(params, clip)
                    scaler.step(optimizer)
                    scaler.update()
                total_loss += loss.item()
                batches += 1
            if scheduler:
                scheduler.step()
            metrics = self.val()
            report = {"epoch": epoch, "loss": total_loss / batches, **metrics}
            with (self.output_dir / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            print(f"epoch {epoch}: loss={report['loss']:.6f} ADE={metrics.get('ADE')} FDE={metrics.get('FDE')}")
            state = adapter.checkpoint(optimizer, epoch)
            state["scheduler"] = scheduler.state_dict() if scheduler else None
            state["scaler"] = scaler.state_dict()
            torch.save(state, self.output_dir / "checkpoint_last.pth")
            if (epoch + 1) % frequency == 0:
                torch.save(state, self.output_dir / f"checkpoint{epoch:04d}.pth")
            self.last_epoch = epoch
        return model

    def val(self):
        adapter = self.adapter()
        if adapter.uses_external_trainer:
            return adapter.validate_external()
        loader = self._loader("val")
        evaluation = self.config.get("evaluation", {})
        accumulator = TrajectoryMetrics(kde_bandwidth=evaluation.get("kde_bandwidth"))
        for batch in loader:
            accumulator.update(adapter.predict_paths(batch), batch)
        metrics = accumulator.compute()
        metrics.update({"label_provenance": loader.dataset.provenance.get("label_provenance"),
                        "dataset_version": loader.dataset.provenance.get("dataset_version"),
                        "evaluation_mode": loader.dataset.evaluation_mode,
                        "detector": loader.dataset.provenance.get("detector"),
                        "tracker": loader.dataset.provenance.get("tracker"), "adapter": adapter.name})
        metrics["dataset_exclusions"] = loader.dataset.exclusion_counts
        if loader.dataset._detected_dataset is not None:
            detected_provenance = loader.dataset._detected_dataset.provenance
            metrics["detector"] = detected_provenance.get("detector")
            metrics["tracker"] = detected_provenance.get("tracker")
            metrics["detected_history_dataset_version"] = detected_provenance.get("dataset_version")
            write_json(self.output_dir / "detected_history_provenance.json", detected_provenance)
        write_json(self.output_dir / "trajectory_metrics.json", metrics)
        return metrics
