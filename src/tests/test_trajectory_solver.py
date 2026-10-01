import json
from types import SimpleNamespace

import pytest
import torch

from engine.config import YAMLConfig, dataset_overrides
from engine.solver.tasks import TASKS


@pytest.mark.parametrize("name", ["constant_velocity", "gru_trajectory"])
def test_cpu_solver_checkpoint_metrics_and_resume(prepared_dataset, tmp_path, name):
    config = {"task": "trajectory", "device": "cpu", "model": {"adapter": name},
              "trajectory": {"hidden_size": 8}, "output_dir": str(tmp_path / name), "epochs": 1,
              "temporal": {"cache": {"max_frames": 8}, "sequence": {"observation_frames": 3, "prediction_frames": 2}},
              "train_dataloader": {"batch_size": 4, "dataset": {"prepared_dir": str(prepared_dataset)}},
              "val_dataloader": {"batch_size": 4, "dataset": {"prepared_dir": str(prepared_dataset)}}}
    solver = TASKS["trajectory"](SimpleNamespace(yaml_cfg=config))
    solver.fit()
    output = tmp_path / name
    assert (output / "checkpoint_last.pth").is_file()
    metrics = json.loads((output / "trajectory_metrics.json").read_text())
    assert metrics["valid_samples"] == 12 and metrics["valid_points"] == 24
    assert metrics["ADE"] >= 0 and metrics["FDE"] >= 0
    assert metrics["label_provenance"] == "detector_tracker_pseudo_labels"
    config.update({"resume": str(output / "checkpoint_last.pth"), "epochs": 2})
    TASKS["trajectory"](SimpleNamespace(yaml_cfg=config)).fit()
    assert torch.load(output / "checkpoint_last.pth", weights_only=False)["epoch"] == 1


def test_yaml_overrides_and_trajectory_task_are_available(tmp_path):
    import yaml
    path = tmp_path / "dataset.yml"
    path.write_text(yaml.safe_dump({"task": "trajectory", "temporal": {"sequence": {"prediction_frames": 4}},
                                     "train_dataloader": {"batch_size": 2, "dataset": {"prepared_dir": "fixture"}}}))
    overrides = dataset_overrides(path)
    cfg = YAMLConfig("conf/models/trajectory/gru/base.yml", **overrides)
    assert cfg.yaml_cfg["temporal"]["sequence"]["prediction_frames"] == 4
    assert cfg.yaml_cfg["train_dataloader"]["batch_size"] == 2
    assert "num_classes" not in overrides and "trajectory" in TASKS


def test_training_cli_runs_from_yaml(prepared_dataset, tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import yaml
    dataset_config = tmp_path / "dataset.yml"
    dataset_config.write_text(yaml.safe_dump({"task": "trajectory",
        "train_dataloader": {"dataset": {"prepared_dir": str(prepared_dataset)}},
        "val_dataloader": {"dataset": {"prepared_dir": str(prepared_dataset)}}}))
    output = tmp_path / "cli-run"
    result = subprocess.run([sys.executable, "train.py", "-c", "conf/models/trajectory/gru/s.yml",
        "--dataset-config", str(dataset_config), "--device", "cpu", "-o", str(output),
        "-u", "epochs=1", "temporal.sequence.observation_frames=3", "temporal.sequence.prediction_frames=2"],
        cwd=Path(__file__).resolve().parents[1], env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert (output / "checkpoint_last.pth").is_file()
    assert "ADE=" in result.stdout and "FDE=" in result.stdout


def test_second_adapter_uses_unchanged_solver_dataset_and_cache(prepared_dataset, tmp_path):
    from engine.trajectory import ConstantVelocityAdapter, TRAJECTORY_ADAPTERS, register_trajectory_adapter
    @register_trajectory_adapter
    class AnotherPredictor(ConstantVelocityAdapter):
        name = "test_another_predictor"
    config = {"device": "cpu", "model": {"adapter": AnotherPredictor.name},
        "output_dir": str(tmp_path / "custom"), "epochs": 1,
        "temporal": {"cache": {"max_frames": 8}, "sequence": {"observation_frames": 3, "prediction_frames": 2}},
        "train_dataloader": {"dataset": {"prepared_dir": str(prepared_dataset)}},
        "val_dataloader": {"dataset": {"prepared_dir": str(prepared_dataset)}}}
    try:
        solver = TASKS["trajectory"](SimpleNamespace(yaml_cfg=config))
        solver.fit()
        assert solver.val()["ADE"] < 1e-6
    finally:
        TRAJECTORY_ADAPTERS.pop(AnotherPredictor.name)
