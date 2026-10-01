import os
from pathlib import Path

import pytest
import torch

from engine.config import YAMLConfig
from engine.trajectory.bitrap_adapter import BiTraPAdapter
from engine.trajectory.pretrained_bitrap import import_pie_checkpoint
from tools.view_trajectories import resolve_checkpoint


def pie_config():
    config = YAMLConfig(str(Path(__file__).resolve().parents[1] /
                            "conf/models/trajectory/bitrap/pretrained_pie.yml"), device="cpu").yaml_cfg
    config = {**config}
    config.pop("resume", None)
    return config


def test_checkpoint_selection_respects_model():
    assert resolve_checkpoint({"model": {"adapter": "bitrap"}, "resume": "pie.pth"}) == "pie.pth"
    assert resolve_checkpoint({"model": {"adapter": "bitrap"}, "resume": "pie.pth"}, "override") == "override"
    assert resolve_checkpoint({"model": {"adapter": "constant_velocity"}}) is None
    assert resolve_checkpoint({"model": {"adapter": "gru_trajectory"}}).endswith("checkpoint_last.pth")


@pytest.mark.parametrize("key,value", [("decoder_with_z", True), ("sampling_fps", 15),
                                      ("reference_image_size", [1280, 720]), ("variant", "gmm")])
def test_pretrained_profile_rejects_wrong_settings(key, value):
    config = pie_config()
    config["bitrap"] = {**config["bitrap"], key: value}
    with pytest.raises(ValueError, match="Pretrained PIE"):
        BiTraPAdapter(config)


def test_import_rejects_unknown_weights_and_overwrite(tmp_path):
    raw, destination = tmp_path / "unknown.pth", tmp_path / "imported.pth"
    torch.save({"anything": torch.ones(1)}, raw)
    with pytest.raises(ValueError, match="SHA-256"):
        import_pie_checkpoint(raw, destination, pie_config())
    assert not destination.exists()
    destination.touch()
    with pytest.raises(FileExistsError):
        import_pie_checkpoint(raw, destination, pie_config())


def test_verified_real_pie_import_inference_roundtrip(tmp_path):
    source = os.environ.get("BITRAP_TEST_WEIGHTS")
    root = os.environ.get("BITRAP_TEST_ROOT")
    if not source or not root:
        pytest.skip("Set BITRAP_TEST_WEIGHTS and BITRAP_TEST_ROOT to audited PIE weights/source")
    config = pie_config()
    config["bitrap"] = {**config["bitrap"], "upstream_root": root}
    checkpoint = tmp_path / "pie.pth"
    provenance = import_pie_checkpoint(source, checkpoint, config)
    assert provenance["dataset"] == "PIE" and provenance["trained_num_samples"] == 20
    adapter = BiTraPAdapter(config)
    envelope = adapter.load(checkpoint)
    assert envelope["input_schema"]["decoder_with_z"] is False
    boxes = torch.tensor([.5, .5, .04, .15]).repeat(1, 15, 1)
    batch = {"observed_points": boxes[..., :2], "observed_boxes": boxes,
             "observed_mask": torch.ones(1, 15, dtype=torch.bool),
             "observed_time_deltas": torch.tensor([[0.] + [1 / 30] * 14]),
             "track_id": ["pedestrian"], "sequence_id": ["test"],
             "metadata": [{"coordinate_system": "image_normalized", "box_format": "normalized_cxcywh",
                           "image_sizes": [[1920, 1080]] * 15, "prediction_dt_s": 1 / 30}]}
    torch.manual_seed(42)
    expected = adapter.predict_paths(batch)[0]
    assert expected.paths.shape == (20, 45, 2) and expected.probabilities is None
    saved = adapter.save(tmp_path / "saved.pth")
    restored = BiTraPAdapter(config)
    assert restored.load(saved)["pretrained_provenance"] == provenance
    torch.manual_seed(42)
    torch.testing.assert_close(restored.predict_paths(batch)[0].paths, expected.paths)
    batch["metadata"][0]["image_sizes"] = [[1280, 720]] * 15
    with pytest.raises(ValueError, match="1920x1080"):
        adapter.predict_paths(batch)
