import os

import pytest
import torch

from engine.trajectory import TRAJECTORY_ADAPTERS
from engine.trajectory.dataset import TrajectoryDataset
from engine.trajectory.collate import trajectory_collate


def config(name, **extra):
    return {"device": "cpu", "model": {"adapter": name}, "trajectory": {"hidden_size": 8},
            "temporal": {"sequence": {"observation_frames": 3, "prediction_frames": 2}}, **extra}


@pytest.mark.parametrize("name", ["constant_velocity", "gru_trajectory"])
def test_native_adapter_lifecycle_one_step_save_load(prepared_dataset, tmp_path, name):
    batch = trajectory_collate([TrajectoryDataset(prepared_dataset, observation_frames=3, prediction_frames=2)[0]])
    adapter = TRAJECTORY_ADAPTERS[name](config(name))
    model = adapter.build()
    prepared = adapter.prepare_batch(batch, training=True)
    loss = adapter.compute_loss(adapter.forward(prepared), prepared)
    if name == "gru_trajectory":
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss.backward()
        optimizer.step()
    else:
        assert loss.item() < 1e-12
    prediction = adapter.predict_paths(batch)[0]
    assert prediction.paths.shape == (2, 2)
    checkpoint = adapter.save(tmp_path / (name + ".pth"), epoch=3)
    restored = TRAJECTORY_ADAPTERS[name](config(name))
    envelope = restored.load(checkpoint)
    assert envelope["task"] == "trajectory" and envelope["epoch"] == 3
    torch.testing.assert_close(restored.predict_paths(batch)[0].paths, prediction.paths)
    incompatible = TRAJECTORY_ADAPTERS[name](config(name))
    incompatible.prediction_frames = 5
    with pytest.raises(ValueError, match="horizons"):
        incompatible.load(checkpoint)


@pytest.mark.parametrize("variant", ["deterministic", "np", "gmm"])
def test_optional_upstream_bitrap_training_and_inference(prepared_dataset, tmp_path, variant):
    root = os.environ.get("BITRAP_TEST_ROOT")
    if not root:
        pytest.skip("Set BITRAP_TEST_ROOT to an optional upstream checkout")
    options = {"upstream_root": root, "variant": variant, "hidden_size": 8, "latent_dim": 3, "num_samples": 4}
    adapter = TRAJECTORY_ADAPTERS["bitrap"](config("bitrap", bitrap=options))
    dataset = TrajectoryDataset(prepared_dataset, observation_frames=3, prediction_frames=2)
    batch = trajectory_collate([dataset[0], dataset[1]])
    model = adapter.build()
    prepared = adapter.prepare_batch(batch, training=True)
    loss = adapter.compute_loss(adapter.forward(prepared), prepared)
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    torch.optim.Adam(model.parameters(), lr=.0001).step()
    paths = adapter.predict_paths(batch)
    assert paths[0].paths.shape == ((2, 2) if variant == "deterministic" else (4, 2, 2))
    assert paths[0].metadata["box_paths_normalized_cxcywh"].shape == (1 if variant == "deterministic" else 4, 2, 4)
    adapter.save(tmp_path / (variant + ".pth"))
    assert adapter.load(tmp_path / (variant + ".pth"))["adapter"] == "bitrap"
    if variant == "gmm":
        assert "component_probabilities" in paths[0].metadata


def test_bitrap_rejects_raw_research_checkpoints(tmp_path):
    path = tmp_path / "published.pth"
    torch.save({"model": {}}, path)
    adapter = TRAJECTORY_ADAPTERS["bitrap"](config("bitrap"))
    with pytest.raises(ValueError, match="envelope"):
        adapter.load(path)
