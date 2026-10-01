import pytest
import torch

from engine.trajectory import TrajectoryPrediction
from engine.trajectory.metrics import evaluate_trajectories


def test_last_valid_fde_multimodal_minima_and_kde():
    batch = {"track_id": ["cam:1"], "future_points": torch.zeros(1, 3, 2),
             "future_mask": torch.tensor([[True, False, True]])}
    paths = torch.tensor([[[3., 4.], [99., 99.], [0., 2.]], [[0., 0.], [0., 0.], [0., 0.]]])
    prediction = TrajectoryPrediction("cam:1", paths, probabilities=torch.tensor([.8, .2]))
    metrics = evaluate_trajectories([prediction], batch, kde_bandwidth=.2)
    assert metrics["ADE"] == 3.5 and metrics["FDE"] == 2.
    assert metrics["minADE"] == 0 and metrics["minFDE"] == 0
    assert metrics["valid_points"] == 2 and metrics["sample_count"] == 2
    assert "KDE_NLL" in metrics
    with pytest.raises(ValueError, match="coordinate"):
        evaluate_trajectories([TrajectoryPrediction("cam:1", paths, coordinate_system="world")], batch)
