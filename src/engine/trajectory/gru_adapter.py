"""Compact mask-aware GRU encoder and residual MLP future decoder."""

import torch

from .base_adapter import TrajectoryAdapter


class GRUTrajectoryModel(torch.nn.Module):
    def __init__(self, hidden_size, horizon):
        super().__init__()
        self.encoder = torch.nn.GRUCell(3, hidden_size)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_size, horizon * 2))
        self.hidden_size, self.horizon = hidden_size, horizon

    def forward(self, points, mask, deltas):
        hidden = points.new_zeros((len(points), self.hidden_size))
        last = points.new_zeros((len(points), 2))
        elapsed = points.new_zeros(len(points))
        for t in range(points.shape[1]):
            elapsed = elapsed + deltas[:, t]
            update = self.encoder(torch.cat([points[:, t], elapsed[:, None]], dim=-1), hidden)
            hidden = torch.where(mask[:, t, None], update, hidden)
            last = torch.where(mask[:, t, None], points[:, t], last)
            elapsed = torch.where(mask[:, t], torch.zeros_like(elapsed), elapsed)
        return last[:, None] + self.decoder(hidden).reshape(-1, self.horizon, 2)


class GRUTrajectoryAdapter(TrajectoryAdapter):
    name = "gru_trajectory"

    def build_model(self, device, checkpoint=None):
        hidden = self.config.get("trajectory", {}).get("hidden_size", 128)
        if hidden <= 0:
            raise ValueError("trajectory.hidden_size must be positive")
        return GRUTrajectoryModel(hidden, self.prediction_frames).to(device)

    def forward(self, batch):
        return self.build()(batch["observed_points"], batch["observed_mask"], batch["observed_time_deltas"])
