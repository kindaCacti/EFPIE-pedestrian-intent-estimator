"""Parameter-free reference model using the last two visible observations."""

import torch

from .base_adapter import TrajectoryAdapter, prediction_times


class ConstantVelocityModel(torch.nn.Module):
    def forward(self, points, mask, deltas, future_times):
        times = deltas.cumsum(-1)
        paths = []
        for i in range(len(points)):
            visible = mask[i].nonzero().flatten()
            last = visible[-1]
            velocity = torch.zeros_like(points[i, last])
            if len(visible) > 1:
                previous = visible[-2]
                duration = times[i, last] - times[i, previous]
                if duration <= 0:
                    raise ValueError("Visible observations must have strictly increasing timestamps")
                velocity = (points[i, last] - points[i, previous]) / duration
            paths.append(points[i, last] + future_times[i, :, None] * velocity)
        return torch.stack(paths)


class ConstantVelocityAdapter(TrajectoryAdapter):
    name = "constant_velocity"

    def build_model(self, device, checkpoint=None):
        return ConstantVelocityModel().to(device)

    def forward(self, batch):
        return self.build()(batch["observed_points"], batch["observed_mask"],
                            batch["observed_time_deltas"], prediction_times(batch, self.prediction_frames))
