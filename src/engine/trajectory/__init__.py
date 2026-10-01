from .base_adapter import TrajectoryAdapter, TrajectoryPrediction
from .constant_velocity_adapter import ConstantVelocityAdapter
from .gru_adapter import GRUTrajectoryAdapter
from .bitrap_adapter import BiTraPAdapter

TRAJECTORY_ADAPTERS = {adapter.name: adapter for adapter in
                       (ConstantVelocityAdapter, GRUTrajectoryAdapter, BiTraPAdapter)}


def register_trajectory_adapter(adapter):
    if not adapter.name or adapter.name in TRAJECTORY_ADAPTERS:
        raise ValueError(f"Trajectory adapter name missing or already registered: {adapter.name!r}")
    TRAJECTORY_ADAPTERS[adapter.name] = adapter
    return adapter
