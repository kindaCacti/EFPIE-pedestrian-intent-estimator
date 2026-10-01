"""Reuse SOVA YAML composition with task-aware dataset overrides."""

import copy

from .core import YAMLConfig, yaml_utils

__all__ = ["YAMLConfig", "yaml_utils", "dataset_overrides"]


def dataset_overrides(path):
    data = yaml_utils.load_config(path)
    if data.get("task") == "trajectory" or "trajectory_dataset" in data:
        return {key: copy.deepcopy(value) for key, value in data.items() if key != "__include__"}
    return yaml_utils.dataset_overrides(path)
