"""Shared offline/live temporal configuration validation."""

from .tracker import create_tracker
from .sequence_builder import SequenceBuilder


def validate_temporal_config(config, category_ids=None):
    temporal = config.get("temporal", config)
    sequence = temporal.get("sequence", {})
    SequenceBuilder(**sequence)
    capacity = temporal.get("cache", {}).get("max_frames", 32)
    if not isinstance(capacity, int) or capacity < sequence.get("observation_frames", 8):
        raise ValueError("temporal.cache.max_frames must be >= observation_frames")
    tracker = create_tracker(temporal.get("tracker", {}))
    if category_ids is not None and not tracker.class_ids.issubset(set(map(int, category_ids))):
        raise ValueError("pedestrian_class_ids are absent from detector category metadata")
    return temporal
