"""Detector -> tracker -> cache -> histories -> canonical path predictions."""

from collections import Counter
from time import perf_counter

import numpy as np
from PIL import Image

from .config import validate_temporal_config
from .frame_cache import FrameCache
from .sequence_builder import SequenceBuilder
from .tracker import create_tracker
from .types import FrameRecord, normalize_detections


class TemporalInferencePipeline:
    def __init__(self, detector, trajectory_adapter, config):
        self.config = config.yaml_cfg if hasattr(config, "yaml_cfg") else config
        detector_config = getattr(getattr(detector, "_cfg", None), "yaml_cfg", {})
        num_classes = detector_config.get("num_classes", len(detector_config.get("class_names", [])))
        metadata_ids = range(num_classes) if num_classes else None
        temporal = validate_temporal_config(self.config, metadata_ids)
        self.detector, self.trajectory_adapter = detector, trajectory_adapter
        self.cache = FrameCache(**temporal.get("cache", {"max_frames": 32}))
        self.tracker = create_tracker(temporal.get("tracker", {}))
        self.builder = SequenceBuilder(**temporal.get("sequence", {}))
        if self.builder.observation_frames != trajectory_adapter.observation_frames:
            raise ValueError("Pipeline history length must match trajectory adapter horizon")
        self.sequence_id = None
        self.drops = Counter()
        self.frames_processed = 0

    def process(self, image, sequence_id, frame_index, timestamp_s):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        # Validate sequencing before advancing stateful detector/tracker.
        snapshot = self.cache.snapshot()
        if snapshot.frames and sequence_id == snapshot.sequence_id:
            last = snapshot.frames[-1]
            if frame_index <= last.frame_index or timestamp_s <= last.timestamp_s:
                raise ValueError("Live frame index and timestamp must strictly increase")
        FrameRecord(sequence_id, frame_index, timestamp_s, None, image.size, ())
        if sequence_id != self.sequence_id:
            self.cache.clear()
            self.tracker.reset(sequence_id)
            self.sequence_id = sequence_id
        start = perf_counter()
        outputs = self.detector.predict([image], threshold=self.tracker.low)
        if len(outputs) != 1:
            raise ValueError("Detector must return one canonical dictionary for one frame")
        rejected = []
        detections = normalize_detections(outputs[0], image.size, self.tracker.class_ids, self.tracker.low, rejected)
        self.drops.update(rejected)
        tracked = self.tracker.update(detections, timestamp_s)
        frame = FrameRecord(sequence_id, frame_index, timestamp_s,
                            np.asarray(image) if self.cache.store_images else None, image.size, tracked)
        self.cache.append(frame)
        histories, statuses = [], {}
        for detection in tracked:
            if detection.track_id is None:
                self.drops["unmatched_detection"] += 1
                continue
            history = self.builder.history_for(self.cache.snapshot(), detection.track_id)
            if history is None:
                statuses[detection.track_id] = "insufficient_history"
                self.drops["insufficient_history"] += 1
            else:
                histories.append(history)
                statuses[detection.track_id] = "ready"
        prediction_start = perf_counter()
        predictions = self.trajectory_adapter.predict_paths(histories) if histories else []
        prediction_ms = (perf_counter() - prediction_start) * 1000
        self.frames_processed += 1
        return {"status": "predicted" if predictions else "insufficient_history",
                "sequence_id": sequence_id, "frame_index": frame_index, "timestamp_s": timestamp_s,
                "detections": tracked, "predictions": predictions, "track_statuses": statuses,
                "statistics": {"cache_occupancy": len(self.cache), "evictions": self.cache.evictions,
                               "active_tracks": self.tracker.active_tracks, "history_ready_tracks": len(histories),
                               "prediction_latency_ms": prediction_ms, "total_latency_ms": (perf_counter() - start) * 1000,
                               "frames_processed": self.frames_processed, "drops_by_reason": dict(self.drops)}}
