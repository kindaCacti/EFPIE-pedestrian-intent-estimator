"""Bounded, thread-safe single-stream frame cache."""

from collections import deque
from dataclasses import replace
from threading import RLock

from .types import FrameRecord, TemporalSnapshot


class FrameCache:
    def __init__(self, max_frames, store_images=False, require_monotonic_timestamps=True):
        if not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self.max_frames = max_frames
        self.store_images = store_images
        self.require_monotonic_timestamps = require_monotonic_timestamps
        self._frames = deque(maxlen=max_frames)
        self._sequence_id = ""
        self._lock = RLock()
        self.evictions = 0

    def append(self, frame: FrameRecord):
        if not isinstance(frame, FrameRecord):
            raise TypeError("FrameCache accepts FrameRecord only")
        with self._lock:
            if frame.sequence_id != self._sequence_id:
                self._frames.clear()
                self._sequence_id = frame.sequence_id
            if self._frames:
                last = self._frames[-1]
                if frame.frame_index <= last.frame_index:
                    raise ValueError("frame_index must increase within a sequence")
                if self.require_monotonic_timestamps and frame.timestamp_s <= last.timestamp_s:
                    raise ValueError("timestamp_s must increase within a sequence")
            if len(self._frames) == self.max_frames:
                self.evictions += 1
            self._frames.append(frame if self.store_images else replace(frame, image=None))

    def snapshot(self, sequence_id=None):
        with self._lock:
            if sequence_id is not None and sequence_id != self._sequence_id:
                return TemporalSnapshot(sequence_id, ())
            # Return independent image storage as well as an immutable tuple.
            return TemporalSnapshot(self._sequence_id, tuple(
                replace(f, image=f.image) if f.image is not None else f for f in self._frames
            ))

    def clear(self, sequence_id=None):
        with self._lock:
            if sequence_id is None or sequence_id == self._sequence_id:
                self._frames.clear()
                self._sequence_id = ""

    def __len__(self):
        with self._lock:
            return len(self._frames)
