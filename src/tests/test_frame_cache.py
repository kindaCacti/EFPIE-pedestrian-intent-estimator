from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from engine.temporal import Detection, FrameCache, FrameRecord


def frame(index, sid="camera", image=None, stamp=None):
    return FrameRecord(sid, index, index / 30 if stamp is None else stamp, image, (20, 20),
                       (Detection((1, 2, 10, 15), .9, 0, track_id=sid + ":1"),))


def test_exact_capacity_order_and_snapshots_survive_eviction():
    cache = FrameCache(3)
    for index in range(3):
        cache.append(frame(index))
    previous = cache.snapshot()
    cache.append(frame(3))
    assert [f.frame_index for f in cache.snapshot().frames] == [1, 2, 3]
    assert [f.frame_index for f in previous.frames] == [0, 1, 2]
    assert cache.evictions == 1 and len(cache) == 3
    with pytest.raises(FrozenInstanceError):
        previous.frames[0].timestamp_s = 10


def test_array_ownership_and_images_off_by_default():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    cached = frame(0, image=image)
    image[:] = 255
    cache = FrameCache(2, store_images=True)
    cache.append(cached)
    snapshot = cache.snapshot()
    assert not snapshot.frames[0].image.any()
    assert not np.shares_memory(snapshot.frames[0].image, cached.image)
    with pytest.raises(ValueError):
        snapshot.frames[0].image.setflags(write=True)
    stripped = FrameCache(2)
    stripped.append(cached)
    assert stripped.snapshot().frames[0].image is None


def test_sequence_change_and_targeted_clear():
    cache = FrameCache(3)
    cache.append(frame(5))
    cache.append(frame(0, sid="next"))
    assert len(cache) == 1 and not cache.snapshot("camera").frames
    cache.clear("camera")
    assert len(cache) == 1
    cache.clear("next")
    assert len(cache) == 0


def test_order_timestamps_and_invalid_records():
    cache = FrameCache(2)
    cache.append(frame(1))
    with pytest.raises(ValueError, match="frame_index"):
        cache.append(frame(1))
    with pytest.raises(ValueError, match="timestamp"):
        cache.append(frame(2, stamp=0))
    with pytest.raises(ValueError):
        Detection((2, 1, 1, 10), .9, 0)
    with pytest.raises(ValueError):
        FrameRecord("cam", 0, 0, None, (0, 20), ())
    with pytest.raises(TypeError, match="numpy"):
        import torch
        frame(0, image=torch.zeros(20, 20, 3))


def test_concurrent_snapshots_are_ordered():
    from concurrent.futures import ThreadPoolExecutor
    cache = FrameCache(8)
    def writer():
        for index in range(100):
            cache.append(frame(index))
    def reader():
        for _ in range(100):
            indices = [f.frame_index for f in cache.snapshot().frames]
            assert indices == sorted(set(indices))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(writer), pool.submit(reader), pool.submit(reader)]
        for future in futures:
            future.result()
    assert len(cache) == 8
