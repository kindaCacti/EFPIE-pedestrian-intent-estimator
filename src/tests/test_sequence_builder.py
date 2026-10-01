import numpy as np

from engine.temporal import Detection, FrameCache, FrameRecord, SequenceBuilder


def test_bottom_center_mask_time_deltas_and_no_implicit_padding():
    cache = FrameCache(4)
    builder = SequenceBuilder(observation_frames=3)
    for index in range(3):
        detections = () if index == 1 else (Detection((10, 20, 30, 60), .9, 0, track_id="cam:1"),)
        cache.append(FrameRecord("cam", index, index * .1, None, (100, 200), detections))
        if index < 2:
            assert builder.history_for(cache.snapshot(), "cam:1") is None
    assert builder.history_for(cache.snapshot(), "cam:1") is None
    sample = SequenceBuilder(observation_frames=3, allow_observation_gaps=True).history_for(cache.snapshot(), "cam:1")
    np.testing.assert_allclose(sample["observed_points"][[0, 2]], [[.2, .3], [.2, .3]])
    np.testing.assert_allclose(sample["observed_time_deltas"], [0, .1, .1])
    assert sample["observed_mask"].tolist() == [True, False, True]
    assert builder.history_for(cache.snapshot(), "other:1") is None


def test_opt_in_padding_and_missing_ingestion_frames():
    cache = FrameCache(4)
    for index in (0, 1):
        cache.append(FrameRecord("cam", index, index / 30, None, (100, 100),
                                 (Detection((10, 10, 30, 40), .9, 0, track_id="cam:1"),)))
    sample = SequenceBuilder(observation_frames=3, allow_padded_history=True).history_for(cache.snapshot(), "cam:1")
    assert sample["observed_mask"].tolist() == [False, True, True]
    assert not sample["observed_points"][0].any()
    cache.append(FrameRecord("cam", 3, .1, None, (100, 100), ()))
    assert SequenceBuilder(observation_frames=3, allow_observation_gaps=True).history_for(cache.snapshot(), "cam:1") is None
