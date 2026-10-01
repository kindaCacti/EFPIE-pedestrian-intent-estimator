import pytest

from engine.temporal import ByteTrackPedestrianTracker, Detection


def det(x=10, score=.9, class_id=0):
    return Detection((x, 10, x + 20, 60), score, class_id)


def test_high_association_low_recovery_and_unmatched_boxes():
    tracker = ByteTrackPedestrianTracker()
    tracker.reset("cam")
    identity = tracker.update([det()], 0)[0].track_id
    assert tracker.update([det(11)], 1 / 30)[0].track_id == identity
    output = tracker.update([det(12, .2), det(80, .2)], 2 / 30)
    assert output[0].track_id == identity
    assert output[1].track_id is None
    assert tracker.active_tracks == 1


def test_expiry_class_filtering_reset_and_lost_track_reactivation():
    tracker = ByteTrackPedestrianTracker(track_buffer_frames=2)
    tracker.reset("cam")
    original = tracker.update([det(), det(class_id=1), det(score=.05)], 0)
    assert len(original) == 1
    tracker.update([], 1 / 30)
    assert tracker.update([det()], 2 / 30)[0].track_id == original[0].track_id
    for t in range(3, 6):
        tracker.update([], t / 30)
    assert tracker.update([det()], 6 / 30)[0].track_id != original[0].track_id
    tracker.reset("next")
    assert tracker.update([det()], 0)[0].track_id == "next:1"


def test_class_aware_identity_and_timestamp_validation():
    tracker = ByteTrackPedestrianTracker(pedestrian_class_ids=[0, 1])
    outputs = tracker.update([det(class_id=0), det(class_id=1)], 0)
    next_outputs = tracker.update([det(class_id=1), det(class_id=0)], 1)
    assert next_outputs[0].track_id == outputs[1].track_id
    with pytest.raises(ValueError, match="timestamps"):
        tracker.update([], 1)
    with pytest.raises(ValueError):
        ByteTrackPedestrianTracker(low_confidence_threshold=.8, high_confidence_threshold=.5)
