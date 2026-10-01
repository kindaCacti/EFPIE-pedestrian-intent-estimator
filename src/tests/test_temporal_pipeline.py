from PIL import Image
import pytest

from conftest import MockPedestrianDetector
from engine.temporal.pipeline import TemporalInferencePipeline
from engine.trajectory import ConstantVelocityAdapter


def test_end_to_end_no_prediction_before_history_and_sequence_reset():
    config = {"device": "cpu", "temporal": {"cache": {"max_frames": 3},
              "sequence": {"observation_frames": 3, "prediction_frames": 2}}}
    pipeline = TemporalInferencePipeline(MockPedestrianDetector(), ConstantVelocityAdapter(config), config)
    for index in range(6):
        result = pipeline.process(Image.new("RGB", (100, 100), (index, 0, 0)), "cam", index, index / 30)
        assert len(result["predictions"]) == (0 if index < 2 else 2)
        assert result["statistics"]["cache_occupancy"] == min(index + 1, 3)
    assert result["predictions"][0].paths[0, 0].item() == pytest.approx(.26)
    assert result["statistics"]["evictions"] == 3
    reset = pipeline.process(Image.new("RGB", (100, 100)), "new", 0, 0)
    assert reset["status"] == "insufficient_history" and len(pipeline.cache) == 1
    assert reset["detections"][0].track_id == "new:1"
    with pytest.raises(ValueError, match="increase"):
        pipeline.process(Image.new("RGB", (100, 100)), "new", 0, 0)
