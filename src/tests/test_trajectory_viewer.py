from collections import deque

import numpy as np
from PIL import Image

from conftest import MockPedestrianDetector
from engine.temporal.pipeline import TemporalInferencePipeline
from engine.trajectory import ConstantVelocityAdapter
from tools.view_trajectories import TrajectoryWindow, render_overlay


def test_viewer_draws_predictions_without_mutating_source():
    config = {"device": "cpu", "temporal": {"cache": {"max_frames": 3},
              "sequence": {"observation_frames": 3, "prediction_frames": 2}}}
    pipeline = TemporalInferencePipeline(MockPedestrianDetector(), ConstantVelocityAdapter(config), config)
    for index in range(3):
        image = Image.new("RGB", (100, 100), (index, 0, 0))
        result = pipeline.process(image, "camera", index, index / 30)
    before = np.asarray(image).copy()
    canvas = np.asarray(render_overlay(image, result, pipeline))
    assert np.array_equal(np.asarray(image), before)
    for color in ((0, 255, 0), (255, 255, 0), (0, 255, 255)):
        assert np.any(np.all(canvas == color, axis=-1))


class FakeCV:
    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 4

    def __init__(self, keys, visible=True):
        self.keys = deque(keys)
        self.visible = visible
        self.polls = 0

    def namedWindow(self, *args):
        pass

    def resizeWindow(self, *args):
        pass

    def waitKey(self, delay):
        self.polls += 1
        return self.keys.popleft()

    def getWindowProperty(self, *args):
        return int(self.visible)


def test_pause_services_events_until_resumed():
    cv = FakeCV([ord(" "), -1, ord(" ")])
    window = TrajectoryWindow(cv)
    assert window.wait()
    assert not window.paused and cv.polls == 3


def test_quit_while_paused_and_window_close():
    window = TrajectoryWindow(FakeCV([ord(" "), ord("q")]))
    assert not window.wait() and window.closed
    window = TrajectoryWindow(FakeCV([-1], visible=False))
    assert not window.wait() and window.closed
