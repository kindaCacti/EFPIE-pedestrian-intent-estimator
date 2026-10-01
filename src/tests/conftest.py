"""Small offline temporal fixtures; no detector downloads."""

import numpy as np
from PIL import Image
import pytest
import yaml

from engine.data_prep import prepare_dataset


class MockPedestrianDetector:
    name = "mock_pedestrian"
    version = "1"

    def __init__(self):
        self.calls = 0

    def build(self):
        return self

    def predict(self, images, threshold=.3):
        self.calls += 1
        outputs = []
        for image in images:
            index = image.getpixel((0, 0))[0]
            outputs.append({"boxes": np.array([[10 + index, 10, 30 + index, 50], [60, 20 + index, 80, 60 + index]], np.float32),
                            "scores": np.array([.9, .8]), "labels": np.array([0, 0])})
        return outputs


@pytest.fixture
def tiny_sources(tmp_path):
    sequences = []
    for split in ("train", "val", "test"):
        directory = tmp_path / (split + "_source")
        directory.mkdir()
        for index in range(10):
            Image.new("RGB", (100, 100), (index, 0, 0)).save(directory / f"{index:06d}.png")
        sequences.append({"sequence_id": split + "_camera", "frame_dir": directory.name, "fps": 30, "split": split})
    manifest = tmp_path / "sources.yml"
    manifest.write_text(yaml.safe_dump({"sequences": sequences}))
    return manifest


@pytest.fixture
def prepared_dataset(tmp_path, tiny_sources):
    path = tmp_path / "prepared-v1"
    cfg = {"num_classes": 1, "class_names": ["pedestrian"],
           "dataset_preparation": {"image_mode": "copy", "inference_batch_size": 3,
                                   "validation": {"min_track_length_frames": 5}}}
    detector = MockPedestrianDetector()
    prepare_dataset(detector, cfg, tiny_sources, path)
    return path
