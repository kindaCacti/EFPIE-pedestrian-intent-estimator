import json

import pytest
import torch

from engine.data_prep.manifest import write_json
from engine.trajectory.collate import trajectory_collate
from engine.trajectory.dataset import TrajectoryDataset


def test_windows_alignment_and_box_retention(prepared_dataset):
    dataset = TrajectoryDataset(prepared_dataset, observation_frames=3, prediction_frames=2)
    assert len(dataset) == 12  # two tracks, six windows each
    sample = dataset[0]
    assert sample["track_id"] == "train_camera:1"
    torch.testing.assert_close(sample["observed_points"][:, 0], torch.tensor([.2, .21, .22]))
    torch.testing.assert_close(sample["future_points"][:, 0], torch.tensor([.23, .24]))
    assert sample["future_boxes"].shape == (2, 4)
    assert sample["observed_mask"].all() and sample["future_mask"].all()
    val = TrajectoryDataset(prepared_dataset, split="val", observation_frames=3, prediction_frames=2)
    assert set(s[0] for s in dataset.windows).isdisjoint(s[0] for s in val.windows)


def test_padding_and_gap_policy(prepared_dataset):
    dataset = TrajectoryDataset(prepared_dataset, observation_frames=3, prediction_frames=2, allow_padded_history=True)
    assert len(dataset) == 16
    assert dataset[0]["observed_mask"].tolist() == [False, False, True]
    first, second = dataset[0], dataset[1]
    first = {**first, "observed_points": first["observed_points"][-1:], "observed_boxes": first["observed_boxes"][-1:],
             "observed_mask": first["observed_mask"][-1:], "observed_time_deltas": first["observed_time_deltas"][-1:]}
    batch = trajectory_collate([first, second])
    assert batch["observed_points"].shape == (2, 3, 2)
    assert batch["observed_mask"][0].tolist() == [False, False, True]


def test_schema_completion_and_annotation_immutability(prepared_dataset):
    with pytest.raises(ValueError, match="ground_truth_history"):
        TrajectoryDataset(prepared_dataset, evaluation_mode="ground_truth_history")
    path = prepared_dataset / "annotations" / "instances_train.json"
    data = json.loads(path.read_text())
    data["images"][0].pop("timestamp_s")
    write_json(path, data)
    with pytest.raises(ValueError, match="changed"):
        TrajectoryDataset(prepared_dataset)
    manifest_path = prepared_dataset / "manifests" / "prepared_dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("annotation_fingerprints")
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="timestamp_s"):
        TrajectoryDataset(prepared_dataset)


def test_split_leakage_is_rejected(prepared_dataset):
    manifest_path = prepared_dataset / "manifests" / "prepared_dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("annotation_fingerprints")
    write_json(manifest_path, manifest)
    path = prepared_dataset / "annotations" / "instances_val.json"
    data = json.loads(path.read_text())
    for image in data["images"]:
        image["sequence_id"] = "train_camera"
    for annotation in data["annotations"]:
        annotation["track_id"] = annotation["track_id"].replace("val_camera", "train_camera")
    write_json(path, data)
    with pytest.raises(ValueError, match="split leakage"):
        TrajectoryDataset(prepared_dataset)
