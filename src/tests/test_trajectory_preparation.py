import json

import numpy as np
import pytest
import yaml

from conftest import MockPedestrianDetector
from engine.data_prep import prepare_dataset, load_source_manifest, validate_coco
from engine.data_prep.manifest import sha256_file


def test_coco_validity_deterministic_batching_and_provenance(tmp_path, tiny_sources):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"test checkpoint")
    config = {"num_classes": 1, "class_names": ["pedestrian"], "resume": str(checkpoint),
              "dataset_preparation": {"inference_batch_size": 1}}
    first, second = tmp_path / "first", tmp_path / "second"
    info = prepare_dataset(MockPedestrianDetector(), config, tiny_sources, first)
    config["dataset_preparation"]["inference_batch_size"] = 4
    prepare_dataset(MockPedestrianDetector(), config, tiny_sources, second)
    seen_images, seen_annotations = set(), set()
    for split in ("train", "val", "test"):
        path = first / "annotations" / f"instances_{split}.json"
        data = json.loads(path.read_text())
        assert data == json.loads((second / "annotations" / path.name).read_text())
        validate_coco(data)
        image_ids = {i["id"] for i in data["images"]}
        annotation_ids = {a["id"] for a in data["annotations"]}
        assert not seen_images.intersection(image_ids) and not seen_annotations.intersection(annotation_ids)
        seen_images.update(image_ids)
        seen_annotations.update(annotation_ids)
        assert data["annotations"][0]["bbox"] == [10., 10., 20., 40.]
        assert data["annotations"][0]["track_id"] == split + "_camera:1"
        assert (first / "images" / data["images"][0]["file_name"]).is_file()
    assert info["detector"]["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert info["label_provenance"] == "detector_tracker_pseudo_labels"
    assert (first / "manifests" / "source_sequences.json").is_file()


def test_clipping_rejection_and_atomic_failure(tmp_path, tiny_sources):
    class InvalidDetector(MockPedestrianDetector):
        def predict(self, images, threshold=.3):
            return [{"boxes": np.array([[-10, -5, 110, 120], [5, 5, 5, 10], [0, 0, np.nan, 1]]),
                     "scores": [.9] * 3, "labels": [0] * 3} for _ in images]
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="failure threshold"):
        prepare_dataset(InvalidDetector(), {}, tiny_sources, output)
    assert not output.exists() and not list(tmp_path.glob(".bad-preparing-*"))
    info = prepare_dataset(InvalidDetector(), {"dataset_preparation": {"validation": {"max_failure_fraction": 1}}}, tiny_sources, output)
    data = json.loads((output / "annotations" / "instances_train.json").read_text())
    assert data["annotations"][0]["bbox"] == [0, 0, 100, 100]
    assert info["rejected_frame_count"] == 30
    assert len((output / "manifests" / "rejected_frames.jsonl").read_text().splitlines()) == 60


def test_sequence_split_and_timestamp_rejection(tiny_sources):
    data = yaml.safe_load(tiny_sources.read_text())
    data["sequences"].append({**data["sequences"][0], "sequence_id": "alias", "split": "val"})
    tiny_sources.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="multiple"):
        load_source_manifest(tiny_sources)
    data["sequences"].pop()
    data["sequences"][0]["timestamps_s"] = [0] * 10
    tiny_sources.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="nonmonotonic"):
        load_source_manifest(tiny_sources)


def test_completed_versions_require_force_and_backup_is_recoverable(prepared_dataset, tiny_sources):
    original = (prepared_dataset / "annotations" / "instances_train.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_dataset(MockPedestrianDetector(), {}, tiny_sources, prepared_dataset)
    info = prepare_dataset(MockPedestrianDetector(), {}, tiny_sources, prepared_dataset, force=True)
    from pathlib import Path
    assert (Path(info["previous_version_backup"]) / "annotations" / "instances_train.json").read_bytes() == original


def test_sparse_source_category_mapping_uses_contiguous_detector_labels(tmp_path, tiny_sources):
    info = prepare_dataset(MockPedestrianDetector(), {"num_classes": 2,
        "class_names": ["person", "bicycle"], "category_ids": [1, 7]},
        tiny_sources, tmp_path / "sparse-v1")
    assert info["category_mapping"]["0"] == {"category_id": 1, "name": "person",
                                             "detector_class_id": 0, "original_category_id": 1}


def test_force_never_replaces_unrelated_directories(tmp_path, tiny_sources):
    directory = tmp_path / "unrelated"
    directory.mkdir()
    with pytest.raises(ValueError, match="unrelated"):
        prepare_dataset(MockPedestrianDetector(), {}, tiny_sources, directory, force=True)


def test_requested_range_preserves_original_indices_and_timestamps(tiny_sources):
    from engine.data_prep import iter_frames
    data = yaml.safe_load(tiny_sources.read_text())
    data["sequences"][0].update({"start_frame": 2, "end_frame": 7})
    tiny_sources.write_text(yaml.safe_dump(data))
    sequence = load_source_manifest(tiny_sources)[0]
    frames = list(iter_frames(sequence, stride=2))
    assert [f[0] for f in frames] == [2, 4, 6]
    assert frames[0][1] == pytest.approx(2 / 30)
    data["sequences"][0]["end_frame"] = 11
    tiny_sources.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="start_frame"):
        load_source_manifest(tiny_sources)


def test_preparation_freezes_adapter_default_weights(tmp_path, tiny_sources):
    class SavingDetector(MockPedestrianDetector):
        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"built detector weights")
            return path

    output = tmp_path / "default-weights-v1"
    info = prepare_dataset(SavingDetector(), {}, tiny_sources, output)
    checkpoint = output / "manifests" / "detector_checkpoint.pth"
    assert checkpoint.read_bytes() == b"built detector weights"
    assert info["detector"]["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert info["detector"]["checkpoint"]["path"] == str(checkpoint)
