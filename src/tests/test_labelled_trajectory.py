import json

import pytest
import torch

from engine.data_prep.labelled_export import import_labelled_dataset
from engine.data_prep.manifest import write_json
from engine.trajectory.dataset import TrajectoryDataset


def test_verified_import_and_detected_history_matching(prepared_dataset, tmp_path):
    inputs = {split: prepared_dataset / "annotations" / f"instances_{split}.json" for split in ("train", "val", "test")}
    output = tmp_path / "human-v1"
    info = import_labelled_dataset(inputs, prepared_dataset / "images", output)
    assert info["label_provenance"] == "human_ground_truth"
    ground_truth = TrajectoryDataset(output, observation_frames=3, prediction_frames=2,
                                     evaluation_mode="ground_truth_history")
    detected = TrajectoryDataset(output, observation_frames=3, prediction_frames=2,
                                 detected_history_dir=prepared_dataset)
    assert len(ground_truth) == len(detected) == 12
    for i in range(len(ground_truth)):
        torch.testing.assert_close(ground_truth[i]["future_points"], detected[i]["future_points"])
        assert detected[i]["metadata"]["detected_track_id"] == ground_truth[i]["track_id"]
    with pytest.raises(FileExistsError):
        import_labelled_dataset(inputs, prepared_dataset / "images", output)


def test_detected_history_does_not_require_detected_future(prepared_dataset, tmp_path):
    inputs = {split: prepared_dataset / "annotations" / f"instances_{split}.json" for split in ("train", "val", "test")}
    human = tmp_path / "human-v1"
    import_labelled_dataset(inputs, prepared_dataset / "images", human)
    # Remove detections after observation frame 2. Human future targets remain.
    path = prepared_dataset / "annotations" / "instances_train.json"
    data = json.loads(path.read_text())
    keep_ids = {i["id"] for i in data["images"] if i["frame_index"] <= 2}
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    write_json(path, data)
    manifest = prepared_dataset / "manifests" / "prepared_dataset.json"
    provenance = json.loads(manifest.read_text())
    provenance.pop("annotation_fingerprints")
    write_json(manifest, provenance)
    dataset = TrajectoryDataset(human, observation_frames=3, prediction_frames=2,
                                detected_history_dir=prepared_dataset)
    assert len(dataset) == 2 and dataset[0]["future_mask"].all()
    assert dataset.exclusion_counts["unmatched_ground_truth_windows"] == 10
