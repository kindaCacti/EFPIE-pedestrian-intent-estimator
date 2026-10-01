"""Explicit offline detector execution and atomic prepared-dataset publication."""

from itertools import islice
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import yaml

from ..temporal.tracker import create_tracker
from ..temporal.types import normalize_detections
from .detection_writer import DetectionWriter
from .manifest import creation_time, fingerprint, write_json
from .sequence_index import iter_frames, load_source_manifest


def prepare_dataset(detector, cfg, source_manifest=None, output_dir=None, force=False):
    raw_config = cfg.yaml_cfg if hasattr(cfg, "yaml_cfg") else cfg
    config = {k: v for k, v in raw_config.items() if k != "__include__"}
    prep = config.get("dataset_preparation", {})
    manifest_path = source_manifest or prep.get("source_manifest")
    destination = output_dir or prep.get("output_dir")
    if not manifest_path or not destination:
        raise ValueError("Preparation requires source_manifest and output_dir")
    destination = Path(destination).absolute()
    if destination.exists() and not force:
        raise FileExistsError(f"Prepared version {destination} exists; use a new version or --force")
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise ValueError("Prepared destination must be a directory, not a file or symlink")
    if destination.exists() and force and not (destination / "manifests" / "prepared_dataset.json").is_file():
        raise ValueError("--force can replace only a prepared dataset version, never an unrelated directory")
    mode = prep.get("image_mode", "symlink")
    stride, batch_size = prep.get("frame_stride", 1), prep.get("inference_batch_size", 8)
    if mode not in {"symlink", "hardlink", "copy"} or stride <= 0 or batch_size <= 0:
        raise ValueError("Invalid image_mode, frame_stride or inference_batch_size")
    class_ids = list(prep.get("pedestrian_class_ids", [0]))
    names = config.get("class_names", [])
    num_classes = config.get("num_classes", len(names))
    original_category_ids = config.get("category_ids", list(range(num_classes)))
    # SOVA's adapter contract uses contiguous detector labels. Sparse COCO
    # category IDs are source metadata, not the detector's prediction labels.
    if num_classes and not set(class_ids).issubset(range(num_classes)):
        raise ValueError("Preparation pedestrian_class_ids are absent from detector metadata")
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("pedestrian_class_ids must be a nonempty unique list")
    mapping = {int(c): (i + 1, names[c] if c < len(names) else f"pedestrian_{c}") for i, c in enumerate(sorted(class_ids))}
    tracker_config = {**config.get("temporal", {}).get("tracker", {}), **prep.get("tracker", {}), "pedestrian_class_ids": class_ids}
    tracker = create_tracker(tracker_config)
    threshold = float(prep.get("confidence_threshold", .3))
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must lie in [0, 1]")
    # Keep recovery boxes through the tracker; apply the export threshold after
    # association so occlusion recovery isn't defeated by detector filtering.
    detector_threshold = min(threshold, tracker.low)
    sequences = load_source_manifest(manifest_path)
    provenance = [s.provenance() for s in sequences]  # freeze source fingerprints before inference
    checkpoint = config.get("resume") or config.get("model", {}).get("checkpoint") or config.get("tuning")
    if checkpoint and not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {checkpoint}")
    checkpoint_info = fingerprint(checkpoint) if checkpoint else None
    detector.build()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-preparing-", dir=destination.parent))
    backup = None
    try:
        writer = DetectionWriter(staging, mapping)
        # When an adapter downloads/defaults its weights internally, freeze its
        # built state in the dataset version so the exact detector is auditable.
        if checkpoint_info is None and hasattr(detector, "save"):
            state_path = staging / "manifests" / "detector_checkpoint.pth"
            detector.save(state_path)
            checkpoint_info = fingerprint(state_path)
            checkpoint_info["path"] = str(destination / "manifests" / state_path.name)
        (staging / "manifests").mkdir(exist_ok=True)
        attempted, failed = 0, set()
        with (staging / "manifests" / "rejected_frames.jsonl").open("w") as rejected:
            def reject(sequence, index, reason, kind="frame"):
                failed.add((sequence.sequence_id, index))
                rejected.write(json.dumps({"sequence_id": sequence.sequence_id, "frame_index": index,
                                           "kind": kind, "reason": reason}) + "\n")

            for sequence in sequences:
                tracker.reset(sequence.sequence_id)
                iterator = iter(iter_frames(sequence, stride))
                while batch := list(islice(iterator, batch_size)):
                    attempted += len(batch)
                    valid = [f for f in batch if f[2] is not None]
                    outputs = detector.predict([f[2] for f in valid], threshold=detector_threshold) if valid else []
                    if len(outputs) != len(valid):
                        raise ValueError("Detector must return exactly one detection dictionary per frame")
                    outputs = iter(outputs)
                    for index, stamp, image, source, error in batch:
                        if image is None:
                            tracker.update((), stamp)
                            reject(sequence, index, error)
                            continue
                        invalid = []
                        try:
                            detections = normalize_detections(next(outputs), image.size, class_ids, detector_threshold, invalid)
                        except (ValueError, TypeError) as exc:
                            tracker.update((), stamp)
                            reject(sequence, index, str(exc), "detection")
                            continue
                        for reason in invalid:
                            reject(sequence, index, reason, "annotation")
                        tracked = tracker.update(detections, stamp)
                        tracked = [d for d in tracked if d.score >= threshold or d.track_id is not None]
                        suffix = source.suffix.lower() if source else ".jpg"
                        relative = Path(sequence.split) / sequence.sequence_id / f"{index:06d}{suffix}"
                        target = staging / "images" / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if source is None:
                            image.save(target, quality=95)
                        elif mode == "symlink":
                            target.symlink_to(source)
                        elif mode == "hardlink":
                            os.link(source, target)
                        else:
                            shutil.copy2(source, target)
                        writer.add_frame(sequence, index, stamp, image.size, relative.as_posix(), tracked)
        validation = prep.get("validation", {})
        max_fraction = validation.get("max_failure_fraction", .05)
        max_failures = validation.get("max_failures", 100)
        if not 0 <= max_fraction <= 1 or max_failures < 0:
            raise ValueError("Validation failure limits must be a fraction in [0,1] and a non-negative count")
        if attempted == 0 or len(failed) > max_failures or len(failed) / attempted > max_fraction:
            raise ValueError(f"Preparation rejected {len(failed)}/{attempted} frames; failure threshold exceeded")
        # Catch source changes during an export, including in-place image edits.
        if provenance != [s.provenance() for s in sequences]:
            raise ValueError("Source files changed during preparation; rerun with immutable inputs")
        counts = writer.finish(validation.get("min_track_length_frames", 20))
        write_json(staging / "manifests" / "source_sequences.json", {"sequences": provenance})
        info = {"format_version": 1, "status": "complete", "dataset_version": destination.name,
                "created_at": creation_time(), "label_provenance": "detector_tracker_pseudo_labels",
                "coordinate_system": "image_normalized", "detector": {"adapter": getattr(detector, "name", type(detector).__name__),
                "version": getattr(detector, "version", "sova-0.1.0"), "checkpoint": checkpoint_info,
                "preprocessing": {k: config.get(k) for k in ("imgsz", "sparse_ids", "class_names", "category_ids")}},
                "category_mapping": {str(k): {"category_id": v[0], "name": v[1],
                    "detector_class_id": k, "original_category_id": original_category_ids[k] if k < len(original_category_ids) else k} for k, v in mapping.items()},
                "tracker": tracker_config, "confidence_threshold": threshold,
                "frame_stride": stride, "inference_batch_size": batch_size,
                "image_mode": mode, "source_manifest": fingerprint(manifest_path),
                "resolved_detector_config": config,
                "source_fps": {s.sequence_id: s.fps for s in sequences}, "counts": counts,
                "rejected_frame_count": len(failed), "attempted_frame_count": attempted,
                "annotation_fingerprints": {s: fingerprint(staging / "annotations" / f"instances_{s}.json")["sha256"] for s in counts}}
        write_json(staging / "manifests" / "prepared_dataset.json", info)
        with (staging / "resolved_config.yml").open("w") as stream:
            resolved = dict(config)
            resolved["dataset_preparation"] = {**prep, "source_manifest": str(Path(manifest_path).resolve()),
                                                "output_dir": str(destination), "tracker": tracker_config}
            yaml.safe_dump(resolved, stream, sort_keys=False)
        if destination.exists():
            backup = destination.with_name(f".{destination.name}-backup-{uuid.uuid4().hex}")
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup:
                backup.rename(destination)
            raise
        if backup:
            # Keep forced replacements recoverable; never recursively delete a
            # pre-existing version supplied by the caller.
            info["previous_version_backup"] = str(backup)
        return info
    finally:
        if staging.exists():
            shutil.rmtree(staging)
