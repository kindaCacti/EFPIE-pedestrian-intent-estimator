"""Import verified annotations through the same COCO temporal contract."""

import json
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image
import yaml

from .detection_writer import validate_coco, validate_split_isolation
from .manifest import creation_time, fingerprint, sha256_file, write_json


def import_labelled_dataset(annotation_files, image_root, output_dir, fps=30,
                            frame_stride=1, image_mode="copy", source_name="custom_human_annotations"):
    destination = Path(output_dir).absolute()
    source_root = Path(image_root).resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"Dataset version {destination} already exists")
    if fps <= 0 or frame_stride <= 0 or image_mode not in {"copy", "hardlink", "symlink"}:
        raise ValueError("Require positive fps/stride and a supported image_mode")
    if not annotation_files or set(annotation_files) - {"train", "val", "test"}:
        raise ValueError("Annotation files must be mapped to train/val/test splits")
    exports, input_fingerprints = {}, {}
    categories = None
    for split, path in annotation_files.items():
        with open(path) as stream:
            data = json.load(stream)
        validate_coco(data)
        if categories is not None and categories != data["categories"]:
            raise ValueError("Human annotation category mappings must agree across splits")
        categories = data["categories"]
        exports[split] = data
        input_fingerprints[split] = fingerprint(path)
    for split in ("train", "val", "test"):
        exports.setdefault(split, {"images": [], "annotations": [], "categories": categories})
    validate_split_isolation(list(exports.values()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-importing-", dir=destination.parent))
    sources, counts, source_fps = [], {}, {}
    seen_paths = set()
    try:
        for split, data in exports.items():
            # Reassign IDs globally across split JSONs; input IDs may restart.
            image_mapping = {}
            for image in data["images"]:
                relative = Path(image["file_name"])
                source = (source_root / relative).resolve(strict=True)
                if not source.is_relative_to(source_root):
                    raise ValueError("Annotation image escapes image_root")
                if source in seen_paths:
                    raise ValueError("Human export reuses a source image across sequences/splits")
                seen_paths.add(source)
                with Image.open(source) as decoded:
                    if decoded.size != (image["width"], image["height"]):
                        raise ValueError(f"Image dimensions disagree with annotations: {source}")
                    decoded.verify()
                image_mapping[image["id"]] = len(sources) + 1
                image["id"] = len(sources) + 1
                target_name = Path(split) / image["sequence_id"] / f"{image['frame_index']:06d}{source.suffix.lower()}"
                target = staging / "images" / target_name
                target.parent.mkdir(parents=True, exist_ok=True)
                if image_mode == "copy":
                    shutil.copy2(source, target)
                elif image_mode == "hardlink":
                    os.link(source, target)
                else:
                    target.symlink_to(source)
                image["file_name"] = target_name.as_posix()
                sources.append({**fingerprint(source), "sequence_id": image["sequence_id"], "split": split})
                source_fps[image["sequence_id"]] = fps
            previous_annotations = sum(count["annotations"] for count in counts.values())
            for i, annotation in enumerate(data["annotations"]):
                annotation["id"] = previous_annotations + i + 1
                annotation["image_id"] = image_mapping[annotation["image_id"]]
            validate_coco(data)
            write_json(staging / "annotations" / f"instances_{split}.json", data)
            counts[split] = {"images": len(data["images"]), "annotations": len(data["annotations"]),
                             "tracks": len({a["track_id"] for a in data["annotations"] if a["track_id"] is not None})}
        info = {"format_version": 1, "status": "complete", "dataset_version": destination.name,
                "created_at": creation_time(), "label_provenance": "human_ground_truth",
                "coordinate_system": "image_normalized", "source_name": source_name,
                "source_fps": source_fps, "frame_stride": frame_stride, "image_mode": image_mode,
                "counts": counts, "input_annotations": input_fingerprints,
                "annotation_fingerprints": {s: sha256_file(staging / "annotations" / f"instances_{s}.json") for s in exports}}
        write_json(staging / "manifests" / "prepared_dataset.json", info)
        write_json(staging / "manifests" / "source_sequences.json", {"images": sources})
        (staging / "manifests" / "rejected_frames.jsonl").touch()
        with (staging / "resolved_config.yml").open("w") as stream:
            yaml.safe_dump({"labelled_import": {"annotations": {k: str(v) for k, v in annotation_files.items()},
                           "fps": fps, "frame_stride": frame_stride, "image_mode": image_mode,
                           "source_name": source_name, "image_root": str(source_root)}} , stream)
        staging.rename(destination)
        return info
    finally:
        if staging.exists():
            shutil.rmtree(staging)
