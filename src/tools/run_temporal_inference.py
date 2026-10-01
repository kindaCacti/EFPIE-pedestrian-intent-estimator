"""Run temporal inference over a sequence manifest; write paths and telemetry."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from engine.config import YAMLConfig, yaml_utils, dataset_overrides
from engine.data_prep.manifest import fingerprint, write_json
from engine.data_prep.sequence_index import iter_frames, load_source_manifest
from engine.temporal.pipeline import TemporalInferencePipeline
from engine.trajectory import TRAJECTORY_ADAPTERS


def json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def render(image, result):
    """Optional rendering outside the temporal/model APIs."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    for det in result["detections"]:
        draw.rectangle(det.box_xyxy, outline="lime", width=2)
        draw.text(det.box_xyxy[:2], str(det.track_id), fill="lime")
    for prediction in result["predictions"]:
        paths = prediction.paths
        if paths.ndim == 2:
            paths = paths.unsqueeze(0)
        for path in paths:
            points = [(float(x) * image.width, float(y) * image.height) for x, y in path]
            if len(points) > 1:
                draw.line(points, fill="cyan", width=2)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("-c", "--config", required=True, help="Trajectory YAML")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--detector-checkpoint")
    parser.add_argument("-r", "--resume", help="Trajectory checkpoint")
    parser.add_argument("--dataset-config", help="Detector category metadata YAML")
    parser.add_argument("--output", default="outputs/temporal")
    parser.add_argument("--device")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--progress-every", type=int, default=30,
                        help="Print progress every N processed frames; 0 disables it")
    parser.add_argument("-u", "--update", nargs="+")
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    updates = yaml_utils.parse_cli(args.update)
    if args.device:
        updates["device"] = args.device
    if args.resume:
        updates["resume"] = args.resume
    trajectory_cfg = YAMLConfig(args.config, **updates)
    detector_updates = dataset_overrides(args.dataset_config) if args.dataset_config else {}
    if args.detector_checkpoint:
        detector_updates["resume"] = args.detector_checkpoint
    if args.device:
        detector_updates["device"] = args.device
    detector_cfg = YAMLConfig(args.detector_config, **detector_updates)
    from engine.models import ADAPTERS
    detector = ADAPTERS[detector_cfg.yaml_cfg["model"]["adapter"]](detector_cfg)
    adapter = TRAJECTORY_ADAPTERS[trajectory_cfg.yaml_cfg["model"]["adapter"]](trajectory_cfg)
    pipeline = TemporalInferencePipeline(detector, adapter, trajectory_cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed = trajectory_cfg.yaml_cfg.get("seed", 42)
    torch.manual_seed(seed)
    detector.build()
    actual_detector_checkpoint = args.detector_checkpoint
    if not actual_detector_checkpoint:
        actual_detector_checkpoint = detector.save(output / "detector_checkpoint.pth")
    with (output / "resolved_config.yml").open("w") as stream:
        yaml.safe_dump({"detector": {k: v for k, v in detector_cfg.yaml_cfg.items() if k != "__include__"},
                        "trajectory": {k: v for k, v in trajectory_cfg.yaml_cfg.items() if k != "__include__"}}, stream, sort_keys=False)
    write_json(output / "experiment_metadata.json", {"evaluation_mode": "detected_history",
               "label_provenance": "detector_tracker_observations", "source_manifest": fingerprint(args.source_manifest),
               "detector_checkpoint": fingerprint(actual_detector_checkpoint),
               "trajectory_checkpoint": fingerprint(args.resume) if args.resume else None,
               "tracker": trajectory_cfg.yaml_cfg.get("temporal", {}).get("tracker")})
    stride = trajectory_cfg.yaml_cfg.get("temporal", {}).get("sequence", {}).get("frame_stride", 1)
    last_statistics = None
    with (output / "predictions.jsonl").open("w") as predictions, (output / "rejected_frames.jsonl").open("w") as rejected:
        for sequence in load_source_manifest(args.source_manifest):
            for index, stamp, image, _, error in iter_frames(sequence, stride):
                if image is None:
                    rejected.write(json.dumps({"sequence_id": sequence.sequence_id, "frame_index": index, "reason": error}) + "\n")
                    continue
                result = pipeline.process(image, sequence.sequence_id, index, stamp)
                last_statistics = result["statistics"]
                serial = {**result, "detections": [asdict(d) for d in result["detections"]],
                          "predictions": [asdict(p) for p in result["predictions"]]}
                predictions.write(json.dumps(serial, default=json_value, allow_nan=False) + "\n")
                if args.render:
                    history_image = render(image.copy(), result)
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(history_image)
                    for pred in result["predictions"]:
                        history = pipeline.builder.history_for(pipeline.cache.snapshot(), pred.track_id)
                        points = history["observed_points"][history["observed_mask"]]
                        if len(points) > 1:
                            draw.line([(float(x) * image.width, float(y) * image.height) for x, y in points], fill="yellow", width=2)
                    folder = output / "rendered" / sequence.sequence_id
                    folder.mkdir(parents=True, exist_ok=True)
                    history_image.save(folder / f"{index:06d}.jpg")
                if args.progress_every and pipeline.frames_processed % args.progress_every == 0:
                    print(f"Processed {pipeline.frames_processed} frames; {sequence.sequence_id} "
                          f"frame {index}, predictions={len(result['predictions'])}", flush=True)
    write_json(output / "inference_statistics.json", last_statistics or {"frames_processed": 0})
    print(f"Temporal results written to {output}")


if __name__ == "__main__":
    main()
