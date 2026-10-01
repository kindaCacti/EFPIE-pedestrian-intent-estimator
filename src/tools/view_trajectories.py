"""Compute pedestrian trajectories while displaying a video in a desktop window."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import ImageDraw
import torch

from engine.config import YAMLConfig
from engine.data_prep.sequence_index import iter_frames, load_source_manifest
from engine.temporal.pipeline import TemporalInferencePipeline
from engine.trajectory import TRAJECTORY_ADAPTERS


def render_overlay(image, result, pipeline):
    """Render tracked pedestrians, measured histories and newly computed futures."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for detection in result["detections"]:
        if detection.track_id is None:
            continue  # Low-confidence unassociated recovery boxes are not tracks.
        draw.rectangle(detection.box_xyxy, outline="lime", width=3)
        label = f"ID {str(detection.track_id).rsplit(':', 1)[-1]}  {detection.score:.2f}"
        draw.text(detection.box_xyxy[:2], label, fill="lime")
    snapshot = pipeline.cache.snapshot()
    for prediction in result["predictions"]:
        history = pipeline.builder.history_for(snapshot, prediction.track_id)
        if history is None:
            continue
        observed = history["observed_points"][history["observed_mask"]]
        points = [(float(x) * image.width, float(y) * image.height) for x, y in observed]
        if len(points) > 1:
            draw.line(points, fill="yellow", width=3)
        paths = prediction.paths.detach().cpu().numpy()
        if paths.ndim == 2:
            paths = paths[None]
        for path in paths:
            future = [points[-1]] + [(float(x) * image.width, float(y) * image.height) for x, y in path]
            draw.line(future, fill="cyan", width=1 if len(paths) > 1 else 3)
            x, y = future[-1]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="cyan")
    return canvas


def resolve_checkpoint(config, explicit=None):
    """Honor selected model config; keep the original GRU command working."""
    return explicit or config.get("resume") or (
        "outputs/trajectory/gru/checkpoint_last.pth"
        if config["model"]["adapter"] == "gru_trajectory" else None)


class TrajectoryWindow:
    def __init__(self, cv2, width=1280, model_label=""):
        self.cv2, self.width = cv2, width
        self.name = "Pedestrian trajectory prediction"
        if model_label:
            self.name += " - " + model_label
        self.paused = False
        self.closed = False
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, width, round(width * 9 / 16))

    def show(self, image, status):
        cv2 = self.cv2
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if frame.shape[1] > self.width:
            height = round(frame.shape[0] * self.width / frame.shape[1])
            frame = cv2.resize(frame, (self.width, height), interpolation=cv2.INTER_AREA)
        frame = frame.copy()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (0, 0, 0), -1)
        cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        cv2.putText(frame, "Green: tracks   Yellow: history   Cyan: possible futures   Space: pause   Q/Esc: quit",
                    (12, 57), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.imshow(self.name, frame)

    def wait(self, milliseconds=1):
        """Service GUI events and pause without consuming any video frames."""
        cv2 = self.cv2
        while True:
            key = cv2.waitKey(max(1, milliseconds)) & 0xff
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                self.closed = True
                return False
            if key == ord(" "):
                self.paused = not self.paused
            if not self.paused:
                return True
            milliseconds = 30

    def close(self):
        self.cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-config", default="conf/rtdetrv4_pedestrian.yml")
    parser.add_argument("-c", "--config", default="conf/models/trajectory/gru/base.yml")
    parser.add_argument("-r", "--resume", help="Override the checkpoint in the selected model config")
    parser.add_argument("--source-manifest", default="conf/dataset/pie_sample_segments.yml")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--loop", action="store_true", help="Replay selected videos until Q/Esc or window close")
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.window_width < 640:
        parser.error("--threads must be positive and --window-width must be at least 640")
    trajectory_cfg = YAMLConfig(args.config, device=args.device)
    checkpoint = resolve_checkpoint(trajectory_cfg.yaml_cfg, args.resume)
    if checkpoint and not Path(checkpoint).is_file():
        parser.error(f"Trajectory checkpoint not found: {checkpoint}")
    model_name = trajectory_cfg.yaml_cfg["model"]["adapter"]
    if not checkpoint and model_name != "constant_velocity":
        parser.error("This learned model needs a checkpoint: set resume in config or pass --resume")
    if checkpoint:
        trajectory_cfg = YAMLConfig(args.config, device=args.device, resume=checkpoint)
    sequences = [s for s in load_source_manifest(args.source_manifest) if args.split == "all" or s.split == args.split]
    if not sequences:
        parser.error(f"No {args.split} sequences in {args.source_manifest}")
    stride = trajectory_cfg.yaml_cfg.get("temporal", {}).get("sequence", {}).get("frame_stride", 1)
    torch.set_num_threads(args.threads)
    torch.manual_seed(trajectory_cfg.yaml_cfg.get("seed", 42))
    import cv2
    from PIL import Image
    from engine.models import ADAPTERS

    options = trajectory_cfg.yaml_cfg.get("bitrap", {})
    model_label = "BiTraP-NP (pretrained PIE)" if options.get("pretrained_profile") == "pie_np" else model_name
    horizon = trajectory_cfg.yaml_cfg["temporal"]["sequence"]["prediction_frames"]
    samples = options.get("num_samples", 1) if options.get("variant") in {"np", "gmm"} else 1
    window = TrajectoryWindow(cv2, args.window_width, model_label)
    try:
        window.show(Image.new("RGB", (1280, 720)), "Loading detector and trained trajectory model...")
        if not window.wait():
            return
        detector_cfg = YAMLConfig(args.detector_config, device=args.device)
        detector = ADAPTERS[detector_cfg.yaml_cfg["model"]["adapter"]](detector_cfg)
        adapter = TRAJECTORY_ADAPTERS[trajectory_cfg.yaml_cfg["model"]["adapter"]](trajectory_cfg)
        detector.build()
        adapter.build()
        print("Live inference window ready. Space: pause/resume; Q/Esc: quit.", flush=True)
        print("Video timestamps are preserved; playback slows down if inference cannot keep up.", flush=True)
        print(f"{model_label}: {horizon} future frames, {samples} possible future(s).", flush=True)
        while not window.closed:
            processed = 0
            for sequence in sequences:
                # Fresh histories/identities at sequence changes and every replay.
                pipeline = TemporalInferencePipeline(detector, adapter, trajectory_cfg)
                frames = iter_frames(sequence, stride)
                try:
                    for index, stamp, image, _, error in frames:
                        if not window.wait():
                            break
                        if image is None:
                            print(f"Skipping {sequence.sequence_id} frame {index}: {error}", flush=True)
                            continue
                        start = perf_counter()
                        result = pipeline.process(image, sequence.sequence_id, index, stamp)
                        elapsed = perf_counter() - start
                        canvas = render_overlay(image, result, pipeline)
                        status = (f"{model_label} | +{horizon}f ({horizon * stride / sequence.fps:.1f}s), {samples} paths | "
                                  f"frame {index} | video {stamp:.2f}s | "
                                  f"{1 / max(elapsed, 1e-9):.1f} inference FPS | "
                                  f"{len(result['predictions'])} predicted tracks")
                        window.show(canvas, status)
                        processed += 1
                        if processed == 1 or processed % 30 == 0:
                            print(status, flush=True)
                        # Never skip source frames: trained temporal sampling stays intact.
                        remaining = stride / sequence.fps - (perf_counter() - start)
                        if not window.wait(round(max(0, remaining) * 1000)):
                            break
                finally:
                    frames.close()
                if window.closed:
                    break
            if window.closed or not args.loop:
                break
            if not processed:
                raise ValueError("No readable frames to replay")
    finally:
        window.close()


if __name__ == "__main__":
    main()
