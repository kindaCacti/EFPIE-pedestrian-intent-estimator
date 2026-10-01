"""Select bounded PIE segments for a CPU demonstration using sparse detection."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from PIL import Image
import torch
import yaml

from engine.config import YAMLConfig
from engine.models import ADAPTERS
from engine.data_prep.sequence_index import load_source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--output", required=True, help="New selected source-manifest YAML")
    parser.add_argument("--segment-frames", type=int, default=180)
    parser.add_argument("--scan-frames", type=int, default=20)
    args = parser.parse_args()
    if args.segment_frames <= 0 or args.scan_frames <= 0:
        parser.error("--segment-frames and --scan-frames must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Selected manifest already exists: {output}")
    torch.set_num_threads(4)
    detector = ADAPTERS["rtdetrv4"](YAMLConfig(args.detector_config, device="cpu"))
    detector.build()
    selected = []
    for sequence in load_source_manifest(args.source_manifest):
        if not sequence.video:
            raise ValueError("Segment selection expects video sequences")
        cap = cv2.VideoCapture(str(sequence.source))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_start, source_end = sequence.start_frame, sequence.end_frame or count
        candidates = []
        try:
            for n in range(args.scan_frames):
                index = source_start + int((n + .5) * (source_end - source_start) / args.scan_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = detector.predict([image], threshold=.5)[0]
                labels, scores, boxes = (result[k].detach().cpu() for k in ("labels", "scores", "boxes"))
                valid = (labels == 0) & (scores >= .5)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                quality = float((scores[valid] * areas[valid]).sum())
                candidates.append((quality, index, int(valid.sum())))
                print(f"{sequence.sequence_id}: frame {index}, pedestrians={int(valid.sum())}, quality={quality:.1f}", flush=True)
        finally:
            cap.release()
        if not candidates or max(candidates)[0] == 0:
            raise ValueError(f"No visible pedestrians found in {sequence.sequence_id}")
        quality, index, pedestrians = max(candidates)
        length = min(args.segment_frames, source_end - source_start)
        start = max(source_start, min(source_end - length, index - length // 2))
        selected.append({"sequence_id": sequence.sequence_id,
                         "video": os.path.relpath(sequence.source, output.parent), "fps": sequence.fps,
                         "split": sequence.split, "start_frame": start, "end_frame": start + length})
        print(f"Selected {sequence.sequence_id}: frames {start}:{start + length} ({pedestrians} pedestrians at centre)", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        yaml.safe_dump({"sequences": selected}, stream, sort_keys=False)


if __name__ == "__main__":
    main()
