"""Import verified PIE/JAAD/custom annotations exported as temporal COCO."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.data_prep.labelled_export import import_labelled_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument("--" + split, help=f"{split} COCO temporal annotations JSON")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--image-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--source-name", default="custom_human_annotations")
    args = parser.parse_args()
    annotations = {s: getattr(args, s) for s in ("train", "val", "test") if getattr(args, s)}
    result = import_labelled_dataset(annotations, args.image_root, args.output, args.fps,
                                     args.frame_stride, args.image_mode, args.source_name)
    print(f"Imported {args.output}: {result['counts']}")


if __name__ == "__main__":
    main()
