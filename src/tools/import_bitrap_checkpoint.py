"""Convert verified official PIE BiTraP-NP weights into a trajectory checkpoint."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import YAMLConfig
from engine.trajectory.pretrained_bitrap import import_pie_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="models/bitrap/pie_np_k20.pth")
    parser.add_argument("-c", "--config", default="conf/models/trajectory/bitrap/pretrained_pie.yml")
    args = parser.parse_args(argv)
    provenance = import_pie_checkpoint(args.source, args.output, YAMLConfig(args.config, device="cpu"))
    print(f"Imported official {provenance['dataset']} K={provenance['trained_num_samples']} weights to {args.output}")
    print(f"SHA-256: {provenance['source_sha256']}")


if __name__ == "__main__":
    main()
