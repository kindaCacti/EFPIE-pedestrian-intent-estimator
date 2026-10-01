"""Prepare an immutable detector/ByteTrack-derived trajectory dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import YAMLConfig, yaml_utils, dataset_overrides
from engine.data_prep import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="Existing detector YAML")
    parser.add_argument("-d", "--dataset-config", help="Detector category metadata YAML")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("-r", "--resume", help="Detector checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true", help="Replace version; previous directory is retained as a backup")
    parser.add_argument("-u", "--update", nargs="+")
    args = parser.parse_args()
    updates = dataset_overrides(args.dataset_config) if args.dataset_config else {}
    updates = yaml_utils.merge_dict(updates, yaml_utils.parse_cli(args.update))
    updates.update({k: v for k, v in {"resume": args.resume, "device": args.device}.items() if v is not None})
    cfg = YAMLConfig(args.config, **updates)
    from engine.models import ADAPTERS
    detector = ADAPTERS[cfg.yaml_cfg["model"]["adapter"]](cfg)
    info = prepare_dataset(detector, cfg, args.source_manifest, args.output, args.force)
    print(f"Prepared {args.output}: {info['counts']}")
    if info.get("previous_version_backup"):
        print("Previous version retained at " + info["previous_version_backup"])


if __name__ == "__main__":
    main()
