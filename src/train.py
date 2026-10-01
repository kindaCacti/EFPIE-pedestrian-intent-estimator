"""Train or evaluate from YAML; supports trajectory and existing detectors."""

import argparse

from engine.config import YAMLConfig, yaml_utils, dataset_overrides
from engine.solver import TASKS


def main(args):
    if args.tuning and args.resume:
        raise ValueError("Choose --resume or --tuning, not both")
    updates = dataset_overrides(args.dataset_config) if args.dataset_config else {}
    updates = yaml_utils.merge_dict(updates, yaml_utils.parse_cli(args.update))
    updates.update({k: v for k, v in vars(args).items() if k not in {"update", "dataset_config", "config"} and v is not None})
    cfg = YAMLConfig(args.config, **updates)
    task = cfg.yaml_cfg.get("task", "trajectory")
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose {sorted(TASKS)}")
    solver = TASKS[task](cfg)
    return solver.val() if args.test_only else solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-d", "--dataset-config")
    parser.add_argument("-o", "--output-dir")
    parser.add_argument("-r", "--resume")
    parser.add_argument("-t", "--tuning")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-amp", action="store_true", default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("-u", "--update", nargs="+")
    main(parser.parse_args())
