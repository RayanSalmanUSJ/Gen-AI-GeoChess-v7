"""Final sparse-board test: every orbit is excluded from training and development."""

import argparse
import json
from pathlib import Path

import torch

from chess_policy.checkpoint import load_checkpoint
from chess_policy.data import atomic_json, sha256
from chess_policy.mate_curriculum import MateDataset
from chess_policy.refinement import measure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("v4_data/mate_test"))
    p.add_argument("--curriculum", type=Path, default=Path("v4_data/mates"))
    p.add_argument("--out", type=Path, default=Path("v4_results/mate_test.json"))
    args = p.parse_args()
    test = MateDataset(args.data, "test")
    blocked = set()
    for split in ["train", "dev"]:
        blocked.update(row["group"] for row in MateDataset(args.curriculum, split).rows)
    if len({r["group"] for r in test.rows}) != len(test) or any(
        r["group"] in blocked for r in test.rows
    ):
        raise ValueError("mate test orbit leakage")
    torch.set_num_threads(2)
    report = {}
    for name, path in [
        ("v3", "checkpoints/pilot.pt"),
        ("control", "checkpoints/control_best.pt"),
        ("targeted", "checkpoints/targeted_best.pt"),
    ]:
        model, _ = load_checkpoint(path)
        report[name] = {**measure(model, test), "checkpoint_sha256": sha256(path)}
    result = {
        "models": report,
        "excluded_train_dev_orbits": len(blocked),
        "test_orbit_overlap": 0,
        "data_sha256": sha256(args.data / "test.jsonl"),
        "limitation": "Synthetic pawnless mate-in-one test; not arbitrary tactics, conversion skill, or Elo.",
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
