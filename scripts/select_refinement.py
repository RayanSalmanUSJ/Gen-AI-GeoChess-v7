"""Select a checkpoint solely by the previously registered development criterion."""

import argparse
import json
from pathlib import Path
import shutil

from chess_policy.checkpoint import load_checkpoint
from chess_policy.data import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("v4_runs"))
    parser.add_argument("--baseline", type=Path, default=Path("checkpoints/pilot.pt"))
    parser.add_argument("--baseline-name", default="v3")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/policy_v4.pt"))
    parser.add_argument("--report", type=Path, default=Path("v4_results/selection.json"))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Selection is frozen; use a separate report for a new experiment.")
    run = json.loads((args.runs / "control/run.json").read_text())
    initial = run["initial"]
    options = [{"name": args.baseline_name, "path": str(args.baseline), "step": 0, "dev": initial}]
    ceiling = initial["partitions"]["original"]["nll"] + run["config"]["original_nll_tolerance"]
    for name in ["control", "targeted"]:
        root = args.runs / name
        if not (root / "final.pt").exists():
            raise RuntimeError(f"{name} training is not complete")
        _, payload = load_checkpoint(root / "best.pt")
        metadata = payload["metadata"]
        if metadata["parent_sha256"] != sha256(args.baseline):
            raise ValueError("parent checkpoint mismatch")
        if metadata["dev"]["partitions"]["original"]["nll"] > ceiling:
            raise ValueError("selected arm violates the retention constraint")
        options.append(
            {
                "name": name,
                "path": str(root / "best.pt"),
                "step": metadata["step"],
                "dev": metadata["dev"],
            }
        )
    selected = min(options, key=lambda item: item["dev"]["weighted_objective"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["path"], args.out)
    shutil.copy2(selected["path"] + ".sha256", str(args.out) + ".sha256")
    for option in options[1:]:
        destination = args.out.parent / (args.prefix + option["name"] + "_best.pt")
        shutil.copy2(option["path"], destination)
        shutil.copy2(option["path"] + ".sha256", str(destination) + ".sha256")
    report = {
        "selected": selected["name"],
        "step": selected["step"],
        "selected_sha256": sha256(args.out),
        "original_nll_ceiling": ceiling,
        "alternatives": options,
        "rule": "minimum weighted development NLL subject to retention constraint",
        "test_metrics_used": False,
        "arena_metrics_used": False,
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
