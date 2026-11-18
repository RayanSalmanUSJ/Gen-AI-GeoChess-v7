"""Development-only retention repair through same-architecture parameter interpolation."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from chess_policy.checkpoint import load_checkpoint, save_checkpoint
from chess_policy.corrections import CorrectionDataset
from chess_policy.data import ShardDataset, atomic_json, sha256
from chess_policy.refinement import measure
from chess_policy.tactics import TacticDataset


def interpolate(parent, child, alpha):
    if not 0 <= alpha <= 1 or parent.keys() != child.keys():
        raise ValueError("incompatible interpolation")
    out = {}
    for name, first in parent.items():
        last = child[name]
        if first.shape != last.shape or first.dtype != last.dtype:
            raise ValueError("state shapes/dtypes disagree")
        if first.is_floating_point():
            out[name] = torch.lerp(first, last, alpha)
        elif torch.equal(first, last):
            out[name] = first.clone()
        else:
            raise ValueError("nonfloating state differs")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-data", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("v5_runs"))
    p.add_argument("--out", type=Path, default=Path("v5_results/selection.json"))
    p.add_argument("--destination", type=Path, default=Path("checkpoints"))
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("frozen selection already exists")
    torch.set_num_threads(2)
    parent_path = Path("checkpoints/policy_v4.pt")
    parent, parent_payload = load_checkpoint(parent_path)
    child, child_payload = load_checkpoint(args.runs / "targeted/final.pt")
    _, control_payload = load_checkpoint(args.runs / "control/best.pt")
    if not (args.runs / "control/final.pt").exists():
        raise RuntimeError("control is not complete")
    for payload in [child_payload, control_payload]:
        if payload["metadata"]["parent_sha256"] != sha256(parent_path):
            raise ValueError("refinement parent mismatch")
    run = json.loads((args.runs / "targeted/run.json").read_text())
    initial = run["initial"]
    ceiling = initial["partitions"]["original"]["nll"] + run["config"]["original_nll_tolerance"]
    sets = {
        "original": ShardDataset(args.base_data, "dev", verify=True),
        "correction": CorrectionDataset("v4_data/corrections", "dev"),
        "mate": TacticDataset("v5_data/tactics", "dev", "mate"),
        "defense": TacticDataset("v5_data/tactics", "dev", "defense"),
    }
    weights = run["selection_weights"]
    options = [
        {
            "name": "v4",
            "path": str(parent_path),
            "alpha": 0.0,
            "step": 0,
            "dev": initial,
            "qualifies": True,
        },
        {
            "name": "control",
            "path": str(args.runs / "control/best.pt"),
            "step": control_payload["metadata"]["step"],
            "dev": control_payload["metadata"]["dev"],
            "qualifies": control_payload["metadata"]["dev"]["partitions"]["original"]["nll"]
            <= ceiling,
        },
    ]
    directory = args.runs / "interpolation"
    directory.mkdir(exist_ok=True)
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        if alpha == 1:
            dev = child_payload["metadata"]["dev"]
            path = args.runs / "targeted/final.pt"
        else:
            child.load_state_dict(
                interpolate(parent_payload["state"], child_payload["state"], alpha)
            )
            values = {
                key: measure(child, data, limit=12000 if key == "original" else 0)
                for key, data in sets.items()
            }
            dev = {
                "partitions": values,
                "weighted_objective": sum(weights[k] * values[k]["nll"] for k in weights),
            }
            path = directory / f"alpha_{int(alpha * 100):03d}.pt"
            save_checkpoint(
                path,
                child,
                {
                    "parent_sha256": sha256(parent_path),
                    "endpoint_sha256": sha256(args.runs / "targeted/final.pt"),
                    "alpha": alpha,
                    "dev": dev,
                    "step": run["config"]["steps"],
                    "method": "parameter interpolation selected only on development data",
                },
            )
        option = {
            "name": "targeted",
            "path": str(path),
            "alpha": alpha,
            "step": run["config"]["steps"],
            "dev": dev,
            "qualifies": dev["partitions"]["original"]["nll"] <= ceiling,
        }
        options.append(option)
        print(json.dumps(option), flush=True)
    feasible = [o for o in options if o["qualifies"]]
    selected = min(feasible, key=lambda o: o["dev"]["weighted_objective"])
    targeted = min(
        [o for o in feasible if o["name"] in ("v4", "targeted")],
        key=lambda o: o["dev"]["weighted_objective"],
    )
    args.destination.mkdir(parents=True, exist_ok=True)
    if (args.destination / "policy_v4.pt").resolve() != parent_path.resolve():
        shutil.copy2(parent_path, args.destination / "policy_v4.pt")
        shutil.copy2(str(parent_path) + ".sha256", args.destination / "policy_v4.pt.sha256")
    for option, dest in [
        (selected, "policy_v5.pt"),
        (targeted, "v5_targeted_best.pt"),
        (options[1], "v5_control_best.pt"),
    ]:
        shutil.copy2(option["path"], args.destination / dest)
        shutil.copy2(option["path"] + ".sha256", args.destination / (dest + ".sha256"))
    report = {
        "selected": selected["name"],
        "alpha": selected.get("alpha"),
        "step": selected["step"],
        "selected_sha256": sha256(args.destination / "policy_v5.pt"),
        "original_nll_ceiling": ceiling,
        "alternatives": options,
        "test_metrics_used": False,
        "arena_metrics_used": False,
        "adaptation": "The initial dev curves violated retention; alpha grid 0,.25,.5,.75,1 was added using dev only before final tests. Retention tolerance was unchanged.",
    }
    atomic_json(args.out, report)
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "alpha": selected.get("alpha"),
                "objective": selected["dev"]["weighted_objective"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
