"""Gate 2: paired, seed-averaged comparison of run groups (e.g. B vs A, C vs A) on one dataset.

For each group, the per-position cp loss (and best-move / blunder indicators) is averaged over
its seeds, then the difference to the reference group is bootstrapped over whole games, as in
v6_eval. Negative cp_loss difference = the group loses fewer centipawns than the reference.

  python scripts/v6_paired_groups.py --data benchmarks/t2_p2_1312 --reference A_played \
      --group A_played --group B_soft --group C_value --out results/v6/gate2_t2.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chess_policy.v6_data import V6Dataset  # noqa: E402
from chess_policy.v6_eval import bootstrap, evaluate_model  # noqa: E402
from chess_policy.v6_model import load_any  # noqa: E402

KEYS = ("cp_loss", "best_rate", "blunder_rate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--group", action="append", required=True, help="run prefix, e.g. B_soft")
    ap.add_argument("--reference", action="append", required=True, help="repeatable")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--runs", default="runs/v6")
    ap.add_argument("--out")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = V6Dataset(a.data)
    per = {}
    for g in a.group:
        seeds = [evaluate_model(load_any(f"{a.runs}/{g}_s{s}/best.pt", device), ds, device)
                 for s in range(a.seeds)]
        per[g] = {k: np.mean([np.asarray(p[k], float) for p in seeds], axis=0) for k in KEYS}
    out = {"data": a.data, "rows": len(ds), "games": int(len(np.unique(ds.game))),
           "seeds": a.seeds, "groups": {g: bootstrap(v, ds.game) for g, v in per.items()}}
    for g, s in out["groups"].items():
        print(f"{g:14s} cp lost {s['cp_loss']['mean']:6.1f} [{s['cp_loss']['lo']:.1f}, {s['cp_loss']['hi']:.1f}]")
    for r in a.reference:
        ref = per[r]
        out[f"paired_vs_{r}"] = {g: bootstrap({k: v[k] - ref[k] for k in KEYS}, ds.game)
                                 for g, v in per.items() if g != r}
        for g, s in out[f"paired_vs_{r}"].items():
            d = s["cp_loss"]
            print(f"{g} - {r}: {d['mean']:+.2f} cp [{d['lo']:+.2f}, {d['hi']:+.2f}]  "
                  f"best-move {100 * s['best_rate']['mean']:+.2f} pts  blunders {100 * s['blunder_rate']['mean']:+.2f} pts")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
