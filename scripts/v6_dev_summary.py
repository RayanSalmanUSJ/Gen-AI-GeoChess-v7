"""Seed-averaged dev cp loss for every run group in runs/v6 (run from the repo root)."""
import collections
import json
from pathlib import Path

import numpy as np

groups = collections.defaultdict(list)
for p in sorted(Path("runs/v6").glob("*/metrics.json")):
    name = p.parent.name
    base = name.rsplit("_s", 1)[0] if name.rsplit("_s", 1)[-1].isdigit() else name
    m = json.loads(p.read_text())
    groups[base].append((m["best_dev_score"], m.get("minutes") or 0, m.get("best_epoch")))
for k, v in sorted(groups.items(), key=lambda kv: np.mean([x[0] for x in kv[1]])):
    s = [x[0] for x in v]
    print(f"{k:16s} n={len(s)}  dev cp lost {np.mean(s):6.1f} (sd {np.std(s):.1f})  "
          f"min/run {np.mean([x[1] for x in v]):4.0f}  best epochs {[x[2] for x in v]}")
