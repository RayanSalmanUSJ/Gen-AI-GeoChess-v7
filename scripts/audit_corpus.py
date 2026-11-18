"""Check all hashes, legal labels, duplicate keys and opening-group isolation."""

import argparse
import json
from pathlib import Path
import numpy as np
from chess_policy.data import read_manifest, atomic_json


def audit(roots):
    seen = set()
    group_splits = {}
    reports = []
    for index, root in enumerate(roots):
        manifest = read_manifest(root, verify=True)
        counts = {}
        for game in manifest["games"]:
            group = game["group"]
            split = f"{index}:{game['split']}"
            if group in group_splits and group_splits[group] != split:
                raise ValueError("opening group straddles partitions")
            group_splits[group] = split
        for shard in manifest["shards"]:
            arrays = {
                k: np.load(root / v["path"], mmap_mode="r") for k, v in shard["files"].items()
            }
            if not ((arrays["legal"] == arrays["target"][:, None]).sum(1) == 1).all():
                raise ValueError("illegal or duplicated target")
            for key in arrays["keys"]:
                key = bytes(key)
                if key in seen:
                    raise ValueError("duplicate position across corpus partitions")
                seen.add(key)
            counts[shard["split"]] = counts.get(shard["split"], 0) + shard["rows"]
        reports.append({"root": str(root), "counts": counts})
    return {
        "corpora": reports,
        "unique_positions": len(seen),
        "opening_groups": len(group_splits),
        "duplicate_positions": 0,
        "illegal_targets": 0,
        "group_partition_overlap": 0,
        "all_shard_checksums_verified": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("roots", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path("results/data_audit.json"))
    a = p.parse_args()
    r = audit(a.roots)
    atomic_json(a.out, r)
    print(json.dumps(r, indent=2))
