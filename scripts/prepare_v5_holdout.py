"""Build a new 400-game reserve after every game used by v3 or v4."""

import argparse
import gzip
import json
from pathlib import Path
import shutil

import chess.pgn
import numpy as np

from chess_policy.corrections import read_keys
from chess_policy.data import atomic_json, build_corpus, read_manifest, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--pgn", type=Path, default=Path("v5_data/fresh_400.pgn.gz"))
    p.add_argument("--base-data", type=Path, default=Path("corpus"))
    p.add_argument(
        "--earlier-tests", type=Path, nargs="+", default=[Path("holdout"), Path("v4_holdout")]
    )
    p.add_argument("--out", type=Path, default=Path("v5_holdout"))
    p.add_argument("--report", type=Path, default=Path("v5_results/fresh_manifest.json"))
    args = p.parse_args()
    if args.source:
        if (
            sha256(args.source)
            != "bef40277bf1996beb8a025258922761ad4c52a47a294b2b8869ed65b78189e09"
        ):
            raise ValueError("unexpected full source archive")
        args.pgn.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.source, "rt") as stream, gzip.open(args.pgn, "wt") as out:
            for i in range(4100):
                game = chess.pgn.read_game(stream)
                if game is None or game.errors:
                    raise ValueError("missing or invalid source game")
                if i >= 3700:
                    out.write(
                        game.accept(
                            chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                        )
                        + "\n\n"
                    )
    build_corpus([args.pgn], args.out, force_test=True, exclude=args.base_data)
    blocked = set()
    groups = set()
    for root in args.earlier_tests:
        blocked.update(read_keys(root, {"test"}))
        groups.update(g["group"] for g in read_manifest(root)["games"])
    for root in [Path("v4_data"), Path("v5_data")]:
        for file in root.rglob("*.jsonl"):
            blocked.update(
                bytes.fromhex(json.loads(line)["key"]) for line in file.read_text().splitlines()
            )
    manifest = read_manifest(args.out, verify=True)
    shards = []
    count = removed = 0
    for shard in manifest["shards"]:
        arrays = {k: np.load(args.out / v["path"]) for k, v in shard["files"].items()}
        mask = np.array(
            [
                bytes(key) not in blocked and bytes(group).hex() not in groups
                for key, group in zip(arrays["keys"], arrays["groups"])
            ]
        )
        removed += len(mask) - int(mask.sum())
        if not mask.any():
            shutil.rmtree((args.out / shard["files"]["pieces"]["path"]).parent)
            continue
        shard["rows"] = int(mask.sum())
        count += shard["rows"]
        for name, array in arrays.items():
            path = args.out / shard["files"][name]["path"]
            np.save(path, array[mask], allow_pickle=False)
            shard["files"][name]["sha256"] = sha256(path)
        shards.append(shard)
    manifest.update(
        shards=shards,
        counts={"train": 0, "dev": 0, "test": count},
        additional_exclusions=removed,
        reservation="Source games 3701–4100; all earlier corpora, earlier opening groups and new curricula excluded.",
    )
    atomic_json(args.out / "manifest.json", manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.out / "manifest.json", args.report)
    print(json.dumps({"positions": count, "additional_exclusions": removed}, indent=2))


if __name__ == "__main__":
    main()
