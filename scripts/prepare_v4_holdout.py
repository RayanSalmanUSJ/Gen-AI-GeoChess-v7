"""Reserve later source games and exclude every known training/development position."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, help="Optional full source to re-extract games 3301–3700."
    )
    parser.add_argument("--pgn", type=Path, default=Path("v4_data/fresh_400.pgn.gz"))
    parser.add_argument("--base-data", type=Path, default=Path("corpus"))
    parser.add_argument("--heldout-data", type=Path, default=Path("holdout"))
    parser.add_argument("--curricula", type=Path, default=Path("v4_data"))
    parser.add_argument("--out", type=Path, default=Path("v4_holdout"))
    parser.add_argument("--report", type=Path, default=Path("v4_results/fresh_manifest.json"))
    args = parser.parse_args()
    expected = "bef40277bf1996beb8a025258922761ad4c52a47a294b2b8869ed65b78189e09"
    compact = args.pgn
    if args.source:
        if sha256(args.source) != expected:
            raise ValueError("unexpected source")
        compact.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.source, "rt") as stream, gzip.open(compact, "wt") as out:
            for i in range(3700):
                game = chess.pgn.read_game(stream)
                if game is None or game.errors:
                    raise ValueError("source parse failure")
                if i >= 3300:
                    out.write(
                        game.accept(
                            chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                        )
                        + "\n\n"
                    )
    root = args.out
    build_corpus([compact], root, force_test=True, exclude=args.base_data)
    old_holdout = args.heldout_data
    excluded = read_keys(old_holdout, {"test"})
    old_groups = {g["group"] for g in read_manifest(old_holdout)["games"]}
    for folder in ["corrections", "mates"]:
        for split in ["train", "dev"]:
            for line in (args.curricula / folder / (split + ".jsonl")).read_text().splitlines():
                excluded.add(bytes.fromhex(json.loads(line)["key"]))
    manifest = read_manifest(root, verify=True)
    filtered = []
    removed = 0
    count = 0
    for shard in manifest["shards"]:
        arrays = {k: np.load(root / v["path"]) for k, v in shard["files"].items()}
        mask = np.array(
            [
                bytes(k) not in excluded and bytes(g).hex() not in old_groups
                for k, g in zip(arrays["keys"], arrays["groups"])
            ]
        )
        removed += len(mask) - int(mask.sum())
        if not mask.any():
            shutil.rmtree((root / shard["files"]["pieces"]["path"]).parent)
            continue
        shard["rows"] = int(mask.sum())
        count += shard["rows"]
        for key, array in arrays.items():
            path = root / shard["files"][key]["path"]
            np.save(path, array[mask], allow_pickle=False)
            shard["files"][key]["sha256"] = sha256(path)
        filtered.append(shard)
    manifest.update(
        shards=filtered,
        counts={"train": 0, "dev": 0, "test": count},
        additional_exclusions=removed,
        reservation="Source games 3301 through 3700; all v3/v4 train/dev positions and earlier test groups excluded.",
    )
    atomic_json(root / "manifest.json", manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "manifest.json", args.report)
    print(json.dumps({"fresh_positions": count, "additional_exclusions": removed}, indent=2))


if __name__ == "__main__":
    main()
