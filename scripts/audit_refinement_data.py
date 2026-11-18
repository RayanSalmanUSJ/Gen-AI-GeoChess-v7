"""Partition firewall for correction and exact-mate curricula, before training."""

import argparse
import json
from pathlib import Path

import chess

from chess_policy.corrections import read_keys
from chess_policy.data import atomic_json, sha256
from chess_policy.encoding import fingerprint
from chess_policy.mate_curriculum import mate_moves, orbit_group


def load_rows(root, split):
    return [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]


def rewrite(root, split, rows):
    path = root / f"{split}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["partitions"][split]["rows"] = len(rows)
    manifest["partitions"][split]["sha256"] = sha256(path)
    manifest.setdefault("postprocessing", []).append(
        "Partition firewall: diagnostic positions and cross-curriculum duplicates removed before training."
    )
    atomic_json(root / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, default=Path("corpus"))
    parser.add_argument("--heldout-data", type=Path, default=Path("holdout"))
    parser.add_argument("--corrections", type=Path, default=Path("v4_data/corrections"))
    parser.add_argument("--mates", type=Path, default=Path("v4_data/mates"))
    parser.add_argument("--diagnostic", type=Path, default=Path("v4_results/v3_loss_audit.json"))
    parser.add_argument("--out", type=Path, default=Path("v4_results/refinement_data_audit.json"))
    args = parser.parse_args()
    corrections = args.corrections
    mates = args.mates
    diagnostic = json.loads(args.diagnostic.read_text())
    diagnostic_keys = {fingerprint(chess.Board(r["fen"])).hex() for r in diagnostic["rows"]}
    counts = {}
    corrected = {}
    for split in ["train", "dev"]:
        rows = load_rows(corrections, split)
        kept = [r for r in rows if r["key"] not in diagnostic_keys]
        counts["correction_" + split + "_removed_diagnostic"] = len(rows) - len(kept)
        if len(kept) != len(rows):
            rewrite(corrections, split, kept)
        corrected[split] = kept
    correction_keys = {r["key"] for rows in corrected.values() for r in rows}
    mating = {}
    for split in ["train", "dev"]:
        rows = load_rows(mates, split)
        kept = [r for r in rows if r["key"] not in correction_keys | diagnostic_keys]
        counts["mate_" + split + "_removed_overlap"] = len(rows) - len(kept)
        if len(kept) != len(rows):
            rewrite(mates, split, kept)
        mating[split] = kept
    base = args.base_data
    heldout = args.heldout_data
    protected = {k.hex() for k in read_keys(base, {"test"}) | read_keys(heldout, {"test"})}
    train_keys = {k.hex() for k in read_keys(base, {"train"})}
    dev_keys = {k.hex() for k in read_keys(base, {"dev"})}
    for split, target_set in [("train", train_keys), ("dev", dev_keys)]:
        for row in corrected[split] + mating[split]:
            board = chess.Board(row["fen"])
            assert board.is_valid() and fingerprint(board).hex() == row["key"]
            targets = row.get("targets", [row.get("target")])
            assert all(chess.Move.from_uci(text) in board.legal_moves for text in targets)
            if "targets" in row:
                assert mate_moves(board) == row["targets"] and orbit_group(board) == row["group"]
            target_set.add(row["key"])
    assert not train_keys & dev_keys and not train_keys & protected and not dev_keys & protected
    assert not {r["group"] for r in mating["train"]} & {r["group"] for r in mating["dev"]}
    report = {
        "train_dev_exact_overlap": 0,
        "train_protected_test_overlap": 0,
        "dev_protected_test_overlap": 0,
        "mate_orbit_train_dev_overlap": 0,
        "removed": counts,
        "corrections": {s: len(r) for s, r in corrected.items()},
        "mates": {s: len(r) for s, r in mating.items()},
        "all_labels_legal": True,
        "all_mate_sets_exhaustively_verified": True,
    }
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
