"""Audit exact-position, seed-family and legal-target separation before training."""

import argparse
import json
from pathlib import Path

import chess

from chess_policy.corrections import read_keys
from chess_policy.data import atomic_json, sha256
from chess_policy.encoding import fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("v5_data/tactics"))
    p.add_argument("--protected-roots", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("v5_results/data_audit.json"))
    args = p.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    keys = {}
    groups = {}
    counts = {}
    protected = set()
    for root in args.protected_roots:
        protected.update(k.hex() for k in read_keys(root, {"train", "dev", "test"}))
    for split in ["train", "dev"]:
        file = args.data / (split + ".jsonl")
        if sha256(file) != manifest["partitions"][split]["sha256"]:
            raise ValueError("tactical file checksum mismatch")
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        keys[split] = set()
        groups[split] = set()
        for row in rows:
            board = chess.Board(row["fen"])
            if not board.is_valid() or fingerprint(board).hex() != row["key"]:
                raise ValueError("invalid position")
            if not 8 <= len(board.piece_map()) <= 22 or len(chess.SquareSet(board.pawns)) < 2:
                raise ValueError("board outside new curriculum")
            legal = {m.uci() for m in board.legal_moves}
            targets = set(row["targets"])
            if not targets or not targets <= legal:
                raise ValueError("illegal/empty target set")
            if row["task"] == "defense" and (
                not row["unsafe"]
                or targets | set(row["unsafe"]) != legal
                or targets & set(row["unsafe"])
            ):
                raise ValueError("invalid defensive partition")
            if row["key"] in keys[split] or row["key"] in protected:
                raise ValueError("duplicate or protected position")
            keys[split].add(row["key"])
            groups[split].add(row["group"])
        counts[split] = len(rows)
    if keys["train"] & keys["dev"] or groups["train"] & groups["dev"]:
        raise ValueError("training/development leakage")
    report = {
        "rows": counts,
        "exact_position_overlap": 0,
        "seed_family_overlap": 0,
        "protected_position_overlap": 0,
        "all_targets_legal": True,
        "note": "Full target-set proofs are generated offline and recomputed on dataset access; this audit checks partitions and legal targets.",
    }
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
