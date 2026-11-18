"""Verify all excerpt starting positions and mainline moves against the raw archive."""

import gzip
import json
from pathlib import Path
import chess.pgn
from chess_policy.data import atomic_json, sha256
from chess_policy.encoding import position_key


def main():
    counts = []
    with gzip.open("data/source.pgn.gz", "rt") as source:
        for name, count in [("fishtest_train.pgn.gz", 2500), ("fishtest_holdout.pgn.gz", 800)]:
            total = 0
            with gzip.open(Path("data") / name, "rt") as derived:
                for _ in range(count):
                    a, b = chess.pgn.read_game(source), chess.pgn.read_game(derived)
                    assert a is not None and b is not None and not a.errors and not b.errors
                    assert position_key(a.board()) == position_key(b.board())
                    ma, mb = list(a.mainline_moves()), list(b.mainline_moves())
                    assert ma == mb
                    total += len(ma)
                assert chess.pgn.read_game(derived) is None
            counts.append(
                {
                    "file": name,
                    "games": count,
                    "verified_moves": total,
                    "sha256": sha256(Path("data") / name),
                }
            )
    report = {"source_sha256": sha256("data/source.pgn.gz"), "excerpts": counts, "verified": True}
    atomic_json("results/source_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
