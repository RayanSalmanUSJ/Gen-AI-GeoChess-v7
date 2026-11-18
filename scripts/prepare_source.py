"""Create reproducible compact PGNs from the checksum-pinned Fishtest source."""

import argparse
import gzip
import json
from pathlib import Path
import chess.pgn
from chess_policy.data import sha256

EXPECTED = "bef40277bf1996beb8a025258922761ad4c52a47a294b2b8869ed65b78189e09"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/source.pgn.gz"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    if sha256(args.source) != EXPECTED:
        raise ValueError("unexpected source checksum")
    args.out.mkdir(parents=True, exist_ok=True)
    counts = [0, 0]
    with gzip.open(args.source, "rt") as src:
        for batch, count in enumerate([2500, 800]):
            dest = args.out / ("fishtest_train.pgn.gz" if batch == 0 else "fishtest_holdout.pgn.gz")
            with gzip.open(dest, "wt") as dst:
                for _ in range(count):
                    game = chess.pgn.read_game(src)
                    if game is None or game.errors:
                        raise ValueError("source ended early or contains invalid game")
                    counts[batch] += sum(1 for _ in game.mainline_moves())
                    dst.write(
                        game.accept(
                            chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                        )
                        + "\n\n"
                    )
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": EXPECTED,
                "source_url": "https://huggingface.co/datasets/official-stockfish/fishtest_pgns/resolve/main/25-01-01/677492ae1a2f267f20548afc/677492ae1a2f267f20548afc.pgn.gz",
                "games": [2500, 800],
                "mainline_moves": counts,
                "transformation": "PGN headers and mainline retained; annotations and variations removed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
