"""Copy a range of games from a (gzipped) PGN, e.g. the Fishtest source archive.

v5 used games 1-2500 (training), 2501-3300 (holdout, T3) and 3301-4100 (v4/v5 holdouts),
so extra training games for D2 start after game 4100:

  python scripts/v6_extract_games.py data/source.pgn.gz --skip 4100 --count 6500 \
      --out data/fishtest_extra.pgn.gz
"""
import argparse
import gzip

import chess.pgn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--skip", type=int, required=True)
    ap.add_argument("--count", type=int, default=0, help="0 = everything after --skip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    opener = gzip.open if a.source.endswith(".gz") else open
    read = written = 0
    with opener(a.source, "rt", encoding="utf-8") as src, gzip.open(a.out, "wt") as dst:
        while not a.count or written < a.count:
            game = chess.pgn.read_game(src)
            if game is None:
                break
            read += 1
            if read <= a.skip or game.errors:
                continue
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            dst.write(game.accept(exporter) + "\n\n")
            written += 1
    print(f"read {read} games, wrote {written} (games {a.skip + 1}..{a.skip + written}) to {a.out}")


if __name__ == "__main__":
    main()
