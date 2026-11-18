"""
Re-score a subsample of positions with a deeper teacher (for the test set).

Test games are played at the training depth (cheap), then every k-th
position is re-scored at a greater depth, so the benchmark's reference is
stronger than the labels the models learned from.

Usage:
  python relabel.py data/test_games.jsonl data/test.jsonl --depth 12 --every 2 --workers 2
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import chess
import chess.engine

from datagen import analyse_all

_ENGINE = None


def _init(path):
    global _ENGINE
    _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    _ENGINE.configure({"Threads": 1, "Hash": 32})


def _relabel(job):
    row, depth = job
    board = chess.Board(row["fen"])
    scores = analyse_all(_ENGINE, board, chess.engine.Limit(depth=depth))
    out = dict(row)
    out["moves"], out["cp"], out["depth"] = list(scores), [int(scores[m]) for m in scores], depth
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--min-ply", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--engine", default=os.environ.get("STOCKFISH", "stockfish"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    pick = [r for r in rows if r["ply"] >= a.min_ply and r["ply"] % a.every == 0]
    t0 = time.time()
    with open(a.out, "w") as f, mp.Pool(a.workers, initializer=_init, initargs=(a.engine,)) as pool:
        for i, r in enumerate(pool.imap(_relabel, [(r, a.depth) for r in pick], chunksize=4), 1):
            f.write(json.dumps(r) + "\n")
            if i % 200 == 0:
                print(f"{i}/{len(pick)} {time.time() - t0:.0f}s", flush=True)
    print(f"relabelled {len(pick)} of {len(rows)} positions at depth {a.depth} -> {a.out}")


if __name__ == "__main__":
    main()
