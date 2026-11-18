"""
Teacher-labelled data for value-optimal training (engine distillation).

Prototype 1 learned the move a human happened to play. Here every position
is labelled with the teacher's score for EVERY legal move (MultiPV = all), so
the model can learn which moves are good, and by how much, rather than
which move someone chose.

Games are produced by the teacher itself, sampling moves from a softmax over
its own scores (hot in the opening for variety, cooler afterwards) plus a
small chance of a uniformly random move. The random moves create the
"hanging piece" positions the notes care about, so the model sees
blunders and learns to punish them.

Output: JSON lines, one per position:
  {"fen", "game", "ply", "moves": [uci...], "cp": [side-to-move cp per move]}

Usage:
  python datagen.py --out data/train.jsonl --games 400 --depth 8 --workers 2 --seed 1
  python datagen.py --out data/test.jsonl  --games 24  --depth 12 --workers 2 --seed 9001
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

import chess
import chess.engine

DEFAULT_ENGINE = os.environ.get("STOCKFISH", "stockfish")


def analyse_all(engine, board, limit):
    """Teacher score (side-to-move cp, mates mapped to +/-10000) for every legal move."""
    n = board.legal_moves.count()
    infos = engine.analyse(board, limit, multipv=n)
    out = {}
    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue
        mv = info["pv"][0]
        out[mv.uci()] = info["score"].pov(board.turn).score(mate_score=10000)
    # Very rarely MultiPV returns fewer lines than legal moves; score the rest singly.
    for mv in board.legal_moves:
        if mv.uci() not in out:
            info = engine.analyse(board, limit, root_moves=[mv])
            out[mv.uci()] = info["score"].pov(board.turn).score(mate_score=10000)
    return out


def expected_score(cp):
    cp = max(-2000, min(2000, cp))
    return 2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0


def sample_move(scores, ply, rng, eps):
    moves = list(scores)
    if ply >= 10 and rng.random() < eps:
        return rng.choice(moves), True
    temp = 0.12 if ply < 10 else 0.04
    vals = [expected_score(scores[m]) for m in moves]
    best = max(vals)
    w = [math.exp((v - best) / temp) for v in vals]
    return rng.choices(moves, weights=w, k=1)[0], False


def play_game(args):
    game_id, depth, seed, engine_path, max_plies, eps = args
    rng = random.Random(seed * 100003 + game_id)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1, "Hash": 16})
    limit = chess.engine.Limit(depth=depth)
    board = chess.Board()
    rows, decided = [], 0
    try:
        for ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            scores = analyse_all(engine, board, limit)
            rows.append({
                "fen": board.fen(),
                "game": game_id,
                "ply": ply,
                "moves": list(scores),
                "cp": [int(scores[m]) for m in scores],
            })
            best = max(scores.values())
            decided = decided + 1 if abs(best) >= 1500 else 0
            if decided >= 4:  # result no longer in doubt; stop spending teacher time
                break
            mv, _ = sample_move(scores, ply, rng, eps)
            board.push(chess.Move.from_uci(mv))
    finally:
        engine.quit()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-plies", type=int, default=150)
    ap.add_argument("--eps", type=float, default=0.03, help="chance of a random move after ply 10")
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jobs = [(g, a.depth, a.seed, a.engine, a.max_plies, a.eps) for g in range(a.games)]
    t0, n_pos = time.time(), 0
    with open(a.out, "w") as f, mp.Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(play_game, jobs), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n_pos += len(rows)
            f.flush()
            if i % 10 == 0 or i == a.games:
                dt = time.time() - t0
                print(f"games {i}/{a.games}  positions {n_pos}  {dt:.0f}s  "
                      f"({1000 * dt / max(1, n_pos):.0f} ms/pos)", flush=True)
    print("wrote", n_pos, "positions to", a.out)


if __name__ == "__main__":
    main()
