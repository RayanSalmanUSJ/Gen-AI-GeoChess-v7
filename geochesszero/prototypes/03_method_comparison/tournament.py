"""
Round-robin tournament between methods, with Elo estimates.

Every pair plays each opening twice (colours reversed). Openings are the first
8 moves of held-out test games, so no method trained on them. A game ends by
the rules or at 200 plies (then a draw). Elo is fitted by maximum likelihood
(Bradley-Terry, draws = half a win) and anchored at Random moves = 0;
intervals come from resampling games.

Usage:
  python tournament.py --config tournament_players.json --openings 8 --workers 2 \
      --out results/tournament.json
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "02_value_distillation")
sys.path.insert(0, HERE)

from gpt_movelm import game_sequences  # noqa: E402

_PLAYERS = {}


def _player(spec):
    if spec not in _PLAYERS:
        import torch
        torch.set_num_threads(1)
        from players import make_player
        _PLAYERS[spec] = make_player(spec)
    return _PLAYERS[spec]


def play_game(job):
    white_spec, black_spec, opening, opening_id, max_plies = job
    w, b = _player(white_spec), _player(black_spec)
    board = chess.Board()
    for m in opening:
        board.push(m)
    t0 = time.time()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mv = (w if board.turn == chess.WHITE else b).move(board)
        board.push(mv)
    res = board.result(claim_draw=True)
    if res == "*":
        res = "1/2-1/2"
    return dict(white=white_spec, black=black_spec, opening=opening_id, result=res,
                plies=board.ply(), seconds=round(time.time() - t0, 1),
                moves=[m.uci() for m in board.move_stack])


def fit_elo(names, games, iters=2000):
    """Bradley-Terry MLE on game scores; Elo anchored so names[0] = 0."""
    k = {n: i for i, n in enumerate(names)}
    r = np.zeros(len(names))
    W = np.zeros((len(names), len(names)))
    N = np.zeros((len(names), len(names)))
    for g in games:
        i, j = k[g["white"]], k[g["black"]]
        s = {"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5)
        W[i, j] += s
        W[j, i] += 1 - s
        N[i, j] += 1
        N[j, i] += 1
    gamma = np.ones(len(names))
    for _ in range(iters):  # minorise-maximise updates (Hunter 2004), with a tiny prior
        num = W.sum(1) + 0.5
        den = np.array([sum(N[i, j] / (gamma[i] + gamma[j]) for j in range(len(names)) if N[i, j])
                        for i in range(len(names))]) + 1.0 / (gamma + 1.0)
        gamma = num / den
        gamma /= gamma[0]
    r = 400 * np.log10(gamma)
    return r - r[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON list of {name, spec}")
    ap.add_argument("--openings", type=int, default=8)
    ap.add_argument("--opening-ply", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    players = json.load(open(a.config))
    names = [p["name"] for p in players]
    spec_of = {p["name"]: p["spec"] for p in players}
    seqs = game_sequences(os.path.join(P2, "data", "test_games.jsonl"))
    openings = [seqs[g][:a.opening_ply] for g in sorted(seqs)[:a.openings]]
    jobs = []
    for x, y in itertools.combinations(names, 2):
        for oi, op in enumerate(openings):
            jobs.append((spec_of[x], spec_of[y], op, oi, a.max_plies))
            jobs.append((spec_of[y], spec_of[x], op, oi, a.max_plies))
    print(f"{len(names)} players, {len(openings)} openings, {len(jobs)} games", flush=True)
    t0 = time.time()
    games = []
    with mp.Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(play_game, jobs, chunksize=1), 1):
            games.append(g)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} games, {time.time() - t0:.0f}s", flush=True)
    name_of = {p["spec"]: p["name"] for p in players}
    for g in games:
        g["white"], g["black"] = name_of[g["white"]], name_of[g["black"]]

    elo = fit_elo(names, games)
    rng = np.random.RandomState(0)
    boots = np.array([fit_elo(names, [games[i] for i in rng.randint(0, len(games), len(games))], 300)
                      for _ in range(a.boot)])
    table = []
    for i, n in enumerate(names):
        mine = [g for g in games if n in (g["white"], g["black"])]
        pts = sum({"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5) if g["white"] == n
                  else 1 - {"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5) for g in mine)
        table.append(dict(name=n, spec=spec_of[n], games=len(mine), points=pts,
                          score=pts / len(mine), elo=float(elo[i]),
                          elo_lo=float(np.percentile(boots[:, i], 2.5)),
                          elo_hi=float(np.percentile(boots[:, i], 97.5))))
    table.sort(key=lambda r: -r["elo"])
    cross = {x: {y: None for y in names} for x in names}
    for x, y in itertools.permutations(names, 2):
        gs = [g for g in games if {g["white"], g["black"]} == {x, y}]
        if gs:
            cross[x][y] = sum({"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5) if g["white"] == x
                              else 1 - {"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5) for g in gs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(players=players, openings=a.openings, table=table, crosstable=cross,
                   games=games), open(a.out, "w"), indent=1)
    print(f"\n{'player':34s} {'score':>6s} {'Elo (Random = 0)':>22s}")
    for r in table:
        print(f"{r['name']:34s} {100 * r['score']:5.1f}% {r['elo']:+7.0f} [{r['elo_lo']:+5.0f}, {r['elo_hi']:+5.0f}]")


if __name__ == "__main__":
    main()
