"""
Play matches between players and report W/D/L with an opening-pair bootstrap.

Players (spec strings):
  model:<run_dir>:policy       argmax of the policy head (no search)
  model:<run_dir>:comparator   1-ply least-bad move by the value head
  model:<run_dir>:search       2-ply minimax: policy top-6 candidates, all replies, value head at leaves
  sf:depth=<d> | sf:nodes=<n> | sf:skill=<0-20>
  random

Each opening is played twice with colours reversed. Openings are taken from
held-out test games at a fixed ply, so the models never trained on them.

Usage:
  python arena.py --a model:runs/ray_s0:policy --b model:runs/standard_s0:policy \
                  --openings data/test.jsonl --out results/arena_ray_vs_standard.json
"""

import argparse
import json
import os
import random

import chess
import chess.engine
import numpy as np
import torch

import encoding as enc
from evaluate import search_choices
from model import load_checkpoint

_MODELS = {}


class ModelPlayer:
    avoid_repetition = False  # set by --avoid-repetition (applies to every model player)

    def __init__(self, run, mode):
        if run not in _MODELS:
            _MODELS[run] = load_checkpoint(os.path.join(run, "best.pt"))[0]
        self.model, self.mode = _MODELS[run], mode
        self.name = f"{os.path.basename(run.rstrip('/'))}:{mode}"

    @torch.no_grad()
    def move(self, board):
        legal = list(board.legal_moves)
        if self.mode == "search":
            acts = [enc.action_index(board, m) for m in legal]
            row = np.array([acts], dtype=np.int64)
            j = int(search_choices(self.model, [board.fen()], row, np.array([len(acts)]))[0])
            return legal[j]
        if self.mode == "policy":
            x = torch.from_numpy(enc.encode(board)).unsqueeze(0)
            occ = torch.from_numpy(enc.occupancy(board)).unsqueeze(0)
            logits, _ = self.model(x, occ)
            idx = torch.tensor([enc.action_index(board, m) for m in legal])
            order = logits[0, idx].argsort(descending=True).tolist()
            if self.avoid_repetition:
                # The network never sees game history, so deterministic play loops.
                # Like an engine's repetition check: skip a move that repeats a
                # position if any non-repeating legal move exists.
                for j in order:
                    board.push(legal[j])
                    rep = board.is_repetition(2)
                    board.pop()
                    if not rep:
                        return legal[j]
            return legal[order[0]]
        # comparator: value of each child from the mover's point of view
        q, xs, occs, slots = np.full(len(legal), -np.inf), [], [], []
        for j, m in enumerate(legal):
            board.push(m)
            if board.is_checkmate():
                q[j] = 1.0
            elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                q[j] = 0.0
            else:
                xs.append(enc.encode(board))
                occs.append(enc.occupancy(board))
                slots.append(j)
            board.pop()
        if xs:
            _, v = self.model(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(occs)))
            q[slots] = -v.numpy()
        return legal[int(q.argmax())]


class EnginePlayer:
    def __init__(self, path, spec):
        key, val = spec.split("=")
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        opts = {"Threads": 1, "Hash": 16}
        if key == "skill":
            opts["Skill Level"] = int(val)
            self.limit = chess.engine.Limit(depth=8)
        else:
            self.limit = chess.engine.Limit(**{key: int(val)})
        self.engine.configure(opts)
        self.name = f"stockfish({spec})"

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


class RandomPlayer:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def move(self, board):
        return self.rng.choice(list(board.legal_moves))


def make_player(spec, engine_path):
    if spec.startswith("model:"):
        _, run, mode = spec.split(":")
        return ModelPlayer(run, mode)
    if spec.startswith("sf:"):
        return EnginePlayer(engine_path, spec[3:])
    if spec == "random":
        return RandomPlayer()
    raise ValueError(spec)


def play(white, black, fen, max_plies=200):
    board = chess.Board(fen)
    moves = []
    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        mv = (white if board.turn == chess.WHITE else black).move(board)
        moves.append(mv.uci())
        board.push(mv)
    res = board.result(claim_draw=True)
    if res == "*":
        res = "1/2-1/2"  # ply cap reached: adjudicated draw
    return res, moves, board.fen()


def openings_from(path, ply, limit):
    seen, out = set(), []
    for line in open(path):
        r = json.loads(line)
        if r["ply"] == ply and r["game"] not in seen:
            seen.add(r["game"])
            out.append(r["fen"])
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--openings", required=True)
    ap.add_argument("--opening-ply", type=int, default=8)
    ap.add_argument("--n-openings", type=int, default=20)
    ap.add_argument("--engine", default=os.environ.get("STOCKFISH", "stockfish"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--avoid-repetition", action="store_true",
                    help="model players skip moves that repeat a position when an alternative exists")
    a = ap.parse_args()
    torch.set_num_threads(1)
    ModelPlayer.avoid_repetition = a.avoid_repetition

    A, B = make_player(a.a, a.engine), make_player(a.b, a.engine)
    fens = openings_from(a.openings, a.opening_ply, a.n_openings)
    games, pair_scores = [], []
    for i, fen in enumerate(fens):
        pair = 0.0
        for a_white in (True, False):
            w, b = (A, B) if a_white else (B, A)
            res, moves, final = play(w, b, fen)
            score_a = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
            if not a_white:
                score_a = 1.0 - score_a
            pair += score_a
            games.append(dict(opening=i, fen=fen, a_white=a_white, result=res,
                              score_a=score_a, plies=len(moves), moves=moves))
        pair_scores.append(pair / 2)
    for p in (A, B):
        if hasattr(p, "close"):
            p.close()

    s = np.array([g["score_a"] for g in games])
    pair_scores = np.array(pair_scores)
    rng = np.random.RandomState(0)
    boots = [pair_scores[rng.randint(0, len(pair_scores), len(pair_scores))].mean()
             for _ in range(2000)]
    summary = dict(a=A.name, b=B.name, games=len(games), avoid_repetition=a.avoid_repetition,
                   w=int((s == 1).sum()), d=int((s == 0.5).sum()), l=int((s == 0).sum()),
                   score=float(s.mean()), lo=float(np.percentile(boots, 2.5)),
                   hi=float(np.percentile(boots, 97.5)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(summary=summary, games=games), open(a.out, "w"), indent=1)
    print(f"{A.name} vs {B.name}: +{summary['w']} ={summary['d']} -{summary['l']}  "
          f"score {100 * summary['score']:.1f}% [{100 * summary['lo']:.1f}, {100 * summary['hi']:.1f}]")


if __name__ == "__main__":
    main()
