"""
Cross-evaluation: the team's Chess Policy System (v3 pilot / v4 / v5) against
Prototype 2's board transformers, on BOTH benchmarks, same positions for all.

  A. Move quality (Prototype 2 benchmark): 1,312 held-out positions priced by
     Stockfish 15.1 depth 12 -> centipawns lost, best-move rate, blunder rate.
  B. Recorded-move agreement (v5's own benchmark): positions from the reserved
     Fishtest holdout games -> top-1 / top-3 agreement with the move Stockfish
     actually played in those games.

Usage:
  python -I crosseval.py --v5-root /path/to/chess_policy_v5 --holdout-games 200 --out results/crosseval.json
"""

import argparse
import glob
import gzip
import json
import os
import sys

import chess
import chess.pgn
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "02_value_distillation")
ROOT = os.path.join(HERE, "..", "..")
for p in (P2, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import encoding as enc  # noqa: E402  (Prototype 2)
from evaluate import choice_metrics, summarize  # noqa: E402
from model import load_checkpoint as load_p2  # noqa: E402
from train import load_npz  # noqa: E402

METRICS = ("cp_loss", "best_rate", "blunder_rate", "es_loss")


# ---------------------------------------------------------------------------
# The two model families behind one interface: per-board logits over legal moves
# ---------------------------------------------------------------------------

class V5Family:
    def __init__(self, v5_root, ckpt):
        sys.path.insert(0, os.path.join(v5_root, "src"))
        from chess_policy.checkpoint import load_checkpoint
        from chess_policy.encoding import collate, encode, id_move
        from chess_policy.model import model_inputs
        self.collate, self.encode, self.id_move, self.inputs = collate, encode, id_move, model_inputs
        self.model, _ = load_checkpoint(ckpt)
        self.model.eval()

    @torch.no_grad()
    def scores(self, boards, bs=256):
        """List of dicts {move: logit} for each board."""
        out = []
        for s in range(0, len(boards), bs):
            chunk = boards[s:s + bs]
            encs = [self.encode(b, validate=False) for b in chunk]
            batch = self.collate(encs)
            logits = self.model(**self.inputs(batch))["logits"]
            for i, e in enumerate(encs):
                moves = [self.id_move(int(t)) for t in e["legal"]]
                out.append({m: float(logits[i, j]) for j, m in enumerate(moves)})
        return out


class P2Family:
    def __init__(self, ckpt):
        self.model, _ = load_p2(ckpt)

    @torch.no_grad()
    def scores(self, boards, bs=512):
        out = []
        for s in range(0, len(boards), bs):
            chunk = boards[s:s + bs]
            x = torch.from_numpy(np.stack([enc.encode(b) for b in chunk]))
            o = torch.from_numpy(np.stack([enc.occupancy(b) for b in chunk]))
            logits, _ = self.model(x, o)
            for i, b in enumerate(chunk):
                out.append({m: float(logits[i, enc.action_index(b, m)]) for m in b.legal_moves})
        return out


# ---------------------------------------------------------------------------
# A. Move quality on Prototype 2's benchmark
# ---------------------------------------------------------------------------

def bench_quality(model, d):
    fens, idx, cp, games = [str(f) for f in d["fen"]], d["idx"], d["cp"], d["game"]
    n_legal = (idx >= 0).sum(1)
    boards = [chess.Board(f) for f in fens]
    sc = model.scores(boards)
    choice = []
    for i, b in enumerate(boards):
        best = max(sc[i], key=sc[i].get)
        choice.append(int(np.where(idx[i] == enc.action_index(b, best))[0][0]))
    return choice_metrics(cp, n_legal, np.array(choice))


# ---------------------------------------------------------------------------
# B. Recorded-move agreement on the v5 Fishtest holdout
# ---------------------------------------------------------------------------

def holdout_positions(pgn_gz, n_games, seed=0):
    games = []
    with gzip.open(pgn_gz, "rt", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games.append(g)
    rng = np.random.RandomState(seed)
    pick = sorted(rng.choice(len(games), min(n_games, len(games)), replace=False))
    seen, boards, targets, gid = set(), [], [], []
    for k in pick:
        g = games[k]
        b = g.board()
        for mv in g.mainline_moves():
            key = " ".join(b.fen().split()[:4])
            if key not in seen and not b.is_game_over():
                seen.add(key)
                boards.append(b.copy(stack=False))
                targets.append(mv)
                gid.append(int(k))
            b.push(mv)
    return boards, targets, np.array(gid), len(games)


def bench_agreement(model, boards, targets):
    sc = model.scores(boards)
    top1, top3, n_legal = [], [], []
    for s, t in zip(sc, targets):
        order = sorted(s, key=s.get, reverse=True)
        top1.append(float(order[0] == t))
        top3.append(float(t in order[:3]))
        n_legal.append(len(s))
    return dict(top1=np.array(top1), top3=np.array(top3),
                uniform=np.array([1.0 / n for n in n_legal]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-root", required=True)
    ap.add_argument("--holdout-games", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "crosseval.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    models = {}
    for name, f in (("v3 pilot", "pilot.pt"), ("v4", "policy_v4.pt"), ("v5", "policy_v5.pt")):
        models[name] = ("v5 family", V5Family(a.v5_root, os.path.join(a.v5_root, "checkpoints", f)))
    for var in ("ray", "static", "standard"):
        for run in sorted(glob.glob(os.path.join(P2, "runs", f"{var}_s[0-9]"))):
            models[os.path.basename(run)] = ("prototype 2", P2Family(os.path.join(run, "best.pt")))

    d = load_npz(os.path.join(P2, "data", "test.npz"))
    games = d["game"]
    boards, targets, gid, n_total = holdout_positions(
        os.path.join(a.v5_root, "data", "fishtest_holdout.pgn.gz"), a.holdout_games)
    print(f"benchmark A: {len(games)} positions | benchmark B: {len(boards)} positions "
          f"from {a.holdout_games} of {n_total} holdout games", flush=True)

    res = {"A_quality": {}, "B_agreement": {}, "n_A": int(len(games)), "n_B": len(boards),
           "B_games": a.holdout_games}
    per_A, per_B = {}, {}
    for name, (fam, m) in models.items():
        per_A[name] = bench_quality(m, d)
        per_B[name] = bench_agreement(m, boards, targets)
        sA = summarize(per_A[name], games)
        sB = summarize(per_B[name], gid)
        res["A_quality"][name] = dict(family=fam, **{k: sA[k] for k in METRICS})
        res["B_agreement"][name] = dict(family=fam, **sB)
        print(f"{name:12s} [{fam}]  A: cp_loss {sA['cp_loss']['mean']:6.1f} best {100 * sA['best_rate']['mean']:5.1f}% "
              f"blunders {100 * sA['blunder_rate']['mean']:5.1f}%  |  B: top1 {100 * sB['top1']['mean']:5.2f}% "
              f"top3 {100 * sB['top3']['mean']:5.2f}% (uniform {100 * sB['uniform']['mean']:.2f}%)", flush=True)

    # Paired: best of each family, seed-averaged where there are seeds
    def avg(names, per):
        return {k: np.mean([per[n][k] for n in names], axis=0) for k in per[names[0]]}
    ray = [n for n in per_A if n.startswith("ray_s")]
    paired = {}
    for other in ("v3 pilot", "v5"):
        dA = {k: avg(ray, per_A)[k] - per_A[other][k] for k in ("cp_loss", "best_rate", "blunder_rate")}
        dB = {k: avg(ray, per_B)[k] - per_B[other][k] for k in ("top1", "top3")}
        paired[f"ray (3 seeds) - {other}"] = dict(A=summarize(dA, games), B=summarize(dB, gid))
    res["paired"] = paired
    for k, v in paired.items():
        print(f"{k}: A cp_loss {v['A']['cp_loss']['mean']:+.1f} [{v['A']['cp_loss']['lo']:+.1f},{v['A']['cp_loss']['hi']:+.1f}]"
              f"  B top1 {100 * v['B']['top1']['mean']:+.2f} pp [{100 * v['B']['top1']['lo']:+.2f},{100 * v['B']['top1']['hi']:+.2f}]")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
