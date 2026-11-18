"""
Method: self-play, the model learns from its own games (no teacher labels).

AlphaZero-style expert iteration, scaled down to a CPU:
  1. The current model plays games against itself (policy sampling, many
     games batched together).
  2. On a sample of those positions it runs its own 2-ply search: the policy
     proposes the top-k moves, every reply is tried, and the value head scores
     the results. Search is the "expert": Prototype 2 showed it plays better
     than the raw policy.
  3. Policy target = policy improvement  pi_new(m) ~ pi(m) * exp(Q(m) / tau)
     (moves outside the top-k get the worst searched Q). Value target mixes
     the searched value with the game's final result. A side resigns when its
     own value head is below -0.75 on 3 consecutive turns; games that hit the
     move cap get no result, so their value targets come from search alone
     (scoring them as draws taught the model that won positions are drawn).
  4. Fine-tune on those targets, then repeat from step 1 with the new model.

Stockfish is never used for training, only afterwards as the referee
(benchmark on Prototype 2's held-out positions, and matches).

Usage:
  python selfplay.py --start ../02_value_distillation/runs/ray_s0/best.pt \
                     --iterations 3 --out runs/selfplay_ray_s0
"""

import argparse
import json
import math
import os
import random
import sys
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "02_value_distillation")
ROOT = os.path.join(HERE, "..", "..")
for p in (P2, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import encoding as enc  # noqa: E402
from evaluate import choice_metrics, policy_choices, summarize  # noqa: E402
from model import load_checkpoint, save_checkpoint  # noqa: E402
from train import batch_losses, load_npz  # noqa: E402

L_MAX = 128


def terminal_value(board):
    """Value for the side to move if the game is over, else None (fast path for search)."""
    if not any(True for _ in board.generate_legal_moves()):
        return -1.0 if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return 0.0
    return None


@torch.no_grad()
def forward_boards(model, boards, bs=512):
    logits, values = [], []
    for s in range(0, len(boards), bs):
        x = torch.from_numpy(np.stack([enc.encode(b) for b in boards[s:s + bs]]))
        o = torch.from_numpy(np.stack([enc.occupancy(b) for b in boards[s:s + bs]]))
        lg, v = model(x, o)
        logits.append(lg)
        values.append(v)
    return torch.cat(logits), torch.cat(values)


# ---------------------------------------------------------------------------
# 1. Self-play games, many at once
# ---------------------------------------------------------------------------

@torch.no_grad()
def self_play(model, n_games, rng, max_plies=160, temp_open=1.0, temp=0.5, open_plies=8,
              resign_v=-0.75, resign_streak=3, record_actions=True):
    boards = [chess.Board() for _ in range(n_games)]
    records = [[] for _ in range(n_games)]
    actions = [[] for _ in range(n_games)]
    streak = [{chess.WHITE: 0, chess.BLACK: 0} for _ in range(n_games)]
    resigned = [None] * n_games
    active = list(range(n_games))
    for ply in range(max_plies):
        active = [g for g in active if resigned[g] is None
                  and not boards[g].is_game_over(claim_draw=True)]
        if not active:
            break
        logits, values = forward_boards(model, [boards[g] for g in active])
        for k, g in enumerate(active):
            b = boards[g]
            # resignation: the side to move is clearly lost on 3 consecutive own turns
            streak[g][b.turn] = streak[g][b.turn] + 1 if float(values[k]) < resign_v else 0
            if streak[g][b.turn] >= resign_streak:
                resigned[g] = b.turn
                continue
            legal = list(b.legal_moves)
            ids = torch.tensor([enc.action_index(b, m) for m in legal])
            t = temp_open if ply < open_plies else temp
            p = F.softmax(logits[k, ids] / t, dim=0).numpy().astype(np.float64)
            # skip moves that repeat a position when possible (no history in the input)
            for j, m in enumerate(legal):
                b.push(m)
                if b.is_repetition(2):
                    p[j] *= 1e-3
                b.pop()
            p /= p.sum()
            j = rng.choice(len(legal), p=p)
            records[g].append((b.fen(), b.turn, float(values[k])))
            actions[g].append(int(ids[j]))
            b.push(legal[j])
    games = []
    for g in range(n_games):
        if resigned[g] is not None:
            res = "0-1 (resign)" if resigned[g] == chess.WHITE else "1-0 (resign)"
            z_white = -1.0 if resigned[g] == chess.WHITE else 1.0
        else:
            res = boards[g].result(claim_draw=True)
            z_white = {"1-0": 1.0, "0-1": -1.0}.get(res, 0.0)
            if res == "*":
                res, z_white = "unfinished (cap)", None   # no result: value targets from search only
        games.append(dict(result=res, actions=actions[g], positions=[
            (fen, None if z_white is None else (z_white if turn == chess.WHITE else -z_white), v)
            for fen, turn, v in records[g]]))
    return games


# ---------------------------------------------------------------------------
# 2. 2-ply search targets
# ---------------------------------------------------------------------------

@torch.no_grad()
def search_targets(model, fens, k=6, tau_q=0.1, chunk=96):
    out = []
    for c in range(0, len(fens), chunk):
        out.extend(_search_chunk(model, fens[c:c + chunk], k, tau_q))
    return out


@torch.no_grad()
def _search_chunk(model, fens, k, tau_q):
    roots = [chess.Board(f) for f in fens]
    root_logits, _ = forward_boards(model, roots)
    lx, lo, plan = [], [], []
    for i, b in enumerate(roots):
        legal = list(b.legal_moves)
        ids = torch.tensor([enc.action_index(b, m) for m in legal])
        logp = F.log_softmax(root_logits[i, ids], dim=0)
        cand = logp.argsort(descending=True)[:k].tolist()
        entry = []
        for j in cand:
            b.push(legal[j])
            tv = terminal_value(b)
            if tv is not None:
                entry.append((j, [("t", -tv)]))        # value for the mover
            else:
                outs = []
                for r in list(b.legal_moves):
                    b.push(r)
                    tv2 = terminal_value(b)
                    if tv2 is not None:
                        outs.append(("t", tv2))        # mover is to move again
                    else:
                        outs.append(("l", len(lx)))
                        lx.append(enc.encode(b))
                        lo.append(enc.occupancy(b))
                    b.pop()
                entry.append((j, outs))
            b.pop()
        plan.append((legal, ids, logp, entry))
    leaf_v = np.zeros(0)
    if lx:
        X, O = torch.from_numpy(np.stack(lx)), torch.from_numpy(np.stack(lo))
        leaf_v = torch.cat([model(X[s:s + 1024], O[s:s + 1024])[1] for s in range(0, len(X), 1024)]).numpy()
    out = []
    for (legal, ids, logp, entry), fen in zip(plan, fens):
        q = {j: min(v if kind == "t" else float(leaf_v[v]) for kind, v in outs) for j, outs in entry}
        qmin = min(q.values())
        qall = np.array([q.get(j, qmin) for j in range(len(legal))])
        improved = logp.numpy() + qall / tau_q              # log pi + Q / tau
        target = np.exp(improved - improved.max())
        target /= target.sum()
        out.append(dict(fen=fen, ids=ids.numpy(), target=target.astype(np.float32),
                        q_root=float(max(q.values()))))
    return out


# ---------------------------------------------------------------------------
# 3. Fine-tune
# ---------------------------------------------------------------------------

def to_tensors(samples):
    N = len(samples)
    x = np.zeros((N, 64, enc.N_CHANNELS), np.float32)
    occ = np.zeros((N, 64), np.float32)
    idx = np.full((N, L_MAX), -1, np.int64)
    tgt = np.zeros((N, L_MAX), np.float32)
    val = np.zeros(N, np.float32)
    for i, s in enumerate(samples):
        b = chess.Board(s["fen"])
        x[i] = enc.encode(b)
        occ[i] = enc.occupancy(b)
        n = min(len(s["ids"]), L_MAX)
        idx[i, :n] = s["ids"][:n]
        tgt[i, :n] = s["target"][:n]
        val[i] = s["value"]
    return dict(x=torch.from_numpy(x), occ=torch.from_numpy(occ), idx=torch.from_numpy(idx),
                target=torch.from_numpy(tgt), value=torch.from_numpy(val))


def finetune(model, T, epochs, lr, seed, batch=256):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    n = T["x"].shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, batch):
            sel = order[s:s + batch]
            b = {k: v[sel] for k, v in T.items()}
            ce, _, mse, _ = batch_losses(model, b)
            loss = ce.mean() + mse.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()


# ---------------------------------------------------------------------------
# Referee: Prototype 2's held-out benchmark (Stockfish depth 12 prices)
# ---------------------------------------------------------------------------

def benchmark(model, d):
    T = dict(x=torch.from_numpy(d["x"].astype(np.float32)),
             occ=torch.from_numpy(d["occ"].astype(np.float32)),
             idx=torch.from_numpy(d["idx"].astype(np.int64)))
    ch, v = policy_choices(model, T)
    n_legal = (d["idx"] >= 0).sum(1)
    m = choice_metrics(d["cp"], n_legal, ch)
    s = summarize(m, d["game"])
    return dict(cp_loss=s["cp_loss"]["mean"], cp_lo=s["cp_loss"]["lo"], cp_hi=s["cp_loss"]["hi"],
                best=s["best_rate"]["mean"], blunder=s["blunder_rate"]["mean"],
                value_r=float(np.corrcoef(v, d["value"])[0, 1]), per_position=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--positions", type=int, default=3000, help="searched positions per iteration")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--contested", type=float, default=0.8,
                    help="only learn from positions whose own |value| is below this")
    ap.add_argument("--z-weight", type=float, default=0.3, help="share of game result in value target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    model, _ = load_checkpoint(a.start)
    test = load_npz(os.path.join(P2, "data", "test.npz"))
    base = benchmark(model, test)
    log = [dict(iteration=0, **{k: v for k, v in base.items() if k != "per_position"})]
    np.savez_compressed(os.path.join(a.out, "per_position_iter0.npz"), **base["per_position"])
    print(f"iter 0 (start): cp_loss {base['cp_loss']:.1f} blunder {100 * base['blunder']:.1f}% "
          f"value r {base['value_r']:.3f}", flush=True)
    for it in range(1, a.iterations + 1):
        t0 = time.time()
        games = self_play(model, a.games, rng)
        results = {}
        for g in games:
            results[g["result"]] = results.get(g["result"], 0) + 1
        # learn where the game is still undecided by the model's own judgement
        pool = [(fen, z) for g in games for fen, z, v in g["positions"] if abs(v) < a.contested]
        pick = rng.choice(len(pool), min(a.positions, len(pool)), replace=False)
        fens = [pool[i][0] for i in pick]
        zs = [pool[i][1] for i in pick]
        t1 = time.time()
        targets = search_targets(model, fens)
        for tg, z in zip(targets, zs):
            tg["value"] = tg["q_root"] if z is None else (1 - a.z_weight) * tg["q_root"] + a.z_weight * z
        t2 = time.time()
        finetune(model, to_tensors(targets), a.epochs, a.lr, a.seed + it)
        save_checkpoint(model, os.path.join(a.out, f"iter{it}.pt"), extra=dict(iteration=it))
        model.eval()
        r = benchmark(model, test)
        np.savez_compressed(os.path.join(a.out, f"per_position_iter{it}.npz"), **r["per_position"])
        row = dict(iteration=it, games=len(games), results=results, positions=len(pool),
                   searched=len(fens), selfplay_s=round(t1 - t0), search_s=round(t2 - t1),
                   train_s=round(time.time() - t2),
                   **{k: v for k, v in r.items() if k != "per_position"})
        log.append(row)
        print(f"iter {it}: {len(games)} games {results}, {len(pool)} positions, searched {len(fens)} | "
              f"cp_loss {r['cp_loss']:.1f} [{r['cp_lo']:.1f},{r['cp_hi']:.1f}] "
              f"blunder {100 * r['blunder']:.1f}% value r {r['value_r']:.3f} | "
              f"{row['selfplay_s']}+{row['search_s']}+{row['train_s']}s", flush=True)
        json.dump(dict(args=vars(a), log=log), open(os.path.join(a.out, "selfplay_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
