"""
Method: self-play that learns only from wins and losses ("learns from its mistakes").

Each iteration the model plays itself (same game loop as selfplay.py, with
resignation), then updates with a policy gradient:
  policy loss = -A * log pi(played move),   A = z - V(s)
where z is the final result from the mover's side and V(s) is what the model
itself expected at the time. A move from a game that went worse than the model
expected gets less likely; a move from a game that went better gets more likely.
Value loss = (V(s) - z)^2. A KL penalty to the starting policy keeps the model
from drifting on so few games. Unfinished (move-cap) games are skipped.

No search, no Stockfish labels: Stockfish is only the referee afterwards.

Usage:
  python selfplay_pg.py --start ../02_value_distillation/runs/ray_s0/best.pt \
                        --iterations 8 --out runs/selfplay_pg_ray_s0
"""

import argparse
import json
import os
import sys
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import selfplay as sp  # noqa: E402  (game loop, benchmark, encoding helpers)
from selfplay import enc, load_checkpoint, load_npz, save_checkpoint, P2  # noqa: E402

L_MAX = 128


def collect(games):
    """Positions from decided games: (fen, played action, z, V at the time)."""
    out = []
    for g in games:
        for (fen, z, v), act in zip(g["positions"], g["actions"]):
            if z is not None:
                out.append((fen, act, z, v))
    return out


def tensors(samples, start_model):
    N = len(samples)
    x = np.zeros((N, 64, enc.N_CHANNELS), np.float32)
    occ = np.zeros((N, 64), np.float32)
    idx = np.full((N, L_MAX), -1, np.int64)
    chosen = np.zeros(N, np.int64)
    z = np.zeros(N, np.float32)
    v_old = np.zeros(N, np.float32)
    for i, (fen, act, zz, vv) in enumerate(samples):
        b = chess.Board(fen)
        x[i], occ[i] = enc.encode(b), enc.occupancy(b)
        ids = [enc.action_index(b, m) for m in b.legal_moves][:L_MAX]
        idx[i, :len(ids)] = ids
        chosen[i] = ids.index(act)
        z[i], v_old[i] = zz, vv
    T = dict(x=torch.from_numpy(x), occ=torch.from_numpy(occ), idx=torch.from_numpy(idx),
             chosen=torch.from_numpy(chosen), z=torch.from_numpy(z), v_old=torch.from_numpy(v_old))
    with torch.no_grad():  # reference policy for the KL anchor
        ref = []
        for s in range(0, N, 512):
            lg, _ = start_model(T["x"][s:s + 512], T["occ"][s:s + 512])
            leg = lg.gather(1, T["idx"][s:s + 512].clamp(min=0)).masked_fill(T["idx"][s:s + 512] < 0, -1e9)
            ref.append(F.log_softmax(leg, 1))
        T["ref_logp"] = torch.cat(ref)
    return T


def update(model, T, lr, kl_weight, seed, batch=256):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n = T["x"].shape[0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    stats = []
    for s in range(0, n, batch):
        b = {k: v[order[s:s + batch]] for k, v in T.items()}
        logits, v = model(b["x"], b["occ"])
        pad = b["idx"] < 0
        leg = logits.gather(1, b["idx"].clamp(min=0)).masked_fill(pad, -1e9)
        logp = F.log_softmax(leg, 1)
        adv = b["z"] - b["v_old"]
        pg = -(adv * logp.gather(1, b["chosen"][:, None]).squeeze(1)).mean()
        ref = b["ref_logp"]
        kl = (ref.exp() * (ref - logp)).masked_fill(pad, 0).sum(1).mean()
        vloss = (v - b["z"]).pow(2).mean()
        loss = pg + vloss + kl_weight * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        stats.append((pg.item(), vloss.item(), kl.item()))
    model.eval()
    return np.mean(stats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kl-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    os.makedirs(a.out, exist_ok=True)
    model, _ = load_checkpoint(a.start)
    start_model, _ = load_checkpoint(a.start)
    test = load_npz(os.path.join(P2, "data", "test.npz"))
    r0 = sp.benchmark(model, test)
    log = [dict(iteration=0, cp_loss=r0["cp_loss"], cp_lo=r0["cp_lo"], cp_hi=r0["cp_hi"],
                blunder=r0["blunder"], best=r0["best"], value_r=r0["value_r"])]
    print(f"iter 0: cp_loss {r0['cp_loss']:.1f} blunder {100 * r0['blunder']:.1f}% value r {r0['value_r']:.3f}",
          flush=True)
    for it in range(1, a.iterations + 1):
        t0 = time.time()
        games = sp.self_play(model, a.games, rng, record_actions=True)
        res = {}
        for g in games:
            res[g["result"]] = res.get(g["result"], 0) + 1
        samples = collect(games)
        T = tensors(samples, start_model)
        pg, vl, kl = update(model, T, a.lr, a.kl_weight, a.seed + it)
        save_checkpoint(model, os.path.join(a.out, f"iter{it}.pt"), extra=dict(iteration=it))
        r = sp.benchmark(model, test)
        row = dict(iteration=it, results=res, samples=len(samples), pg_loss=float(pg), value_loss=float(vl),
                   kl=float(kl), seconds=round(time.time() - t0), cp_loss=r["cp_loss"], cp_lo=r["cp_lo"],
                   cp_hi=r["cp_hi"], blunder=r["blunder"], best=r["best"], value_r=r["value_r"])
        log.append(row)
        print(f"iter {it}: {res} | {len(samples)} positions | value loss {vl:.3f} kl {kl:.4f} | "
              f"cp_loss {r['cp_loss']:.1f} [{r['cp_lo']:.1f},{r['cp_hi']:.1f}] blunder {100 * r['blunder']:.1f}% "
              f"value r {r['value_r']:.3f} | {row['seconds']}s", flush=True)
        json.dump(dict(args=vars(a), log=log), open(os.path.join(a.out, "selfplay_pg_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
