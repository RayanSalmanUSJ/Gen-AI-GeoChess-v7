"""
Train GeoChessZero v2 by distilling a teacher's move values.

Loss = soft cross-entropy(policy over legal moves, teacher soft policy)
     + value_weight * MSE(value head, teacher value)

The policy target is the teacher's softmax over expected score for all legal
moves, not the move a human played. That is the switch from behavioural
cloning (Prototype 1) to value-optimal play.

Usage:
  python train.py --data data/train.npz --variant ray --seed 0 --epochs 8 --out runs/ray_s0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GeoChessZeroV2, count_parameters, save_checkpoint


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def to_tensors(d, sel):
    return dict(
        x=torch.from_numpy(d["x"][sel].astype(np.float32)),
        occ=torch.from_numpy(d["occ"][sel].astype(np.float32)),
        idx=torch.from_numpy(d["idx"][sel].astype(np.int64)),
        target=torch.from_numpy(d["target"][sel].astype(np.float32)),
        value=torch.from_numpy(d["value"][sel]),
    )


def batch_losses(model, b):
    logits, v = model(b["x"], b["occ"])
    pad = b["idx"] < 0
    legal_logits = logits.gather(1, b["idx"].clamp(min=0))
    legal_logits = legal_logits.masked_fill(pad, -1e9)
    logp = F.log_softmax(legal_logits, dim=1)
    t = b["target"]
    ce = -(t * logp).sum(1)
    ent = -(t * torch.log(t.clamp(min=1e-12))).sum(1)
    mse = (v - b["value"]).pow(2)
    top1 = (legal_logits.argmax(1) == t.argmax(1)).float()
    return ce, ce - ent, mse, top1


def iterate(T, batch_size, shuffle, gen=None):
    n = T["x"].shape[0]
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for s in range(0, n, batch_size):
        sel = order[s:s + batch_size]
        yield {k: v[sel] for k, v in T.items()}


@torch.no_grad()
def evaluate(model, T, batch_size=512):
    model.eval()
    sums, n = np.zeros(4), 0
    for b in iterate(T, batch_size, False):
        ce, kl, mse, top1 = batch_losses(model, b)
        sums += [ce.sum().item(), kl.sum().item(), mse.sum().item(), top1.sum().item()]
        n += ce.shape[0]
    model.train()
    ce, kl, mse, top1 = sums / n
    return dict(ce=ce, kl=kl, value_mse=mse, top1_teacher=top1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["standard", "static", "ray"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=0, help="cap on training positions (0 = all)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    d = load_npz(a.data)
    games = np.unique(d["game"])
    rng = np.random.RandomState(12345)  # same split for every variant and seed
    val_games = set(rng.choice(games, max(1, int(len(games) * a.val_frac)), replace=False))
    is_val = np.array([g in val_games for g in d["game"]])
    tr_idx = np.where(~is_val)[0]
    if a.max_train:
        tr_idx = np.random.RandomState(a.seed).permutation(tr_idx)[:a.max_train]
    va_idx = np.where(is_val)[0]
    TR, VA = to_tensors(d, tr_idx), to_tensors(d, va_idx)

    model = GeoChessZeroV2(a.variant, d_model=a.d_model, n_layers=a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = math.ceil(len(tr_idx) / a.batch_size)
    total = steps_per_epoch * a.epochs
    warm = max(1, min(200, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    print(f"variant={a.variant} seed={a.seed} params={count_parameters(model):,} "
          f"train={len(tr_idx)} val={len(va_idx)} steps={total}", flush=True)
    log, best, gen = [], None, torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        run = np.zeros(3)
        nb = 0
        for b in iterate(TR, a.batch_size, True, gen):
            ce, kl, mse, _ = batch_losses(model, b)
            loss = ce.mean() + a.value_weight * mse.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += [ce.mean().item(), kl.mean().item(), mse.mean().item()]
            nb += 1
        tr = dict(zip(["ce", "kl", "value_mse"], (run / nb).tolist()))
        va = evaluate(model, VA)
        score = va["ce"] + a.value_weight * va["value_mse"]
        row = dict(epoch=ep, train=tr, val=va, val_loss=score, seconds=round(time.time() - t0, 1))
        log.append(row)
        print(f"ep {ep}: train ce {tr['ce']:.3f} kl {tr['kl']:.3f} vmse {tr['value_mse']:.4f} | "
              f"val ce {va['ce']:.3f} kl {va['kl']:.3f} vmse {va['value_mse']:.4f} "
              f"top1 {va['top1_teacher']:.3f} | {row['seconds']:.0f}s", flush=True)
        if best is None or score < best:
            best = score
            save_checkpoint(model, os.path.join(a.out, "best.pt"),
                            extra=dict(epoch=ep, val=va, args=vars(a)))
    json.dump(dict(args=vars(a), params=count_parameters(model), log=log,
                   best_val_loss=best), open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
