"""
Method: GPT-style next-move predictor (the course's "Idea 3").

A decoder-only transformer reads the game as a token sequence
  [BOS] m1 m2 ... m_t
and predicts move t+1, exactly like GPT predicts the next word. It never sees
the board: it must infer the position from the move history.

For a fair comparison with Prototype 2 it trains on the SAME games, the SAME
positions and the SAME targets (Stockfish's soft distribution over all legal
moves, plus a value head). Only the input representation differs: a move
sequence here, a 64-square board there.

Tokens = Prototype 1's 4,272-move vocabulary in absolute (unmirrored)
coordinates, + BOS. Causal self-attention, learned positions, pre-LayerNorm.

Usage:
  python gpt_movelm.py train --seed 0 --out runs/gpt_s0
  python gpt_movelm.py eval --runs 'runs/gpt_s*' --out results/gpt_eval.json
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "02_value_distillation")
ROOT = os.path.join(HERE, "..", "..")
for p in (P2, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import encoding as enc  # noqa: E402  (Prototype 2)
from dataset import load_jsonl  # noqa: E402
from shared.move_encoder import NUM_ACTIONS, move_to_index  # noqa: E402

BOS = NUM_ACTIONS
VOCAB = NUM_ACTIONS + 1
MAX_LEN = 160
L_MAX = 128


# ---------------------------------------------------------------------------
# Data: games as move sequences, with per-ply teacher targets
# ---------------------------------------------------------------------------

def position_key(fen):
    return " ".join(fen.split()[:4])


def move_between(fen_a, fen_b):
    """The legal move that turns position a into position b."""
    board = chess.Board(fen_a)
    target = position_key(fen_b)
    for m in board.legal_moves:
        board.push(m)
        hit = position_key(board.fen()) == target
        board.pop()
        if hit:
            return m
    raise ValueError(f"no move links {fen_a} -> {fen_b}")


def played_moves(rows):
    """rows of one game sorted by ply -> list of played moves (len = len(rows) - 1)."""
    for a, b in zip(rows[:-1], rows[1:]):
        if b["ply"] != a["ply"] + 1:
            raise ValueError("gap in game record")
    return [move_between(a["fen"], b["fen"]) for a, b in zip(rows[:-1], rows[1:])]


def build_sequences(path, temperature=0.06):
    games = {}
    for r in load_jsonl(path):
        games.setdefault(r["game"], []).append(r)
    seqs = []
    for g, rows in sorted(games.items()):
        rows.sort(key=lambda r: r["ply"])
        if rows[0]["ply"] != 0:
            continue
        moves = played_moves(rows)
        T = min(len(rows), MAX_LEN)
        tokens = np.array([BOS] + [move_to_index(m) for m in moves[:T - 1]], np.int64)
        idx = np.full((T, L_MAX), -1, np.int64)
        tgt = np.zeros((T, L_MAX), np.float32)
        val = np.zeros(T, np.float32)
        for t in range(T):
            r = rows[t]
            b = chess.Board(r["fen"])
            scores = np.clip(np.array(r["cp"], np.int64), -enc.MATE_CP, enc.MATE_CP)
            n = len(r["moves"])
            for j, u in enumerate(r["moves"]):
                idx[t, j] = move_to_index(chess.Move.from_uci(u))
            tgt[t, :n] = enc.soft_policy_target(scores, temperature)
            val[t] = enc.cp_to_value(scores.max())
        seqs.append(dict(game=g, tokens=tokens, idx=idx, target=tgt, value=val))
    return seqs


def game_sequences(path):
    """game id -> list of played moves (for building test prefixes)."""
    games = {}
    for r in load_jsonl(path):
        games.setdefault(r["game"], []).append(r)
    out = {}
    for g, rows in games.items():
        rows.sort(key=lambda r: r["ply"])
        if rows[0]["ply"] == 0:
            out[g] = played_moves(rows)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x, causal):
        y = self.n1(x)
        x = x + self.att(y, y, y, attn_mask=causal, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class MoveGPT(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.config = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.policy = nn.Linear(d_model, NUM_ACTIONS)
        self.value = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tokens):
        B, T = tokens.shape
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        h = self.tok(tokens) + self.pos[:, :T]
        for blk in self.blocks:
            h = blk(h, causal)
        h = self.norm(h)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def collate(batch):
    B, T = len(batch), max(len(s["tokens"]) for s in batch)
    tokens = torch.full((B, T), BOS, dtype=torch.long)
    idx = torch.full((B, T, L_MAX), -1, dtype=torch.long)
    tgt = torch.zeros(B, T, L_MAX)
    val = torch.zeros(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, s in enumerate(batch):
        n = len(s["tokens"])
        tokens[i, :n] = torch.from_numpy(s["tokens"])
        idx[i, :n] = torch.from_numpy(s["idx"])
        tgt[i, :n] = torch.from_numpy(s["target"])
        val[i, :n] = torch.from_numpy(s["value"])
        mask[i, :n] = True
    return tokens, idx, tgt, val, mask


def losses(model, batch):
    tokens, idx, tgt, val, mask = batch
    logits, v = model(tokens)
    pad = idx < 0
    legal = logits.gather(2, idx.clamp(min=0)).masked_fill(pad, -1e9)
    logp = F.log_softmax(legal, dim=2)
    ce = -(tgt * logp).sum(2)
    ent = -(tgt * torch.log(tgt.clamp(min=1e-12))).sum(2)
    mse = (v - val).pow(2)
    top1 = (legal.argmax(2) == tgt.argmax(2)).float()
    # legality of the unmasked argmax: does the model know the rules on its own?
    raw = logits.argmax(2, keepdim=True)
    is_legal = ((idx == raw) & ~pad).any(2).float()
    m = mask.float()
    n = m.sum()
    return dict(ce=(ce * m).sum() / n, kl=((ce - ent) * m).sum() / n, mse=(mse * m).sum() / n,
                top1=(top1 * m).sum() / n, legal=(is_legal * m).sum() / n, n=n)


def train(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    seqs = build_sequences(os.path.join(P2, "data", "train.jsonl"))
    games = np.unique([s["game"] for s in seqs])
    rng = np.random.RandomState(12345)  # identical game split to Prototype 2's train.py
    val_games = set(rng.choice(games, max(1, int(len(games) * 0.1)), replace=False))
    tr = [s for s in seqs if s["game"] not in val_games]
    va = [s for s in seqs if s["game"] in val_games]
    for path in a.extra:  # extra games only ever go to training; validation is unchanged
        more = build_sequences(path)
        print(f"extra training games from {path}: {len(more)}", flush=True)
        tr += more
    model = MoveGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    os.makedirs(a.out, exist_ok=True)
    n_tr = sum(len(s["tokens"]) for s in tr)
    print(f"gpt seed={a.seed} params={count_parameters(model):,} train games={len(tr)} "
          f"positions={n_tr} val games={len(va)}", flush=True)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    gen = np.random.RandomState(a.seed)
    best, log, t0, bad = None, [], time.time(), 0
    for ep in range(1, a.epochs + 1):
        model.train()
        order = gen.permutation(len(tr))
        for s in range(0, len(tr), a.batch):
            L = losses(model, collate([tr[i] for i in order[s:s + a.batch]]))
            loss = L["ce"] + L["mse"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            V = losses(model, collate(va))
        row = dict(epoch=ep, val={k: float(v) for k, v in V.items() if k != "n"},
                   seconds=round(time.time() - t0, 1))
        log.append(row)
        score = float(V["ce"] + V["mse"])
        if ep % 10 == 0 or ep == 1:
            print(f"ep {ep}: val ce {V['ce']:.3f} kl {V['kl']:.3f} vmse {V['mse']:.4f} "
                  f"top1 {V['top1']:.3f} raw-legal {V['legal']:.3f} | {row['seconds']:.0f}s", flush=True)
        if best is None or score < best:
            best, bad = score, 0
            torch.save(dict(config=model.config, state_dict=model.state_dict(), epoch=ep,
                            val=row["val"]), os.path.join(a.out, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep}", flush=True)
                break
    json.dump(dict(args=vars(a), params=count_parameters(model), log=log, best_val_loss=best),
              open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


def load_gpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MoveGPT(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


# ---------------------------------------------------------------------------
# Evaluation on Prototype 2's test positions
# ---------------------------------------------------------------------------

@torch.no_grad()
def gpt_choices(model, prefixes, legal_abs):
    """prefixes: list of move lists; legal_abs: list of arrays of absolute action ids."""
    choice, raw_legal = [], []
    for s in range(0, len(prefixes), 64):
        chunk = prefixes[s:s + 64]
        T = max(len(p) for p in chunk) + 1
        toks = torch.full((len(chunk), T), BOS, dtype=torch.long)
        last = []
        for i, p in enumerate(chunk):
            ids = [move_to_index(m) for m in p]
            if ids:
                toks[i, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
            last.append(len(ids))
        logits, _ = model(toks)
        for i, p in enumerate(chunk):
            lg = logits[i, last[i]]
            la = torch.from_numpy(legal_abs[s + i])
            choice.append(int(lg[la].argmax()))
            raw_legal.append(float(int(lg.argmax()) in set(la.tolist())))
    return np.array(choice), np.array(raw_legal)


def evaluate_runs(a):
    sys.path.insert(0, P2)
    from evaluate import choice_metrics, summarize
    from train import load_npz
    d = load_npz(os.path.join(P2, "data", "test.npz"))
    seqs = game_sequences(os.path.join(P2, "data", "test_games.jsonl"))
    fens, games, plies = [str(f) for f in d["fen"]], d["game"], d["ply"]
    idx, cp = d["idx"], d["cp"]
    n_legal = (idx >= 0).sum(1)
    prefixes, legal_abs = [], []
    for i, fen in enumerate(fens):
        pre = seqs[int(games[i])][:int(plies[i])]
        b = chess.Board()
        for m in pre:
            b.push(m)
        assert position_key(b.fen()) == position_key(fen), "prefix mismatch"
        by_index = {enc.action_index(b, m): m for m in b.legal_moves}
        legal_abs.append(np.array([move_to_index(by_index[int(x)]) for x in idx[i, :n_legal[i]]]))
        prefixes.append(pre)
    players = {}
    for run in sorted(glob.glob(a.runs)):
        if not os.path.isfile(os.path.join(run, "best.pt")):
            continue
        model, ck = load_gpt(os.path.join(run, "best.pt"))
        ch, raw_legal = gpt_choices(model, prefixes, legal_abs)
        m = choice_metrics(cp, n_legal, ch)
        m["raw_legal_rate"] = raw_legal
        name = os.path.basename(run.rstrip("/"))
        players[f"{name}:policy"] = summarize(m, games)
        print(f"{name} (epoch {ck['epoch']}): cp_loss {players[name + ':policy']['cp_loss']['mean']:.1f} "
              f"best {100 * players[name + ':policy']['best_rate']['mean']:.1f}% "
              f"blunder {100 * players[name + ':policy']['blunder_rate']['mean']:.1f}% "
              f"raw-legal {100 * raw_legal.mean():.1f}%", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(players=players, n_positions=len(fens)), open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--patience", type=int, default=25)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--extra", nargs="*", default=[], help="extra JSONL game files for training")
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--runs", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate_runs(a)


if __name__ == "__main__":
    main()
