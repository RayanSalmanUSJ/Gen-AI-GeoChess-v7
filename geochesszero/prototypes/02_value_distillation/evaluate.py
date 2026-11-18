"""
Benchmark trained models on held-out games labelled by a DEEPER teacher
(depth 12, vs. depth 8 for training labels), so the score is not just
agreement with the training labels.

Per move chosen by a model we measure:
  cp_loss       best teacher score minus chosen move's score (scores clipped to +/-1000)
  es_loss       expected-score points lost (0-100 scale): the "win % thrown away"
  best_rate     chosen move is (tied) best by the teacher
  mistake_rate  cp_loss >= 100
  blunder_rate  cp_loss >= 200
and for the value head: Pearson r and MAE vs. the teacher's value.

Move-choice modes:
  policy      argmax of the policy head (no search, one forward pass)
  comparator  1-ply "least-bad move" from the notes: score every legal move by
              -V(child) with the value head and pick the best

Reference players on the same positions: uniform random, Stockfish depth 1
and depth 4.

95% intervals are cluster bootstraps over test GAMES (positions inside a game
are correlated). Variant differences are paired on identical positions.
"""

import argparse
import glob
import json
import os

import chess
import chess.engine
import numpy as np
import torch

import encoding as enc
from model import load_checkpoint
from train import load_npz

CLIP = 1000


def move_metrics(cp_row, n, j):
    """Metrics for choosing legal move j in a position with teacher scores cp_row[:n]."""
    c = np.clip(cp_row[:n].astype(np.float64), -CLIP, CLIP)
    best = c.max()
    loss = best - c[j]
    es = (enc.cp_to_value(cp_row[:n].max()) - enc.cp_to_value(cp_row[j])) * 50.0
    return loss, es


def summarize(per_pos, games, n_boot=2000, seed=0):
    """Mean of each metric plus a game-cluster bootstrap 95% interval."""
    rng = np.random.RandomState(seed)
    ug = np.unique(games)
    groups = [np.where(games == g)[0] for g in ug]
    out = {}
    for k, arr in per_pos.items():
        arr = np.asarray(arr, dtype=np.float64)
        sums = np.array([arr[gi].sum() for gi in groups])
        cnts = np.array([len(gi) for gi in groups])
        boots = []
        for _ in range(n_boot):
            s = rng.randint(0, len(groups), len(groups))
            boots.append(sums[s].sum() / cnts[s].sum())
        out[k] = dict(mean=float(arr.mean()), lo=float(np.percentile(boots, 2.5)),
                      hi=float(np.percentile(boots, 97.5)))
    return out


def choice_metrics(cp, n_legal, choice):
    loss, es = zip(*[move_metrics(cp[i], n_legal[i], choice[i]) for i in range(len(choice))])
    loss, es = np.array(loss), np.array(es)
    return dict(cp_loss=loss, es_loss=es, best_rate=(loss <= 0).astype(float),
                mistake_rate=(loss >= 100).astype(float), blunder_rate=(loss >= 200).astype(float))


@torch.no_grad()
def policy_choices(model, T, bs=512):
    choice, values = [], []
    for s in range(0, T["x"].shape[0], bs):
        x, occ, idx = T["x"][s:s + bs], T["occ"][s:s + bs], T["idx"][s:s + bs]
        logits, v = model(x, occ)
        legal = logits.gather(1, idx.clamp(min=0)).masked_fill(idx < 0, -1e9)
        choice.append(legal.argmax(1))
        values.append(v)
    return torch.cat(choice).numpy(), torch.cat(values).numpy()


def expand_children(fens, idx_rows, n_legal):
    """Encode every child position (for the 1-ply comparator)."""
    xs, occs, owner, slot, terminal = [], [], [], [], {}
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        by_index = {enc.action_index(b, m): m for m in b.legal_moves}
        for j in range(n_legal[i]):
            m = by_index[int(idx_rows[i, j])]
            b.push(m)
            if b.is_checkmate():
                terminal[(i, j)] = 1.0          # mover delivers mate
            elif b.is_stalemate() or b.is_insufficient_material() or b.can_claim_draw():
                terminal[(i, j)] = 0.0
            else:
                xs.append(enc.encode(b))
                occs.append(enc.occupancy(b))
                owner.append(i)
                slot.append(j)
            b.pop()
    return (np.array(xs, np.float32), np.array(occs, np.float32),
            np.array(owner), np.array(slot), terminal)


@torch.no_grad()
def comparator_choices(model, children, n_pos, L, bs=1024):
    xs, occs, owner, slot, terminal = children
    q = np.full((n_pos, L), -np.inf)
    vals = []
    for s in range(0, len(xs), bs):
        _, v = model(torch.from_numpy(xs[s:s + bs]), torch.from_numpy(occs[s:s + bs]))
        vals.append(v.numpy())
    vals = np.concatenate(vals) if vals else np.zeros(0)
    q[owner, slot] = -vals                      # child value is from the opponent's side
    for (i, j), t in terminal.items():
        q[i, j] = t
    return q.argmax(1)


@torch.no_grad()
def search_choices(model, fens, idx_rows, n_legal, k=6, bs=1024):
    """2-ply minimax: the policy proposes its top-k moves, every opponent reply is
    tried, and the value head scores the resulting positions (mover to move)."""
    leaves_x, leaves_o, owner = [], [], []
    q = []  # per position: list of (slot j, list of leaf ids or terminal values)
    T = dict(x=torch.from_numpy(np.stack([enc.encode(chess.Board(f)) for f in fens])),
             occ=torch.from_numpy(np.stack([enc.occupancy(chess.Board(f)) for f in fens])))
    logits = torch.cat([model(T["x"][s:s + bs], T["occ"][s:s + bs])[0]
                        for s in range(0, len(fens), bs)])
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        by_index = {enc.action_index(b, m): m for m in b.legal_moves}
        legal = logits[i].gather(0, torch.from_numpy(idx_rows[i, :n_legal[i]].astype(np.int64)))
        cand = legal.argsort(descending=True)[:k].tolist()
        entry = []
        for j in cand:
            m = by_index[int(idx_rows[i, j])]
            b.push(m)
            if b.is_checkmate():
                entry.append((j, [("t", 1.0)]))
            elif b.is_stalemate() or b.is_insufficient_material():
                entry.append((j, [("t", 0.0)]))
            else:
                outs = []
                for r in b.legal_moves:
                    b.push(r)
                    if b.is_checkmate():
                        outs.append(("t", -1.0))
                    elif b.is_stalemate() or b.is_insufficient_material():
                        outs.append(("t", 0.0))
                    else:
                        outs.append(("l", len(leaves_x)))
                        leaves_x.append(enc.encode(b))
                        leaves_o.append(enc.occupancy(b))
                    b.pop()
                entry.append((j, outs))
            b.pop()
        q.append(entry)
    vals = []
    X, O = np.array(leaves_x, np.float32), np.array(leaves_o, np.float32)
    for s in range(0, len(X), bs):
        vals.append(model(torch.from_numpy(X[s:s + bs]), torch.from_numpy(O[s:s + bs]))[1].numpy())
    vals = np.concatenate(vals) if vals else np.zeros(0)
    choice = []
    for entry in q:
        best_j, best_q = None, -np.inf
        for j, outs in entry:
            qv = min(v if kind == "t" else vals[v] for kind, v in outs)  # leaf is mover-to-move
            if qv > best_q:
                best_j, best_q = j, qv
        choice.append(best_j)
    return np.array(choice)


def engine_choices(fens, idx_rows, engine_path, depth):
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    eng.configure({"Threads": 1, "Hash": 16})
    out = []
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        mv = eng.play(b, chess.engine.Limit(depth=depth)).move
        a = enc.action_index(b, mv)
        out.append(int(np.where(idx_rows[i] == a)[0][0]))
    eng.quit()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--runs", required=True, help="glob of run dirs, e.g. 'runs/*'")
    ap.add_argument("--engine", default=os.environ.get("STOCKFISH", "stockfish"))
    ap.add_argument("--comparator", action="store_true")
    ap.add_argument("--search-every", type=int, default=0,
                    help="also run the 2-ply search mode on every n-th test position (0 = off)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    d = load_npz(a.test)
    T = dict(x=torch.from_numpy(d["x"].astype(np.float32)),
             occ=torch.from_numpy(d["occ"].astype(np.float32)),
             idx=torch.from_numpy(d["idx"].astype(np.int64)))
    cp, idx, games, fens = d["cp"], d["idx"], d["game"], [str(f) for f in d["fen"]]
    n_legal = (idx >= 0).sum(1)
    N, L = idx.shape
    print(f"test positions {N} from {len(np.unique(games))} games")

    results, per_position, per_subset = {}, {}, {}

    rng = np.random.RandomState(0)
    rand = np.array([rng.randint(n_legal[i]) for i in range(N)])
    per_position["random"] = choice_metrics(cp, n_legal, rand)
    for depth in (1, 4):
        ch = engine_choices(fens, idx, a.engine, depth)
        per_position[f"stockfish_d{depth}"] = choice_metrics(cp, n_legal, ch)

    children = expand_children(fens, idx, n_legal) if a.comparator else None
    for run in sorted(glob.glob(a.runs)):
        ckpt = os.path.join(run, "best.pt")
        if not os.path.exists(ckpt):
            continue
        model, extra = load_checkpoint(ckpt)
        name = os.path.basename(run.rstrip("/"))
        ch, v = policy_choices(model, T)
        m = choice_metrics(cp, n_legal, ch)
        m["value_abs_err"] = np.abs(v - d["value"])
        per_position[f"{name}:policy"] = m
        results.setdefault("value_r", {})[name] = float(np.corrcoef(v, d["value"])[0, 1])
        if children is not None:
            per_position[f"{name}:comparator"] = choice_metrics(
                cp, n_legal, comparator_choices(model, children, N, L))
        if a.search_every:
            sub = np.arange(0, N, a.search_every)
            ch_s = search_choices(model, [fens[i] for i in sub], idx[sub], n_legal[sub])
            per_subset[f"{name}:search"] = choice_metrics(cp[sub], n_legal[sub], ch_s)
            per_subset[f"{name}:policy"] = {k: v[sub] for k, v in m.items() if k != "value_abs_err"}
        print("evaluated", name, "best epoch", extra.get("epoch"), flush=True)

    results["players"] = {k: summarize(v, games) for k, v in per_position.items()}
    if per_subset:
        sub = np.arange(0, N, a.search_every)
        for k in ("random", "stockfish_d1", "stockfish_d4"):
            per_subset[k] = {m: v[sub] for m, v in per_position[k].items()}
        results["search_subset"] = dict(
            n_positions=int(len(sub)),
            players={k: summarize(v, games[sub]) for k, v in per_subset.items()})
    results["n_positions"] = int(N)
    results["n_games"] = int(len(np.unique(games)))

    # Paired variant comparison, averaged over seeds, on identical positions.
    variants = sorted({k.split(":")[0].rsplit("_s", 1)[0] for k in per_position if ":" in k})
    paired = {}
    for mode in ("policy", "comparator"):
        avg = {}
        for var in variants:
            keys = [k for k in per_position if k.startswith(var + "_s") and k.endswith(":" + mode)]
            if keys:
                avg[var] = {met: np.mean([per_position[k][met] for k in keys], axis=0)
                            for met in ("cp_loss", "best_rate", "blunder_rate", "es_loss")}
        for x in avg:
            for y in avg:
                if x < y:
                    diff = {met: avg[x][met] - avg[y][met] for met in avg[x]}
                    paired[f"{mode}: {x} - {y}"] = summarize(diff, games)
    results["paired_differences"] = paired
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)

    print(f"\n{'player':32s} {'cp_loss':>18s} {'best%':>7s} {'blunder%':>9s} {'es_loss':>8s}")
    for k, s in results["players"].items():
        print(f"{k:32s} {s['cp_loss']['mean']:7.1f} [{s['cp_loss']['lo']:5.1f},{s['cp_loss']['hi']:5.1f}]"
              f" {100 * s['best_rate']['mean']:6.1f} {100 * s['blunder_rate']['mean']:8.1f}"
              f" {s['es_loss']['mean']:8.2f}")
    if per_subset:
        print(f"\n2-ply search subset ({len(np.arange(0, N, a.search_every))} positions):")
        for k, s in results["search_subset"]["players"].items():
            print(f"  {k:30s} cp_loss {s['cp_loss']['mean']:6.1f} [{s['cp_loss']['lo']:5.1f},{s['cp_loss']['hi']:5.1f}]"
                  f" best {100 * s['best_rate']['mean']:5.1f}% blunder {100 * s['blunder_rate']['mean']:5.1f}%")
    print("\nvalue head Pearson r:", {k: round(v, 3) for k, v in results.get("value_r", {}).items()})
    print("\npaired differences (cp_loss, negative = first is better):")
    for k, s in paired.items():
        c = s["cp_loss"]
        print(f"  {k:40s} {c['mean']:+6.1f} [{c['lo']:+6.1f},{c['hi']:+6.1f}]")


if __name__ == "__main__":
    main()
