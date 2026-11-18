"""v6 evaluation on any labelled v6 dataset directory (T1, T2, dev) or unlabelled one (T3).

  python -m chess_policy.v6_eval --data benchmarks/t2_p2_1312 \
      --model v5=checkpoints/policy_v5.pt --model B=runs/v6/B_soft_s0/best.pt \
      [--search B] [--reference v5] --out results/v6/t2.json

Per model and mode (policy = argmax of the policy head; search = 2-ply search that needs a
value head) it reports centipawns lost per move, best-move rate, mistakes (>=100 cp),
blunders (>=200 cp) and agreement with the move played, each with a 95% interval from a
bootstrap over whole games (positions inside a game are correlated). --reference adds
paired differences on identical positions.
"""

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch

from .encoding import collate, encode, move_id
from .v6_data import V6Dataset
from .v6_model import load_any
from .v6_train import choice_metrics, to_device


def bootstrap(per_position, games, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    unique = np.unique(games)
    groups = [np.flatnonzero(games == g) for g in unique]
    out = {}
    for key, values in per_position.items():
        values = np.asarray(values, dtype=np.float64)
        sums = np.array([values[g].sum() for g in groups])
        counts = np.array([len(g) for g in groups])
        boots = []
        for _ in range(n_boot):
            pick = rng.randint(0, len(groups), len(groups))
            boots.append(sums[pick].sum() / counts[pick].sum())
        out[key] = dict(mean=float(values.mean()), lo=float(np.percentile(boots, 2.5)),
                        hi=float(np.percentile(boots, 97.5)))
    return out


@torch.no_grad()
def policy_order(model, dataset, device, batch_size=512):
    """Per position: legal-slot indices sorted by the policy, best first."""
    orders = []
    for s in range(0, len(dataset), batch_size):
        idx = np.arange(s, min(len(dataset), s + batch_size))
        b = to_device(dataset.batch(idx), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(b["pieces"], b["geometry"], b["attacks"], b["legal"])["logits"]
        order = logits.float().argsort(-1, descending=True).cpu().numpy()
        for row, i in zip(order, idx):
            orders.append(row[: dataset.lengths[i]])
    return orders


@torch.no_grad()
def values_of(model, boards, device, batch_size=1024):
    """Value head for each board, from the side to move's point of view."""
    out = []
    for s in range(0, len(boards), batch_size):
        b = to_device(collate([encode(x, validate=False) for x in boards[s:s + batch_size]]),
                      device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            v = model(b["pieces"], b["geometry"], b["attacks"], b["legal"])["value"]
        out.append(v.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _terminal(board):
    if board.is_checkmate():
        return -1.0  # side to move is mated
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    return None


def search_choices(model, fens, orders, device, k=6):
    """2-ply search (Prototype 2): the policy proposes its top-k moves, every reply is
    tried, the value head scores the resulting positions (original mover to move);
    each candidate is worth its worst reply, the best candidate is chosen."""
    plans, leaves = [], []
    for fen, order in zip(fens, orders):
        board = chess.Board(fen)
        legal = sorted(board.legal_moves, key=move_id)
        entry = []
        for j in order[:k]:
            move = legal[j]
            board.push(move)
            t = _terminal(board)
            if t is not None:
                entry.append((j, [("t", -t)]))
            else:
                outs = []
                for reply in board.legal_moves:
                    board.push(reply)
                    t2 = _terminal(board)
                    if t2 is not None:
                        outs.append(("t", t2))
                    else:
                        outs.append(("l", len(leaves)))
                        leaves.append(board.copy(stack=False))
                    board.pop()
                entry.append((j, outs))
            board.pop()
        plans.append(entry)
    vals = values_of(model, leaves, device)
    choice = []
    for entry in plans:
        best_j, best_q = entry[0][0], -np.inf
        for j, outs in entry:
            q = min(v if kind == "t" else float(vals[v]) for kind, v in outs)
            if q > best_q:
                best_j, best_q = j, q
        choice.append(best_j)
    return np.array(choice)


def evaluate_model(model, dataset, device, mode="policy", k=6):
    orders = policy_order(model, dataset, device)
    if mode == "search":
        if not model.config.value_head:
            raise ValueError("search needs a model with a value head")
        fens = (dataset.dir / "fen.txt").read_text().split("\n")[: len(dataset)]
        choice = search_choices(model, fens, orders, device, k)
    else:
        choice = np.array([o[0] for o in orders])
    per = dict(agree_top1=(choice == dataset.target).astype(float),
               agree_top3=np.array([t in o[:3] for t, o in zip(dataset.target, orders)], float))
    if dataset.cp is not None:
        offsets = dataset.offsets
        rows = [np.asarray(dataset.cp[offsets[i]:offsets[i + 1]]) for i in range(len(dataset))]
        per.update(choice_metrics(rows, dataset.lengths, choice))
    return per


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chess_policy.v6_eval")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint")
    ap.add_argument("--search", action="append", default=[], help="also run 2-ply search for NAME")
    ap.add_argument("--reference", help="name (or name:search) for paired differences")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    dataset = V6Dataset(a.data)
    if a.max_rows:
        keep = slice(0, a.max_rows)
        dataset.target, dataset.game = dataset.target[keep], dataset.game[keep]
        dataset.offsets = dataset.offsets[: a.max_rows + 1]
        dataset.lengths = dataset.lengths[: a.max_rows]
    games = dataset.game
    per, report = {}, {"data": a.data, "rows": len(dataset), "games": int(len(np.unique(games))),
                       "models": {}}
    for spec in a.model:
        name, path = spec.split("=", 1)
        model = load_any(path, device)
        modes = ["policy"] + (["search"] if name in a.search else [])
        for mode in modes:
            key = name if mode == "policy" else f"{name}:search"
            per[key] = evaluate_model(model, dataset, device, mode, a.k)
            report["models"][key] = dict(checkpoint=path, mode=mode,
                                         **bootstrap(per[key], games))
            s = report["models"][key]
            line = f"{key:28s}"
            if "cp_loss" in s:
                line += (f" cp lost {s['cp_loss']['mean']:6.1f} [{s['cp_loss']['lo']:.1f}, "
                         f"{s['cp_loss']['hi']:.1f}]  best {100 * s['best_rate']['mean']:5.1f}%"
                         f"  blunders {100 * s['blunder_rate']['mean']:5.1f}%")
            line += f"  agree {100 * s['agree_top1']['mean']:5.2f}%"
            print(line, flush=True)
    if a.reference:
        ref = per[a.reference]
        report["paired_vs_" + a.reference] = {
            key: bootstrap({k: np.asarray(v) - np.asarray(ref[k]) for k, v in vals.items()},
                           games)
            for key, vals in per.items() if key != a.reference
        }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    main()
