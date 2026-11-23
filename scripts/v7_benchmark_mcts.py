"""Benchmark PUCT MCTS on labelled chess datasets (e.g. dev_d10).

Evaluates centipawn loss, best-move rate, and blunder rate across:
1. Raw policy (argmax)
2. MCTS with configurable N (simulations) and c_puct.
Computes paired bootstrap differences against raw policy.
"""

import argparse
import json
from pathlib import Path
import time

import chess
import numpy as np
import torch

from chess_policy.encoding import move_id
from chess_policy.mcts import MCTS, MCTSConfig
from chess_policy.v6_data import V6Dataset
from chess_policy.v6_eval import bootstrap, policy_order
from chess_policy.v6_model import load_v6
from chess_policy.v6_train import choice_metrics


def evaluate_mcts(model, fens, dataset, device, mcts_config):
    """Run MCTS on each position in fens and return choice array (indices into legal moves)."""
    mcts = MCTS(model, device, mcts_config)
    choices = []
    times = []

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        legal_moves = sorted(board.legal_moves, key=move_id)

        t0 = time.monotonic()
        res = mcts.search(board)
        times.append(time.monotonic() - t0)

        best_move = res["best_move"]
        if best_move is None or best_move not in legal_moves:
            # Fallback to first legal move
            idx = 0
        else:
            idx = legal_moves.index(best_move)
        choices.append(idx)

    return np.array(choices), np.mean(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MCTS vs Policy.")
    parser.add_argument("--data", default="data/v6/dev_d10")
    parser.add_argument("--model", action="append", required=True, help="name=checkpoint")
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--simulations", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--c-puct", type=float, nargs="+", default=[1.5, 2.0])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vl-weight", type=float, default=0.1)
    parser.add_argument("--fpu-reduction", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/v7/mcts_benchmark.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = V6Dataset(args.data)
    n_rows = min(len(dataset), args.max_rows) if args.max_rows else len(dataset)

    # Slice dataset
    dataset.target = dataset.target[:n_rows]
    dataset.game = dataset.game[:n_rows]
    dataset.offsets = dataset.offsets[: n_rows + 1]
    dataset.lengths = dataset.lengths[:n_rows]

    fens = (dataset.dir / "fen.txt").read_text().split("\n")[:n_rows]
    games = dataset.game
    offsets = dataset.offsets
    cp_rows = [np.asarray(dataset.cp[offsets[i]:offsets[i + 1]]) for i in range(n_rows)]

    results = {
        "data": args.data,
        "rows": n_rows,
        "games": int(len(np.unique(games))),
        "models": {},
    }

    for spec in args.model:
        name, path = spec.split("=", 1)
        print("\n========================================================")
        print(f"Benchmarking Model: {name} ({path}) on {n_rows} positions")
        print("========================================================")
        model, _ = load_v6(path)
        model.to(device)

        # 1. Raw Policy
        orders = policy_order(model, dataset, device)
        policy_choice = np.array([o[0] for o in orders])
        policy_per = choice_metrics(cp_rows, dataset.lengths, policy_choice)
        policy_boot = bootstrap(policy_per, games)

        pol_key = f"{name}:policy"
        results["models"][pol_key] = {
            "mode": "policy",
            "cp_loss": policy_boot["cp_loss"],
            "best_rate": policy_boot["best_rate"],
            "blunder_rate": policy_boot["blunder_rate"],
        }
        print(f"{pol_key:28s}  cp lost: {policy_boot['cp_loss']['mean']:5.1f} "
              f"[{policy_boot['cp_loss']['lo']:.1f}, {policy_boot['cp_loss']['hi']:.1f}]  "
              f"best: {100 * policy_boot['best_rate']['mean']:4.1f}%  "
              f"blunder: {100 * policy_boot['blunder_rate']['mean']:4.1f}%")

        # 2. MCTS Grid
        for c in args.c_puct:
            for n in args.simulations:
                cfg = MCTSConfig(
                    simulations=n,
                    c_puct=c,
                    batch_size=args.batch_size,
                    virtual_loss_weight=args.vl_weight,
                    fpu_reduction=args.fpu_reduction,
                    temperature=0.0,
                )
                mcts_key = f"{name}:mcts_N{n}_c{c}"
                t0 = time.time()
                mcts_choice, avg_time = evaluate_mcts(model, fens, dataset, device, cfg)
                elapsed = time.time() - t0

                mcts_per = choice_metrics(cp_rows, dataset.lengths, mcts_choice)
                mcts_boot = bootstrap(mcts_per, games)

                # Paired difference vs raw policy
                paired_diff = bootstrap(
                    {k: np.asarray(mcts_per[k]) - np.asarray(policy_per[k]) for k in mcts_per},
                    games,
                )

                results["models"][mcts_key] = {
                    "mode": "mcts",
                    "simulations": n,
                    "c_puct": c,
                    "avg_sec_per_move": round(avg_time, 4),
                    "total_seconds": round(elapsed, 1),
                    "cp_loss": mcts_boot["cp_loss"],
                    "best_rate": mcts_boot["best_rate"],
                    "blunder_rate": mcts_boot["blunder_rate"],
                    "paired_vs_policy": paired_diff,
                }

                diff_cp = paired_diff["cp_loss"]["mean"]
                diff_lo = paired_diff["cp_loss"]["lo"]
                diff_hi = paired_diff["cp_loss"]["hi"]
                sign = "+" if diff_cp > 0 else ""

                print(f"{mcts_key:28s}  cp lost: {mcts_boot['cp_loss']['mean']:5.1f} "
                      f"[{mcts_boot['cp_loss']['lo']:.1f}, {mcts_boot['cp_loss']['hi']:.1f}]  "
                      f"best: {100 * mcts_boot['best_rate']['mean']:4.1f}%  "
                      f"blunder: {100 * mcts_boot['blunder_rate']['mean']:4.1f}%  "
                      f"diff: {sign}{diff_cp:4.1f} [{diff_lo:.1f}, {diff_hi:.1f}]  "
                      f"({avg_time * 1000:4.0f}ms/mv)")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote full benchmark results to {args.out}")


if __name__ == "__main__":
    main()
