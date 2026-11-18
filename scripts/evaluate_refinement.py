"""Evaluate frozen arms on fresh games and revisit excluded diagnostic mate positions."""

import argparse
import json
from pathlib import Path

import chess

from chess_policy.data import atomic_json, sha256
from chess_policy.evaluation import evaluate
from chess_policy.inference import Policy
from chess_policy.mate_curriculum import mate_moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("v4_holdout"))
    p.add_argument("--out", type=Path, default=Path("v4_results"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = p.parse_args()
    selection = json.loads((args.out / "selection.json").read_text())
    if selection["selected_sha256"] != sha256(args.checkpoints / "policy_v4.pt"):
        raise ValueError("selected checkpoint changed")
    paths = {
        "v3": Path("checkpoints/pilot.pt"),
        "control": args.checkpoints / "control_best.pt",
        "targeted": args.checkpoints / "targeted_best.pt",
    }
    reports = {}
    diagnostic = json.loads((args.out / "v3_loss_audit.json").read_text())
    opportunities = [r for r in diagnostic["rows"] if r["missed_mate_in_one"]]
    old_mates = {}
    for name, path in paths.items():
        reports[name] = evaluate(
            path, args.data, args.out / f"fresh_{name}.json", threads=args.threads
        )
        policy = Policy(path, threads=args.threads)
        rows = []
        for row in opportunities:
            board = chess.Board(row["fen"])
            predicted = policy.predict(board)["move"]
            rows.append(
                {
                    "fen": row["fen"],
                    "prediction": predicted,
                    "mates": mate_moves(board),
                    "correct": predicted in mate_moves(board),
                }
            )
        old_mates[name] = {
            "correct": sum(r["correct"] for r in rows),
            "opportunities": len(rows),
            "rows": rows,
        }
        print(
            json.dumps(
                {
                    "model": name,
                    "top1": reports[name]["top1"],
                    "nll": reports[name]["nll"],
                    "diagnostic_mates_correct": old_mates[name]["correct"],
                }
            ),
            flush=True,
        )
    atomic_json(
        args.out / "diagnostic_revisit.json",
        {
            "models": old_mates,
            "limitation": "Repeated opportunities from previously inspected games; excluded from training, but this is a diagnostic, not an independent strength test.",
        },
    )
    atomic_json(
        args.out / "fresh_comparison.json",
        {"models": reports, "selection_sha256": sha256(args.out / "selection.json")},
    )


if __name__ == "__main__":
    main()
