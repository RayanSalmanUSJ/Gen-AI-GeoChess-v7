"""Pre-registered paired-opening comparison with frozen models and explicit budgets."""

import argparse
import importlib.util
import json
from pathlib import Path

import chess
import numpy as np

from chess_policy.data import atomic_json, sha256

OPENINGS = [
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6",
    "c2c4 c7c5 g1f3 g8f6 b1c3 b8c6",
    "d2d4 f7f5 g2g3 g8f6 f1g2 g7g6",
    "e2e4 e7e5 b1c3 g8f6 f1c4 b8c6",
]


def run(baseline, candidate, stockfish, out):
    for opening in OPENINGS:
        b = chess.Board()
        for uci in opening.split():
            b.push_uci(uci)
    spec = importlib.util.spec_from_file_location("v4_arena", Path(__file__).with_name("arena.py"))
    arena = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arena)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = {name: [] for name in ["baseline", "candidate"]}
    # Alternate engine order across opening pairs. Models are frozen before this call.
    for pair, opening in enumerate(OPENINGS):
        arena.OPENINGS = [opening]
        order = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        for name in order:
            checkpoint = baseline if name == "baseline" else candidate
            result = arena.run(
                checkpoint,
                stockfish,
                out / f"{name}_pair_{pair}.json",
                pairs=1,
                elo=1600,
                seconds=0.1,
            )
            results[name].append((pair, result))
    summary = {}
    scores = {}
    for name, entries in results.items():
        entries.sort(key=lambda item: item[0])
        games = [game for _, result in entries for game in result["games"]]
        counts = {
            status: sum(g["status"] == status for g in games)
            for status in ["win", "draw", "loss", "unfinished"]
        }
        scores[name] = np.array(
            [
                sum(
                    1 if g["status"] == "win" else 0.5 if g["status"] == "draw" else 0
                    for g in result["games"]
                )
                / 2
                for _, result in entries
            ]
        )
        summary[name] = {
            "counts": counts,
            "completed_score": (counts["win"] + 0.5 * counts["draw"])
            / max(1, len(games) - counts["unfinished"]),
            "checkpoint_sha256": sha256(baseline if name == "baseline" else candidate),
        }
    difference = scores["candidate"] - scores["baseline"]
    rng = np.random.default_rng(427)
    interval = np.quantile(
        [
            difference[rng.integers(0, len(difference), len(difference))].mean()
            for _ in range(10000)
        ],
        [0.025, 0.975],
    ).tolist()
    report = {
        "models": summary,
        "opening_pairs": len(OPENINGS),
        "games_per_model": 2 * len(OPENINGS),
        "paired_score_difference": float(difference.mean()),
        "pair_bootstrap_95": interval,
        "limitations": "One small paired sample; Stockfish strength limiting is stochastic. Unequal inference budgets. No calibrated Elo claim. Unfinished games, if present, are excluded from completed_score and pessimistically scored zero in pair differences.",
    }
    atomic_json(out / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, default=Path("checkpoints/pilot.pt"))
    p.add_argument("--candidate", type=Path, default=Path("checkpoints/policy_v4.pt"))
    p.add_argument("--stockfish", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("v4_results/paired_arena"))
    run(**vars(p.parse_args()))
