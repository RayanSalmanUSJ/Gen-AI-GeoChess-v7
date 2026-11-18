"""Measure warmed single-position encode and inference latency over varied positions."""

import argparse
import json
from pathlib import Path
import random
import time
import chess
import numpy as np
from chess_policy.data import atomic_json
from chess_policy.encoding import encode
from chess_policy.inference import Policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/policy_v5.pt"))
    p.add_argument("--out", type=Path, default=Path("results/runtime.json"))
    p.add_argument("--positions", type=int, default=100)
    a = p.parse_args()
    if a.positions < 1:
        raise ValueError("positive positions required")
    policy = Policy(a.checkpoint)
    rng = random.Random(903)
    board = chess.Board()
    timings = []
    encoding = []
    policy.predict(board)
    for _ in range(a.positions):
        if board.is_game_over():
            board = chess.Board()
        start = time.perf_counter()
        encode(board)
        encoding.append((time.perf_counter() - start) * 1000)
        result = policy.predict(board)
        assert chess.Move.from_uci(result["move"]) in board.legal_moves
        timings.append(result["milliseconds"])
        board.push(rng.choice(list(board.legal_moves)))
    report = {
        "positions": len(timings),
        "legal_outputs": len(timings),
        "encoding_median_ms": float(np.median(encoding)),
        "prediction_median_ms": float(np.median(timings)),
        "prediction_p95_ms": float(np.quantile(timings, 0.95)),
        "note": "CPU, four PyTorch threads, warmed one-position calls; encoding plus model plus output included. Random walk is a legality/runtime probe, not a strength test.",
    }
    atomic_json(a.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
