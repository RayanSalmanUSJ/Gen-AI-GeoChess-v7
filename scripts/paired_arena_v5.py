"""Frozen v4 versus v5 on eight openings reserved for the v5 experiment."""

import argparse
from pathlib import Path

import paired_arena

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4",
    "d2d4 d7d5 c2c4 c7c6 b1c3 g8f6",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6",
    "e2e4 e7e5 f2f4 e5f4 g1f3 g7g5",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("checkpoints/policy_v4.pt"))
    parser.add_argument("--candidate", type=Path, default=Path("checkpoints/policy_v5.pt"))
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("v5_results/paired_arena"))
    paired_arena.OPENINGS = OPENINGS
    paired_arena.run(**vars(parser.parse_args()))
