"""Exact offline audit of mate opportunities in completed paired-arena traces."""

import argparse
import json
from pathlib import Path

import chess

from chess_policy.data import atomic_json
from chess_policy.mate_curriculum import mate_moves
from chess_policy.tactics import safe_moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arena", type=Path, default=Path("v5_results/paired_arena"))
    p.add_argument("--out", type=Path, default=Path("v5_results/game_tactic_audit.json"))
    args = p.parse_args()
    models = {}
    for name in ["baseline", "candidate"]:
        decisions = opportunities = misses = allows = avoidable = 0
        events = []
        for file in sorted(args.arena.glob(name + "_pair_*.json")):
            report = json.loads(file.read_text())
            for game_index, game in enumerate(report["games"]):
                board = chess.Board(game["initial_fen"])
                color = chess.WHITE if game["candidate_color"] == "white" else chess.BLACK
                for ply, uci in enumerate(game["moves"]):
                    if board.turn == color:
                        decisions += 1
                        mating = mate_moves(board)
                        opportunities += bool(mating)
                        missed = bool(mating and uci not in mating)
                        misses += missed
                        board.push_uci(uci)
                        replies = (
                            mate_moves(board) if not board.is_game_over(claim_draw=False) else []
                        )
                        board.pop()
                        prevents = []
                        if replies:
                            allows += 1
                            prevents, _ = safe_moves(board)
                            avoidable += bool(prevents)
                        if missed or replies:
                            events.append(
                                {
                                    "file": file.name,
                                    "game": game_index,
                                    "ply": ply,
                                    "fen": board.fen(),
                                    "chosen": uci,
                                    "missed_mates": mating if missed else [],
                                    "opponent_mating_replies": replies,
                                    "preventing_moves": prevents,
                                }
                            )
                    board.push_uci(uci)
        models[name] = {
            "decisions": decisions,
            "mate_in_one_opportunities": opportunities,
            "missed_mate_in_one": misses,
            "allows_mate_in_one": allows,
            "avoidable_immediate_mate_blunders": avoidable,
            "events": events,
        }
    result = {
        "models": models,
        "method": "Exact legal-move enumeration offline; no teacher evaluation or runtime intervention.",
        "limitation": "Models reach different positions. Counts diagnose each trajectory and are not a matched-position causal comparison.",
    }
    atomic_json(args.out, result)
    print(
        json.dumps(
            {k: {q: v for q, v in d.items() if q != "events"} for k, d in models.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
