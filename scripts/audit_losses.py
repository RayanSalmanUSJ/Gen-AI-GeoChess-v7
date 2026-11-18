"""Offline teacher audit of recorded decisions. Never used by the playing policy."""

import argparse
import json
from pathlib import Path
import time

import chess
import chess.engine
import numpy as np

from chess_policy.data import atomic_json, sha256


def mating_moves(board):
    found = []
    for move in list(board.legal_moves):
        board.push(move)
        if board.is_checkmate():
            found.append(move.uci())
        board.pop()
    return found


def analyze(engine, board, nodes, root_moves=None):
    result = engine.analyse(
        board, chess.engine.Limit(nodes=nodes), root_moves=root_moves, game=object()
    )
    score = result["score"].pov(board.turn)
    return {
        "cp": score.score(mate_score=100000),
        "mate": score.mate(),
        "depth": result.get("depth"),
        "nodes": result.get("nodes"),
        "move": result["pv"][0].uci() if result.get("pv") else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", type=Path, default=Path("results/arena.json"))
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=Path("v4_results/v3_loss_audit.json"))
    args = parser.parse_args()
    match = json.loads(args.arena.read_text())
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve()))
    engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": False})
    rows = []
    started = time.monotonic()
    try:
        for game_id, game in enumerate(match["games"]):
            board = chess.Board()
            for uci in game["opening"].split():
                board.push_uci(uci)
            color = chess.WHITE if game["candidate_color"] == "white" else chess.BLACK
            for ply, uci in enumerate(game["moves"]):
                move = chess.Move.from_uci(uci)
                if board.turn == color:
                    best = analyze(engine, board, args.nodes)
                    chosen = (
                        best if uci == best["move"] else analyze(engine, board, args.nodes, [move])
                    )
                    mates = mating_moves(board)
                    before = board.fen()
                    board.push(move)
                    allowed_mates = mating_moves(board)
                    board.pop()
                    row = {
                        "game": game_id,
                        "ply": ply,
                        "fen": before,
                        "chosen": uci,
                        "best": best,
                        "chosen_analysis": chosen,
                        "regret_cp_proxy": max(0, best["cp"] - chosen["cp"]),
                        "missed_mate_in_one": bool(mates and uci not in mates),
                        "allows_mate_in_one": bool(allowed_mates),
                        "already_lost_proxy": best["cp"] < -500,
                    }
                    rows.append(row)
                board.push(move)
            print(json.dumps({"audited_games": game_id + 1, "decisions": len(rows)}), flush=True)
        viable = [r for r in rows if not r["already_lost_proxy"]]
        result = {
            "opponent_id": dict(engine.id),
            "nodes_per_query": args.nodes,
            "source_arena_sha256": sha256(args.arena),
            "decisions": len(rows),
            "regret_at_least_100cp": sum(r["regret_cp_proxy"] >= 100 for r in rows),
            "regret_at_least_300cp": sum(r["regret_cp_proxy"] >= 300 for r in rows),
            "viable_decisions": len(viable),
            "viable_regret_at_least_300cp": sum(r["regret_cp_proxy"] >= 300 for r in viable),
            "median_regret_cp_proxy": float(np.median([r["regret_cp_proxy"] for r in rows])),
            "missed_mate_in_one": sum(r["missed_mate_in_one"] for r in rows),
            "allows_mate_in_one": sum(r["allows_mate_in_one"] for r in rows),
            "rows": rows,
            "seconds": time.monotonic() - started,
            "limitations": "Independent finite-node teacher searches are noisy proxies. Negative score differences are clipped to zero. Scores involving mate use a finite sentinel. This diagnostic does not prove causal blame or optimal play.",
        }
        atomic_json(args.out, result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
