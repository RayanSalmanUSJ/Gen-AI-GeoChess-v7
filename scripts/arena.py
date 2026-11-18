"""Paired-opening arena. Stockfish is the opponent only, never a runtime advisor."""

import argparse
import json
from pathlib import Path
import time
import chess
import chess.engine
import chess.pgn
from chess_policy.data import atomic_json, sha256
from chess_policy.inference import Policy

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
]


def run(checkpoint, stockfish, out, *, pairs=2, offset=0, elo=1600, seconds=0.1, max_plies=300):
    if pairs < 1 or offset < 0 or offset + pairs > len(OPENINGS) or seconds <= 0 or max_plies < 1:
        raise ValueError("invalid arena limits")
    policy = Policy(checkpoint)
    records, times = [], []
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(str(Path(stockfish).resolve()))
    started = time.monotonic()
    try:
        option = engine.options.get("UCI_Elo")
        if option is None or not option.min <= elo <= option.max:
            raise ValueError("opponent does not support requested Elo setting")
        engine.configure({"Threads": 1, "Hash": 32, "UCI_LimitStrength": True, "UCI_Elo": elo})
        identity = dict(engine.id)
        with out.with_suffix(".pgn").open("w") as pgn_file:
            for opening in OPENINGS[offset : offset + pairs]:
                for color in (chess.WHITE, chess.BLACK):
                    board = chess.Board()
                    for uci in opening.split():
                        board.push_uci(uci)
                    start_fen = board.fen()
                    game = chess.pgn.Game.from_board(board)
                    game.headers["White"] = "GeometricPolicy" if color else identity["name"]
                    game.headers["Black"] = identity["name"] if color else "GeometricPolicy"
                    node = game.end()
                    trace = []
                    game_id = object()
                    for _ in range(max_plies):
                        if board.is_game_over(claim_draw=True):
                            break
                        if board.turn == color:
                            result = policy.predict(board)
                            times.append(result["milliseconds"])
                            move = chess.Move.from_uci(result["move"])
                        else:
                            move = engine.play(
                                board, chess.engine.Limit(time=seconds), game=game_id
                            ).move
                        if move not in board.legal_moves:
                            raise AssertionError("illegal move in arena")
                        trace.append(move.uci())
                        board.push(move)
                        node = node.add_variation(move)
                    outcome = board.outcome(claim_draw=True)
                    result = outcome.result() if outcome else "*"
                    status = (
                        (
                            "draw"
                            if outcome.winner is None
                            else "win"
                            if outcome.winner == color
                            else "loss"
                        )
                        if outcome
                        else "unfinished"
                    )
                    game.headers["Result"] = result
                    pgn_file.write(str(game) + "\n\n")
                    pgn_file.flush()
                    record = {
                        "opening": opening,
                        "candidate_color": "white" if color else "black",
                        "initial_fen": start_fen,
                        "moves": trace,
                        "final_fen": board.fen(),
                        "result": result,
                        "status": status,
                        "reason": outcome.termination.name if outcome else "ply_cap",
                    }
                    # Independently replay the move trace before accepting the record.
                    check = chess.Board(start_fen)
                    for text in trace:
                        check.push_uci(text)
                    assert check.fen() == board.fen()
                    records.append(record)
                    print(
                        json.dumps({"game": len(records), "status": status, "plies": len(trace)}),
                        flush=True,
                    )
        report = {
            "checkpoint_sha256": sha256(checkpoint),
            "opponent": identity,
            "opponent_binary_sha256": sha256(stockfish),
            "uci_elo": elo,
            "opponent_seconds_per_move": seconds,
            "candidate": "one neural forward pass; no search",
            "candidate_mean_ms": sum(times) / len(times),
            "max_plies": max_plies,
            "counts": {
                x: sum(r["status"] == x for r in records)
                for x in ("win", "draw", "loss", "unfinished")
            },
            "games": records,
            "seconds": time.monotonic() - started,
            "limitations": "Unequal time budgets; rating is an opponent configuration, not a calibrated Elo estimate. Ply caps are unfinished, not draws. Both sides claim available draws.",
        }
        atomic_json(out, report)
        return report
    finally:
        engine.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/policy_v5.pt"))
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/arena.json"))
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--elo", type=int, default=1600)
    parser.add_argument("--seconds", type=float, default=0.1)
    parser.add_argument("--max-plies", type=int, default=300)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
