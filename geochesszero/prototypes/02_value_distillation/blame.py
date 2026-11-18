"""
Hindsight blame report for one game: the "BLAME" idea from the notes, made
concrete with a teacher instead of a theorem.

For every move by the chosen side, the teacher scores the best move and the
move actually played from the position before it. The regret is the
difference. Blame shares follow blame_transformer.pdf:

    beta_i = max(0, regret_i - tau) / sum_j max(0, regret_j - tau)

so tiny inaccuracies (< tau cp) are not blamed and the shares sum to 1. This
(game, blame vector) pair is exactly the training target that the proposed
backward "blame transformer" would learn to predict. This script produces the
labels; it does not train that model.

Usage:
  python blame.py --arena results/arena_ray_vs_sf.json --game 0 --side a
"""

import argparse
import json
import os

import chess
import chess.engine


def regrets(moves, fen, side_white, engine, depth):
    board = chess.Board(fen)
    out = []
    lim = chess.engine.Limit(depth=depth)
    for ply, u in enumerate(moves):
        mv = chess.Move.from_uci(u)
        if board.turn == (chess.WHITE if side_white else chess.BLACK):
            best = engine.analyse(board, lim)
            played = engine.analyse(board, lim, root_moves=[mv])
            b = best["score"].pov(board.turn).score(mate_score=10000)
            p = played["score"].pov(board.turn).score(mate_score=10000)
            b, p = max(-1000, min(1000, b)), max(-1000, min(1000, p))
            out.append(dict(ply=ply, move_no=board.fullmove_number,
                            san=board.san(mv), best_san=board.san(best["pv"][0]),
                            score_best=b, score_played=p, regret=max(0, b - p)))
        board.push(mv)
    return out


def blame_shares(rows, tau=50):
    raw = [max(0, r["regret"] - tau) for r in rows]
    tot = sum(raw)
    for r, x in zip(rows, raw):
        r["blame"] = x / tot if tot > 0 else 0.0
    return rows


def narrative(rows, side_white, result, top=3):
    who = "White" if side_white else "Black"
    dots = "." if side_white else "..."
    ranked = sorted([r for r in rows if r["blame"] > 0], key=lambda r: -r["blame"])[:top]
    if not ranked:
        return f"{who} ({result}): no move lost more than the threshold; nothing to blame."
    parts = [f"{r['move_no']}{dots}{r['san']} carries {100 * r['blame']:.0f}% of the blame "
             f"(it left {who} at {r['score_played']:+d} cp; {r['move_no']}{dots}{r['best_san']} "
             f"kept {r['score_best']:+d})"
             for r in ranked]
    return f"{who} ({result}): " + "; ".join(parts) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", required=True)
    ap.add_argument("--game", type=int, default=None, help="game index; default = first loss of side a")
    ap.add_argument("--side", choices=["a", "b"], default="a")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--tau", type=int, default=50)
    ap.add_argument("--engine", default=os.environ.get("STOCKFISH", "stockfish"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = json.load(open(a.arena))
    games = data["games"]
    if a.game is None:
        lost = [i for i, g in enumerate(games) if (g["score_a"] == 0) == (a.side == "a")
                and g["score_a"] != 0.5]
        a.game = lost[0] if lost else 0
    g = games[a.game]
    side_white = g["a_white"] if a.side == "a" else not g["a_white"]
    eng = chess.engine.SimpleEngine.popen_uci(a.engine)
    eng.configure({"Threads": 1, "Hash": 64})
    rows = blame_shares(regrets(g["moves"], g["fen"], side_white, eng, a.depth), a.tau)
    eng.quit()
    text = narrative(rows, side_white, g["result"])
    player = data["summary"]["a" if a.side == "a" else "b"]
    print(f"game {a.game}: {player} as {'White' if side_white else 'Black'}, result {g['result']}")
    print(text)
    print(f"\n{'move':>8s} {'played':>8s} {'best':>8s} {'regret':>7s} {'blame':>6s}")
    for r in rows:
        if r["regret"] > 0:
            print(f"{r['move_no']:>8d} {r['san']:>8s} {r['best_san']:>8s} {r['regret']:>7d} "
                  f"{100 * r['blame']:5.0f}%")
    if a.out:
        json.dump(dict(game=a.game, player=player, side_white=side_white, result=g["result"],
                       narrative=text, moves=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
