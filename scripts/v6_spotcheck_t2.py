"""Re-score T2 rows with our Stockfish (depth 12) and compare to the stored labels.

  HASH=16 python scripts/v6_spotcheck_t2.py benchmarks/t2_p2_1312 <stockfish.exe> 0 500 1000
"""
import os
import sys
import numpy as np
import chess
import chess.engine
from chess_policy.v6_data import score_all_moves
from chess_policy.encoding import move_id

root, engine_path = sys.argv[1], sys.argv[2]
rows = [int(x) for x in sys.argv[3:]] or [0, 500, 1000]
fens = open(f"{root}/fen.txt").read().split("\n")
cp, off, legal = (np.load(f"{root}/{n}.npy") for n in ("cp", "offsets", "legal"))
eng = chess.engine.SimpleEngine.popen_uci(engine_path)
eng.configure({"Threads": 1, "Hash": int(os.environ.get("HASH", 16))})
for r in rows:
    board = chess.Board(fens[r])
    ours, missing = score_all_moves(eng, board, chess.engine.Limit(depth=12))
    stored = cp[off[r]:off[r + 1]].astype(int)
    ids = sorted(move_id(m) for m in board.legal_moves)
    same_ids = list(legal[off[r]:off[r + 1]]) == ids
    diff = np.abs(np.array(ours) - stored)
    best_ours, best_stored = int(np.argmax(ours)), int(np.argmax(stored))
    print(f"row {r}: {len(ours)} moves, move order matches={same_ids}, exact={int((diff == 0).sum())}/{len(diff)}, "
          f"max diff={int(diff.max())} cp, median diff={float(np.median(diff))}, same best move={best_ours == best_stored}, missing={missing}")
eng.quit()
