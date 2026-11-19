"""Gate 1: total missing for a label dir + re-score 3 rows with a fresh engine.

  python scripts/v6_gate1_check.py data/v6/positions_train.npz data/v6/labels_dev_d10 <stockfish.exe>
(run with src on PYTHONPATH; a fresh engine differs by ~10-20 cp from labels made with a warm hash)
"""
import sys
import numpy as np
import chess
import chess.engine
from chess_policy.v6_data import load_labels, score_all_moves

pos_file, label_dir, engine_path = sys.argv[1:4]
rows, cp, lens, missing, meta = load_labels(label_dir)
print(f"{label_dir}: {len(rows)} positions, {len(cp)} move scores, depth {meta['depth']}, "
      f"missing (scored singly) = {missing} ({100 * missing / len(cp):.3f}% of moves)")
pos = np.load(pos_file)
offs = np.concatenate([[0], np.cumsum(lens)])
eng = chess.engine.SimpleEngine.popen_uci(engine_path)
eng.configure({"Threads": 1, "Hash": 16})
for i in (7, len(rows) // 2, len(rows) - 3):
    board = chess.Board(pos["fen"][rows[i]].decode())
    ours, _ = score_all_moves(eng, board, chess.engine.Limit(depth=meta["depth"]))
    stored = cp[offs[i]:offs[i + 1]].astype(int)
    d = np.abs(np.array(ours) - stored)
    print(f"  row {i}: {len(stored)} moves, same best move={int(np.argmax(ours)) == int(np.argmax(stored))}, "
          f"best score stored {stored.max()} vs direct {max(ours)}, median diff {np.median(d):.0f} cp, "
          f"exact {int((d == 0).sum())}/{len(d)}")
eng.quit()
