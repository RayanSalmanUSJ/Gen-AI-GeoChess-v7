"""
Turn teacher-labelled JSONL into compact training tensors.

Per position we store only the legal moves (padded to L_MAX):
  x        (N, 64, 17)  uint8   oriented board features
  occ      (N, 64)      uint8   occupancy (for ray masks)
  idx      (N, L)       int16   oriented action index of each legal move, -1 = pad
  cp       (N, L)       int16   teacher score of each legal move (side-to-move cp)
  target   (N, L)       float16 soft teacher policy (softmax of expected score / T)
  value    (N,)         float32 teacher value of the position (best move), in [-1, 1]
  game     (N,)         int32   source game id (splits and bootstrap are by game)

Positions are de-duplicated by (board, side, castling, ep) within a file.
"""

import argparse
import json

import chess
import numpy as np

import encoding as enc

L_MAX = 128  # more legal moves than this is vanishingly rare in real games


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # tolerate a truncated final line
                continue
    return rows


def position_key(fen):
    return " ".join(fen.split()[:4])


def build(rows, temperature=0.06, dedupe=True, exclude=()):
    seen, keep = set(exclude), []
    for r in rows:
        key = position_key(r["fen"])
        if key in exclude or (dedupe and key in seen):
            continue
        seen.add(key)
        if len(r["moves"]) == 0 or len(r["moves"]) > L_MAX:
            continue
        keep.append(r)
    N = len(keep)
    x = np.zeros((N, 64, enc.N_CHANNELS), np.uint8)
    occ = np.zeros((N, 64), np.uint8)
    idx = np.full((N, L_MAX), -1, np.int16)
    cp = np.zeros((N, L_MAX), np.int16)
    target = np.zeros((N, L_MAX), np.float16)
    value = np.zeros(N, np.float32)
    game = np.zeros(N, np.int32)
    ply = np.zeros(N, np.int16)
    fens = []
    for i, r in enumerate(keep):
        b = chess.Board(r["fen"])
        x[i] = enc.encode(b)
        occ[i] = enc.occupancy(b)
        n = len(r["moves"])
        scores = np.clip(np.array(r["cp"], np.int64), -enc.MATE_CP, enc.MATE_CP)
        for j, u in enumerate(r["moves"]):
            idx[i, j] = enc.action_index(b, chess.Move.from_uci(u))
        cp[i, :n] = np.clip(scores, -30000, 30000)
        target[i, :n] = enc.soft_policy_target(scores, temperature)
        value[i] = enc.cp_to_value(scores.max())
        game[i] = r["game"]
        ply[i] = r["ply"]
        fens.append(r["fen"])
    return dict(x=x, occ=occ, idx=idx, cp=cp, target=target, value=value,
                game=game, ply=ply, fen=np.array(fens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--temperature", type=float, default=0.06)
    ap.add_argument("--exclude", default=None,
                    help="JSONL whose positions must not appear here (keeps the test set held out)")
    a = ap.parse_args()
    rows = load_jsonl(a.inp)
    exclude = {position_key(r["fen"]) for r in load_jsonl(a.exclude)} if a.exclude else set()
    d = build(rows, a.temperature, exclude=exclude)
    np.savez_compressed(a.out, **d)
    n_moves = (d["idx"] >= 0).sum(1)
    print(f"{a.inp}: {len(rows)} rows -> {len(d['value'])} unique positions, "
          f"{len(np.unique(d['game']))} games, legal moves mean {n_moves.mean():.1f} "
          f"max {n_moves.max()}; teacher-best probability mean "
          f"{d['target'].astype(np.float32).max(1).mean():.3f}")


if __name__ == "__main__":
    main()
