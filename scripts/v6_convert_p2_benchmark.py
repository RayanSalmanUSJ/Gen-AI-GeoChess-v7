"""Convert Prototype 2's 1,312-position benchmark (Stockfish 15.1 depth 12, every legal
move scored) into a v6 dataset directory, so v6_eval scores models on the exact
positions of the 18 Nov cross-evaluation.

  python scripts/v6_convert_p2_benchmark.py <P2 data/test.npz> <P2 folder> benchmarks/t2_p2_1312
"""
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chess_policy.encoding import encode, move_id  # noqa: E402


def main(test_npz, p2_dir, out):
    sys.path.insert(0, p2_dir)
    import encoding as p2enc  # Prototype 2's action indexing

    d = np.load(test_npz, allow_pickle=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pieces, geometry, attacks, legal, cps, lengths, targets = [], [], [], [], [], [], []
    fens = [str(f) for f in d["fen"]]
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        e = encode(board, validate=False)
        moves = sorted(board.legal_moves, key=move_id)
        slot = {int(a): j for j, a in enumerate(d["idx"][i]) if a >= 0}
        row = [int(d["cp"][i, slot[p2enc.action_index(board, m)]]) for m in moves]
        pieces.append(e["pieces"]); geometry.append(e["geometry"]); attacks.append(e["attacks"])
        legal.append(e["legal"]); cps.append(row); lengths.append(len(moves))
        # the benchmark has no recorded move; mark Stockfish's best as "target"
        targets.append(int(np.argmax(row)))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    arrays = dict(pieces=np.stack(pieces), geometry=np.stack(geometry), attacks=np.stack(attacks),
                  legal=np.concatenate(legal).astype(np.int16), offsets=offsets,
                  target=np.array(targets, np.int16), result=np.zeros(len(fens), np.int8),
                  game=np.asarray(d["game"]).astype(np.int32),
                  ply=np.asarray(d["ply"]).astype(np.int16),
                  source_row=np.arange(len(fens)), cp=np.concatenate(cps).astype(np.int16))
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v, allow_pickle=False)
    (out / "fen.txt").write_text("\n".join(fens) + "\n")
    (out / "meta.json").write_text(json.dumps(dict(
        split="test", rows=len(fens), labelled=True, depth=12,
        source="Prototype 2 benchmark: 24 held-out games, Stockfish 15.1 depth 12",
        note="no recorded moves: target = Stockfish's best move"), indent=2) + "\n")
    print(len(fens), "positions written to", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
