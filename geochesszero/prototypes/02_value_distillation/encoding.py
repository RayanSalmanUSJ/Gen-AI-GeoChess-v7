"""
Board / move encoding for Prototype 2.

Differences from Prototype 1 (shared/chess_utils.py):
  * Side-to-move relative: if Black is to move, the board is mirrored
    (python-chess `Board.mirror()`), so the network always plays "White".
    This is the colour-swap symmetry from the notes, enforced exactly in the
    input instead of being learned. It is applied identically to every
    attention variant, so the attention comparison still changes one thing.
  * 17 channels per square: 6 "us" pieces, 6 "them" pieces,
    4 castling rights (us K, us Q, them K, them Q), en-passant square.
  * The 4,272-action vocabulary of Prototype 1 is reused unchanged
    (shared/move_encoder.py).
"""

import os
import sys

import chess
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.move_encoder import NUM_ACTIONS, move_to_index  # noqa: E402

N_CHANNELS = 17


def oriented(board: chess.Board) -> chess.Board:
    """Return the board seen from the side to move (always White to move)."""
    return board if board.turn == chess.WHITE else board.mirror()


def orient_move(move: chess.Move, turn: bool) -> chess.Move:
    """Map a real move into the oriented frame (or back: the map is an involution)."""
    if turn == chess.WHITE:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def encode(board: chess.Board) -> np.ndarray:
    """(64, 17) float32 features of the side-to-move-oriented board."""
    b = oriented(board)
    x = np.zeros((64, N_CHANNELS), dtype=np.float32)
    for sq, piece in b.piece_map().items():
        offset = 0 if piece.color == chess.WHITE else 6
        x[sq, offset + piece.piece_type - 1] = 1.0
    x[:, 12] = float(b.has_kingside_castling_rights(chess.WHITE))
    x[:, 13] = float(b.has_queenside_castling_rights(chess.WHITE))
    x[:, 14] = float(b.has_kingside_castling_rights(chess.BLACK))
    x[:, 15] = float(b.has_queenside_castling_rights(chess.BLACK))
    if b.ep_square is not None:
        x[b.ep_square, 16] = 1.0
    return x


def occupancy(board: chess.Board) -> np.ndarray:
    """(64,) occupancy of the oriented board, used by the ray-aware masks."""
    b = oriented(board)
    occ = np.zeros(64, dtype=np.float32)
    for sq in b.piece_map():
        occ[sq] = 1.0
    return occ


def action_index(board: chess.Board, move: chess.Move) -> int:
    """Action index of a real move, in the oriented frame."""
    return move_to_index(orient_move(move, board.turn))


def legal_actions(board: chess.Board):
    """List of (real move, oriented action index) for every legal move."""
    return [(m, action_index(board, m)) for m in board.legal_moves]


# ---------------------------------------------------------------------------
# Value conversions
# ---------------------------------------------------------------------------

MATE_CP = 10000
CLIP_CP = 2000


def cp_to_value(cp):
    """Centipawns (side-to-move POV) -> expected score in [-1, 1].

    Uses the logistic curve Lichess uses for win chances
    (2 / (1 + exp(-0.00368208 * cp)) - 1). Mate scores saturate.
    """
    cp = np.clip(np.asarray(cp, dtype=np.float64), -CLIP_CP, CLIP_CP)
    return 2.0 / (1.0 + np.exp(-0.00368208 * cp)) - 1.0


def soft_policy_target(cps, temperature=0.06):
    """Teacher distribution over legal moves: softmax of expected score / T.

    Working in expected-score space (not raw centipawns) means a 50 cp
    difference matters near equality and barely matters at +8.
    With T = 0.06 on the [-1, 1] scale, a move that gives away about
    3 percentage points of expected score gets weight e^-1 of the best.
    """
    v = cp_to_value(cps)
    z = (v - v.max()) / temperature
    p = np.exp(z)
    return (p / p.sum()).astype(np.float32)


if __name__ == "__main__":
    b = chess.Board()
    b.push_san("e4")
    x = encode(b)
    print("encoded", x.shape, "pieces:", int(x[:, :12].sum()))
    print("legal actions:", len(legal_actions(b)), "of", NUM_ACTIONS)
    print("value(+100cp) =", float(cp_to_value(100)))
