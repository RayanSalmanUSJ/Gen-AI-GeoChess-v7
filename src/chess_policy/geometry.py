"""Finite non-wrapping lattice. D4 is square geometry, not chess-rule symmetry."""

from dataclasses import dataclass
import chess
import numpy as np

FULL = (1 << 64) - 1


def sq(file: int, rank: int) -> int:
    if not 0 <= file < 8 or not 0 <= rank < 8:
        raise ValueError("coordinates outside board")
    return 8 * rank + file


def coords(square: int) -> tuple[int, int]:
    if not 0 <= square < 64:
        raise ValueError("square outside board")
    return square % 8, square // 8


def name(square: int) -> str:
    coords(square)
    return chess.square_name(square)


def from_name(text: str) -> int:
    return chess.parse_square(text)


@dataclass(frozen=True)
class BB:
    """Immutable 64-bit set. XOR is addition in GF(2)^64."""

    bits: int = 0

    def __post_init__(self):
        object.__setattr__(self, "bits", self.bits & FULL)

    def __or__(self, other):
        return BB(self.bits | other.bits)

    def __and__(self, other):
        return BB(self.bits & other.bits)

    def __xor__(self, other):
        return BB(self.bits ^ other.bits)

    def __invert__(self):
        return BB(~self.bits)

    def __lshift__(self, count):
        return BB(self.bits << count)

    def __rshift__(self, count):
        return BB(self.bits >> count)

    def __iter__(self):
        return chess.scan_forward(self.bits)

    def __contains__(self, square):
        coords(square)
        return bool(self.bits & (1 << square))


FILE_MASKS = tuple(BB(x) for x in chess.BB_FILES)
RANK_MASKS = tuple(BB(x) for x in chess.BB_RANKS)


def dihedral_group():
    permutations = []
    for reflect in (False, True):
        for rotations in range(4):
            permutation = []
            for square in range(64):
                x, y = coords(square)
                if reflect:
                    x = 7 - x
                for _ in range(rotations):
                    x, y = 7 - y, x
                permutation.append(sq(x, y))
            permutations.append(permutation)
    return np.array(permutations, dtype=np.int64)


D4 = dihedral_group()
representatives = np.min(D4, axis=0)
ORBIT_IDS = np.searchsorted(np.unique(representatives), representatives)


def geometric_tables():
    squares = np.arange(64)
    dx = squares[None, :] % 8 - squares[:, None] % 8
    dy = squares[None, :] // 8 - squares[:, None] // 8
    delta = np.full((65, 65), 225, dtype=np.int64)
    delta[:64, :64] = (dx + 7) * 15 + dy + 7
    relations = np.zeros((65, 65, 5), dtype=np.float32)
    relations[:64, :64] = np.stack(
        [
            dx == 0,
            dy == 0,
            np.abs(dx) == np.abs(dy),
            np.maximum(np.abs(dx), np.abs(dy)) == 1,
            np.abs(dx) * np.abs(dy) == 2,
        ],
        axis=-1,
    )
    return delta, relations


def perft(board: chess.Board, depth: int) -> int:
    if depth < 0:
        raise ValueError("negative depth")
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        try:
            total += perft(board, depth - 1)
        finally:
            board.pop()
    return total
