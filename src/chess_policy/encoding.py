"""Versioned symbolic text tokens, exact attack graphs and legal-move baskets."""

import hashlib
import chess
import numpy as np
import torch

ENCODING_VERSION = 1
PROMOTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
MOVE_VOCAB = 64 * 64 * 5
FEATURE_SCALE = (8, 8, 16, 16, 1, 1, 1, 100)


def move_id(move):
    if not move or move.drop is not None:
        raise ValueError("only orthodox non-null moves have tokens")
    return (move.from_square * 64 + move.to_square) * 5 + PROMOTIONS.index(move.promotion)


def id_move(token):
    if not 0 <= token < MOVE_VOCAB:
        raise ValueError("move token out of range")
    pair, promotion = divmod(int(token), 5)
    return chess.Move(pair // 64, pair % 64, promotion=PROMOTIONS[promotion])


def position_key(board):
    return " ".join(board.fen(en_passant="fen").split()[:4])


def fingerprint(board):
    return hashlib.blake2b(position_key(board).encode(), digest_size=16).digest()


def encode(board, *, validate=True):
    if board.chess960 or (validate and not board.is_valid()):
        raise ValueError("expected a valid orthodox position")
    pieces = np.zeros(64, dtype=np.uint8)
    geometry = np.zeros((64, 8), dtype=np.uint8)
    attacks = np.zeros((64, 64), dtype=np.uint8)
    for square, piece in board.piece_map().items():
        pieces[square] = piece.piece_type + (0 if piece.color else 6)
        targets = list(chess.scan_forward(board.attacks_mask(square)))
        attacks[square, targets] = 1
        geometry[targets, 0 if piece.color else 1] += 1
    legal = sorted(board.legal_moves, key=move_id)
    for move in legal:
        geometry[move.from_square, 2] += 1
        geometry[move.to_square, 3] += 1
    if board.ep_square is not None:
        geometry[board.ep_square, 4] = 1
    for square in chess.scan_forward(board.clean_castling_rights()):
        geometry[square, 5] = 1
    geometry[:, 6] = int(board.turn)
    geometry[:, 7] = min(board.halfmove_clock, 100)
    return {
        "pieces": pieces,
        "geometry": geometry,
        "attacks": np.packbits(attacks, axis=-1, bitorder="little"),
        "legal": np.array([move_id(m) for m in legal], dtype=np.int16),
    }


def text_tokens(board):
    pieces = board.piece_map()
    return (
        [f"{chess.square_name(s)}:{pieces[s].symbol() if s in pieces else '.'}" for s in range(64)]
        + board.fen(en_passant="fen").split()[1:5]
        + ["<move>"]
    )


def collate(examples):
    if not examples or any(len(x["legal"]) == 0 for x in examples):
        raise ValueError("an empty or terminal batch has no policy distribution")
    width = max(len(x["legal"]) for x in examples)
    legal = np.full((len(examples), width), -1, dtype=np.int64)
    targets = []
    for row, item in enumerate(examples):
        legal[row, : len(item["legal"])] = item["legal"]
        if "target" in item:
            where = np.flatnonzero(item["legal"] == item["target"])
            if len(where) != 1:
                raise ValueError("target must occur exactly once among legal moves")
            targets.append(int(where[0]))
    batch = {
        "pieces": torch.from_numpy(np.stack([x["pieces"] for x in examples]).astype(np.int64)),
        "geometry": torch.from_numpy(np.stack([x["geometry"] for x in examples])),
        "attacks": torch.from_numpy(
            np.unpackbits(np.stack([x["attacks"] for x in examples]), axis=-1, bitorder="little")
        ),
        "legal": torch.from_numpy(legal),
    }
    if targets:
        if len(targets) != len(examples):
            raise ValueError("mixed labeled and unlabeled batch")
        batch["target"] = torch.tensor(targets, dtype=torch.long)
    return batch
