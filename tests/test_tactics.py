import json
import random

import chess
import pytest

from chess_policy.data import sha256
from chess_policy.encoding import fingerprint
from chess_policy.mate_curriculum import mate_moves
from chess_policy.tactics import TacticDataset, augment, safe_moves


def test_defensive_targets_partition_legal_moves_and_restore_board():
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 b - - 0 1")
    # Add a defender with both protective and irrelevant legal moves.
    board.set_piece_at(chess.A7, chess.Piece(chess.ROOK, chess.BLACK))
    assert board.is_valid()
    before = board.fen()
    good, bad = safe_moves(board)
    assert good and bad and not set(good) & set(bad)
    assert set(good) | set(bad) == {m.uci() for m in board.legal_moves}
    assert before == board.fen() and not board.move_stack
    for uci in bad:
        board.push_uci(uci)
        assert mate_moves(board)
        board.pop()
    for uci in good:
        board.push_uci(uci)
        assert not mate_moves(board)
        board.pop()


def test_augmentation_preserves_original_pieces_and_adds_pawns():
    seed = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    board = augment(seed, random.Random(61))
    assert board is not None and board.is_valid()
    assert 8 <= len(board.piece_map()) <= 22 and len(chess.SquareSet(board.pawns)) >= 2
    assert all(board.piece_at(s) == p for s, p in seed.piece_map().items())


def test_automatic_draw_is_not_an_opponent_mating_opportunity():
    board = chess.Board("7k/r7/5KQ1/8/8/8/8/8 b - - 149 1")
    assert board.is_valid() and not board.is_game_over(claim_draw=False)
    good, bad = safe_moves(board)
    assert not bad and set(good) == {m.uci() for m in board.legal_moves}


def test_tactic_loader_rejects_incomplete_exact_set(tmp_path):
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    row = {
        "fen": board.fen(),
        "key": fingerprint(board).hex(),
        "targets": ["g6g5"],
        "unsafe": [],
        "task": "mate",
        "group": "test",
    }
    file = tmp_path / "train.jsonl"
    file.write_text(json.dumps(row) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partitions": {"train": {"rows": 1, "tasks": {"mate": 1}, "sha256": sha256(file)}},
            }
        )
    )
    with pytest.raises(ValueError, match="incorrect tactical labels"):
        TacticDataset(tmp_path, "train", "mate")[0]
