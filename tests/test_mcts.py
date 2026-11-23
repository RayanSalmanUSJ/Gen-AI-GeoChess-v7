"""Unit tests for batched PUCT MCTS."""

import chess
import pytest
import torch

from chess_policy.mcts import MCTS, MCTSConfig, _terminal_value
from chess_policy.v6_model import V6Config, V6Policy


@pytest.fixture
def v6_model():
    cfg = V6Config(width=64, layers=2, heads=4, experts=4, attention="hard_ray", value_head=True)
    model = V6Policy(cfg)
    model.eval()
    return model


def test_terminal_detection():
    # Scholar's mate final position (Black is checkmated)
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert board.is_checkmate()
    assert _terminal_value(board) == -1.0

    # Stalemate position
    board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    assert board.is_stalemate()
    assert _terminal_value(board) == 0.0

    # Ongoing position
    board = chess.Board()
    assert _terminal_value(board) is None


def test_mcts_forced_move(v6_model):
    # Position with only one legal move: Black king on a8, White rook on b2 attacks b-file
    board = chess.Board("k7/8/8/8/8/8/1R6/2K5 b - - 0 1")
    assert len(list(board.legal_moves)) == 1
    mcts = MCTS(v6_model, torch.device("cpu"), MCTSConfig(simulations=32))
    res = mcts.search(board)
    assert res["status"] == "forced"
    assert res["best_move"] == chess.Move.from_uci("a8a7")


def test_mcts_terminal_search(v6_model):
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    mcts = MCTS(v6_model, torch.device("cpu"))
    res = mcts.search(board)
    assert res["status"] == "terminal"
    assert res["best_move"] is None


def test_mcts_simulation_counts_and_visits(v6_model):
    board = chess.Board()
    cfg = MCTSConfig(simulations=32, batch_size=8, c_puct=2.0)
    mcts = MCTS(v6_model, torch.device("cpu"), cfg)
    res = mcts.search(board)
    assert res["status"] == "ok"
    assert res["best_move"] in board.legal_moves
    total_visits = sum(res["visits"].values())
    # Total visits at root's children equals simulations - 1 (since root itself is 1 visit)
    assert total_visits == 31
    # Check policy_target sums to 1.0
    assert sum(res["policy_target"].values()) == pytest.approx(1.0)


def test_mcts_batched_leaf_evaluation(v6_model):
    board = chess.Board()
    # Test batch_size=1 (sequential) vs batch_size=16 (batched)
    mcts_seq = MCTS(v6_model, torch.device("cpu"), MCTSConfig(simulations=16, batch_size=1))
    res_seq = mcts_seq.search(board)
    assert res_seq["best_move"] in board.legal_moves

    mcts_batch = MCTS(v6_model, torch.device("cpu"), MCTSConfig(simulations=16, batch_size=16))
    res_batch = mcts_batch.search(board)
    assert res_batch["best_move"] in board.legal_moves


def test_mcts_temperature_scaling(v6_model):
    board = chess.Board()
    # Argmax (T=0)
    mcts_greedy = MCTS(v6_model, torch.device("cpu"), MCTSConfig(simulations=32, temperature=0.0))
    res_greedy = mcts_greedy.search(board)
    assert res_greedy["best_move"] in board.legal_moves
    assert res_greedy["visits"][res_greedy["best_move"].uci()] == max(res_greedy["visits"].values())

    # Soft distribution (T=1.0)
    mcts_soft = MCTS(v6_model, torch.device("cpu"), MCTSConfig(simulations=32, temperature=1.0))
    res_soft = mcts_soft.search(board)
    assert res_soft["best_move"] in board.legal_moves
    assert sum(res_soft["policy_target"].values()) == pytest.approx(1.0)
