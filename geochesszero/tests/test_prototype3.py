"""Tests for the method-comparison prototype (03)."""

import os
import subprocess
import sys

import chess
import pytest

HERE = os.path.dirname(__file__)
P3 = os.path.abspath(os.path.join(HERE, "..", "prototypes", "03_method_comparison"))
P2 = os.path.abspath(os.path.join(HERE, "..", "prototypes", "02_value_distillation"))
sys.path.insert(0, P3)
sys.path.insert(0, HERE)

import llm_player  # noqa: E402
import mock_ollama  # noqa: E402
from gpt_movelm import MoveGPT, move_between, BOS  # noqa: E402


def test_parse_move_handles_uci_san_and_thinking():
    b = chess.Board()
    assert llm_player.parse_move(b, "<think>e2e5?</think> I play e2e4.") == chess.Move.from_uci("e2e4")
    assert llm_player.parse_move(b, "Nf3 is best") == chess.Move.from_uci("g1f3")
    assert llm_player.parse_move(b, "a1a8") is None


def test_llm_player_retries_and_always_returns_a_legal_move():
    srv, host = mock_ollama.start()
    try:
        p = llm_player.LLMPlayer("mock", host=host, retries=2)
        b = chess.Board()
        for _ in range(30):
            mv = p.move(b)
            assert mv in b.legal_moves
            b.push(mv)
            if b.is_game_over():
                break
        s = p.stats
        assert s["moves"] >= 1 and s["first_try_legal"] <= s["legal_after_retries"] <= s["moves"]
    finally:
        srv.shutdown()


def test_move_between_and_gpt_shapes():
    a = chess.Board()
    b = a.copy()
    b.push_san("e4")
    assert move_between(a.fen(), b.fen()) == chess.Move.from_uci("e2e4")
    import torch
    m = MoveGPT(d_model=32, n_heads=4, n_layers=1)
    logits, v = m(torch.full((2, 5), BOS))
    assert logits.shape == (2, 5, 4272) and v.shape == (2, 5)


@pytest.mark.skipif(not os.path.exists(os.path.join(P2, "data", "test.npz")),
                    reason="needs Prototype 2's generated test data")
def test_llm_bench_end_to_end_with_mock(tmp_path):
    srv, host = mock_ollama.start()
    try:
        out = tmp_path / "llm.json"
        r = subprocess.run([sys.executable, os.path.join(P3, "llm_player.py"), "bench", "--model", "mock",
                            "--host", host, "--every", "50", "--limit", "8", "--out", str(out)],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr
        import json
        d = json.load(open(out))
        assert d["positions"] == 8 and 0 <= d["first_try_legal_rate"] <= 1
    finally:
        srv.shutdown()
