"""v6 tests. Fast ones always run; the Stockfish ones need STOCKFISH=<path to binary>.

  pytest tests/test_v6.py -q
"""

import json
import os
from pathlib import Path
import shutil

import chess
import numpy as np
import pytest
import torch

from chess_policy.encoding import collate, encode
from chess_policy.model import model_inputs
from chess_policy.v6_data import V6Dataset, assemble, build_positions, concat, label
from chess_policy.v6_eval import evaluate_model
from chess_policy.v6_model import V6Config, V6Policy, from_v5, load_v6, save_v6
from chess_policy.v6_train import TrainV6Config, soft_targets, train

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = os.environ.get("STOCKFISH")


def _batch(fens):
    return collate([encode(chess.Board(f)) for f in fens])


def test_v5_checkpoint_loads_unchanged():
    from chess_policy.checkpoint import load_checkpoint

    v5, _ = load_checkpoint(ROOT / "checkpoints" / "policy_v5.pt")
    v6 = from_v5(ROOT / "checkpoints" / "policy_v5.pt")
    b = _batch([chess.STARTING_FEN, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"])
    with torch.no_grad():
        a = v5(**model_inputs(b))["logits"]
        c = v6(**model_inputs(b))["logits"]
    finite = torch.isfinite(a)
    assert torch.equal(finite, torch.isfinite(c))
    assert (a[finite] - c[finite]).abs().max() < 1e-5


def _visible(model, fen, head, square):
    b = _batch([fen])
    mask = model.head_mask(b["pieces"])
    return {chess.square_name(s) for s in range(64) if mask[0, head, square, s] == 0}


def test_geometry_heads():
    hard = V6Policy(V6Config(attention="hard"))
    ray = V6Policy(V6Config(attention="hard_ray"))
    start = chess.STARTING_FEN
    rook_hard = _visible(hard, start, 0, chess.E4)
    rook_ray = _visible(ray, start, 0, chess.E4)
    assert {"e8", "e1", "a4", "h4"} <= rook_hard
    assert "e8" not in rook_ray and "e7" in rook_ray and "e2" in rook_ray and "e1" not in rook_ray
    assert _visible(hard, start, 2, chess.G1) == {"g1", "e2", "f3", "h3"}
    assert _visible(hard, start, 3, chess.A1) == {"a1", "a2", "b1", "b2"}
    bishop = _visible(ray, start, 1, chess.C1)
    assert bishop == {"c1", "b2", "d2"}


@pytest.mark.parametrize("mode", ["soft", "plain", "hard", "hard_ray", "hard_only"])
def test_forward_all_modes(mode):
    model = V6Policy(V6Config(attention=mode, value_head=True))
    b = _batch([chess.STARTING_FEN])
    out = model(**model_inputs(b))
    assert torch.isfinite(out["logits"]).sum() == 20
    assert out["value"].shape == (1,) and out["value"].abs().item() <= 1


def test_soft_targets_sum_to_one_over_legal_moves():
    cp = torch.tensor([[100.0, 0.0, -50.0, -32000.0]])
    legal = torch.tensor([[1, 2, 3, -1]])
    q = soft_targets(cp, legal, 100.0)
    assert torch.allclose(q.sum(-1), torch.ones(1))
    assert q[0, 3] == 0 and q[0, 0] > q[0, 1] > q[0, 2]


def test_save_and_load_roundtrip(tmp_path):
    model = V6Policy(V6Config(attention="hard_ray", value_head=True))
    save_v6(tmp_path / "m.pt", model, {"note": "test"})
    loaded, payload = load_v6(tmp_path / "m.pt")
    assert payload["metadata"]["note"] == "test"
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_positions_reproduce_v5_splits(tmp_path):
    summary = build_positions([ROOT / "data" / "fishtest_train.pgn.gz"], tmp_path / "p.npz",
                              max_games=40)
    assert summary["games"] == 40 and summary["positions"] > 3000
    pos = np.load(tmp_path / "p.npz")
    assert set(np.unique(pos["split"])) <= {0, 1, 2}
    assert len(np.unique(pos["key"])) == len(pos["key"])  # de-duplicated


def test_write_summary_handles_non_ascii(tmp_path, monkeypatch):
    # crashed on Windows (cp1252) at the "Δ" in the paired-difference table
    from chess_policy import v6_experiments as ex

    monkeypatch.setattr(ex, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(ex, "RUNS", tmp_path / "runs")
    (tmp_path / "results").mkdir()
    stat = {"mean": 50.0, "lo": 45.0, "hi": 55.0}
    model = {"cp_loss": stat, "best_rate": {"mean": 0.4}, "blunder_rate": {"mean": 0.1}}
    rep = {"rows": 10, "games": 2, "models": {"v5": model, "A_played_s0": model},
           "paired_vs_v5": {"A_played_s0": {"cp_loss": stat}}}
    (tmp_path / "results" / "t2.json").write_text(json.dumps(rep))
    monkeypatch.setattr("locale.getpreferredencoding", lambda *a, **k: "cp1252")
    ex.write_summary()
    assert "Δ cp lost" in (tmp_path / "results" / "SUMMARY.md").read_text(encoding="utf-8")


def test_label_bad_engine_fails_fast(tmp_path):
    # used to hang forever: the pool initializer failed and was respawned endlessly
    build_positions([ROOT / "data" / "fishtest_train.pgn.gz"], tmp_path / "p.npz", max_games=2)
    with pytest.raises(OSError):
        label(tmp_path / "p.npz", tmp_path / "lab", "no/such/engine.exe", depth=1, workers=1,
              splits=("train",), limit=5)


@pytest.mark.skipif(not STOCKFISH, reason="set STOCKFISH=<path> to run")
def test_label_forward_slash_relative_engine_path(tmp_path, monkeypatch):
    # CLAUDE.md's $env:STOCKFISH = "tools/stockfish/stockfish.exe" failed on Windows
    build_positions([ROOT / "data" / "fishtest_train.pgn.gz"], tmp_path / "p.npz", max_games=2)
    engine = Path(STOCKFISH).resolve()
    monkeypatch.chdir(engine.parent.parent)
    label(tmp_path / "p.npz", tmp_path / "lab", f"{engine.parent.name}/{engine.name}", depth=2,
          workers=1, chunk=5, splits=("train",), limit=5)
    assert (tmp_path / "lab" / "chunk_00000.npz").exists()


@pytest.mark.skipif(not STOCKFISH, reason="set STOCKFISH=<path> to run")
def test_label_resume_relabels_corrupt_chunk(tmp_path):
    # a reboot during labelling left a chunk that was not a valid .npz; assemble then crashed
    from chess_policy.v6_data import load_labels

    build_positions([ROOT / "data" / "fishtest_train.pgn.gz"], tmp_path / "p.npz", max_games=2)
    args = dict(depth=2, workers=1, chunk=5, splits=("train",), limit=10)
    label(tmp_path / "p.npz", tmp_path / "lab", STOCKFISH, **args)
    good = np.load(tmp_path / "lab" / "chunk_00001.npz")["cp"].copy()
    (tmp_path / "lab" / "chunk_00001.npz").write_bytes(b"\0" * 300)
    label(tmp_path / "p.npz", tmp_path / "lab", STOCKFISH, **args)
    rows, cp, lengths, missing, meta = load_labels(tmp_path / "lab")
    assert len(rows) == 10 and len(cp) == lengths.sum()
    assert np.load(tmp_path / "lab" / "chunk_00001.npz")["cp"].shape == good.shape


@pytest.mark.skipif(not STOCKFISH, reason="set STOCKFISH=<path> to run")
def test_label_assemble_train_eval(tmp_path):
    build_positions([ROOT / "data" / "fishtest_train.pgn.gz"], tmp_path / "p.npz", max_games=12)
    label(tmp_path / "p.npz", tmp_path / "lab", STOCKFISH, depth=4, workers=2, chunk=50,
          splits=("train",), limit=150)
    meta = assemble(tmp_path / "p.npz", tmp_path / "ds", "train", tmp_path / "lab", workers=2)
    assert meta["rows"] == 150
    ds = V6Dataset(tmp_path / "ds")
    b = ds.batch(np.arange(10))
    assert (b["cp"][b["legal"] >= 0] > -20000).all()
    concat(tmp_path / "big", tmp_path / "ds", tmp_path / "ds")
    assert len(V6Dataset(tmp_path / "big")) == 300
    cfg = TrainV6Config(train=str(tmp_path / "ds"), dev=str(tmp_path / "ds"),
                        out=str(tmp_path / "run"), policy_target="soft", value_weight=1.0,
                        attention="hard_ray", epochs=1, batch_size=32, warmup_steps=2,
                        device="cpu", log_every=2)
    summary = train(cfg)
    assert json.loads((tmp_path / "run" / "metrics.json").read_text())["best_epoch"] == 0
    model, _ = load_v6(tmp_path / "run" / "best.pt")
    per = evaluate_model(model, ds, torch.device("cpu"), mode="search", k=2)
    assert len(per["cp_loss"]) == 150 and summary["parameters"] > 0
    shutil.rmtree(tmp_path / "big")
