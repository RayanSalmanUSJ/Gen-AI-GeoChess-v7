import json
import chess
import pytest
import torch
from torch.nn import functional as F

from chess_policy.encoding import encode, move_id, fingerprint
from chess_policy.mate_curriculum import mate_moves, orbit_group, MateDataset
from chess_policy.corrections import CorrectionDataset
from chess_policy.data import sha256
from chess_policy.refinement import collate_targets, set_nll


def test_set_loss_equals_cross_entropy_for_single_label():
    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 1.0, 0.0]], requires_grad=True)
    target = torch.tensor([2, 0])
    accepted = F.one_hot(target, 3).bool()
    assert torch.allclose(
        set_nll(logits, accepted), F.cross_entropy(logits, target, reduction="none")
    )


def test_multiple_mates_have_joint_probability_and_padding_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0, float("-inf")]], requires_grad=True)
    accepted = torch.tensor([[False, True, True, False]])
    loss = set_nll(logits, accepted)
    expected = -torch.log(logits.softmax(-1)[0, 1:3].sum())
    assert torch.allclose(loss[0], expected)
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 3] == 0


def test_mate_labels_and_safe_geometric_orbits():
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    assert board.is_valid()
    before = board.fen()
    mates = mate_moves(board)
    assert "g6g7" in mates and before == board.fen()
    assert orbit_group(board) == orbit_group(board.transform(chess.flip_horizontal))
    assert orbit_group(board) == orbit_group(board.mirror())
    item = encode(board)
    item.update(
        target=move_id(chess.Move.from_uci(mates[0])),
        acceptable=[move_id(chess.Move.from_uci(m)) for m in mates],
    )
    batch = collate_targets([item])
    assert batch["accepted"].sum() == len(mates)


def test_correction_data_rejects_wrong_fingerprint(tmp_path):
    row = {"fen": chess.STARTING_FEN, "target": "e2e4", "key": "bad", "group": "a"}
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "partitions": {"train": {"sha256": sha256(path), "rows": 1}}}
        )
    )
    data = CorrectionDataset(tmp_path, "train")
    with pytest.raises(ValueError, match="fingerprint"):
        data[0]


def test_mate_dataset_rejects_incorrect_target(tmp_path):
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    row = {
        "fen": board.fen(),
        "targets": ["g6g5"],
        "key": fingerprint(board).hex(),
        "group": orbit_group(board),
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "partitions": {"train": {"sha256": sha256(path), "rows": 1}}}
        )
    )
    with pytest.raises(ValueError, match="mate labels"):
        MateDataset(tmp_path, "train")[0]
