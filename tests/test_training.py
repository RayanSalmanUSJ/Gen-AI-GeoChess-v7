import json
import random
from dataclasses import replace
import chess
import chess.pgn
import numpy as np
import pytest
import torch
from chess_policy.data import build_corpus, read_manifest, ShardDataset, EpochSampler
from chess_policy.encoding import collate
from chess_policy.model import ModelConfig
from chess_policy.checkpoint import load_checkpoint
from chess_policy.training import TrainConfig, train


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(122)
    path = tmp_path / "fixture.pgn"
    with path.open("w") as stream:
        for _ in range(60):
            board = chess.Board()
            for _ in range(12):
                board.push(rng.choice(list(board.legal_moves)))
            game = chess.pgn.Game()
            game.setup(board)
            node = game
            for _ in range(4):
                if board.is_game_over():
                    break
                move = rng.choice(list(board.legal_moves))
                board.push(move)
                node = node.add_variation(move)
            stream.write(str(game) + "\n\n")
    root = tmp_path / "corpus"
    build_corpus([path], root, shard_size=32)
    return root


def test_data_integrity_and_sampling(corpus):
    meta = read_manifest(corpus, verify=True)
    keys = {s: set() for s in ["train", "dev", "test"]}
    groups = {s: set() for s in keys}
    for shard in meta["shards"]:
        local = np.load(corpus / shard["files"]["keys"]["path"])
        assert len(set(bytes(k) for k in local)) == len(local)
        keys[shard["split"]].update(bytes(k) for k in local)
    for game in meta["games"]:
        groups[game["split"]].add(game["group"])
    assert not keys["train"] & keys["dev"] and not keys["train"] & keys["test"]
    assert not groups["train"] & groups["dev"] and not groups["train"] & groups["test"]
    dataset = ShardDataset(corpus, "train")
    a = list(EpochSampler(dataset, 42, 0))
    assert a == list(EpochSampler(dataset, 42, 0))
    assert a != list(EpochSampler(dataset, 42, 1))
    assert sorted(a) == list(range(len(dataset)))
    batch = collate([dataset[i] for i in a[:8]])
    assert batch["legal"].gather(1, batch["target"][:, None]).min() >= 0
    target = corpus / meta["shards"][0]["files"]["pieces"]["path"]
    with target.open("ab") as stream:
        stream.write(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(corpus, verify=True)


def test_epoch_resume_matches_uninterrupted(corpus, tmp_path, monkeypatch):
    import chess_policy.training as training

    model = ModelConfig(width=16, layers=2, heads=4, experts=2, dropout=0.15)
    config = TrainConfig(
        epochs=2,
        batch_size=7,
        accumulation=3,
        max_train_rows=37,
        max_dev_rows=12,
        warmup_steps=1,
        threads=1,
    )
    train(corpus, tmp_path / "full", model, config)
    original = training.save_checkpoint

    def interrupted(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path.name == "last.pt":
            raise InterruptedError("simulated interruption after an atomic checkpoint")

    monkeypatch.setattr(training, "save_checkpoint", interrupted)
    with pytest.raises(InterruptedError):
        train(corpus, tmp_path / "resume", model, config)
    monkeypatch.setattr(training, "save_checkpoint", original)
    train(corpus, tmp_path / "resume", model, config, tmp_path / "resume" / "last.pt")
    a, _ = load_checkpoint(tmp_path / "full" / "last.pt")
    b, _ = load_checkpoint(tmp_path / "resume" / "last.pt")
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    with pytest.raises(ValueError, match="identical"):
        train(
            corpus,
            tmp_path / "resume",
            model,
            replace(config, epochs=3),
            tmp_path / "resume" / "last.pt",
        )


def test_invalid_pgn_does_not_commit(tmp_path):
    source = tmp_path / "broken.pgn"
    source.write_text('[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]\n[SetUp "1"]\n\n*\n')
    with pytest.raises(ValueError):
        build_corpus([source], tmp_path / "out")
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_manifest_rejects_path_escape(corpus):
    path = corpus / "manifest.json"
    obj = json.loads(path.read_text())
    obj["shards"][0]["files"]["pieces"]["path"] = "../../outside.npy"
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="escapes"):
        read_manifest(corpus)
