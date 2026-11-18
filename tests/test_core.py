from io import StringIO
import chess
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from chess_policy.geometry import BB, FULL, D4, ORBIT_IDS, sq, coords, name, from_name, perft
from chess_policy.encoding import encode, collate, id_move, move_id
from chess_policy.model import ChessPolicy, ModelConfig, model_inputs
from chess_policy.checkpoint import save_checkpoint, load_checkpoint
from chess_policy.inference import Policy
from chess_policy.uci import serve_uci, parse_position

torch.set_num_threads(2)


@pytest.fixture
def model():
    torch.manual_seed(7)
    return ChessPolicy(ModelConfig(width=32, layers=2, heads=4, experts=4))


@pytest.fixture
def batch():
    board = chess.Board()
    examples = []
    for text in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        item = encode(board)
        item["target"] = item["legal"][0]
        examples.append(item)
        board.push_uci(text)
    return collate(examples)


def test_lattice_bitboards_and_dihedral_group():
    assert all(from_name(name(s)) == sq(*coords(s)) == s for s in range(64))
    assert list(BB(9)) == [0, 3] and 3 in BB(9)
    assert (~BB()).bits == FULL and (BB(9) ^ BB(1)).bits == 8
    assert len({tuple(p) for p in D4}) == 8
    assert sum((p == np.arange(64)).sum() for p in D4) == 80
    assert len(set(ORBIT_IDS)) == 10
    for p in D4:
        for q in D4:
            assert any(np.array_equal(p[q], r) for r in D4)
    with pytest.raises(ValueError):
        sq(8, 0)


@pytest.mark.parametrize(
    "fen,depth,count",
    [
        (chess.STARTING_FEN, 4, 197281),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),
    ],
)
def test_perft(fen, depth, count):
    board = chess.Board(fen)
    assert perft(board, depth) == count
    assert board.fen() == fen and not board.move_stack


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ],
)
def test_legality_and_attack_graph(fen):
    board = chess.Board(fen)
    item = encode(board)
    assert {id_move(int(t)) for t in item["legal"]} == set(board.legal_moves)
    assert all(move_id(id_move(int(t))) == t for t in item["legal"])
    attack = np.unpackbits(item["attacks"], axis=-1, bitorder="little")
    for s in range(64):
        assert set(np.flatnonzero(attack[s])) == (
            set(board.attacks(s)) if board.piece_at(s) else set()
        )


def test_special_rules():
    b = chess.Board("4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert chess.D6 in b.attacks(chess.E5)
    assert chess.Move.from_uci("e5d6") not in b.legal_moves
    b = chess.Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1")
    assert {m.promotion for m in b.legal_moves if m.from_square == chess.A7} == {2, 3, 4, 5}


def test_gradient_flow(model, batch):
    o = model(**model_inputs(batch))
    (
        F.cross_entropy(o["logits"], batch["target"]) + 0.01 * o["balance"] + 0.001 * o["z_loss"]
    ).backward()
    assert all(p.grad is not None for p in model.parameters())
    names = dict(model.named_parameters())
    for parameter_name in [
        "blocks.0.attention.qkv.weight",
        "blocks.0.attention.edge_weights",
        "blocks.1.ffn.router.weight",
        "decoder.q.weight",
        "decoder.kv.weight",
    ]:
        assert names[parameter_name].grad.abs().sum() > 0, parameter_name
    for e in range(4):
        assert names[f"blocks.1.ffn.experts.{e}.down.weight"].grad.abs().sum() > 0


def test_bidirectional_attention(model):
    x = torch.randn(1, 65, 32, requires_grad=True)
    output = model.blocks[0].attention(x, model.deltas, model.relations, torch.zeros(1, 65, 65, 4))
    output[0, 0].sum().backward()
    assert x.grad[0, 63].abs().sum() > 0


def test_candidate_permutation_and_padding(model, batch):
    model.eval()
    with torch.inference_mode():
        a = model(**model_inputs(batch))["logits"]
        perm = torch.randperm(batch["legal"].shape[1])
        b = model(**model_inputs(dict(batch, legal=batch["legal"][:, perm])))["logits"]
        assert torch.allclose(a[:, perm], b, atol=1e-6)
        c = model(**model_inputs(dict(batch, legal=F.pad(batch["legal"], (0, 5), value=-1))))[
            "logits"
        ]
        assert torch.allclose(a, c[:, : a.shape[1]], atol=1e-6)
        assert (c.softmax(-1)[:, a.shape[1] :] == 0).all()
    assert "target" not in model_inputs(batch)


def test_overfit(model, batch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = F.cross_entropy(model(**model_inputs(batch))["logits"], batch["target"]).item()
    for _ in range(70):
        optimizer.zero_grad()
        o = model(**model_inputs(batch))
        loss = F.cross_entropy(o["logits"], batch["target"]) + 0.01 * o["balance"]
        loss.backward()
        optimizer.step()
    final = model(**model_inputs(batch))["logits"]
    assert F.cross_entropy(final, batch["target"]).item() < initial * 0.1
    assert (final.argmax(-1) == batch["target"]).all()


def test_checkpoint_and_inference(model, batch, tmp_path):
    path = tmp_path / "policy.pt"
    model.eval()
    save_checkpoint(path, model, {"test": True})
    loaded, _ = load_checkpoint(path)
    with torch.inference_mode():
        assert torch.equal(
            model(**model_inputs(batch))["logits"], loaded(**model_inputs(batch))["logits"]
        )
    policy = Policy(path, threads=2)
    prediction = policy.predict(chess.Board(), top_k=0)
    assert len(prediction["moves"]) == 20
    assert sum(m["probability"] for m in prediction["moves"]) == pytest.approx(1)
    assert (
        chess.Move.from_uci(policy.predict(chess.Board(), temperature=1e-300, seed=7)["move"])
        in chess.Board().legal_moves
    )
    with pytest.raises(ValueError):
        policy.predict(chess.Board(), temperature=float("nan"))
    terminal = policy.predict("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert terminal["status"] == "terminal" and terminal["move"] is None
    source = StringIO("uci\nisready\nposition startpos moves e2e4\ngo searchmoves e7e5\nquit\n")
    sink = StringIO()
    serve_uci(policy, source, sink)
    assert "uciok" in sink.getvalue() and "bestmove e7e5" in sink.getvalue()
    assert parse_position("startpos moves e2e4".split()).turn == chess.BLACK
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path)


def test_http_contract(model, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from chess_policy.api import create_app

    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, {})
    client = TestClient(create_app(path, threads=2))
    assert client.get("/health").json()["status"] == "ready"
    answer = client.post("/predict", json={"fen": chess.STARTING_FEN}).json()
    assert chess.Move.from_uci(answer["move"]) in chess.Board().legal_moves
    assert client.post("/predict", json={"fen": "invalid"}).status_code == 422


def test_bfloat16_router_stability(model, batch):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(**model_inputs(batch))
        loss = F.cross_entropy(output["logits"], batch["target"]) + 0.01 * output["balance"]
    assert output["balance"].dtype == torch.float32
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_medium_configuration_forward_backward(batch):
    model = ChessPolicy(ModelConfig(width=256, layers=6, heads=8, experts=4))
    subset = {k: v[:1] for k, v in batch.items()}
    output = model(**model_inputs(subset))
    loss = F.cross_entropy(output["logits"], subset["target"]) + 0.01 * output["balance"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.blocks[-1].ffn.router.weight.grad.abs().sum() > 0
