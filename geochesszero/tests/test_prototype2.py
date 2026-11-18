"""Tests for Prototype 2 (value distillation + ray-aware geometry attention)."""

import os
import sys

import chess
import numpy as np
import torch

HERE = os.path.dirname(__file__)
P2 = os.path.abspath(os.path.join(HERE, "..", "prototypes", "02_value_distillation"))
sys.path.insert(0, P2)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import encoding as enc  # noqa: E402
from model import GeoChessZeroV2, build_geometry  # noqa: E402
from evaluate import move_metrics  # noqa: E402


def sq(name):
    return chess.parse_square(name)


def test_static_masks_match_prototype1_geometry():
    g = build_geometry()
    e4 = sq("e4")
    rook = {chess.square_name(t) for t in range(64) if g["rook"][e4, t] and t != e4}
    knight = {chess.square_name(t) for t in range(64) if g["knight"][e4, t] and t != e4}
    assert len(rook) == 14 and "e8" in rook and "a4" in rook
    assert knight == {"d2", "f2", "c3", "g3", "c5", "g5", "d6", "f6"}
    assert int(g["king"][e4].sum()) == 9  # 8 neighbours + itself


def test_between_table():
    g = build_geometry()["between"]
    assert {chess.square_name(s) for s in range(64) if g[sq("a1"), sq("a4"), s]} == {"a2", "a3"}
    assert {chess.square_name(s) for s in range(64) if g[sq("c1"), sq("f4"), s]} == {"d2", "e3"}
    assert not g[sq("a1"), sq("a2")].any()
    assert not g[sq("e4"), sq("f6")].any()  # knight jump: nothing between


def test_ray_masks_stop_at_first_blocker():
    m = GeoChessZeroV2("ray", d_model=32, n_heads=4, n_layers=1)
    occ = torch.zeros(1, 64)
    occ[0, sq("a2")] = 1  # blocker directly above a1
    occ[0, sq("b2")] = 1  # blocker on a1's diagonal
    rook, bishop, knight, king = m.relation_masks(occ)[0]
    a1 = sq("a1")
    assert rook[a1, sq("a2")] and not rook[a1, sq("a3")]   # blocker itself is included
    assert rook[a1, sq("h1")]                                # open rank still visible
    assert bishop[a1, sq("b2")] and not bishop[a1, sq("c3")]
    assert knight[a1, sq("b3")] and knight[a1, sq("c2")]     # knights jump over blockers
    assert all(mask[a1, a1] for mask in (rook, bishop, knight, king))


def test_orientation_is_an_exact_colour_swap():
    b = chess.Board()
    b.push_san("e4")  # Black to move
    x = enc.encode(b)
    # After mirroring, Black's pawns are "our" pawns on rank 2 and White's e-pawn is on e5.
    assert x[sq("e2"), 0] == 1 and x[sq("e5"), 6] == 1
    mv = chess.Move.from_uci("e7e5")
    assert enc.orient_move(enc.orient_move(mv, chess.BLACK), chess.BLACK) == mv
    actions = [a for _, a in enc.legal_actions(b)]
    assert len(actions) == len(set(actions)) == 20


def test_promotion_actions_are_unique():
    b = chess.Board("8/P6k/8/8/8/8/6Kp/8 b - - 0 1")
    actions = [a for _, a in enc.legal_actions(b)]
    assert len(actions) == len(set(actions))


def test_soft_target_prefers_best_move():
    p = enc.soft_policy_target(np.array([30, 10, -400, 10000]))
    assert abs(p.sum() - 1) < 1e-6 and p.argmax() == 3
    p2 = enc.soft_policy_target(np.array([50, 40, -300]))
    assert p2[0] > p2[1] > p2[2]


def test_model_variants_forward_and_backward():
    x, occ = torch.zeros(2, 64, 17), torch.zeros(2, 64)
    occ[:, :16] = 1
    for variant in ("standard", "static", "ray"):
        m = GeoChessZeroV2(variant, d_model=32, n_heads=4, n_layers=1)
        logits, v = m(x, occ)
        assert logits.shape == (2, 4272) and v.shape == (2,)
        (logits.logsumexp(1).sum() + v.sum()).backward()
        assert torch.isfinite(m.embed.weight.grad).all()


def test_move_metrics():
    cp = np.array([100, 40, -300, 0], dtype=np.int16)
    loss, es = move_metrics(cp, 3, 0)
    assert loss == 0 and es == 0
    loss, es = move_metrics(cp, 3, 2)
    assert loss == 400 and es > 0
