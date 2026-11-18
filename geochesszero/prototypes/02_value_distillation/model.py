"""
GeoChessZero v2: geometry-aware transformer with policy AND value heads.

Board = 64 square tokens = nodes of a multi-relational graph. Each of the
first four attention heads is restricted to one relation; the other four are
unrestricted (same split as Prototype 1).

Attention variants (the only thing that differs between compared models):
  standard : all 8 heads unrestricted (Prototype 1 baseline)
  static   : Prototype 1's empty-board masks: rook lines, bishop
             diagonals, knight jumps, king neighbourhood
  ray      : NEW. Rook and bishop heads follow occupancy-aware rays: a ray
             stops at the first occupied square (that square is included, as
             an attack/defence target). Knight and king masks stay static,
             since those pieces jump. Prototype 1's README listed "masks do
             not model piece blocking" as a limitation; this variant tests
             fixing it.

Heads:
  policy : Prototype 1's origin x destination bilinear head + promotion head
           over the same 4,272-action vocabulary
  value  : scalar in [-1, 1], expected score for the side to move. This is
           the "single number that says who is better" from the notes.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.move_encoder import NUM_ACTIONS, PROMOTION_PATHS  # noqa: E402

VARIANTS = ("standard", "static", "ray")


def _coords(s):
    return s % 8, s // 8


def build_geometry():
    """Static relation masks and the 'squares strictly between' table."""
    rook = torch.zeros(64, 64, dtype=torch.bool)
    bishop = torch.zeros(64, 64, dtype=torch.bool)
    knight = torch.zeros(64, 64, dtype=torch.bool)
    king = torch.zeros(64, 64, dtype=torch.bool)
    between = torch.zeros(64, 64, 64, dtype=torch.bool)
    for s in range(64):
        sx, sy = _coords(s)
        for t in range(64):
            tx, ty = _coords(t)
            dx, dy = tx - sx, ty - sy
            adx, ady = abs(dx), abs(dy)
            if s == t:
                continue
            rook[s, t] = sx == tx or sy == ty
            bishop[s, t] = adx == ady
            knight[s, t] = (adx, ady) in ((1, 2), (2, 1))
            king[s, t] = max(adx, ady) == 1
            if rook[s, t] or bishop[s, t]:
                stepx = (dx > 0) - (dx < 0)
                stepy = (dy > 0) - (dy < 0)
                x, y = sx + stepx, sy + stepy
                while (x, y) != (tx, ty):
                    between[s, t, x + 8 * y] = True
                    x, y = x + stepx, y + stepy
    eye = torch.eye(64, dtype=torch.bool)
    return {
        "rook": rook | eye,
        "bishop": bishop | eye,
        "knight": knight | eye,
        "king": king | eye,
        "between": between,
    }


class GeometryAttention(nn.Module):
    def __init__(self, d_model, n_heads, variant, geometry):
        super().__init__()
        assert variant in VARIANTS
        assert d_model % n_heads == 0 and n_heads >= 4
        self.h, self.dh, self.variant = n_heads, d_model // n_heads, variant
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        static = torch.stack([geometry["rook"], geometry["bishop"],
                              geometry["knight"], geometry["king"]])
        self.register_buffer("static_masks", static, persistent=False)

    def forward(self, x, masks=None, return_attention=False):
        B, T, D = x.shape
        q, k, v = (self.qkv(x).view(B, T, 3, self.h, self.dh)
                   .permute(2, 0, 3, 1, 4))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if self.variant != "standard":
            allowed = masks if masks is not None else self.static_masks.unsqueeze(0)
            neg = torch.finfo(scores.dtype).min
            geo = scores[:, :4].masked_fill(~allowed, neg)
            scores = torch.cat([geo, scores[:, 4:]], dim=1)
        att = scores.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        y = self.out(y)
        return (y, att) if return_attention else y


class Block(nn.Module):
    def __init__(self, d_model, n_heads, variant, geometry, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.att = GeometryAttention(d_model, n_heads, variant, geometry)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, masks):
        x = x + self.att(self.n1(x), masks)
        return x + self.ff(self.n2(x))


class GeoChessZeroV2(nn.Module):
    def __init__(self, variant="ray", d_model=128, n_heads=8, n_layers=4,
                 in_channels=17, dropout=0.1):
        super().__init__()
        self.variant = variant
        self.config = dict(variant=variant, d_model=d_model, n_heads=n_heads,
                           n_layers=n_layers, in_channels=in_channels, dropout=dropout)
        g = build_geometry()
        self.register_buffer("rook_line", g["rook"], persistent=False)
        self.register_buffer("bishop_line", g["bishop"], persistent=False)
        self.register_buffer("knight_mask", g["knight"], persistent=False)
        self.register_buffer("king_mask", g["king"], persistent=False)
        self.register_buffer("between", g["between"].reshape(64 * 64, 64).float(),
                             persistent=False)

        self.embed = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, variant, g, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        # Policy head (same structure as Prototype 1)
        self.from_proj = nn.Linear(d_model, 64)
        self.to_proj = nn.Linear(d_model, 64)
        self.promo = nn.Linear(d_model, 4)
        self.register_buffer("promo_from", torch.tensor([p[0] for p in PROMOTION_PATHS]),
                             persistent=False)
        self.register_buffer("promo_to", torch.tensor([p[1] for p in PROMOTION_PATHS]),
                             persistent=False)
        # Value head
        self.value = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Linear(d_model, 1))

    # -- geometry ----------------------------------------------------------
    def relation_masks(self, occ):
        """(B, 4, 64, 64) allowed-attention masks for the four geometric heads."""
        B = occ.shape[0]
        if self.variant == "static":
            return None  # attention falls back to the shared static masks
        blocked = (self.between @ occ.t()).t().reshape(B, 64, 64) > 0.5
        rook = self.rook_line.unsqueeze(0) & ~blocked
        bishop = self.bishop_line.unsqueeze(0) & ~blocked
        knight = self.knight_mask.unsqueeze(0).expand(B, -1, -1)
        king = self.king_mask.unsqueeze(0).expand(B, -1, -1)
        return torch.stack([rook, bishop, knight, king], dim=1)

    # -- forward -----------------------------------------------------------
    def forward(self, x, occ):
        masks = self.relation_masks(occ) if self.variant == "ray" else None
        h = self.embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h, masks)
        h = self.norm(h)
        moves = (self.from_proj(h) @ self.to_proj(h).transpose(1, 2)).flatten(1)
        promo = self.promo(h[:, self.promo_from] + h[:, self.promo_to]).flatten(1)
        logits = torch.cat([moves, promo], dim=1)
        value = torch.tanh(self.value(h.mean(1))).squeeze(-1)
        return logits, value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path, extra=None):
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = GeoChessZeroV2(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("extra", {})


if __name__ == "__main__":
    import time
    for variant in VARIANTS:
        m = GeoChessZeroV2(variant)
        x = torch.zeros(256, 64, 17)
        occ = torch.zeros(256, 64)
        occ[:, :16] = 1
        occ[:, 48:] = 1
        t = time.time()
        logits, v = m(x, occ)
        loss = logits.logsumexp(1).mean() + v.pow(2).mean()
        loss.backward()
        print(f"{variant:9s} params={count_parameters(m):,} logits={tuple(logits.shape)} "
              f"fwd+bwd(256)={time.time() - t:.2f}s")
    assert logits.shape[1] == NUM_ACTIONS
