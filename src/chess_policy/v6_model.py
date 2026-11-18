"""v6: v5's board transformer + one attention head per piece geometry + a value head.

What changes relative to v5 (model.py is untouched and reused):

* attention modes
    "soft"     v5 exactly: learned geometric biases, every head sees every square
    "plain"    no geometric biases at all (an ordinary transformer over 65 tokens)
    "hard"     head h may only attend along ONE movement geometry
               (h % 4: 0 rook lines, 1 bishop diagonals, 2 knight jumps, 3 king ring),
               plus itself and the state token; v5's learned biases act inside the mask
    "hard_ray" like "hard", but rook and bishop lines stop at the first piece
               (the square of that piece is still visible), as in Prototype 2's ray masks
    "hard_only" the masks of "hard" and nothing else (no learned geometric biases):
               the project statement taken literally, for the ablation
* value head (optional): who is winning, in [-1, 1], from the side to move,
  read from the state token after the last block.

With attention="soft" and value_head=False the parameter names and the forward pass
are identical to v5, so v5 checkpoints load into V6Policy unchanged (tested).
"""

from dataclasses import asdict, dataclass, fields
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import ORBIT_IDS, geometric_tables
from .model import (
    FEATURE_SCALE,
    Block,
    BoardAttention,
    ModelConfig,
    MoveDecoder,
    RMSNorm,
)

ATTENTION_MODES = ("soft", "plain", "hard", "hard_ray", "hard_only")
GEOMETRIES = ("rook", "bishop", "knight", "king")


@dataclass(frozen=True)
class V6Config:
    width: int = 64
    layers: int = 2
    heads: int = 4
    experts: int = 4
    top_k: int = 2
    expansion: int = 3
    moe_every: int = 2
    dropout: float = 0.0
    attention: str = "soft"
    value_head: bool = False

    def __post_init__(self):
        if self.attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}")
        if self.attention.startswith("hard") and self.heads % 4:
            raise ValueError("hard geometry masks need a multiple of 4 heads")
        self.model_config()  # reuse v5's validation

    def model_config(self):
        names = {f.name for f in fields(ModelConfig)}
        return ModelConfig(**{k: v for k, v in asdict(self).items() if k in names})


def geometry_tables():
    """allowed[g, i, j] for the four geometries on 65 tokens, and between[i, j, k]
    = square k lies strictly between squares i and j on a shared line."""
    squares = np.arange(64)
    fx, fy = squares % 8, squares // 8
    dx = fx[None, :] - fx[:, None]
    dy = fy[None, :] - fy[:, None]
    adx, ady = np.abs(dx), np.abs(dy)
    same = (dx == 0) & (dy == 0)
    rook = (dx == 0) | (dy == 0)
    bishop = adx == ady
    knight = (adx * ady == 2) | same
    king = (np.maximum(adx, ady) <= 1)
    allowed = np.ones((4, 65, 65), dtype=bool)
    for g, table in enumerate((rook, bishop, knight, king)):
        allowed[g, :64, :64] = table
    between = np.zeros((64, 64, 64), dtype=bool)
    for i in range(64):
        for j in range(64):
            if i == j or not (rook[i, j] or bishop[i, j]):
                continue
            n = max(adx[i, j], ady[i, j])
            sx, sy = np.sign(dx[i, j]), np.sign(dy[i, j])
            for step in range(1, n):
                between[i, j, (fy[i] + sy * step) * 8 + fx[i] + sx * step] = True
    return allowed, between


class V6BoardAttention(BoardAttention):
    """v5's BoardAttention (same parameters) with an optional additive head mask."""

    def __init__(self, config, mode):
        super().__init__(config)
        self.mode = mode

    def forward(self, x, deltas, relations, edges, head_mask=None):
        batch, tokens, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        if self.mode == "plain":
            bias = None
        elif self.mode == "hard_only":
            bias = head_mask.to(q.dtype)
        else:
            static = (self.delta_bias(deltas) + self.relation_bias(relations)).permute(2, 0, 1)
            dynamic = torch.einsum(
                "bijn,hn->bhij", edges.to(q.dtype), self.edge_weights.to(q.dtype)
            )
            bias = static.to(q.dtype)[None] + dynamic
            if head_mask is not None:
                bias = bias + head_mask.to(q.dtype)
        values = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0, is_causal=False
        )
        return self.output(values.transpose(1, 2).reshape(batch, tokens, width))


class V6Block(Block):
    def __init__(self, config, use_moe, mode):
        super().__init__(config, use_moe)
        self.attention = V6BoardAttention(config, mode)

    def forward(self, x, deltas, relations, edges, head_mask=None):
        x = x + self.dropout(
            self.attention(self.norm_attention(x), deltas, relations, edges, head_mask)
        )
        if hasattr(self.ffn, "router"):
            update, aux = self.ffn(self.norm_ffn(x))
        else:
            update, aux = self.ffn(self.norm_ffn(x)), None
        return x + self.dropout(update), aux


class V6Policy(nn.Module):
    """Same tokens, embeddings, blocks and move decoder as v5's ChessPolicy."""

    def __init__(self, config: V6Config):
        super().__init__()
        self.config = config
        base = config.model_config()
        width = base.width
        self.piece = nn.Embedding(13, width)
        self.file, self.rank = nn.Embedding(8, width), nn.Embedding(8, width)
        self.orbit = nn.Embedding(10, width)
        self.geometry = nn.Linear(8, width, bias=False)
        self.state_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(
            V6Block(base, base.experts > 1 and (i + 1) % base.moe_every == 0, config.attention)
            for i in range(base.layers)
        )
        self.norm = RMSNorm(width)
        self.decoder = MoveDecoder(base)
        if config.value_head:
            self.value = nn.Sequential(
                nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
            )
        deltas, relations = geometric_tables()
        self.register_buffer("deltas", torch.from_numpy(deltas), persistent=False)
        self.register_buffer("relations", torch.from_numpy(relations), persistent=False)
        self.register_buffer("orbit_ids", torch.from_numpy(ORBIT_IDS), persistent=False)
        self.register_buffer("squares", torch.arange(64), persistent=False)
        self.register_buffer("feature_scale", torch.tensor(FEATURE_SCALE), persistent=False)
        allowed, between = geometry_tables()
        head_geometry = torch.arange(config.heads) % 4
        self.register_buffer(
            "head_allowed", torch.from_numpy(allowed)[head_geometry], persistent=False
        )  # (H, 65, 65) bool
        self.register_buffer(
            "between", torch.from_numpy(between).reshape(64 * 64, 64).float(), persistent=False
        )
        self.register_buffer("head_is_line", head_geometry < 2, persistent=False)
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, std=0.02 / math.sqrt(2 * base.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def head_mask(self, pieces):
        """Additive mask (B or 1, H, 65, 65): 0 where a head may look, -inf elsewhere."""
        mode = self.config.attention
        if mode not in ("hard", "hard_ray", "hard_only"):
            return None
        allowed = self.head_allowed[None]  # (1, H, 65, 65)
        if mode == "hard_ray":
            occupied = (pieces > 0).float()  # (B, 64)
            blocked = (occupied @ self.between.T).reshape(-1, 64, 64) > 0  # (B, 64, 64)
            line = self.head_is_line[None, :, None, None]
            open_line = torch.ones(
                (pieces.shape[0], 65, 65), dtype=torch.bool, device=pieces.device
            )
            open_line[:, :64, :64] = ~blocked
            allowed = allowed & (~line | open_line[:, None])
        mask = torch.zeros(allowed.shape, dtype=torch.float32, device=pieces.device)
        return mask.masked_fill(~allowed, float("-inf"))

    def forward(self, pieces, geometry, attacks, legal):
        batch = pieces.shape[0]
        scaled = geometry.float() / self.feature_scale
        x = (
            self.piece(pieces)
            + self.file(self.squares % 8)
            + self.rank(self.squares // 8)
            + self.orbit(self.orbit_ids)
            + self.geometry(scaled)
        )
        state = self.state_token.expand(batch, -1, -1) + self.geometry(scaled.mean(1, keepdim=True))
        x = torch.cat([x, state], 1)
        edges = x.new_zeros((batch, 65, 65, 4))
        edges[:, :64, :64, 0] = attacks
        edges[:, :64, :64, 1] = attacks.transpose(1, 2)
        valid = legal >= 0
        pair = legal.clamp_min(0) // 5
        rows = torch.arange(batch, device=x.device)[:, None].expand_as(pair)
        edges[rows[valid], (pair // 64)[valid], (pair % 64)[valid], 2] = 1
        edges[rows[valid], (pair % 64)[valid], (pair // 64)[valid], 3] = 1
        mask = self.head_mask(pieces)
        auxiliaries = []
        for block in self.blocks:
            x, aux = block(x, self.deltas, self.relations, edges, mask)
            if aux is not None:
                auxiliaries.append(aux)
        board = self.norm(x)
        zero = x.sum() * 0
        out = {
            "logits": self.decoder(board, legal),
            "balance": torch.stack([a["balance"] for a in auxiliaries]).mean()
            if auxiliaries
            else zero,
            "z_loss": torch.stack([a["z_loss"] for a in auxiliaries]).mean()
            if auxiliaries
            else zero,
            "expert_loads": torch.stack([a["loads"] for a in auxiliaries])
            if auxiliaries
            else x.new_empty((0, 0)),
        }
        if self.config.value_head:
            out["value"] = torch.tanh(self.value(board[:, -1]).squeeze(-1).float())
        return out

    def attention_maps(self, pieces, geometry, attacks, legal, layer=0):
        """Per-head attention weights of one layer (for the report's figures)."""
        captured = {}
        block = self.blocks[layer]
        original = block.attention.forward

        def spy(x, deltas, relations, edges, head_mask=None):
            att = block.attention
            b, t, w = x.shape
            q, k, _ = (
                att.qkv(x).reshape(b, t, 3, att.heads, w // att.heads).permute(2, 0, 3, 1, 4)
            )
            scores = q @ k.transpose(-1, -2) / math.sqrt(w // att.heads)
            if att.mode == "hard_only":
                scores = scores + head_mask
            elif att.mode != "plain":
                static = (att.delta_bias(deltas) + att.relation_bias(relations)).permute(2, 0, 1)
                dynamic = torch.einsum("bijn,hn->bhij", edges.to(q.dtype), att.edge_weights)
                scores = scores + static[None] + dynamic
                if head_mask is not None:
                    scores = scores + head_mask
            captured["weights"] = scores.softmax(-1)
            return original(x, deltas, relations, edges, head_mask)

        block.attention.forward = spy
        try:
            with torch.no_grad():
                self(pieces, geometry, attacks, legal)
        finally:
            block.attention.forward = original
        return captured["weights"]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def from_v5(path, device="cpu"):
    """Load a v5 checkpoint (v3 pilot / v4 / v5) as a V6Policy(attention='soft')."""
    from .checkpoint import load_checkpoint

    v5, payload = load_checkpoint(path)
    config = V6Config(**payload["config"], attention="soft", value_head=False)
    model = V6Policy(config)
    model.load_state_dict(v5.state_dict(), strict=True)
    return model.to(device).eval()


def save_v6(path, model, metadata):
    """Weights-only payload + sha256 sidecar, mirroring v5's checkpoint format."""
    from pathlib import Path

    from .data import sha256
    from .encoding import ENCODING_VERSION

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 6,
        "encoding_version": ENCODING_VERSION,
        "config": asdict(model.config),
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": metadata,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256(path) + "\n")


def load_v6(path, device="cpu"):
    from pathlib import Path

    from .data import sha256

    path = Path(path)
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.exists() or checksum.read_text().strip() != sha256(path):
        raise ValueError(f"missing or mismatched checksum for {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") != 6:
        raise ValueError("not a v6 checkpoint (use from_v5 for v3/v4/v5)")
    model = V6Policy(V6Config(**payload["config"]))
    model.load_state_dict(payload["state"], strict=True)
    return model.to(device).eval(), payload


def load_any(path, device="cpu"):
    """v6 checkpoint, or a v5-family checkpoint wrapped as V6Policy."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") == 6:
        return load_v6(path, device)[0]
    return from_v5(path, device)
