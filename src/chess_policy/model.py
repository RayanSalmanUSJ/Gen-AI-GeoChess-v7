"""Learned QKV attention, dense SwiGLU FFNs, dropless top-k MoE and move decoding.

The complete board is a visible conditioning prefix. Each candidate move is a
single output token, represented by its source, destination and promotion.
Outputs are policy logits, with no centipawn, outcome or search-value head.
"""

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .encoding import FEATURE_SCALE
from .geometry import ORBIT_IDS, geometric_tables


@dataclass(frozen=True)
class ModelConfig:
    width: int = 128
    layers: int = 4
    heads: int = 4
    experts: int = 4
    top_k: int = 2
    expansion: int = 3
    moe_every: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.expansion, self.moe_every) < 1:
            raise ValueError("dimensions must be positive")
        if self.width % self.heads or not 1 <= self.top_k <= self.experts:
            raise ValueError("incompatible attention or expert dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.experts > 1 and self.layers < self.moe_every:
            raise ValueError("configuration would have no MoE layers")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width, expansion):
        super().__init__()
        self.up_gate = nn.Linear(width, 2 * width * expansion, bias=False)
        self.down = nn.Linear(width * expansion, width, bias=False)

    def forward(self, x):
        gate, up = self.up_gate(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


class SparseMoE(nn.Module):
    """Local, dropless top-k dispatch. Not expert-parallel across devices."""

    def __init__(self, config):
        super().__init__()
        self.count, self.top_k = config.experts, config.top_k
        self.router = nn.Linear(config.width, config.experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(config.width, config.expansion) for _ in range(config.experts)
        )

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(flat.float(), self.router.weight.float())
            probabilities = logits.softmax(-1)
        weights, indices = probabilities.topk(self.top_k, -1)
        if self.top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            values = expert(flat[rows]) * weights[rows, slots, None].to(flat.dtype)
            output.index_add_(0, rows, values.to(output.dtype))
        assignments = F.one_hot(indices, self.count).float().mean((0, 1))
        balance = self.count * (probabilities.mean(0) * assignments.detach()).sum()
        z_loss = logits.logsumexp(-1).square().mean()
        return output.reshape(shape), {
            "balance": balance,
            "z_loss": z_loss,
            "loads": assignments.detach(),
        }


class BoardAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dropout = config.heads, config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.delta_bias = nn.Embedding(226, config.heads)
        self.relation_bias = nn.Linear(5, config.heads, bias=False)
        self.edge_weights = nn.Parameter(torch.zeros(config.heads, 4))

    def forward(self, x, deltas, relations, edges):
        batch, tokens, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        static = (self.delta_bias(deltas) + self.relation_bias(relations)).permute(2, 0, 1)
        dynamic = torch.einsum("bijn,hn->bhij", edges.to(q.dtype), self.edge_weights.to(q.dtype))
        bias = static.to(q.dtype)[None] + dynamic
        values = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0, is_causal=False
        )
        return self.output(values.transpose(1, 2).reshape(batch, tokens, width))


class Block(nn.Module):
    def __init__(self, config, use_moe):
        super().__init__()
        self.norm_attention = RMSNorm(config.width)
        self.norm_ffn = RMSNorm(config.width)
        self.attention = BoardAttention(config)
        self.ffn = SparseMoE(config) if use_moe else SwiGLU(config.width, config.expansion)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, deltas, relations, edges):
        x = x + self.dropout(self.attention(self.norm_attention(x), deltas, relations, edges))
        if isinstance(self.ffn, SparseMoE):
            update, aux = self.ffn(self.norm_ffn(x))
        else:
            update, aux = self.ffn(self.norm_ffn(x)), None
        return x + self.dropout(update), aux


class MoveDecoder(nn.Module):
    """Geometric queries cross-attend to the board; candidate order has no meaning."""

    def __init__(self, config):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.source = nn.Linear(width, width, bias=False)
        self.destination = nn.Linear(width, width, bias=False)
        self.context = nn.Linear(width, width, bias=False)
        self.promotion = nn.Embedding(5, width)
        self.delta = nn.Embedding(225, width)
        self.norm = RMSNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.kv = nn.Linear(width, 2 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.ffn_norm = RMSNorm(width)
        self.ffn = SwiGLU(width, config.expansion)
        self.readout = nn.Linear(width, 1, bias=False)

    def forward(self, board, legal):
        batch, moves = legal.shape
        width = board.shape[-1]
        ids = legal.clamp_min(0)
        source, destination = ids // 320, (ids // 5) % 64
        rows = torch.arange(batch, device=board.device)[:, None]
        delta = (destination % 8 - source % 8 + 7) * 15 + destination // 8 - source // 8 + 7
        query = (
            self.source(board[rows, source])
            + self.destination(board[rows, destination])
            + self.context(board[:, -1:])
            + self.promotion(ids % 5)
            + self.delta(delta)
        )
        q = (
            self.q(self.norm(query))
            .reshape(batch, moves, self.heads, width // self.heads)
            .transpose(1, 2)
        )
        k, v = (
            self.kv(board)
            .reshape(batch, 65, 2, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        context = (
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, moves, width)
        )
        query = query + self.proj(context)
        query = query + self.ffn(self.ffn_norm(query))
        return self.readout(query).squeeze(-1).float().masked_fill(legal < 0, float("-inf"))


class ChessPolicy(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.piece = nn.Embedding(13, width)
        self.file, self.rank = nn.Embedding(8, width), nn.Embedding(8, width)
        self.orbit = nn.Embedding(10, width)
        self.geometry = nn.Linear(8, width, bias=False)
        self.state_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(
            Block(config, config.experts > 1 and (i + 1) % config.moe_every == 0)
            for i in range(config.layers)
        )
        self.norm = RMSNorm(width)
        self.decoder = MoveDecoder(config)
        deltas, relations = geometric_tables()
        self.register_buffer("deltas", torch.from_numpy(deltas), persistent=False)
        self.register_buffer("relations", torch.from_numpy(relations), persistent=False)
        self.register_buffer("orbit_ids", torch.from_numpy(ORBIT_IDS), persistent=False)
        self.register_buffer("squares", torch.arange(64), persistent=False)
        self.register_buffer("feature_scale", torch.tensor(FEATURE_SCALE), persistent=False)
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

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
        auxiliaries = []
        for block in self.blocks:
            x, aux = block(x, self.deltas, self.relations, edges)
            if aux is not None:
                auxiliaries.append(aux)
        zero = x.sum() * 0
        return {
            "logits": self.decoder(self.norm(x), legal),
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


def model_inputs(batch):
    return {k: batch[k] for k in ("pieces", "geometry", "attacks", "legal")}
