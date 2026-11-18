"""
Every method as a chess player with the same interface: move(board) -> chess.Move.

Games always start from the initial position plus opening moves, so the move
history is on board.move_stack (the GPT-style model needs it; others ignore it).

Specs (used by tournament.py):
  random
  board:<checkpoint.pt>:policy | :search      Prototype 2 board transformers (+ self-play)
  gpt:<checkpoint.pt>                          GPT-style move-sequence model
  uci:<binary>:depth=<d>[:name=<label>]        any UCI engine (team C++ engine, Stockfish)
  llm:<ollama model>[:nolist]                  local LLM through Ollama
  v5:<chess_policy_v5 root>:<checkpoint.pt>[:<label>]   team's Chess Policy System (v3/v4/v5)
  v6:<checkpoint.pt>:policy|search[:k=6]       v6 models (GeoChess repo); search needs a value head
  uci:<binary>:elo=<1350..2850>[:name=<label>] Stockfish at limited strength (0.1 s per move)

Paths inside specs must not contain ':' (Windows drive letters): use relative paths.
"""

import os
import random
import sys

import chess
import chess.engine
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "02_value_distillation")
ROOT = os.path.join(HERE, "..", "..")
for p in (P2, ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import encoding as enc  # noqa: E402
from evaluate import search_choices  # noqa: E402
from model import load_checkpoint  # noqa: E402
from shared.move_encoder import move_to_index  # noqa: E402


def _avoid_repeats(board, ordered_moves):
    """First move in preference order that does not repeat a position (else the first)."""
    for m in ordered_moves:
        board.push(m)
        rep = board.is_repetition(2)
        board.pop()
        if not rep:
            return m
    return ordered_moves[0]


class RandomPlayer:
    def __init__(self, seed=0):
        self.rng, self.name = random.Random(seed), "Random moves"

    def move(self, board):
        return self.rng.choice(list(board.legal_moves))


class BoardModelPlayer:
    """Prototype 2 board transformer: policy head, or 2-ply search with the value head."""

    def __init__(self, ckpt, mode="policy", name=None):
        self.model, _ = load_checkpoint(ckpt)
        self.mode = mode
        self.name = name or f"{os.path.basename(os.path.dirname(ckpt))}:{mode}"

    @torch.no_grad()
    def move(self, board):
        legal = list(board.legal_moves)
        acts = [enc.action_index(board, m) for m in legal]
        if self.mode == "search":
            row = np.array([acts], dtype=np.int64)
            j = int(search_choices(self.model, [board.fen()], row, np.array([len(acts)]))[0])
            return legal[j]
        x = torch.from_numpy(enc.encode(board)).unsqueeze(0)
        occ = torch.from_numpy(enc.occupancy(board)).unsqueeze(0)
        logits, _ = self.model(x, occ)
        order = logits[0, torch.tensor(acts)].argsort(descending=True).tolist()
        return _avoid_repeats(board, [legal[j] for j in order])


class GPTPlayer:
    """GPT-style model: reads the move history from the initial position."""

    def __init__(self, ckpt, name=None):
        from gpt_movelm import BOS, load_gpt
        self.model, _ = load_gpt(ckpt)
        self.bos = BOS
        self.name = name or f"gpt:{os.path.basename(os.path.dirname(ckpt))}"

    @torch.no_grad()
    def move(self, board):
        from gpt_movelm import MAX_LEN
        hist = [move_to_index(m) for m in board.move_stack][-(MAX_LEN - 1):]
        toks = torch.tensor([[self.bos] + hist])
        logits, _ = self.model(toks)
        legal = list(board.legal_moves)
        ids = torch.tensor([move_to_index(m) for m in legal])
        order = logits[0, -1, ids].argsort(descending=True).tolist()
        return _avoid_repeats(board, [legal[j] for j in order])


class V5Player:
    """Team's Chess Policy System checkpoints (v3 pilot / v4 / v5), loaded unchanged."""

    def __init__(self, root, ckpt, name=None):
        src = os.path.join(root, "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from chess_policy.checkpoint import load_checkpoint as load_v5
        from chess_policy.encoding import collate, encode, id_move
        from chess_policy.model import model_inputs
        self._collate, self._encode, self._id_move, self._inputs = collate, encode, id_move, model_inputs
        self.model, _ = load_v5(os.path.join(root, "checkpoints", ckpt))
        self.name = name or f"v5:{ckpt}"

    @torch.no_grad()
    def move(self, board):
        e = self._encode(board, validate=False)
        logits = self.model(**self._inputs(self._collate([e])))["logits"][0]
        moves = [self._id_move(int(t)) for t in e["legal"]]
        order = logits[: len(moves)].argsort(descending=True).tolist()
        return _avoid_repeats(board, [moves[j] for j in order])


class V6Player:
    """v6 checkpoint (or a v5-family one) from the GeoChess repo: policy or 2-ply search."""

    def __init__(self, ckpt, mode="policy", k=6, name=None):
        src = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
        if src not in sys.path:
            sys.path.insert(0, src)
        from chess_policy.encoding import collate, encode, move_id
        from chess_policy.v6_eval import search_choices
        from chess_policy.v6_model import load_any
        self._collate, self._encode, self._move_id = collate, encode, move_id
        self._search = search_choices
        self.model = load_any(ckpt)
        self.mode, self.k = mode, k
        if mode == "search" and not self.model.config.value_head:
            raise ValueError("search needs a v6 model with a value head")
        self.name = name or f"v6:{os.path.basename(os.path.dirname(ckpt))}:{mode}"

    @torch.no_grad()
    def move(self, board):
        legal = sorted(board.legal_moves, key=self._move_id)
        b = self._collate([self._encode(board, validate=False)])
        logits = self.model(b["pieces"], b["geometry"], b["attacks"], b["legal"])["logits"][0]
        order = logits[: len(legal)].argsort(descending=True).tolist()
        if self.mode == "search":
            j = int(self._search(self.model, [board.fen()], [order], torch.device("cpu"), self.k)[0])
            order = [j] + [x for x in order if x != j]
        return _avoid_repeats(board, [legal[j] for j in order])


class UCIPlayer:
    def __init__(self, path, depth, name=None, elo=None):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            self.engine.configure({"Threads": 1, "Hash": 16})
        except chess.engine.EngineError:
            pass  # engines without these options
        if elo:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
            self.limit = chess.engine.Limit(time=0.1)
        else:
            self.limit = chess.engine.Limit(depth=depth)
        self.name = name or (f"{os.path.basename(path)} elo{elo}" if elo
                             else f"{os.path.basename(path)} d{depth}")

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


def make_player(spec):
    kind, _, rest = spec.partition(":")
    if kind == "random":
        return RandomPlayer()
    if kind == "board":
        ckpt, mode, *name = rest.split(":")
        return BoardModelPlayer(ckpt, mode, name[0] if name else None)
    if kind == "gpt":
        ckpt, *name = rest.split(":")
        return GPTPlayer(ckpt, name[0] if name else None)
    if kind == "uci":
        parts = rest.split(":")
        opts = dict(p.split("=", 1) for p in parts[1:])
        return UCIPlayer(parts[0], int(opts.get("depth", 4)), opts.get("name"), opts.get("elo"))
    if kind == "v6":
        ckpt, mode, *rest = rest.split(":")
        opts = dict(p.split("=", 1) for p in rest if "=" in p)
        return V6Player(ckpt, mode, int(opts.get("k", 6)), opts.get("name"))
    if kind == "v5":
        root, ckpt, *name = rest.split(":")
        return V5Player(root, ckpt, name[0] if name else None)
    if kind == "llm":
        from llm_player import LLMPlayer
        model, *flags = rest.split(":")
        return LLMPlayer(model, legal_list="nolist" not in flags)
    raise ValueError(spec)
