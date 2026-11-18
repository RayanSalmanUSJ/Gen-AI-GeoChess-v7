"""Validated direct inference. No external engine or search is invoked here."""

from pathlib import Path
import math
import threading
import time
import chess
import numpy as np
import torch
from .checkpoint import load_checkpoint
from .encoding import collate, encode, id_move
from .model import model_inputs


class Policy:
    def __init__(self, checkpoint: Path, device="cpu", threads=4):
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.model, self.checkpoint = load_checkpoint(checkpoint, self.device)
        self.lock = threading.Lock()

    def predict(self, position, *, top_k=5, temperature=0.0, seed=None):
        if not math.isfinite(temperature) or temperature < 0 or top_k < 0:
            raise ValueError("temperature and top_k must be nonnegative and finite")
        board = chess.Board(position) if isinstance(position, str) else position.copy(stack=True)
        if not board.is_valid() or board.chess960:
            raise ValueError("expected valid orthodox chess")
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            return {
                "status": "terminal",
                "move": None,
                "moves": [],
                "result": outcome.result(),
                "reason": outcome.termination.name.lower(),
            }
        start = time.perf_counter()
        encoded = encode(board)
        batch = {k: v.to(self.device) for k, v in collate([encoded]).items()}
        with self.lock, torch.inference_mode():
            logits = self.model(**model_inputs(batch))["logits"][0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite policy logits")
            probabilities = logits.softmax(-1).cpu().numpy().astype(float)
            sampling = (
                ((logits.double() - logits.double().max()) / temperature)
                .softmax(-1)
                .cpu()
                .numpy()
                .astype(float)
                if temperature > 0
                else None
            )
        moves = [id_move(int(token)) for token in encoded["legal"]]
        order = sorted(range(len(moves)), key=lambda i: (-probabilities[i], moves[i].uci()))
        selected = (
            int(np.random.default_rng(seed).choice(len(moves), p=sampling / sampling.sum()))
            if sampling is not None
            else order[0]
        )
        count = min(top_k or len(moves), len(moves))
        return {
            "status": "ok",
            "move": moves[selected].uci(),
            "legal_moves": len(moves),
            "moves": [
                {
                    "uci": moves[i].uci(),
                    "san": board.san(moves[i]),
                    "probability": float(probabilities[i]),
                }
                for i in order[:count]
            ],
            "entropy_nats": float(
                -(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum()
            ),
            "milliseconds": 1000 * (time.perf_counter() - start),
            "temperature": temperature,
            "search_nodes": 0,
        }
