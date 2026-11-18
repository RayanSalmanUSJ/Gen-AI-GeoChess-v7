"""Versioned state dictionaries, safe loading and atomic writes with checksums."""

from dataclasses import asdict
import os
from pathlib import Path
import random
import numpy as np
import torch
from .data import sha256
from .encoding import ENCODING_VERSION
from .model import ChessPolicy, ModelConfig


def save_checkpoint(path, model, metadata, training=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 1,
        "encoding_version": ENCODING_VERSION,
        "config": asdict(model.config),
        "state": model.state_dict(),
        "metadata": metadata,
    }
    if training is not None:
        payload["training"] = training
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    checksum = path.with_suffix(path.suffix + ".sha256")
    temp_sum = checksum.with_suffix(".tmp")
    temp_sum.write_text(sha256(path) + "\n")
    temp_sum.replace(checksum)


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.exists() or checksum.read_text().strip() != sha256(path):
        raise ValueError("missing or mismatched checkpoint checksum")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["checkpoint_version"] != 1 or payload["encoding_version"] != ENCODING_VERSION:
        raise ValueError("checkpoint format mismatch")
    model = ChessPolicy(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state"], strict=True)
    return model.to(device).eval(), payload


def capture_rng():
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), *state[2:]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0], np.array(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
