"""Frozen earlier model on exactly the same held-out positions as this release."""

import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from chess_policy.data import ShardDataset, atomic_json, sha256
from chess_policy.encoding import collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from previous_move_gpt import load  # noqa: E402


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("holdout"))
    p.add_argument("--checkpoint", type=Path, default=Path("baselines/previous_policy.pt"))
    p.add_argument("--out", type=Path, default=Path("results/previous_holdout.json"))
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    model, metadata = load(a.checkpoint)
    dataset = ShardDataset(a.data, "test", verify=True)
    total = right = top3 = 0
    nll = 0.0
    started = time.monotonic()
    for batch in DataLoader(dataset, batch_size=64, collate_fn=collate):
        pieces = batch["pieces"]
        geometry = batch["geometry"].float()
        size = len(pieces)
        ids = torch.cat([pieces, torch.full((size, 1), 13, dtype=torch.long)], dim=1)
        features = torch.zeros(size, 65, 8)
        squares = torch.arange(64)
        features[:, :64, 0] = squares % 8 / 7
        features[:, :64, 1] = squares // 8 / 7
        features[:, :64, 2:4] = geometry[:, :, :2] / 8
        features[:, :64, 4:6] = geometry[:, :, 4:6]
        features[:, :64, 6] = geometry[:, :, 6]
        features[:, :64, 7] = geometry[:, :, 7] / 100
        features[:, 64, 6:8] = features[:, 0, 6:8]
        logits = model(ids, features).gather(1, batch["legal"].clamp_min(0))
        logits = logits.masked_fill(batch["legal"] < 0, float("-inf"))
        labels = batch["target"]
        nll += F.cross_entropy(logits, labels, reduction="sum").item()
        top = logits.topk(min(3, logits.shape[1]), dim=-1).indices
        right += (top[:, 0] == labels).sum().item()
        top3 += (top == labels[:, None]).any(-1).sum().item()
        total += size
    result = {
        "positions": total,
        "top1": right / total,
        "top3": top3 / total,
        "nll": nll / total,
        "checkpoint_sha256": sha256(a.checkpoint),
        "config": metadata["config"],
        "seconds": time.monotonic() - started,
        "limitations": "Same heldout positions, but architecture and training corpus both differ; not an isolated architecture ablation. Historical training overlap is not established.",
    }
    atomic_json(a.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
