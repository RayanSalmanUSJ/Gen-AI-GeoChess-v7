"""Held-out imitation metrics with opening-group bootstrap, not an Elo estimate."""

from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .checkpoint import load_checkpoint
from .data import ShardDataset, atomic_json, sha256
from .encoding import collate
from .model import model_inputs


def collate_groups(examples):
    return collate(examples), [x["group"] for x in examples]


@torch.inference_mode()
def evaluate(checkpoint, data, output, *, split="test", device="cpu", threads=4, batch_size=64):
    torch.set_num_threads(threads)
    model, payload = load_checkpoint(checkpoint, device)
    dataset = ShardDataset(data, split, verify=True)
    groups = defaultdict(lambda: np.zeros(7, dtype=np.float64))
    calibration = np.zeros((10, 3), dtype=np.float64)
    latencies = []
    started = time.monotonic()
    for batch, ids in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_groups):
        batch = {k: v.to(device) for k, v in batch.items()}
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        logits = model(**model_inputs(batch))["logits"]
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - begin)
        target = batch["target"]
        losses = F.cross_entropy(logits, target, reduction="none")
        probabilities = logits.softmax(-1)
        confidence, predicted = probabilities.max(-1)
        top = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
        correct = predicted == target
        count = (batch["legal"] >= 0).sum(-1).float()
        brier = (
            probabilities.square().sum(-1)
            - 2 * probabilities.gather(1, target[:, None]).squeeze(1)
            + 1
        )
        values = (
            torch.stack(
                [
                    torch.ones_like(losses),
                    losses,
                    correct.float(),
                    (top == target[:, None]).any(-1).float(),
                    1 / count,
                    count.log(),
                    brier,
                ],
                -1,
            )
            .cpu()
            .numpy()
        )
        for group, value in zip(ids, values):
            groups[group] += value
        for conf, hit in zip(confidence.cpu().numpy(), correct.cpu().numpy()):
            calibration[min(int(conf * 10), 9)] += [1, conf, hit]
    rows = np.stack(list(groups.values()))
    totals = rows.sum(0)
    n = int(totals[0])
    rng = np.random.default_rng(2037)
    bootstrap = []
    for _ in range(2000):
        sample = rows[rng.integers(0, len(rows), len(rows))].sum(0)
        bootstrap.append(sample[2] / sample[0])
    report = {
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_metadata": payload["metadata"],
        "data_manifest_sha256": sha256(Path(data) / "manifest.json"),
        "split": split,
        "positions": n,
        "opening_groups": len(groups),
        "nll": float(totals[1] / n),
        "top1": float(totals[2] / n),
        "top3": float(totals[3] / n),
        "uniform_top1": float(totals[4] / n),
        "uniform_nll": float(totals[5] / n),
        "multiclass_brier": float(totals[6] / n),
        "teacher_token_ece": float(sum(abs(r[1] - r[2]) for r in calibration) / n),
        "top1_opening_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "median_forward_batch_ms": float(np.median(latencies) * 1000),
        "seconds": time.monotonic() - started,
        "limitations": "Recorded-move agreement is not optimal-move accuracy. Positions are correlated; bootstrap clusters opening groups within one Fishtest run. No Elo is inferred.",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report
