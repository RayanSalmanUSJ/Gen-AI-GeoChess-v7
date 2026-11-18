"""Update-matched policy continuation: replay-only control versus targeted data.

All supervised outputs remain move tokens. No value regression or search is
introduced. Exact multi-solution mate examples optimize total valid-move mass.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .checkpoint import load_checkpoint, save_checkpoint
from .corrections import CorrectionDataset
from .data import EpochSampler, ShardDataset, atomic_json, sha256
from .encoding import collate
from .mate_curriculum import MateDataset
from .model import model_inputs


def collate_targets(examples):
    batch = collate(examples)
    accepted = torch.zeros_like(batch["legal"], dtype=torch.bool)
    for row, example in enumerate(examples):
        choices = example.get("acceptable", [int(example["target"])])
        for choice in choices:
            accepted[row] |= batch["legal"][row] == choice
        if not accepted[row].any():
            raise ValueError("no acceptable legal target")
    batch["accepted"] = accepted
    return batch


def set_nll(logits, accepted):
    """-log probability of selecting any admissible target; CE for a singleton."""
    return -torch.logsumexp(logits.log_softmax(-1).masked_fill(~accepted, float("-inf")), dim=-1)


@torch.inference_mode()
def measure(model, dataset, *, limit=0, batch_size=64):
    model.eval()
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    total = correct = top3 = 0
    loss = 0.0
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_targets):
        output = model(**model_inputs(batch))
        logits = output["logits"]
        top = logits.topk(min(3, logits.shape[1]), dim=-1).indices
        accepted = batch["accepted"]
        total += len(top)
        loss += set_nll(logits, accepted).sum().item()
        correct += accepted.gather(1, top[:, :1]).sum().item()
        top3 += accepted.gather(1, top).any(-1).sum().item()
    return {"rows": total, "nll": loss / total, "top1": correct / total, "top3": top3 / total}


@dataclass(frozen=True)
class RefinementConfig:
    steps: int = 1800
    batch: int = 64
    pool_rows: int = 64000
    original_dev_rows: int = 12000
    evaluate_every: int = 300
    learning_rate: float = 0.0002
    seed: int = 8327
    threads: int = 2
    original_nll_tolerance: float = 0.05


def train_refinement(
    checkpoint, base_data, corrections, mates, output, *, mode="targeted", config=RefinementConfig()
):
    if (
        mode not in ("targeted", "control")
        or config.batch % 8
        or min(config.steps, config.batch, config.pool_rows, config.evaluate_every) < 1
    ):
        raise ValueError("invalid refinement configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("use a fresh refinement output directory")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model, _ = load_checkpoint(checkpoint)
    original = ShardDataset(base_data, "train", verify=True)
    original_dev = ShardDataset(base_data, "dev")
    correction_train = CorrectionDataset(corrections, "train")
    correction_dev = CorrectionDataset(corrections, "dev")
    mate_train = MateDataset(mates, "train")
    mate_dev = MateDataset(mates, "dev")
    pool = []
    for index in EpochSampler(original, config.seed, 0, config.pool_rows):
        example = original[index]
        pool.append(
            {
                k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                for k, v in example.items()
            }
        )
    original.cache.clear()
    devsets = {"original": original_dev, "correction": correction_dev, "mate": mate_dev}

    def validate():
        scores = {
            key: measure(model, value, limit=config.original_dev_rows if key == "original" else 0)
            for key, value in devsets.items()
        }
        objective = (
            0.5 * scores["original"]["nll"]
            + 0.375 * scores["correction"]["nll"]
            + 0.125 * scores["mate"]["nll"]
        )
        return {"partitions": scores, "weighted_objective": objective}

    started = time.monotonic()
    initial = validate()
    ceiling = initial["partitions"]["original"]["nll"] + config.original_nll_tolerance
    best = initial["weighted_objective"]
    selected = 0
    provenance = {
        "mode": mode,
        "config": asdict(config),
        "parent_sha256": sha256(checkpoint),
        "base_manifest_sha256": sha256(Path(base_data) / "manifest.json"),
        "correction_manifest_sha256": sha256(Path(corrections) / "manifest.json"),
        "mate_manifest_sha256": sha256(Path(mates) / "manifest.json"),
    }
    atomic_json(
        output / "run.json",
        {
            **provenance,
            "initial": initial,
            "selection": "minimum weighted dev NLL subject to original dev NLL <= initial + tolerance",
        },
    )
    save_checkpoint(output / "best.pt", model, {**provenance, "step": 0, "dev": initial})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    history = []
    train_loss = 0.0
    for step in range(1, config.steps + 1):
        if mode == "control":
            examples = [pool[i] for i in rng.integers(0, len(pool), config.batch)]
        else:
            examples = [pool[i] for i in rng.integers(0, len(pool), config.batch // 2)]
            examples += [
                correction_train[int(i)]
                for i in rng.integers(0, len(correction_train), config.batch * 3 // 8)
            ]
            examples += [
                mate_train[int(i)] for i in rng.integers(0, len(mate_train), config.batch // 8)
            ]
        batch = collate_targets(examples)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output_values = model(**model_inputs(batch))
        policy_loss = set_nll(output_values["logits"], batch["accepted"]).mean()
        loss = policy_loss + 0.01 * output_values["balance"] + 0.001 * output_values["z_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite refinement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        factor = min(1.0, step / 50) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / config.steps)))
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * factor
        optimizer.step()
        train_loss += policy_loss.item()
        if step % config.evaluate_every == 0 or step == config.steps:
            dev = validate()
            qualifies = dev["partitions"]["original"]["nll"] <= ceiling
            if qualifies and dev["weighted_objective"] < best:
                best = dev["weighted_objective"]
                selected = step
                save_checkpoint(output / "best.pt", model, {**provenance, "step": step, "dev": dev})
            row = {
                "step": step,
                "mean_train_nll": train_loss / step,
                "dev": dev,
                "qualifies": qualifies,
                "selected_step": selected,
                "seconds": time.monotonic() - started,
            }
            history.append(row)
            atomic_json(output / "history.json", {"initial": initial, "evaluations": history})
            print(json.dumps(row), flush=True)
    save_checkpoint(output / "final.pt", model, {**provenance, "step": config.steps, "dev": dev})
    return {
        "mode": mode,
        "selected_step": selected,
        "selected_objective": best,
        "seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pilot.pt"))
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, default=Path("v4_data/corrections"))
    parser.add_argument("--mates", type=Path, default=Path("v4_data/mates"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["targeted", "control"], default="targeted")
    args = parser.parse_args()
    print(json.dumps(train_refinement(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
