"""V5 controlled policy refinement on pawnful attack and defense examples."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .corrections import CorrectionDataset
from .data import EpochSampler, ShardDataset, atomic_json, sha256
from .mate_curriculum import MateDataset
from .model import model_inputs
from .refinement import collate_targets, measure, set_nll
from .tactics import TacticDataset


@dataclass(frozen=True)
class Config:
    steps: int = 1800
    batch: int = 64
    pool_rows: int = 64000
    original_dev_rows: int = 12000
    evaluate_every: int = 300
    learning_rate: float = 0.00015
    seed: int = 28327
    threads: int = 2
    original_nll_tolerance: float = 0.02


def train(
    checkpoint, base_data, corrections, mates, tactics, output, *, mode="targeted", config=Config()
):
    if (
        mode not in ("control", "targeted")
        or config.batch % 8
        or min(config.steps, config.batch, config.threads) < 1
    ):
        raise ValueError("invalid v5 configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("v5 output directory must be empty")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model, _ = load_checkpoint(checkpoint)
    original = ShardDataset(base_data, "train", verify=True)
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
    sources = {
        "original": pool,
        "correction": CorrectionDataset(corrections, "train"),
        "sparse": MateDataset(mates, "train"),
        "mate": TacticDataset(tactics, "train", "mate"),
        "defense": TacticDataset(tactics, "train", "defense"),
    }
    devsets = {
        "original": ShardDataset(base_data, "dev"),
        "correction": CorrectionDataset(corrections, "dev"),
        "mate": TacticDataset(tactics, "dev", "mate"),
        "defense": TacticDataset(tactics, "dev", "defense"),
    }
    weights = {"original": 0.5, "correction": 0.25, "mate": 0.125, "defense": 0.125}
    mixture = (
        {"original": 4, "correction": 3, "sparse": 1}
        if mode == "control"
        else {"original": 4, "correction": 2, "mate": 1, "defense": 1}
    )

    def validate():
        values = {
            key: measure(model, data, limit=config.original_dev_rows if key == "original" else 0)
            for key, data in devsets.items()
        }
        return {
            "partitions": values,
            "weighted_objective": sum(weights[k] * values[k]["nll"] for k in weights),
        }

    started = time.monotonic()
    initial = validate()
    ceiling = initial["partitions"]["original"]["nll"] + config.original_nll_tolerance
    best = initial["weighted_objective"]
    selected = 0
    provenance = {
        "mode": mode,
        "config": asdict(config),
        "parent_sha256": sha256(checkpoint),
        "selection_weights": weights,
        "mixture_eighths": mixture,
        "dataset_manifests": {
            k: sha256(Path(root) / "manifest.json")
            for k, root in [
                ("original", base_data),
                ("correction", corrections),
                ("sparse", mates),
                ("tactics", tactics),
            ]
        },
    }
    atomic_json(
        output / "run.json",
        {
            **provenance,
            "initial": initial,
            "selection": "lowest weighted dev NLL subject to original <= parent + tolerance",
        },
    )
    save_checkpoint(output / "best.pt", model, {**provenance, "step": 0, "dev": initial})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    history = []
    train_loss = 0.0
    for step in range(1, config.steps + 1):
        examples = []
        for name, fraction in mixture.items():
            data = sources[name]
            examples.extend(
                data[int(i)] for i in rng.integers(0, len(data), fraction * config.batch // 8)
            )
        batch = collate_targets(examples)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(**model_inputs(batch))
        policy_loss = set_nll(result["logits"], batch["accepted"]).mean()
        loss = policy_loss + 0.01 * result["balance"] + 0.001 * result["z_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite v5 loss")
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
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/policy_v4.pt"))
    p.add_argument("--base-data", type=Path, required=True)
    p.add_argument("--corrections", type=Path, default=Path("v4_data/corrections"))
    p.add_argument("--mates", type=Path, default=Path("v4_data/mates"))
    p.add_argument("--tactics", type=Path, default=Path("v5_data/tactics"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["control", "targeted"], default="targeted")
    print(json.dumps(train(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
