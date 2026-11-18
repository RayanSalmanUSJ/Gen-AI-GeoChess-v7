"""Single-device training: AMP, accumulation, warmup/cosine, safe epoch resume.

The shipped implementation intentionally has no untested distributed trainer.
Large configurations need suitable hardware; all experts reside on one device.
"""

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from .checkpoint import save_checkpoint, load_checkpoint, capture_rng, restore_rng
from .data import ShardDataset, EpochSampler, sha256, atomic_json
from .encoding import collate
from .model import ChessPolicy, model_inputs


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 64
    accumulation: int = 1
    learning_rate: float = 0.0008
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_train_rows: int = 160000
    max_dev_rows: int = 12000
    balance_weight: float = 0.01
    z_weight: float = 0.001
    seed: int = 20261116
    threads: int = 4
    workers: int = 0
    device: str = "cpu"
    precision: str = "fp32"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.accumulation, self.threads) < 1:
            raise ValueError("training counts must be positive")
        if min(self.max_train_rows, self.max_dev_rows, self.warmup_steps, self.workers) < 0:
            raise ValueError("limits must be nonnegative")
        if (
            self.learning_rate <= 0
            or min(self.weight_decay, self.balance_weight, self.z_weight) < 0
        ):
            raise ValueError("invalid optimizer parameters")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("unsupported precision")


def amp(config, device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if config.precision == "bf16" else torch.float16,
        enabled=config.precision != "fp32",
    )


def statistics(output, batch, loss):
    top = output["logits"].detach().topk(min(3, output["logits"].shape[1]), dim=-1).indices
    target = batch["target"]
    return np.array(
        [
            len(target),
            float(loss.detach()) * len(target),
            (top[:, 0] == target).sum().item(),
            (top == target[:, None]).any(-1).sum().item(),
            (1.0 / (batch["legal"] >= 0).sum(-1)).sum().item(),
        ]
    )


def finish(totals):
    n, loss, top1, top3, uniform = map(float, totals)
    if not n:
        raise ValueError("no rows processed")
    return {
        "positions": int(n),
        "nll": loss / n,
        "top1": top1 / n,
        "top3": top3 / n,
        "uniform_top1": uniform / n,
    }


@torch.inference_mode()
def validate(model, loader, config, device):
    model.eval()
    totals, loads = np.zeros(5), None
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp(config, device):
            output = model(**model_inputs(batch))
            loss = F.cross_entropy(output["logits"], batch["target"])
        totals += statistics(output, batch, loss)
        current = output["expert_loads"].cpu().numpy() * len(batch["target"])
        loads = current if loads is None else loads + current
    result = finish(totals)
    result["expert_assignment_fractions"] = (loads / totals[0]).tolist()
    return result


def train(data, output, model_config, config, resume=None):
    data, output = Path(data), Path(output)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("distributed training is not implemented; use one process")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if device.type == "cpu" and config.precision == "fp16":
        raise ValueError("CPU fp16 is unsupported; choose fp32 or bf16")
    if device.type == "cuda" and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16")
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "last.pt").exists():
        raise FileExistsError("run exists; resume or choose a fresh output")
    if resume and Path(resume).resolve().parent != output.resolve():
        raise ValueError("resume must use the existing run directory")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dataset = ShardDataset(data, "train", verify=True)
    development = ShardDataset(data, "dev")
    manifest_hash = sha256(data / "manifest.json")
    model = ChessPolicy(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.precision == "fp16"
    )
    first_epoch, step, best, history, rng = 0, 0, float("inf"), [], None
    if resume:
        restored, payload = load_checkpoint(resume, device)
        state = payload.get("training")
        if state is None:
            raise ValueError("inference checkpoint cannot resume training")
        if (
            payload["config"] != asdict(model_config)
            or state["config"] != asdict(config)
            or state["manifest_sha256"] != manifest_hash
        ):
            raise ValueError("resume requires identical config and data")
        model.load_state_dict(restored.state_dict())
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first_epoch, step, best = state["epoch"], state["step"], state["best_dev_nll"]
        history, rng = state["history"], state["rng"]
    dev_count = min(len(development), config.max_dev_rows or len(development))
    dev_loader = DataLoader(
        Subset(development, range(dev_count)),
        batch_size=config.batch_size,
        collate_fn=collate,
        num_workers=config.workers,
    )
    train_count = min(len(dataset), config.max_train_rows or len(dataset))
    total_steps = (
        math.ceil(math.ceil(train_count / config.batch_size) / config.accumulation) * config.epochs
    )
    started = time.monotonic()
    if resume is None:
        initial = validate(model, dev_loader, config, device)
        atomic_json(
            output / "run.json",
            {
                "model": asdict(model_config),
                "train": asdict(config),
                "parameters": sum(p.numel() for p in model.parameters()),
                "manifest_sha256": manifest_hash,
                "torch": str(torch.__version__),
                "device": str(device),
                "initial_development": initial,
            },
        )
        print(json.dumps({"initial_development": initial}), flush=True)
    if rng is not None:
        restore_rng(rng)
    for epoch in range(first_epoch, config.epochs):
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            collate_fn=collate,
            sampler=EpochSampler(dataset, config.seed, epoch, config.max_train_rows),
            num_workers=config.workers,
            pin_memory=device.type == "cuda",
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(5)
        for index, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            window_start = index // config.accumulation * config.accumulation
            window_rows = min(
                config.accumulation * config.batch_size,
                len(loader.sampler) - window_start * config.batch_size,
            )
            # Weight the final short microbatch correctly within an accumulation window.
            fraction = len(batch["target"]) / window_rows
            with amp(config, device):
                result = model(**model_inputs(batch))
                ce = F.cross_entropy(result["logits"], batch["target"])
                loss = (
                    ce
                    + config.balance_weight * result["balance"]
                    + config.z_weight * result["z_loss"]
                ) * fraction
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss; previous checkpoint preserved")
            scaler.scale(loss).backward()
            totals += statistics(result, batch, ce)
            if (index + 1) % config.accumulation == 0 or index + 1 == len(loader):
                step += 1
                warmup = min(config.warmup_steps, max(0, total_steps - 1))
                scale = (
                    step / warmup
                    if warmup and step <= warmup
                    else 0.1
                    + 0.45
                    * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
                )
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * scale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % 200 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "batch": index + 1,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        dev = validate(model, dev_loader, config, device)
        history.append(
            {
                "epoch": epoch + 1,
                "step": step,
                "train": finish(totals),
                "dev": dev,
                "seconds": time.monotonic() - started,
            }
        )
        improved = dev["nll"] < best
        best = min(best, dev["nll"])
        metadata = {
            "epoch": epoch + 1,
            "dev": dev,
            "manifest_sha256": manifest_hash,
            "selection": "minimum development NLL only",
        }
        if improved:
            save_checkpoint(output / "best.pt", model, metadata)
        save_checkpoint(
            output / "last.pt",
            model,
            metadata,
            training={
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": asdict(config),
                "epoch": epoch + 1,
                "step": step,
                "best_dev_nll": best,
                "history": history,
                "manifest_sha256": manifest_hash,
                "rng": capture_rng(),
            },
        )
        atomic_json(output / "history.json", {"epochs": history})
        print(json.dumps(history[-1]), flush=True)
    return {"epochs": history, "best_dev_nll": best}
