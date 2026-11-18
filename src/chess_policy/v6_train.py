"""v6 trainer: one loop for every objective of the core experiment.

  --policy-target played   behavioral cloning: cross-entropy with the move that was played (v5)
  --policy-target soft     value distillation: cross-entropy with q(m) = softmax(cp(m) / tau)
                           over the legal moves (team paper eq. 21)
  --value-weight L         adds L * (V - v_target)^2 with a value head (team paper eq. 19);
                           v_target = (1 - mix) * expected_score(best cp) + mix * game result

Same optimiser and schedule for every objective (AdamW, warmup + cosine), MoE auxiliary
losses as in v5. The checkpoint kept is the epoch with the lowest DEVELOPMENT centipawn
loss (or development NLL when the development set has no scores), so every objective is
selected by the same rule.

Example (run from the repo root, with src on PYTHONPATH):
  python -m chess_policy.v6_train --train data/v6/train_d10 --dev data/v6/dev_d10 \
      --out runs/v6/B_soft_s0 --policy-target soft --tau 100 --seed 0
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .v6_data import V6Dataset, cp_to_value
from .v6_model import V6Config, V6Policy, load_any, parameter_count, save_v6

CLIP = 1000


@dataclass
class TrainV6Config:
    train: str
    dev: str
    out: str
    policy_target: str = "soft"  # played | soft
    tau: float = 100.0
    value_weight: float = 0.0
    value_mix: float = 0.25
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 300
    max_train_rows: int = 0
    max_dev_rows: int = 20000
    balance_weight: float = 0.01
    z_weight: float = 0.001
    seed: int = 0
    device: str = "auto"
    width: int = 64
    layers: int = 2
    heads: int = 4
    experts: int = 4
    attention: str = "soft"
    init: str = ""
    log_every: int = 200


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if k == "game":
            continue
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device, non_blocking=True)
    return out


def soft_targets(cp, legal, tau):
    """q(m) proportional to exp(cp(m) / tau) over legal moves; cp clipped to +/-1000."""
    logits = (cp.clamp(-CLIP, CLIP) / tau).masked_fill(legal < 0, float("-inf"))
    return logits.softmax(-1)


def policy_loss(logits, batch, config):
    logp = F.log_softmax(logits.float(), -1)
    if config.policy_target == "played":
        return F.nll_loss(logp, batch["target"])
    q = soft_targets(batch["cp"], batch["legal"], config.tau)
    return -torch.where(batch["legal"] >= 0, q * logp, torch.zeros_like(logp)).sum(-1).mean()


def value_target(batch, mix):
    best = batch["best_cp"].float().clamp(-2000, 2000)
    engine = 2.0 / (1.0 + torch.exp(-0.00368208 * best)) - 1.0
    return (1 - mix) * engine + mix * batch["result"].float()


def choice_metrics(cp_rows, lengths, choice):
    """Prototype 2's metrics: centipawns lost by the chosen move (scores clipped to
    +/-1000), best-move rate, mistakes (>=100) and blunders (>=200)."""
    loss = np.empty(len(choice))
    for i, (row, n, j) in enumerate(zip(cp_rows, lengths, choice)):
        c = np.clip(row[:n].astype(np.float64), -CLIP, CLIP)
        loss[i] = c.max() - c[j]
    return dict(cp_loss=loss, best_rate=(loss <= 0).astype(float),
                mistake_rate=(loss >= 100).astype(float),
                blunder_rate=(loss >= 200).astype(float))


@torch.no_grad()
def evaluate(model, dataset, device, max_rows=0, batch_size=512):
    model.eval()
    n = len(dataset) if not max_rows else min(max_rows, len(dataset))
    sums = dict(nll=0.0, top1=0.0, top3=0.0, cp_loss=0.0, best_rate=0.0, blunder_rate=0.0,
                value_se=0.0)
    values, targets = [], []
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(n, s + batch_size))
        raw = dataset.batch(idx)
        b = to_device(raw, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(b["pieces"], b["geometry"], b["attacks"], b["legal"])
        logits = out["logits"].float()
        sums["nll"] += F.cross_entropy(logits, b["target"], reduction="sum").item()
        order = logits.argsort(-1, descending=True)
        sums["top1"] += (order[:, 0] == b["target"]).sum().item()
        sums["top3"] += (order[:, :3] == b["target"][:, None]).any(-1).sum().item()
        if "cp" in raw:
            m = choice_metrics(raw["cp"], dataset.lengths[idx], order[:, 0].cpu().numpy())
            for k in ("cp_loss", "best_rate", "blunder_rate"):
                sums[k] += m[k].sum()
            if "value" in out:
                vt = value_target(b, 0.0)
                values.append(out["value"].float().cpu().numpy())
                targets.append(vt.cpu().numpy())
    result = {k: float(v / n) for k, v in sums.items() if k != "value_se"}
    if dataset.cp is None:
        for k in ("cp_loss", "best_rate", "blunder_rate"):
            result.pop(k)
    if values:
        v, t = np.concatenate(values), np.concatenate(targets)
        result["value_mse_vs_engine"] = float(np.mean((v - t) ** 2))
        result["value_corr_vs_engine"] = float(np.corrcoef(v, t)[0, 1])
    result["rows"] = n
    model.train()
    return result


def train(config: TrainV6Config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto"
        else config.device
    )
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    log = (out / "log.jsonl").open("a")

    def emit(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    train_set, dev_set = V6Dataset(config.train), V6Dataset(config.dev)
    needs_cp = config.policy_target == "soft" or config.value_weight > 0
    if needs_cp and train_set.cp is None:
        raise ValueError("this objective needs a labelled training set (cp.npy)")
    # Keep the hot arrays in RAM (a few hundred MB for ~280K positions).
    for name in ("pieces", "geometry", "attacks", "legal"):
        setattr(train_set, name, np.asarray(getattr(train_set, name)))
    if train_set.cp is not None:
        train_set.cp = np.asarray(train_set.cp)

    if config.init:
        model = load_any(config.init)
        if config.value_weight > 0 and not model.config.value_head:
            cfg = V6Config(**{**asdict(model.config), "value_head": True})
            fresh = V6Policy(cfg)
            fresh.load_state_dict(model.state_dict(), strict=False)
            model = fresh
    else:
        model = V6Policy(V6Config(width=config.width, layers=config.layers, heads=config.heads,
                                  experts=config.experts, top_k=min(2, config.experts),
                                  attention=config.attention,
                                  value_head=config.value_weight > 0))
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    pool = np.arange(len(train_set))
    if config.max_train_rows and config.max_train_rows < len(pool):
        pool = np.sort(rng.choice(pool, config.max_train_rows, replace=False))
    steps_per_epoch = math.ceil(len(pool) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    emit(dict(event="start", device=str(device), parameters=parameter_count(model),
              train_rows=int(len(pool)), dev_rows=min(len(dev_set), config.max_dev_rows or 10**9),
              steps=total_steps, model=asdict(model.config)))
    best, best_epoch, step, history = float("inf"), -1, 0, []
    started = time.monotonic()
    for epoch in range(config.epochs):
        order = rng.permutation(pool)
        running = dict(loss=0.0, policy=0.0, value=0.0, n=0)
        for s in range(0, len(order), config.batch_size):
            b = to_device(train_set.batch(order[s:s + config.batch_size]), device)
            warm = min(config.warmup_steps, max(1, total_steps - 1))
            scale = (step + 1) / warm if step < warm else 0.5 * (
                1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * scale
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                res = model(b["pieces"], b["geometry"], b["attacks"], b["legal"])
            p_loss = policy_loss(res["logits"], b, config)
            loss = p_loss + config.balance_weight * res["balance"] + config.z_weight * res["z_loss"]
            v_loss = torch.zeros((), device=device)
            if config.value_weight > 0:
                v_loss = F.mse_loss(res["value"].float(), value_target(b, config.value_mix))
                loss = loss + config.value_weight * v_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            running["loss"] += loss.item()
            running["policy"] += p_loss.item()
            running["value"] += v_loss.item()
            running["n"] += 1
            if step % config.log_every == 0:
                emit(dict(event="train", epoch=epoch, step=step, of=total_steps,
                          lr=round(config.learning_rate * scale, 7),
                          **{k: round(running[k] / running["n"], 4)
                             for k in ("loss", "policy", "value")},
                          minutes=round((time.monotonic() - started) / 60, 2)))
                running = dict(loss=0.0, policy=0.0, value=0.0, n=0)
        dev = evaluate(model, dev_set, device, config.max_dev_rows)
        score = dev.get("cp_loss", dev["nll"])
        improved = bool(score < best)
        if improved:
            best, best_epoch = score, epoch
            save_v6(out / "best.pt", model, dict(epoch=epoch, dev=dev, train=asdict(config)))
        save_v6(out / "last.pt", model, dict(epoch=epoch, dev=dev, train=asdict(config)))
        history.append(dict(epoch=epoch, **dev))
        emit(dict(event="dev", epoch=epoch, improved=improved,
                  minutes=round((time.monotonic() - started) / 60, 2), **dev))
    summary = dict(best_epoch=best_epoch, best_dev_score=best, history=history,
                   parameters=parameter_count(model),
                   minutes=round((time.monotonic() - started) / 60, 2))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit(dict(event="done", **{k: v for k, v in summary.items() if k != "history"}))
    log.close()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chess_policy.v6_train")
    defaults = TrainV6Config(train="", dev="", out="")
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if name in ("train", "dev", "out"):
            ap.add_argument(flag, required=True)
        else:
            ap.add_argument(flag, type=type(value), default=value)
    args = ap.parse_args(argv)
    train(TrainV6Config(**{k.replace("-", "_"): v for k, v in vars(args).items()}))


if __name__ == "__main__":
    main()
