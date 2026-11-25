"""Big -> Small Model Knowledge Distillation for GeoChess (Chunk 2, Step 5).

1. Caches teacher (e.g. runs/v6/scale_L_1M/best.pt) legal-move logits and value
   predictions across data/v6/train_d10 once under data/v7/ so multi-epoch CPU
   training avoids repeating expensive teacher forward passes.
2. Trains a compact S student (width=64, layers=2, heads=4, hard_ray, ~307K params)
   with:
   - Policy distillation loss: alpha * T^2 * KL(softmax(z_T / T) || softmax(z_S / T))
     + (1 - alpha) * CrossEntropy(soft_sf_targets(tau=100), softmax(z_S))
   - Value MSE loss against blended teacher value and pure Stockfish expected score
     (value_mix = 0.0 against noisy game outcomes).
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chess_policy.v6_data import V6Dataset  # noqa: E402
from chess_policy.v6_model import (  # noqa: E402
    V6Config,
    V6Policy,
    load_any,
    parameter_count,
    save_v6,
)
from chess_policy.v6_train import (  # noqa: E402
    CLIP,
    evaluate,
    soft_targets,
    to_device,
    value_target,
)


def cache_teacher_targets(
    teacher_ckpt: str,
    dataset_dir: str,
    cache_dir: str,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    """Run teacher once over dataset_dir and store logits + values on disk."""
    cache_path = Path(cache_dir)
    meta_path = cache_path / "meta.json"
    logits_path = cache_path / "teacher_logits.npy"
    value_path = cache_path / "teacher_value.npy"

    ds = V6Dataset(dataset_dir)
    n_rows = len(ds)
    n_legal = len(ds.legal)

    if meta_path.exists() and logits_path.exists() and value_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            old.get("teacher") == str(teacher_ckpt)
            and old.get("dataset") == str(dataset_dir)
            and old.get("rows") == n_rows
            and old.get("legal_len") == n_legal
        ):
            print(f"Reusing cached teacher targets from {cache_path} ({n_rows} rows).")
            return old

    cache_path.mkdir(parents=True, exist_ok=True)
    teacher = load_any(teacher_ckpt, device)
    teacher.eval()
    if not teacher.config.value_head:
        raise ValueError(f"Teacher checkpoint {teacher_ckpt} has no value head")

    t_logits = np.zeros(n_legal, dtype=np.float32)
    t_value = np.zeros(n_rows, dtype=np.float32)

    print(f"Caching teacher ({teacher_ckpt}) predictions on {dataset_dir} ({n_rows} rows)...")
    started = time.monotonic()
    next_log = 25000

    with torch.inference_mode():
        for s in range(0, n_rows, batch_size):
            end = min(n_rows, s + batch_size)
            idx = np.arange(s, end)
            b = to_device(ds.batch(idx), device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                out = teacher(b["pieces"], b["geometry"], b["attacks"], b["legal"])

            logits_np = out["logits"].float().cpu().numpy()
            vals_np = out["value"].float().cpu().numpy()

            for row_j, i in enumerate(idx):
                lo, hi = int(ds.offsets[i]), int(ds.offsets[i + 1])
                t_logits[lo:hi] = logits_np[row_j, : hi - lo]
                t_value[i] = float(vals_np[row_j])

            if end >= next_log or end == n_rows:
                elapsed = time.monotonic() - started
                rate = end / max(elapsed, 1e-6)
                eta_min = (n_rows - end) / max(rate, 1e-6) / 60.0
                print(
                    f"  cached {end:6d}/{n_rows} ({100.0 * end / n_rows:5.1f}%) | "
                    f"{rate:5.1f} pos/s | elapsed {elapsed / 60:4.1f}m | eta {eta_min:4.1f}m",
                    flush=True,
                )
                next_log += 25000

    np.save(logits_path, t_logits, allow_pickle=False)
    np.save(value_path, t_value, allow_pickle=False)
    meta = dict(
        teacher=str(teacher_ckpt),
        dataset=str(dataset_dir),
        rows=n_rows,
        legal_len=n_legal,
        teacher_params=parameter_count(teacher),
        seconds=round(time.monotonic() - started, 2),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved teacher cache to {cache_path} in {meta['seconds']}s.")
    return meta


class DistillDataset(V6Dataset):
    """V6Dataset paired with precomputed teacher legal-move logits and values."""

    def __init__(self, directory: str, cache_dir: str):
        super().__init__(directory)
        cache_path = Path(cache_dir)
        self.teacher_logits = np.load(cache_path / "teacher_logits.npy")
        self.teacher_value = np.load(cache_path / "teacher_value.npy")
        if len(self.teacher_logits) != len(self.legal):
            raise ValueError("teacher_logits length mismatch with legal array")
        if len(self.teacher_value) != len(self):
            raise ValueError("teacher_value length mismatch with dataset rows")

    def batch(self, idx):
        out = super().batch(idx)
        sorted_idx = np.sort(np.asarray(idx))
        tl, _ = self._ragged(self.teacher_logits, sorted_idx, float("-inf"))
        out["teacher_logits"] = tl.astype(np.float32)
        out["teacher_value"] = self.teacher_value[sorted_idx].astype(np.float32)
        return out


@dataclass
class DistillConfig:
    teacher: str = "runs/v6/scale_L_1M/best.pt"
    train: str = "data/v6/train_d10"
    dev: str = "data/v6/dev_d10"
    cache_dir: str = "data/v7/teacher_cache_L_train_d10"
    out: str = "runs/v7/v7_S_distill_L"
    distill_alpha: float = 0.7
    distill_temp: float = 2.0
    tau: float = 100.0
    value_weight: float = 1.0
    value_alpha: float = 0.5
    epochs: int = 8
    batch_size: int = 256
    cache_batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 300
    max_dev_rows: int = 20000
    balance_weight: float = 0.01
    z_weight: float = 0.001
    width: int = 64
    layers: int = 2
    heads: int = 4
    experts: int = 4
    attention: str = "hard_ray"
    init: str = ""
    seed: int = 0
    device: str = "cpu"
    log_every: int = 200
    cache_only: bool = False


def distillation_loss(res, batch, config: DistillConfig):
    """Compute Hinton temperature-scaled KD + Stockfish soft target policy loss and value MSE."""
    legal = batch["legal"]
    valid = legal >= 0
    s_logits = res["logits"].float()
    t_logits = batch["teacher_logits"].float()
    T = float(config.distill_temp)

    t_masked = t_logits.masked_fill(~valid, float("-inf"))
    t_probs_T = F.softmax(t_masked / T, dim=-1)
    t_logp_T = F.log_softmax(t_masked / T, dim=-1)
    s_logp_T = F.log_softmax(s_logits / T, dim=-1)

    kd_per_row = torch.where(
        valid, t_probs_T * (t_logp_T - s_logp_T), torch.zeros_like(s_logp_T)
    ).sum(-1)
    kd_loss = (T * T) * kd_per_row.mean()

    q_sf = soft_targets(batch["cp"], legal, config.tau)
    s_logp_1 = F.log_softmax(s_logits, dim=-1)
    sf_loss = -torch.where(valid, q_sf * s_logp_1, torch.zeros_like(s_logp_1)).sum(-1).mean()

    p_loss = config.distill_alpha * kd_loss + (1.0 - config.distill_alpha) * sf_loss

    v_sf = value_target(batch, 0.0)
    v_teacher = batch["teacher_value"].float()
    v_tgt = config.value_alpha * v_teacher + (1.0 - config.value_alpha) * v_sf
    v_loss = F.mse_loss(res["value"].float(), v_tgt)

    total = (
        p_loss
        + config.value_weight * v_loss
        + config.balance_weight * res["balance"]
        + config.z_weight * res["z_loss"]
    )
    return total, p_loss, kd_loss, sf_loss, v_loss


@torch.no_grad()
def evaluate_dev_with_ece(model, dataset, device, max_rows=20000, batch_size=512):
    metrics = evaluate(model, dataset, device, max_rows=max_rows, batch_size=batch_size)
    n = len(dataset) if not max_rows else min(max_rows, len(dataset))
    preds, t_sf = [], []
    model.eval()
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(n, s + batch_size))
        b = to_device(dataset.batch(idx), device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            out = model(b["pieces"], b["geometry"], b["attacks"], b["legal"])
        if "value" in out:
            preds.append(out["value"].float().cpu().numpy())
            t_sf.append(value_target(b, 0.0).cpu().numpy())
    if preds:
        pv = (np.concatenate(preds) + 1.0) / 2.0
        psf = (np.concatenate(t_sf) + 1.0) / 2.0
        edges = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for i in range(10):
            mask = (pv >= edges[i]) & (pv <= edges[i + 1] if i == 9 else pv < edges[i + 1])
            cnt = int(mask.sum())
            if cnt > 0:
                ece += (cnt / len(pv)) * abs(float(pv[mask].mean() - psf[mask].mean()))
        metrics["value_ece_vs_engine"] = float(ece)
    model.train()
    return metrics


def train_distill(config: DistillConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else config.device
    )

    cache_teacher_targets(
        config.teacher,
        config.train,
        config.cache_dir,
        device,
        batch_size=config.cache_batch_size,
    )
    if config.cache_only:
        return {"cached_only": True}

    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    log = (out / "log.jsonl").open("w", encoding="utf-8")

    def emit(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    train_set = DistillDataset(config.train, config.cache_dir)
    dev_set = V6Dataset(config.dev)

    for name in ("pieces", "geometry", "attacks", "legal"):
        setattr(train_set, name, np.asarray(getattr(train_set, name)))
    if train_set.cp is not None:
        train_set.cp = np.asarray(train_set.cp)

    if config.init:
        model = load_any(config.init, device)
    else:
        model = V6Policy(
            V6Config(
                width=config.width,
                layers=config.layers,
                heads=config.heads,
                experts=config.experts,
                top_k=min(2, config.experts),
                attention=config.attention,
                value_head=config.value_weight > 0,
            )
        )
    model.to(device).train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed)
    pool = np.arange(len(train_set))
    steps_per_epoch = math.ceil(len(pool) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    emit(
        dict(
            event="start",
            device=str(device),
            parameters=parameter_count(model),
            train_rows=len(pool),
            dev_rows=min(len(dev_set), config.max_dev_rows or 10**9),
            steps=total_steps,
            model=asdict(model.config),
        )
    )

    best, best_epoch, step, history = float("inf"), -1, 0, []
    started = time.monotonic()

    for epoch in range(config.epochs):
        order = rng.permutation(pool)
        running = dict(loss=0.0, policy=0.0, kd=0.0, sf=0.0, value=0.0, n=0)

        for s in range(0, len(order), config.batch_size):
            b = to_device(train_set.batch(order[s : s + config.batch_size]), device)
            warm = min(config.warmup_steps, max(1, total_steps - 1))
            scale = (
                (step + 1) / warm
                if step < warm
                else 0.5
                * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
            )
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * scale

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                res = model(b["pieces"], b["geometry"], b["attacks"], b["legal"])

            loss, p_loss, kd_loss, sf_loss, v_loss = distillation_loss(res, b, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            running["loss"] += loss.item()
            running["policy"] += p_loss.item()
            running["kd"] += kd_loss.item()
            running["sf"] += sf_loss.item()
            running["value"] += v_loss.item()
            running["n"] += 1

            if step % config.log_every == 0:
                emit(
                    dict(
                        event="train",
                        epoch=epoch,
                        step=step,
                        of=total_steps,
                        lr=round(config.learning_rate * scale, 7),
                        **{
                            k: round(running[k] / running["n"], 4)
                            for k in ("loss", "policy", "kd", "sf", "value")
                        },
                        minutes=round((time.monotonic() - started) / 60, 2),
                    )
                )
                running = dict(loss=0.0, policy=0.0, kd=0.0, sf=0.0, value=0.0, n=0)

        dev = evaluate_dev_with_ece(model, dev_set, device, config.max_dev_rows)
        score = dev.get("cp_loss", dev["nll"])
        improved = bool(score < best)
        if improved:
            best, best_epoch = score, epoch
            save_v6(
                out / "best.pt",
                model,
                dict(epoch=epoch, dev=dev, train=asdict(config)),
            )
        save_v6(
            out / "last.pt",
            model,
            dict(epoch=epoch, dev=dev, train=asdict(config)),
        )
        history.append(dict(epoch=epoch, **dev))
        emit(
            dict(
                event="dev",
                epoch=epoch,
                improved=improved,
                minutes=round((time.monotonic() - started) / 60, 2),
                **dev,
            )
        )

    summary = dict(
        best_epoch=best_epoch,
        best_dev_score=best,
        history=history,
        parameters=parameter_count(model),
        minutes=round((time.monotonic() - started) / 60, 2),
        clip=CLIP,
    )
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    emit(dict(event="done", best_epoch=best_epoch, best_dev_score=best, minutes=summary["minutes"]))
    log.close()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Knowledge distillation (Big L -> Small S).")
    defaults = DistillConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            ap.add_argument(flag, action="store_true")
        else:
            ap.add_argument(flag, type=type(value), default=value)
    args = ap.parse_args(argv)
    cfg = DistillConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    train_distill(cfg)


if __name__ == "__main__":
    main()
