"""AlphaZero-Style Target Training & Replay Fine-Tuning for GeoChess (Chunk 2, Step 4).

Supports:
1. Policy cross-entropy against MCTS visit distributions pi(a) from selfplay_visits.jsonl,
   or Stockfish depth-10 soft targets (control) on the same self-play positions.
2. Value MSE against Stockfish, pure game outcome, MCTS search value, or blended targets.
3. Fine-tuning from an existing checkpoint with optional replay mixing from train_d10.
4. Full dev_d10 evaluation (policy cp_loss + value calibration MSE/ECE/correlation).
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Optional

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chess_policy.encoding import move_id  # noqa: E402
from chess_policy.v6_data import V6Dataset  # noqa: E402
from chess_policy.v6_model import load_any, parameter_count, save_v6  # noqa: E402
from chess_policy.v6_train import (  # noqa: E402
    CLIP,
    evaluate,
    soft_targets,
    to_device,
    value_target,
)


class AlphaZeroSelfplayDataset(V6Dataset):
    """V6Dataset augmented with aligned MCTS visit distributions and root search values."""

    def __init__(self, directory: str, visits_jsonl: str):
        super().__init__(directory)
        visits_path = Path(visits_jsonl)
        lines = visits_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(self):
            raise ValueError(
                f"Visits count ({len(lines)}) != dataset rows ({len(self)})"
            )

        fens = (self.dir / "fen.txt").read_text(encoding="utf-8").splitlines()
        visit_probs = np.zeros(len(self.legal), dtype=np.float32)
        root_vals = np.zeros(len(self), dtype=np.float32)

        for i, line in enumerate(lines):
            rec = json.loads(line)
            if rec["fen"] != fens[i]:
                raise ValueError(f"FEN mismatch at row {i}: {rec['fen']} != {fens[i]}")
            root_vals[i] = float(rec.get("root_val", 0.0))

            start, end = int(self.offsets[i]), int(self.offsets[i + 1])
            row_legal = np.asarray(self.legal[start:end])
            id_to_slot = {int(mid): slot for slot, mid in enumerate(row_legal)}

            row_probs = np.zeros(end - start, dtype=np.float32)
            for uci, prob in rec.get("policy_target", {}).items():
                mid = move_id(chess.Move.from_uci(uci))
                slot = id_to_slot.get(mid)
                if slot is not None:
                    row_probs[slot] = float(prob)

            total_p = float(row_probs.sum())
            if total_p > 0:
                row_probs /= total_p
            else:
                # Fallback to played move one-hot if visit counts were empty
                row_probs[int(self.target[i])] = 1.0
            visit_probs[start:end] = row_probs

        self.visit_probs = visit_probs
        self.root_vals = root_vals

    def batch(self, idx):
        out = super().batch(idx)
        sorted_idx = np.sort(np.asarray(idx))
        vp, _ = self._ragged(self.visit_probs, sorted_idx, 0.0)
        out["visit_probs"] = vp.astype(np.float32)
        out["root_val"] = self.root_vals[sorted_idx].astype(np.float32)
        return out


@dataclass
class AlphaZeroTrainConfig:
    selfplay: str = "data/v7/selfplay_d10"
    visits: str = "data/v7/selfplay_visits.jsonl"
    dev: str = "data/v6/dev_d10"
    init: str = "runs/v7/v7_S_pure_sf/best.pt"
    out: str = "runs/v7/v7_S_az_s0"
    policy_target: str = "alphazero"  # alphazero | soft | played
    value_mode: str = "blend_sf_search"  # sf | outcome | search | blend_sf_outcome | blend_sf_search | blend_search_outcome
    value_weight: float = 1.0
    value_mix: float = 0.25
    tau: float = 100.0
    replay_train: str = ""
    replay_ratio: float = 0.0
    epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 30
    max_dev_rows: int = 20000
    balance_weight: float = 0.01
    z_weight: float = 0.001
    seed: int = 0
    device: str = "auto"
    log_every: int = 10


def compute_policy_dist(
    batch: Dict[str, torch.Tensor], mode: str, tau: float
) -> torch.Tensor:
    """Return dense target distribution (B, W) aligned with batch['legal']."""
    legal = batch["legal"]
    if mode == "alphazero":
        return batch["visit_probs"].float()
    if mode == "soft":
        return soft_targets(batch["cp"], legal, tau)
    if mode == "played":
        dist = torch.zeros_like(legal, dtype=torch.float32)
        rows = torch.arange(legal.shape[0], device=legal.device)
        dist[rows, batch["target"]] = 1.0
        return dist
    raise ValueError(f"Unknown policy_target mode: {mode}")


def compute_value_target(
    batch: Dict[str, torch.Tensor], mode: str, mix: float
) -> torch.Tensor:
    """Return scalar value targets in [-1, 1] for each row in batch."""
    if mode == "sf":
        return value_target(batch, 0.0)
    if mode == "outcome":
        return batch["result"].float()
    if mode == "search":
        return batch["root_val"].float()
    if mode == "blend_sf_outcome":
        return (1.0 - mix) * value_target(batch, 0.0) + mix * batch["result"].float()
    if mode == "blend_sf_search":
        return (1.0 - mix) * value_target(batch, 0.0) + mix * batch["root_val"].float()
    if mode == "blend_search_outcome":
        return (1.0 - mix) * batch["root_val"].float() + mix * batch["result"].float()
    raise ValueError(f"Unknown value_mode: {mode}")


def _pad_width(tensor: torch.Tensor, target_width: int, pad_value) -> torch.Tensor:
    diff = target_width - tensor.shape[1]
    if diff <= 0:
        return tensor
    return F.pad(tensor, (0, diff), value=pad_value)


def _merge_batches(
    sp_batch: Dict[str, torch.Tensor],
    sp_pdist: torch.Tensor,
    sp_vtgt: torch.Tensor,
    rp_batch: Optional[Dict[str, torch.Tensor]],
    rp_pdist: Optional[torch.Tensor],
    rp_vtgt: Optional[torch.Tensor],
):
    """Concatenate self-play and replay batches, padding legal/policy_dist to common width."""
    if rp_batch is None:
        return (
            sp_batch["pieces"],
            sp_batch["geometry"],
            sp_batch["attacks"],
            sp_batch["legal"],
            sp_pdist,
            sp_vtgt,
        )
    max_w = max(sp_batch["legal"].shape[1], rp_batch["legal"].shape[1])
    pieces = torch.cat([sp_batch["pieces"], rp_batch["pieces"]], dim=0)
    geometry = torch.cat([sp_batch["geometry"], rp_batch["geometry"]], dim=0)
    attacks = torch.cat([sp_batch["attacks"], rp_batch["attacks"]], dim=0)
    legal = torch.cat(
        [
            _pad_width(sp_batch["legal"], max_w, -1),
            _pad_width(rp_batch["legal"], max_w, -1),
        ],
        dim=0,
    )
    pdist = torch.cat(
        [_pad_width(sp_pdist, max_w, 0.0), _pad_width(rp_pdist, max_w, 0.0)], dim=0
    )
    vtgt = torch.cat([sp_vtgt, rp_vtgt], dim=0)
    return pieces, geometry, attacks, legal, pdist, vtgt


@torch.no_grad()
def evaluate_with_ece(model, dataset, device, max_rows=20000, batch_size=512):
    """Run standard dev evaluation plus 10-bin Expected Calibration Error vs Stockfish."""
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


def train_alphazero(config: AlphaZeroTrainConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else config.device
    )

    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    log = (out / "log.jsonl").open("w", encoding="utf-8")

    def emit(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    sp_set = AlphaZeroSelfplayDataset(config.selfplay, config.visits)
    dev_set = V6Dataset(config.dev)

    # Keep small self-play dataset arrays in RAM for speed
    for name in ("pieces", "geometry", "attacks", "legal", "visit_probs"):
        setattr(sp_set, name, np.asarray(getattr(sp_set, name)))
    if sp_set.cp is not None:
        sp_set.cp = np.asarray(sp_set.cp)

    replay_set = None
    if config.replay_train and config.replay_ratio > 0:
        if not 0.0 < config.replay_ratio < 1.0:
            raise ValueError("replay_ratio must be strictly between 0.0 and 1.0")
        replay_set = V6Dataset(config.replay_train)

    model = load_any(config.init, device)
    if config.value_weight > 0 and not model.config.value_head:
        raise ValueError(f"Checkpoint {config.init} must have a value head")
    model.to(device).train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    rng = np.random.default_rng(config.seed)
    n_sp_per_batch = (
        max(1, int(round(config.batch_size * (1.0 - config.replay_ratio))))
        if replay_set is not None
        else config.batch_size
    )
    n_rp_per_batch = config.batch_size - n_sp_per_batch if replay_set is not None else 0

    sp_pool = np.arange(len(sp_set))
    steps_per_epoch = math.ceil(len(sp_pool) / n_sp_per_batch)
    total_steps = steps_per_epoch * config.epochs

    emit(
        dict(
            event="start",
            device=str(device),
            parameters=parameter_count(model),
            selfplay_rows=len(sp_set),
            replay_rows=len(replay_set) if replay_set is not None else 0,
            n_sp_per_batch=n_sp_per_batch,
            n_rp_per_batch=n_rp_per_batch,
            dev_rows=min(len(dev_set), config.max_dev_rows or 10**9),
            steps=total_steps,
        )
    )

    # Baseline evaluation (epoch -1) before any fine-tuning updates
    started = time.monotonic()
    baseline_dev = evaluate_with_ece(model, dev_set, device, config.max_dev_rows)
    emit(dict(event="baseline", epoch=-1, minutes=0.0, **baseline_dev))

    best = float("inf")
    best_epoch = -1
    step = 0
    history = []

    for epoch in range(config.epochs):
        order = rng.permutation(sp_pool)
        running = dict(loss=0.0, policy=0.0, value=0.0, n=0)

        for s in range(0, len(order), n_sp_per_batch):
            sp_idx = order[s : s + n_sp_per_batch]
            sp_b = to_device(sp_set.batch(sp_idx), device)
            sp_pdist = compute_policy_dist(sp_b, config.policy_target, config.tau)
            sp_vtgt = compute_value_target(sp_b, config.value_mode, config.value_mix)

            rp_b, rp_pdist, rp_vtgt = None, None, None
            if replay_set is not None and n_rp_per_batch > 0:
                rp_idx = rng.integers(0, len(replay_set), size=n_rp_per_batch)
                rp_b = to_device(replay_set.batch(rp_idx), device)
                rp_pdist = soft_targets(rp_b["cp"], rp_b["legal"], config.tau)
                rp_vtgt = value_target(rp_b, 0.0)

            pieces, geometry, attacks, legal, pdist, vtgt = _merge_batches(
                sp_b, sp_pdist, sp_vtgt, rp_b, rp_pdist, rp_vtgt
            )

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
                res = model(pieces, geometry, attacks, legal)

            logp = F.log_softmax(res["logits"].float(), -1)
            p_loss = (
                -torch.where(legal >= 0, pdist * logp, torch.zeros_like(logp))
                .sum(-1)
                .mean()
            )
            loss = (
                p_loss
                + config.balance_weight * res["balance"]
                + config.z_weight * res["z_loss"]
            )
            v_loss = torch.zeros((), device=device)
            if config.value_weight > 0:
                v_loss = F.mse_loss(res["value"].float(), vtgt)
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
                emit(
                    dict(
                        event="train",
                        epoch=epoch,
                        step=step,
                        of=total_steps,
                        lr=round(config.learning_rate * scale, 7),
                        **{
                            k: round(running[k] / running["n"], 4)
                            for k in ("loss", "policy", "value")
                        },
                        minutes=round((time.monotonic() - started) / 60, 2),
                    )
                )
                running = dict(loss=0.0, policy=0.0, value=0.0, n=0)

        dev = evaluate_with_ece(model, dev_set, device, config.max_dev_rows)
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
        baseline=baseline_dev,
        best_epoch=best_epoch,
        best_dev_score=best,
        history=history,
        parameters=parameter_count(model),
        minutes=round((time.monotonic() - started) / 60, 2),
        clip=CLIP,
    )
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit(dict(event="done", best_epoch=best_epoch, best_dev_score=best, minutes=summary["minutes"]))
    log.close()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="AlphaZero-style training on self-play data.")
    defaults = AlphaZeroTrainConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        ap.add_argument(flag, type=type(value), default=value)
    args = ap.parse_args(argv)
    cfg = AlphaZeroTrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    train_alphazero(cfg)


if __name__ == "__main__":
    main()
