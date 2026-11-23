"""Value head calibration analysis for GeoChess.

Measures calibration of predicted value V in (-1, 1) against:
1. Stockfish 15.1 expected score: v_sf = 2 / (1 + exp(-0.00368208 * best_cp)) - 1
2. Actual game outcome: v_game = result in {-1, 0, 1}
3. Training blend target: v_target = 0.75 * v_sf + 0.25 * v_game

Outputs MSE, MAE, Expected Calibration Error (ECE), Pearson and Spearman
correlations, sign classification accuracy, and 10-bin reliability diagram tables.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from chess_policy.v6_data import V6Dataset
from chess_policy.v6_model import load_v6
from chess_policy.v6_train import to_device


def compute_calibration_metrics(pred_v, target_sf, target_game, target_blend, n_bins=10):
    """pred_v, target_sf, target_game, target_blend are in [-1, 1]."""
    # Mapped probabilities in [0, 1]
    p_pred = (pred_v + 1.0) / 2.0
    p_sf = (target_sf + 1.0) / 2.0
    p_game = (target_game + 1.0) / 2.0

    # Errors
    mse_sf = float(np.mean((pred_v - target_sf) ** 2))
    mse_game = float(np.mean((pred_v - target_game) ** 2))
    mse_blend = float(np.mean((pred_v - target_blend) ** 2))

    mae_sf = float(np.mean(np.abs(pred_v - target_sf)))
    mae_game = float(np.mean(np.abs(pred_v - target_game)))

    # Correlations
    r_sf = float(np.corrcoef(pred_v, target_sf)[0, 1])
    rank_p = np.argsort(np.argsort(pred_v))
    rank_sf = np.argsort(np.argsort(target_sf))
    rho_sf = float(np.corrcoef(rank_p, rank_sf)[0, 1])

    # Sign accuracy on decisive positions (|best_cp| >= 50, i.e. not dead equal)
    decisive = np.abs(target_sf) > 0.1
    if np.any(decisive):
        sign_acc = float(np.mean(np.sign(pred_v[decisive]) == np.sign(target_sf[decisive])))
    else:
        sign_acc = float(np.mean(np.sign(pred_v) == np.sign(target_sf)))

    # Reliability diagram with equal-width bins in [0, 1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []
    ece_sf = 0.0
    ece_game = 0.0
    max_err_sf = 0.0

    n_total = len(pred_v)
    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (p_pred >= low) & (p_pred <= high)
        else:
            mask = (p_pred >= low) & (p_pred < high)
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(p_pred[mask]))
            mean_sf = float(np.mean(p_sf[mask]))
            mean_game = float(np.mean(p_game[mask]))
            gap_sf = abs(mean_pred - mean_sf)
            gap_game = abs(mean_pred - mean_game)
            ece_sf += (count / n_total) * gap_sf
            ece_game += (count / n_total) * gap_game
            max_err_sf = max(max_err_sf, gap_sf)
        else:
            mean_pred = (low + high) / 2.0
            mean_sf = 0.0
            mean_game = 0.0
            gap_sf = 0.0
            gap_game = 0.0

        bins_data.append({
            "bin": b,
            "range": [round(float(low), 2), round(float(high), 2)],
            "count": count,
            "pct": round(count / n_total * 100, 2),
            "mean_pred_prob": round(mean_pred, 4),
            "mean_sf_prob": round(mean_sf, 4),
            "mean_game_prob": round(mean_game, 4),
            "gap_sf": round(gap_sf, 4),
            "gap_game": round(gap_game, 4),
        })

    return {
        "mse_sf": round(mse_sf, 4),
        "mse_game": round(mse_game, 4),
        "mse_blend": round(mse_blend, 4),
        "mae_sf": round(mae_sf, 4),
        "mae_game": round(mae_game, 4),
        "pearson_r_sf": round(r_sf, 4),
        "spearman_rho_sf": round(rho_sf, 4),
        "sign_acc_decisive": round(sign_acc * 100, 2),
        "ece_sf": round(float(ece_sf), 4),
        "ece_game": round(float(ece_game), 4),
        "mce_sf": round(float(max_err_sf), 4),
        "bins": bins_data,
    }


def evaluate_value_calibration(model, dataset, device, batch_size=512, max_rows=0):
    n_rows = len(dataset) if not max_rows else min(len(dataset), max_rows)
    preds = []
    t_sf = []
    t_game = []
    t_blend = []

    model.eval()
    with torch.no_grad():
        for s in range(0, n_rows, batch_size):
            idx = np.arange(s, min(n_rows, s + batch_size))
            b = to_device(dataset.batch(idx), device)
            out = model(b["pieces"], b["geometry"], b["attacks"], b["legal"])
            if "value" not in out:
                raise ValueError("Model has no value head!")
            val = out["value"].cpu().numpy()
            preds.append(val)

            best_cp = np.clip(b["best_cp"].cpu().numpy().astype(np.float64), -2000, 2000)
            engine = 2.0 / (1.0 + np.exp(-0.00368208 * best_cp)) - 1.0
            game = b["result"].cpu().numpy().astype(np.float64)
            blend = 0.75 * engine + 0.25 * game

            t_sf.append(engine)
            t_game.append(game)
            t_blend.append(blend)

    pred_v = np.concatenate(preds)
    target_sf = np.concatenate(t_sf)
    target_game = np.concatenate(t_game)
    target_blend = np.concatenate(t_blend)

    return compute_calibration_metrics(pred_v, target_sf, target_game, target_blend)


def main():
    parser = argparse.ArgumentParser(description="Evaluate value head calibration.")
    parser.add_argument("--data", default="data/v6/dev_d10", help="Labelled dataset path")
    parser.add_argument("--model", action="append", default=[], help="name=path to model checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0, help="0 for all rows")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    models = dict(m.split("=", 1) for m in args.model)
    if not models:
        models = {
            "scale_S_1M": "runs/v6/scale_S_1M/best.pt",
            "scale_L_1M": "runs/v6/scale_L_1M/best.pt",
            "C_value_s0": "runs/v6/C_value_s0/best.pt",
        }

    device = torch.device(args.device)
    print(f"Loading dataset: {args.data} ...")
    t0 = time.time()
    ds = V6Dataset(args.data)
    print(f"Loaded {len(ds)} rows in {time.time() - t0:.1f}s.")

    results = {}
    for name, path in models.items():
        print(f"\nEvaluating value calibration for '{name}' ({path}) ...")
        t_start = time.time()
        model, cfg = load_v6(path)
        model.to(device)
        res = evaluate_value_calibration(model, ds, device, args.batch_size, args.max_rows)
        results[name] = res
        print(f"Done in {time.time() - t_start:.2f}s.")
        print(f"  MSE vs Stockfish: {res['mse_sf']:.4f} | MAE: {res['mae_sf']:.4f}")
        print(f"  MSE vs Game:      {res['mse_game']:.4f} | MAE: {res['mae_game']:.4f}")
        print(f"  Pearson r:        {res['pearson_r_sf']:.4f} | Spearman rho: {res['spearman_rho_sf']:.4f}")
        print(f"  ECE vs Stockfish: {res['ece_sf']:.4f} | MCE: {res['mce_sf']:.4f}")
        print(f"  Sign accuracy:    {res['sign_acc_decisive']:.1f}%")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote full metrics to {args.out}")


if __name__ == "__main__":
    main()
