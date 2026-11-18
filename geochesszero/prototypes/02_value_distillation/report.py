"""
Collect evaluation, training logs and match results into results/summary.json
and a readable results/SUMMARY.md.

Usage: python report.py
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def fmt(s, scale=1.0, digits=1):
    return (f"{scale * s['mean']:.{digits}f} [{scale * s['lo']:.{digits}f}, "
            f"{scale * s['hi']:.{digits}f}]")


def main():
    ev = json.load(open(os.path.join(RES, "eval.json")))
    players = ev["players"]

    # Average each variant over its seeds (per-seed numbers are in eval.json).
    rows = []
    for name, s in players.items():
        if ":" not in name:
            rows.append(dict(player=name, mode="reference", seeds=1,
                             cp_loss=s["cp_loss"]["mean"], best=s["best_rate"]["mean"],
                             blunder=s["blunder_rate"]["mean"], es=s["es_loss"]["mean"]))
    by_var = {}
    for name, s in players.items():
        if ":" in name:
            run, mode = name.split(":")
            var = run.rsplit("_s", 1)[0]
            by_var.setdefault((var, mode), []).append(s)
    for (var, mode), ss in sorted(by_var.items()):
        rows.append(dict(player=var, mode=mode, seeds=len(ss),
                         cp_loss=float(np.mean([s["cp_loss"]["mean"] for s in ss])),
                         cp_loss_seed_sd=float(np.std([s["cp_loss"]["mean"] for s in ss])),
                         best=float(np.mean([s["best_rate"]["mean"] for s in ss])),
                         blunder=float(np.mean([s["blunder_rate"]["mean"] for s in ss])),
                         es=float(np.mean([s["es_loss"]["mean"] for s in ss]))))

    value_r = {}
    for run, r in ev.get("value_r", {}).items():
        value_r.setdefault(run.rsplit("_s", 1)[0], []).append(r)

    train = {}
    for log in sorted(glob.glob(os.path.join(HERE, "runs", "*", "train_log.json"))):
        d = json.load(open(log))
        run = os.path.basename(os.path.dirname(log))
        best = min(d["log"], key=lambda r: r["val_loss"])
        train[run] = dict(best_epoch=best["epoch"], val_ce=best["val"]["ce"],
                          val_kl=best["val"]["kl"], val_value_mse=best["val"]["value_mse"],
                          val_top1=best["val"]["top1_teacher"], seconds=d["log"][-1]["seconds"],
                          params=d["params"], train_positions=None)

    matches, pair_scores = {}, {}
    for prefix, pat in (("", "arena_*.json"), ("no-repeat ", "norep_*.json")):
        for f in sorted(glob.glob(os.path.join(RES, pat))):
            key = prefix + os.path.basename(f)[6:-5]
            d = json.load(open(f))
            matches[key] = d["summary"]
            per_open = {}
            for g in d["games"]:
                per_open.setdefault(g["opening"], []).append(g["score_a"])
            pair_scores[key] = [float(np.mean(v)) for v in per_open.values()]

    # Pool seed-by-seed matches of the same pairing.
    pooled = {}
    for k, m in matches.items():
        base = k.rsplit("_s", 1)[0] if k[-3:-1] == "_s" else k
        p = pooled.setdefault(base, dict(w=0, d=0, l=0, games=0, seeds=0))
        for key in ("w", "d", "l", "games"):
            p[key] += m[key]
        p["seeds"] += 1
    rng = np.random.RandomState(0)
    for base, p in pooled.items():
        p["score"] = (p["w"] + 0.5 * p["d"]) / max(1, p["games"])
        pairs = np.concatenate([pair_scores[k] for k in matches
                                if (k.rsplit("_s", 1)[0] if k[-3:-1] == "_s" else k) == base])
        boots = [pairs[rng.randint(0, len(pairs), len(pairs))].mean() for _ in range(4000)]
        p["lo"], p["hi"] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        sc = min(max(p["score"], 1e-3), 1 - 1e-3)
        p["elo"] = float(400 * np.log10(sc / (1 - sc)))

    summary = dict(n_positions=ev["n_positions"], n_games=ev["n_games"], table=rows,
                   paired=ev["paired_differences"],
                   value_r={k: (float(np.mean(v)), len(v)) for k, v in value_r.items()},
                   train=train, matches=matches, pooled_matches=pooled)
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)

    L = [f"# Prototype 2 results\n",
         f"Test: {ev['n_positions']} positions from {ev['n_games']} held-out games, "
         f"scored by Stockfish 15.1 at depth 12.\n",
         "| Player | Move choice | Seeds | Centipawns lost / move | Best move | Blunders (>=200 cp) | Expected score lost |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: r["cp_loss"]):
        sd = f" (seed sd {r['cp_loss_seed_sd']:.1f})" if r.get("seeds", 1) > 1 else ""
        L.append(f"| {r['player']} | {r['mode']} | {r['seeds']} | {r['cp_loss']:.1f}{sd} | "
                 f"{100 * r['best']:.1f}% | {100 * r['blunder']:.1f}% | {r['es']:.2f} |")
    L.append("\nValue head Pearson r vs teacher: " +
             ", ".join(f"{k} {v[0]:.3f} ({v[1]} seeds)" for k, v in summary["value_r"].items()))
    L.append("\n## Paired differences (seed-averaged, same positions; negative = first is better)\n")
    L.append("| Comparison | Centipawns lost | Best-move rate (pp) | Blunder rate (pp) |")
    L.append("|---|---:|---:|---:|")
    for k, s in ev["paired_differences"].items():
        L.append(f"| {k} | {fmt(s['cp_loss'])} | {fmt(s['best_rate'], 100)} | {fmt(s['blunder_rate'], 100)} |")
    sub = ev.get("search_subset")
    if sub:
        L.append(f"\n## 2-ply search (every 4th test position: {sub['n_positions']} positions; seed-averaged)\n")
        L.append("| Variant | Policy: centipawns lost | 2-ply search: centipawns lost | Policy blunders | Search blunders |")
        L.append("|---|---:|---:|---:|---:|")
        search_rows = {}
        for var in ("ray", "static", "standard"):
            def avg(mode, met):
                vals = [s[met]["mean"] for k, s in sub["players"].items()
                        if k.startswith(var + "_s") and k.endswith(":" + mode)]
                return float(np.mean(vals)) if vals else float("nan")
            r = dict(policy=avg("policy", "cp_loss"), search=avg("search", "cp_loss"),
                     policy_blunder=avg("policy", "blunder_rate"), search_blunder=avg("search", "blunder_rate"))
            search_rows[var] = r
            L.append(f"| {var} | {r['policy']:.1f} | {r['search']:.1f} | {100 * r['policy_blunder']:.1f}% | "
                     f"{100 * r['search_blunder']:.1f}% |")
        for k in ("stockfish_d1", "stockfish_d4", "random"):
            p = sub["players"][k]
            L.append(f"| {k} (reference) | {p['cp_loss']['mean']:.1f} | | {100 * p['blunder_rate']['mean']:.1f}% | |")
        summary["search_subset"] = search_rows
        json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    if pooled:
        L.append("\n## Matches (colours reversed per opening; 'no-repeat' = both models skip "
                 "moves that repeat a position when an alternative exists)\n")
        L.append("| Pairing | Seeds | Games | +W =D -L | Score of first [95% CI] | Elo gap |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for k, p in pooled.items():
            L.append(f"| {k} | {p['seeds']} | {p['games']} | +{p['w']} ={p['d']} -{p['l']} | "
                     f"{100 * p['score']:.1f}% [{100 * p['lo']:.1f}, {100 * p['hi']:.1f}] | {p['elo']:+.0f} |")
    if train:
        L.append("\n## Training (best validation epoch)\n")
        L.append("| Run | Best epoch | Val KL to teacher | Val value MSE | Val top-1 vs teacher | Minutes |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for k, t in train.items():
            L.append(f"| {k} | {t['best_epoch']} | {t['val_kl']:.3f} | {t['val_value_mse']:.4f} | "
                     f"{100 * t['val_top1']:.1f}% | {t['seconds'] / 60:.0f} |")
    open(os.path.join(RES, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
