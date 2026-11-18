"""Run the Phase 3 plan unattended, one stage at a time, skipping finished runs.

  python -m chess_policy.v6_experiments --stage baseline   # A: behavioral cloning, 3 seeds (needs only
                                                          #    dev labels; can start before train labels)
  python -m chess_policy.v6_experiments --stage tau        # pick the soft-target temperature
  python -m chess_policy.v6_experiments --stage core       # A played / B soft / C soft+value, 3 seeds
  python -m chess_policy.v6_experiments --stage attention  # plain / hard / hard_ray / hard_only (+ C = soft)
  python -m chess_policy.v6_experiments --stage moe        # C without mixture of experts
  python -m chess_policy.v6_experiments --stage scaling    # sizes S/M/L x data sizes (needs D2 for 1M)
  python -m chess_policy.v6_experiments --stage eval       # score every run on T2 (and T1), write tables
  python -m chess_policy.v6_experiments --stage all        # everything above, in order

Defaults assume the layout written by the data steps in CLAUDE.md:
  data/v6/train_d10, data/v6/dev_d10, data/v6/big_d10 (D2, optional),
  benchmarks/t2_p2_1312, data/v6/t1_d16 (optional)
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "v6"
RESULTS = ROOT / "results" / "v6"

SIZES = {  # width, layers, heads, learning rate
    "S": (64, 2, 4, 1e-3),
    "M": (128, 6, 8, 6e-4),
    "L": (256, 6, 8, 4e-4),
}


def run(name, args, data):
    out = RUNS / name
    if (out / "metrics.json").exists():
        print(f"[skip] {name} (finished)", flush=True)
        return out
    cmd = [sys.executable, "-m", "chess_policy.v6_train", "--train", data["train"],
           "--dev", data["dev"], "--out", str(out)] + [str(a) for a in args]
    print("[run]", name, " ".join(cmd[3:]), flush=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    with (RUNS / f"{name}.stdout.txt").open("w") as log:
        code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise SystemExit(f"run {name} failed (exit {code}); see {RUNS / (name + '.stdout.txt')}")
    return out


def dev_score(name):
    path = RUNS / name / "metrics.json"
    return json.loads(path.read_text())["best_dev_score"] if path.exists() else None


def choose_tau():
    scores = {t: dev_score(f"tau{t}_s0") for t in (50, 100, 200)}
    scores = {t: s for t, s in scores.items() if s is not None}
    if not scores:
        raise SystemExit("run --stage tau first")
    return min(scores, key=scores.get)


def mean_score(prefix):
    vals = [dev_score(f"{prefix}_s{s}") for s in range(3)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def choose_attention():
    options = {"soft": mean_score("C_value")}
    for mode in ("plain", "hard", "hard_ray", "hard_only"):
        options[mode] = mean_score(f"att_{mode}")
    options = {k: v for k, v in options.items() if v is not None}
    if not options:
        raise SystemExit("run --stage attention first")
    return min(options, key=options.get)


def stage_tau(data):
    for tau in (50, 100, 200):
        run(f"tau{tau}_s0", ["--policy-target", "soft", "--tau", tau, "--seed", 0], data)


def stage_baseline(data):
    """Run A (the v5 recipe) on the same positions; unlabelled train data is enough."""
    train = data["train_played"] if Path(data["train_played"]).exists() else data["train"]
    for seed in range(3):
        run(f"A_played_s{seed}", ["--policy-target", "played", "--seed", seed],
            dict(data, train=train))


def stage_core(data):
    stage_baseline(data)
    tau = choose_tau()
    for seed in range(3):
        run(f"B_soft_s{seed}", ["--policy-target", "soft", "--tau", tau, "--seed", seed], data)
        run(f"C_value_s{seed}", ["--policy-target", "soft", "--tau", tau, "--value-weight", 1.0,
                                 "--seed", seed], data)


def stage_attention(data):
    tau = choose_tau()
    for seed in range(3):
        for mode in ("plain", "hard", "hard_ray", "hard_only"):
            run(f"att_{mode}_s{seed}", ["--policy-target", "soft", "--tau", tau,
                                        "--value-weight", 1.0, "--attention", mode,
                                        "--seed", seed], data)


def stage_moe(data):
    tau = choose_tau()
    for seed in range(3):
        run(f"moe_off_s{seed}", ["--policy-target", "soft", "--tau", tau, "--value-weight", 1.0,
                                 "--experts", 1, "--seed", seed], data)


def stage_scaling(data):
    tau, attention = choose_tau(), choose_attention()
    plans = [("70K", 70000, data["train"]), ("140K", 140000, data["train"]),
             ("279K", 0, data["train"])]
    if Path(data["big"]).exists():
        plans.append(("1M", 0, data["big"]))
    for size, (width, layers, heads, lr) in SIZES.items():
        for label, rows, train in plans:
            epochs = 4 if label == "1M" else 6
            run(f"scale_{size}_{label}", ["--policy-target", "soft", "--tau", tau,
                                          "--value-weight", 1.0, "--attention", attention,
                                          "--width", width, "--layers", layers, "--heads", heads,
                                          "--learning-rate", lr, "--epochs", epochs,
                                          "--max-train-rows", rows, "--seed", 0],
                dict(data, train=train))


def stage_eval(data):
    """Score every finished run on T2 (and T1 when present); write summary tables."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    finished = sorted(p.parent.name for p in RUNS.glob("*/metrics.json"))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    for bench_name, bench in (("t2", data["t2"]), ("t1", data["t1"])):
        if not Path(bench).exists():
            continue
        models = ["v3=checkpoints/pilot.pt", "v4=checkpoints/policy_v4.pt",
                  "v5=checkpoints/policy_v5.pt"]
        search = []
        for name in finished:
            models.append(f"{name}={RUNS / name / 'best.pt'}")
            cfg = json.loads((RUNS / name / "config.json").read_text())
            if cfg.get("value_weight", 0) > 0:
                search.append(name)
        cmd = [sys.executable, "-m", "chess_policy.v6_eval", "--data", bench,
               "--reference", "v5", "--out", str(RESULTS / f"{bench_name}.json")]
        for m in models:
            cmd += ["--model", m]
        for s in search:
            cmd += ["--search", s]
        print("[eval]", bench_name, len(models), "models,", len(search), "with search", flush=True)
        subprocess.check_call(cmd, cwd=ROOT, env=env)
    write_summary()


def _fmt(stat, pct=False):
    if pct:
        return f"{100 * stat['mean']:.1f}%"
    return f"{stat['mean']:.1f} [{stat['lo']:.1f}, {stat['hi']:.1f}]"


def write_summary():
    lines = ["# v6 results (auto-generated by v6_experiments --stage eval)", ""]
    for bench in ("t2", "t1"):
        path = RESULTS / f"{bench}.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text())
        lines += [f"## {bench.upper()}: {rep['rows']} positions from {rep['games']} games", "",
                  "| Model | cp lost / move [95% CI] | Best move | Blunders | Dev cp lost |",
                  "| --- | --- | --- | --- | --- |"]
        rows = []
        for key, s in rep["models"].items():
            run_name = key.split(":")[0]
            dev = dev_score(run_name)
            rows.append((s["cp_loss"]["mean"], f"| {key} | {_fmt(s['cp_loss'])} | "
                         f"{_fmt(s['best_rate'], True)} | {_fmt(s['blunder_rate'], True)} | "
                         f"{'' if dev is None else f'{dev:.1f}'} |"))
        lines += [r for _, r in sorted(rows)] + [""]
        paired = rep.get("paired_vs_v5", {})
        if paired:
            lines += ["Paired difference vs v5 (negative = fewer centipawns lost than v5):", "",
                      "| Model | Δ cp lost [95% CI] |", "| --- | --- |"]
            for key, s in sorted(paired.items(), key=lambda kv: kv[1]["cp_loss"]["mean"]):
                lines.append(f"| {key} | {_fmt(s['cp_loss'])} |")
            lines.append("")
    # seed-averaged groups on T2
    path = RESULTS / "t2.json"
    if path.exists():
        rep = json.loads(path.read_text())["models"]
        groups = {}
        for key, s in rep.items():
            base, _, mode = key.partition(":")
            if "_s" in base and base.rsplit("_s", 1)[1].isdigit():
                groups.setdefault(base.rsplit("_s", 1)[0] + (":" + mode if mode else ""), []).append(
                    s["cp_loss"]["mean"])
        if groups:
            lines += ["## Seed-averaged centipawn loss on T2", "", "| Group | Seeds | Mean | Std |",
                      "| --- | --- | --- | --- |"]
            for g, vals in sorted(groups.items(), key=lambda kv: np.mean(kv[1])):
                lines.append(f"| {g} | {len(vals)} | {np.mean(vals):.1f} | {np.std(vals):.1f} |")
            lines.append("")
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("wrote", RESULTS / "SUMMARY.md", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chess_policy.v6_experiments")
    ap.add_argument("--stage", required=True,
                    choices=["baseline", "tau", "core", "attention", "moe", "scaling", "eval",
                             "all"])
    ap.add_argument("--train", default="data/v6/train_d10")
    ap.add_argument("--train-played", default="data/v6/train_nolabel",
                    help="unlabelled copy of the same train positions, for the baseline stage")
    ap.add_argument("--dev", default="data/v6/dev_d10")
    ap.add_argument("--big", default="data/v6/big_d10")
    ap.add_argument("--t2", default="benchmarks/t2_p2_1312")
    ap.add_argument("--t1", default="data/v6/t1_d16")
    a = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    data = dict(train=a.train, train_played=a.train_played, dev=a.dev, big=a.big, t2=a.t2,
                t1=a.t1)
    stages = {"baseline": stage_baseline, "tau": stage_tau, "core": stage_core, "attention": stage_attention,
              "moe": stage_moe, "scaling": stage_scaling, "eval": stage_eval}
    for name in (list(stages) if a.stage == "all" else [a.stage]):
        print(f"=== stage {name}", flush=True)
        stages[name](data)


if __name__ == "__main__":
    main()
