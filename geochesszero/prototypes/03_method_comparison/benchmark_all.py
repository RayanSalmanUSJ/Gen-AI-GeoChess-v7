"""
One benchmark table for every method, on the same 1,312 held-out positions
(Prototype 2's test set, priced by Stockfish 15.1 at depth 12).

Rows: reference players, the team's C++ alpha-beta engine, the GPT-style
move model (normal and scaled data), the three board transformers, the
self-play model, 2-ply search, and any local-LLM results found in
results/llm_*.json (those use a subset of positions; flagged in the table).

Multi-seed methods are averaged per position over seeds, then summarised with
a game-cluster bootstrap 95% interval, like every other row.

Usage: python benchmark_all.py
"""

import glob
import json
import os
import sys

import chess
import chess.engine
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "02_value_distillation")
ROOT = os.path.join(HERE, "..", "..")
for p in (P2, ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import encoding as enc  # noqa: E402
from evaluate import choice_metrics, policy_choices, search_choices, summarize  # noqa: E402
from model import load_checkpoint  # noqa: E402
from train import load_npz  # noqa: E402

TEAM_ENGINE = os.environ.get("TEAM_ENGINE", os.path.join(HERE, "engines", "november_adapter", "november_uci"))
STOCKFISH = os.environ.get("STOCKFISH", "stockfish")
METRICS = ("cp_loss", "best_rate", "blunder_rate", "es_loss")


def uci_choices(path, fens, idx, depth):
    eng = chess.engine.SimpleEngine.popen_uci(path)
    out = []
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        mv = eng.play(b, chess.engine.Limit(depth=depth)).move
        out.append(int(np.where(idx[i] == enc.action_index(b, mv))[0][0]))
    eng.quit()
    return np.array(out)


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    d = load_npz(os.path.join(P2, "data", "test.npz"))
    fens, idx, cp, games = [str(f) for f in d["fen"]], d["idx"], d["cp"], d["game"]
    n_legal = (idx >= 0).sum(1)
    T = dict(x=torch.from_numpy(d["x"].astype(np.float32)),
             occ=torch.from_numpy(d["occ"].astype(np.float32)),
             idx=torch.from_numpy(idx.astype(np.int64)))
    rows = []

    def add(name, family, per_pos_list, note=""):
        avg = {m: np.mean([pp[m] for pp in per_pos_list], axis=0) for m in METRICS}
        s = summarize(avg, games)
        rows.append(dict(name=name, family=family, seeds=len(per_pos_list), note=note,
                         positions=int(len(games)), **{m: s[m] for m in METRICS}))
        print(f"{name:42s} cp_loss {s['cp_loss']['mean']:6.1f} blunders {100 * s['blunder_rate']['mean']:5.1f}%",
              flush=True)

    rng = np.random.RandomState(0)
    add("Random moves", "reference",
        [choice_metrics(cp, n_legal, np.array([rng.randint(n) for n in n_legal]))])
    for depth in (1, 4):
        add(f"Stockfish 15.1, depth {depth}", "reference",
            [choice_metrics(cp, n_legal, uci_choices(STOCKFISH, fens, idx, depth))])
    for depth in (3, 4):
        add(f"Team C++ engine (alpha-beta), depth {depth}", "classical search",
            [choice_metrics(cp, n_legal, uci_choices(TEAM_ENGINE, fens, idx, depth))],
            "november.zip module 12, unchanged")

    # GPT-style move models (needs move histories)
    from gpt_movelm import gpt_choices, load_gpt, game_sequences
    from shared.move_encoder import move_to_index
    seqs = game_sequences(os.path.join(P2, "data", "test_games.jsonl"))
    prefixes, legal_abs = [], []
    for i, fen in enumerate(fens):
        pre = seqs[int(games[i])][:int(d["ply"][i])]
        b = chess.Board()
        for m in pre:
            b.push(m)
        by_index = {enc.action_index(b, m): m for m in b.legal_moves}
        legal_abs.append(np.array([move_to_index(by_index[int(x)]) for x in idx[i, :n_legal[i]]]))
        prefixes.append(pre)
    raw_legal = {}
    for label, pattern in (("GPT-style move model (204 games)", "runs/gpt_s[0-9]"),
                           ("GPT-style move model (+1,200 games)", "runs/gptscale_s[0-9]")):
        pp, rl = [], []
        for run in sorted(glob.glob(os.path.join(HERE, pattern))):
            if os.path.isfile(os.path.join(run, "best.pt")):
                m, _ = load_gpt(os.path.join(run, "best.pt"))
                ch, raw = gpt_choices(m, prefixes, legal_abs)
                pp.append(choice_metrics(cp, n_legal, ch))
                rl.append(float(raw.mean()))
        if pp:
            add(label, "sequence model (GPT)", pp,
                f"top move legal without the legal-move filter: {100 * np.mean(rl):.1f}%")
            raw_legal[label] = float(np.mean(rl))

    # Board transformers from Prototype 2
    labels = {"standard": "Standard transformer (board)", "static": "Geometry masks, static (Prototype 1)",
              "ray": "Geometry masks, ray (Prototype 2)"}
    for var, label in labels.items():
        pp = []
        for run in sorted(glob.glob(os.path.join(P2, "runs", f"{var}_s[0-9]"))):
            m, _ = load_checkpoint(os.path.join(run, "best.pt"))
            pp.append(choice_metrics(cp, n_legal, policy_choices(m, T)[0]))
        add(label, "board transformer", pp)

    # Search and self-play
    ray0, _ = load_checkpoint(os.path.join(P2, "runs", "ray_s0", "best.pt"))
    add("Ray + 2-ply search (seed 0)", "board transformer + search",
        [choice_metrics(cp, n_legal, search_choices(ray0, fens, idx, n_legal))])
    for pattern, label, note in (
            ("selfplay_ray_s0", "Ray + self-play, search-guided ({n} round)",
             "AlphaZero-style expert iteration; no Stockfish labels"),
            ("selfplay_pg_ray_s0", "Ray + self-play, wins/losses only ({n} rounds)",
             "policy gradient on game results; no Stockfish labels")):
        ckpts = sorted(glob.glob(os.path.join(HERE, "runs", pattern, "iter*.pt")),
                       key=lambda p: int(os.path.basename(p)[4:-3]))
        if ckpts:
            sp_model, _ = load_checkpoint(ckpts[-1])
            n = int(os.path.basename(ckpts[-1])[4:-3])
            add(label.format(n=n), "self-play", [choice_metrics(cp, n_legal, policy_choices(sp_model, T)[0])],
                note)

    # Local LLM results (run on a machine with Ollama), if any
    for f in sorted(glob.glob(os.path.join(HERE, "results", "llm_*.json"))):
        r = json.load(open(f))
        rows.append(dict(name=r["player"], family="local LLM", seeds=1, positions=r["positions"],
                         note=f"subset; first-try legal {100 * r['first_try_legal_rate']:.0f}%",
                         **{m: r[m] for m in METRICS}))

    json.dump(dict(rows=rows, raw_legal=raw_legal, n_positions=len(fens)),
              open(os.path.join(HERE, "results", "comparison.json"), "w"), indent=1)
    L = ["# Method comparison", "",
         f"{len(fens)} held-out positions from {len(np.unique(games))} games, priced by Stockfish 15.1 at depth 12. "
         "Centipawns lost per move (scores capped at +/-1000); 95% intervals resample whole games.", "",
         "| Method | Family | Seeds | Centipawns lost / move [95% CI] | Best move | Blunders (200+ cp) | Note |",
         "|---|---|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: r["cp_loss"]["mean"]):
        c = r["cp_loss"]
        L.append(f"| {r['name']} | {r['family']} | {r['seeds']} | {c['mean']:.1f} [{c['lo']:.1f}, {c['hi']:.1f}] | "
                 f"{100 * r['best_rate']['mean']:.1f}% | {100 * r['blunder_rate']['mean']:.1f}% | {r['note']} |")
    open(os.path.join(HERE, "results", "COMPARISON.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
