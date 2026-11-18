# GeoChess — instructions for Claude Code (chunk 1 of 3)

Read this whole file before doing anything. It carries over a long conversation the user had
with Claude in the Claude app; you start without that memory.

## Who you are working for

- The user leads a 3-person team for a **Generative AI university course project**. After you
  finish chunk 1, two teammates continue (chunk 2 and chunk 3) from your `HANDOFF.md`.
- The user is not deeply technical. Keep updates short and plain: what finished, what is
  running, what you found, how long until the next result. No walls of logs.
- They said they want big chunks done properly, not lots of small pieces, and they do not want
  scope creep: stay inside the plan below. Ask before anything outside the project folder or
  any download over ~1 GB.
- Machine: Windows desktop, RTX 3060 (12 GB), Python via Miniconda. Stockfish and the team's
  C++ engine are not installed yet.

## The project

Project statement (from the team): *chess as next-move prediction over a 64-node
multi-relational graph, where attention heads are masked to rook, bishop, knight and king
movement geometries. The objective is value-optimal play, not behavioral cloning of human
moves. The architecture remains isomorphic to GPT.* Other ideas in scope: model distillation
and self-generated training data (chunk 2). Out of scope: fine-tuning an LLM on chess books,
the GPT move-sequence model (dead end), the hand-crafted evaluator.

The team's own paper (projet_data.pdf, not in this folder) defines a 4-phase curriculum —
behavioral cloning → engine distillation with soft targets q(m) ∝ exp(cp(m)/τ) → self-play →
oracle distillation — plus dual policy/value heads, search, and an evaluation protocol
(Elo vs a fixed pool, centipawn loss, blunder rate, value calibration, search-scaling curve).

Full roadmap (Phases 0–6, schedule, risks, who does what):
https://claude.ai/artifact/Kz4qUXjPHppPxYD59FPw7w

## Where things stand (18 Nov 2026)

Cross-evaluation of every earlier prototype (all on the same positions; details in
`geochesszero/prototypes/04_crosseval/RESULTS.md`):

| Model | cp lost / move (T2, lower = better) | Elo vs Random = 0 (round robin) |
| --- | --- | --- |
| Team v5 (this repo's base) | 63.2 | +565 |
| Team v3 pilot | 62.3 | +414 |
| Prototype 2 ray masks + 2-ply search | 75.6 | +275 |
| Prototype 2 ray masks | 86.4 | +204 |
| Team C++ engine depth 4 | 46.7 | +927 |

Decision: **v5 is the base.** It is behavioral cloning of the move Stockfish played, with no
value head and no search. v6 (this chunk) adds exactly what the statement asks for and v5 lacks.
Caveat that the core experiment resolves: v5 trained on 12× more positions than Prototype 2.

## Your job: chunk 1 = Phases 0–3 + 2-ply search, then HANDOFF.md

Already written and tested (in the cloud, on CPU) — do not rewrite, run it:

| File | What it does |
| --- | --- |
| `src/chess_policy/v6_model.py` | `V6Policy`: v5's network + attention modes `soft` (= v5 exactly; v5 checkpoints load unchanged), `plain`, `hard` (head h sees only geometry h%4: rook / bishop / knight / king, plus v5's learned biases inside), `hard_ray` (rook/bishop lines stop at the first piece), `hard_only` (masks, no learned biases) + optional value head (tanh, state token). `attention_maps()` for figures. |
| `src/chess_policy/v6_data.py` | `positions` (v5's exact splits: reproduces 279,365 / 22,341 / 25,319 and holdout 103,772), `label` (Stockfish scores every legal move, multiprocess, chunked, resumable, `--stride`, `--shard/--of` for several machines), `assemble` (pre-encoded tensors), `concat`, `V6Dataset`. |
| `src/chess_policy/v6_train.py` | One trainer for every objective: `--policy-target played` (A, v5's objective) or `soft` (B), `--value-weight 1` (C). Same optimiser for all. Keeps the epoch with the lowest **dev** centipawn loss. bf16 autocast on CUDA. |
| `src/chess_policy/v6_eval.py` | cp lost / best move / blunders / agreement with 95% game-bootstrap intervals; `--search NAME` = 2-ply search with the value head; `--reference v5` = paired differences. |
| `src/chess_policy/v6_experiments.py` | Runs the plan stage by stage, skips finished runs, writes `results/v6/SUMMARY.md`. |
| `tests/test_v6.py` | 11 tests (the Stockfish ones need `$env:STOCKFISH`). v5's own tests: 30 pass, 1 skipped. |
| `benchmarks/t2_p2_1312/` | T2: the 18 Nov benchmark (1,312 positions, 24 games, Stockfish 15.1 depth 12, every legal move scored). |
| `scripts/v6_extract_games.py`, `scripts/v6_convert_p2_benchmark.py` | Helpers (D2 games; how T2 was made). |
| `geochesszero/` | Prototypes 1–4 for the record + the tournament harness (`prototypes/03_method_comparison/tournament.py`, `players.py` now has `v6:` and `uci:...:elo=` players, `tournament_v6_template.json`). |

v5's original files (`model.py`, `data.py`, `training.py`, ...) are untouched. Keep it that way:
extend in `v6_*.py`.

### Step 0 — environment (PowerShell, from this folder)

```powershell
conda create -n geochess python=3.12 -y
conda activate geochess
# CUDA PyTorch: take the exact command from https://pytorch.org/get-started/locally/
# (Stable, Windows, Pip, Python, CUDA 12.x), e.g. pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import os; print(os.cpu_count())"
```

Stockfish **15.1** (the version that produced every existing label; never mix versions): the
Windows build from the official release page
https://github.com/official-stockfish/Stockfish/releases/tag/sf_15.1 (the x86-64 avx2 zip;
fall back to the plain x86-64 build if the CPU lacks AVX2). Put the exe at
`tools/stockfish/stockfish.exe` and check that it answers `uci` with `id name Stockfish 15.1`.

```powershell
$env:STOCKFISH = "tools/stockfish/stockfish.exe"
python -m pytest tests -q          # expect v5: 30 passed 1 skipped; v6: 11 passed
python -m chess_policy.v6_eval --data benchmarks/t2_p2_1312 --model v3=checkpoints/pilot.pt `
    --model v4=checkpoints/policy_v4.pt --model v5=checkpoints/policy_v5.pt --device cpu
```

**Gate 0:** v3 62.3, v4 60.5, v5 63.2 cp lost (exact on `--device cpu`; on CUDA bf16 may move
these by a fraction). `git init` + first commit if there is no repo yet.

### Step 1 — data (Phase 1). Labelling is CPU-bound and takes hours: run it in the background, log to a file, keep the GPU busy meanwhile.

Measured in the cloud: depth 10 ≈ 0.24 s per position per core. With N cores, use
`--workers N-1` (or N-2 while training runs). Every `label` command is resumable: rerun the
same command after an interruption.

```powershell
python -m chess_policy.v6_data positions data/fishtest_train.pgn.gz --out data/v6/positions_train.npz
#   expect train 279365 / dev 22341 / test 25319
python -m chess_policy.v6_data positions data/fishtest_holdout.pgn.gz --out data/v6/positions_holdout.npz --force-test --exclude data/v6/positions_train.npz
#   expect 103772
python -m chess_policy.v6_data assemble data/v6/positions_train.npz --split train --out data/v6/train_nolabel
python -m chess_policy.v6_data assemble data/v6/positions_holdout.npz --split test --out data/v6/t3_holdout

# dev first (~22K positions), then train (~279K: about 2.5 h on 8 cores)
python -m chess_policy.v6_data label data/v6/positions_train.npz --splits dev --depth 10 --out data/v6/labels_dev_d10 --engine $env:STOCKFISH
python -m chess_policy.v6_data assemble data/v6/positions_train.npz --split dev --labels data/v6/labels_dev_d10 --out data/v6/dev_d10
python -m chess_policy.v6_data label data/v6/positions_train.npz --splits train --depth 10 --out data/v6/labels_train_d10 --engine $env:STOCKFISH
python -m chess_policy.v6_data assemble data/v6/positions_train.npz --split train --labels data/v6/labels_train_d10 --out data/v6/train_d10

# T1: every 10th holdout position (~10,400, all 800 games) at depth 16.
# Time the first chunk; if the ETA is over ~4 h use --depth 14 and say so in HANDOFF.md.
python -m chess_policy.v6_data label data/v6/positions_holdout.npz --splits test --stride 10 --depth 16 --out data/v6/labels_t1_d16 --engine $env:STOCKFISH
python -m chess_policy.v6_data assemble data/v6/positions_holdout.npz --split test --labels data/v6/labels_t1_d16 --out data/v6/t1_d16
```

**Gate 1:** counts match; `missing` (moves MultiPV skipped and that were scored singly) is
tiny; spot-check 3 positions against Stockfish directly. Never train or select on T1/T3.

### Step 2 — core experiment (Phase 3)

```powershell
python -m chess_policy.v6_experiments --stage baseline   # A x3; needs only dev_d10 + train_nolabel
python -m chess_policy.v6_experiments --stage tau        # soft targets at tau 50/100/200
python -m chess_policy.v6_experiments --stage core       # B x3, C x3 with the chosen tau
python -m chess_policy.v6_experiments --stage attention  # plain / hard / hard_ray / hard_only x3
python -m chess_policy.v6_experiments --stage moe        # C without experts x3
python -m chess_policy.v6_experiments --stage eval       # T2 + T1 (+ search for value models) -> results/v6/SUMMARY.md
```

Before the stages, time one short run (`python -m chess_policy.v6_train ... --epochs 1`). If an
S-size run (6 epochs on 279K) would take over ~15 min, check `nvidia-smi`: the bottleneck is
probably numpy batch building on the CPU (labelling competes for cores). Fix it once for all
runs (e.g. batch 512 with lr 1.5e-3), never differently per objective.

**Gate 2 (the headline result):** does B or C lose fewer centipawns than A, with the paired
intervals? Report it honestly either way — a null result is still a result.

### Step 3 — D2 and the scaling curve (overnight)

```powershell
python scripts/download_source.py        # pinned Fishtest archive -> data/source.pgn.gz (check size first)
python scripts/v6_extract_games.py data/source.pgn.gz --skip 4100 --count 6500 --out data/fishtest_extra.pgn.gz
python -m chess_policy.v6_data positions data/fishtest_extra.pgn.gz --out data/v6/positions_extra.npz --exclude data/v6/positions_train.npz --exclude data/v6/positions_holdout.npz
python -m chess_policy.v6_data label data/v6/positions_extra.npz --splits train --depth 10 --out data/v6/labels_extra_d10 --engine $env:STOCKFISH
python -m chess_policy.v6_data assemble data/v6/positions_extra.npz --split train --labels data/v6/labels_extra_d10 --out data/v6/extra_d10
python -m chess_policy.v6_data concat data/v6/train_d10 data/v6/extra_d10 --out data/v6/big_d10
python -m chess_policy.v6_experiments --stage scaling
python -m chess_policy.v6_experiments --stage eval
```

Games 1–2500 are training, 2501–3300 the T3 holdout, 3301–4100 were v4/v5 holdouts, so extra
games start at 4101. If the archive has fewer games than needed, use what it has and say so; a
second archive from https://huggingface.co/datasets/official-stockfish/fishtest_pgns is the
fallback (ask the user first if it is large).

### Step 4 — search and a first round robin

`--stage eval` already scores 2-ply search for every value-head model. Then copy
`geochesszero/prototypes/03_method_comparison/tournament_v6_template.json`, replace
`REPLACE_ME` with the best run, and from that folder:

```powershell
python tournament.py --config <your copy>.json --openings 8 --workers <cores-2> --out results/tournament_v6.json
```

Paths inside player specs must not contain `:` (Windows drive letters): keep them relative.
Stockfish 15.1's `UCI_Elo` range is 1350–2850. The team C++ engine
(`engines/november_adapter`) needs a C++ compiler; skip it if none is installed.

### Step 5 — HANDOFF.md (required, at the end)

Write `HANDOFF.md` at the repo root for the two teammates who continue:

1. What was done, phase by phase, and every decision taken at a gate (and why).
2. Results: the SUMMARY tables, the headline answer (A vs B vs C), attention ablation,
   scaling curve, search, tournament; every checkpoint with its scores.
3. Where every file lives (data, labels, checkpoints, logs, results) and their sizes.
4. Exact commands to rerun anything; environment (Python, PyTorch, CUDA, driver, Stockfish
   build, CPU core count); label depths actually used.
5. **Chunk 2 (teammate 1):** MCTS with PUCT, oracle-labelled self-play rounds,
   AlphaZero-style targets, big-to-small distillation — first steps and what numbers to expect.
   **Chunk 3 (teammate 2):** full Phase 5 evaluation (Elo pool incl. Stockfish Elo levels,
   T1 cp loss, calibration, search-scaling curve, LLM baseline via Ollama), report, slides,
   demo (v5's UCI/HTTP adapters work for v6 via `load_any`).
6. Known issues, anything unfinished, and how to share checkpoints (`runs/` is git-ignored:
   zip `runs/v6/*/best.pt` + `metrics.json` + `config.json` for the teammates).

Then tell the user it is ready, and that they can paste HANDOFF.md into the Claude app chat
so it gets added as a tab to the roadmap doc.

## Ground rules

- One Stockfish version (15.1) for every label; record the depth used for each dataset.
- Selection only on dev; T1/T2/T3 are for reporting.
- Identical hyperparameters across objectives in the core comparison.
- Commit after each step (code + `results/`), with a one-line message of what changed.
- If something breaks, fix it in the v6 files and add a test for it.
- Rough timing on this machine is unknown: measure, then tell the user the real ETA.
