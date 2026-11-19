# GeoChess — instructions for Claude Code (chunks 2 and 3)

Read this whole file, then `HANDOFF.md` in full, before doing anything. Together they replace
the memory of the earlier sessions. You start without that memory, so do not guess: if
something here or in HANDOFF.md is unclear, ask the person you work for.

## Status (19 Nov 2026)

**Chunk 1 is finished.** The team lead ran it with Claude Code on a Windows PC (RTX 3060):
environment, Stockfish 15.1 labels for ~1.3M positions, 39 training runs (the core experiment
A/B/C, the attention ablation, MoE, the scaling curve up to ~1M positions), evaluation on
T1/T2, and a first round robin. Everything it did, decided and found is in `HANDOFF.md`.
Its original briefing is kept for the record in `docs/CLAUDE_chunk1.md`; **do not run its
steps again**.

Key numbers (details and intervals in HANDOFF.md):

| | v5 (old base) | best v6: `runs/v6/scale_L_1M/best.pt` |
| --- | --- | --- |
| T1 cp lost / move (10,378 positions, depth 14) | 101.7 | **49.2** |
| T2 cp lost / move (1,312 positions, depth 12) | 63.2 | **39.4** (CPU) / 40.0 (CUDA) |
| Round robin Elo (Random = 0) | +680 | **+1032** (vs v5: +13 =3 −0) |

Gate 2 answer: soft Stockfish targets (B) and soft targets + value head (C) both beat
behavioural cloning (A) on T1; the geometry-masked heads (`hard_ray`) are the largest single
gain; **2-ply search with the current value head makes play worse** — fixing that is chunk 2's
first job.

## Who you are working for

A 3-person team, Generative AI university course project. The person you work for is
**teammate 1 (chunk 2)** or **teammate 2 (chunk 3)** — ask which if they have not said.

* **Chunk 2:** fix/calibrate the value head, MCTS with PUCT, oracle-labelled self-play,
  AlphaZero-style targets, big-to-small distillation. Plan and first steps: HANDOFF.md §5.
* **Chunk 3:** full Phase 5 evaluation (Elo pool with Stockfish UCI_Elo levels, full T1 pass,
  value calibration, search-scaling curve, LLM baseline via Ollama), report, slides, demo.
  Plan: HANDOFF.md §5.

Project statement: *chess as next-move prediction over a 64-node multi-relational graph,
where attention heads are masked to rook, bishop, knight and king movement geometries. The
objective is value-optimal play, not behavioural cloning of human moves. The architecture
remains isomorphic to GPT.* Out of scope: fine-tuning an LLM on chess books, the GPT
move-sequence model, the hand-crafted evaluator.

## Setting up (new machine)

1. Clone this repo. Python 3.12, then PyTorch with CUDA from pytorch.org, then
   `pip install -e ".[dev]"` (exact versions used: HANDOFF.md §4).
2. Get the three zips from the team's Google Drive folder and **unzip each at the repo root**
   (their paths already start with `runs/`, `data/` or `tools/`):
   * `GeoChess_v6_checkpoints.zip` → `runs/v6/<run>/best.pt, config.json, metrics.json`
     for all 39 runs (`load_v6` checks each `best.pt.sha256`).
   * `GeoChess_v6_data.zip` → `data/v6/positions_*.npz` and every Stockfish label folder
     `data/v6/labels_*` (~16 CPU-hours of Stockfish; **never relabel them**).
   * `GeoChess_stockfish_15.1_windows_avx2.zip` → `tools/stockfish/stockfish.exe` (Windows).
     On Linux/macOS build Stockfish **15.1** from the official source tag `sf_15.1`
     (`make -j build ARCH=x86-64-avx2` in `src/`). Either way it must answer
     `id name Stockfish 15.1` and its `bench` must report **3467381** nodes.
3. Rebuild the training tensors from positions + labels (a few minutes, no Stockfish):

   ```bash
   export PYTHONPATH=src   # PowerShell: $env:PYTHONPATH = "src"
   python -m chess_policy.v6_data assemble data/v6/positions_train.npz --split dev --labels data/v6/labels_dev_d10 --out data/v6/dev_d10
   python -m chess_policy.v6_data assemble data/v6/positions_train.npz --split train --labels data/v6/labels_train_d10 --out data/v6/train_d10
   python -m chess_policy.v6_data assemble data/v6/positions_train.npz --split train --out data/v6/train_nolabel
   python -m chess_policy.v6_data assemble data/v6/positions_holdout.npz --split test --labels data/v6/labels_t1_d14 --out data/v6/t1_d14
   python -m chess_policy.v6_data assemble data/v6/positions_holdout.npz --split test --out data/v6/t3_holdout
   python -m chess_policy.v6_data assemble data/v6/positions_extra.npz --split train --labels data/v6/labels_extra_d10 --out data/v6/extra_d10
   python -m chess_policy.v6_data concat data/v6/train_d10 data/v6/extra_d10 --out data/v6/big_d10
   ```

   Expected rows: dev 22,341 · train 279,365 · t1 10,378 · t3 103,772 · extra 707,896 ·
   big 987,261.
4. Check the setup before any new work — all three must match:
   * `STOCKFISH=<path to stockfish> python -m pytest tests -q` → **45 passed, 1 skipped**.
   * `python -m chess_policy.v6_eval --data benchmarks/t2_p2_1312 --model v5=checkpoints/policy_v5.pt --model best=runs/v6/scale_L_1M/best.pt --device cpu`
     → v5 **63.2**, best **39.4** cp lost (exact on CPU; CUDA bf16 moves them by a fraction).
   * `python -m chess_policy.v6_eval --data data/v6/t1_d14 --model best=runs/v6/scale_L_1M/best.pt`
     → **49.2** cp lost (CUDA).

## Ground rules (carried over from chunk 1 — keep them)

* **One Stockfish version, 15.1, for every label.** Record the depth of every new dataset.
  Existing depths: dev/train/D2 10, T1 14, T2 12, T3 unlabelled.
* **Select only on dev** (`data/v6/dev_d10`). T1, T2 and T3 are for reporting only — never
  train on them, tune on them or pick checkpoints with them.
* Compare methods with **identical hyperparameters**, **3 seeds**, and **paired intervals**
  (`scripts/v6_paired_groups.py`). Training is not bit-for-bit reproducible on GPU: two
  identical runs differed by 1.5 cp, so differences under ~2 cp are noise.
* **Do not modify or delete chunk 1's runs** (`runs/v6/*`) or label folders. New runs go in
  new folders (e.g. `runs/v7/` or new run names). Do not overwrite `results/v6/` either:
  write new results next to it (e.g. `results/v7/`).
* v5's original files (`model.py`, `data.py`, `training.py`, …) stay untouched; extend in
  `v6_*.py` or new modules. Do not rewrite working v6 code; fix only what breaks, and add a
  test for every fix.
* CI (`.github/workflows/tests.yml`) runs `ruff check src scripts tests` and `pytest` on every
  push — keep both green.
* Commit after each step with a one-line message; push to `origin master` (private repo
  `HoviGtaV/GeoChess`). Large files (checkpoints, data, zips) never go in git: `data/v6/`,
  `runs/`, `tools/`, `share_drive/` are git-ignored; share them through the Drive folder.
* Ask before downloading anything over ~1 GB or touching anything outside the repo.

## Where things are

| Path | What |
| --- | --- |
| `HANDOFF.md` | **Read first.** Everything chunk 1 did, decided and found; how to rerun; next steps |
| `results/v6/SUMMARY.md` | every run's T2/T1 scores (auto-generated); `t1.json`, `t2.json`, `t1_1M.json`, `gate2_*.json` |
| `src/chess_policy/v6_*.py` | the v6 model, data pipeline, trainer, evaluation, experiment runner |
| `scripts/v6_*.py` | Gate 1/2 helpers, game extraction, T2 conversion, dev summary |
| `geochesszero/prototypes/03_method_comparison/` | tournament harness; `tournament_v6.json`, `results/tournament_v6.json` |
| `logs/` | chunk 1's logs and the exact driver scripts (a record — see `logs/README.md`) |
| `docs/CLAUDE_chunk1.md` | chunk 1's original briefing (historical) |
| `benchmarks/t2_p2_1312` | T2 benchmark (in git) |

## Practical notes

* Windows: give Stockfish an absolute path or a backslash path. Long jobs started from a
  Claude Code session die if the app restarts; start them as separate processes (chunk 1 used
  PowerShell `Start-Process`) and make them resumable. `v6_data label` is resumable and
  re-labels corrupt chunks.
* `v6_eval` is single-threaded; for big passes run several processes in parallel
  (HANDOFF.md §6).
* Labelling speed on the team lead's PC: ~20 positions/s at depth 10 with 10 Stockfish
  processes (i5-11400, 6 cores).
* Keep updates to the person short and plain: what finished, what is running, what you
  found, how long until the next result.
