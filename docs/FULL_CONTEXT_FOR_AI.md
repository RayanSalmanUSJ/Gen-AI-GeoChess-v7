# Full context for the next AI assistant (written by the AI that ran chunk 1)

You are picking up a project another Claude Code session worked on for two days. You do not
have its memory. This file is that memory: what it knew, what it decided and why, what it
checked, what it got wrong, and the traps it hit. Read it together with `CLAUDE.md` (rules and
setup) and `HANDOFF.md` (results tables, commands, file sizes). Where the three disagree,
`HANDOFF.md` has the numbers and this file has the reasons; tell the person if you find a
contradiction rather than guessing.

---

## 1. The people

* **Team of 3**, a Generative AI university course project. The **team lead** (GitHub
  `HoviGtaV`, owns the repo and the Drive folder) ran chunk 1 with Claude. **Teammate 1 does
  chunk 2, teammate 2 does chunk 3.** You are working for one of the two teammates. Ask which
  chunk if they have not said, and ask how they like to work. The team lead wanted: short,
  plain updates (what finished, what is running, what you found, how long is left); no walls
  of logs; big chunks done properly; no scope creep; long jobs in the background with logs;
  ask before downloads over ~1 GB or touching anything outside the project folder; and
  "do what you think is best, but tell me at the end" for judgement calls. Do not assume your
  person wants the same: ask once.
* The team lead is not deeply technical. Explain decisions in plain words.

## 2. The project in one paragraph

Chess as **next-move prediction over a 64-node multi-relational graph**, where attention heads
are **masked to rook, bishop, knight and king movement geometries**; the objective is
**value-optimal play** (learn from Stockfish's evaluation of every legal move), not
behavioural cloning of the move played; the architecture stays isomorphic to GPT. The team's
paper (`projet_data.pdf`, **not in the repo**; ask the person for it if you need the equations)
defines a 4-phase curriculum: behavioural cloning → engine distillation with soft targets
q(m) ∝ exp(cp(m)/τ) → self-play → oracle distillation, plus dual policy/value heads, search,
and an evaluation protocol (Elo vs a fixed pool, centipawn loss, blunder rate, value
calibration, search-scaling curve). A roadmap (Phases 0–6) lives at
https://claude.ai/artifact/Kz4qUXjPHppPxYD59FPw7w (a claude.ai page; you may not be able to
open it from Claude Code: ask the person to paste it if you need it).

Out of scope (decided before chunk 1): fine-tuning an LLM on chess books, the GPT
move-sequence model (a dead end), the hand-crafted evaluator.

## 3. History before chunk 1

* **v3 → v4 → v5**: the team's earlier models (`README*.md`, `RESULTS*.md`, `v4_*`, `v5_*`
  folders). v5 = behavioural cloning of the move Stockfish played, no value head, no search.
  `checkpoints/pilot.pt` = v3, `policy_v4.pt` = v4, `policy_v5.pt` = v5.
* **Prototypes 1–4** in `geochesszero/prototypes/` (Prototype 2 = "ray masks" + value
  distillation; Prototype 3 = method comparison + tournament harness; Prototype 4 = the
  18 Nov cross-evaluation, `04_crosseval/RESULTS.md`).
* Cross-eval of 18 Nov (T2, cp lost/move): v5 63.2, v3 62.3, P2 ray masks 86.4 (75.6 with
  2-ply search), **team C++ engine depth 4: 46.7** (+927 Elo vs Random in that pool).
  Decision: **v5 is the base** for v6; v6 adds exactly what the project statement asks for and
  v5 lacks (value-optimal targets, geometry masks as an option, value head, search).
* v6 code (`src/chess_policy/v6_*.py`) was written and unit-tested in the cloud on CPU before
  chunk 1; chunk 1's job was to run it on the real data and GPU, fixing only what broke.

## 4. What v6 is, precisely

**Model** (`v6_model.py`, `V6Policy`): v5's network (64 square tokens + a state token,
learned Q/K/V attention with geometry and attack relations, SwiGLU feed-forward, top-2 mixture
of experts every 2nd block, `expansion` 3, dropout 0) with:

* `attention` modes: `soft` (= v5 exactly; v5 checkpoints load unchanged), `plain` (no
  geometry), `hard` (head h only attends along geometry h % 4: rook, bishop, knight, king;
  v5's learned biases kept inside the mask), `hard_ray` (rook/bishop lines stop at the first
  piece), `hard_only` (masks without the learned biases).
* optional **value head** (tanh, read from the state token).
* sizes used: **S** width 64 / 2 layers / 4 heads (307K params, lr 1e-3), **M** 128/6/8
  (2.93M, lr 6e-4), **L** 256/6/8 (11.6M, lr 4e-4).

**Training** (`v6_train.py`, one trainer for every objective): AdamW (weight decay 0.01),
300 warmup steps then cosine, gradient clip 1.0, batch 256, bf16 autocast on CUDA, MoE
balance loss 0.01 and z-loss 0.001. 6 epochs (4 for the 1M runs).

* `--policy-target played` = **A** (behavioural cloning, v5's objective).
* `--policy-target soft --tau T` = **B**: cross-entropy with q(m) = softmax(cp(m)/τ) over legal
  moves, cp clipped to ±1000.
* `--value-weight 1` added = **C**: + MSE(V, v_target) with
  v_target = 0.75 · expected_score(best cp) + 0.25 · game result, where
  expected_score(cp) = 2 / (1 + exp(−0.00368208 · cp)) − 1 (Lichess logistic, cp clipped ±2000).
* **Checkpoint selection**: the epoch with the lowest **dev cp loss**, computed on the **first
  20,000 rows** of the 22,341-row dev set (`max_dev_rows` 20000). 36 of the 39 runs
  picked their **last** epoch (the other 3 the second-to-last): the models are under-trained.

**Metrics** (`choice_metrics`, Prototype 2's definitions): per position, **cp lost** =
(best move's cp − chosen move's cp), scores clipped to ±1000, so it is ≥ 0; best-move rate =
cp lost ≤ 0; mistake ≥ 100; blunder ≥ 200. "agree" = the move actually played in the game.
95 % intervals = bootstrap over **whole games** (positions in a game are correlated).

**2-ply search** (`v6_eval.search_choices`): take the policy's top **6** moves; for each, play
every opponent reply; the value head scores each resulting position; a candidate is worth its
worst reply; pick the best candidate. Terminal positions use exact values.

## 5. Data: exactly what exists

| Set | Source | Rows | Labels |
| --- | --- | --- | --- |
| train / dev / test | games 1–2500 of the Fishtest archive (`data/fishtest_train.pgn.gz`), v5's exact splits | 279,365 / 22,341 / 25,319 | train & dev: Stockfish 15.1 **depth 10** |
| T3 holdout | games 2501–3300 (`data/fishtest_holdout.pgn.gz`) | 103,772 | none |
| **T1** | every 10th T3 position, all 800 games | 10,378 | **depth 14** |
| **T2** | Prototype 2's benchmark, 24 held-out games (`benchmarks/t2_p2_1312`, in git) | 1,312 | depth 12 (Prototype 2's labels; target = Stockfish's best move) |
| D2 ("extra") | games 4101–10600 of `data/source.pgn.gz` (pinned Fishtest archive, sha256-checked by `scripts/download_source.py`) | 707,896 (train split only) | depth 10 |
| big ("1M") | train + D2 | 987,261 | depth 10 |

Games 3301–4100 were v4/v5 holdouts and are not used. D2's dev/test-split positions
(126,466) were extracted but not labelled or used.

**How labels were made** (`v6_data.label`): Stockfish 15.1, Threads 1, Hash 16, MultiPV = all
legal moves at fixed depth, moves MultiPV skipped are scored singly (`missing`; it was **0**
everywhere). Positions are processed in chunks (1,000; T1 used 250), one engine per worker
reused across a chunk, so the hash carries over between positions. Consequence: re-scoring a
position with a fresh engine gives per-move scores ~10–25 cp different and sometimes swaps
close best moves. This is normal search noise, not a bug: replaying a chunk in the
labeller's order reproduces the stored scores **exactly**. Never "fix" labels by relabelling.

**Stockfish build**: the official 15.1 Windows binaries are gone from GitHub, so it was built
from the official `sf_15.1` source tag with WinLibs GCC 12.4 (`ARCH=x86-64-avx2`) and the
official net `nn-ad9b42354671.nnue` (embedded in the exe). Identity check: `id name Stockfish
15.1` and `bench` = **3,467,381** nodes (the official 15.1 signature). **Never label with a
different version**: every number in the project assumes 15.1.

## 6. The runs (all in `runs/v6/`, all on the Drive zip)

Naming: `<group>_s<seed>` for 3-seed groups, `<stage>_<size>_<data>` for scaling.

| Runs | What | Objective | Attention |
| --- | --- | --- | --- |
| `A_played_s0..2` | baseline | played (BC) | soft |
| `tau50_s0`, `tau100_s0`, `tau200_s0` | temperature pick | soft | soft |
| `B_soft_s0..2` | core B | soft, τ 100 | soft |
| `C_value_s0..2` | core C | soft τ 100 + value | soft |
| `att_{plain,hard,hard_ray,hard_only}_s0..2` | attention ablation | as C | as named |
| `moe_off_s0..2` | C with 1 expert | as C | soft |
| `scale_{S,M,L}_{70K,140K,279K,1M}` | scaling curve, seed 0 | as C | hard_ray |

All S-size except the M/L scaling runs. **Best run: `scale_L_1M`** (best dev score of all:
44.5; T1 49.2; T2 40.0 on CUDA / 39.4 on CPU; +1032 Elo). `scale_M_1M` is within noise of
it on T2 (38.6) and a bit worse on dev/T1.

## 7. Results you must know (details in HANDOFF.md §2)

* **Gate 2 (T1, paired, 3 seeds):** B − A = −6.8 cp [−8.5, −5.0]; C − A = −2.3 [−4.0, −0.5];
  B − C = −4.5 (the value head costs policy quality). On T2 the three are within noise.
* **Attention (T1, vs C):** hard_ray −10.1, hard −9.0, hard_only −8.5, plain +19.6. The
  geometry masks are the biggest single effect in the project.
* **MoE off:** no significant difference.
* **Scaling (T1):** S/M/L on 279K = 72.1 / 61.3 / 59.9; on ~1M = 65.5 / 50.7 / 49.2.
* **Search hurts:** policy → 2-ply search, T1: L_1M 49.2 → 57.7; hard_ray 71.6 → 78.2; in the
  round robin −93 Elo. Cause: the value head is not accurate enough at the leaves. Value
  calibration was **never measured**.
* **Round robin** (8 openings, 448 games; Elo vs Random = 0): Stockfish depth 1 +1196,
  **v6 best +1032**, Stockfish UCI_Elo 1350 +1012, v6 best + search +939, v5 +680, v3 +534,
  P2 +232. v6 vs v5 head-to-head +13 =3 −0. Intervals are ±150–250 Elo: not precise.
* A (v5's recipe) beats the v5 checkpoint by ~8 cp (T2) / ~17 cp (T1) only because the v6
  trainer runs 6 epochs (v5: 3) and selects on dev cp loss (v5: dev NLL). Always compare new
  methods against **A or C**, not against the v5 checkpoint.
* GPU training is not bit-for-bit reproducible: identical config + seed gave dev 63.0 vs 64.5.
  Treat < ~2 cp as noise; use 3 seeds and `scripts/v6_paired_groups.py`.

## 8. Decisions taken in chunk 1, and why

1. **No Miniconda** (not installed): a `.venv` in the repo with Python 3.12.3 + CUDA PyTorch
   2.14.1+cu126. A second CPU-only venv `tools/venv-cpu` let work start during the 2.6 GB
   download. Both are local to the team lead's PC.
2. **Stockfish built from source** (no official binary left), verified by bench signature.
3. **T1 at depth 14, not 16**: depth 16 measured at ~4.2 h for T1 (> the plan's ~4 h limit),
   depth 14 ~1.8 h.
4. **Batch 256 / lr 1e-3 kept** for every core run: an S run took ~9–11 min, under the plan's
   15-min threshold for changing it.
5. **τ = 100** (dev 72.5 / 72.4 / 73.1 for 50 / 100 / 200: a near tie, one seed each).
6. **hard_ray** chosen for scaling (best dev of the attention modes).
7. **T1 evaluated on a subset** (all seeds of A/B/C, seed 0 of the other groups, all scaling
   runs, search for 3 models + L_1M), because the full pass would take 12 h+ with
   single-threaded search on a shared CPU. The full T1 pass is chunk 3's.
8. **"v6 best" for the round robin = `scale_L_1M`**, chosen by dev score (never by a test set).
9. **Team C++ engine skipped**: its source (`module1..12.cpp`) is not in the repo.
10. **Repo made public** at the team lead's request; the Drive folder is "anyone with the link,
    view only".

## 9. Bugs found and fixed in chunk 1 (all with tests)

1. `v6_data.label`: a relative engine path with `/` cannot be started by Windows
   CreateProcess, and a pool initializer that fails is respawned forever (silent hang). Now:
   absolute path + one engine started up front to fail fast.
2. `v6_experiments.write_summary`: crashed on Windows (cp1252 cannot encode "Δ"); now UTF-8.
3. `v6_data.label`: a reboot left a chunk renamed into place before its data hit the disk
   (invalid .npz). Now chunks are fsynced before the rename, and a resumed `label` re-labels
   unreadable chunks.
4. CI was red (ruff: semicolons, ambiguous `l`, an unused import). Fixed; CI is green.

## 10. Traps (read these before touching anything)

* **`v6_experiments` is hard-wired to `runs/v6` and `results/v6`.** Every run in
  `runs/v6/*/metrics.json` is picked up by `--stage eval`, which **overwrites**
  `results/v6/t2.json`, `t1.json` and `SUMMARY.md`. Do not put new runs in `runs/v6` and do not
  run `--stage eval` without deciding where the results go. For new work, either add
  `--runs`/`--results` options to `v6_experiments.py` (with a test) or call `v6_train` /
  `v6_eval` directly with your own `--out` paths (e.g. `runs/v7/`, `results/v7/`).
* `--stage scaling` and `--stage attention` read the attention/τ choices from existing runs
  (`choose_tau`, `choose_attention`). Adding runs with those names changes the choices.
* `--t1` defaults to `data/v6/t1_d16`, which **does not exist** (T1 is depth 14). Always pass
  `--t1 data/v6/t1_d14`, or T1 is silently skipped.
* `run()` skips any run whose `metrics.json` exists, so re-running a stage is a no-op for
  finished runs. To redo one, use a new name; never delete chunk 1's runs.
* `results/v6/SUMMARY.md`'s T1 table is only the subset; the 1M runs' T1 numbers are in
  `results/v6/t1_1M.json`; the paired Gate 2 numbers in `gate2_t1.json` / `gate2_t2.json`.
* **The demo adapters do not load v6 models yet.** `chess_policy.uci` / `chess_policy.api` go
  through `inference.Policy`, which uses v5's `load_checkpoint`; on a v6 checkpoint it raises
  "checkpoint format mismatch" (checked on `scale_L_1M`). The chunk-1 plan claimed they work
  "via `load_any`": that was never true. Fix: a small v6 wrapper that uses
  `v6_model.load_any` (keep v5's `inference.py` untouched), plus a test. This is chunk 3's
  (demo) job.
* The tournament harness (`geochesszero/prototypes/03_method_comparison/tournament.py`) is run
  **from its own folder**; player spec paths must be relative and must not contain `:`
  (Windows drive letters break the `kind:path:mode` spec). Specs: `random`,
  `v5:<root>:<ckpt>`, `board:<P2 ckpt>:policy`, `v6:<ckpt>:policy|search[:k=6]`,
  `uci:<exe>:depth=<d>`, `uci:<exe>:elo=<1350..2850>` (0.1 s/move). `tournament_v6.json` is
  the config used.
* `v6_eval` is **single-threaded** (~13 % CPU, ~8 % GPU): the search builds and encodes every
  reply in Python. Parallelise big passes with several processes (HANDOFF.md §6).
* `V6Dataset` loads whole arrays; `big_d10` is 1.2 GB. Fine with 32 GB RAM.
* **Windows**: long jobs started from a Claude Code session die if the app restarts (it did,
  at 05:42 on 19 Nov, probably a Windows-update reboot). Start long jobs as separate
  processes (PowerShell `Start-Process`) and keep them resumable. Stopping a background Git
  Bash job does not kill its native children (stray Stockfish/Python processes keep running):
  kill with `taskkill /F /T /PID <winpid>` and check with `tasklist` before relaunching
  anything that writes to the same folder. Give Stockfish a backslash or absolute path.
* Line-ending warnings ("LF will be replaced by CRLF") are harmless. The team lead's clone has
  `core.filemode false` because Windows checkouts flip the executable bit on 3 `.sh` files.

## 11. Machine facts (team lead's PC; yours will differ, so measure)

RTX 3060 12 GB, driver 610.88; i5-11400, 6 cores / 12 threads (AVX2); 32 GB RAM; Windows 11;
drive C: nearly full (~13 GB free). Timings: Stockfish depth 10 ≈ 20 positions/s with 10
processes (~0.47 s per position per process); depth 14 ≈ 5.6 s, depth 16 ≈ 13 s per position
per process. Training on 279K: S ~9 min, M ~20, L ~23 (6 epochs); on ~1M: S 8, M 28, L 45
(4 epochs). One epoch of S on 279K ≈ 1.8 min. T2 eval of all 72 model/search entries ≈ 1.5 h;
448-game round robin with 8 workers ≈ 53 min.

## 12. What I would do next (if I were continuing)

**Chunk 2** (start from `runs/v6/scale_L_1M/best.pt`; use `scale_S_1M` for fast iteration):
1. Measure value calibration on dev (predicted vs `cp_to_value(best cp)` and vs result).
2. Make 2-ply search beat the policy **on dev** before anything else: value weight 2–4,
   `value_mix` 0, a separate value MLP, more epochs. Compare with paired intervals.
3. Only then PUCT MCTS (c_puct 1.5–2.5, 50–400 simulations), self-play games labelled by
   Stockfish 15.1 depth 10 (`v6_data label` is resumable and shardable with `--shard/--of`),
   AlphaZero-style targets, and distillation L → S (targets: S on Stockfish labels 65.5 cp on
   T1, teacher L 49.2).

**Chunk 3**: a v6 demo adapter (trap above); the full T1 pass in parallel processes; an Elo
pool with Stockfish UCI_Elo 1350/1600/1800/2000 + v5 + the best B/C/hard_ray/S/M/L + chunk 2's
MCTS player with 16–32 openings; value calibration plot; search-scaling curve; LLM baseline via
Ollama (`03_method_comparison/llm_player.py`); attention-map figures with
`V6Policy.attention_maps()` (the hard_ray masks make a good slide); report and slides.

## 13. How to start, concretely

1. `CLAUDE.md` → "Setting up": clone, unzip the three Drive zips at the repo root, rebuild the
   tensors, run the three checks (45 passed / 1 skipped; T2 CPU v5 63.2 & best 39.4; T1 49.2).
   Tested end to end from a fresh clone on 19 Nov: data byte-identical, numbers exact.
2. Tell the person the checks passed, then agree the plan for their chunk before running
   anything long.
3. Commit after each step; keep CI green; never overwrite chunk 1's runs, labels or results.

Links: repo https://github.com/HoviGtaV/GeoChess (public) · Drive
https://drive.google.com/drive/folders/1Cjq4l4tPoTfZXwlZTNJubaNhhkj3J7ep (anyone with the
link) · roadmap https://claude.ai/artifact/Kz4qUXjPHppPxYD59FPw7w.
