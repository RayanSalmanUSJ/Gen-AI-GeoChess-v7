# GeoChess v6 — HANDOFF after chunk 1 (Phases 0–3 + 2-ply search)

Written 19 Nov 2026 at the end of chunk 1, for the two teammates who continue with chunk 2
(teammate 1) and chunk 3 (teammate 2). Everything below was run on the team lead's Windows PC.
`CLAUDE.md` is the original plan; this file says what actually happened.

**Headline.** Value-optimal training beats behavioural cloning on the large test set, and the
geometry-masked attention heads are the biggest single win:

* **B (soft Stockfish targets) − A (behavioural cloning) = −6.8 cp lost per move [−8.5, −5.0]**
  on T1 (10,378 positions, 800 games, paired, averaged over 3 seeds).
* **C (soft targets + value head) − A = −2.3 cp [−4.0, −0.5]**: the value head costs about
  4.5 cp of policy quality versus B, but it is what makes search possible.
* **Hard rook/bishop/knight/king masks, rays stopping at the first piece − C = −10.1 cp
  [−11.9, −8.2]**; plain (unmasked) attention is +19.6 cp worse than C.
* On the small T2 (1,312 positions, 24 games) A, B and C are within noise of each other; the
  mask results hold there too.
* Scaling: more data and bigger models both help a lot. **The best model, `scale_L_1M`
  (L size, hard_ray, ~1M positions), loses 49.2 cp/move on T1 against v5's 101.7 — half —
  with 5.8 % blunders against 15.3 %.**
* Round robin: **v6 best +1032 Elo vs v5 +680 (Random = 0); head-to-head vs v5 +13 =3 −0**;
  level with Stockfish 15.1 at UCI_Elo 1350.
* **2-ply search with the value head makes every model worse** (by 1–10 cp on T1, and ~90 Elo
  in the round robin): the value head is not yet good enough to steer moves. That is the first
  thing chunk 2 (MCTS) has to fix.

---

## 1. What was done, phase by phase, and the decisions taken at each gate

### Step 0 — environment (18 Nov)

* GPU RTX 3060 12 GB (driver 610.88), CPU Intel i5-11400 (**6 cores / 12 threads**), 32 GB RAM,
  Windows 11. Drive C: had only 15–24 GB free throughout; keep an eye on it.
* **Decision: no Miniconda.** It was not installed. Instead a venv inside the repo, `.venv`,
  made with the existing Python 3.12.3, with CUDA PyTorch 2.14.1+cu126. A second, CPU-only venv
  `tools/venv-cpu` (PyTorch CPU) was made so tests, data prep and labelling could start while the
  2.6 GB CUDA wheel downloaded over a slow connection (~250 KB/s per connection; it was
  fetched over 12 parallel ranged connections and checked against PyTorch's sha256).
* **Decision: Stockfish 15.1 built from the official source.** The GitHub release page for
  sf_15.1 no longer carries Windows binaries. Built from the `sf_15.1` source tag with a portable
  GCC 12.4 (WinLibs, in `tools/gcc`, never installed system-wide) and the official net
  `nn-ad9b42354671.nnue` (sha256 prefix checked):
  `make all ARCH=x86-64-avx2 COMP=mingw` from Git Bash. It answers `id name Stockfish 15.1` and
  its `bench` node count is **3,467,381**, the official 15.1 signature, so the search is
  identical to the official build.
* Tests: v5 30 passed + 1 skipped; v6 now 15 (11 original + 4 added for the Windows fixes below).
  Full suite: **45 passed, 1 skipped** with `STOCKFISH` set.
* **Gate 0 passed exactly** on `--device cpu`: v3 62.3, v4 60.5, v5 63.2 cp lost on T2. On CUDA
  (bf16) they come out 62.1 / 60.6 / 63.0, the "fraction" CLAUDE.md expected.

### Step 1 — data and labels (18–19 Nov)

* Positions reproduce v5's splits exactly: **train 279,365 / dev 22,341 / test 25,319**, holdout
  **103,772**.
* Labelling speed on this CPU: **~20 positions/s with 10 Stockfish processes at depth 10**
  (~0.47 s per position per process, slower than the cloud's 0.24 s because 6 real cores serve
  12 threads). Train labels took ~4 h.
* **Gate 1 passed.** `missing` (moves MultiPV skipped and that were scored singly) was **0** for
  dev, train, T1 and D2. Replaying the first 8 dev positions the way the labeller does (one
  engine, in order) reproduces the stored scores **exactly**, so labels line up with positions.
* **Label noise to know about:** re-scoring a position with a *fresh* engine gives per-move scores
  10–25 cp different from the stored labels and sometimes a different best move between close
  moves, because the hash table carries over between positions. Changing only the Hash size does
  the same. This is ordinary depth-10 search noise, not a bug. The same holds against T2
  (made by Prototype 2 with Hash 32): same best move in the 3 spot-checked positions, scores
  within ~10–25 cp.
* **Decision: T1 at depth 14, not 16.** Measured on 20 T1 positions under load: depth 16 =
  13.3 s/position (→ ~4.2 h with 9 workers, over CLAUDE.md's ~4 h limit), depth 14 = 5.6 s
  (→ ~1.8 h). T1 = every 10th holdout position, **10,378 positions from 800 games, depth 14**.
  T1 was labelled with `--chunk 250` so the 9 workers stay busy (11 chunks of 1,000 would leave
  7 of them idle for half the time).
* **Fix 1 (v6_data.label):** on Windows, a relative engine path written with `/` (CLAUDE.md's
  `tools/stockfish/stockfish.exe`) cannot be started by CreateProcess, and a pool initializer that
  fails is respawned forever, so labelling hung silently. `label()` now makes the path absolute
  and starts one engine up front to fail fast. Tests:
  `test_label_bad_engine_fails_fast`, `test_label_forward_slash_relative_engine_path`.

### Step 2 — core experiment (18–19 Nov)

* Timing: one epoch of the S model on 279K positions = **~1.8 min on the GPU** (GPU 20–35 % busy:
  batch building on a CPU shared with Stockfish is the bottleneck). A 6-epoch S run is ~9–11 min,
  under the 15 min threshold, so **the default batch 256 / lr 1e-3 was kept for every run**.
* Stages run: baseline (A ×3), tau (50/100/200), core (B ×3, C ×3), attention (plain / hard /
  hard_ray / hard_only ×3, all with C's objective), moe (C without experts ×3), scaling
  (S/M/L × 70K/140K/279K). 36 runs, all on CUDA.
* **tau = 100 chosen** on dev (dev cp lost 72.5 / 72.4 / 73.1 for tau 50 / 100 / 200 — a near tie).
* **attention = hard_ray chosen** for scaling on dev (65.1 vs hard 65.6, hard_only 67.0,
  soft 74.0, plain 92.2).
* **Fix 2 (v6_experiments.write_summary):** writing SUMMARY.md crashed on Windows (cp1252 cannot
  encode the "Δ" in the paired table). Now written as UTF-8; test
  `test_write_summary_handles_non_ascii`.
* **Eval decision:** the full T1 pass (66 model/search entries × 10,378 positions, with the CPU
  shared with D2 labelling) would have taken 12 h+. So T2 scores **every** run (stage eval), and
  T1 scores a subset: v3/v4/v5, all seeds of A/B/C, seed 0 of each attention/moe group, all 9
  scaling runs, and search for C_value_s0, att_hard_ray_s0, scale_L_279K.
* **Gate 2 resolved** with `scripts/v6_paired_groups.py` (per-position cp loss averaged over the 3
  seeds of each group, paired differences bootstrapped over whole games): see section 2.1.
* Note: A (v5's objective on v5's positions, trained by the v6 trainer) loses ~8 cp less than the
  v5 checkpoint on T2 and ~17 cp less on T1: the v6 trainer runs 6 epochs (v5: 3) and keeps the
  epoch with the lowest dev cp loss (v5: lowest dev NLL). So compare B and C with **A**, not v5.

### Step 3 — D2 and the scaling curve (19 Nov)

* `data/source.pgn.gz` (159,278,139 bytes, sha256 checked by `download_source.py`) has 10,600
  games; games 4101–10600 (6,500) were extracted. After excluding train/holdout positions:
  **834,362 positions, of which 707,896 in the train split** (the dev/test-split positions of
  D2 are not used). Labelled at depth 10. `big_d10` = train_d10 + extra_d10 ≈ **987K positions**
  (the "1M" runs).
* **Fix 3 (v6_data.label):** at the 05:42 reboot one D2 chunk (`chunk_00297.npz`) was left as a
  file that was not a valid .npz (renamed into place before its data reached the disk), and
  `assemble` crashed at 17:00. Chunks are now fsynced before the rename, and a resumed `label`
  re-labels any unreadable chunk. Test: `test_label_resume_relabels_corrupt_chunk`. All other
  label folders were checked: no other bad chunk.
* 1M runs (S/M/L, 4 epochs) on `big_d10`, hard_ray, tau 100, value weight 1: section 2.4.
  `scale_L_1M` has the best dev score of every run (44.5).

### Step 4 — search and a first round robin

* `--stage eval` scored 2-ply search for every value-head model (section 2.3).
* Round robin with `scale_L_1M` as "v6 best" (section 2.5), 8 openings, 8 workers (not 10:
  the T2 re-score was running at the same time). The team C++ engine was **skipped**: its source
  (`module1..module12 .cpp`, needed by `engines/november_adapter/build.sh`) is not in this repo.
  A compiler is now available in `tools/gcc` if someone adds the source.

### Things that went wrong (and are fixed)

* The night of 18 Nov a replaced wait-script was not really stopped (on Windows, stopping a Git
  Bash job leaves its native children running), so for ~3 min two T1 labellings and two copies of
  the first tau run ran at once. Everything was killed, the half-written run and label folders
  were deleted, and the night restarted cleanly. No reported result comes from those minutes.
* At 05:42 on 19 Nov the Claude app restarted and killed the jobs it had started (the eval stage
  and D2 labelling at 297K/708K). Both resumed (labelling is chunk-resumable). Later jobs were
  started as independent Windows processes (`Start-Process`), which survive app restarts.

---

## 2. Results

Lower cp lost is better. Intervals are 95 % bootstrap over whole games.
Full tables: `results/v6/SUMMARY.md` (auto-generated), `results/v6/t2.json`, `results/v6/t1.json`,
`results/v6/gate2_t2.json`, `results/v6/gate2_t1.json`.

### 2.1 Headline: A vs B vs C (Gate 2), seed-averaged and paired

| Group (3 seeds each) | T2 cp lost | T1 cp lost | Dev cp lost |
| --- | --- | --- | --- |
| v5 checkpoint (reference) | 63.0 | 101.7 | — |
| A: behavioural cloning (`played`) | 54.8 | 84.7 | 76.5 |
| B: soft targets, tau 100 | 54.0 | **77.9** | **71.0** |
| C: soft targets + value head | 56.6 | 82.4 | 74.0 |

| Paired difference | T2 | **T1** |
| --- | --- | --- |
| B − A | −0.8 [−5.0, +3.2] | **−6.8 [−8.5, −5.0]** |
| C − A | +1.8 [−2.6, +6.7] | **−2.3 [−4.0, −0.5]** |
| B − C | −2.6 [−4.9, −0.3] | **−4.5 [−5.9, −3.0]** |

Answer: **yes — B and C both lose fewer centipawns than A on T1, with intervals excluding zero;
B is best.** T2 is too small (24 games) to separate them. T1 positions are harder on average (all
phases of 800 games, depth-14 labels), which is why every number is higher on T1 than on T2.

### 2.2 Attention ablation (all with C's objective: soft targets + value head)

| Attention | Dev | T2 (3 seeds) | T1 (seed 0) | Paired − C on T1 (3 seeds) |
| --- | --- | --- | --- | --- |
| plain (no geometry) | 92.2 | 64.2 | 102.2 | +19.6 [+17.0, +22.2] |
| soft (= v5: learned geometry biases) | 74.0 | 56.6 | 82.9 | — |
| hard (head h sees geometry h%4 only, + biases) | 65.6 | **50.4** | 72.5 | −9.0 [−10.9, −7.2] |
| **hard_ray** (rays stop at first piece) | **65.1** | 52.2 | **71.6** | **−10.1 [−11.9, −8.2]** |
| hard_only (masks, no learned biases) | 67.0 | 52.9 | 73.2 | −8.5 [−10.4, −6.6] |
| C without experts (MoE off) | 75.1 | 53.5 | 83.1 | +0.9 [−0.4, +2.1] |

The project statement's core idea — heads masked to rook, bishop, knight and king movement —
is the largest effect in the whole study. Mixture of experts makes no clear difference.

### 2.3 Search (2-ply, policy top-6, value head scores the replies)

| Model | T2 policy → search | T1 policy → search |
| --- | --- | --- |
| C (3 seeds) | 56.6 → 63.9 | 82.9 → 84.1 (s0) |
| hard_ray (3 seeds) | 52.2 → 60.1 | 71.6 → 78.2 (s0) |
| scale_L_279K | 49.1 → 53.8 | 59.9 → 70.1 |
| scale_L_1M | 40.0 → 49.1 | 49.2 → 57.7 |
| scale_M_1M / scale_S_1M | 38.6 → 51.5 / 47.3 → 63.2 | — |
| Round robin (scale_L_1M) | +1032 Elo → +939 | |

Search hurts every model. The value head (trained with MSE to 0.75 × expected score of the best
cp + 0.25 × game result) is too inaccurate at the leaves to beat the policy's own choice. This is
the first thing chunk 2 must fix before MCTS can help (section 5).

### 2.4 Scaling curve (hard_ray, soft targets + value head, seed 0)

| Size (width/layers/heads) | 70K | 140K | 279K | ~987K (1M) |
| --- | --- | --- | --- | --- |
| S (64/2/4), dev | 77.7 | 68.7 | 63.0 | 58.4 |
| S, T1 | 84.6 | 78.3 | 72.1 | 65.5 |
| M (128/6/8), dev | 67.6 | 60.7 | 55.4 | 45.8 |
| M, T1 | 77.9 | 69.4 | 61.3 | 50.7 |
| L (256/6/8), dev | 66.8 | 60.4 | 54.2 | **44.5** |
| L, T1 | 74.1 | 69.6 | 59.9 | **49.2 [46.8, 51.4]** |
| S / M / L, T2 | 60.7 / 55.1 / 54.0 | 52.3 / 50.2 / 50.5 | 50.3 / 50.2 / 49.1 | 47.3 / **38.6** / 40.0 |

On T2 the 1M models beat the team C++ engine at depth 4 (46.7 cp in the 18 Nov cross-eval)
without any search. M and L on 1M are within noise of each other on T2 (38.6 vs 40.0); L is
better on dev (44.5 vs 45.8) and T1 (49.2 vs 50.7).

(70K–279K: 6 epochs; 1M: 4 epochs, as in `stage_scaling`.) Data helps at every size and model
size helps once there is enough data: going from 279K to ~1M positions cuts the L model's T1
loss by another 10.7 cp. **The best model, `scale_L_1M`, loses 49.2 cp/move on T1 against v5's
101.7** — half. Training time on ~1M positions: S 8 min, M 28 min, L 45 min.

Every run's best epoch was its last one: all models were still improving and are
under-trained. More epochs is a cheap win.

### 2.5 First round robin

8 players, 8 openings × both colours × every pair = 448 games, 8 workers, ~53 min.
"v6 best" = `scale_L_1M` (best dev cp loss of all runs, so chosen without looking at any test
set). Config: `geochesszero/prototypes/03_method_comparison/tournament_v6.json`; results:
`.../results/tournament_v6.json` (every game with moves) and `logs/step4_tournament.log`.

| Player | Score | Elo (Random = 0) [95% CI] |
| --- | --- | --- |
| Stockfish 15.1, depth 1 | 88.4 % | +1196 [+1053, +1484] |
| **v6 best (policy)** | **75.0 %** | **+1032 [+892, +1288]** |
| Stockfish 15.1, UCI_Elo 1350 (0.1 s/move) | 73.2 % | +1012 [+865, +1259] |
| v6 best (2-ply search) | 66.5 % | +939 [+811, +1198] |
| Team v5 | 44.2 % | +680 [+563, +895] |
| Team v3 pilot | 33.5 % | +534 [+417, +787] |
| Prototype 2 ray masks | 15.2 % | +232 [+131, +387] |
| Random moves | 4.0 % | 0 |

Head-to-head for v6 best (policy): **vs v5 +13 =3 −0**; vs Stockfish Elo 1350 +6 =2 −8;
vs Stockfish depth 1 +4 =5 −7; vs its own 2-ply search +6 =6 −4.

So v6 is **~350 Elo above v5** and level with Stockfish's weakest UCI_Elo setting; search
again costs strength. For reference, the team C++ engine at depth 4 scored +927 in the
18 Nov cross-eval round robin (a different pool, so not directly comparable). The intervals are
wide (16 games per pair): chunk 3 should use 16–32 openings.

### 2.6 Every checkpoint with its scores

See `results/v6/SUMMARY.md` (every run with T2 cp lost, best-move %, blunders %, dev cp lost)
and the T1 table there. Training time per run: S ~9 min, M ~20 min, L ~23 min on 279K.

---

## 3. Where everything lives (sizes as of 19 Nov)

| Path | What | Size |
| --- | --- | --- |
| `data/v6/positions_train.npz` (+ `.json`) | 2,500 games → train/dev/test positions | 51 MB |
| `data/v6/positions_holdout.npz` | T3 holdout positions (800 games) | 17 MB |
| `data/v6/positions_extra.npz` | D2 positions (games 4101–10600) | 130 MB |
| `data/v6/labels_dev_d10`, `labels_train_d10` | Stockfish 15.1 depth-10 labels (chunks) | 1.5 / 18 MB |
| `data/v6/labels_t1_d14` | T1 labels, depth 14, stride 10 | 0.8 MB |
| `data/v6/labels_extra_d10` | D2 labels, depth 10 | 46 MB |
| `data/v6/dev_d10`, `train_d10`, `train_nolabel` | assembled tensors | 28 / 340 / 326 MB |
| `data/v6/t1_d14`, `t3_holdout` | T1 (labelled), T3 (unlabelled) | 13 / 121 MB |
| `data/v6/extra_d10`, `big_d10` | D2 tensors (707,896 rows), train+D2 (987,261 rows) | 862 MB / 1.2 GB |
| `benchmarks/t2_p2_1312` | T2 (Prototype 2, Stockfish 15.1 depth 12) | in git |
| `runs/v6/<run>/` (39 runs) | `best.pt`, `last.pt`, `config.json`, `metrics.json`, `log.jsonl` | 517 MB total |
| `runs/v6_checkpoints_chunk1.zip` | every run's `best.pt` (+ sha256), `metrics.json`, `config.json` | 240 MB |
| `runs/v6/<run>.stdout.txt` | training stdout | small |
| `results/v6/` | SUMMARY.md, t2.json, t1.json, gate2_*.json | 1.6 MB, in git |
| `logs/` | every log and driver script of chunk 1 | 0.2 MB, in git |
| `tools/stockfish/stockfish.exe` | the Stockfish 15.1 build | 50 MB |
| `tools/src/Stockfish-sf_15.1` | its source + net | 50 MB |
| `tools/gcc` | portable GCC 12.4 (WinLibs) | 844 MB |
| `tools/venv-cpu`, `.venv` | CPU-only and CUDA Python environments | 788 MB / 4.3 GB |
| `geochesszero/prototypes/03_method_comparison/results/tournament_v6.json` | round robin, every game | in git |

`data/v6/`, `runs/`, `tools/` and `.venv/` are git-ignored.

---

## 4. How to rerun anything

### Environment

| | |
| --- | --- |
| OS | Windows 11 Pro, Git Bash for scripts |
| Python | 3.12.3 (python.org build), venv `.venv` |
| PyTorch | 2.14.1+cu126 (CUDA 12.6, cuDNN 9.10.02), numpy 2.5.5, python-chess 1.11.2 |
| GPU / driver | RTX 3060 12 GB, NVIDIA driver 610.88 |
| CPU | i5-11400, 6 cores / 12 threads, AVX2 |
| Stockfish | 15.1 from source tag `sf_15.1`, GCC 12.4, `ARCH=x86-64-avx2`, bench 3467381, Threads 1, Hash 16 |
| Label depths | dev/train/D2: 10 · T1: **14** · T2: 12 (Prototype 2) · T3: none |

### Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Stockfish 15.1 (from Git Bash, with tools/gcc/mingw64/bin on PATH, net already in src/):
#   cd tools/src/Stockfish-sf_15.1/src && mingw32-make -j12 all ARCH=x86-64-avx2 COMP=mingw
$env:STOCKFISH = "tools\stockfish\stockfish.exe"   # backslashes, or any path after Fix 1
.venv\Scripts\python -m pytest tests -q              # 45 passed, 1 skipped
```

### Commands actually used (from the repo root, `PYTHONPATH=src`)

The CLAUDE.md commands work unchanged except for:
* `--engine` given as `C:\Users\User\Documents\GeoChess\tools\stockfish\stockfish.exe`
  (any form works after Fix 1), `--workers 10` for dev/train, `--workers 9` while training ran;
* T1: `... label data/v6/positions_holdout.npz --splits test --stride 10 --depth 14 --chunk 250
  --out data/v6/labels_t1_d14 ...` then `assemble ... --out data/v6/t1_d14`;
* eval: `python -m chess_policy.v6_experiments --stage eval --t1 data/v6/t1_d14` (scores every
  run on T1 — ~12 h on this PC while labelling; `logs/day2_eval_t1.sh` is the subset actually run);
* Gate 2: `python scripts/v6_paired_groups.py --data data/v6/t1_d14 --reference A_played
  --reference C_value --group A_played --group B_soft --group C_value --group att_hard
  --group att_hard_ray --group att_hard_only --group att_plain --group moe_off
  --out results/v6/gate2_t1.json`;
* Gate 1 helpers: `scripts/v6_gate1_check.py`, `scripts/v6_spotcheck_t2.py`;
  dev table: `scripts/v6_dev_summary.py`.

The exact driver scripts are in `logs/` (`run_step1_labels.sh`, `run_overnight.sh`,
`run_overnight_cpu.sh`, `resume_day2.sh`, `day2_eval*.sh`, `day2_scaling_1M.sh`).
Every stage skips finished runs, so rerunning a stage only does what is missing.

---

## 5. Next chunks

### Chunk 2 (teammate 1): MCTS with PUCT, oracle self-play, AlphaZero targets, distillation

Start from `runs/v6/scale_L_1M/best.pt` (best run: hard_ray attention, soft targets + value
head, L size, ~1M positions; T1 49.2 cp, +1032 Elo). Use `scale_S_1M` (65.5 cp) when you need
speed. Suggested first steps, in order:

1. **Fix the value head first.** 2-ply search currently *loses* 5–10 cp (section 2.3). Measure
   value calibration on dev (predicted expected score vs `cp_to_value(best cp)` and vs game
   result), then try: value weight 2–4 instead of 1; value_mix 0 (pure Stockfish expected score
   instead of 25 % game result); a separate value MLP on the state token; training longer
   (every run peaked at its last epoch). Success = 2-ply search beating the policy on dev.
2. **PUCT MCTS** with the policy as prior and the value head at leaves (c_puct ~1.5–2.5, 50–400
   simulations). Expect it to hurt until step 1 works; once the value head is calibrated, expect
   a few cp/move and a clear Elo gain over the raw policy in the round robin.
3. **Oracle-labelled self-play:** play games with the model (+ MCTS), label positions with
   Stockfish 15.1 depth 10 (`v6_data label` is resumable and shardable: `--shard/--of`), add to
   training. Budget: ~20 positions/s on this PC, i.e. ~70K positions/hour.
4. **AlphaZero-style targets:** MCTS visit counts as policy targets, game result / search value
   as value targets; compare against soft Stockfish targets on T1 with
   `scripts/v6_paired_groups.py`.
5. **Distillation big → small:** `scale_L_1M` as teacher, S as student, soft targets from the
   teacher's policy. Reference numbers on T1: S trained on Stockfish labels = 65.5 cp (1M) /
   72.1 (279K); teacher L = 49.2. A distilled S that lands between them is a success.

### Chunk 3 (teammate 2): full Phase 5 evaluation, report, slides, demo

1. **Elo pool:** extend `tournament_v6.json` (section 2.5) with more Stockfish 15.1 `UCI_Elo`
   levels (range 1350–2850; 1350/1600/1800/2000 are the useful ones here), v5, the best B, C,
   hard_ray, S/M/L, and chunk 2's MCTS player. Use more openings (16–32) for tighter intervals.
2. **T1 cp loss** for every run (the full `--stage eval --t1 data/v6/t1_d14` pass; run it when the
   CPU is free, ~3–5 h).
3. **Value calibration** plot (reliability diagram) for C, hard_ray and L.
4. **Search-scaling curve:** cp loss vs search budget (policy, 2-ply, MCTS 50/100/200/400).
5. **LLM baseline via Ollama** (`geochesszero/prototypes/03_method_comparison/llm_player.py`).
6. Figures: attention maps from `V6Policy.attention_maps()` for hard_ray (the masks make a
   good slide).
7. **Demo:** v5's UCI and HTTP adapters (`chess_policy.uci`, `chess_policy.api`) load v6
   checkpoints through `load_any`.

---

## 6. Known issues, unfinished work, sharing checkpoints

* **Training is not bit-for-bit reproducible on GPU**: `scale_S_279K` and `att_hard_ray_s0` have
  identical configs and seed, yet dev 63.0 vs 64.5. Treat differences under ~2 cp as noise; use
  3 seeds and paired intervals for any claim.
* Label noise of ~10–25 cp per move at depth 10 (section 1, Step 1).
* tau 50 / 100 / 200 were a near tie on dev (one seed each).
* All models were still improving at their last epoch.
* T1 eval covers a subset of runs (section 1, Step 2); the full pass is chunk 3's.
* **`v6_eval` is single-threaded** (CPU ~13 %, GPU ~8 % on this PC): the 2-ply search builds and
  encodes every reply position in pure Python, one at a time. For big passes, run several
  `v6_eval` processes in parallel, each with a few `--model`s (include `--model v5=...` and
  `--reference v5` in each so the paired numbers come out), then merge the `models` and
  `paired_vs_v5` dicts of the JSON files into `results/v6/t1.json` and call
  `v6_experiments.write_summary()`. With 6–8 processes this should be ~6× faster.
* The team C++ engine was not built (source missing).
* Drive C: is nearly full (~15–18 GB free). `tools/gcc` (~1 GB) can be deleted once nobody
  needs to rebuild Stockfish.
* Windows notes: give Stockfish an absolute or backslash path to other tools; long jobs started
  from the Claude app die when the app restarts — start them with `Start-Process` (see `logs/`).

**Sharing checkpoints** (`runs/` is git-ignored): **`runs/v6_checkpoints_chunk1.zip`** (240 MB)
holds `best.pt` + `best.pt.sha256` + `metrics.json` + `config.json` for all 39 runs. Unzip it at
the repo root (paths inside start with `runs/v6/`); `load_v6` checks each sha256. Share it by
Drive or similar (too big for git). The data (`data/v6`, ~3 GB) is rebuilt with the commands in
section 4; the Stockfish labels alone (`data/v6/labels_*`, ~66 MB) are worth sharing too, since
they took ~16 CPU-hours to make.
