# GeoChess Chunk 2 Handoff — Search, Value Calibration, Self-Play & Distillation (`v7`)

## Summary of Achievements
Chunk 2 resolved the Chunk 1 search degradation (where unpruned 2-ply minimax worsened `scale_L_1M` from `49.2 cp` to `57.7 cp` lost) and delivered calibrated value heads, batched PUCT MCTS, an 80-game Stockfish-labelled self-play dataset, AlphaZero visit-target ablations, and `L -> S` model distillation.

### Key Results Across Steps 1–5
1. **Step 1 — Value Head Calibration (`scripts/v7_value_calibration.py`, `runs/v7/v7_S_pure_sf`)**:
   - Diagnosed `value_mix=0.25` noisy game-outcome targets as an 11x higher MSE source than Stockfish targets.
   - Training `v7_S_pure_sf` (`value_mix=0.0`) cut Expected Calibration Error by **8.5x** on `dev_d10` (`0.0308` -> `0.0036`) and improved raw S-model policy loss by **3.9 cp** (`62.4` -> `58.6 cp` on 1k dev slice; `65.1` -> `62.7 cp` on full 20k dev).
2. **Step 2 — Batched PUCT MCTS (`src/chess_policy/mcts.py`, `tests/test_mcts.py`, `scripts/v7_benchmark_mcts.py`)**:
   - Implemented batched PUCT MCTS with First Play Urgency (FPU), calibrated virtual loss (`vl_weight=0.1`), exact terminal checkmate/draw detection, prior tie-breaking, and AlphaZero visit distributions.
   - On 300 positions of `dev_d10`, `scale_L_1M` + MCTS (`N=64, c_puct=1.5, batch_size=4`) improved centipawn loss from **50.6 cp** (raw policy) down to **38.0 cp/move** (**-12.6 cp** paired delta, 95% CI `[-28.4, -6.0]`) and reduced blunders by **32%** (`6.3%` -> `4.3%`).
3. **Step 3 — Oracle-Labelled Self-Play (`scripts/v7_selfplay.py`, `data/v7/selfplay_d10`)**:
   - Generated 80 MCTS self-play games (`scale_L_1M`, `N=32`, `c_puct=1.5`, Dirichlet noise `eps=0.25, alpha=0.3` and `T=1.0` for plies < 15), yielding **6,664 positions** (27 White wins, 28 Black wins, 25 draws) and **204,215 legal moves** labelled by Stockfish 15.1 at depth 10, plus MCTS root visit targets in `data/v7/selfplay_visits.jsonl`.
4. **Step 4 — AlphaZero Target Training (`scripts/v7_train_alphazero.py`, `runs/v7/v7_L_az_s0`)**:
   - Showed 80% `train_d10` replay prevents opening-distribution forgetting and preserves calibration (`v7_S_az_replay80_s0` matched the Stockfish soft control within `+0.35 cp [-0.60, +1.27]` on `t1_d14`).
   - A 1-epoch fine-tune of `scale_L_1M` (`v7_L_az_s0`) cut `dev_d10` value ECE nearly in half (`0.0342` -> `0.0186`) and value MSE by 13% (`0.0223` -> `0.0194`) with a `+1.26 cp` raw-policy tradeoff on `t1_d14`.
5. **Step 5 — Big -> Small Model Distillation (`scripts/v7_distill.py`, `runs/v7/v7_S_distill_L`)**:
   - Distilled `scale_L_1M` (11.6M params) into a fresh S student (307K params) on `data/v6/train_d10` (`distill_alpha=0.7, T=2.0, value_alpha=0.5`).
   - On `data/v6/t1_d14` (10,378 unseen positions, 800 games), `v7_S_distill_L` reached **67.9 cp/move** (`-3.79 cp [-5.98, -1.72]` vs `att_hard_ray_s0` at `71.7 cp`, and `-3.26 cp [-5.26, -1.33]` vs `v7_S_pure_sf` at `71.2 cp`), with best-move rate `33.5%` and blunders `8.9%`.

## Code & Result Artifacts
- **Engine & Tests**: `src/chess_policy/mcts.py`, `tests/test_mcts.py` (52 total `pytest` tests passing with Stockfish enabled; `ruff check` clean).
- **Scripts**: `scripts/v7_value_calibration.py`, `scripts/v7_benchmark_mcts.py`, `scripts/v7_selfplay.py`, `scripts/v7_train_alphazero.py`, `scripts/v7_distill.py`, and updated `scripts/v6_paired_groups.py` (`--model name=path` support).
- **Saved Metrics (`results/v7/`)**: All JSON outputs for calibration, 2-ply vs MCTS benchmarks, AlphaZero ablations, and T1 paired bootstrap comparisons.

## Notes for Chunk 3
- **Demo Adapter & Live Play**: Use `chess_policy.mcts.MCTS` with `runs/v6/scale_L_1M/best.pt` (or `runs/v7/v7_L_az_s0/best.pt`) and recommended inference defaults `num_simulations=64, c_puct=1.5, batch_size=4, vl_weight=0.1` (~570 ms/move on CPU; substantially faster on RTX 3060 CUDA).
- **Final Evaluation & Plots**: All raw JSON numbers needed for the report/slides (reliability bin tables, MCTS vs raw policy paired deltas, and S vs Distilled-S vs L scaling comparisons) are ready in `results/v7/`.