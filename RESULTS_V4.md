# V4 measured results — 17 November 2026

V4 substantially improves the synthetic mating task. It does **not** establish
stronger full-game play, and the goal of beating the 1600 setting remains unmet.
The experimental candidate, v3 baseline, continued-training control and raw
results are included.

## Final tests after frozen selection

| Measurement | V3 | Continued-training control | Targeted v4 |
|---|---:|---:|---:|
| Unseen sparse mate-in-one top-1, 1,000 positions | 33.6% | 39.3% | **88.4%** |
| Sparse mate-in-one top-3 | 60.3% | 71.8% | **97.5%** |
| Sparse mate set NLL, lower is better | 2.40781 | 2.21100 | **0.33028** |
| Fresh game move agreement, 51,611 positions | 29.100% | **29.219%** | 28.662% |
| Fresh game top-3 | **54.702%** | 54.680% | 54.512% |
| Fresh game token NLL, lower is better | 2.29708 | **2.29377** | 2.30932 |
| Previously missed game mates repaired, 11 opportunities | 0 | 0 | 2 |

The synthetic test excludes all 3,000 training/development symmetry orbits. The
54.8 percentage-point improvement reduces failures from 664 to 116 (82.5%) on
this specific task. It does not imply the same improvement in game blunder rate.
All 11 previously inspected game positions contain 12 pieces and five or six
pawns; the synthetic task contains three to five pieces and no pawns. Those
11 opportunities are correlated diagnostics, not another independent test set.

Fresh game testing uses 400 later games and 200 opening groups from the same
Fishtest run. Earlier positions/groups and new curriculum positions are excluded.
Uniform legal prediction would give 6.513% expected top-1 agreement. Recorded-move
agreement is not optimal-move accuracy: other moves can be equally good. V4's
top-1 opening-group bootstrap interval is 28.176%–29.180%.

## Match against Stockfish 15.1, UCI_Elo=1600

| Model | Wins | Draws | Losses | Unfinished | Score |
|---|---:|---:|---:|---:|---:|
| V3 | 1 | 1 | 14 | 0 | 9.375% |
| V4 | 0 | 2 | 14 | 0 | 6.250% |

Eight new openings, both colors, 16 games per model. Opponent: 100 ms per move,
one thread, 32 MiB hash, strength limiting enabled. Policy: one forward pass,
no search. Model order alternates across pairs. Both sides claim available draws.
Every move is checked and replayed; PGNs and model/engine hashes are saved.

V4 minus v3 score is **−3.125 percentage points**. The 95% bootstrap interval,
resampling whole opening pairs, is **−21.875 to +12.500 points**. This sample does
not establish which model has higher underlying strength. It does not establish
that v4 beats 1600. Opponent strength limiting is stochastic, time budgets are
unequal, and other CPU evaluation jobs overlapped part of the match. A larger
isolated match is required for reliable strength estimation. No Elo is claimed.

## Development selection

Both arms start at v3 and receive 1,800 updates of 64 examples, seed 8327, and the
same optimizer/schedule. The control uses original replay. V4 uses 50% replay,
37.5% learner-rollout corrections and 12.5% exact mates. This tests the combined
intervention, not the separate contribution of each new dataset. One seed per arm.

| Development quantity | V3 | Control, step 1,200 | V4, step 1,800 |
|---|---:|---:|---:|
| Original NLL, 12,000 rows | 2.28215 | 2.28451 | 2.30273 |
| Correction NLL, 2,795 rows | 2.28378 | 2.29575 | 2.21956 |
| Mate set NLL, 500 rows | 2.32683 | 2.13002 | 0.27781 |
| Weighted selection objective | 2.28835 | 2.26942 | **2.01843** |

The prespecified objective is `0.5 original + 0.375 correction + 0.125 mate`,
subject to original NLL at most v3 + 0.05. V4 qualifies and wins that objective.
This explicitly permits a small original-data regression; it does not guarantee
stronger games. Selection was frozen before final tests and the arena. Their
results did not trigger another training or selection round.

Control training: 335.65 seconds; targeted training: 359.84 seconds. The arms ran
concurrently. Each consumes 115,200 presentations with replacement, matched by
updates rather than exact FLOPs. Correction collection took 306.88 seconds;
mate train/development sampling took 70.98 seconds. Preparation/evaluation are
additional costs.

## Engineering checks

- 26 automated tests pass; Ruff passes. Coverage includes perft, special rules,
  D4, attention/router/expert gradients, full-board interactions, basket padding,
  checkpoint recovery/corruption, data integrity, HTTP/UCI, set loss and labels.
- Zero audited prohibited exact-position overlaps and zero mate-orbit overlap.
  All new labels are legal; all mate target sets are exhaustively verified.
- Warmed CPU inference on 100 random-walk positions: median **4.160 ms**, p95
  **7.352 ms**, including encoding/output. Encoding median: 0.410 ms. Four PyTorch
  threads. All 100 outputs legal. This tests latency/legality, not strength.
- **303,064 parameters**; unchanged QKV/FFN/MoE architecture; no value head,
  runtime teacher, search or mating override. Larger presets are untrained.
  No CUDA performance measurement is claimed.

## Remaining bottleneck

The clearest observed failure is poor transfer from narrow tactical supervision
to real game positions. Three-to-five-piece mating accuracy does not ensure
correct choices on pawnful 12-piece boards. Ordinary imitation quality also
barely improves under short continuation. Geometry and legality alone do not
determine move quality.

The next controlled intervention should collect pawnful, crowded tactical and
conversion positions from learner trajectories, covering both mating and
preventing immediate loss while retaining ordinary-game performance. Use fresh
evaluation groups and multiple seeds. Model capacity is a separate experiment;
these results do not isolate size as the limiting cause. Promotion should require
game-outcome improvement as well as development metrics. This release records a
useful local gain and a failed strength goal.

## Evidence map

| Claim | Artifact |
|---|---|
| Initial failure audit | `v4_results/v3_loss_audit.json` |
| Plan and frozen selection | `v4_results/experiment_plan.json`, `selection.json` |
| Learning curves/provenance | `v4_results/control_history.json`, `targeted_history.json`, corresponding `*_run.json` |
| Fresh game metrics | `v4_results/fresh_comparison.json`, `fresh_*.json` |
| Separate mate test | `v4_results/mate_test.json` |
| Game diagnostic revisit | `v4_results/diagnostic_revisit.json` |
| All 32 games | `v4_results/paired_arena/` |
| Partition/label audit | `v4_results/refinement_data_audit.json` |
| Tests/environment/runtime | `v4_results/tests.log`, `lint.log`, `environment.json`, `runtime.json` |

V4 checkpoint SHA-256:
`63fce2e23ca31d739392a6bcf8068bfc32393f930409e87cb1d13e3c02571ce6`.
The release manifest records every included file's size and checksum.
