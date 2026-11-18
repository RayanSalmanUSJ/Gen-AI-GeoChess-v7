# V5 results — 17 November 2026

V5 improves pawnful tactical tests and repairs more of the previously inspected
mating failures. **It still does not beat the opponent configured at 1600.** The
experimental weights, unchanged baselines, labels, tests, traces and source code
are included. No calibrated Elo or production deployment history is claimed.

## Final tests after frozen selection

| Measurement | V4 | Control | V5 |
|---|---:|---:|---:|
| Pawnful mate-in-one top-1, 1,000 positions | 34.9% | 37.1% | 79.7% |
| Pawnful mate top-3 | 61.2% | 62.9% | 92.3% |
| Immediate-mate prevention top-1, 600 positions | 48.00% | 48.67% | 58.33% |
| Immediate-mate prevention top-3 | 73.17% | 75.17% | 83.67% |
| Fresh game recorded-move agreement, 52,830 positions | 28.607% | 28.713% | 28.139% |
| Fresh game token NLL, lower is better | 2.30446 | 2.29919 | 2.32335 |
| Old inspected mate opportunities repaired | 2/11 | 2/11 | 5/11 |

The attacking test spans 408 held-out sparse seed families; defense spans 335.
Variants sharing a seed are correlated. All 500 source test seeds were new, and
excluded every earlier sparse train/development/test family. Test boards contain
8–22 pieces and at least two pawns. They are constructed rule-valid states, not
a natural-game sample or a proof of historical reachability.

Uniform legal-move sampling would succeed on 4.714% of attacking examples and
25.466% of defensive examples. A defensive success means preventing immediate
opposing mate; it can still lose material or allow a longer forced mate. These
metrics must not be relabeled optimal-move accuracy.

V5's mating gain over v4 is 44.8 percentage points. Its defensive gain is 10.33
points. The inspected-game repair count rises from 2 to 5 of 11, but those positions
were used to diagnose the previous failure and are not an independent final test.

Fresh game evaluation uses source games 3,701–4,100, after every v3/v4 source game.
Exclusions leave 52,830 positions in 200 opening groups. These still come from the
same Fishtest run. V5's top-1 opening-group bootstrap interval is 27.603%–28.702%.
Recorded-move agreement is not optimal-move accuracy: alternative moves can be
just as strong. The original-game regression remains visible rather than hidden.

## Paired game match against Stockfish 15.1

| Model | Wins | Draws | Losses | Unfinished | Score |
|---|---:|---:|---:|---:|---:|
| V4 | 0 | 1 | 15 | 0 | 3.125% |
| V5 | 0 | 2 | 14 | 0 | 6.250% |

Eight additional openings, both colors, 16 games per model. Opponent:
`UCI_Elo=1600`, `UCI_LimitStrength=true`, one thread, 32 MiB hash, 100 ms per move.
Candidate: one neural forward pass with no search. Model order alternates across
opening pairs. Both sides claim available draws. Every move is checked and replayed.
No training or batch evaluation runs concurrently with this arena.

The v5 minus v4 score difference is **+3.125 percentage points**.
The 95% whole-opening-pair bootstrap interval is **-6.250 to +12.500 points**.
This sample does not establish stronger underlying play. Strength limiting is
stochastic and the time budgets differ. The configured opponent rating is not a
calibrated rating for the policy. The zero-win result leaves the strength goal unmet.

## What the game traces reveal

| Exact offline observation | V4 | V5 |
|---|---:|---:|
| Policy decisions | 642 | 509 |
| Immediate mating opportunities on the policy's turn | 0 | 0 |
| Decisions permitting an immediate mating reply | 23 | 24 |
| Such decisions with a legal way to prevent that immediate reply | 9 | 11 |

Neither model had an immediate mating opportunity in this match, so improved
mate-in-one execution had no direct opportunity to help. This does not mean
neither ever had a winning position. V5 still allowed immediate mate in 11 decisions
where at least one legal move could prevent it on the next reply. Some positions
may already have been lost by other means. These are local tactical diagnostics,
not proof that those decisions caused otherwise avoidable game losses.

The models reach different positions, so their event counts are not a controlled
same-position comparison. The practical lesson is nonetheless concrete: broader
synthetic tactical learning still transfers incompletely to naturally reached
states, and local mating skill alone is not full-game decision quality.

## Training and selection

Both arms start from v4. Each performs 1,800 updates of 64 examples, seed 28327,
with AdamW, warmup/cosine scheduling, gradient clipping and MoE regularization.
Each has 115,200 presentations, sampled with replacement. The replay pool has
64,000 original positions.

| Arm | Original replay | Teacher corrections | Sparse mate | Pawnful mate | Defense |
|---|---:|---:|---:|---:|---:|
| Control, examples per batch | 32 | 24 | 8 | 0 | 0 |
| Targeted | 32 | 16 | 0 | 8 | 8 |

Training data: 6,000 new attacking examples and 3,000 defensive examples.
Development: 1,000 attacking and 600 defensive examples, plus the same 12,000
original and 2,795 correction development rows. Split audits found zero prohibited
exact-position or seed-family overlap. All tactical target sets are computed
exhaustively offline and checked again by the dataset loader.

The development objective weights original NLL by 0.5, correction by 0.25,
attacking by 0.125 and defense by 0.125. Original NLL must remain at most v4 + 0.02.
Early targeted checkpoints exceeded that limit, prompting a development-only
parameter-interpolation grid at alpha 0, .25, .5, .75 and 1. The initial plan and
the adaptation are both retained; the grid was not prespecified from the outset.

The final unblended endpoint recovered enough original performance to qualify.
It won at **alpha = 1.0**, so interpolation did not change the delivered weights.
Original development NLL rose from 2.302732 to 2.319214, within the fixed ceiling
of 2.322732. The threshold was not relaxed. Selection was frozen
before final testing and arena play. No test-driven retraining occurred.

Control training took 320.07 seconds; targeted training took 342.53 seconds, while
both ran concurrently. This comparison matches updates, not exact FLOPs. The
curriculum took 30.61 seconds to generate. Preparation, interpolation checks and
final evaluation add time. One training seed per arm does not establish robustness
or separate the individual effects of attack examples, defense examples and mix changes.

## Architecture, verification and runtime

The network remains the same **303,064-parameter QKV/FFN/MoE policy**. Board geometry,
exact attacks and legal moves are encoded. Its output is a legal move token, not
a scalar evaluation. The deployment code does not call Stockfish, minimax, MCTS,
the offline mate enumerator or a tactical override. Public transformer primitives
are used; no undisclosed GPT-4 architecture is claimed.

**31 automated tests pass**, covering legality/perft, geometry, attention/router/
expert gradients, set-valued loss, integrity, recovery, APIs, exact defensive labels,
automatic-draw handling and interpolation. Ruff passes. Warmed CPU inference on
100 random-walk positions: median **3.535 ms**, p95 **5.478 ms**,
including encoding and output. All 100 moves were legal. Encoding median:
0.385 ms. Four PyTorch threads. No GPU performance was measured.

The refinement runner is a bounded CPU experiment and does not resume optimizer
state. The original trainer retains AMP and epoch-boundary recovery. Larger model
presets remain untrained. Safe weights-only loading and checksums are integrity
controls, not evidence of stronger play or production deployment experience.

## Evidence and next priority

The next data intervention should use naturally reached learner positions, with
hard examples of material loss, checks, defense and conversion, rather than
further increasing performance on this synthetic family alone. Multiple seeds,
broader source runs and preserved game-data performance are necessary controls.
Model capacity should be tested separately; this experiment does not prove it
is or is not the limiting cause.

| Evidence | Path |
|---|---|
| Original plan and development adaptation | `v5_results/initial_experiment_plan.json`, `experiment_plan.json` |
| Training and frozen model choice | `v5_results/control_history.json`, `targeted_history.json`, `selection.json` |
| Full final evaluation | `v5_results/final_evaluation.json` |
| All 32 games, PGNs and pair statistics | `v5_results/paired_arena/` |
| Exact game-tactic audit | `v5_results/game_tactic_audit.json` |
| Split checks | `v5_results/data_audit.json` |
| Tests, environment, runtime | `v5_results/tests.log`, `lint.log`, `environment.json`, `runtime.json` |

Checkpoint SHA-256: `f63b794a58ac4bdc2a42ca828d2689b8e4e6d8995ca2421279471140989eb54a`.
