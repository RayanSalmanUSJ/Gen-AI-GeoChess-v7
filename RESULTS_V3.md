# Measured release results — 17 November 2026

The new geometry-aware transformer learns recorded Stockfish moves substantially
better than the earlier direct-move prototype on the same test positions. Its
playing strength is still inadequate to beat the tested 1600-configured opponent.
Those two findings must be kept separate.

## Held-out prediction

Evaluation used **103,772 unique positions from 800 reserved games**, grouped into
400 opening pairs. Neither these positions nor their opening groups overlap with
the initial 2,500-game corpus. Both excerpts come from the same Fishtest archive,
so this is not an independent-source generalization test.

| Model | Top-one agreement | Top-three agreement | Cross-entropy, nats |
|---|---:|---:|---:|
| Uniform random legal move | 6.34% expected | — | 3.1453 |
| Earlier shipped move-GPT prototype | 8.97% | 22.52% | 3.1843 |
| New QKV + SwiGLU + top-two MoE pilot | **28.53%** | **54.15%** | **2.3141** |

The new model's top-one agreement is **3.18 times** the previous model's;
cross-entropy is **27.33% lower**. Both training data and architecture changed,
so this is not an isolated test of geometric attention or MoE. The old model's
historical training overlap with this corpus has not been established.

Resampling whole opening groups gives a top-one 95% bootstrap interval of
**28.16%–28.88%**. Multiclass Brier score is 0.8231; 10-bin ECE against recorded-move
labels is 0.00772. These are imitation metrics. They do not establish game-theoretic
optimality, blunder avoidance or a chess rating.

Evidence: `results/heldout.json`, `results/previous_holdout.json`.

## Training that actually ran

- **303,064 parameters**, width 64, two board blocks, four attention heads.
- One dense SwiGLU block and one four-expert SwiGLU MoE block, top-two routing.
- A geometric move-query decoder with cross-attention and its own dense FFN.
- **279,365 available training positions**, 22,341 development and 25,319 initial
  test positions from the first 2,500 games. The initial test partition was unused
  for selecting or evaluating this release.
- Three epochs sampled 160,000 training positions each: **480,000 presentations**,
  covering **254,789 distinct positions**, or 91.20% of the available training set.
- Development selection used the same 12,000-position subset each epoch. The
  minimum-NLL checkpoint was epoch three; its hash was fixed before reserved evaluation.
- Training took **691.18 seconds** on this CPU environment with four PyTorch threads.
  No CUDA training occurred.

| Stage | Development top-one | Development top-three | Development NLL |
|---|---:|---:|---:|
| Random initialization | 7.63% | 18.96% | 3.0715 |
| Epoch 1 | 25.66% | 48.72% | 2.4519 |
| Epoch 2 | 27.65% | 52.67% | 2.3410 |
| Epoch 3 | 28.64% | 54.68% | 2.2822 |

Final expert assignment fractions on development were **24.24%, 24.23%, 25.51%,
26.02%**. Routing is active and not collapsed. This alone does not establish
semantic specialization or an advantage over a compute-matched dense model.

Evidence: `results/pilot_run.json`, `results/pilot_history.json`,
`results/final_selection.json`.

## Game match

Against **Stockfish 15.1 with UCI_Elo=1600**, one thread, 32 MB hash and 100 ms
per opponent move, the new policy scored **0 wins, 1 draw, 3 losses** over two
paired openings and both colors: **0.5/4, or 12.5%**. All four games terminated
under chess rules; none was assigned a draw merely for reaching a length cap.

The candidate used one direct neural decision per move, averaging **5.43 ms**
during the match. This is an unequal-compute pilot, not a calibrated Elo estimate.
The draw was a claimable threefold repetition. The three losses ended in checkmate.
Every recorded move was checked and independently replayed.

Evidence: `results/arena.json`, `results/arena.pgn`.

## Runtime and correctness

The final isolated runtime probe used 100 varied positions from a seeded random
legal walk, after warmup, with four CPU threads. All 100 predictions were legal.
Median encode time was **0.415 ms**; median complete prediction was **3.31 ms** and
p95 was **3.98 ms**. This is a runtime probe, not a strategic evaluation.

**21 tests passed**, and Ruff reported no lint failures. Tests include exact perft
counts of 197,281 and 97,862; full board attention; gradients through QKV, router,
geometric biases and experts; candidate/padding invariance; tiny-data overfit;
HTTP/UCI handling; checksum rejection; corpus partition integrity; CPU BF16 router
stability; a real medium-model forward/backward pass; and an interrupted training
run that resumes to bit-identical CPU weights.

The audit checked **430,797 unique positions** across all partitions: zero illegal
targets, zero repeated position fingerprints and zero crossing opening groups.
All shard checksums were verified. Separately, all **459,082 mainline moves** and
3,300 starting positions in the compact excerpts were checked against the original
checksum-pinned archive. The two test warnings came from dependency deprecations.

## Scaling configurations

| Configuration | Parameters | What was executed |
|---|---:|---|
| Pilot | 303,064 | Full training, holdout evaluation and game match |
| RTX 3050 starting preset | 2,040,752 | Parameter/configuration inspection; CUDA untested |
| Medium | 11,558,416 | CPU forward/backward correctness test; untrained |
| Large | 284,427,328 | Meta-device construction and parameter count; untrained |

No large trained model, 1600 win, CUDA speedup, distributed training result or
production deployment is claimed. The code supplies these larger single-device
architectures; the release weight file is the measured pilot.

## Current diagnosis

The system now satisfies the intended learning contract: elite-game move tokens
supervise a geometry-aware, fully learned policy with working attention and MoE.
Held-out agreement improves markedly. Nevertheless, about 71.5% of top-one choices
differ from the recorded action, and the game sample contains three losses.
Different moves are not automatically errors, but the match shows that average
imitation quality has not produced reliable play.

The next experiments should use diverse source archives and controlled architecture
ablations, then measure tactical failures and paired games. More parameters alone
are not a measured remedy. The current result warrants further training experiments;
it does not warrant claiming the playing-strength objective has been reached.
