# Geometric Chess Policy System — v0.4

A current position becomes a probability distribution over legal UCI move tokens.
The model uses learned Q/K/V attention, geometric and attack relations, SwiGLU
FFNs, and top-two mixture-of-experts routing. Inference is one neural forward pass.

**V4 is a measured training experiment, not a demonstrated strength upgrade.**
Unseen sparse mate-in-one accuracy rises from 33.6% to 88.4%, but fresh game move
agreement falls from 29.10% to 28.66%. Against Stockfish configured at 1600, v4
scores 0 wins, 2 draws, 14 losses. The 1600 target remains unmet. See `RESULTS.md`.

## Run the trained model

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,serve]'
chess-policy predict --show-tokens
chess-policy uci
chess-policy serve --host 127.0.0.1 --port 8000
```

Install a CPU-only PyTorch wheel first if appropriate, or a CUDA build compatible
with your GPU/driver. Tested versions are in `v4_results/environment.json`.
The default checkpoint is the experimental `checkpoints/policy_v4.pt`.
The untouched v3 baseline remains available:

```bash
chess-policy predict --checkpoint checkpoints/pilot.pt
```

HTTP: `GET /health` reports checkpoint identity; `POST /predict` accepts
`{"fen":"...", "top_k":5}`. Use `top_k=0` for the complete legal basket. Displayed
probabilities refer to that complete basket. Temperature zero takes the maximum;
positive temperature samples with an optional seed. UCI handles move histories,
terminal states and `go searchmoves`. It chooses immediately without searching.

No Stockfish executable is needed to play or train on the included labels.
Stockfish is only an optional teacher during data preparation or an opponent.
No value head, minimax, MCTS or hardcoded tactical override selects policy moves.

## What changed

The 303,064-parameter model and encoder are unchanged from v3. V4 adds:

- Offline decision auditing with exact mate detection and estimated teacher regret.
- Teacher labels for learner trajectories: 14,714 training and 2,795 development examples.
- Exact mate targets: 2,500 training, 500 development and 1,000 separate test positions.
- Set-valued next-token loss that accepts every immediate mating move.
- An update-matched continued-training control and frozen development-only selection.
- Fresh game testing, 32 paired arena games, PGNs, integrity audits and raw logs.

Only 2 of 11 previously missed real-game mates are repaired. Those boards have
12 pieces and five or six pawns; the curriculum has three to five pieces and no
pawns. This identifies a specific coverage mismatch to investigate.

## Core code

| Layer | Files under `src/chess_policy/` | Responsibility |
|---|---|---|
| Rules/geometry | `geometry.py`, `encoding.py` | Lattice, D4 geometry, attacks and legal move basket |
| Network | `model.py` | QKV, residuals, SwiGLU, MoE and candidate cross-attention |
| Original training | `data.py`, `training.py` | Streaming PGNs, splits, deduplication, AMP, epoch resume |
| New supervision | `corrections.py`, `mate_curriculum.py` | Learner-distribution labels and exact mate sets |
| Refinement | `refinement.py` | Set loss, replay mixture and retention constraint |
| Deployment/evidence | `inference.py`, `evaluation.py` | Direct prediction and independent metrics |

The rules engine is `python-chess`. The representation has 64 square tokens plus
one state token. D4 describes square geometry; it is not imposed as a symmetry of
pawn direction or castling. The network sees the current position, not a move
transcript. The rules layer retains history when needed for repetition.

Read `docs/ARCHITECTURE.md`, `docs/REFINEMENT.md` and `docs/OPERATIONS.md` for
mathematics, decisions and supported behavior. This uses public transformer
techniques, not undisclosed GPT-4 internals.

## Reproduce and test

```bash
python -m pytest -q
python -m ruff check src scripts tests
python scripts/benchmark_runtime.py --out v4_results/my_runtime.json
python scripts/evaluate_mates.py --out v4_results/my_mate_test.json
bash scripts/reproduce_refinement.sh my_reproduction
```

The reproduction script builds corpora from included PGNs, audits labels, trains
both arms from v3, freezes its choice, then evaluates fresh game data. Use a fresh
output directory. Original corpora require roughly 516 MiB; the fresh test adds
about 65 MiB. Preparation and evaluation are additional to the roughly six-minute
CPU refinement measurement. Timings/results can vary across environments.

The refinement runner is a fixed CPU experiment. It saves inference weights, not
resumable optimizer snapshots. The main trainer retains AMP and epoch-boundary
recovery. `README_V3.md` and `RESULTS_V3.md` preserve source-training instructions
and historical evidence; their version/default references describe v3.

Optional 32-game match:

```bash
python scripts/paired_arena.py --stockfish /usr/games/stockfish --out my_arena
```

The opponent must support the 1600 setting. Capped games remain unfinished.
Configured opponent Elo is not a calibrated rating for this policy.

Larger presets in `configs/` contain approximately 2.04M, 11.56M and 284.43M
parameters. They are untrained. The RTX 3050 preset uses small microbatches with
accumulation; GPU performance and memory were not measured here. This is a tested
research/engineering package, not a claim of production deployment history.

Attribution is in `THIRD_PARTY.md`. The package includes weights, compact PGNs,
new labels, tests and evidence. Regenerable mmap arrays and engine binaries are omitted.
