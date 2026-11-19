> **Current status (19 Nov 2026): v6 chunk 1 is done.** New contributors (and their AI assistants): read
> [`CLAUDE.md`](CLAUDE.md) and [`HANDOFF.md`](HANDOFF.md) first. The rest of this README describes v5,
> which is the base v6 builds on. Best v6 model: 49.2 cp lost/move on T1 vs 101.7 for v5; +1032 vs +680 Elo.

# Geometric Chess Policy System — v0.5

A current chess position becomes probabilities over its legal UCI move tokens.
The trained network uses learned Q/K/V attention, geometric and attack relations,
SwiGLU feed-forward layers and top-two mixture-of-experts routing. It chooses by
one forward pass. No value evaluator, search, engine query or tactical override
participates in deployment.

**V5 improves tactical tests but still fails the 1600 game-strength goal.**
On new pawnful boards, mate-in-one accuracy improves from 34.9% to 79.7%, and
immediate-mate prevention from 48.0% to 58.3%. In the new paired match against
Stockfish 15.1 set to 1600, v5 scores 0 wins, 2 draws, 14 losses; v4 scores
0 wins, 1 draw, 15 losses. These small samples establish no rating. `RESULTS.md`
reports the regressions and limitations as well as gains.

## Run the included trained policy

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
with your GPU/driver. Tested versions are in `v5_results/environment.json`.
The default is the experimental `checkpoints/policy_v5.pt`. V4 and v3 remain:

```bash
chess-policy predict --checkpoint checkpoints/policy_v4.pt
chess-policy predict --checkpoint checkpoints/pilot.pt
```

HTTP: `GET /health` identifies the checkpoint; `POST /predict` accepts
`{"fen":"...", "top_k":5}`. `top_k=0` returns the full legal basket. Probabilities
refer to that full basket. Temperature zero selects the maximum; positive
values sample with an optional seed. UCI supports histories for rules handling,
terminal states and `go searchmoves`; it chooses immediately, without searching.
No Stockfish binary is needed to play or train on the bundled labels.

## V5 changes

- New 8–22-piece pawnful tactical examples: 6,000 attack and 3,000 defense training rows.
- Exact offline target sets for immediate mate and prevention of immediate opposing mate.
- Separate development and final tests, grouped by underlying mating configurations.
- A continued-training control, fixed retention constraint and development-only interpolation checks.
- A fresh 52,830-position game reserve and a new 32-game paired match.
- Exact audits of the played trajectories, including 11 v5 decisions allowing immediate mate despite a local prevention option.

The model architecture is unchanged. Final selection uses the unblended targeted
weights, alpha 1.0. The package retains both histories, all candidate selection
metrics, labels, PGNs and checkpoints. There is no claim that the local tactical
loss is an optimal-move oracle or that improved puzzle accuracy guarantees wins.

## Core study order

| Code under `src/chess_policy/` | What to master |
|---|---|
| `encoding.py`, `geometry.py` | Finite lattice, square tokens, attacks and legal UCI basket |
| `model.py` | Learned QKV, geometry bias, residuals, SwiGLU, MoE routing and candidate decoder |
| `data.py`, `training.py` | Source ingestion, splits, deduplication, gradients and recovery |
| `corrections.py` | Teacher labels on learner-generated trajectories |
| `tactics.py` | Offline exact tactical targets with full position context |
| `refinement.py`, `refinement_v5.py` | Set-valued token loss, mixtures and retention constraints |
| `inference.py`, `evaluation.py` | Direct prediction and independent evidence |

The network represents 64 squares plus a state token. The rules dependency is
`python-chess`. D4 describes square geometry; it is not imposed as a symmetry of
pawn direction or castling. The model sees the current position; the rules layer
uses history when necessary for repetition. Public transformer techniques are
used, not undisclosed GPT-4 internals.

Read `docs/ARCHITECTURE.md`, `docs/REFINEMENT_V5.md` and `docs/OPERATIONS.md` for
equations, experimental decisions and limits. `README_V4.md`, `RESULTS_V4.md`
and their v3 counterparts preserve historical behavior and measurements.

## Reproduce and benchmark

```bash
python -m pytest -q
python -m ruff check src scripts tests
python scripts/benchmark_runtime.py --out v5_results/my_runtime.json
bash scripts/reproduce_v5.sh my_reproduction
python scripts/paired_arena_v5.py --stockfish /usr/games/stockfish --out my_arena
```

Reproduction builds corpora from bundled compact PGNs, audits the tactical data,
trains both arms, performs development selection and tests the frozen models.
Use a fresh experiment directory. Generated arrays need approximately 650 MiB.
Data preparation/evaluation are additional to the measured 5–6 minute CPU training
runs. Timings and game outcomes vary with hardware and opponent randomness.

The refinement runners are CPU experiments and do not resume optimizer state;
the original trainer retains AMP and epoch-boundary recovery. Larger presets
(approximately 2.04M, 11.56M and 284.43M parameters) remain untrained. The RTX 3050
preset uses small microbatches with accumulation; no CUDA speed or VRAM claim
is made. This is a tested research/engineering package, not a proven production
service or a 1600-rated engine.

The archive includes code, trained weights, data, reports, tests and all game traces.
Regenerable mmap arrays and external engine binaries are omitted. Attribution
and licenses are in `THIRD_PARTY.md`.
