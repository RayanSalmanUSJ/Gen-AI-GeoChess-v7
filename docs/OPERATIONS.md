# Engineering contracts and release limits

## Interfaces

The model has four tensor inputs: `pieces[B,64]`, `geometry[B,64,8]`,
`attacks[B,64,64]`, and `legal[B,M]`, where `-1` denotes padding. Targets are local
indices into the basket and are never passed to the forward method. Shapes and
indices are constructed by the validated encoder/collator rather than supplied by
HTTP callers. The inference boundary validates orthodox FENs and terminal states.

The checkpoint pins the model configuration, encoding version and data-manifest
hash. Safe `weights_only=True` loading and a SHA-256 sidecar catch accidental
corruption. The sidecar is an integrity check, not an authentication signature:
both files still need a trusted distribution channel.

Checkpoint replacement is atomic for the weights file and then atomic for its
sidecar. A crash between those two replacements can leave a checksum mismatch;
loading fails closed. The previous `best.pt` and the most recent completed
`last.pt` have independent identities. Model serialization round-trips are tested.

## Training recovery

This section describes the original `training.py` trainer. The bounded CPU
`refinement.py` and `refinement_v5.py` experiments save best/final inference checkpoints but do not
save resumable optimizer state. Restart that experiment in a fresh directory.

The last checkpoint carries optimizer and AMP-scaler state, the epoch, optimizer
step, RNG states, schedule configuration, development history and corpus hash.
Only an identical configuration and corpus can resume. The tested interruption
restarts from the end of a completed epoch and produces bit-identical CPU weights
to uninterrupted training, including dropout. GPU numerical reproducibility is
not asserted. The partially completed current epoch is repeated after a crash.

An interrupted data build has no final manifest and is never accepted as a complete
corpus. Rebuild in a fresh directory. The importer refuses to overwrite an existing
directory. Invalid PGNs fail the build; they are not silently used for training.

## Observability

Training writes structured JSON progress, `run.json`, development metrics and
expert assignment fractions. Each completed epoch records training CE, development
CE, top-one and top-three recorded-move agreement, sample count and elapsed time.
The held-out evaluator reports NLL, agreement, Brier score, ECE relative to recorded
move labels, and a bootstrap interval resampling opening groups.

`/health` identifies the loaded checkpoint by hash. `/predict` returns measured
latency and policy entropy. The service serializes forward passes with a lock;
dynamic request batching, GPU queue scheduling, multi-tenant isolation, persistent
request metrics and overload control are not implemented. For exposed deployments,
an operator must add authentication, TLS, request-size limits and rate control at
the service boundary. No deployment is performed by this package.

## Test gates

- Exact start-position and Kiwipete perft counts and make/unmake restoration.
- Promotion, castling, en-passant and pinned-piece attack/legality distinctions.
- D4 permutations, group closure, Burnside count and square round-trips.
- Gradients through QKV, geometric attention weights, router and all experts.
- Full board interaction, including a gradient from a1's output to h8's input.
- Candidate permutation invariance and zero probability for padded candidates.
- Small-corpus overfit, checkpoint recovery and corruption rejection.
- Dataset checksums, partition separation, invalid input and path traversal rejection.
- Validated HTTP contract and UCI move restriction behavior.

The rules implementation is a dependency. Perft confirms the tested integration;
it does not prove every feature or learned strategic decision correct.

## Scale and compute limits

CPU execution was measured. CUDA configurations exist but are unmeasured here.
The implementation uses PyTorch SDPA and local sparse expert dispatch, not a fused
expert GEMM kernel. Small expert batches and Python dispatch can dominate latency.
All expert weights reside on one device, and the trainer rejects multi-process
world sizes. Larger settings should be profiled and validated on their target
hardware before long training jobs. No large-model quality claim follows from the
ability to instantiate its configuration on a meta device.

## Experimental discipline

V4's intervention is documented in `REFINEMENT.md` and `RESULTS_V4.md`; v5's is
in `REFINEMENT_V5.md` and the root `RESULTS.md`. The default weights are an
experimental development-selected candidate. Tactical accuracy alone is not a
release promotion criterion or a certified playing-strength result.

Use development loss to choose checkpoints, then evaluate the reserved set once.
Use additional source archives for the next release's external test. The current
reserve is a different group of games from the same Fishtest run, which limits
claims about distribution shift. Position deduplication ignores counters and keeps
the first occurrence; identical piece/turn/castling/EP states cannot span partitions.

A high move-agreement score does not establish low blunder rate. Rare catastrophic
decisions compound across a game. Paired-opening game matches, tactical stress
positions and larger independent samples are needed for strength claims. Changing
data and architecture together is an engineering comparison, not a causal ablation.

The next controlled experiments should compare dense FFNs against routed experts
at matched data and compute, geometric bias against no bias, and source variety
against simply repeating the same games. The package makes these components
separate; it does not claim those ablations have already demonstrated a benefit.
