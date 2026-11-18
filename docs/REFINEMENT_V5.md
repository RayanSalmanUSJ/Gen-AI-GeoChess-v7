# V5: pawnful tactical learning with a retention constraint

## Why another experiment

V4 improved separate sparse mate-in-one accuracy from 33.6% to 88.4%, yet its
paired game score was 0 wins, 2 draws, 14 losses against the 1600 setting. It
repaired only two of 11 previously missed game mates. All those boards contained
12 pieces and five or six pawns; the curriculum had three to five pieces and no
pawns. This motivated a distribution intervention, not an inference-time search
fallback or a claim that the transformer architecture had been disproved.

## The output contract remains unchanged

The 303,064-parameter transformer maps the current position to logits over legal
UCI tokens. Q/K/V attention, encoded geometry and attacks, SwiGLU, residuals and
top-two routed experts are unchanged. Legal moves receive neural probabilities;
argmax selects the move. The production inference module never imports the
offline tactical generator. No scalar evaluator, minimax, engine consultation,
hardcoded mating choice or safety override is added to play.

## Constructing broader examples

`tactics.py` expands sparse mating configurations with randomly placed pieces.
Each resulting position has 8–22 pieces and at least two pawns. Added pieces must
preserve orthodox board validity. The mating target is recomputed after adding
context: previous targets are never presumed valid through a blocker or attack.
All legal immediate mating moves form the accepted target set.

For defensive examples, the side to move is reversed when that position remains
valid. Every legal move is examined offline. It is rejected if the opponent then
has an immediate mating move; the remaining moves form the accepted set. Positions
with no safe move or no unsafe move are omitted. Automatic terminal draws count
as safe; draw claims are outside this local target definition.

The labels mean **immediate mate** or **absence of immediate opposing mate**. A
defensive target may lose material or allow a longer forced mate. It is not an
optimal-move oracle. This deliberate separation makes the new loss auditable.

The training set has 6,000 attacking and 3,000 defensive positions; development
has 1,000 attacking and 600 defensive positions. Whole sparse-mate configuration
families retain their original train/development ownership. The synthetic seeds
were grouped by pawnless D4/color equivalence. No full-D4 transformation is imposed
on the expanded pawnful boards. Exact duplicates and earlier corpus positions are
excluded. These boards pass rule-state checks, but historical reachability and
natural-game representativeness are not guaranteed.

Final tactical testing starts from 500 newly generated sparse configurations,
excluding every earlier sparse training, development and test configuration.
It expands them into 1,000 attacking and 600 defensive test positions. Several
contexts can share a seed family, so the 1,600 rows are not independent games.

## Learning and the control

The existing set-valued token objective is retained:

$$L(s)=-\log\sum_{a\in A(s)}\pi_\theta(a\mid s).$$

An elite-game or correction label makes $A(s)$ a singleton; tactical examples
can accept multiple moves. The training targets never enter the forward pass.

Both arms start from v4 and perform 1,800 updates of 64 examples with seed 28327,
AdamW, warmup/cosine scheduling, gradient clipping, and MoE balance/z losses.
Both use a deterministic 64,000-position original replay pool. The control
continues v4's mixture; the intervention swaps some correction/sparse supervision
for pawnful attack/defense examples.

| Arm | Original | Teacher correction | Sparse mate | Pawnful mate | Defense |
|---|---:|---:|---:|---:|---:|
| Control | 32 | 24 | 8 | 0 | 0 |
| Targeted | 32 | 16 | 0 | 8 | 8 |

This is a combined intervention compared at equal update/presentation counts.
It does not isolate each component or match exact FLOPs. Candidate baskets and
expert loads vary. One seed per arm is an engineering pilot, not a robustness study.

Development selection uses weights 0.5 original, 0.25 correction, 0.125 pawnful
mate and 0.125 defense. Original development NLL must remain within +0.02 of v4.
The unchanged parent is always an eligible fallback. Final game and tactical tests
play no part in selection.

## Repairing the observed retention failure

The first 900 updates improved pawnful tactical development performance but
violated the original-game retention constraint. That observation prompted an
additional *development-only* selection step, recorded explicitly in the updated
experiment plan. The original plan is preserved separately. This adaptation was
not prespecified from the outset and must not be described as such.

We evaluate parameter interpolations

$$\theta_\alpha=(1-\alpha)\theta_{v4}+\alpha\theta_{\mathrm{targeted}},
\qquad \alpha\in\{0,0.25,0.5,0.75,1\}.$$

The endpoint is the final 1,800-update targeted model. Model structure and
nonfloating geometric buffers must agree. Only floating tensors are interpolated.
The same fixed development objective and +0.02 constraint choose among these
models and the control. The threshold is never loosened to make a candidate pass.
There is no assumption that neural behavior changes linearly with these weights;
each intermediate model is measured. This is one deployed parameter set, not an
inference ensemble, so the forward-pass architecture/cost remains unchanged.

The final unblended endpoint subsequently recovered enough original-game loss to
pass the fixed constraint. It won selection at alpha = 1. The extra interpolation
checks therefore did not change the shipped weights; they remain recorded as a
development-only adaptation, not a necessary cause of this result.

## Fresh final evidence

The game-data reserve uses source games 3,701–4,100, after every source game used
by v3/v4. Excluding earlier positions/opening groups and new curricula leaves
52,830 positions. These remain from the same Fishtest run, not an independent
source distribution. Recorded-move agreement is a proxy, not optimal-move accuracy.

The final match uses eight additional openings, both colors, 16 games per model.
Frozen v4 and the selected v5 face Stockfish 15.1, UCI_Elo 1600, 100 ms per
opponent move, one thread and 32 MiB hash. No training or batch evaluation runs
concurrently with this match. Both models use a single neural forward pass.
Draw claims and unfinished-game handling follow the existing audited arena.

`RESULTS.md` records the measured outcomes, including regressions. All model hashes,
learning curves, selection candidates, legal targets, PGNs and traces are retained.
The trained model is still a CPU pilot; larger configurations are untrained and
neither this dataset nor parameter interpolation establishes a particular Elo.
