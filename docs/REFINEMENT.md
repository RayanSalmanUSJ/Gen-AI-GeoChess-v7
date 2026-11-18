# V4: correcting the distribution the policy actually encounters

## The machine and the measured fault

The interface stays `current position -> probabilities over legal UCI moves -> argmax`.
The 303,064-parameter pilot still uses learned Q/K/V self-attention, geometric and
attack biases, dense SwiGLU FFNs, top-two routed experts, and candidate-to-board
cross-attention. There is no scalar value head, minimax, MCTS, teacher query, or
mate-in-one override in deployment. A move such as `g6g7` wins selection only if
its neural logit is highest. Exact rules determine the available basket.

V3's four saved arena games exposed failures that aggregate imitation accuracy
concealed. An offline audit inspected 181 policy decisions: 26 had at least 100 cp
of estimated teacher regret, 10 at least 300 cp, 11 missed immediate mating moves,
and four allowed an immediate mating reply. Repeated opportunities in one game
are correlated. These counts are not 11 independently lost games.

The regret audit uses independent full-strength Stockfish searches of 20,000 nodes
for the unconstrained best move and the played move. It is a finite-search proxy,
with negative differences clipped to zero and mate scores represented by a finite
sentinel. The immediate-mate findings are exact legal-tree checks; the centipawn
findings are estimates. Neither establishes counterfactual causal blame.

A teacher-game dataset largely depicts strong play. A learner also encounters
positions created by its own errors, including unusual won endgames. We changed
the supervised distribution while preserving the architecture. This is inspired
by learner-rollout dataset aggregation, not a claim that every assumption or
convergence guarantee of DAgger is satisfied:
https://proceedings.mlr.press/v15/ross11a.html

## New supervision

### Learner-rollout corrections

`corrections.py` starts from 192 original training opening groups and 40 original
development groups. A frozen v3 chooses each played move with probability 0.7;
the full-strength teacher chooses with probability 0.3. Both colors use the same
mixture. Each visited position is labeled with a Stockfish 15.1 move from a
6,000-node search, regardless of which actor played the next move. Rollouts stop
at a terminal state, a claimable draw, or 96 plies.

This produced 14,714 training and 2,795 development examples. Legal teacher tokens
are targets; teacher evaluations, search depths and chosen actions are metadata,
never network inputs. The teacher runs only during preparation. No Stockfish
binary is needed to train on the included labels or to play with the checkpoint.

### Exact terminal-action curriculum

`mate_curriculum.py` samples valid pawnless positions with both kings, a queen or
rook, and occasional additional pieces. It enumerates *all* legal immediate mating
moves offline. The training set has 2,500 positions; development has 500. Whole
color-normalized D4 orbits are kept distinct. Full D4 is valid for this restricted
pawnless, castling-free family; it is not imposed as a symmetry of ordinary chess.

After checkpoint selection, another 1,000 positions were generated with seed 9941,
excluding all 3,000 training/development orbits and all reserved corpus positions.
They provide a separate sparse-board test. They do not test multi-move tactics,
general strategy, or winning an entire endgame.

If a position has three mating moves, ordinary single-label cross-entropy would
penalize two correct answers. We instead optimize

$$L(s)=-\log\sum_{a\in A_*(s)}\pi_\theta(a\mid s).$$

Here $A_*(s)$ is the exact mating set for curriculum positions, and a singleton
teacher token for ordinary examples. The latter reduces exactly to cross-entropy.
The network still emits the same legal-token logits. Tests cover singleton
equivalence, joint probability, and zero gradient through padded candidates.

## Controlled comparison

Both arms start from the same v3 checkpoint, receive 1,800 optimizer updates of
64 examples, and use the same seed, optimizer, learning-rate schedule and router
regularization. They are matched by update count and example presentations,
not exactly by FLOPs: legal-basket sizes and expert loads vary.

| Arm | Original replay per batch | Correction positions | Exact mates |
|---|---:|---:|---:|
| Continued-training control | 64 | 0 | 0 |
| Targeted refinement | 32 | 24 | 8 |

The replay pool is a deterministic 64,000-position sample from original training.
Each arm receives 115,200 presentations with replacement. The targeted arm sees
57,600 original, 43,200 correction and 14,400 mate presentations. These are
presentations, not counts of unique positions. The included histories record every
300-update development evaluation. All optimizer steps finish even when an earlier
checkpoint remains the best.

Selection uses the prespecified objective

$$J=0.5L_{\mathrm{original}}+0.375L_{\mathrm{correction}}+0.125L_{\mathrm{mate}},$$

subject to original development NLL increasing by no more than 0.05 from v3.
Original development is the same first 12,000 rows used for both arms. V3 itself
is a valid candidate. This objective explicitly trades a small amount of original
imitation performance for tactical and learner-distribution performance; it is
not a theorem about game strength.

The selected control is step 1,200; the selected targeted model is step 1,800.
Targeted refinement wins this development criterion and becomes `policy_v4.pt`.
The choice and model hashes were written before fresh-test and arena evaluation.
`select_refinement.py` refuses to overwrite an existing selection report.

## Leakage controls and final tests

The correction partitions follow the original opening-group split. Exact position
keys prevent crossing original train/development/test boundaries. The original
800-game reserve and inspected diagnostic positions are excluded. Mate orbit
groups do not cross partitions; cross-curriculum duplicates are also excluded.
Every new label is checked for legality; every mate label set is exhaustively
verified. The audit reported zero prohibited overlaps.

Final game-data testing uses source games 3,301–3,700, after the original 2,500
training-source games and 800 reserved games. Earlier corpus positions, reserved
opening groups, and every new curriculum position are excluded. This leaves
51,611 positions. They are fresh within this project, but still come from the
same Fishtest run; an independent source run would be a stronger distribution test.

The game match uses eight additional opening pairs, both colors, 16 games per
model. V3 and v4 face Stockfish 15.1 at `UCI_Elo=1600`, 100 ms per opponent move,
one thread and 32 MiB hash. Policies take one forward pass. Model order alternates
across opening pairs. PGNs and full move traces are saved. Claimable draws are
claimed; capped games remain unfinished. Match outcomes never select the checkpoint.

All results, including regressions, belong in `RESULTS.md`. A better mate score
does not establish a higher Elo. Sixteen games per model are a pilot, not rating
certification. Stockfish strength limiting is stochastic. CPU evaluation jobs ran
concurrently with part of the arena, so this is not an isolated timing benchmark.

## Boundaries of this release

The main trainer retains AMP and epoch-boundary resume. The refinement experiment
is a fixed CPU pilot; it saves best and final inference weights, but not resumable
optimizer snapshots. Restart an interrupted refinement in a new directory. It
does not implement distributed training, online continual updates or automatic
release promotion. Larger v3 configurations remain untrained in this package.

The control isolates the *combined* correction-plus-mate intervention against
continued training. It does not separately identify the contribution of each new
dataset, nor demonstrate that MoE outperforms dense FFNs. Those require additional
frozen-data ablations and multiple seeds. The current evidence supports the
specific observed changes, not every proposed architectural thesis.
