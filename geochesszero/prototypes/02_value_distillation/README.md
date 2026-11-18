# Prototype 2: value distillation + occupancy-aware geometric attention

Prototype 1 taught a geometry-aware transformer to copy human moves.
Prototype 2 keeps the architecture and changes the objective to
**value-optimal play**. The model learns a teacher's score for *every*
legal move, plus a scalar value head. It compares three attention variants
under identical conditions:

| Variant | Heads 1-4 | Heads 5-8 |
|---|---|---|
| `standard` | unrestricted | unrestricted |
| `static` (Prototype 1) | rook lines, bishop diagonals, knight jumps, king ring (empty board) | unrestricted |
| `ray` (new) | rook/bishop rays stop at the first occupied square; knight/king static | unrestricted |

## Pipeline

1. `datagen.py`: Stockfish 15.1 self-play. Moves are sampled from a softmax over its own scores, with a 3% random move after ply 10. Every legal move is scored (MultiPV = all, depth 8).
2. `relabel.py`: held-out test games, every 2nd position re-scored at depth 12.
3. `dataset.py`: compact arrays. Soft policy target = softmax(expected score / 0.06); value = best move's expected score in [-1, 1]. Test positions that also occur in training are removed (`--exclude`).
4. `model.py`: the three variants (0.84M parameters each; Prototype 1's size), side-to-move-oriented input, policy head over Prototype 1's 4,272 actions, and a value head.
5. `train.py`: soft cross-entropy + value MSE, AdamW 3e-4, cosine schedule, by-game validation split, best epoch kept.
6. `evaluate.py`: centipawn loss, best-move rate, blunder rate and expected-score loss of the chosen move against the depth-12 teacher. Two move-choice modes:
   - `policy`: argmax of the policy head
   - `comparator`: least-bad move by the value head, one ply

   Yardsticks: random play and Stockfish depth 1 and 4. Intervals come from a game-cluster bootstrap, with paired differences between variants.
7. `arena.py`: matches, colours reversed per opening.
8. `blame.py`: hindsight blame shares for one game. These are the training labels the proposed blame transformer would learn.
9. `report.py`: writes `results/SUMMARY.md` and `results/summary.json`.

## Run

```bash
STOCKFISH=/path/to/stockfish ./run_all.sh
python -m pytest ../../tests/test_prototype2.py
```

This package includes the generated data (`data/`) and the nine trained checkpoints (`runs/*/best.pt`), so `evaluate.py` and `arena.py` run without retraining. They are large; keep them out of git.

## Results (3 seeds per variant, 1,312 held-out positions, Stockfish 15.1 depth 12 as reference)

| Variant | Centipawns lost / move (policy, no search) | Blunders (200+ cp) | 2-ply search, cp lost* |
|---|---:|---:|---:|
| ray | 86.4 | 13.4% | 70 |
| static (Prototype 1 masks) | 93.9 | 15.1% | 67 |
| standard | 142.7 | 24.1% | 94 |
| Stockfish depth 1 (reference) | 40.1 | 4.6% | |
| random moves (reference) | 238.7 | 38.6% | |

\*every 4th test position (328).

- Paired, seed-averaged: ray - standard = -56 cp (95% CI -81 to -34); ray - static = -8 cp (-13 to -2).
- Matches with both sides avoiding repetitions, 120 games each: ray vs standard +52 =67 -1 (71%); static vs standard +45 =75 -0 (69%); ray vs static +22 =83 -15 (53%, not significant).
- The value head tracks the teacher across positions (r = 0.91), but used alone as a 1-ply "least bad move" comparator it plays near random: it grabs material without seeing the recapture. A 2-ply search fixes most of that.
- In absolute terms the models are weak: 6-8% against Stockfish depth 1.

Full tables: `results/SUMMARY.md`; raw numbers: `results/eval.json`, `results/summary.json`, match records `results/arena_*.json` and `results/norep_*.json`, blame example `results/blame_example.json`.
