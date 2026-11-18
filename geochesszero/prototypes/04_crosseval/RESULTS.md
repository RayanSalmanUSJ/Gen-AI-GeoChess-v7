# Cross-evaluation: team Chess Policy System (v3/v4/v5) vs Prototypes 1–3

Every model is scored on the same positions and in the same games. The team's checkpoints were loaded unchanged.

## A. Move quality (Prototype 2 benchmark)

1,312 held-out positions, scored by Stockfish 15.1 at depth 12.

| Model | Params | Train positions | cp lost / move | Best move | Blunders (≥200 cp) |
|---|---|---|---|---|---|
| Team v4 | 303K | ~279K + corrections | **60.5** | 43.8% | **8.6%** |
| Team v3 pilot | 303K | ~279K | 62.3 | 42.7% | 8.8% |
| Team v5 | 303K | ~279K + corrections | 63.2 | **45.0%** | 8.7% |
| Ray + 2-ply search (P2) | 0.84M | 22.9K | 75.6 | 40.1% | 11.9% |
| Geometry masks, ray (P2, 3 seeds) | 0.84M | 22.9K | 86.4 | 40.5% | 13.4% |
| Geometry masks, static (P1, 3 seeds) | 0.84M | 22.9K | 93.9 | 39.8% | 15.1% |
| Standard transformer (3 seeds) | 0.84M | 22.9K | 142.7 | 37.8% | 24.1% |
| Team C++ engine, depth 4 | — | — | 46.7 | 45.5% | 4.6% |
| Stockfish, depth 1 | — | — | 40.1 | 51.4% | 4.6% |

Paired difference, ray (3 seeds) minus v5: +23.2 cp/move [+10.8, +37.0]. The v5 family is significantly better.

## B. Recorded-move agreement (team benchmark)

27,803 positions from 200 of the 800 Fishtest holdout games.

| Model | Top-1 | Top-3 |
|---|---|---|
| Team v3 pilot | **28.55%** | **54.29%** |
| Team v4 | 28.08% | 53.54% |
| Team v5 | 27.78% | 53.36% |
| Ray (P2, best seed) | 22.37% | 43.61% |
| Static (P1, best seed) | 20.53% | 40.84% |
| Standard | 16.25% | 31.94% |
| Uniform over legal moves | 6.54% | — |

These numbers reproduce the team's reported 28.53% for v3.

## C. Games

Round robin, 8 openings × both colours. 336 games, 16 per pair. Elo is anchored at Random = 0.

| Player | Score | Elo [95% CI] |
|---|---|---|
| Stockfish, depth 1 | 93.2% | +1071 [+908, +1338] |
| Team C++ engine, depth 4 | 84.9% | +927 [+772, +1200] |
| **Team v5** | 59.4% | **+565 [+448, +719]** |
| Team v3 pilot | 45.8% | +414 [+299, +550] |
| Ray + 2-ply search (P2) | 32.3% | +275 [+186, +391] |
| Geometry masks, ray (P2) | 25.5% | +204 [+119, +297] |
| Random moves | 8.9% | 0 |

Head-to-head results (W-D-L, from v5's side):

| Opponent | v5 result |
|---|---|
| P2 ray | 16-0-0 |
| Ray + search | 12-2-2 |
| v3 | 8-5-3 |
| Team C++ engine d4 | 2-0-14 |
| Stockfish d1 | 0-0-16 |

## Takeaways

- The team's v5 is the strongest learned model on every measure. It plays the best games, and v3/v4 are close behind on move quality.
- **Agreement % is the wrong headline metric.** v5 agrees with Stockfish less often than v3, yet it beats v3 head-to-head (8-5-3) and finds the best move more often. Report cp loss, blunders and Elo.
- Confound: v3/v5 trained on about 12× more positions than Prototype 2. So this comparison does not show that their objective beats value distillation.
- Geometry helps in both families. On the same data: ray masks 86 cp, static masks 94 cp, plain attention 143 cp.
- Search on a value head adds about 70 Elo to the P2 model. v5 has no value head, so it cannot search yet.
- Classical alpha-beta at depth 4 (team C++ engine) still beats every learned policy.

Reproduce:

```
python -I crosseval.py --v5-root <chess_policy_v5>
python -I ../03_method_comparison/tournament.py --config ../03_method_comparison/tournament_v5.json ...
```
