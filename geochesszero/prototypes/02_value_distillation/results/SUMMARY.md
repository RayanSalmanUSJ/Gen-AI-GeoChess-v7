# Prototype 2 results

Test: 1312 positions from 24 held-out games, scored by Stockfish 15.1 at depth 12.

| Player | Move choice | Seeds | Centipawns lost / move | Best move | Blunders (>=200 cp) | Expected score lost |
|---|---|---:|---:|---:|---:|---:|
| stockfish_d4 | reference | 1 | 36.7 | 50.2% | 3.7% | 2.34 |
| stockfish_d1 | reference | 1 | 40.1 | 51.4% | 4.6% | 2.46 |
| ray | policy | 3 | 86.4 (seed sd 2.3) | 40.5% | 13.4% | 5.40 |
| static | policy | 3 | 93.9 (seed sd 1.7) | 39.8% | 15.1% | 5.86 |
| standard | policy | 3 | 142.7 (seed sd 0.8) | 37.8% | 24.1% | 8.65 |
| ray | comparator | 3 | 201.2 (seed sd 7.2) | 33.7% | 36.5% | 12.82 |
| static | comparator | 3 | 208.4 (seed sd 8.4) | 33.0% | 36.6% | 13.17 |
| standard | comparator | 3 | 219.6 (seed sd 2.8) | 33.3% | 38.9% | 13.88 |
| random | reference | 1 | 238.7 | 20.7% | 38.6% | 14.66 |

Value head Pearson r vs teacher: ray 0.913 (3 seeds), standard 0.914 (3 seeds), static 0.912 (3 seeds)

## Paired differences (seed-averaged, same positions; negative = first is better)

| Comparison | Centipawns lost | Best-move rate (pp) | Blunder rate (pp) |
|---|---:|---:|---:|
| policy: ray - standard | -56.3 [-81.4, -33.6] | 2.8 [0.2, 5.5] | -10.7 [-14.7, -6.7] |
| policy: ray - static | -7.6 [-12.6, -2.3] | 0.8 [-0.7, 2.2] | -1.8 [-2.6, -0.9] |
| policy: standard - static | 48.8 [28.0, 71.7] | -2.0 [-4.2, 0.2] | 9.0 [5.2, 12.9] |
| comparator: ray - standard | -18.3 [-33.6, -4.9] | 0.5 [-0.8, 1.9] | -2.4 [-5.4, 0.1] |
| comparator: ray - static | -7.2 [-16.0, 1.2] | 0.8 [-0.4, 1.9] | -0.1 [-1.5, 1.3] |
| comparator: standard - static | 11.2 [-1.3, 23.6] | 0.3 [-1.0, 1.6] | 2.4 [0.3, 4.6] |

## 2-ply search (every 4th test position: 328 positions; seed-averaged)

| Variant | Policy: centipawns lost | 2-ply search: centipawns lost | Policy blunders | Search blunders |
|---|---:|---:|---:|---:|
| ray | 87.9 | 69.8 | 13.1% | 8.9% |
| static | 98.9 | 67.0 | 15.2% | 8.4% |
| standard | 153.6 | 93.7 | 24.9% | 15.2% |
| stockfish_d1 (reference) | 46.2 | | 5.5% | |
| stockfish_d4 (reference) | 40.9 | | 4.3% | |
| random (reference) | 236.4 | | 38.4% | |

## Matches (colours reversed per opening; 'no-repeat' = both models skip moves that repeat a position when an alternative exists)

| Pairing | Seeds | Games | +W =D -L | Score of first [95% CI] | Elo gap |
|---|---:|---:|---:|---:|---:|
| ray_vs_random | 1 | 40 | +18 =22 -0 | 72.5% [66.2, 78.8] | +168 |
| ray_vs_sf_d1 | 1 | 40 | +0 =6 -34 | 7.5% [1.2, 15.0] | -436 |
| ray_vs_standard | 3 | 120 | +11 =109 -0 | 54.6% [52.1, 57.5] | +32 |
| ray_vs_static | 3 | 120 | +5 =114 -1 | 51.7% [50.0, 53.8] | +12 |
| raysearch_vs_sf_d1 | 1 | 40 | +0 =1 -39 | 1.2% [0.0, 3.8] | -759 |
| standard_vs_random | 1 | 40 | +4 =36 -0 | 55.0% [51.2, 60.0] | +35 |
| standard_vs_sf_d1 | 1 | 40 | +0 =5 -35 | 6.2% [2.5, 11.2] | -470 |
| static_vs_random | 1 | 40 | +15 =25 -0 | 68.8% [61.3, 76.2] | +137 |
| static_vs_sf_d1 | 1 | 40 | +0 =6 -34 | 7.5% [2.5, 12.5] | -436 |
| static_vs_standard | 3 | 120 | +6 =114 -0 | 52.5% [50.8, 54.6] | +17 |
| no-repeat ray_vs_standard | 3 | 120 | +52 =67 -1 | 71.2% [66.2, 76.2] | +158 |
| no-repeat ray_vs_static | 3 | 120 | +22 =83 -15 | 52.9% [47.9, 57.9] | +20 |
| no-repeat static_vs_standard | 3 | 120 | +45 =75 -0 | 68.8% [64.2, 73.3] | +137 |

## Training (best validation epoch)

| Run | Best epoch | Val KL to teacher | Val value MSE | Val top-1 vs teacher | Minutes |
|---|---:|---:|---:|---:|---:|
| ray_s0 | 6 | 0.806 | 0.0773 | 26.0% | 20 |
| ray_s1 | 6 | 0.802 | 0.0766 | 25.9% | 20 |
| ray_s2 | 6 | 0.815 | 0.0805 | 25.4% | 16 |
| standard_s0 | 6 | 0.940 | 0.0835 | 21.4% | 19 |
| standard_s1 | 6 | 0.942 | 0.0803 | 21.8% | 14 |
| standard_s2 | 6 | 0.944 | 0.0812 | 21.8% | 13 |
| static_s0 | 6 | 0.840 | 0.0767 | 24.6% | 19 |
| static_s1 | 6 | 0.842 | 0.0762 | 24.6% | 16 |
| static_s2 | 6 | 0.846 | 0.0853 | 24.4% | 16 |
