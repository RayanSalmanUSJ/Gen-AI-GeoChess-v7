# Method comparison

1312 held-out positions from 24 games, priced by Stockfish 15.1 at depth 12. Centipawns lost per move (scores capped at +/-1000); 95% intervals resample whole games.

| Method | Family | Seeds | Centipawns lost / move [95% CI] | Best move | Blunders (200+ cp) | Note |
|---|---|---:|---:|---:|---:|---|
| Stockfish 15.1, depth 4 | reference | 1 | 36.7 [32.2, 41.7] | 50.2% | 3.7% |  |
| Stockfish 15.1, depth 1 | reference | 1 | 40.1 [34.1, 46.4] | 51.4% | 4.6% |  |
| Team C++ engine (alpha-beta), depth 4 | classical search | 1 | 46.7 [40.9, 52.7] | 45.5% | 4.6% | november.zip module 12, unchanged |
| Team C++ engine (alpha-beta), depth 3 | classical search | 1 | 69.3 [58.4, 81.6] | 42.5% | 10.1% | november.zip module 12, unchanged |
| Ray + 2-ply search (seed 0) | board transformer + search | 1 | 75.6 [63.9, 89.9] | 40.1% | 11.9% |  |
| Ray + self-play, wins/losses only (8 rounds) | self-play | 1 | 84.5 [71.1, 99.0] | 41.1% | 13.5% | policy gradient on game results; no Stockfish labels |
| Geometry masks, ray (Prototype 2) | board transformer | 3 | 86.4 [71.2, 103.2] | 40.5% | 13.4% |  |
| Geometry masks, static (Prototype 1) | board transformer | 3 | 93.9 [78.4, 111.8] | 39.8% | 15.1% |  |
| Ray + self-play, search-guided (1 round) | self-play | 1 | 98.8 [79.4, 121.4] | 40.3% | 16.4% | AlphaZero-style expert iteration; no Stockfish labels |
| Standard transformer (board) | board transformer | 3 | 142.7 [110.5, 180.3] | 37.8% | 24.1% |  |
| GPT-style move model (+1,200 games) | sequence model (GPT) | 1 | 190.0 [159.5, 226.8] | 30.1% | 29.9% | top move legal without the legal-move filter: 0.8% |
| GPT-style move model (204 games) | sequence model (GPT) | 3 | 200.6 [167.1, 241.2] | 27.5% | 31.7% | top move legal without the legal-move filter: 0.0% |
| Random moves | reference | 1 | 238.7 [207.1, 275.3] | 20.7% | 38.6% |  |
