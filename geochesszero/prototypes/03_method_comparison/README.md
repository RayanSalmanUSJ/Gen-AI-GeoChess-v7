# Prototype 3: compare the methods on one benchmark

The course question is whether a GPT-style next-token predictor can play chess,
and how it compares with the other approaches in the team's notes. Every
method here is scored on **Prototype 2's test bench**: 1,312 held-out positions
priced by Stockfish 15.1 at depth 12. A round-robin tournament then ranks
them in real games.

Nothing from the team's earlier work is rebuilt. The team's C++ engine
(november.zip, module 12) is used unchanged behind a small UCI adapter, and
the board transformers come straight from Prototype 2.

| Method | File | What it is |
|---|---|---|
| GPT-style move model | `gpt_movelm.py` | Decoder-only transformer over move tokens (`[BOS] m1 m2 ...`); same games and Stockfish targets as Prototype 2; never sees the board. Also trained with ~7x more games (`--extra`) |
| Self-play, search-guided | `selfplay.py` | AlphaZero-style expert iteration: self-play games, 2-ply search as the improvement operator, resignation; no Stockfish labels |
| Self-play, outcome-driven | `selfplay_pg.py` | Plays itself and learns only from results: policy gradient with its own value as baseline, KL anchor to the start |
| Classical alpha-beta | `engines/november_adapter/` | The team's own engine (alpha-beta, TT, null-move, LMR, trained handcrafted eval) behind a UCI adapter |
| Local LLMs | `llm_player.py` | Qwen / Llama through Ollama; legality, retries, same benchmark; runs on a machine with Ollama |
| Board transformers | `../02_value_distillation/` | Standard, static-mask and ray-mask attention (Prototype 2) |

`benchmark_all.py` builds the single table (`results/COMPARISON.md`).
`tournament.py` plays the round-robin (`results/tournament.json`), with Elo
anchored at random moves = 0. `players.py` gives every method the same
`move(board)` interface.

## Run

```bash
STOCKFISH=/path/to/stockfish NOVEMBER=/path/to/november ./run_all.sh
python -m pytest ../../tests/test_prototype3.py
```

Local LLMs, on a laptop with Ollama:

```bash
ollama pull llama3.2:3b
python llm_player.py bench --model llama3.2:3b --out results/llm_llama3.2-3b.json
python llm_player.py bench --model llama3.2:3b --no-legal-list --out results/llm_llama3.2-3b_nolist.json
python benchmark_all.py        # picks up results/llm_*.json
```

To include an LLM in the tournament, add `{"name": "Llama 3.2 3B", "spec": "llm:llama3.2:3b"}`
to `tournament_players.json`.

## Results

| Method | Centipawns lost / move (1,312 positions) | Tournament Elo vs random [95% CI] |
|---|---:|---:|
| Stockfish 15.1, depth 1 (reference) | 40.1 | +877 [+761, +1061] |
| Team C++ engine (alpha-beta), depth 4 | 46.7 | +827 [+738, +967] |
| Ray masks + 2-ply search | 75.6 | +300 [+246, +371] |
| Ray masks (Prototype 2) | 86.4 | +259 [+198, +323] |
| Ray + self-play, wins/losses (8 rounds) | 84.5 | +210 [+161, +258] |
| Static masks (Prototype 1) | 93.9 | +198 [+149, +249] |
| Ray + self-play, search-guided (1 round) | 98.8 | not entered |
| Standard transformer (board) | 142.7 | +93 [+43, +137] |
| GPT-style move model, +1,200 games | 190.0 | +75 [+28, +123] |
| GPT-style move model (204 games, 3 seeds) | 200.6 | +10 [-35, +52] |
| Random moves (reference) | 238.7 | 0 |

- **Reading the board beats reading moves.** The GPT-style model never learns legal moves on its own: its top choice is legal 0% of the time without the legal-move filter, and 0.8% with 7x more games. In games it is indistinguishable from random play.
- **Search wins games.** The team's alpha-beta engine is level with Stockfish depth 1 within error, and 2-ply search adds about 40 Elo to the ray network.
- **Self-play without a teacher did not help at this scale.** Search-guided self-play got worse after one round (+2 =27 -11 against its start). Wins/losses-only self-play stayed level after 8 rounds (+4 =32 -4).

Full table: `results/COMPARISON.md`. Games and crosstable: `results/tournament.json`.
Self-play logs: `runs/selfplay*_ray_s0/*.json` and `results/selfplay_*_h2h.json`.
