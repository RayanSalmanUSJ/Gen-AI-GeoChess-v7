#!/usr/bin/env bash
# Prototype 3: compare methods on Prototype 2's benchmark.
#   STOCKFISH=/path/to/stockfish NOVEMBER=/path/to/november ./run_all.sh
# Needs Prototype 2's data and checkpoints (run ../02_value_distillation/run_all.sh first).
set -euo pipefail
cd "$(dirname "$0")"
: "${STOCKFISH:?set STOCKFISH to a Stockfish binary}"
: "${NOVEMBER:?set NOVEMBER to the folder holding the team module*.cpp files}"
export STOCKFISH OMP_NUM_THREADS=1
mkdir -p runs results data

# Team's C++ engine (november.zip, unchanged) behind a UCI adapter
engines/november_adapter/build.sh "$NOVEMBER"

# GPT-style move model: same games and targets as Prototype 2, 3 seeds
for s in 0 1 2; do python gpt_movelm.py train --seed $s --out runs/gpt_s$s; done
# ... and with ~6x more games (labelled at depth 4)
[ -f data/gpt_scale_d4.jsonl ] || (cd ../02_value_distillation && \
  python datagen.py --out ../03_method_comparison/data/gpt_scale_d4.jsonl --games 1200 --depth 4 --workers 2 --seed 77)
python gpt_movelm.py train --seed 0 --extra data/gpt_scale_d4.jsonl --out runs/gptscale_s0
python gpt_movelm.py eval --runs 'runs/gpt*_s*' --out results/gpt_eval.json

# Self-play from the ray model, no Stockfish labels
python selfplay.py --start ../02_value_distillation/runs/ray_s0/best.pt --iterations 3 --out runs/selfplay_ray_s0

# One table for everything, then the round-robin
python benchmark_all.py
sed "s|uci:stockfish:|uci:${STOCKFISH}:|" tournament_players.json > results/tournament_players.local.json
python tournament.py --config results/tournament_players.local.json --openings 8 --workers 2 --out results/tournament.json

# Local LLMs (on a machine with Ollama):
#   python llm_player.py bench --model llama3.2:3b --out results/llm_llama3.2-3b.json
#   python llm_player.py bench --model qwen3.5:9b  --out results/llm_qwen3.5-9b.json
#   then re-run benchmark_all.py, and add {"name": ..., "spec": "llm:llama3.2:3b"} to the tournament.
