#!/usr/bin/env bash
# Reproduce Prototype 2 end to end on CPU.
#   STOCKFISH=/path/to/stockfish-15.1 ./run_all.sh
# Takes roughly 2-3 hours on a 2-core CPU; a GPU makes the training part minutes.
set -euo pipefail
cd "$(dirname "$0")"
: "${STOCKFISH:?set STOCKFISH to a Stockfish binary (15.1 used for the reported results)}"
export STOCKFISH OMP_NUM_THREADS=1
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-6}"

mkdir -p data runs results

# 1. Teacher-labelled data: training at depth 8, held-out test games at depth 12.
[ -f data/train.jsonl ] || python datagen.py --out data/train.jsonl --games 204 --depth 8 --workers 2 --seed 1
[ -f data/test_games.jsonl ] || python datagen.py --out data/test_games.jsonl --games 24 --depth 8 --workers 2 --seed 9001
[ -f data/test.jsonl ] || python relabel.py data/test_games.jsonl data/test.jsonl --depth 12 --every 2 --workers 2
python dataset.py data/train.jsonl data/train.npz
python dataset.py data/test.jsonl  data/test.npz --exclude data/train.jsonl

# 2. Train the three attention variants x seeds (two at a time, one thread each).
for seed in $SEEDS; do
  for v in standard static ray; do
    [ -f "runs/${v}_s${seed}/best.pt" ] && continue
    python train.py --data data/train.npz --variant "$v" --seed "$seed" --epochs "$EPOCHS" \
      --out "runs/${v}_s${seed}" > "runs/${v}_s${seed}.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 5; done
  done
done
wait

# 3. Matches between variants, seed by seed. Plain argmax play loops into threefold
#    repetitions (the network never sees game history), so the main matches use
#    --avoid-repetition for both sides; the plain version is kept for comparison.
for seed in $SEEDS; do
  for pair in "ray standard" "static standard" "ray static"; do
    set -- $pair
    python arena.py --a "model:runs/${1}_s${seed}:policy" --b "model:runs/${2}_s${seed}:policy" \
      --openings data/test_games.jsonl --avoid-repetition --out "results/norep_${1}_vs_${2}_s${seed}.json"
    python arena.py --a "model:runs/${1}_s${seed}:policy" --b "model:runs/${2}_s${seed}:policy" \
      --openings data/test_games.jsonl --out "results/arena_${1}_vs_${2}_s${seed}.json"
  done
done

# 4. Yardsticks: random (floor) and Stockfish depth 1 (ceiling at this scale).
for v in ray static standard; do
  python arena.py --a "model:runs/${v}_s0:policy" --b random     --openings data/test_games.jsonl --out "results/arena_${v}_vs_random.json"
  python arena.py --a "model:runs/${v}_s0:policy" --b sf:depth=1 --openings data/test_games.jsonl --out "results/arena_${v}_vs_sf_d1.json"
done
python arena.py --a model:runs/ray_s0:search --b sf:depth=1 --openings data/test_games.jsonl --out results/arena_raysearch_vs_sf_d1.json

# 5. Benchmark against the deeper teacher (policy, 1-ply comparator, 2-ply search on every 4th position).
python evaluate.py --test data/test.npz --runs 'runs/*_s*' --comparator --search-every 4 --threads 2 --out results/eval.json

# 6. Hindsight blame report on one lost game, then the summary.
python blame.py --arena results/arena_ray_vs_sf_d1.json --side a --out results/blame_example.json
python report.py
