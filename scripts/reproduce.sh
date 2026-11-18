#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e '.[dev,serve]'
python -m pytest -q
chess-policy build-data --pgn data/fishtest_train.pgn.gz --out corpus
chess-policy build-data --pgn data/fishtest_holdout.pgn.gz --out holdout --force-test --exclude corpus
python scripts/audit_corpus.py corpus holdout
chess-policy train --data corpus --out runs/reproduction --config configs/pilot.json
chess-policy evaluate --data holdout --checkpoint runs/reproduction/best.pt --out results/reproduction.json
