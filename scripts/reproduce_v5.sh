#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
experiment_dir="${1:-reproduction_v5}"
if [[ -e "$experiment_dir" ]]; then
  echo "Choose a fresh directory: $experiment_dir already exists." >&2
  exit 1
fi
mkdir -p "$experiment_dir/results"
python -m pytest -q
if [[ ! -f corpus/manifest.json ]]; then
  python -m chess_policy build-data --pgn data/fishtest_train.pgn.gz --out corpus
fi
if [[ ! -f holdout/manifest.json ]]; then
  python -m chess_policy build-data --pgn data/fishtest_holdout.pgn.gz --out holdout --force-test --exclude corpus
fi
if [[ ! -f v4_holdout/manifest.json ]]; then
  python scripts/prepare_v4_holdout.py --report "$experiment_dir/results/v4_manifest.json"
fi
python scripts/audit_v5_data.py --protected-roots corpus holdout v4_holdout --out "$experiment_dir/results/data_audit.json"
for arm in control targeted; do
  python -u -m chess_policy.refinement_v5 --base-data corpus --mode "$arm" --output "$experiment_dir/runs/$arm"
done
python scripts/select_v5.py --base-data corpus --runs "$experiment_dir/runs" --destination "$experiment_dir/checkpoints" --out "$experiment_dir/results/selection.json"
if [[ ! -f v5_holdout/manifest.json ]]; then
  python scripts/prepare_v5_holdout.py --report "$experiment_dir/results/fresh_manifest.json"
fi
python scripts/evaluate_v5.py --data v5_holdout --checkpoints "$experiment_dir/checkpoints" --out "$experiment_dir/results"
echo "Finished. Evidence: $experiment_dir/results"
