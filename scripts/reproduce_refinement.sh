#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
experiment_dir="${1:-reproduction_v4}"
if [[ -e "$experiment_dir" ]]; then
  echo "Choose a fresh output directory: $experiment_dir already exists." >&2
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
python scripts/audit_refinement_data.py --out "$experiment_dir/results/data_audit.json"
for arm in control targeted; do
  python -u -m chess_policy.refinement --base-data corpus --mode "$arm" --output "$experiment_dir/runs/$arm"
done
python scripts/select_refinement.py --runs "$experiment_dir/runs" --out "$experiment_dir/checkpoints/policy_v4.pt" --report "$experiment_dir/results/selection.json"
if [[ ! -f v4_holdout/manifest.json ]]; then
  python scripts/prepare_v4_holdout.py --report "$experiment_dir/results/fresh_manifest.json"
fi
cp v4_results/v3_loss_audit.json "$experiment_dir/results/v3_loss_audit.json"
python scripts/evaluate_refinement.py --data v4_holdout --checkpoints "$experiment_dir/checkpoints" --out "$experiment_dir/results"
echo "Finished. Results: $experiment_dir/results"
