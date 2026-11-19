#!/usr/bin/env bash
# CPU track (replaces the CPU half of run_overnight.sh): T1 at depth 14 in chunks of 250 so all
# 9 workers stay busy, then D2 labels -> assemble -> big_d10. run_overnight.sh waits for T1_DONE.
set -e
cd /c/Users/User/Documents/GeoChess
export PYTHONPATH=src
P=tools/venv-cpu/Scripts/python.exe
SF='C:\Users\User\Documents\GeoChess\tools\stockfish\stockfish.exe'
$P -m chess_policy.v6_data label data/v6/positions_holdout.npz --splits test --stride 10 --depth 14 \
    --chunk 250 --out data/v6/labels_t1_d14 --engine "$SF" --workers 9 > logs/step1_t1_label.log 2>&1
$P -m chess_policy.v6_data assemble data/v6/positions_holdout.npz --split test --labels data/v6/labels_t1_d14 \
    --out data/v6/t1_d14 --workers 9 >> logs/step1_t1_label.log 2>&1
echo "T1_DONE $(date)"
$P -m chess_policy.v6_data label data/v6/positions_extra.npz --splits train --depth 10 \
    --out data/v6/labels_extra_d10 --engine "$SF" --workers 9 > logs/step3_extra_label.log 2>&1
$P -m chess_policy.v6_data assemble data/v6/positions_extra.npz --split train --labels data/v6/labels_extra_d10 \
    --out data/v6/extra_d10 --workers 9 >> logs/step3_extra_label.log 2>&1
$P -m chess_policy.v6_data concat data/v6/train_d10 data/v6/extra_d10 --out data/v6/big_d10 >> logs/step3_extra_label.log 2>&1
echo "D2_DONE $(date)"
