#!/usr/bin/env bash
# Finish D2 after chunk 297 was found corrupt (written during the 05:42 reboot): relabel it, assemble, concat.
set -e
cd /c/Users/User/Documents/GeoChess
export PYTHONPATH=src
P=tools/venv-cpu/Scripts/python.exe
SF='C:\Users\User\Documents\GeoChess\tools\stockfish\stockfish.exe'
{
$P -m chess_policy.v6_data label data/v6/positions_extra.npz --splits train --depth 10 \
    --out data/v6/labels_extra_d10 --engine "$SF" --workers 10
$P -m chess_policy.v6_data assemble data/v6/positions_extra.npz --split train --labels data/v6/labels_extra_d10 \
    --out data/v6/extra_d10 --workers 10
$P -m chess_policy.v6_data concat data/v6/train_d10 data/v6/extra_d10 --out data/v6/big_d10
} >> logs/step3_extra_label.log 2>&1
echo "D2_DONE $(date)" >> logs/overnight_cpu.log
