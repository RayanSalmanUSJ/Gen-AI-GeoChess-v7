#!/usr/bin/env bash
# Step 1 labelling chain (Stockfish 15.1, depth 10). Resumable: rerun to continue.
set -e
cd /c/Users/User/Documents/GeoChess
P=tools/venv-cpu/Scripts/python.exe
SF='C:\Users\User\Documents\GeoChess\tools\stockfish\stockfish.exe'
W=${WORKERS:-11}
$P -m chess_policy.v6_data label data/v6/positions_train.npz --splits dev --depth 10 --out data/v6/labels_dev_d10 --engine "$SF" --workers $W
$P -m chess_policy.v6_data assemble data/v6/positions_train.npz --split dev --labels data/v6/labels_dev_d10 --out data/v6/dev_d10 --workers $W
echo DEV_DONE
$P -m chess_policy.v6_data label data/v6/positions_train.npz --splits train --depth 10 --out data/v6/labels_train_d10 --engine "$SF" --workers $W
$P -m chess_policy.v6_data assemble data/v6/positions_train.npz --split train --labels data/v6/labels_train_d10 --out data/v6/train_d10 --workers $W
echo TRAIN_DONE
