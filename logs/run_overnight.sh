#!/usr/bin/env bash
# Overnight plan, started 18 Nov. Two tracks after the train labels are ready:
#   CPU: T1 labels (depth 14: depth 16 measured at ~4.2 h > CLAUDE.md's ~4 h limit) -> assemble,
#        then D2 labels (707,896 extra train positions, depth 10) -> assemble -> concat big_d10
#   GPU: stages tau, core, attention, moe, scaling (S/M/L x 70K/140K/279K; 1M added later), then
#        eval on T2 + T1 once T1 is assembled.
cd /c/Users/User/Documents/GeoChess
P=tools/venv-cpu/Scripts/python.exe
G=.venv/Scripts/python.exe
SF='C:\Users\User\Documents\GeoChess\tools\stockfish\stockfish.exe'
export PYTHONPATH=src
until grep -q TRAIN_DONE logs/step1_labels.log; do
  grep -qE "Traceback|Error" logs/step1_labels.log && { echo "train labelling failed"; exit 1; }
  sleep 30
done
echo "train labels ready $(date)"
(
  set -e
  $P -m chess_policy.v6_data label data/v6/positions_holdout.npz --splits test --stride 10 --depth 14 \
      --out data/v6/labels_t1_d14 --engine "$SF" --workers 9 > logs/step1_t1_label.log 2>&1
  $P -m chess_policy.v6_data assemble data/v6/positions_holdout.npz --split test --labels data/v6/labels_t1_d14 \
      --out data/v6/t1_d14 --workers 9 >> logs/step1_t1_label.log 2>&1
  echo "T1_DONE $(date)"
  $P -m chess_policy.v6_data label data/v6/positions_extra.npz --splits train --depth 10 \
      --out data/v6/labels_extra_d10 --engine "$SF" --workers 9 > logs/step3_extra_label.log 2>&1
  $P -m chess_policy.v6_data assemble data/v6/positions_extra.npz --split train --labels data/v6/labels_extra_d10 \
      --out data/v6/extra_d10 --workers 9 >> logs/step3_extra_label.log 2>&1
  $P -m chess_policy.v6_data concat data/v6/train_d10 data/v6/extra_d10 --out data/v6/big_d10 >> logs/step3_extra_label.log 2>&1
  echo "D2_DONE $(date)"
) > logs/overnight_cpu.log 2>&1 &
for s in tau core attention moe scaling; do
  echo "=== $s $(date)"
  $G -m chess_policy.v6_experiments --stage $s > logs/step2_stage_$s.log 2>&1 || { echo "stage $s failed"; exit 1; }
done
echo "GPU stages done $(date); waiting for T1"
until grep -q T1_DONE logs/overnight_cpu.log; do
  grep -qE "Traceback|Error" logs/overnight_cpu.log logs/step1_t1_label.log && { echo "T1 failed"; break; }
  sleep 30
done
$G -m chess_policy.v6_experiments --stage eval --t1 data/v6/t1_d14 > logs/step2_stage_eval.log 2>&1 && echo "EVAL_DONE $(date)"
wait
echo "ALL_DONE $(date)"
