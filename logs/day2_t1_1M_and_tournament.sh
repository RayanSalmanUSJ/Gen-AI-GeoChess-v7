#!/usr/bin/env bash
# T1 for the three 1M runs (+ search for L), and the Step 4 round robin with the best dev run (scale_L_1M).
cd /c/Users/User/Documents/GeoChess
export PYTHONPATH=src
G=.venv/Scripts/python.exe
$G -m chess_policy.v6_eval --data data/v6/t1_d14 --model v5=checkpoints/policy_v5.pt \
   --model scale_S_1M=runs/v6/scale_S_1M/best.pt --model scale_M_1M=runs/v6/scale_M_1M/best.pt \
   --model scale_L_1M=runs/v6/scale_L_1M/best.pt --search scale_L_1M --reference v5 \
   --out results/v6/t1_1M.json > logs/step3_eval_t1_1M.log 2>&1 &
cd geochesszero/prototypes/03_method_comparison
../../../.venv/Scripts/python.exe tournament.py --config tournament_v6.json --openings 8 --workers 8 \
   --out results/tournament_v6.json > ../../../logs/step4_tournament.log 2>&1 && echo "TOURNAMENT_DONE $(date)" >> ../../../logs/run_overnight.log
wait
echo "T1_1M_DONE $(date)" >> ../../../logs/run_overnight.log
