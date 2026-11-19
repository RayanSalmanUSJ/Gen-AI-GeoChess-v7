#!/usr/bin/env bash
# T1 eval (subset), split out of day2_eval.sh after its SUMMARY.md write crashed (fixed).
cd /c/Users/User/Documents/GeoChess
export PYTHONPATH=src
G=.venv/Scripts/python.exe
M="--model v3=checkpoints/pilot.pt --model v4=checkpoints/policy_v4.pt --model v5=checkpoints/policy_v5.pt"
for r in A_played_s0 A_played_s1 A_played_s2 B_soft_s0 B_soft_s1 B_soft_s2 C_value_s0 C_value_s1 C_value_s2 \
         att_plain_s0 att_hard_s0 att_hard_ray_s0 att_hard_only_s0 moe_off_s0 \
         scale_S_70K scale_S_140K scale_S_279K scale_M_70K scale_M_140K scale_M_279K scale_L_70K scale_L_140K scale_L_279K; do
  M="$M --model $r=runs/v6/$r/best.pt"
done
$G -m chess_policy.v6_eval --data data/v6/t1_d14 $M --search C_value_s0 --search att_hard_ray_s0 --search scale_L_279K \
   --reference v5 --out results/v6/t1.json > logs/step2_eval_t1.log 2>&1 || exit 1
$G -c "from chess_policy.v6_experiments import write_summary; write_summary()" >> logs/step2_eval_t1.log 2>&1
echo "T1_EVAL_DONE $(date)" >> logs/run_overnight.log
