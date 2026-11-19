#!/usr/bin/env bash
# After D2 labels: the 1M-position scaling runs (S/M/L, 4 epochs), then re-score T2 for all runs
# once the T1 subset eval has finished (stage eval skips nothing, so it re-runs T2 for everyone).
cd /c/Users/User/Documents/GeoChess
export PYTHONPATH=src
G=.venv/Scripts/python.exe
until grep -q D2_DONE logs/overnight_cpu.log; do sleep 60; done
$G -m chess_policy.v6_experiments --stage scaling > logs/step3_stage_scaling_1M.log 2>&1 || exit 1
echo "SCALING_1M_DONE $(date)" >> logs/run_overnight.log
until grep -q T1_EVAL_DONE logs/run_overnight.log; do sleep 60; done
$G -m chess_policy.v6_experiments --stage eval --t1 data/v6/no_t1_this_pass > logs/step3_stage_eval.log 2>&1 \
  && echo "EVAL_WITH_1M_DONE $(date)" >> logs/run_overnight.log
