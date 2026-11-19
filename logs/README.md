# logs/ — the record of chunk 1 (18–19 Nov 2026)

These are the logs and the exact driver scripts used to run chunk 1 on the team lead's PC.
They are kept so every number in `HANDOFF.md` can be traced back. **They are a record, not
something to run again**: the scripts contain that PC's absolute paths
(`C:\Users\User\Documents\GeoChess`), assume its two virtual environments (`.venv`,
`tools/venv-cpu`), and their work is finished. To redo any step, use the commands in
`HANDOFF.md` §4 or `CLAUDE.md`.

| File | What |
| --- | --- |
| `step0_*` | environment setup, tests, Gate 0 |
| `run_step1_labels.sh`, `step1_labels.log` | dev + train labelling (depth 10) |
| `step1_gate1_*`, `step1_spotcheck_t2.log` | Gate 1 checks |
| `run_overnight.sh`, `run_overnight_cpu.sh`, `run_overnight.log`, `overnight_cpu.log` | the overnight plan (GPU stages; T1 + D2 labelling) |
| `step2_stage_*.log` | each `v6_experiments` stage |
| `step2_gate2_t1.log`, `step2_gate2_t2.log` | the Gate 2 paired comparison |
| `step2_eval_t1.log`, `day2_eval*.sh` | T1 evaluation (subset) |
| `step3_*`, `day2_d2_finish.sh`, `day2_scaling_1M.sh` | D2 labelling, 1M scaling runs, T2 re-score |
| `step4_tournament.log`, `day2_t1_1M_and_tournament.sh` | first round robin |
| `resume_day2.sh` | restart after the 05:42 reboot on 19 Nov |
