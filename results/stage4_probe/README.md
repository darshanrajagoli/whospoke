# Stage-4 evidence (November 2026)

What was run for D37–D39, kept so the claims there can be checked. None of it is read by the scripts.

| folder | what it is |
|---|---|
| `2026-11-10_s4eval_r1/` | The full scored run (D39): `run_meta.json` (command, model file, server settings, machine, timing) and `eval_stdout.log` (every report's time and every conversation's cpWER, then the totals). Its output is `results/eval_postprocess.csv` and `results/eval_postprocess/`. |
| `2026-11-10_live/`, `2026-11-10_recheck/` | Two real end-to-end Stage-4 runs on a stored Stage-3 transcript with the real Airavata server, before and after D38: the report (`.md`, `.json`), the settings and the log. |
| `actions_g07/` | The action-item probe on test group 7 only (D37): the two probe scripts, their logs and `findings.json`. |

The reports these runs replaced (made in October on two machines), their CSV and their run summaries are in git at
commit `df497d8` (`git show df497d8:results/eval_postprocess.csv`).
