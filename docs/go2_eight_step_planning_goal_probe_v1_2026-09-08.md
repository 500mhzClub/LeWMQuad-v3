# Eight-step nominal path planning probe V1

The completed short-horizon probe failed both fixed models at tick 28. Its saved
banks predicted later nominal-clearance conflicts against already-observed
cells while selected first steps still passed. Test a distinct planning change
with the same final seed-2026091001 full-JEPA and full-direct models, without
retraining, model selection, residual correction or changing the action bank.

For waypoint decisions, score the existing distance/bearing potential and
cumulative contact cost at the trained 800-ms endpoint. Keep coefficients
0.4 m, 0.35 m and 1.2 m unchanged. For view acquisition, retain the existing
100-ms scan utility and phase allowances. In both phases require every nominal
segment, from the current observed pose through all eight 100–800-ms predicted
poses, to pass the existing strict 0.45-m occupied-square clearance test. Check
all currently observed occupied squares. Require exact reproduction of the
old first-segment check and preserve its first-step articulated surface veto.

Execute only one 100-ms command and observe again. The planning horizon does
not extend open-loop execution, the 240-tick mission budget or the ten-command
drain. Preserve public sensing, observed mapping, overlap-retention observer,
route and view logic, sensor/physical stops, 0.04-m observed arrival and the
original native 0.06-m one-second quiet/contact gate. No candidate means the
existing terminal stop. Keep both failures visible.

The path guard is a nominal prediction check, not a model-error, articulated
motion, future gait, continuous curved-motion, unknown-space or terminal
viability certificate. Later articulated surface checks are not added; the
existing first-step check remains in force. No clearance threshold is reduced.

Freeze the source closure, completed short-horizon fit/readout and predecessor
native/readout identities before the exclusive
`go2_eight_step_planning_goal_probe_v1_attempt_001` root. Run both fresh CPU scenes
on `family_episode_039`, in the same order fixed by seed 2026091501. Keep the
passing four-process native benchmark and completed two-scene resource evidence,
check 32 GiB available RAM and an 8 GiB output allowance above the 40 GiB reserve,
and use two fresh processes with one numerical thread each. Monitor resources
and report complete-iteration timing; physics remains paused during compute.

Verify exact original warmup public/physical prefixes. Replay every decision
and command with a fresh unchanged copy of its assigned model and the original
actuator/native-goal auditors. Reverify all bound sources and artifacts after
execution. Preserve terminal failures; no retry or resume. The separate mapping
optimization prototype is not adopted in this intervention.

This reused integration layout supplies zero independent novel mazes. Success
would still leave matched reactive/nonpredictive, planning and memory baselines,
physical backtracking, independent layouts, realistic continuous timing and
bounded hardware evidence outstanding.
