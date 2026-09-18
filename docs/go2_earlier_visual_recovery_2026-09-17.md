# Earlier visual recovery: fixed two-run JEPA experiment

The completed four-run repeatability batch yielded one success and one failure
for each model. Exact sensor replay reproduced the last JEPA tracking failure
at frame 465. Weak-view recovery triggered at 47.10 s, was published at 47.29 s,
and first requested the reverse turn at 47.80 s. Tracking failed at 48.00 s.

Test one parameter change: increase weak-feature onset and aligned-view release
support from 48 to 72, retaining the 96-feature strong-reference threshold.
The saved reference, locality limit, actual pose fitting and every navigation
guard remain unchanged. This is an uncalibrated warning heuristic, not a new
confidence claim. Four focused onset/release/reference tests passed in 1.90 s.

The saved-state probe reproduced all 116 original planning recovery states.
Starting both variants from the same final strong reference (45.90 s), the
72-feature rule triggers at 46.60 s, 0.50 s earlier than the original 47.10 s.
The first probe incorrectly described a 27.70-s difference between different
episodes/references as warning gain. Its retained V1 receipt is superseded by
`earlier_visual_recovery_saved_probe_v2.json`, which explicitly separates the
same-reference comparison from the diverging full-trajectory state. Neither
probe executes an alternative action or proves tracking failure is prevented.

Run exactly two JEPA missions on exposed layout 1, sequentially, using unchanged
weights, six candidates, 4800-tick budget, ideal gyro and 2-mm depth noise.
No tuning between runs or additional attempts to obtain a favorable result.
Primary outcome remains independently verified goal-and-home arrival with no
disallowed contact. Also inspect tracking, actual warning/recovery timing,
view exclusions, coverage stalls and planning deadlines. Keep every failure.

Launcher: `scripts/run_go2_earlier_visual_recovery_development.py`.
Plan: `docs/go2_earlier_visual_recovery_plan_2026-09-17.json`.
The launcher reuses the completed repeatability runner/evaluator and records
the actual threshold/runtime in each launch. Per-run readout filenames and
schemas remain `view_replan_repeatability_readout_v1.json`.
Recordings use the project directory on `/mnt/steam_drive`; 38 GiB was free
after the previous batch. No other timed native job is running.

## Outcomes

Assignment 1 launched in session 90236, owner PID 4113813. Launch metadata
identifies `EarlierVisualRecoveryRuntime`, JEPA, thresholds 72/72/96 and two
planned assignments. Owner exited zero after full archival; evaluation session
96660 exited zero. **Budget exhausted at 480.88 simulated seconds, no arrivals,
zero contacts, no pipeline faults.** Tracking remained available for the full
mission. 989/1195 plans were on time (82.8%); sixteen views were interrupted.
The robot approached the goal but did not complete its quiet arrival dwell.
This is a navigation failure despite preserved tracking. The publisher recorded
79 recovery onsets, 77 at maximum camera support between 48 and 71, confirming
the new threshold actually affected execution. In the last 100 simulated
seconds, 233/249 plans selected pure turns and 242/249 were on time; 127 plans
requested weak-view recovery. Only one of those final plans had a translation
coverage rejection. Timing or the translation coverage filter alone therefore
does not explain the terminal turn cycle. Nine requested coverage patches were
resolved during the mission.

Assignment 2 launched unchanged in session 71067, owner PID 4116436. Owner
exited one after tracking failure and archival; evaluation session 3774 exited
zero. **No arrivals, zero contacts, tracking failure after 537 acquired frames.**
129/132 plans were on time (97.7%). Two camera views were interrupted.
Exact sensor replay matched all 532 accepted raw poses, then reproduced failure
at frame 532 (sensor stamp 54.70 s). All eight recent references failed their
unchanged registration checks. Preserve both full failure recordings.

The second failure was not simply a late warning. Recovery triggered at
51.90 s with [50, 69] features, using a strong view at 50.80 s. Publication
cancelled the old command at 52.08 s. However, plans 504–520 continued selecting
left turns although their preferred action before clearance filtering was right
turn toward the saved heading. At frame 504, the recorded model predicted
minimum stored clearance 0.47303 m for right turn, versus 0.48172 m for left
turn. Right turn cleared the nominal 0.45-m footprint but failed the additional
0.03-m reserve. The alternative-turn recovery chose the long way around, away
from the visually supported heading. The right turn became eligible at frame
524 and was requested at 54.20 s, only 0.50 s before tracking failed.

The field `nominal_predicted_path_clear` in the final selection already includes
turn-reserve eligibility; it must not be interpreted as nominal 0.45-m footprint
clearance. `nominal_footprint_path_clear` preserves the separate nominal result.

## Complete result and next diagnosis

The fixed two-run experiment is complete: **0/2 round trips, no arrivals, zero
contacts.** One run kept tracking but exhausted its budget near the goal; the
other lost tracking after the clearance selector directed recovery away from
the requested view. The higher threshold is not established as an improvement
and is not promoted as a fix. Do not add further repetitions to this batch.

The saved-data check is complete:
`scripts/probe_go2_view_recovery_forecast_clearance_development.py` reconstructed
the recorded maps through the final recovery, matched stored floor/obstacle
counts and all neural segment-clearance values for seven planning states
(504–528, every four frames). Delivered sensor digests were checked. No native
state or wall geometry entered this comparison.

At frame 504, observed current clearance was 0.48450 m. The neural forecast
predicted 0.47303 m for right turn, failing the unchanged 0.48-m requirement;
the fitted pose-command forecast predicted 0.48133 m and the command-history
forecast 0.48450 m, both passing. Across the seven saved states, neural right
turn passed the isolated reserve rules twice, pose-command six times, and
command-history seven times. These diagnostics include full/stepwise reserve
recovery but do not reexecute the whole selector, hold-relative overrides,
stopping or dispatch. They are not forecasts of alternative mission outcomes.
The receipt is `view_recovery_forecast_clearance_probe_v1.json` in assignment 2.

This identifies a concrete interaction between learned turn-drift forecasts
and visual recovery. It does not prove a falsely blocked safe turn: the right
turn was unexecuted at those states, and a simpler forecast can underestimate
drift. Next compare action-specific forecast errors and investigate the
recovery selector's long-way turn behavior. Keep this negative threshold result
separate; do not continue threshold tuning or launch a larger maze cohort yet.
Fresh-layout reliability, JEPA advantage,
realistic sensing, real-time execution and hardware validation remain unproven.
