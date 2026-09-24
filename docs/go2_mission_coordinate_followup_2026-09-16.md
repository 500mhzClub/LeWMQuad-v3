# Mission-coordinate correction: development follow-up

Status: the unchanged 36-assignment pilot is complete and retained. Runtime
integration and all four native follow-up assignments are complete: four verified
round trips and zero contacts. The correction showed no speed benefit in either
targeted pair. Whole-mission timing is also affected by asynchronous trajectory
differences before the terminal intervention activates. Keep the correction
available explicitly; do not promote it as an established navigation improvement.
Thirteen focused coordinate/scoring/arrival tests pass. The original mode's
candidate scores and alignment scales exactly reproduce 1,104 saved plans
from the two selected pilot cases. This does not promise identical asynchronous
native trajectories on repetition.

Launcher: `scripts/run_go2_mission_coordinate_followup_development.py --assignment N`.
Evaluator: `scripts/evaluate_go2_mission_coordinate_followup_development.py --assignment N`.
The updated sources are shared by both modes; each selected plan records the
coordinate treatment. The implementation captures the observation's planning
goal, applies the matrix only during terminal position approach, and clears it
after selection. Predicted body-XY speed keeps its original definition; measured
3D quiet-dwell checks remain separate and unchanged.

The pilot exposed a roughly 3 mm difference between its terminal planning
target and mission-arrival metric. The discrepancy can make a predicted stop
look inside the 20 mm arrival radius while the current observed goal remains
outside. A coordinate-consistent post-hoc calculation disqualified 69 recorded
hold decisions in five of the first 28 runs. It does not establish a better
executed action, and 43 of 44 still-eligible holds in the detailed maze-0 case
remained outside the observed arrival radius later. This correction addresses
coordinate consistency, not all forecast bias or terminal turning.

Use `planar_body_to_initial_xy` to score terminal displacement in the same
initial-frame XY metric as the mission. Capture the planning goal used for
that observation, derive its body-XY target under the gravity-horizontal motion
assumption, and apply the resulting metric consistently to positional progress,
predicted-stop distances and the arrival-entry check. Keep body-frame forecasts
unchanged for clearance/stopping guards. Keep trained weights, residual fits,
prediction horizons, action commands, translation pulses, heading fallback,
measured settling and independent physical arrival checks unchanged. Record
the active metric and target in the selected plan.

The assumption eliminates unpredicted body Z using constant gravity-aligned
height. Its evaluator-only error was at most 0.2042 mm on the diagnosed held
windows; this is not a bound on moving actions or actual vertical dynamics.

Run a targeted four-assignment comparison in this fixed order:

| Assignment | Existing model and maze | Coordinate mode |
|---|---|---|
| 1 | seed 2026091402, no-RGB JEPA, pilot maze 0 | Original |
| 2 | seed 2026091402, no-RGB JEPA, pilot maze 0 | Consistent |
| 3 | seed 2026091402, full-input JEPA, pilot maze 1 | Consistent |
| 4 | seed 2026091402, full-input JEPA, pilot maze 1 | Original |

The first case exhibited the detailed holding mismatch; the second exhibited
the distinct terminal turning/arrival-entry interaction. These are deliberately
selected, already-exposed debugging cases, not new environmental replication
or an RGB/JEPA comparison. Both modes use the same updated source and identical
physical/sensor settings, with the original mode retaining the original math.
Use sequential native owners and the pilot's CPU groups. Reuse existing
physical, forecast, command and terminal-window evaluators.

Report every success or failure, contacts, observed and physical arrivals,
terminal and nonterminal time, on-time plans and peak lag. Compare predicted
hold eligibility and heading-restoration decisions, while recognizing that
closed-loop trajectories may diverge before a terminal window. Keep all original
pilot results. Do not turn a worse outcome into a favorable result by changing
the 20 mm observed or 40 mm physical arrival radius.

This is a bounded controller-correctness experiment. Regardless of its outcome,
the broader goal still needs learned-prediction benefit against matched simple
motion controls, more independent environments, realistic sensor/timing failure
tests and physical-platform validation. Do not respond to the current mixed
learning results with another larger training sweep on the unchanged task.

Assignment 1 (original metric, no-RGB JEPA, maze 0) verified goal/home at
frames 1,332/1,821: 1,823 poses, zero contacts, maximum position error
8.172 mm, 438/446 plans on time and 182.42 simulated seconds. Owner exit 0
after 4:26.08 with no swaps. Actual model/input/correction checks and the
original-coordinate treatment passed. Peak simulator lag was 450.782 ms.
This baseline repeats the configuration, not the earlier pilot's asynchronous
trajectory or its 232.88 s completion time. Assignment 2 uses the consistent
metric with the same model, fit, maze, sensors and shared sources.

Assignment 1's terminal windows were 5.4 s outbound and 3.6 s return, versus
46.2/15.6 s in its original pilot configuration. Thus the fresh baseline alone
reduced terminal time by 52.8 s and whole-mission time by 50.46 s; outside
terminal windows it was 2.34 s slower. This repeat variability is substantial
relative to the proposed intervention. Do not attribute a comparison against
the old 232.88 s pilot run to the correction or claim statistical improvement
from the small targeted follow-up.

Assignment 2 (consistent metric, no-RGB JEPA, maze 0) verified goal/home at
frames 1,388/2,084: 2,086 poses, zero contacts, maximum position error
9.649 mm, 499/508 plans on time and 208.74 simulated seconds. Owner exit 0
after 5:02.18 with no swaps; model/input/correction checks passed. The corrected
coordinate treatment was verified on all 45 active terminal plans. Peak lag
was 442.610 ms. It was 26.32 s slower than its fresh original-mode comparator;
retain this result without tuning the remaining assignments. Assignment 3
(consistent metric, full-input JEPA, maze 1) launched next.

The maze-0 paired report has 178 identical common source identities and
inspected PNG/SVG trajectories. Original/consistent terminal windows were
5.4/1.6 s outbound and 3.6/21.1 s on return. The consistent run therefore
spent 13.7 s more in terminal windows and 12.62 s more outside them. Its
return window contained 17.9 s zero command. Predicted-hold eligibility was
2/14 plans, and arrival-entry heading restorations were 5/0. The coordinate
correction is exercised but is not a demonstrated navigation improvement.

All 14 corrected-run predicted-arrival holds were on time and have matched
executed command sequences through 700 ms. The outbound hold worked. All
13 return holds predicted both terminal distances inside 20 mm, yet the later
observed goal and the evaluator's planar projection of actual displacement
remained outside. Their corrected endpoint errors were 3.511–8.551 mm.
This residual error persists after coordinate consistency; 800 ms is not
evaluated and the overlapping windows are not independent experiments.
The pair retains `corrected_terminal_hold_diagnostic_v1.json`.

For those same 13 return command windows, saved raw-neural/corrected-neural/
fitted-pose-command endpoint RMSE was 7.736/6.269/4.538 mm. Both neural versions
passed the predicted distance/speed arrival test in all 13; the fitted control
did so in six. These are stored alternatives on the actually executed hold
windows, not outcomes of a different navigation policy, and alternative
clearance selection was not evaluated. The comparison retains
`corrected_terminal_hold_motion_controls_v1.json`. Keep this negative evidence
when choosing subsequent learning work; do not assume the coordinate fix
resolves the predictor's terminal dynamics error.

Assignment 3 (consistent metric, full-input JEPA, maze 1) verified goal/home
at frames 1,041/1,608: 1,610 poses, zero contacts, maximum position error
5.580 mm, 383/394 plans on time and 161.16 simulated seconds. Owner exit 0
after 4:08.59 with no swaps. Actual model/input/correction checks passed;
the corrected metric was verified on three active terminal plans. Peak lag
was 536.859 ms. The diagnosed maze-0 pair's depth was retired under policy,
preserving its complete non-depth evidence and original pilot references.
Assignment 4 (original metric, same full-input JEPA on maze 1) launched last.

The maze-0 command traces first diverge at measured time 15.42 s: original
applies yaw 0.45 rad/s under the nominal obstacle check, while consistent
applies approximately 0.10 rad/s with `CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE`.
The first corrected-metric plan is frame 1,372 at measured time 138.7 s.
Thus trajectories differ long before the treatment activates; whole-mission
timing is not an isolated coordinate-effect estimate. The pair retains
`pre_intervention_command_divergence_v1.json`. This is a direct observed
timing/observation difference, not a claim to know its operating-system cause.

Assignment 4 (original metric, full-input JEPA, maze 1) verified goal/home at
frames 1,016/1,588: 1,590 poses, zero contacts, maximum position error
6.135 mm, 381/390 plans on time and 159.14 simulated seconds. Owner exit 0
after 4:05.84 with no swaps; actual model/input/correction/coordinate checks
passed. Peak simulator lag was 559.884 ms. The final pair has 178 identical
common sources and inspected PNG/SVG trajectories. Its original/consistent
terminal windows were 2.3/2.3 s outbound and 1.7/1.6 s return. Both had two
predicted-arrival hold selections and no arrival-entry heading restorations.

| Targeted case | Original completion s | Consistent completion s | Original terminal s | Consistent terminal s |
|---|---:|---:|---:|---:|
| No-RGB JEPA, maze 0 | 182.42 | 208.74 | 9.0 | 22.7 |
| Full-input JEPA, maze 1 | 159.14 | 161.16 | 4.0 | 3.9 |

The complete four-assignment result is retained at
`go2_mission_coordinate_complete_comparison_v1_attempt_001/result.json` under
the development-artifact base. Both pairs share the same 178 source identities.
No native owner remains active. The next scientific focus is prediction quality
and training/execution coverage, not another series of terminal-rule changes.
The full pilot's same-window motion-control readout is documented in
`docs/go2_neural_rgb_motion_controls_result_2026-09-16.md`.
