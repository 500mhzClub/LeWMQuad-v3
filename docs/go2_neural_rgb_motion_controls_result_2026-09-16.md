# Same-window motion controls for the completed RGB pilot

The frozen pose/command predictor has lower XY endpoint RMSE than the corrected
neural predictor in **30/36 pilot runs**. The learned version wins in six.
Both alternatives are evaluated on the same actual selected-action windows;
this is stronger prediction evidence than comparing errors on separate policies'
different trajectories. It still does not establish the result of executing
the alternative navigation policy.

All 36 assignments, including the tracking failure, contribute their available
matched windows. The readout covers 15,801 overlapping windows through 700 ms
on two maze layouts. The pooled table below is descriptive, not 15,801
independent trials or a statistical generalization claim.

| Executed action group | Windows | Raw neural RMSE mm | Corrected neural RMSE mm | Fitted pose/command RMSE mm |
|---|---:|---:|---:|---:|
| All | 15,801 | 21.013 | 7.183 | 6.500 |
| Hold | 669 | 10.184 | 4.498 | 4.519 |
| Translation | 8,390 | 25.841 | 8.218 | 7.152 |
| Short translation pulse | 346 | 18.156 | 10.659 | 9.663 |
| Turn | 6,396 | 13.648 | 5.535 | 5.486 |

Counts of runs with lower corrected-neural error are JEPA full 2/6, JEPA
no-RGB 0/6, direct full 0/6, direct no-RGB 2/6, supervised-rollout full 0/6 and
supervised-rollout no-RGB 2/6. These repetitions are not independent maze units.
The earlier yaw readout already found command-based yaw prediction better in
all 36 runs. Neither result supports assuming the learned motion model is
currently adding useful prediction accuracy over these simple controls.

The alternatives were saved during each original neural run, not refitted
after observing these mazes. They use the same observation-time pose history
and requested action prefix. `FrozenPoseCommandXY.predict` uses the actual
terminal-pulse command sequences, so short-pulse windows are included with
their proper commands. The fitted model identity is constant across all 36
runs. Recomputed corrected-neural errors match the completed executed-window
evaluations within 1e-12 m. Native displacement is an evaluation target only;
no native state enters either deployed predictor.

The controller-correctness follow-up gives a concrete example of the remaining
error: all 13 selected return-home holds in the corrected maze-0 case forecast
arrival inside 20 mm, but actual matched endpoints remain outside. On those
windows, fitted/corrected-neural RMSE is 4.538/6.269 mm; the fitted alternative
still falsely passes the distance/speed test in six windows. Coordinate
consistency alone does not address biased stopping forecasts or make point
predictions a calibrated arrival guarantee.

Artifacts:

- `go2_neural_rgb_motion_controls_readout_v1_attempt_001/result.json` under
  the configured development-artifact base: every run and action-group result.
- `scripts/read_go2_neural_rgb_motion_controls_development.py`: readout from
  saved alternatives and matched execution windows; existing output is preserved.
- `docs/go2_mission_coordinate_followup_2026-09-16.md`: completed four-run
  coordinate follow-up and the retained terminal-error diagnosis.

Next, inspect whether the training inputs and action histories cover the
measured motion state, braking, stopping and short pulses actually encountered
online. Use this to specify a bounded data/model successor and matched simple
controls before fitting again. Do not add another training seed or a larger
unchanged-task sweep to search for a favorable result. Better prediction on
independent situations must then be tested through actual online choices and
maze outcomes; this offline readout does not complete that requirement.
