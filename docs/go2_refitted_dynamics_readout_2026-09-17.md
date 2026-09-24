# Adapt physical-motion readouts to the refitted latent dynamics

**All three readout fits and the fixed branch evaluation are complete.** The
latent predictions and every encoded target remain exactly equal to the
completed predictor-refit assay. Only motion readout outputs changed. The
installed heads reloaded exactly and the fixed numerical-fit checks passed.

| Representation | Transfer 800-ms XY RMSE before (mm) | After (mm) | Yaw RMSE before (degrees) | After (degrees) |
| --- | ---: | ---: | ---: | ---: |
| JEPA | 6.561 | 5.462 | 0.862 | 0.708 |
| Supervised | 5.628 | 4.819 | 0.797 | 0.722 |
| Untrained | 5.644 | 5.509 | 0.733 | 0.547 |
| Command-history reference | 8.232 | 8.232 | 0.937 | 0.937 |

These are all 18 fixed transfer pulse contexts, not independent navigation
executions. Improvements follow predictor refitting plus the same motion-head
fit procedure; they cannot be attributed to the JEPA objective alone. Supervised
features still have lower XY error, and untrained features lower yaw error.
The original learned representations and all original results remain unchanged.

Branch result: `docs/go2_refitted_dynamics_readout_result_2026-09-17.json`.

## Broader recorded-navigation evaluation

**Complete:** all 2,404 windows from all four recordings were evaluated in
127.28 seconds; the process exited zero. Both failed returns are included.
Original JEPA predictions and the saved per-run baseline errors reproduced.

| Model | 700-ms XY RMSE (mm) | Action-increment XY RMSE (mm) | Yaw RMSE (degrees) |
| --- | ---: | ---: | ---: |
| Original JEPA | 9.201 | 6.070 | 0.898 |
| Refitted JEPA + adapted readout | 8.631 | 6.157 | 0.834 |
| Original supervised | 8.746 | 6.183 | 0.867 |
| Refitted supervised + adapted readout | 8.758 | 5.994 | 0.899 |
| Original untrained representation + fitted readout | 7.556 | 5.316 | 0.806 |
| Refitted untrained representation + adapted readout | 7.570 | 5.451 | 0.748 |
| Command history | 7.950 | 5.601 | 0.770 |
| Pose + command | 6.805 | 4.890 | 1.807 |

JEPA whole-window position error improves by 6.2%, with improvement in each
of the four recordings, but its action-increment error slightly worsens.
Whole-window errors improve for forward and arc actions and worsen for hold
and both in-place turns. JEPA remains behind both untrained-representation
readouts and command history in pooled position and yaw errors. Pose + command
has the lowest position error but worse yaw error. There is no established
JEPA advantage, no new closed-loop navigation result, and no model promotion.
The overlapping windows cover two exposed development layouts, not 2,404
independent trials. This result does not establish nonexecuted action accuracy.

Complete aggregate and per-run/action results:
`docs/go2_refitted_dynamics_navigation_forecast_result_2026-09-17.json`.

The next population was fixed before its outcomes: all 2,404 matched 700-ms
execution windows in all four completed return-routing-memory runs, including
both failed returns. Compare all three original readouts, all three refitted
predictor/adapted-readout models, command-history and pose-command controls on
identical RGB/body/control histories. Reproduce original JEPA predictions and
existing per-run XY errors; report prefix, action-increment and whole-window
errors, yaw and action breakdowns. No new model fitting, depth access or
alternative command execution occurs. Overlapping windows are not independent
trials, and prediction improvement does not establish navigation improvement.

Plan: `docs/go2_refitted_dynamics_navigation_forecast_plan_2026-09-17.json`.
Runner: `scripts/evaluate_go2_refitted_dynamics_navigation_forecasts_development.py`.
The evaluation runs on core 0 after every readout fit finishes. The small branch
evaluation uses core 8 independently; there is no timed native simulation.

The completed frozen-representation predictor study improves JEPA action-branch
discrimination, but its previous motion head is stale. Fit the same fixed ridge
readout for all three refitted predictors, using exactly the original 4,694
training contexts, scheduled draw weights and 33,904 valid motion rows. Preserve
the existing 256-feature hidden motion layer, penalty 1, explicit float64
standardization and four motion outputs; only their fitted final readout changes.
No transfer/navigation labels or new sensor data enter fitting. This isolates
the practical effect of the predictor refit followed by the same readout protocol;
it is not a pure representation-only comparison of every upstream component.

Keep the previous complete models/readouts and command reference as comparators.
Evaluate the unchanged 36 action-branch contexts after all fits finish. Require
exact equality of latent predictions, persistence targets, RGB interventions
and encoded future targets with the completed predictor-refit assay. Only
the motion readout should change. Report every model and all horizons, with
800 ms primary, and distinguish training from exposed development transfer.
No new navigation or model promotion follows automatically from these scores.

Plan: `docs/go2_refitted_dynamics_readout_plan_2026-09-17.json`.
Runner: `scripts/fit_go2_refitted_dynamics_readout_development.py`.
Output: `go2_refitted_dynamics_readout_v1_attempt_001` on the artifact volume.
The three preceding fit owners have exited. Reuse their capacity assessment:
one CPU process on core 8, one numerical thread, bounded per-trial inputs,
transient feature arrays, no new image/depth archive or concurrent simulation.
