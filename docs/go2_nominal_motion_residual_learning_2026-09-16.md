# Learning deviations from commanded motion

The prospective-switch experiment left every neural model worse overall than
command integration, and all three regressed on braking. This successor tests
a different outcome parameterization on the same 4,514 training contexts,
same schedule, seed 2026091001, batch size six and 1,200 updates per method.
JEPA, direct and supervised rollout all run; no checkpoint or seed search.

The nominal reference integrates the known 100-ms commands using midpoint
heading, exactly as in the preceding evaluation. The network predicts XY
deviations in the departure body frame and a residual sine/cosine yaw vector.
Composition adds nominal XY and rotates the residual yaw by nominal yaw.
The first four rows of each output head start at zero weights and biases
`[0,0,0,1]`, so initial motion prediction equals the command reference.
Contact-head initialization is unchanged. Encoder, recurrent latent dynamics,
RGB future prediction and input modalities retain their original structure.

Losses compare composed absolute predictions against the original absolute
targets. Position scaling, yaw/contact losses, regularization, JEPA targets
and optimizer settings retain their definitions. Thus residual labels do not
silently change the yaw objective. This experiment changes parameterization
and its associated nominal initialization together; it does not separately
attribute their effects. There is no new measured-pose input.

Evaluate final checkpoints on the unchanged 924-context development transfer
population, with its 300/700/800-ms masks, braking and switch subsets and
two geometry clusters. Compare to the three absolute-head models trained on
the same augmented schedule, and command integration/persistence. Record all
outcomes, including short-horizon or braking regressions. Inference gets only
causal observations and prospective commands. Native values remain targets.

The snapshot schema explicitly identifies residual heads so their weights
cannot accidentally be loaded as absolute-motion heads. No old model or
running controller is replaced. Better offline predictions would still need
actual online benefit, independent mazes and realistic sensor/timing evidence.

Training entry point:
`scripts/train_go2_nominal_motion_residual_development.py`.
Fit root: `go2_nominal_motion_residual_matched_fits_v1_attempt_001`.

Three focused tests passed in 2.26 seconds: analytic nominal motion and masking,
matching composed training/inference outcomes with finite gradients for every
objective, and unchanged default loss behavior for the old model. Training is
running under the unchanged schedule.

Before seeing transfer predictions, a further diagnostic was fixed on the
first-seed full-JEPA pilot trajectories, maze 0 and maze 1. Recompute the three
residual and three same-data absolute-head predictions on exactly their saved
selected command windows with verified 700-ms execution. Include command
integration and the saved corrected-JEPA/fitted-pose alternatives. Use retained
RGB/body packets only, with no depth regeneration or native artifact reads.
Ground truth stays in the existing evaluator records. Report position/yaw
errors separately for hold, turn, translation and short translation pulses.
Yaw uses the prior pilot's saved wrapped world-heading change convention.
This is exposed-trajectory prediction diagnosis, not an alternate-policy
navigation result or new environmental replication.

Entry points:
`scripts/evaluate_go2_nominal_motion_residual_development.py` and
`scripts/read_go2_nominal_residual_executed_windows_development.py`.

## Completed prediction results

All three fits completed in 298.71 seconds including loading, with 3,600 total
updates and 9.75 GB peak RSS. Transfer evaluation took 17.15 seconds. At
700 ms on its 870 valid targets:

| Predictor | XY RMSE mm | Yaw RMSE degrees |
|---|---:|---:|
| Residual JEPA | 21.713 | 1.792 |
| Same-data absolute JEPA | 23.279 | 6.845 |
| Residual direct | 15.280 | 1.838 |
| Same-data absolute direct | 27.893 | 3.714 |
| Residual supervised rollout | 15.872 | 2.206 |
| Same-data absolute supervised rollout | 18.077 | 3.357 |
| Command integration | 16.192 | 2.286 |

Residual direct improves over command integration at 700/800 ms, including
braking (12.828/13.736 mm versus 14.230/15.619 mm), but regresses at 300 ms
(12.373 versus 9.323 mm overall; braking 10.946 versus 7.298 mm). Residual
supervised rollout improves slightly at longer horizons but remains worse on
braking. JEPA yaw improves, while its XY remains substantially worse than the
command reference. Directions are consistent across the two geometry clusters;
this one-seed development comparison establishes no method superiority.

The fixed exposed-maze readout completed 775 matched executed windows: 402 on
maze 0 and 373 on maze 1, in 19.90 seconds. At 700 ms:

| Predictor | XY RMSE mm | Yaw RMSE degrees |
|---|---:|---:|
| Residual JEPA | 22.728 | 1.337 |
| Residual direct | 13.460 | 1.465 |
| Residual supervised rollout | 11.175 | 1.437 |
| Command integration | 11.759 | 1.473 |
| Saved corrected original JEPA | 6.304 | 3.428 |
| Saved fitted XY / command yaw | 6.564 | 1.473 |

The supervised residual's small XY advantage over command integration appears
on each maze, but comes from ordinary translations. It is worse on holds
(12.368 versus 9.148 mm), turns (12.696 versus 10.442 mm), and the five short
translation windows (12.880 versus 12.369 mm). Five short pulses are far too
few to support a reliability claim. The fitted pose-based control remains
substantially more accurate. No alternative navigation outcome was executed.

The executed-readout script initially had a missing closing parenthesis and
exited during module parsing, before reading inputs or producing output. It
was corrected without changing the roster, weights or data. The incident is
retained in the fit root's `executed_readout_setup_failure_v1.json`.

Outputs:
`go2_nominal_motion_residual_transfer_comparison_v1_attempt_001` and
`go2_nominal_residual_executed_windows_v1_attempt_001`.
Every result and fixed checkpoint is retained; none replaces a navigation model.

## Next measured-motion question

The remaining practical gap is observed motion state and stopping, not a reason
to choose another favorable seed. Retained switch training recordings contain
public RGB, depth and gyro packets, so causal visual motion may be recoverable
without another simulation collection. A separate primary-camera observer
probe on the first training switch trial checks this feasibility. It must not
silently substitute native pose or claim equivalence with the deployed
dual-camera registered observer. Any added motion channel needs the same
causal definition in training and deployment and a matched simple control.

The feasibility probe is complete at
`go2_training_visual_motion_probe_v1_attempt_001`: 40 poses accepted, then a
terminal registration failure at frame 40 of 64 because no recent reference
passed the unchanged gates. It provides 37 complete four-pose histories on
that accepted prefix, not a complete training feature population. The failure
and all accepted poses are retained. Do not silently fill the remaining
history with native poses or zeros. The recordings contain the required public
modalities, but observer coverage and a matched deployment definition still
need work. This was a primary RGB-D/gyro observer probe, not a change to the
live dual-camera tracking system.

On its accepted prefix only, a separate comparison against existing native
target labels gives 0.434/0.623/0.993 mm XY RMSE at 100/300/700 ms across
33/31/27 overlapping windows. Labels were never observer inputs. This suggests
useful measured-motion information is present when tracking succeeds, while
the retained failure prevents claiming complete coverage or robust sensing.
The comparison is `accepted_prefix_target_comparison_v1.json` in the probe root.

Scene-dependent prediction, short-pulse/multiple-transition data and actual
online benefit remain necessary. These motion results alone do not establish
useful visual world prediction, JEPA advantage or the broader navigation goal.
