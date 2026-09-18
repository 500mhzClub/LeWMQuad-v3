# Turn forecast diagnosis and fixed longer-training experiment

All six recent missions (four repeatability runs and two earlier-warning runs)
were included, with successes and failures. There were 3726 windows whose
requested 700-ms command sequence matched the recorded plan. The new signed
error readout reproduces every existing endpoint metric and separately measures
the first 300-ms committed prefix and the following 400-ms action increment.
No alternative action or navigation policy is executed.

Across four JEPA missions, 537 executed left-turn windows and 459 right-turn
windows show errors in both portions. Left/right prefix RMSE was 6.50/7.45 mm
for JEPA, versus 3.50/3.12 mm for pose-command. Action-increment RMSE was
6.61/7.08 mm versus 4.49/3.94 mm. Right-turn whole-window mean signed JEPA error
was +3.36/+4.80 mm in the original body x/y axes. The error is not simply a
uniform overestimate of displacement: median left-turn displacement is
underestimated, and prefix and action errors contribute differently.

The same ordering holds separately in each of the six missions: for both left
and right turns, pose-command has lower prefix and action-increment RMSE than
the actually used neural model. This is not solely an effect of pooling one
large stalled recording with successful missions.

These are pooled, overlapping, policy-selected windows. They do not establish
the accuracy of the blocked, unexecuted recovery turn. Per-run, per-action and
prefix-category metrics remain in
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_action_forecast_bias_readout_v1_attempt_001/result.json`.
The standalone PNG/SVG turn-error chart there was visually inspected.
Reader: `scripts/read_go2_action_forecast_bias_development.py`.

## Fixed training-budget test

The existing models received 1200 updates, batch size six: 7200 draws from 4694
training contexts, with intentional sampling weights. Test 6000 updates using
five exact repetitions of that schedule, retaining seed, architecture, learning
rate, data, targets and losses. Train fresh JEPA and supervised models. Their
1200-update model states must reproduce the frozen predecessors. Only the fixed
6000-update endpoints are evaluated; no checkpoint selection or resume.
No recent navigation observation or target enters training.

Training is CPU-only and uses one thread per condition on separate cores (8
and 0). Available memory was 64 GiB; predecessor peak memory was about 10 GiB
per fit. The two conditions run concurrently, with no native simulation. New
checkpoint outputs use the project directory on `/mnt/steam_drive` (34 GiB free
before launch); existing training inputs remain at their original locations.

Plan: `docs/go2_longer_residual_fit_plan_2026-09-17.json`.
Trainer: `scripts/train_go2_longer_residual_fit_development.py`.
JEPA session 64354; supervised session 69668.

The evaluator runs all four models (old/new JEPA and old/new supervised) on the
same causal RGB/body/command inputs from all six recordings. It must reproduce
the actually recorded original forecasts, reports planar prefix/action/whole
errors and whole-window yaw error, and saves forecasts at the seven diagnosed
clearance-conflict states. This is prediction evaluation on exposed development
trajectories, not a new navigation result or an independent generalization claim.
Evaluator: `scripts/evaluate_go2_longer_residual_fit_development.py`.

## Outcomes

Both conditions passed the 1200-update state comparison, reproducing the
frozen predecessors exactly, and continued toward the fixed 6000-update endpoint.
Both fits completed at 6000 updates and their final snapshots reloaded. The
supervised owner (session 69668) and JEPA owner (session 64354) exited zero.
Prediction evaluation (session 5332) completed and exited zero in 153.39 s.
Every actually recorded original forecast was reproduced exactly: maximum
difference was zero across the first four output components. All 3726 selected
executed windows from the six preselected missions were evaluated with all
four models on identical inputs.

| Model | Prefix XY RMSE (mm) | Whole 700-ms XY RMSE (mm) | Action-increment XY RMSE (mm) | Yaw RMSE (degrees) |
| --- | ---: | ---: | ---: | ---: |
| JEPA, 1200 updates | 6.12 | 9.76 | 7.50 | 2.77 |
| JEPA, 6000 updates | 7.86 | 9.90 | 7.52 | 1.25 |
| Supervised, 1200 updates | 5.39 | 8.66 | 6.56 | 1.19 |
| Supervised, 6000 updates | 10.40 | 13.26 | 7.03 | 1.24 |

Longer JEPA training improved yaw but did not improve overall XY error; its
committed-prefix error worsened. Whole-window left/right turn XY error improved
slightly (9.42/9.35 to 8.83/8.78 mm), but turn-prefix and turn-increment errors
worsened. This is not a demonstrated solution to the recovery conflict.
Longer supervised training worsened whole-window XY error in every one of the
six recordings. The original supervised model remains best among these four
for pooled whole-window XY error. These comparisons do not establish a general
JEPA advantage or an independent generalization result.

Training rollout-outcome loss averaged 0.139 versus 0.0355 for JEPA over the
first versus final 1200-update cycles; supervised averaged 0.153 versus 0.0303.
Thus this fixed increase in optimization reduced training outcome loss without
a corresponding navigation-distribution position-error gain. It does not show
that arbitrary additional training, another learning-rate schedule or more
diverse data would fail.

Neither new checkpoint is promoted into navigation. No native mission used
these models. All original and new snapshots, update logs and evaluation
records are retained under
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_longer_residual_matched_fits_v1_attempt_001/`.
Final model state identities: JEPA
`141dd5e79d0b5710199f9b4df151c3aca90fe9458fb54fa84eb0daeaa2bbd001`;
supervised
`aee30c8c2e8f0602230b68b25259804ff2150d37c45525bd6e56f3418a69df7f`.

The additional saved-map comparison (`--conflicts`, session 26751) also
completed. Right turn passed the full 0.48-m reserve check at 2/7 conflict
states for original JEPA, 5/7 for longer JEPA, 7/7 for original supervised and
5/7 for longer supervised. Original JEPA clearances reproduced exactly.
This shows changed candidate eligibility, not improved prediction truth for
the unexecuted turn. Favorable eligibility in a few saved states does not
override the complete motion-error results or establish navigation success.
Receipt: `prediction_evaluation/saved_conflict_clearance.json`.

Next diagnose observation dependence and training-to-navigation coverage, and
the recovery selector's long-way turn conflict, before another large navigation
batch. Preserve the complete negative training-budget result. The overall goal
remains active: reliable fresh-layout navigation, contribution tests, realistic
sensing/timing and hardware evidence are not established by this diagnostic.
