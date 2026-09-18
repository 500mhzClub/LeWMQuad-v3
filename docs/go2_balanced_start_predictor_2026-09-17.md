# Balanced start-action coverage: matched predictor continuation

Status: **FIT, FIXED EVALUATION AND CONTROL COMPARISON COMPLETE**.
The supplemented predictor corrects the action ranking at the exposed right
start. Transfer prediction effects are mixed; the completed local control
comparison achieved 2/4 final arrivals versus 0/4 matched original-data
continuation, both contact-free. See `go2_balanced_start_goal_pilot_2026-09-17.md`.
Independent complete-maze navigation remains unproven for this dense model.

Training PID 63063/session 21525 exited 0 after all eight epochs and 1,760
updates per arm, taking 2,787.49 seconds (46.46 minutes). Encoding 4,019 unique
images took 845.83 seconds. All four final weights are retained; no fit failed.
The corrected fixed evaluation completed in 49.78 seconds, session 21196,
exit 0. Earlier progress statements below are historical.

| Predictor | Right-start dense MSE | Right-start chosen action | Transfer dense MSE (18 branches) | Transfer goal-embedding MSE |
|---|---:|---|---:|---:|
| Original-data continuation, action input | 0.348277 | Forward | 0.235534 | 0.644430 |
| Supplemented continuation, action input | 0.274036 | Right arc | 0.247789 | 0.525702 |
| Original-data continuation, action blind | 0.594887 | All six tied | 0.343325 | 1.459909 |
| Supplemented continuation, action blind | 0.501588 | All six tied | 0.357447 | 1.975056 |
| Original parent, action input | 0.348247 | Forward | 0.243141 | 0.770751 |
| Persistence | 0.628064 | All six tied | 0.387830 | 0.641084 |

At the exposed right-opening start, adding the balanced examples lowers dense
prediction error by 21.3% against matched continued training and changes the
choice to the physically best right arc (physical regret 33.90 to zero).
Both continued action predictors retrieve all six action outcomes correctly.
On the 18 transfer pulse branches, dense MSE worsens by 5.2%, while goal-
embedding MSE improves by 18.4%; both retain 18/18 dense action retrieval.
Both predict hold in all four retained early/late goal-state comparisons.
These are exposed development results with one fit seed, not proof of
navigation improvement or a JEPA encoder-objective contribution.

The first evaluation (session 47462, exit 0) revealed small batched GEMM
differences that created false tie breaks for identical blind/persistence
forecasts at the right start. The evaluator now computes their goal cost once
and expands it, exactly as the online controller does. Original results are
preserved in `go2_balanced_start_predictor_evaluation_before_numerical_tie_fix_2026-09-17.json`
and the output's `fixed_evaluation_before_numerical_tie_fix` directory, with
an explanatory hash-bound receipt. The corrected pass changes no models,
training data or native trials; action-conditioned scores are unchanged.

The completed coverage diagnostic found only two exact 500-ms start-from-rest
examples per moving action in the original dense predictor training set,
both in left-opening layouts. Collecting a balanced panel tests whether this
limited coverage contributes to prediction errors. It does not assume that
all failures are explained by training data; the current local planner also
lacks established collision avoidance and long-horizon route selection.

Collection completed all **48/48** preselected training cases without contact
or physical stop: two training clusters, both opening directions, both
appearance seeds, and all six actions. Each records ten quiet ticks followed
by five action ticks; context frames are 0/5/10 and the target is frame 15.
RGB context hashes match within each of the eight environment groups.
Native execution took 8.33–8.70 seconds per case, excluding process startup.
Two workers ran on CPU groups 4–7 and 8–11. The tapes are training examples,
not closed-loop navigation evidence. All RGB, applied commands, native
physics and sensor timing are retained; no depth arrays were written.

The initial proposed output volume failed the batch-space check before
creating an attempt or starting a simulator. Both prematurely invoked worker
commands also exited before simulation because that output directory did not
exist. The collection was then prepared on the workspace volume and ran once
per case (worker sessions 43032 and 50624, both exit 0). Reader 97467 completed
with 48 eligible cases, zero contacts and eight matching context groups.
The original 220-MiB size estimate was low because scene assets are retained
per case; free workspace space after collection was approximately 880 MiB.

Physical-diversity analysis completed in session 73280. Within each action,
all eight recordings have exactly identical native base-pose, joint-position,
joint-velocity and applied-command traces. Thus these are **six distinct
physical trajectories shown in eight visual environments**, not 48 independent
motion responses. This is appropriate for testing visual-context coverage,
but cannot establish learned geometry-dependent contact dynamics. The result
is recorded in `go2_balanced_start_physical_diversity_2026-09-17.json`; no
recordings are discarded or training schedules changed after this analysis.

The continuation has four arms:

| Data schedule | Future-action input | Initial checkpoint |
|---|---|---|
| Original samples | Present | Original native-adapted action predictor |
| Original plus balanced starts | Present | Same action predictor |
| Original samples | Zeroed | Original matched action-blind predictor |
| Original plus balanced starts | Zeroed | Same action-blind predictor |

Each arm receives eight epochs and 1,760 optimizer updates, batch size 16
(the final batch in an epoch has 14 samples). Every pair restores identical
parent model and AdamW state. Loss, learning rate, clipping, architecture,
frozen V-JEPA encoder, and control normalization remain unchanged. Within
each batch, the mixed schedule replaces two original examples with balanced
start samples. The remaining positions are identical to the control batch.
All 3,518 original examples remain represented over the mixed schedule;
each of the 48 new examples receives 73 or 74 draws. Schedules and the final
epoch are fixed before training, without transfer-loss selection.

Training uses one GPU process on CPU cores 8–11, with normalized FP16 features
kept in RAM. The existing batch-eight encoder and batch-16 predictor settings
were already measured on this hardware. Estimated duration is about 45
minutes, including encoding. Only four final model-weight files are saved;
optimizer states remain in RAM. Interrupted runs must remain recorded failures
and must not be silently restarted. No existing checkpoints are removed.

After training, run `evaluate_go2_balanced_start_predictor_development.py`.
It compares all four final models, both original parents and persistence on
the retained branch panel, including the exposed right-opening start that
motivated the diagnosis. The fixed cross-trajectory goal metric is shared.
Report dense prediction error, action discrimination and physical regret
separately, then run matched prospective online control. These exposed local
panels do not replace independent complete-maze evaluation.

The subsequent control design is now fixed while encoding is still in
progress: `go2_balanced_start_goal_pilot_design_2026-09-17.json`. Run the
four exposed tasks with old-data/action, mixed-data/action and the shared
action-blind policy, for 12 new trials. Both action-blind weights are scored
offline; their controller is identical because a single forecast is expanded
across candidates, producing the same seeded uniform ties, while observed
arrival recognition is independent of predictor weights. Execute that policy
once per task, not as two purportedly distinct replications. Retain the
previous cross-metric parent and direct-feedback results as explicit reused
references. Initial RGB must match the corresponding earlier tasks.

Runner: `scripts/run_go2_balanced_start_goal_pilot_development.py` (design
session 90574 completed). Reader:
`scripts/read_go2_balanced_start_goal_pilot_development.py`. All cases,
controller sources and the criterion were fixed before new fit/evaluation
outcomes. Storage is split prospectively between workspace and root volumes,
six cases each, with the existing reserve. No control trial has launched.

Training process PID 63063, session 21525, was confirmed alive with encoding
progress (1,008/4,019 unique images at 211.7 encoder seconds). Continue this
same process; do not launch another fit while it is live. Once terminal,
inspect the result or failure and run the fixed evaluator before preparing
the control batch. The full navigation goal remains incomplete.

Full-maze integration remains a separate scientific step. The existing
sparse-corner navigator consumes eight 100-ms motion forecasts, whereas this
dense predictor supplies a single 500-ms visual endpoint. The previously
fitted visual motion readout did not beat command history on transfer (8.189
versus 5.113 mm XY RMSE) and also had error with actual future images. See
`go2_dense_visual_motion_readout_2026-09-17.md`. Do not interpolate an endpoint
and describe it as a learned swept trajectory, or replace that navigator
without explicitly addressing horizon, motion decoding and collision scoring.
The old full-maze successes and current dense local experiments must remain
separate claims.

This isolates added data coverage within each predictor family. It does not
isolate the contribution of JEPA encoder training, and model-only gains would
not complete the navigation goal. Exploration, persistent memory, physical
backtracking, independent maze success and deployment validation remain open.

Collection result: `go2_balanced_start_actions_result_2026-09-17.json`.
Training plan: `go2_balanced_start_predictor_plan_2026-09-17.json`.
Output: `.generated/navigation_development_artifacts_v1/go2_balanced_start_predictor_v1_attempt_001`.
