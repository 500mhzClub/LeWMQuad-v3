# Predict action-dependent change from a frozen current visual state

**Both 1,200-update fits, the fixed branch evaluation and the full
recorded-navigation visual evaluation are complete.**
The common representation is the completed visual-target JEPA model. Freeze
its online/EMA encoders and encode the current RGB image into the same visual
target space as the future. A new predictor adds cumulative visual innovations
to that current state. Its context is four frozen multimodal observation
embeddings, processed by a trainable 32-dimensional history GRU. Each transition
receives context, current visual state, its previous innovation, horizon,
validity and three command coordinates; a 256-unit hidden layer predicts the
next increment. Zero final-layer initialization makes the initial forecast
exactly equal to persistence.

Two models have identical parameters, initialization, training data, schedule,
loss and optimizer. One sees future commands; the other replaces their three
coordinates with zeros while retaining horizon/validity. The latter may still
infer motion from past body/control inputs. Neither sees future sensors.
Both targets are the exact same frozen visual representations, making their
latent errors directly comparable. This experiment isolates action conditioning
in a predictor; it does not isolate a JEPA representation-training advantage.

Training uses all 4,694 training contexts, the original 7,200 scheduled draws,
1,200 updates per arm, six-context batches, AdamW at 0.001, zero weight decay,
gradient clipping at 1, and final checkpoints only. Per-coordinate innovation
RMS scales are fitted on training data with the original draw weights and equal
weight per available horizon within each context; the floor is 0.0001.
Anchors receive training-only centering/scaling as predictor inputs. The loss
is squared normalized innovation error. No motion loss or EMA updates occur.

Both models start at exactly the same persistence predictions. The frozen
representation is unchanged; checkpoint reloads reproduce predictions exactly.
The no-future-action model exactly ignores changes in future command values.
Inputs match the original training population. The raw tensor cache is released
after one shared encoding pass; no new image/depth archive is retained.

## Training and branch results

| Predictor | Full weighted training latent MSE | Transfer 800-ms latent MSE | Transfer error / persistence | Centered action-effect error / no-effect baseline |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.051314 | 0.003251 | 1.000 | 1.000 |
| Anchored, with future actions | 0.025136 | 0.036537 | 11.240 | 0.838 |
| Anchored, no future actions | 0.026202 | 0.047465 | 14.602 | 1.000 |
| Original visual-target JEPA | not recomputed here | 0.092360 | 28.414 | 1.004 |

The action model reduces full transfer error by about 23% relative to its
matched no-future-action control, and action-effect error by 16.2% relative to
predicting no action effect. However, every model retrieves only 6/18 transfer
action branches at 800 ms, and both new predictors lose to persistence at every
evaluated transfer horizon. On the training branch subset, they also lose to
persistence (6.72 and 9.93 times its 800-ms error), despite halving error over
the broader training distribution. This is not a demonstrated useful visual
world model or a JEPA superiority result.

Post-hoc decomposition of the transfer result: subtracting centered action
error from full error gives common branch-mean error 0.034443 for the action
model, about 94% of its total error. Persistence's corresponding common error
is only 0.000752. The predictor's overestimated common visual change dominates
the result. This decomposition uses all realized branches for diagnosis, not
as an inference correction. No outcome-dependent correction or candidate
centering is applied to improve the reported scores.

This is one training seed, 18 exposed transfer contexts and two geometries,
with six identical-history action groups. Keep these observations exploratory.
Training-mean/prefix baselines and full horizon curves are retained. There is
no new physical motion readout, controller integration or navigation promotion.

Result: `docs/go2_anchored_visual_dynamics_result_2026-09-17.json`.
Plan: `docs/go2_anchored_visual_dynamics_plan_2026-09-17.json`.
Training: `scripts/train_go2_anchored_visual_dynamics_development.py`.
Branch evaluation: `scripts/evaluate_go2_anchored_visual_dynamics_development.py`.
Artifact: `go2_anchored_visual_dynamics_v1_attempt_001` on the artifact volume.

The full 2,404-window recorded-navigation visual forecast population is fixed
in `docs/go2_anchored_visual_navigation_plan_2026-09-17.json`. Its primary horizon
is 700 ms, the full interval for which requested commands match execution.
Both failed return trajectories are included. Only RGB targets are used, with
no depth or native physics. These overlapping windows remain retrospective
visual-forecast evidence, not new navigation or counterfactual-action outcomes.

## Complete recorded-navigation visual results

All 2,404 windows completed in 62.12 seconds, with process exit zero.
These errors are in one common frozen visual target space; they are not
millimetres or physical motion errors.

| Predictor | Pooled 700-ms visual MSE | Relative to persistence |
| --- | ---: | ---: |
| Persistence | 0.027610 | 1.000 |
| Anchored, with future actions | 0.030777 | 1.115 |
| Anchored, no future actions | 0.031301 | 1.134 |
| Original visual-target JEPA | 0.089297 | 3.234 |

Action conditioning reduces error by 1.7% versus its matched no-action control.
It improves three of four recording averages relative to that control, but
loses to persistence in every recording and at every pooled horizon. Right-turn
windows improve over persistence (0.083817 vs 0.106751), while all other action
categories worsen. Retain the full population; do not select that category to
claim general success. The much larger persistence gap on the short-pulse
branch assay demonstrates sensitivity to the motion/observation distribution.

Result: `docs/go2_anchored_visual_navigation_result_2026-09-17.json`.
The artifact retains every window's errors, all horizons, each recording and
each action category. Both failed return trajectories are included; no encoder,
predictor or normalization changed during evaluation. No native navigation or
new motion-readout fitting was launched from these scores.

## Next scientific comparison

The current anchor is explicitly visual, but its temporal context still comes
from frozen mixed RGB/body/control embeddings. A useful next controlled factor
is replacing that history with recent visual-state embeddings, so the predictor
can directly observe whether and how the visual state is changing. Preserve the
current frozen target, action/no-action pairing, schedule and persistence
baseline. This would test temporal visual information rather than modifying
sensors, adding planning candidates or applying a post-hoc bias correction.
It has not been implemented or tested by this experiment.
