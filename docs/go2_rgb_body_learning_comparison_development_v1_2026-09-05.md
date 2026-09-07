# Fixed-budget RGB/body learning comparison, development V1

Specified before any fitting or validation model scoring. Prerequisite: all 120
counterfactual branches in the V2 composite corpus pass raw audit. The original
nineteen physical branches remain at their original paths; no recollection.
All five actions share exactly one selected, real canonical starting observation
per layout. Sixteen procedural layouts are training; eight are development
validation, not final evaluation. All five branches must be available in every
layout; otherwise this fixed study stops without silently selecting examples.

## Question and comparisons

Does action-conditioned latent prediction add useful information about executed
motion and contact beyond the same observation encoder and supervised outcomes?

Three conditions share initialization, spatial CNN, ordered body/control GRUs,
direct action-prefix head, input tensors, training examples and update order:

- `direct`: direct outcome loss plus common variance/covariance regularization.
- `supervised_rollout`: additionally supervised recurrent latent rollout and
  its outcome decoder, but no future-embedding prediction loss.
- `jepa`: the same supervised rollout condition plus prediction of stop-gradient
  EMA future embeddings. EMA momentum 0.99; autoregressive 0.5-s transitions.

Every condition's online encoder sees the same current and valid future
RGB/body/control observations through the common regularizer on their joined
embeddings. Future data enter training targets/regularization, never the current
inference API. This matches observation exposure, not all operations: JEPA has
an additional EMA forward pass. Direct has fewer active trainable parameters;
report those counts and measured training/inference time, not artificial parity.

The most interpretable latent-prediction contrast is JEPA versus supervised
rollout, evaluated with their identical direct head. JEPA versus direct is a
composite comparison that also changes supervised rollout training. Within the
JEPA checkpoint, direct versus rollout inference tests these particular trained
readouts; it does not establish that every possible online planner benefits.

## Frozen budget and targets

Three seeds: 2026091200, 2026091201, 2026091202. CPU, one Torch thread,
deterministic algorithms, no pretrained weights or augmentation. Sixty epochs,
five updates per epoch. Each batch has sixteen independent layouts with one
action per layout; each epoch covers every training branch exactly once using
seed/epoch-fixed action permutations. AdamW learning rate 0.0003, weight decay
0.0001, global gradient norm cap 5. No schedule, validation selection, early
stopping, restart, seed replacement or parameter search. Use only final update
300. Preserve all nine checkpoints and loss histories. Nonfinite values or
integrity errors stop the study and retain partial artifacts.

Outcomes every 0.5 s through 4 s: body-start projected x/y displacement,
sin/cos wrapped world-yaw difference, cumulative contact logit. Motion smooth-L1
and contact BCE each have weight 1 within an active outcome loss. Direct loss
weight 1; rollout loss weight 1 when active; JEPA embedding MSE weight 1.
Common variance weight 0.1 and off-diagonal covariance weight 0.01. Invalid
post-stop motion and unavailable future observations are masked before encoding
or arithmetic. Cumulative contact remains known after an observed contact stop.
These are privileged *training/evaluation labels*, not input channels.

## Evaluation and decision discipline

Evaluate final checkpoints once on fixed validation layouts. Primary descriptive
metrics: per-layout mean positional error (metres), wrapped yaw error (radians),
and contact Brier score across available branch/horizon targets; then average
layouts equally. Report target coverage, contact accuracy at a fixed 0.5
threshold, training metrics, per-seed outcomes and all paired layout differences.
Also report each horizon separately, positive-contact counts and violations of
the required nondecreasing cumulative-contact probability. Pooling early safe
horizons must not hide poor long-horizon contact prediction.
For direct-head condition contrasts, average three seed differences within each
layout and use 10,000 fixed-seed layout bootstrap resamples for a descriptive
95% percentile interval. Eight layouts are a small development sample; these
intervals are not final generalization certification and do not account for
multiple comparisons. Steps/actions/horizons are not independent maze samples.

Include training-only action/horizon mean outcomes, zero-motion/no-contact
persistence, and prospective command-kinematic/no-contact baselines. Evaluate
fixed next-layout same-action RGB shuffle and body-history shuffle, and next
action within-layout plan shuffle. These are dependence diagnostics, not
necessarily physically realizable sensor interventions. Report independent
current-context latent variance/effective rank and future embedding prediction
versus latent persistence; latent loss alone cannot establish utility.

This study has no pass threshold designed to force a JEPA claim. A lower direct
validation loss with preserved motion quality and consistent layout/seed
differences motivates a *new*, bounded executed-choice comparison. A null or
negative predictive contribution is retained and motivates observable-state,
coverage and inference diagnostics, not another automatic coefficient search.
No result here proves closed-loop safety, memory, unknown-goal exploration,
beacon discovery, novel-maze return or physical transfer. Those remain separate
stages in EXECUTION_PLAN.md.
