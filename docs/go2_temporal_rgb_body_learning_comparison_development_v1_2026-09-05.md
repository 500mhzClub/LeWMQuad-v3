# Temporal RGB/body comparison V1: fixed scientific design

This follows the [audited causal windows](go2_causal_subtrajectory_development_v1_result_2026-09-05.md)
and the earlier negative/limited online comparison. It is a **new development
comparison**, not a retry of any completed model or a final maze benchmark.
This document fixes the design before fitting. The training runner, schedule
artifact, metric implementation and source/input bindings must still pass review
and preflight before launch. No training run is launched merely by this document.

## Question and limits

With equal causal temporal inputs and outcome supervision, does latent future
prediction improve decision-relevant state estimation beyond direct supervision
or supervised recurrent rollout? Compare JEPA's **direct head** for predictive-
training utility separately from its **rollout head** for runtime predictor
utility. Within-model head comparisons do not alone isolate runtime compute:
report parameter counts and inference time, and retain the direct-only arm.

This package introduces temporal context, later-state supervision and explicit
remaining-plan masks together. It therefore does not isolate the benefit of RGB
history versus expanded training states relative to the old study. Only contrasts
between the matched new arms isolate their specified objective differences.
Do not call improvement over old models a clean temporal-context ablation.

The dataset still has16 training and8 repeatedly used development-validation
layouts. Later-state windows supply only the executed branch outcome, not all
candidate outcomes. No final held-out, sealed or historical runtime material is
accessed. There is no claim of maze exploration, beacon return or real transfer.

## Model and conditions

Use `TemporalRGBBodyJEPA` at128 latent dimensions. The existing spatial CNN and
ordered body/control encoders process each of four past packets. A chronological
128-unit GRU and LayerNorm form current context. A direct action-prefix GRU head
predicts physical outcomes; a recurrent action-conditioned latent transition and
shared outcome decoder supply the rollout head. Targets are **single actually
observed future packet embeddings** from the EMA observation encoder, not a
fabricated future four-frame context. No future observation enters inference.

Plans are normalized B×8×5×3 commands with B×8×5 boolean validity. Only nonempty
contiguous full .5-second blocks are supported. Unavailable commands must be
zero padding with false validity. No transition runs for an unavailable block,
and no output at that horizon is valid. Earlier predictions cannot depend on
later commands or total remaining plan length. Future feedback commands from a
gyro controller are not known prospective action inputs.

Three arms, each trained from the same per-seed initial weights:

1. `direct`: current-history encoder and direct outcome head only.
2. `supervised_rollout`: same direct objective plus recurrent outcome supervision.
3. `jepa`: same two outcome objectives plus EMA future-embedding prediction MSE.

Direct active parameters:575,045. Each rollout arm:712,138. EMA parameters are
not optimized. The two rollout arms are architecturally matched; direct-only has
less active capacity and runtime work and is not a capacity-matched JEPA control.

Each arm sees the identical four past frames and valid single future frames
through a common variance/covariance regularizer. Apply it to current context,
all four online past embeddings and all valid online future embeddings. Weights:
direct outcome1, rollout outcome1 when present, latent prediction1 for JEPA,
variance.1, covariance.01. EMA momentum.99. No coefficient sweep.

Outcome channels remain dx,dy,sin(yaw),cos(yaw),contact logit. Use the previous
SmoothL1 motion and BCE contact definitions but average the per-window losses
equally over the batch: variable remaining duration must not make a layout count
more solely because it has more labeled horizons. Censored motion is indexed out
before arithmetic. Future latent MSE averages valid embeddings; the common
regularizer likewise weights embeddings, not independent layouts. These weighting
choices are identical across arms and must be reported, not hidden as equivalent
to uniform-layout weighting of every objective term.

## Fixed fitting budget and schedule

Use seeds2026091700,2026091701,2026091702 for all three arms, CPU execution with one
compute thread. AdamW learning rate3e-4, weight decay1e-4, gradient clipping norm5.
Train240 epochs of5 updates each, exactly1,200 updates per model and9 models.
This fixed budget is four times the old300-update budget, chosen before fitting
because the corpus contains substantially more training contexts; it is not a
convergence claim or a controlled equal-budget comparison against the old study.

Each update contains one window from each of16 training layouts. Across five
updates, every layout contributes all five actions once in seeded shuffled order.
Sample one of that action's actually observed offsets uniformly. Use the same
materialized index schedule for every arm with a given seed; do not resample
missing post-contact contexts or balance on validation labels. Every epoch gives
each layout/action equal exposure, not every repeated window equal exposure.

Use only the final1,200-update checkpoint. No early stopping, best-validation
checkpoint, extra seeds, extensions or automatic replacement after failure.
Infrastructure or nonfinite failures retain partial artifacts and stop this
package pending diagnosis. Record every update's losses, source/input/checkpoint
identities, parameter counts and timings. No physical navigation is launched by
this training protocol.

## Predeclared evaluation

Report all three seeds separately and aggregate by **layout first**, with paired
layout differences and development uncertainty. Repeated windows/horizons are not
independent trials. Report coverage, contact prevalence, censored horizons and
failures. Empty metric subsets are explicit nulls, not zeros or JSON NaNs.

Primary prediction endpoints: contact Brier and position error over valid
later-state (offset>0) horizons, equally averaged within each layout then across
layouts. Position error is conditional on pre-contact observation; it cannot
replace the contact endpoint. Also report initial contexts separately, yaw error,
each offset/horizon and the last known horizon. Retain zero-motion/no-contact,
command-kinematics/no-contact and training-only action/remaining-plan mean controls.
Report missing empirical-control cells and a predeclared action/horizon fallback;
never use validation targets to fill them. Unknown padded horizons are excluded
from scores and monotonicity checks.

At initial contexts only, evaluate the five actually executed actions for each
layout with the old fixed local intents(.8,0),(0,.8),(0,-.8), cost10×contact
probability plus distance to intent at4seconds, deterministic action-order ties,
and all-stop comparator. Score actual contact and realized action regret from
the recorded counterfactual branches. This is offline initial-context decision
evidence, not later-state action ranking or fresh online navigation.

For scene-use diagnosis, apply a fixed cyclic cross-layout RGB-history shuffle
and a separate body-history shuffle at matched action and offset; only include
cells where all required donors exist and report omitted cells. Preserve target
labels and command inputs. Compare these diagnostics across all arms/heads;
do not select the best shuffle or use augmentation results as physical evidence.

No JEPA utility claim is warranted from latent loss alone, improvement only
against a weak control, or better motion paired with worse contact/decisions.
A negative outcome remains valid. After preserving this comparison, specify
fresh successive sensor-only replanning with supplied local intents, then online
place/frontier memory and exploration/return. Turning clearance, sensor delay/
drift, relative odometry and physical hardware evaluation remain separate needs.
