# JEPA Navigation Phase 2B/2C Findings

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

## Retrospective Audit Note

A later same-date repository and artifact audit identified important
limitations in the action controls, complete-valid filtering, evaluation split
usage, and shared spatial projector. These limitations do not change the
registered persistence failure or collapse result, but they narrow the
permitted interpretation of action sensitivity and target-geometry failure.

The evidence-qualified claims and revised next experiment are recorded in:

- `docs/lewm_jepa_repository_research_audit_2026-06-14.md`;
- `docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`.

## Purpose

The purpose of these experiments is not to maximize navigation success by any
available means. It is to determine whether a JEPA-style world model can learn
a latent state whose predicted futures are sufficiently spatial and
action-sensitive to choose safe, useful navigation actions.

Phase 1 showed that the existing pooled LeWM does not make competitive
counterfactual decisions. Phase 2A showed that exposing its frozen patch tokens
does not fix the problem. Phase 2B therefore trained the visual encoder and
spatial predictor end to end. Phase 2C tested a more faithful JEPA target
construction using a stop-gradient exponential-moving-average teacher.

The result is a useful negative finding:

> The current image-aligned patch-token objective does not learn usable
> action-conditioned dynamics. Anti-collapse regularisation is necessary, but
> preventing collapse is not sufficient. A naive EMA teacher also does not fix
> the target and action-identifiability problems.

This is a reason to redesign the learned state and objective before scaling. It
is not evidence that JEPA-style world models should be abandoned.

## Lay Explanation

Imagine asking the model to picture what the robot will see after turning left.
A useful world model should produce a better answer when told "turn left" than
when told "do nothing" or "turn right."

The experiments found that changing the action usually changed the model's
error by almost nothing. The model produced roughly the same quality answer
with the correct action, no action, or another action. It also remained worse
than the very simple strategy "assume the view stays the same."

One unregularised model appeared to achieve very low prediction error and good
action-selection scores. That result was false progress: its latent space had
collapsed until nearly every image looked the same internally. Predicting a
future is easy if the representation erases the differences between futures.

The anti-collapse term stopped that failure, so it is not merely working
against the task. However, it only ensured that the representation remained
varied. It did not ensure that the variation represented obstacles, openings,
or the consequences of actions.

## Controlled Data And Evaluation

The bounded gate uses scene-disjoint data:

- eight training scenes and eight evaluation scenes;
- one scene from every registered scene family in each split;
- eight source states per scene;
- nine outcome-stratified action sequences per source state;
- two action blocks and two future-observation slots per sequence.

The generated artifacts are:

- `.generated/jepa_counterfactual/phase2b_train_8scene_spatial_v1.jsonl`;
- `.generated/jepa_counterfactual/phase2b_eval_8scene_spatial_v1.jsonl`;
- `.generated/jepa_counterfactual/phase2b_bounded_factorial/analysis.json`;
- `.generated/jepa_counterfactual/phase2c_ema_gate/analysis.json`.

Of 576 planned candidate sequences per split, 394 training and 422 evaluation
sequences have complete valid rendered future observations. Invalid
observations remain explicitly recorded and are not treated as collision
labels. Evaluation action-selection metrics are therefore conditional on
complete valid future observations and are not complete safety claims.

Every model is evaluated against:

- **persistence:** predict that the future latent is unchanged;
- **zero action:** replace the real command with no motion;
- **shuffled action:** give the model another sequence's command;
- **representation collapse:** measure whether different observations remain
  meaningfully different;
- **counterfactual selection:** rank the rendered candidate futures by
  goal-latent distance and inspect the selected consequence labels.

Action advantage is now normalized by the target's actual latent change. A
model must improve over a wrong action by at least 10% of that change to count
as meaningfully action-sensitive. This avoids passing a gate on numerical
noise.

## Phase 2B Factorial

The reduced-capacity learnability gate compares:

| Cell | Representation | Anti-collapse |
| --- | --- | --- |
| `pooled` | one CLS token | CLS SIGReg |
| `spatial_var` | ordered patch tokens | appearance SIGReg plus spatial variance floor |
| `spatial_no_var` | ordered patch tokens | appearance SIGReg only |

The pooled and spatial models have 561,904 and 574,048 trainable parameters
respectively. They use identical data, seed, epochs, optimizer, action inputs,
and scene-disjoint evaluation.

### Final prediction result

| Cell | Feature std | One-step / persistence | Two-step / persistence | Meaningful action advantage |
| --- | ---: | ---: | ---: | --- |
| pooled | 0.777 | 2.07x | 4.60x | none |
| spatial + variance | 0.947 | 2.69x | 3.68x | none |
| spatial without variance | 0.027 | 21.53x | 116.49x | none; collapsed |

Lower ratios are better; a useful model must be below `1.0x`.

Persistence beats every cell at the first block. This is the decisive failure.
It means recursive rollout drift is not the primary problem: the learned model
already adds more error than useful information on its first prediction.

The regularized spatial model improves the two-step ratio over pooled
(`3.68x` versus `4.60x`), but that is only a relative improvement between two
failing models. It does not pass the learnability gate.

### Action sensitivity

For the regularized spatial model at one step:

- replacing the real action with zero increases MSE by only `0.00205`;
- that is only `2.87%` of the target's actual latent change;
- a shuffled action is slightly better than the real action;
- shuffled-action advantage is `-0.62%` of target change.

For the pooled model, the real action's advantage over zero is only `0.67%` of
target change and over shuffled action only `0.18%`.

Lay interpretation: under the implemented controls, there is no evidence that
the learned future is meaningfully determined by the supplied action. A later
audit found that many zero and shuffled comparisons are not valid negative
actions, so the exact degree of action insensitivity remains unresolved.

### Anti-collapse result

Removing the spatial variance floor produces:

- mean feature standard deviation `0.027`;
- target-change MSE `0.000395`;
- a collapse warning in every epoch;
- essentially zero difference between real, zero, and shuffled actions.

Its raw prediction MSE becomes small because the target space itself becomes
nearly constant. Its apparently strong safe-positive-progress and
oracle-match selection numbers are not meaningful because distances in a
collapsed latent cannot reliably rank futures.

This directly answers the regularisation question:

> The spatial anti-collapse term is not the cause of the failure. Without it,
> the representation collapses. With it, the representation remains varied but
> still fails to organize that variation around action consequences.

The current variance floor is therefore necessary but insufficient.

### Conditional selection result

The regularized spatial model selects safe positive progress on `46.67%` of
the complete-valid evaluation target groups versus `26.67%` for pooled.
However, it also selects newly unsafe outcomes on `11.11%` versus `8.89%`, and
its exact oracle-sequence match rate is only `11.11%`.

These numbers do not override the prediction failure:

- they cover only 45 target groups with enough complete-valid candidates;
- renderer invalidity is outcome-dependent;
- direct goal-latent distance can rank candidates using appearance shortcuts;
- the model does not meaningfully use the supplied action;
- the collapsed model demonstrates that selection metrics can look strong even
  when the latent dynamics are invalid.

The correct decision is therefore `stop_and_redesign_before_scaling`.

## Phase 2C: Stop-Gradient EMA Teacher

### Why this experiment was required

The Phase 2B online objective lets the same trainable encoder define both the
prediction and the target. That allows the target representation to move toward
an easier solution while the predictor learns.

A standard JEPA-style correction is to:

1. encode the current observation with an online student;
2. encode future targets with a stop-gradient teacher;
3. update the teacher slowly from the student using exponential moving average.

Phase 2C implemented this control with EMA momentum `0.99`, keeping all other
data, capacity, regularisation, and evaluation settings fixed.

### Result

The EMA teacher does not collapse, but it also fails:

- persistence beats it at one step in all eight epochs;
- the best one-step ratio is `1.97x`;
- the final one-step ratio is `8.37x`;
- the final two-step ratio is `5.06x`;
- final real-action advantage over zero is only `0.24%` of target change;
- final real-action advantage over shuffled is only `0.20%` of target change;
- safe-positive-progress selection falls from `46.67%` to `40.00%`;
- newly unsafe selection rises from `11.11%` to `28.89%`.

The teacher's mean feature standard deviation grows from `0.291` to `1.795`,
while training prediction loss grows from `0.136` to `1.623`. This is
consistent with a scale-drift failure: the variance floor prevents low
variance but imposes no upper bound or normalized target geometry. The
predictor chases a moving, expanding target space.

This is not proof that EMA targets are unsuitable. It shows that adding EMA to
the current unnormalized patch-target objective is insufficient and unstable.

## What The Experiments Establish

The evidence now supports the following conclusions:

1. The current pooled LeWM is not an appropriate primary planning state.
2. Frozen patch tokens are not a hidden spatial solution.
3. Training patch tokens end to end improves some conditional ranking metrics
   but does not create action-usable dynamics.
4. Persistence wins at the first step for every valid non-collapsed model.
5. Under the implemented controls, real actions do not show a meaningful
   measured advantage over zero or batch-rolled actions. A later audit found
   that those controls contain many false negatives, so this does not cleanly
   estimate the degree of action use.
6. Removing the spatial anti-collapse term causes immediate collapse.
7. The current variance floor prevents collapse but does not stabilize scale or
   create navigation semantics.
8. A naive stop-gradient EMA teacher does not rescue the current target.
9. Larger models, longer horizons, or navigation heuristics are not justified
   as simultaneous compensating changes. Recurrent belief should be tested
   separately on registered history-required subsets rather than assumed to
   solve the single-frame failure.

## Why The Current Spatial Target Is Probably Mismatched

The target is a grid of image-aligned patch tokens. This preserves pixel
location, but it does not directly represent stable world structure.

When the robot moves:

- the same wall shifts across many image patches;
- newly visible regions appear and old regions disappear;
- depth and occlusion change nonlinearly;
- most pixels remain visually similar while the small action-relevant region
  may be an opening or collision boundary;
- direct average MSE gives static appearance much more weight than the small
  part that distinguishes candidate actions.

The objective therefore rewards predicting generic appearance and permits the
action signal to remain weak. The problem is not simply insufficient
transformer capacity.

Lay interpretation: the model is being graded mostly on reproducing the whole
photograph, while navigation may depend on one doorway near the edge. It can
earn a reasonable average score without understanding which command reaches
the doorway.

## Next Falsifiable Experiment: Phase 2D

The original Phase 2D outline below is superseded by the corrected
preregistration in
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`.
That plan retains normalized EMA targets and masked action identifiability, but
requires corrected hard negatives, per-slot masks, separate projection paths,
new test splits, and hierarchical statistical analysis before training.

### Controlled changes

1. **Normalize spatial prediction and target tokens.**

   Apply a fixed per-token normalization before prediction loss and evaluation.
   This removes feature scale as an easy degree of freedom and makes EMA target
   geometry stable enough to evaluate.

2. **Add an action-identifiability objective.**

   For the same current state and true future target, require the prediction
   conditioned on the real action to be closer than predictions conditioned on
   zero or mismatched actions. True hold actions and identical negative actions
   must be masked so the objective does not create false negatives.

3. **Report action advantage as a first-class metric.**

   The main action gate is not a boolean comparison. The correct action must
   improve prediction by at least `10%` of the target's actual latent change
   over both zero and shuffled actions.

### Bounded factorial

Use the existing scene-disjoint Phase 2B data and reduced architecture:

| Cell | Teacher | Target geometry | Action-identifiability loss |
| --- | --- | --- | --- |
| existing `spatial_var` | online | unnormalized | no |
| `ema_normalized` | EMA | normalized | no |
| `ema_normalized_action` | EMA | normalized | yes |

The normalized EMA cell isolates target stability. The final cell tests whether
the objective can force the latent transition to retain real action
consequences.

### Promotion gate

Do not add recurrent belief, longer rollout, or full model capacity unless the
winning cell:

- remains non-collapsed with stable feature scale;
- beats persistence at one block on disjoint scenes;
- meaningfully beats zero and shuffled actions at one block;
- improves the two-block ratio over the online spatial control;
- improves safe positive progress without increasing newly unsafe selection;
- retains explicit invalid-observation and physics-validity limitations.

If normalized action-identifiable targets still fail the one-step gate, stop
using image-aligned patch-token prediction as the primary dynamics state.
The next redesign should then test learned egocentric/world-aligned slots or an
affordance/dynamics branch whose target is explicitly equivariant under robot
motion.

## Research Decision

Do not scale Phase 2B or Phase 2C. Do not add controller heuristics to compensate
for their failure.

Retain the counterfactual benchmark, collapse diagnostics, meaningful action
gates, and EMA-teacher implementation as research infrastructure. Proceed only
with the bounded Phase 2D objective redesign above.
