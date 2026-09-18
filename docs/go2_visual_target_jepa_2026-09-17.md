# Visual-only future target with unchanged multimodal context

**Training, branch evaluation, matched motion-readout fitting and the full
recorded-navigation forecast evaluation are complete.**
The 1,200-update fit took 154.36 seconds and peaked at 10.12 GB RSS. Training
inputs and initialization exactly match the mixed-target control; saved causal
predictions reload exactly. The target accepts RGB alone and is unchanged when
future body/control tensors contain NaNs. All processes for these steps exited
zero, as did the separate full recorded-navigation forecast evaluation below.

## Completed branch findings

The visual-target change does not establish a better predictive world model.
On the 18 exposed transfer contexts at 800 ms:

| Diagnostic | Visual-target JEPA | Original mixed-target JEPA |
| --- | ---: | ---: |
| Prediction MSE / own persistence MSE | 28.414 | 0.366 |
| Action-branch retrieval | 6/18 | 6/18 |
| Scene retrieval | 12/18 | 10/18 |
| Own persistence scene retrieval | 15/18 | 10/18 |
| Centered action-effect error / zero-effect error | 1.004 | 1.647 |
| Target effective rank, all transfer horizons | 1.283/32 | 2.770/32 |
| Mean target coordinate standard deviation | 0.4360 | 0.0959 |

These are different target spaces; raw cross-model latent errors are not a
ranking. Visual-target MSE is 0.092360 versus persistence's 0.003251. It beats
training horizon/prefix means (0.318303/0.142760), but not the observation-based
persistence baseline. It loses to persistence at every horizon and also on the
training branch population (800-ms ratio approximately 80.75). Target variance
is nonzero; low effective rank on this small repeated-motion population alone
does not prove collapse. Static appearance dominates much of the target change
across contexts, and action effects remain poorly predicted.

The same training-only ridge readout protocol gives the following common
physical-target scores; supervised/untrained fitted-model hashes reproduce the
original readout comparison exactly:

| Model | Transfer 800-ms XY RMSE (mm) | Yaw RMSE (degrees) |
| --- | ---: | ---: |
| Visual-target JEPA + fitted readout | 7.853 | 1.148 |
| Original mixed-target JEPA + fitted readout | 6.561 | 0.862 |
| Supervised + fitted readout | 5.628 | 0.797 |
| Untrained representation + fitted readout | 5.644 | 0.733 |
| Command-history reference | 8.232 | 0.937 |

The visual model's original jointly trained motion head scored 7.227 mm and
0.893 degrees. Its training-only adapted head is worse on this transfer assay;
both results are retained. No checkpoint/head was selected using these scores.

Raw-head branch result: `docs/go2_visual_target_raw_branch_result_2026-09-17.json`.
Adapted result: `docs/go2_visual_target_readout_branch_result_2026-09-17.json`.
Each artifact's `branch_evaluation/posthoc_decomposition.json` retains the
training-only means and centered-action diagnostic. Those centered targets
use all realized branches for diagnosis and are not a deployable baseline.

## Complete recorded-navigation forecast comparison

The fixed population contains all 2,404 matched executed windows from the
latest four return-memory recordings, including both failed returns. Evaluation
took 64.07 seconds. Original JEPA predictions and every corresponding window's
position errors reproduced. Only policy packets and target-only physics were
read; no depth or new navigation execution was needed.

| Model with fitted readout where applicable | 700-ms XY RMSE (mm) | Yaw RMSE (degrees) |
| --- | ---: | ---: |
| Visual-target JEPA | 9.804 | 0.985 |
| Original mixed-target JEPA | 9.201 | 0.898 |
| Refitted mixed-target JEPA | 8.631 | 0.834 |
| Original supervised | 8.746 | 0.867 |
| Refitted supervised | 8.758 | 0.899 |
| Original untrained representation | 7.556 | 0.806 |
| Refitted untrained representation | 7.570 | 0.748 |
| Command history | 7.950 | 0.770 |
| Pose + command | 6.805 | 1.807 |

Visual-target JEPA position error is worse than original mixed-target JEPA in
each recording (10.806 vs 10.088; 8.776 vs 8.771; 10.434 vs 8.921; 10.636 vs
9.518 mm). Pooled action-increment error is 6.663 vs 6.070 mm. Left-arc error
improves slightly, but every other action category worsens, particularly
in-place turns. The same two exposed layouts and overlapping windows limit
generalization and statistical claims. These are executed-action forecasts,
not validation of alternative action outcomes or closed-loop decisions.

Result: `docs/go2_visual_target_navigation_forecast_result_2026-09-17.json`.
Runner: `scripts/evaluate_go2_visual_target_navigation_forecasts_development.py`.
The new compact per-window errors are retained alongside references to the
unchanged preceding comparison population. No model is promoted.

## Decision after the experiment

Removing future body/control from the target is insufficient by itself. The
new target varies with images, but the predictor loses badly to current-visual
persistence even on the training branch population. This directs the next
model-learning experiment toward representing current visual state explicitly
and learning action-dependent changes from it, with persistence as an explicit
baseline. Check that the training objective learns visual changes before
spending more native simulation time. A persistence-anchored visual predictor
has not yet been implemented or tested here. Do not infer that this one
negative architecture/objective comparison refutes JEPA generally.

## Controlled intervention

The target-modality assay found that future body/control dominated the mixed
JEPA target. This experiment changes only the latent prediction target: use
the EMA visual trunk, then the existing fusion projection with its 64 body and
32 control embedding coordinates fixed to zero. No future body or control
values are read by that target. All parameter tensors and initialization remain
identical to the previous mixed-target model; no projection parameters are added.

Causal RGB/body/control histories, known action sequences, motion reference,
direct and rollout outcome losses, their weights, and the common online
variance/covariance regularizer remain unchanged. That regularizer still sees
available future multimodal observations on the training side, as it does for
the controls. This is a controlled target change in the existing joint objective,
not a claim of training exclusively on visual self-supervision. The visual
target shares EMA projection weights with the mixed-context online encoder;
there is no new visual-specific variance penalty. Measure its variance rather
than assuming this target cannot collapse.

Fresh training uses the same seed and 4,694 training contexts, exact scheduled
7,200 draws, 1,200 updates, batch size 6, AdamW learning rate 0.0001, zero weight
decay, gradient clipping at 1 and EMA momentum 0.99. Reuse the completed
mixed-target JEPA and supervised controls with that identical budget and seed.
Only the final checkpoint is evaluated. No transfer outcomes select an update
or hyperparameter. Training input identities must match the original fit.

Evaluate the existing identical-history action branches with current-target
persistence, wrong actions, scene controls, variance and all horizons. Raw
latent errors across visual-only and mixed targets are not directly comparable.
Then fit the unchanged training-only ridge motion-readout protocol and compare
physical targets. Existing controls/readouts remain available. No native
navigation is launched merely because the new latent loss decreases.

Target independence is checked by passing RGB alone and replacing future
body/control with NaNs; targets must be identical. The final saved model must
reload to identical causal predictions. These checks address the actual
scientific intervention without a separate infrastructure test suite.

Resources before launch: 72 GiB RAM available, 4.4 GiB on the artifact volume,
both GPUs idle, no competing training/native process. One CPU worker on core 8
with one numerical thread; earlier equivalent fits peaked near 10.1 GB and took
139–166 seconds. Reuse completed controls rather than duplicate their training.
Only compact checkpoints and results are retained; no new depth/image archive.

Training plan: `docs/go2_visual_target_jepa_plan_2026-09-17.json`.
Training runner: `scripts/train_go2_visual_target_jepa_development.py`.
Model: `lewm/visual_target_jepa_development.py`.
Evaluation/readout runner: `scripts/evaluate_go2_visual_target_jepa_development.py`.
Training artifact: `go2_visual_target_jepa_v1_attempt_001` on the artifact volume.
