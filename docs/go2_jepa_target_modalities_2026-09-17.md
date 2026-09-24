# Which future modalities drive the existing JEPA target?

**Complete.** All three frozen representations and all eight current/future
modality combinations were evaluated in 5.56 seconds; the process exited zero.
The full future targets exactly reproduce the previous branch assay. There was
no training, new navigation, depth access or change to deployed models.

The primary population is the same 18 exposed development-transfer contexts
at 800 ms, comprising six groups of three actions with identical causal inputs.
All 36 training/transfer contexts and all eight horizons are retained. Replace
RGB, body history and control history independently with their current values
or their actual future values. These hybrids measure encoder sensitivity;
they are not physically realized counterfactual observations.

Let T be the full future target and C the current target. The first column below
is MSE(T, target with current RGB and future body/control) / MSE(T, C).
The second is MSE(target with future RGB and current body/control, C) / MSE(T, C).
They are different diagnostics, not complementary or additive variance shares.

| Frozen representation | Freezing RGB: change relative to full temporal change | Updating RGB alone: change relative to full temporal change |
| --- | ---: | ---: |
| JEPA | 0.0254% | 0.0239% |
| Supervised | 2.6073% | 2.3290% |
| Untrained | 0.000225% | 0.000227% |

For JEPA, full temporal target MSE is 0.0585766. Updating only future control
history produces 0.0465005; updating only future body history produces
0.0030631; updating only future RGB produces 0.0000140. The fusion is nonlinear,
so these values must not be presented as an additive attribution or percentages
of explained variance. In this assay, changes in the mixed target are dominated
by body/control information, particularly the applied-command history.

This is not a static-image artifact: normalized RGB temporal pixel MSE is
0.008632, and within-group action-dependent RGB pixel variance is 0.003108.
Nevertheless, these aggregate pixel differences do not by themselves establish
semantically useful scene changes.

RGB still contributes some action-dependent information. For JEPA, freezing
RGB changes the centered action effect by 5.55% of full centered action-effect
energy. Updating RGB alone yields only 0.00001046 action variance compared with
0.00019330 for the full mixed target. The refitted predictor retrieves 11/18
full-target action branches and 12/18 when future RGB is frozen. The latter is
a diagnostic using an altered target, not an improved task score. Its latent
MSE similarly changes only from 0.002098 to 0.002067.

## Scientific implication and next experiment

The earlier improvement in mixed-target latent prediction is weak evidence
of learning future visual/environment state. It remains a valid improvement
on its stated multimodal prediction objective; this assay does not establish
that JEPA methods generally fail, or that RGB is irrelevant in every scene.
The population is small, short-horizon and already exposed during development.

The next training comparison should remove future body and control information
from the prediction target while preserving RGB, body state and known actions
as causal context. Train a visual-target JEPA arm against the current mixed-target
objective and matched supervised/untrained controls on the same data/budget.
Evaluate common downstream targets, action-specific visual change, persistence
and action-independent baselines, representation variance, then practical
forecasting. Compare losses within each target space, never use raw loss across
different target definitions to declare superiority. Do not advance to another
navigation sweep merely because a new latent loss falls.

Plan: `docs/go2_jepa_target_modalities_plan_2026-09-17.json`.
Complete curves: `docs/go2_jepa_target_modalities_result_2026-09-17.json`.
Runner: `scripts/probe_go2_jepa_target_modalities_development.py`.
Compact target arrays and duplicate result are retained under
`go2_jepa_target_modalities_v1_attempt_001` on the artifact volume.

No visual-target model has been trained by this experiment. Sensor robustness,
sim-to-real work, candidate-count changes and non-maze tests remain deferred.
