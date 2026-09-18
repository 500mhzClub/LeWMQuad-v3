# Native dense-predictor adaptation: completed branch results

The frozen V-JEPA 2.1 encoder plus the adapted action-conditioned predictor
beats persistence and the matched no-future-action predictor on the exposed
native branch panel. This is positive component evidence, not navigation proof
or an isolated benefit from training the visual encoder with JEPA.

Training completed successfully in 3,830.1 seconds (63.8 minutes), including
1,013.3 seconds of feature extraction. Both arms completed the fixed 24 epochs,
5,280 updates and identical training sequence orders. Final training L1 was
0.227766 with actions and 0.272244 without. Neither checkpoint was selected
using transfer outcomes. The encoder remained frozen throughout.

## Exposed geometry-transfer panel, 500 ms

| Model | Feature MSE | MSE / persistence | Strict correct-action wins | Centered action-effect error / zero effect |
| --- | ---: | ---: | ---: | ---: |
| Unchanged historical one-step | 0.622875 | 1.6061 | 11/18 | 1.0260 |
| Unchanged historical rollout | 0.653284 | 1.6845 | 11/18 | 1.0246 |
| **Adapted, future actions** | **0.243141** | **0.6269** | **18/18** | **0.8239** |
| Adapted, no future actions | 0.328250 | 0.8464 | 0/18 (all ties) | 1.0000 |
| Current-feature persistence | 0.387830 | 1.0000 | 0/18 (all ties) | 1.0000 |

The adapted action model reduces feature error by 37.3% against persistence and
25.9% against the matched no-action model. It improves mean error against the
no-action arm in every one of the six transfer history groups. Centered
action-effect error also beats zero effect in all six groups, by 17.6% pooled.
Thus the gain includes action-dependent changes, not just a better prediction
of the common future. The historical rollout is the shared initialization of
both adapted arms; the historical one-step is an additional unchanged reference.

Training-role branch scores are in-sample: action MSE 0.115091, no-action
0.253294, persistence 0.393201. The action arm retrieves 18/18 there also.

In the retrospective reachable-visual-goal diagnostic, the adapted action model
chooses all 18 correct transfer branches, giving zero selected-endpoint regret
within this three-action recorded candidate set. This does **not** mean zero
physical motion prediction error: no physical-motion decoder was evaluated.
Uniform tie-breaking for no-action/persistence gives 6/18 expected correct choices,
1.100 mm expected endpoint error and 1.791 degrees expected heading error.

The 3,518 admitted training sequences all have role `train` and clusters 00/01
(family, switch and short-pulse sources). The transfer panel uses clusters 02/03,
with geometry references `family_episode_026` and `family_episode_003`; none of
its sample IDs enter this fit. These two geometries were already exposed in
earlier development evaluation: held out from this training, not untouched final
tests. There are six shared-history groups and 18 dependent goals, with tiny
physical branch differences. No independent navigation, sustained-command benefit,
hardware result, or superiority over a matched DINO encoder follows.

The prepared 384-window recorded-mission comparison is the next diagnostic before
a physical-motion/decision interface.

## Evaluation failure and correction

The first evaluator completed forecasting but stopped at an exact bitwise
identity assertion for the no-action arm. A diagnostic with identical inputs
measured within-batch maximum difference 3.624e-5 and MSE 2.043e-13, all finite:
GPU floating-point roundoff. Its failure JSON, log, source and diagnostic remain
in the original `branch_evaluation/` directory.

Attempt 002 computes one no-action forecast for each group after verifying exact
input equality, then reuses it for the three candidates. This enforces the
model's mathematical action independence without letting numerical noise create
retrieval wins. The strict identity check remains. No checkpoint, target, action
model or scientific selection changed. The corrected evaluation completed with
exit 0 in 23.3 seconds and reproduced the reference persistence score.

Results: `go2_frozen_vjepa_native_adaptation_branch_result_2026-09-17.json` and
`go2_frozen_vjepa_adapted_visual_goal_result_2026-09-17.json`.
Artifacts: the native adaptation attempt's `branch_evaluation_attempt_002/`.
No dense feature cache was retained on disk.
