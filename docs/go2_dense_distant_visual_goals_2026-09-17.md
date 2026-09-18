# Dense predictor decisions toward later visual goals

Status: **COMPLETE**, tool session 25520 exited 0. This is a retrospective
development diagnostic, not a navigation result. No weights were changed.

The adapted predictor forecasts the native image representation 500 ms after
the common departure. It ranks the same three recorded action candidates by
normalized dense-feature MSE to goal images at 500, 800 and 1,200 ms. The 800-ms
goal was designated primary before execution; 500 ms reproduces the earlier
control. An oracle ranks actual 500-ms successor features against each later
goal. Source-branch identity alone is not correctness for a later goal: a
different 500-ms branch can be closer. Exact ties are averaged uniformly.

| Transfer goal time | Predictor/oracle action agreement | Action XY regret mm | Blind XY regret mm | Action yaw regret degrees | Blind yaw regret degrees |
|---|---:|---:|---:|---:|---:|
| 500 ms | 18/18 | 0.000 | 1.100 | 0.000 | 1.791 |
| 800 ms | 17/18 | 0.434 | 0.655 | 0.115 | 1.542 |
| 1,200 ms | 17/18 | 0.572 | 0.652 | 0.043 | 1.363 |

Blind means either the no-future-action predictor or persistence: neither can
distinguish candidate actions given the same history, so both have identical
expected decision outcomes. Expected agreement with the visual oracle is 6/18.
XY and yaw regrets are scored separately against the best actual 500-ms endpoint
for each physical criterion, using physical labels only after selection.

For the actual-future visual oracle, XY regret is 0.397 mm at 800 ms and 0.672 mm
at 1,200 ms. Thus visual distance is not an exact physical distance, even with
perfect forecasts. The predictor's smaller physical regret at 1,200 ms does
not mean it outperforms perfect prediction: the objectives differ.

All 36 original branches are retained, 18 per role. All training-role scores
and every cost matrix/choice are in the result JSON. Two previously exposed
geometries per role and six shared histories make these dependent small tasks.
The branches contain a one-tick pulse followed by hold. The apparent temporal
extension does not establish long-horizon rollout or sustained navigation.

The first attempt failed during physical scoring because composing projected
XY/yaw labels does not reproduce the full 3D body transform. Its source, plan,
forecast-completion record and failure remain in `distant_visual_goals_attempt_001`.
The corrected attempt derives goal displacements directly from recorded 3D
poses and reproduces all existing 500/800-ms labels to 1e-12. It also reproduces
the prior 500-ms forecast cost matrices to 1e-6. Forecasts complete before future
RGB and motion are loaded. Goal RGB is explicitly supplied as the task.

One GPU process reused 136 distinct encoded images with four CPU threads;
available RAM was 72 GiB and output-volume free space 4 GiB. No competing compute
process was present. No dense feature cache was retained on disk.

Decision: advance to a bounded prospective native image-goal control experiment
using the unchanged encoder/predictor and this visual cost. Use real newly
acquired observations after each selected command, preserve the action-blind
comparison, and measure physical goal-reaching separately. Start with the
predictor's actual 500-ms action interface; do not impersonate the current
eight-horizon motion controller. This pilot will still leave full-maze routing,
matched reactive comparisons, JEPA-training attribution and real-time/hardware
validation outstanding.

Source: `scripts/evaluate_go2_dense_distant_visual_goals_development.py`.
Result: `go2_dense_distant_visual_goals_result_2026-09-17.json`.
Output: `go2_frozen_vjepa_native_adaptation_v1_attempt_001/distant_visual_goals_attempt_002`
under `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/`.
