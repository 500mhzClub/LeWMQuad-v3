# Learned physical distance for dense visual-goal planning

Status: **FIT, NEAR-GOAL EVALUATION AND LIVE COMPARISON COMPLETE**. Training PID 36304,
tool session 29941 exited 0. All 24 fixed epochs and 4,560 updates completed
in 863.2 seconds, including 835.6 seconds to encode 3,979 distinct images.
Training log-distance MSE fell from 0.233906 to 0.006400. Zero identity and
symmetry checks passed. The evaluator (tool session 29163) also exited 0. The four-case prospective
live comparison is complete: see `go2_dense_metric_goal_pilot_2026-09-17.md`.
Mean action-trial final XY error fell from 118.71 cm to 3.16 cm, but neither
trial finished within both position and heading tolerances.

The matched overshoot diagnostic showed that raw dense-feature MSE prefers
forward even with perfect future images at both tested near-goal states. This
experiment changes the goal-distance function while keeping the V-JEPA encoder
and the action/no-action predictors frozen.

## Training definition

- Use the same 5,552 admitted training image paths from the native adaptation,
  covering 138 recordings. Every episode is checked as training through its
  admitted sample IDs. Pair endpoints precede any native contact. No transfer
  branches, live-pilot recordings or overshoot alternatives enter fitting.
- 24,294 within-recording pairs at fixed separations 100, 200, 500, 1,000, 2,000
  and 3,000 ms, whenever both admitted image slots exist. Source counts:
  family 7,776; switch 15,132; short pulse 1,386. Moving and stopped observations
  remain included; no selection by model errors or transfer outcomes.
- Squared target distance is `XY_separation^2 / 0.03^2 + wrapped_yaw^2 /
  radians(5)^2`, using world-planar positions and world headings. Scales are the
  preceding pilot's physical tolerances. Native poses supply training targets
  only; neither pose, body measurements nor commands enter the new model.
- Average adjacent 2x2 normalized patches to a 12x16 grid, then use a shared
  1024-to-32 GELU projection and spatial flattening into a 64-dimensional linear
  embedding. Mean squared embedding difference gives the pair cost. There are
  426,016 parameters. Sharing gives symmetry, nonnegativity and zero distance
  for identical features.
- MSE between `log1p(predicted_cost)` and `log1p(target_cost)`; AdamW lr .001,
  weight decay .0001, gradient clip 1, batch 128, seed 2026091707. Fixed final
  epoch after 24 epochs; no checkpoint or threshold selection on transfer.
- Fit only observed-observed feature pairs. Apply this same cost later to
  observed versus predicted successors to expose any distribution gap.
  This uses additional physical supervision and does not isolate JEPA training.

The target's quartiles are 0.0224, 1.1317 and 18.9074; maximum 484.4112.
Log loss avoids allowing distant pairs alone to dominate. This is one fixed
architecture and seed, not a hyperparameter search.

## Execution and evaluation

One GPU process uses CPU cores 8-11 and four CPU threads. Available RAM was
72 GiB, R9700 VRAM 31.86 GiB and output storage 2.6 GiB. Batch-eight encoder
extraction reuses the preceding measured implementation and precision. Up to
2.03 GiB of pooled FP16 features stay in RAM and transfer to GPU for float32
head fitting. Only the latest resumable small checkpoint and metrics are kept
on disk. No competing compute job was present. Do not alter the running fit.

The near-goal evaluator is **complete**:
`scripts/evaluate_go2_dense_goal_metric_development.py`. It reproduces original
live raw costs and actual-successor raw costs, then compares raw MSE and the
learned metric for actual future, action prediction, no-action prediction and
persistence. Forecast inputs remain causal; physical outcomes are scoring-only.
Blind/persistence costs are computed once and broadcast so candidates tie.

The prospective cost-only controller and runner are also implemented and
**complete with four executed trials**:
`lewm/dense_metric_goal_control_development.py` and
`scripts/run_go2_dense_metric_goal_pilot_development.py`. They reuse the original
native task loop, goals, six commands, 500-ms timing, seeds, budgets and stops.
Only the goal cost changes. With actual future images, the learned metric
selects hold (the physically best action) at both diagnosed states. With
predicted futures it selects right arc at case 0 and hold at case 3. Raw MSE
selects forward at both states, even with actual future images. This separates
a corrected cost-ranking error from a remaining prediction/distribution gap.
See `go2_dense_goal_metric_near_goal_result_2026-09-17.json`. The final checkpoint is `metric.pt`, SHA-256
`9f89fc391a27875075a47e9383879153e8ebc22dd5e19cd0209041caedc6a7f1`.

These are exposed post-hoc development states. Improved ranking would justify
a prospective controller experiment, not establish navigation or JEPA
superiority. A symmetric distance is not a stopping detector, collision
predictor, exploration policy, or memory system.

Plan: `go2_dense_goal_metric_plan_2026-09-17.json`.
Sources: `lewm/dense_goal_metric_development.py` and
`scripts/train_go2_dense_goal_metric_development.py`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_dense_goal_metric_v1_attempt_001/`.
