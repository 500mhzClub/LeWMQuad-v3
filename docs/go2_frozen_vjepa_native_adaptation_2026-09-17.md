# Native adaptation of the existing frozen-encoder dense predictor

Status: **TRAINING COMPLETE, BRANCH EVALUATION COMPLETE**.
Training process exited 0 after all 24 epochs and 5,280 updates per arm.
The corrected branch evaluator also exited 0. See
`go2_frozen_vjepa_native_adaptation_branch_2026-09-17.md` for positive transfer
prediction/action-discrimination results and the preserved numerical-check
failure. The recorded-mission evaluation also completed; see
`go2_dense_native_recordings_2026-09-17.md`. Neither job is still running.

Feature extraction completed in 1,013.3 seconds, with 3,979 distinct images
covering all 5,552 frame paths. All 24 epochs completed, averaging about 117.2
seconds each. Total wall time was 3,830.1 seconds. Final training L1 was 0.227766
with future actions and 0.272244 without them; these are training losses, not
transfer results. Both final checkpoints are saved and their identities are
recorded in `go2_frozen_vjepa_native_adaptation_result_2026-09-17.json`.

The unchanged August predictors lost to persistence on the native 500-ms panel
(`go2_frozen_vjepa_native_branches_2026-09-17.md`). This experiment keeps the
evidence-backed V-JEPA 2.1 ViT-L encoder frozen and adapts the existing dense
predictor to current native images and post-limiter commands.

- 3,518 admitted training sequences from 138 recordings, 5,552 frame paths.
  Three observed frames at -1000/-500/0 ms; target at +500 ms. Training-role only.
- Both arms start from the same historical RGB-rollout seed-2026080901,
  epoch-21 predictor. The action-blind arm zeros only future commands; RGB and
  past applied-command history remain identical.
- Fixed 24 epochs, 220 updates per epoch per arm, effective batch 16. Same
  shuffled order for both arms. Dense normalized-token L1, AdamW lr 3e-4,
  weight decay .01, clip 1. Final epoch only; no native-transfer checkpoint
  selection. This is one-step native adaptation of a rollout-trained parent.
- Frozen float32 encoder; normalized FP16 features cached only in RAM (upper
  bound 8.13 GiB). Predictor uses BF16 autocast and float32 normalized loss.
- GPU microbatch measurements: 4/8/16 give 59.99/63.47/64.21 samples/s;
  batch 16 selected, 6.00 GiB peak allocation in the training-step benchmark.
  One GPU process interleaves the two arms and shares the feature cache.
- At 600 processed frame paths, extraction took 142.8 seconds and GPU
  utilization was observed at 83%. Initial measured total estimate: 65–75
  minutes including extraction, training and checkpoint writes; not a deadline.

Plan: `go2_frozen_vjepa_native_adaptation_plan_2026-09-17.json`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_vjepa_native_adaptation_v1_attempt_001/`.
Completed training handle: tool session 33534, PID 13975, exit 0. `process.json`
is a historical locator. Latest per-arm checkpoints contain optimizer/RNG state.

`scripts/evaluate_go2_frozen_vjepa_native_adaptation_development.py` completed
with the same venv and repository PYTHONPATH, using all 36 fixed native branches.
It
preserves training versus exposed-transfer roles, predicts before future RGB
loading, verifies action-blind equality and the unchanged persistence score,
and saves compact error/action-effect matrices. Attempt 002 completed after the
preserved identical-input floating-point assertion failure in attempt 001.
Do not rerun the completed evaluation or use intermediate checkpoints.

No new simulation, collection, encoder fitting, navigation evaluation, or
deployment occurs in this experiment. The positive adaptation result concerns
prediction using pretrained JEPA features, not an advantage from training this
repository's encoder with JEPA. Prospective decision and navigation comparisons
remain necessary.

## Existing motion-label coverage for a downstream readout

A CPU-only join of this attempt's `samples.json` to the existing family,
pre-switch and short-pulse `windows.json` records confirms that **all 3,518
admitted sequences have valid 500-ms XY/yaw targets**. The label horizon is the
same as the dense predictor's native adaptation horizon.

| Training source | Sequences | Median / p90 displacement (mm) | Median / p90 absolute yaw (degrees) |
| --- | ---: | ---: | ---: |
| Family | 1,122 | 8.275 / 83.000 | 3.714 / 13.226 |
| Moving-action switch | 2,252 | 7.965 / 84.746 | 6.694 / 13.235 |
| Short pulse | 144 | 8.045 / 12.674 | 1.530 / 6.719 |

Exactly 2,041 sequences have nonzero applied commands at all five future ticks;
the training data are not limited to the millimetre-scale pulse differences in
the exposed branch diagnostic. This supports trying a training-only motion
readout without new collection if the adapted predictions warrant it. It does
not establish that the visual features permit accurate motion decoding. There
are zero positive contact labels in this admitted population, so it cannot
support a learned collision-discrimination claim.

The current planner needs eight 100-ms physical forecasts, while this predictor
produces one 500-ms visual forecast. A 500-ms motion readout alone would not
satisfy that controller interface. See
`go2_frozen_vjepa_native_visual_goal_2026-09-17.md` for that integration gap and
the limited positive action-selection evidence already present in the unchanged
historical models. No readout fit or controller change has been launched.
