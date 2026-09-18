# Common dense-visual motion readout

Status: **COMPLETE; physical planner benefit not established**. Training
PID 27125/session 33156 and evaluation session 21897 both exited 0. The fixed
24 epochs completed 1,320 updates in 844.9 seconds, including 837.5 seconds of
feature extraction. Evaluation took 23.7 seconds. No job remains live.

The adapted dense predictor now improves visual prediction and branch action
selection. This experiment asks whether its future features support physical
motion estimates useful to the waypoint planner. It trains one common decoder
on **observed current and observed future features**, then evaluates exactly the
same decoder on action-conditioned predictions, no-action predictions and
persistence. The observed-future evaluation is an oracle diagnostic, not a
deployable forecast. No candidate actions, body state or command history enter
the neural readout directly.

- Same 3,518 admitted training sequences and 500-ms body-frame XY/yaw targets.
  All roles are training-only. Neither exposed branch targets nor mission
  recordings enter the fit.
- Frozen V-JEPA image encoder and existing dense predictor checkpoints remain
  unchanged. No predictor forward/backward pass is needed during decoder fitting.
- Normalize dense tokens exactly as before; average adjacent 2x2 patches to keep
  a 12x16 grid. Concatenate current features and future-minus-current features.
  Shared 2048-to-32 GELU projection, flatten spatial cells, 128-unit GELU head,
  three XY/yaw outputs: 852,515 trainable parameters.
- Training-only target mean/std; standardized MSE, AdamW lr .001, weight decay
  .0001, gradient clip 1, batch 64, fixed 24 epochs, seed 2026091705. Final epoch
  only, without transfer-based checkpoint selection. Latest resumable checkpoint
  is overwritten atomically; pooled features are held only in RAM.
- Encoder benchmark on eight training images: batches 1/4/8 achieved
  4.151/5.033/5.053 frames/s. Batch 8 selected, peak allocated VRAM 1.70 GB;
  normalized output MSE against batch one was zero for these examples. Actual
  extraction completed all 3,979 unique images in 837.5 seconds. Final training
  standardized MSE was 0.005907, down from 0.536843 in the first epoch.
- Resources before preparation: 72 GiB available RAM and 4.0 GiB output-volume
  free space. Maximum pooled feature cache is about 2.03 GiB. One GPU process,
  four CPU threads, CPUs 8-11. The independent command-reference check used CPU
  cores 4-7 and completed; no competing encoder or simulation is running.

The existing command-history baseline produced finite 500-ms XY/yaw predictions
for all 36 branches. Keep it, plus zero motion, as physical controls. That existing
baseline used its original larger training population and four command histories;
it is a strong existing comparator, not a newly matched-data fit.

The completed branch evaluator reports
XY RMSE, wrapped-yaw RMSE and centered action-effect errors for all 36 original
branches, preserving training versus exposed-transfer roles. Predictions precede
future RGB access. No-action forecasts are computed once per identical input
group to prevent GPU roundoff from creating action preferences.

## Results

All errors are at 500 ms. Each role contains 18 branches in six shared-history
groups across two geometries; training-role results are in sample.

| Common readout input / control | Train XY RMSE mm | Train yaw RMSE degrees | Transfer XY RMSE mm | Transfer yaw RMSE degrees |
|---|---:|---:|---:|---:|
| Action-conditioned prediction | 3.608 | 0.491 | 8.189 | 1.581 |
| No-future-action prediction | 6.553 | 1.810 | 10.921 | 2.847 |
| Visual persistence | 8.599 | 2.088 | 6.338 | 2.197 |
| Existing command history | 5.113 | 0.588 | 5.113 | 0.588 |
| Zero motion | 9.447 | 1.961 | 9.447 | 1.961 |
| Observed future (oracle) | 2.189 | 0.289 | 10.064 | 1.052 |

Action conditioning helps relative to the blind model, but the common visual
motion readout does not beat command history. Its poor transfer even with actual
future RGB identifies a readout limitation; it cannot isolate a predictor
failure or establish that these features contain no motion information.

A post-hoc physical decision diagnostic also fails to establish planner utility.
For 0.25-m point goals at -45/0/+45 degrees, transfer action-choice regret is
1.048 mm for the action model, 0.873 mm for uniform blind choices and 0.153 mm
for command history. For +/-30-degree scan goals, action and command history
both have zero regret; blind choices give 2.015 degrees. Subtracting the decoded
persistence offset does not broadly improve errors and is not adopted.

Corresponding actual motion labels are exactly identical across all four
geometries for each prefix/action combination. This free-space pulse panel tests
visual transfer and action discrimination, but cannot demonstrate an advantage
in geometry-dependent physical dynamics. All admitted training contact labels
are zero. These limitations do not erase the positive visual-prediction and
visual-goal branch-selection results.

Next assess direct visual-goal costs beyond the exact forecast target, including
an observed-future oracle to distinguish goal-cost failure from forecast error.
Do not tune this decoder repeatedly against the same exposed transfer panel.
Prospective closed-loop navigation remains necessary.

Authoritative results: `go2_dense_visual_motion_readout_fit_result_2026-09-17.json`,
`go2_dense_visual_motion_readout_branch_result_2026-09-17.json`, and
`go2_dense_motion_interface_diagnostic_2026-09-17.json`.

A failed observed-future readout would not establish that motion information is
absent: this is one architecture with spatial pooling. A gap between observed-
future and predicted-future decoding may reflect prediction error or a readout
distribution mismatch. Even a successful 500-ms endpoint readout does not supply
the current controller's eight 100-ms forecasts or establish navigation benefit.

Plan: `go2_dense_visual_motion_readout_plan_2026-09-17.json`.
Training: `scripts/train_go2_dense_visual_motion_readout_development.py`.
Readout: `lewm/dense_visual_motion_readout_development.py`.
Evaluation: `scripts/evaluate_go2_dense_visual_motion_readout_development.py`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_dense_visual_motion_readout_v1_attempt_001/`.
