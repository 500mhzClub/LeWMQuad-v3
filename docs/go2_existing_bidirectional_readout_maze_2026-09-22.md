# Existing bidirectional visual readout on the fixed maze-motion panel

Status: frozen-checkpoint evaluation completed on CPU cores 4--7, session 73368,
with exit zero in 540.58 seconds. All 128 rows completed. The reused decoder
performs worse than the starting mixed-data head and is not promoted. No
training or navigation setting changes. The prospective maze coordinator
continues separately in session 64661, with eleven completed assignments.

The all-motion horizon comparison found a large transfer gap even with actual
future images: 700-ms translation XY RMSE 11.19 mm on selected training-exposed
examples versus 77.38 mm on the maze panel. Training the same head at longer
horizons did not repair that gap. Before fitting another variation, this study
reuses the already completed direct visual-goal readout from September 17.

That head was trained on 24,294 original-training pairs at 100--3,000 ms with
balanced forward/reverse order. It has a different shared projection/head and
enforces exactly zero displacement for identical images during training and
inference. Its checkpoint SHA-256 is
`c270cffac40b0e917f05a4f83edf105d132b970586290ef610a0afc8077ef74c`.
Its earlier local-goal successes and fresh-task failures remain unchanged:
this is a new motion-decoding diagnostic, not another local-navigation trial.
See `go2_direct_visual_goal_readout_2026-09-17.md`.

The evaluator uses exactly the existing 32 translation and 32 turn windows
from the completed maze-00 action run, at 500 and 700 ms. It compares the
unchanged head on actual-future, action-predicted and action-blind features.
The 317-image context population, predictor checkpoints, applied command tapes
and window selection are unchanged. Starting mixed-head forecasts must reproduce
the previous evaluation within 2e-5; the remaining saved head and command-history
predictions are reused. All outputs, including failures, are retained.

The direct head was trained on planar body-heading XY and world-yaw change;
the motion heads were trained on full-body-frame XY. Both target conventions
are therefore scored, without correcting predictions. On this fixed panel,
planar versus body XY targets differ by 0.1145 mm RMS and at most 0.5277 mm.
Primary results use common planar targets, and secondary body-target results
preserve comparability with the preceding report. Native poses remain offline
evaluation inputs only.

This is a practical comparison of existing decoders. Architecture, data-pair
coverage, direction augmentation, normalization and update budget differ;
none of their effects is isolated, and this is not a JEPA-objective experiment.
The head was trained on actual images, so its application to predicted features
also tests that transfer. One exposed trajectory with overlapping windows does
not demonstrate independent navigation, cross-family transfer or real-time
validity. No head is automatically promoted and the live cohort stays fixed.

Runner: `scripts/evaluate_go2_bidirectional_readout_maze_development.py`.
Output:
`.generated/navigation_development_artifacts_v1/go2_all_motion_horizon_readout_v1_attempt_001/existing_bidirectional_readout_maze_v1/`.
The saved `plan.json` fixes the checkpoint and input identities before inference.
Only small metadata and predictions are written; features remain in RAM.

## Completed result

Each cell below is common-planar-target XY RMSE in mm / yaw RMSE in degrees.
The population has 32 windows in each motion group at each horizon.

| Readout / input or reference | Translation 500 ms | Translation 700 ms | Turn 500 ms | Turn 700 ms |
|---|---:|---:|---:|---:|
| Existing bidirectional / actual future | 85.57 / 16.57 | 111.73 / 20.42 | 56.51 / 14.93 | 73.62 / 20.49 |
| Existing bidirectional / action prediction | 59.86 / 15.46 | 69.58 / 17.67 | 49.31 / 14.22 | 52.23 / 16.52 |
| Existing bidirectional / blind prediction | 51.17 / 17.04 | 68.42 / 19.46 | 38.16 / 12.04 | 44.00 / 14.98 |
| Starting mixed / actual future | 53.21 / 5.44 | 78.29 / 7.47 | 17.30 / 5.92 | 19.98 / 10.35 |
| Starting mixed / action prediction | 42.49 / 4.91 | 67.18 / 7.02 | 16.83 / 6.34 | 18.37 / 10.24 |
| Command history | 6.13 / 0.99 | 8.66 / 1.06 | 6.01 / 0.59 | 6.72 / 0.77 |
| Zero motion | 63.90 / 8.22 | 92.67 / 11.26 | 16.41 / 9.81 | 17.77 / 14.32 |

The existing head's action-conditioned estimates are worse than the starting
mixed head in both XY and yaw in every group/horizon cell. Its actual-future
estimates are also worse than zero motion in both XY and yaw throughout this
panel. Removing predictor error therefore does not make this decoder useful
here. Blind estimates have smaller XY errors than action estimates, while the
yaw comparison varies by motion group. Neither implies a navigation advantage.

Scoring against the full-body XY targets leaves these conclusions unchanged:
for example, 700-ms translation action XY is 69.59 mm versus planar 69.58 mm,
and actual-future XY is 111.72 versus 111.73 mm. The small target-convention
difference cannot explain the large failure on this panel. All three
starting-head future-input predictions reproduced the preceding evaluator
within 2e-5, and identical-image output was exactly zero at every selected
context. These checks validate this diagnostic's execution, not transfer.

The broader and bidirectional training plus zero-identity architecture of this
existing checkpoint is not a practical replacement for the deployed head.
The result does not isolate which ingredient failed or prove that reversal
augmentation is ineffective in general. It strengthens the case to investigate
training-view coverage and transfer before another decoder variation. Retain
the checkpoint and negative result; no new fit or controller promotion follows.

`result.json` retains all 128 predictions, both target conventions, complete
metrics for every reference head, and the plan identity. No features were
persisted and no new sensor recordings were collected.
