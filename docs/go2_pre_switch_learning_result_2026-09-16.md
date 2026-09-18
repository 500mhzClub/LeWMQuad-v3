# Prospective-switch data successor

The experiment changes training coverage while retaining the original neural
inputs, architecture, losses and 1,200-update budget. JEPA, direct and
supervised-rollout models use seed 2026091001 and the same 4,514-context
schedule. The 504 added windows precede switches already executed in the
original training recordings. All 600 family batches and all per-trial draw
totals remain unchanged. No new independent training episodes are claimed.

Training uses separate causal-input and private-future readers and the existing
observation tensor and target materializers. Packets are loaded once per trial;
materialized samples stay in memory. This avoids new depth storage or a disk
tensor cache. Checkpoints retain the existing evaluation-load format. The
small-output worker requires 1 GiB free, instead of inheriting the historical
40 GiB reserve intended for larger concurrent jobs. Actual initial model state
matches the predecessor seed.

The evaluation population is fixed before running model predictions: 420
original available development transfer contexts plus 504 pre-switch contexts
from the corresponding 72 transfer recordings. Their two parameter clusters,
02 and 03, are disjoint from training clusters 00 and 01. They are previously
used development geometry, not new independent maze trials. All six final
models—three predecessors and three successors—are evaluated on identical
causal observations, commands and motion targets. Native pose is evaluation
truth only; future images are not inference inputs.

Report XY endpoint and yaw error at 300, 700 and 800 ms, separating original
contexts, pre-switch departures, stable future commands, future switches,
braking and the two geometry clusters. Command-integrated motion and stationary
persistence provide simple references. Predictions are raw, without a residual
fit, so this isolates the neural data change. No best-checkpoint selection,
new seed search or controller change is part of this experiment.

Artifacts under the configured development base:

- `go2_pre_switch_matched_fits_v1_attempt_001`: fixed-budget fits and ledgers.
- `go2_pre_switch_transfer_targets_v1_attempt_001`: fixed development targets.
- `go2_pre_switch_transfer_comparison_v1_attempt_001`: planned prediction readout.

Entry points are `scripts/train_go2_pre_switch_successor_development.py`,
`scripts/prepare_go2_pre_switch_transfer_development.py`, and
`scripts/evaluate_go2_pre_switch_successor_development.py`.

## Completed result

All three fits completed: 3,600 optimizer updates total, 301.21 seconds
including 65.37 seconds of loading. Peak process RSS was 9.75 GB; the three
checkpoints total 20.68 MB. Prediction evaluation completed in 17.04 seconds.
There are 924 available contexts and 870 valid 700-ms motion targets.

| Method | Original XY RMSE mm | Added-switch XY RMSE mm | Original yaw RMSE deg | Added-switch yaw RMSE deg |
|---|---:|---:|---:|---:|
| JEPA | 25.951 | 23.279 | 3.947 | 6.845 |
| Direct | 21.990 | 27.893 | 3.724 | 3.714 |
| Supervised rollout | 22.406 | 18.077 | 2.989 | 3.357 |
| Command integration reference | 16.192 | 16.192 | 2.286 | 2.286 |

These are pooled descriptive errors on identical targets, not independent
navigation trials. The direction of the 700-ms XY changes is the same in each
of the two geometry clusters. At 300 ms supervised rollout worsens
14.276 to 15.105 mm; at 800 ms it improves 25.712 to 20.796 mm. JEPA's XY
improvement is accompanied by substantially worse yaw at all three horizons.
Direct XY worsens at all three. Command integration has lower aggregate XY
and yaw error than every neural model at all three horizons.

The future-braking subset contains 100 valid targets at each reported horizon.
All three data successors regress in its 700-ms XY error:

| Method | Original braking XY RMSE mm | Added-switch braking XY RMSE mm |
|---|---:|---:|
| JEPA | 22.538 | 27.106 |
| Direct | 22.014 | 30.209 |
| Supervised rollout | 15.891 | 19.745 |
| Command integration reference | 14.230 | 14.230 |

Thus increased command coverage is not sufficient. Preserving the per-trial
draw budget necessarily changes the within-trial allocation among switch,
steady and braking contexts. This experiment does not isolate coverage from
that allocation change. It provides no basis for assuming the larger sample
set is automatically better or for promoting the successors to navigation.

The first evaluation setup exited before model predictions because original
switch metadata nests causal clock/index fields in its receipt. The reader
was corrected to use that recorded receipt for all rows. The incident is
retained as `evaluation_setup_failure_v1.json` in the transfer-target root;
weights, target population and fitting were unchanged.

Next, test a motion-prediction parameterization that starts from the known
command-integration reference and learns observation-conditioned deviations.
This addresses a now-measured weakness: the current neural heads must relearn
basic commanded motion and still lose to that elementary reference. Keep the
same data, seed, update budget and all three objectives for this separate
comparison, so any change can be attributed to the parameterization. Preserve
RGB-conditioned latent prediction; do not substitute a command-only navigator
for the learned-world-model objective. Evaluate short-horizon and braking
regressions as well as aggregate error before any online integration.

This parameterization successor has now been implemented, fitted and evaluated;
see `docs/go2_nominal_motion_residual_learning_2026-09-16.md`. It improves some
motion forecasts but still trails the fitted pose-based control on recorded
maze execution, and has not been promoted to navigation.
Explicitly observed motion, multiple future switches, actual short pulses,
obstacle interaction, independent-maze navigation and realistic sensing/timing
remain outstanding. No successor navigation trial was launched.
