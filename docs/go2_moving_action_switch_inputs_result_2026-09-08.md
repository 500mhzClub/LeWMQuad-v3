# Moving-action switch causal input result

The whole-population input check passed for all 144 available branches after
the complete native collection and exact-prefix audit. Every model input uses
actual frames 10–13 and the known suffix plan. All 24 matched prefix groups
remain exactly equal after RGB/body/control preprocessing.

Training materialization used 72 cells, 454 valid motion/future-observation
targets, all 576 contact targets and 122 positive cumulative-contact targets.
The 72 geometry-transfer cells used past-only inference: zero future images
were read as inputs or training targets. No future image entered any model
input. Contact-censored motion remains masked NaN, rather than fabricated zeros.

The existing untrained cumulative-contact model accepted the inputs, returned
finite active predictions with exact clocks/masks, and produced a finite full
training-target JEPA objective. Its state remained unchanged and it acquired
no gradients. Optimizer steps were zero. Runtime after launch was
35.6750734330 seconds on one CPU thread; 848 source paths were bound.

Three separate branch-only schedules were checked for seeds 2026091001,
2026091401 and 2026091402. Each contains 1,200 six-sibling batches and exactly
100 draws per training cell, with no transfer cells. The separately recorded
augmented-fit protocol uses the first 600 batches from each branch schedule
alongside 600 original-family batches; no fitting occurred in this input check.

The root is `go2_moving_action_switch_inputs_v1_attempt_001` under the owned
navigation development artifact base.

| Receipt | SHA-256 |
|---|---|
| `launch.json` | `9e55de54170d64cb7707344b79953ff4c6f5c33d39ee0c5f181ddea6b32789a6` |
| `training_schedules.json` | `c2ee75774336efec58bd489a7feddb176116bd9906fc31c2cbc16ee90d612bef` |
| `tensor_index.json` | `4ed2256f7c88933f89e03686cc7e314286b5835dbe1f8a4fa5de965ef6b723b7` |
| `resource_monitor.jsonl` | `bf34ab09e5cc4dec6e1e7b3c7e13a54a656122fa1d0bb9054d469b5786d13fbe` |
| `result.json` | `2a699ba2a37ce26565324c4e7dbf4f97b957f952c709c470fcf3fb4aaedf2083` |

The bound collection result is
`e89ea05cb4590164bcdc48e45136f022cdc941a7b63ea13bedcd4cfc1e6dffa1`.
Twelve focused new sample/view/stream tests passed, covering target separation,
censoring, assignment and clock rejection, cache isolation, balanced schedules
and transfer-role exclusion. This establishes development fitting inputs, with
no newly trained model, native command selection, verified arrival, independent
maze evaluation, real-time or deployment result.
