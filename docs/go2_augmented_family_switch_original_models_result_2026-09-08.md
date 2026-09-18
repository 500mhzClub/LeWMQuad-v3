# Original-model predictions on the new switch population

All six unchanged original models completed predictions on both source
populations. For every model, all saved original-family prediction fields
reproduced exactly: 336 training and 348 geometry-transfer windows. All twelve
model/role prediction arrays were saved before scoring this comparison.

The new geometry-transfer population contains 72 branch executions, including
60 switched-action and 12 repeated-action cells in two reused clusters. Mean
errors below use the same 72 valid first-500-ms motion targets. The final two
columns use all 464 valid motion targets or all 576 contact targets across the
eight horizons, respectively.

| Original seed-2026091001 model | First 500-ms XY error (mm) | First 500-ms yaw error (rad) | All-horizon XY error (mm) | All-horizon contact Brier |
|---|---:|---:|---:|---:|
| Full direct | 15.617 | 0.04940 | 24.040 | 0.13460 |
| Full supervised rollout | 17.935 | 0.05594 | 26.694 | 0.12291 |
| Full JEPA | 42.606 | 0.12918 | 51.398 | 0.13190 |
| No-RGB direct | 17.478 | 0.03762 | 24.859 | 0.09509 |
| No-RGB supervised rollout | 21.608 | 0.03050 | 35.448 | 0.09343 |
| No-RGB JEPA | 20.661 | 0.04446 | 34.541 | 0.09175 |

For full direct, first-half-second switched-action XY error is 16.432 mm versus
11.545 mm on repeated actions. That pattern is not universal: no-RGB direct has
17.032 mm on switches versus 19.711 mm on repeats. The full-JEPA model is weaker
on these motion targets; this comparison does not demonstrate a general JEPA
or RGB advantage. The augmented fitting design and settings were fixed before
these scores were examined and remain unchanged.

Every geometry-transfer first-half-second contact target is negative; its Brier
score alone cannot assess contact detection. All-horizon contact scoring includes
112 positive targets. These are uncalibrated model scores, and repeated horizons,
suffix siblings and optimization outputs are not independent mazes. No winner
or new checkpoint was selected, and no model was trained in this comparison.

The run took 167.8590913930 seconds after launch, using one CPU thread, and bound
886 source paths. Root is `go2_augmented_family_switch_original_models_v1_attempt_001`
under the owned navigation development artifact base.

| Receipt | SHA-256 |
|---|---|
| `launch.json` | `4306eb08d4c790c2e35dd8693ed4edc9e8d217688007e8db208ff158b54418c7` |
| `prediction_phase_complete.json` | `799008021aa5c668b7e3e101602934922e26788c1f6464514f2b6aaf6b215653` |
| `scores.json` | `12a965f35de8b70f430693309a4e946eec8fdc9c3c24a634902106816ba1071e` |
| `result.json` | `d312fc88fd13c5d0a9bb4543d6bead3e99f68cbf8f068af452a1e090c1da6c96` |

The original fit result remains
`ed3e2f6385991439fd390ffc64e647f6763fb3576b35f7c767fab19e4a29398c`;
the new branch input result is
`2a699ba2a37ce26565324c4e7dbf4f97b957f952c709c470fcf3fb4aaedf2083`.
This is a predictive reference, with no new native navigation, verified arrival,
independent-maze evaluation or hardware-deployment evidence.
