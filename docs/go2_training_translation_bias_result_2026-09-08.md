# Training-only translation correction result

All eighteen frozen models received the prescribed closed-form correction for
every trained head: thirty heads and 480 scalar coefficients. All coefficients
were frozen before corrected transfer evaluation and reconstructed exactly in
the separate readout. Original neural weights received zero updates.

For each XY coordinate and horizon, the estimator used only valid training
motion residuals, weighted by the context's exact original draw count divided
by its number of valid motion horizons. Float64 means were frozen as float32
inference coefficients. Native trajectories and transfer residuals were not
used to fit them. This is an output-intercept fit, not probability calibration
or a model-error bound.

All 36 primary model/source mean position errors improved at 100 ms on training
data and all 36 improved on geometry-transfer data. These are eighteen models
times two source families, not independent maze replications. Complete results
for every head, role, stratum and horizon remain in the readout.

| Treatment | Family-transfer 100-ms error before → after, mm | Switch-transfer 100-ms error before → after, mm |
| --- | ---: | ---: |
| Full direct | 21.405 → 7.871 | 20.055 → 8.588 |
| Full supervised rollout | 18.169 → 7.863 | 17.738 → 7.958 |
| Full JEPA | 31.013 → 14.868 | 32.834 → 14.866 |
| No-RGB direct | 19.016 → 9.317 | 17.929 → 8.536 |
| No-RGB supervised rollout | 19.818 → 7.616 | 20.213 → 8.550 |
| No-RGB JEPA | 26.379 → 17.011 | 24.705 → 17.022 |

Values are descriptive means across the three optimization seeds. After
correction, the switch-transfer standard deviations are respectively 1.642,
1.061, 6.246, 1.950, 0.639 and 6.138 mm in the table's order. There is still no
general JEPA advantage; no-RGB direct remains comparable to full direct. This
readout does not establish that vision or learned planning helps a maze mission.

The fixed seed-2026091001 full-direct 100-ms applied XY bias is
[0.0031142050866037607, 0.012625064700841904] m. Its family/switch transfer errors
changed from 17.798/17.891 to 9.943/10.442 mm. Full-JEPA's corresponding applied
bias is [-0.017337942495942116, 0.022331422194838524] m, and errors changed from
29.849/29.838 to 9.197/7.749 mm. These assignments were fixed before fitting;
the better first-seed corrected JEPA result does not select a new winner.

Across all eight switch-transfer horizons, three-seed full-direct mean error
changed from 22.592 to 11.341 mm; full-JEPA changed from 35.639 to 15.614 mm.
Their 500-ms errors changed from 23.601 to 12.266 mm and 36.020 to 15.313 mm.
Raw yaw/contact components, scores, clocks, masks, target denominators and
undefined-yaw accounting remained exactly unchanged for every evaluated head.
Both corrected roles were saved before either was scored for each model.

The coefficient phase bound 1,139 source paths and took 10.982156 seconds after
full-model admission/launch. Preflight measured 82,563,440,640 bytes available
RAM and 69,141,897,216 bytes free artifact storage. The readout bound 1,142 source
paths and 92 artifacts, taking 15.864053 seconds after its admission/launch;
available RAM/storage were 82,324,307,968/69,124,554,752 bytes. Both used one CPU
process and one numerical thread. These phase timings exclude earlier admission.

Artifact roots:
`go2_training_translation_bias_v1_attempt_001` and
`go2_training_translation_bias_readout_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| Original fit result | `45b4680b85c87bd69dcaed6a0058f091105632909661dba319d6c05f5b533418` |
| Correction launch | `4f9ed2efa03281eb3eb89e64641467c446e48fe46b0fceb8a23821e4481650d4` |
| Coefficients | `4a5b9f3e0660f62619682076362489db7a5cf4023e8dd573f1d506a1f4d7414a` |
| Coefficient completion receipt | `6800a6d67babd2f4b59e6a68d84907ebf9287e858a56b50cfbd070bc19cb88f9` |
| Correction result | `a425d3ab1398df9312663e35dee665e5800153bd60d29334ae71a1d95e4f21d5` |
| Readout launch | `36bb6bebb00e3bf294c001368e4111676f48a74e5f624bf9fdfe321761d30f33` |
| Readout metrics | `14d46d976449fe9b831e19307d0ecf4f98e38e4029359a51d05d45fa98c7eede` |
| Readout result | `79753350e317eb096cdfe676d2fc9804a4e1f4c2af8c936241ff75229f869bcf` |

This result includes no corrected native execution. There remain zero verified
arrivals, no independent novel-maze demonstration and no real-time or hardware
qualification. The context-phase coverage gap remains. Matched reactive and
nonpredictive baselines, planning/memory contribution tests, physical backtracking
and independent-layout evidence are still required.
