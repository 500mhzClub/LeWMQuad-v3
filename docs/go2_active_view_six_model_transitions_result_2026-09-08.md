# Six-model executed-transition result — 2026-09-08

The severe stationary-hold forecast failure is concentrated in the full-RGB JEPA
fit. Other fits predict the same executed holds substantially better, though
every fitted model overpredicts their tiny actual translations. The evidence
supports a matched native comparison of the existing models before collecting
more training data or changing the scan cost. It does not show that any alternative
model would navigate successfully, or that JEPA generally causes this failure.

The [diagnostic protocol](go2_active_view_six_model_transitions_v1_2026-09-08.md)
reconstructed all 94 recorded causal prediction contexts from the two failed
active-view runs. All six final models received identical past public packets
and original candidate plans, with each model's trained input treatment and head.
Every original full-JEPA forecast reproduced exactly. Native labels were loaded
after inference. Model and input hashes remained unchanged.

Ninety-three half-second intervals completed the predicted selected action: 44
holds, 28 right turns, 14 left turns and seven left arcs. Case 052 tick 228 was
excluded because its final command changed to the sensor-failure zero drain.
All exclusions remain recorded. Only the executed action was scored; the other
five candidates have predictions but no counterfactual outcome claims.

| Model | Case 052 mean XY error (m) | Case 052 mean yaw error (rad) | Case 039 hold mean XY error (m) | Case 039 hold mean yaw error (rad) |
|---|---:|---:|---:|---:|
| RGB direct | 0.009563 | 0.072463 | 0.008844 | 0.023124 |
| RGB supervised rollout | 0.013818 | 0.060876 | 0.015243 | 0.055017 |
| RGB JEPA | 0.039475 | 0.104568 | 0.056343 | 0.218997 |
| No-RGB direct | 0.018017 | 0.041935 | 0.017439 | 0.006756 |
| No-RGB supervised rollout | 0.023141 | 0.031870 | 0.018862 | 0.029674 |
| No-RGB JEPA | 0.020528 | 0.067267 | 0.021183 | 0.006793 |
| Zero-motion numerical reference | 0.016914 | 0.207327 | 0.000273 | 0.001815 |

Case 052 has 45 eligible nonzero-command intervals and no holds. Case 039 has 44
holds and four nonzero intervals. All model yaw predictions were defined. The
reference predicts zero displacement, zero yaw and zero contact; it is not a
navigation controller. Full per-case all/hold/nonzero means, medians, maxima and
contact Brier scores are retained. There were no contacts in these scored
intervals, so contact sensitivity and calibrated risk cannot be established.

At the previously identified case 039 tick 238, measured yaw change was
-0.001831 rad. RGB direct predicted -0.020839, RGB supervised rollout +0.054687,
RGB JEPA -0.219449, no-RGB direct -0.006446, no-RGB supervised rollout +0.030393,
and no-RGB JEPA -0.006433 rad. RGB JEPA's mean hold-yaw error was about 32 times
the no-RGB JEPA error on this trajectory. These are dependent windows selected
by the original RGB-JEPA controller, from one optimization seed and known mirrored
layouts. They cannot establish a general ranking or a causal navigation benefit.

All six fits passed the existing complete ledger, sample schedule, source/input,
snapshot and raw-score admission. Three focused tests passed in 1.61 s. Hardware
had 16 physical/32 logical CPUs, approximately 82 GB available RAM, idle GPUs and
approximately 95 GB free artifact space. Two inference workers took mean 0.056684 s
versus 0.098247 s for one on the matched benchmark (1.73 times throughput), with
bitwise-identical predictions. The selected two-worker full inference took about
0.87–0.90 s per model, with three waves. Recorded diagnostic work after launch
took 19.098 s; this excludes prelaunch full source/model admission. No simulator
or training job ran.

Root under the existing development artifact base:
`go2_active_view_six_model_transitions_v1_attempt_001`.

| Artifact | SHA-256 |
|---|---|
| Launch | `f806c41d767e23185b8591709e82ad4cffecd218ab2a5a6965ec51a355fb621b` |
| Result | `5d3cfc47b9d4a2cd0417b7987cb2b159df6a05a52888bb774185afb328e42657` |
| Transitions and complete predictions | `463eae7505777082525a5f824c7d82c7fc8ec200546f4595734dabbc42aaf527` |
| Grouped scores | `7e5bd80db9f41d483ec5637b048309e1a0bee08507a31865ed036836d2bccdf8` |
| Workload benchmark | `0da42b1a586ec4329a229d0c70d0264b0d34b15e5d6dc44cf4f5130a7fc6d758` |

The next experiment should change only the fitted model/head/input treatment in
the active-view controller, keep the six candidates, map, observer, scan cost,
goal, command commitment and native goal/stop audits fixed, and retain every
fresh case. A measured-heading scan baseline remains useful as a separately
declared controller comparison. Long-horizon visual continuity, observed route
recovery, physical backtracking, independent-maze navigation, realistic timing
and real-platform evidence remain unresolved. The full goal remains active.
