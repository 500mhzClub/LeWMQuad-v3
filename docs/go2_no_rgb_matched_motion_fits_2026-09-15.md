# Matched corrections for the existing no-RGB models

The recent complete navigation cohorts do not establish a learned-model or
JEPA advantage over fitted motion. Before attributing any future difference
to visual learning, prepare the existing no-RGB neural controls for the same
current navigation interface. This is not another neural training sweep.

Fit corrections for all nine already trained no-RGB models: training seeds
2026091001, 2026091401 and 2026091402, each with JEPA, direct and supervised
rollout. Reuse the full-input fitter's four training recordings, validation
recording, 1,207 training windows, 448 validation windows, per-horizon 42-feature
ridge regression, penalty 1 and stationary thinning. Require exact original
target, validity-mask and window/group equality. Freeze each correction before
reading its validation data. Do not select a seed or method using these results.

Only the neural model's RGB input is removed. Camera-based pose estimation,
causal pose history, command inputs and the geometric navigation system remain.
Thus subsequent comparisons can address the neural RGB contribution, not
whether the entire robot can navigate without vision. Corrections depend on
their corresponding frozen model and are part of the compared pipeline.
The original no-RGB JEPA training also removed RGB from its future-observation
targets. This compares the existing matched training input/target treatments;
it is not inference-only image removal from the same trained weights.

After the reserved-terminal native pair finishes, run three CPU workers, one
per training method, with each worker processing the three seeds in order.
Each model inference and small linear solve uses one CPU thread. These fits
reuse compact RGB/body/pose/command artifacts; no depth generation or new
simulation is required. Preserve all results, including worse validation
scores. Compare full/no-RGB errors on identical windows before and after
correction. These overlapping windows from one validation recording cannot
establish navigation benefit or independent-maze generalization.

Prospective native RGB ablations remain necessary. Choose their fresh maze
cohort and complete fixed assignment list before observing its outcomes; keep
the shared perception and predictive controller fixed across input treatments.

## Completed results

All nine corrections finished in the three parallel workers, each worker
exiting 0. Every fit matched the original 1,207 training and 448 validation
windows, targets, validity masks and groups. Neural weights remained unchanged;
every actual model call was checked for zero RGB input. All full/no-RGB training
and validation arrays were compared directly when assembling the result.
No native state entered these fits. Validation targets are registered visual
pose changes, not independent physical truth.

The 700-ms moving-command comparison uses the same 119 overlapping validation
windows for every model. XY RMSE in millimetres:

| Seed | Method | Full before correction | No-RGB before correction | Full corrected | No-RGB corrected |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026091001 | JEPA | 20.855 | 14.886 | 8.033 | 7.934 |
| 2026091001 | Direct | 22.987 | 26.401 | 8.467 | 8.180 |
| 2026091001 | Supervised rollout | 18.621 | 22.020 | 7.962 | 7.861 |
| 2026091401 | JEPA | 19.927 | 30.021 | 8.162 | 8.237 |
| 2026091401 | Direct | 24.523 | 19.076 | 8.377 | 8.282 |
| 2026091401 | Supervised rollout | 15.896 | 14.280 | 8.218 | 8.335 |
| 2026091402 | JEPA | 22.277 | 15.799 | 8.314 | 8.006 |
| 2026091402 | Direct | 21.983 | 21.195 | 8.337 | 8.122 |
| 2026091402 | Supervised rollout | 18.641 | 17.863 | 8.257 | 7.896 |

Across the three fixed seeds, corrected full/no-RGB means are 8.170/8.059 mm
for JEPA, 8.394/8.194 mm for direct and 8.146/8.031 mm for supervised rollout.
No-RGB has lower corrected error in seven of nine paired cells. These small
differences establish neither superiority nor equivalence. The raw neural
differences vary substantially across seeds, while the shared correction
procedure brings the pipelines close together. This population gives no
positive evidence that neural RGB improves corrected XY prediction.

The descriptive result includes every group and scored horizon:
`go2_no_rgb_matched_motion_comparison_v1_attempt_001/result.json`.
The separate nine-model registry is
`docs/go2_no_rgb_navigation_models_2026-09-15.json`, SHA-256
`2fc9d21c99cba417ca45be2caf99f6b85ccd724ca71ad69d3b88057d10b1cda6`.
The earlier nine full-input model registry remains unchanged.

Next, wire these frozen no-RGB assignments into the current shared predictive
runtime and test complete matched full/no-RGB navigation on a fixed fresh
development cohort. The required change is model/input/correction assignment,
not another perception or recovery variant. Include fresh full-input runs in
that comparison. Retain all assigned seeds and methods; do not choose the
best validation cell. These fits supply no new navigation evidence, yaw/contact
accuracy result, independent-maze reliability estimate or hardware validation.
