# Body-centric range coverage qualification V1 result

Primary classification: `SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO`.

This was a deterministic, CPU-only, no-training development qualification. True-future clouds use actual transition geometry as an observability upper bound; they do not establish pre-action prediction or deployment safety.

## Bindings and execution

- Source-freeze commit: `6fb55dec810b8fb8337d4519096f17a294c78425`
- Source lineage: `10b3a190d506830e6a87e04a0f1c832b92295bd7`
- Contract SHA-256: `f79eedb072dcfb32853b90881699ccbe3a3066ffd395970a655f980410713980` (content `ad31dfc7c33bb071c18250034d3b7bb1b586fd6ece8242574dafbead27f1faca`)
- Source-closure SHA-256: `14ef33f786bc9f931b6bac214c020d694931c9efaf55bea11f48f131f7862dd5`
- Environment: Python 3.12.3, NumPy 2.4.6, SciPy 1.17.1; CPU-only; Torch/Genesis/JEPA not imported
- Materialisation: 176 states, 29470 transitions, 13385 unique applied-action representatives, 1473500 physics frames, 192 raw-audit artifacts
- Materialisation runtime/storage: 1865.9 s / 2597105664 bytes
- Output storage before result: 6352715776 bytes of 20000000000 byte ceiling on `/dev/mapper/data-root` (ext4)
- Fixture gate: PASS; content digest `340678b7657e7fab7946a00c5076f44cece6db1ecce4a18fabbc52370082ef88`

## Frozen sensor and mounts

The realistic condition is the explicit development assumption `ASSUMED_GO2_HEAD_LIDAR_L2` using `APPROXIMATED_REALISTIC_PLATFORM_SCAN`; it is not a final deployment selection. The approximation uses 6,400 ideal rays per 100 ms, 64 kpoints/s effective rate, 5.55 Hz azimuth, 216 Hz triangular elevation over −6° to +90°, 0.05–30 m range, and planning-boundary-hash phases.

Platform mount (`base`→`radar`): translation `[0.28945, 0.0, -0.046825]` m, RPY `[0.0, 2.8782, 0.0]` rad. Body-centric mount: translation `[0.0, 0.0, 0.067]` m, level RPY `[0.0, 0.0, 0.0]` rad.

Specifications: [Unitree Go2](https://www.unitree.com/go2/), [Unitree L2](https://www.unitree.com/L2/), [L2 manual](https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf).

## Contact metrics and frozen calibration thresholds

| Condition | Mode | Threshold m | Cur AUC/AP | Succ AUC/AP | Combined recall/FNR | Negative retention | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| CURRENT_SPARSE_RANGE_BASELINE | PLANNING_TIME_CAUSAL_CLOUD | 0.147511 | 0.5/0.255952 | 0.5/0.143429 | 1/0 | 0 | n/a (causal diagnostic) |
| CURRENT_SPARSE_RANGE_BASELINE | TRUE_FUTURE_OBSERVABILITY_CLOUD | 0.147511 | 0.5/0.255952 | 0.5/0.143429 | 1/0 | 0 | FAIL |
| REALISTIC_PLATFORM_SCAN | PLANNING_TIME_CAUSAL_CLOUD | 0.00284784 | 0.482233/0.249062 | 0.507171/0.145213 | 0.989796/0.0102041 | 0.015702 | n/a (causal diagnostic) |
| REALISTIC_PLATFORM_SCAN | TRUE_FUTURE_OBSERVABILITY_CLOUD | 0.496014 | 0.5/0.255952 | 0.50934/0.145761 | 1/0 | 0 | FAIL |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | PLANNING_TIME_CAUSAL_CLOUD | 0.00274718 | 0.482233/0.249062 | 0.509506/0.145803 | 1/0 | 0.0200123 | n/a (causal diagnostic) |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | TRUE_FUTURE_OBSERVABILITY_CLOUD | 0.155083 | 0.5/0.255952 | 0.511674/0.146356 | 1/0 | 0 | FAIL |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | PLANNING_TIME_CAUSAL_CLOUD | 0.493773 | 0.5/0.255952 | 0.5/0.143429 | 1/0 | 0 | n/a (causal diagnostic) |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | TRUE_FUTURE_OBSERVABILITY_CLOUD | 0.509875 | 0.5/0.255952 | 0.5/0.143429 | 1/0 | 0 | FAIL |

## Safe-count and two-ply viability

| Condition | Mode | Count MAE/Spearman/exact/zero-nonzero | False zero/nonzero | Viable retained | Nonviable abstain | Unsafe selected (now/next) | Progress fraction/regret | Top-1/Top-3 | Margins ≥1/2/3 |
|---|---|---|---|---:|---:|---|---|---|---|
| CURRENT_SPARSE_RANGE_BASELINE | PLANNING_TIME_CAUSAL_CLOUD | 6.02304/0/0.327189/0.327189 | 1/0 | 0/20 | 4/4 | 0/0 | 0/1 | 0/0 | 0/0/0 |
| CURRENT_SPARSE_RANGE_BASELINE | TRUE_FUTURE_OBSERVABILITY_CLOUD | 6.02304/0/0.327189/0.327189 | 1/0 | 0/20 | 4/4 | 0/0 | 0/1 | 0/0 | 0/0/0 |
| REALISTIC_PLATFORM_SCAN | PLANNING_TIME_CAUSAL_CLOUD | 5.94009/0.0427017/0.336406/0.341014 | 0.979452/0 | 1/20 | 4/4 | 0/0 | 0.0468917/0.953108 | 0.05/0.05 | 1/1/1 |
| REALISTIC_PLATFORM_SCAN | TRUE_FUTURE_OBSERVABILITY_CLOUD | 6.02304/0/0.327189/0.327189 | 1/0 | 0/20 | 4/4 | 0/0 | 0/1 | 0/0 | 0/0/0 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | PLANNING_TIME_CAUSAL_CLOUD | 5.89401/0.0730049/0.341014/0.345622 | 0.972603/0 | 1/20 | 4/4 | 0/0 | 0.0468917/0.953108 | 0.05/0.05 | 1/1/1 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | TRUE_FUTURE_OBSERVABILITY_CLOUD | 6.02304/0/0.327189/0.327189 | 1/0 | 0/20 | 4/4 | 0/0 | 0/1 | 0/0 | 0/0/0 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | PLANNING_TIME_CAUSAL_CLOUD | 6.02304/0/0.327189/0.327189 | 1/0 | 0/20 | 4/4 | 0/0 | 0/1 | 0/0 | 0/0/0 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | TRUE_FUTURE_OBSERVABILITY_CLOUD | 6.02304/0/0.327189/0.327189 | 1/0 | 0/20 | 4/4 | 0/0 | 0/1 | 0/0 | 0/0/0 |

## True-future per-family coverage

| Condition | Family | Contact AUC/recall | Support | Mean unsupported | Viable retained | Nonviable abstain |
|---|---|---|---:|---:|---:|---:|
| CURRENT_SPARSE_RANGE_BASELINE | large_enclosed_maze | 0.5/1 | 0.0280248 | 0.971975 | 0/6 | 0/0 |
| CURRENT_SPARSE_RANGE_BASELINE | loop_alias_stress | 0.5/1 | 0.113518 | 0.886482 | 0/5 | 1/1 |
| CURRENT_SPARSE_RANGE_BASELINE | medium_enclosed_maze | 0.5/1 | 0.0332302 | 0.96677 | 0/5 | 1/1 |
| CURRENT_SPARSE_RANGE_BASELINE | small_enclosed_maze | 0.5/1 | 0.055792 | 0.944208 | 0/4 | 2/2 |
| REALISTIC_PLATFORM_SCAN | large_enclosed_maze | 0.5/1 | 0.443766 | 0.556234 | 0/6 | 0/0 |
| REALISTIC_PLATFORM_SCAN | loop_alias_stress | 0.538251/1 | 0.356033 | 0.643967 | 0/5 | 1/1 |
| REALISTIC_PLATFORM_SCAN | medium_enclosed_maze | 0.5/1 | 0.385103 | 0.614897 | 0/5 | 1/1 |
| REALISTIC_PLATFORM_SCAN | small_enclosed_maze | 0.5/1 | 0.436936 | 0.563064 | 0/4 | 2/2 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | large_enclosed_maze | 0.5/1 | 0.57784 | 0.42216 | 0/6 | 0/0 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | loop_alias_stress | 0.547814/1 | 0.58185 | 0.41815 | 0/5 | 1/1 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | medium_enclosed_maze | 0.5/1 | 0.553767 | 0.446233 | 0/5 | 1/1 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | small_enclosed_maze | 0.5/1 | 0.604032 | 0.395968 | 0/4 | 2/2 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | large_enclosed_maze | 0.5/1 | 0.126613 | 0.873387 | 0/6 | 0/0 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | loop_alias_stress | 0.5/1 | 0.195222 | 0.804778 | 0/5 | 1/1 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | medium_enclosed_maze | 0.5/1 | 0.142246 | 0.857754 | 0/5 | 1/1 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | small_enclosed_maze | 0.5/1 | 0.237402 | 0.762598 | 0/4 | 2/2 |

## True-future per-link coverage

| Condition | Link | Region | Contact AUC/recall | Support | Direct visibility | Mean unsupported | Unresolved oracle rows |
|---|---|---|---|---:|---:|---:|---:|
| CURRENT_SPARSE_RANGE_BASELINE | FL_calf | calf | n/a/n/a | 0 | 0 | 1 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | FL_hip | front_limb | 0.53329/1 | 0.11731 | 0 | 0.88269 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | FL_thigh | front_limb | n/a/n/a | 0.0544108 | 0 | 0.945589 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | FR_calf | calf | 0.5/1 | 0 | 0 | 1 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | FR_hip | front_limb | 0.539368/1 | 0.114411 | 6.25652e-05 | 0.885589 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | FR_thigh | front_limb | n/a/n/a | 0 | 0 | 1 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | RL_calf | calf | 0.5/1 | 0 | 0 | 1 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | RL_hip | rear_limb | n/a/n/a | 0.177596 | 0 | 0.822404 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | RL_thigh | rear_limb | n/a/n/a | 0 | 0 | 1 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | RR_calf | calf | 0.5/1 | 0 | 0 | 1 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | RR_hip | rear_limb | n/a/n/a | 0.0786444 | 0 | 0.921356 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | RR_thigh | rear_limb | 0.5/1 | 0 | 0 | 1 | 9 |
| CURRENT_SPARSE_RANGE_BASELINE | base | trunk | 0.521935/1 | 0.168618 | 0 | 0.831382 | 9 |
| REALISTIC_PLATFORM_SCAN | FL_calf | calf | n/a/n/a | 0.639114 | 0.933415 | 0.360886 | 9 |
| REALISTIC_PLATFORM_SCAN | FL_hip | front_limb | 0.766055/1 | 0.560209 | 0.307492 | 0.439791 | 9 |
| REALISTIC_PLATFORM_SCAN | FL_thigh | front_limb | n/a/n/a | 0.614515 | 0.742258 | 0.385485 | 9 |
| REALISTIC_PLATFORM_SCAN | FR_calf | calf | 0.737796/1 | 0.570688 | 0.914927 | 0.429312 | 9 |
| REALISTIC_PLATFORM_SCAN | FR_hip | front_limb | 0.595602/1 | 0.515109 | 0.125865 | 0.484891 | 9 |
| REALISTIC_PLATFORM_SCAN | FR_thigh | front_limb | n/a/n/a | 0.543019 | 0.636184 | 0.456981 | 9 |
| REALISTIC_PLATFORM_SCAN | RL_calf | calf | 0.683171/1 | 0.410443 | 0.562284 | 0.589557 | 9 |
| REALISTIC_PLATFORM_SCAN | RL_hip | rear_limb | n/a/n/a | 0.0383629 | 0 | 0.961637 | 9 |
| REALISTIC_PLATFORM_SCAN | RL_thigh | rear_limb | n/a/n/a | 0.225808 | 0.0378154 | 0.774192 | 9 |
| REALISTIC_PLATFORM_SCAN | RR_calf | calf | 0.625543/1 | 0.364645 | 0.433285 | 0.635355 | 9 |
| REALISTIC_PLATFORM_SCAN | RR_hip | rear_limb | n/a/n/a | 0.0667362 | 0.00135558 | 0.933264 | 9 |
| REALISTIC_PLATFORM_SCAN | RR_thigh | rear_limb | 0.575924/1 | 0.193525 | 0.0876747 | 0.806475 | 9 |
| REALISTIC_PLATFORM_SCAN | base | trunk | 0.480287/1 | 0.538384 | 0.498243 | 0.461616 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | FL_calf | calf | n/a/n/a | 0.965474 | 0.965474 | 0.0345255 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | FL_hip | front_limb | 0.818349/1 | 0.697174 | 0.697174 | 0.302826 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | FL_thigh | front_limb | n/a/n/a | 0.877252 | 0.877252 | 0.122748 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | FR_calf | calf | 0.950328/1 | 0.95878 | 0.95878 | 0.04122 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | FR_hip | front_limb | 0.637087/1 | 0.567351 | 0.567351 | 0.432649 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | FR_thigh | front_limb | n/a/n/a | 0.801512 | 0.801512 | 0.198488 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | RL_calf | calf | 0.820385/1 | 0.724572 | 0.724572 | 0.275428 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | RL_hip | rear_limb | n/a/n/a | 0.0383629 | 0.0383629 | 0.961637 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | RL_thigh | rear_limb | n/a/n/a | 0.243869 | 0.243869 | 0.756131 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | RR_calf | calf | 0.740316/1 | 0.636663 | 0.636663 | 0.363337 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | RR_hip | rear_limb | n/a/n/a | 0.0675495 | 0.0675495 | 0.93245 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | RR_thigh | rear_limb | 0.595201/1 | 0.242388 | 0.242388 | 0.757612 | 9 |
| DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT | base | trunk | 0.577112/1 | 0.68804 | 0.68804 | 0.31196 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | FL_calf | calf | n/a/n/a | 0 | 0 | 1 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | FL_hip | front_limb | 0.634993/1 | 0.295855 | 0.295855 | 0.704145 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | FL_thigh | front_limb | n/a/n/a | 0.119093 | 0.119093 | 0.880907 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | FR_calf | calf | 0.5/1 | 0 | 0 | 1 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | FR_hip | front_limb | 0.684119/1 | 0.39792 | 0.39792 | 0.60208 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | FR_thigh | front_limb | n/a/n/a | 0.0144734 | 0.0144734 | 0.985527 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | RL_calf | calf | 0.5/1 | 0 | 0 | 1 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | RL_hip | rear_limb | n/a/n/a | 0.435209 | 0.435209 | 0.564791 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | RL_thigh | rear_limb | n/a/n/a | 0.123968 | 0.123968 | 0.876032 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | RR_calf | calf | 0.5/1 | 0 | 0 | 1 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | RR_hip | rear_limb | n/a/n/a | 0.447503 | 0.447503 | 0.552497 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | RR_thigh | rear_limb | 0.537241/1 | 0.138858 | 0.138858 | 0.861142 | 9 |
| DENSE_BODY_CENTRIC_SINGLE_ORIGIN | base | trunk | 0.55/1 | 0.236152 | 0.236152 | 0.763848 | 9 |

## Coverage errors, regression, and compute

Coverage-error rows: 25874; aggregate classes: `{"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 11027, "SCAN_PATTERN_SPARSITY": 243, "SCAN_TIMING_LIMITATION": 184, "SINGLE_ORIGIN_LIMITATION": 70, "UNRESOLVED": 58, "VERTICAL_COVERAGE_LIMITATION": 14292}`.

Per-condition/mode attribution: `{"CURRENT_SPARSE_RANGE_BASELINE": {"PLANNING_TIME_CAUSAL_CLOUD": {"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 0, "SCAN_PATTERN_SPARSITY": 0, "SCAN_TIMING_LIMITATION": 0, "SINGLE_ORIGIN_LIMITATION": 0, "UNRESOLVED": 0, "VERTICAL_COVERAGE_LIMITATION": 3248}, "TRUE_FUTURE_OBSERVABILITY_CLOUD": {"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 0, "SCAN_PATTERN_SPARSITY": 0, "SCAN_TIMING_LIMITATION": 0, "SINGLE_ORIGIN_LIMITATION": 0, "UNRESOLVED": 0, "VERTICAL_COVERAGE_LIMITATION": 3248}}, "DENSE_BODY_CENTRIC_SINGLE_ORIGIN": {"PLANNING_TIME_CAUSAL_CLOUD": {"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 3248, "SCAN_PATTERN_SPARSITY": 0, "SCAN_TIMING_LIMITATION": 0, "SINGLE_ORIGIN_LIMITATION": 0, "UNRESOLVED": 0, "VERTICAL_COVERAGE_LIMITATION": 0}, "TRUE_FUTURE_OBSERVABILITY_CLOUD": {"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 3248, "SCAN_PATTERN_SPARSITY": 0, "SCAN_TIMING_LIMITATION": 0, "SINGLE_ORIGIN_LIMITATION": 0, "UNRESOLVED": 0, "VERTICAL_COVERAGE_LIMITATION": 0}}, "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT": {"PLANNING_TIME_CAUSAL_CLOUD": {"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 42, "SCAN_PATTERN_SPARSITY": 0, "SCAN_TIMING_LIMITATION": 0, "SINGLE_ORIGIN_LIMITATION": 0, "UNRESOLVED": 0, "VERTICAL_COVERAGE_LIMITATION": 3141}, "TRUE_FUTURE_OBSERVABILITY_CLOUD": {"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 22, "SCAN_PATTERN_SPARSITY": 0, "SCAN_TIMING_LIMITATION": 0, "SINGLE_ORIGIN_LIMITATION": 70, "UNRESOLVED": 0, "VERTICAL_COVERAGE_LIMITATION": 3156}}, "REALISTIC_PLATFORM_SCAN": {"PLANNING_TIME_CAUSAL_CLOUD": {"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 2409, "SCAN_PATTERN_SPARSITY": 126, "SCAN_TIMING_LIMITATION": 103, "SINGLE_ORIGIN_LIMITATION": 0, "UNRESOLVED": 6, "VERTICAL_COVERAGE_LIMITATION": 559}, "TRUE_FUTURE_OBSERVABILITY_CLOUD": {"HORIZONTAL_COVERAGE_LIMITATION": 0, "NEAR_BLIND_REGION": 0, "PLATFORM_MOUNT_LIMITATION": 0, "POINT_ACCUMULATION_ERROR": 0, "ROBOT_SELF_OCCLUSION": 2058, "SCAN_PATTERN_SPARSITY": 117, "SCAN_TIMING_LIMITATION": 81, "SINGLE_ORIGIN_LIMITATION": 0, "UNRESOLVED": 52, "VERTICAL_COVERAGE_LIMITATION": 940}}}`.

Legacy sparse-regression summary: `{"both_finite_rows": 29470, "current_nonfinite_rows": 0, "exact_within_1e_6_fraction": 0.0010858500169664065, "legacy_nonfinite_rows": 0, "maximum_absolute_clearance_difference_m": 0.15894758328795433, "mean_absolute_clearance_difference_m": 0.024211601454550973, "p95_absolute_clearance_difference_m": 0.07808118965476747, "rows": 29470}`.

Complete-heldout reduction benchmark: P50/P90/P95/P99/max = 54.0456/54.8479/55.548/98.2485/102.205 ms; misses at 50/80/100 ms = 1000/29/5; peak RSS 496095232 bytes; peak VRAM 0 bytes.

Scan materialisation totals: `{"CURRENT_SPARSE_RANGE_BASELINE": {"PLANNING_TIME_CAUSAL_CLOUD": {"environment_returns": 7740310, "ground_returns": 367177, "near_blind_first_hits": 0, "rays": 9780480, "representative_scans": 13584, "robot_self_returns": 0, "unique_rendered_rays": 1019520, "unique_rendered_scans": 1416}, "TRUE_FUTURE_OBSERVABILITY_CLOUD": {"environment_returns": 15483138, "ground_returns": 735432, "near_blind_first_hits": 0, "rays": 19560960, "representative_scans": 13584, "robot_self_returns": 0, "unique_rendered_rays": 19560960, "unique_rendered_scans": 13584}}, "REALISTIC_PLATFORM_SCAN": {"PLANNING_TIME_CAUSAL_CLOUD": {"environment_returns": 28009951, "ground_returns": 55699595, "near_blind_first_hits": 0, "rays": 86937600, "representative_scans": 13584, "robot_self_returns": 2464069, "unique_rendered_rays": 9062400, "unique_rendered_scans": 1416}, "TRUE_FUTURE_OBSERVABILITY_CLOUD": {"environment_returns": 27339684, "ground_returns": 56349776, "near_blind_first_hits": 1154, "rays": 86937600, "representative_scans": 13584, "robot_self_returns": 2502483, "unique_rendered_rays": 86937600, "unique_rendered_scans": 13584}}}`.

## Classification and custody

Secondary classifications: `FOUR_CHANNEL_RANGE_BASELINE_INSUFFICIENT`, `FRONT_LIMB_VISIBILITY_FAILURE`, `REAR_LIMB_VISIBILITY_FAILURE`, `CALF_VISIBILITY_FAILURE`, `TRUNK_VISIBILITY_FAILURE`, `ROBOT_BODY_SELF_OCCLUSION`, `ASSUMED_SENSOR_CONTRACT`.

Next decision: `SPECIFY_MULTIPLE_SENSOR_ORIGINS_OR_NARROW_PROTECTED_CONTACT_SCOPE`. No training is authorized.

No model was trained; no fresh panel, G2 evaluation, JEPA predictor, checkpoint, memory, navigation, routing, or beacon system was opened or executed. Superseded occupancy and recurrent-memory experiments were not restarted.

Machine-readable authority: the external `result.json`, calibration receipts, transition ledger, per-link ledger, coverage-error ledger, materialisation index, raw-audit manifest, and persistence receipt.
