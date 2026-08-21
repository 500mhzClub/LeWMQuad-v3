# WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_V1

Date: 2026-08-21
Source commit: `58d91deb37d41a129e64a6a0c17ae8b9b6f135d2`
Final classification: **`WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY`**

## Claim boundary

The target is `SIMULATED_DISALLOWED_CONTACT_PROXY`: any robot–environment contact included by the unchanged frozen binary definition. Ordinary expected foot–ground support and robot self-contact remain excluded. This is a simulated operational separation/contact-avoidance proxy only. It does not establish prevention of injury or property damage, acceptable impact, material-contact safety, or safety around people or fragile infrastructure.

The preserved ontology terminal remains `CONTACT_HAZARD_ONTOLOGY_OR_INSTRUMENTATION_INSUFFICIENT`. Stuck is reported only as recoverability/task-performance evidence and never rejects a candidate in this experiment.

## Stage 0: frozen-score composition

The frozen LiDAR and depth-plus-embodied ledgers reproduced their committed branch metrics and selected candidates. Contact counts and decisions were exact; route aggregates reproduced within `2e-7` because the historical ledger stored route labels as FP32.

The continuous composition diagnostic is the probability union `1-(1-p_lidar)(1-p_fusion)`. Admission is separate: a branch is admitted only below both component thresholds.

| Quantity | Calibration-selected result |
|---|---:|
| LiDAR threshold | 0.1919558942 |
| Fusion threshold | 0.8373786211 |
| Calibration recall / FNR | 0.9926 / 0.0074 |
| Calibration negative retention | 0.2829 |
| Calibration states retaining | 10/24 |
| Held-out union AUC / AP | 0.9119 / 0.9078 |
| Held-out recall / FNR | 0.9756 / 0.0244 |
| Held-out negative retention | 0.2485 |
| Held-out states retaining | 11/24 |
| Held-out selected contacts | 1 |
| Held-out false abstentions | 12 |
| Held-out oracle-progress fraction | 0.6478 |
| Held-out normalized regret / top-3 | 0.2617 / 0.1667 |

The calibration and held-out frontiers each contain 84,100 threshold pairs (`290×290`). The post-hoc held-out frontier has zero complete-gate pairs. Stage 0 therefore classified `WIDE_GEOMETRY_SCORE_FRONTIER_NO_GO`, authorising Stage 1 without adopting any held-out threshold.

## Data bindings

The fit corpus contains 240 distinct observed states and 2,880 branches: the original 48 purpose-built states, the former factorised-safety calibration/held-out 48, and all 144 scaling states. Their existing geometry, enhanced embodied, action/control, and contact evidence became development fit data.

The new panel contains exactly 48 prospectively frozen scenes and clusters:

- calibration identities: `wide-cal-0-00` through `wide-cal-3-05`, six per family;
- held-out identities: `wide-held-0-00` through `wide-held-3-05`, six per family;
- 24 calibration and 24 held-out states, 12 candidates each, 576 new branches;
- zero scene overlap with the fit-240 corpus or predictor scenes;
- panel digest: `2466d7e36bcd46cd06e4e08fad3e89c0242dd45406a75c53acb6e9cd61f60be4`.

| Split | Contact positive / negative | Stuck positive | Contact–stuck overlap | States with a negative candidate |
|---|---:|---:|---:|---:|
| Fit-240 | 1,252 / 1,628 | 1,542 | 811 | 234/240 |
| Calibration | 91 / 197 | 149 | 62 | 24/24 |
| Held-out | 121 / 167 | 150 | 76 | 22/24 |

All 576 new branches passed finite-channel, candidate/action, pose, contact, stuck, and aggregate verification. No branch or state was replaced.

## Sensor and model contract

- Front depth: ideal metric `64×48`, RGB-camera pose/FOV, `[0.05,10] m`, validity mask.
- LiDAR: ideal 360°, 180 azimuth bins, four vertical channels at `[-15,-5,5,15]°`, `[0.05,10] m`, validity mask. This is `CHANGED_DEPLOYMENT_SENSOR_CONTRACT` unless an equivalent physical sensor is deployed.
- Enhanced embodied state: the unchanged 73-channel acceleration, gyro, gravity, joint/motor, calf-force, previous-action, applied-command, candidate-action, and control-history contract.

The model has independent depth, LiDAR, and embodied encoders, a one-layer causal GRU of width 160, and per-tick active/cumulative contact outputs. It has **415,282 trainable parameters**. Seed `2026082013` was used once. Final checkpoint SHA-256: `3e556531a0442df214d0667ad42110e42806ec3aa7aa240c2b2746d7c304af31`.

Balanced active/cumulative BCE and the fixed `0.25` H3 within-state ranking loss were used for 60 final-epoch-only epochs. Total loss changed from 1.186064 at epoch 1 to 0.001494 at epoch 60. A post-training row-persistence field-name fault occurred after the checkpoint was saved; it was repaired without retraining, and evaluation resumed from the frozen checkpoint.

## Calibration and held-out result

Fresh-calibration temperature: `3.7706889163`

Fresh-calibration threshold: `0.0692083538`

| Branch/event metric | Held-out |
|---|---:|
| Contact AUC | 0.9315 |
| Average precision | 0.9348 |
| Recall / FNR | 0.9421 / 0.0579 |
| Contact-negative retention | 0.5928 (99/167) |
| Admitted contact-positive | 7/121 |
| ECE / Brier | 0.0559 / 0.0929 |
| Event-tick recall | 0.9061 |
| Branch event-detection recall | 0.9504 |
| Median delay | 0 ticks |
| Missed transient-contact rate | 0.2143 |

| Filtering/planning metric | Held-out |
|---|---:|
| States retaining a negative candidate | 21/24 |
| States admitting only positive candidates | 0 |
| Correct / false abstentions | 2 / 1 |
| Selected contacts | 1 |
| Selected stuck | 8 |
| Selected progress | 0.0993 m |
| Oracle-contact progress | 0.1447 m |
| Oracle-progress fraction | 0.6862 |
| Normalized regret | 0.3474 |
| Best-negative top-1 / top-3 | 0.2273 / 0.5000 |

The condition passed AUC, AP, negative retention, ECE, event recall, detection delay, state availability, no-positive-only-state, and abstention gates. It failed recall/FNR, zero selected contact, oracle-progress fraction, regret, top-3, and no-family-collapse gates.

## Per-family held-out result

| Family | AUC | Recall | Negative retention | States retaining | Selected contacts | Progress (m) | Oracle fraction | Regret | Top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Large enclosed | 0.9583 | 0.9231 | 0.6949 | 6/6 | 0 | 0.2661 | 0.8710 | 0.2314 | 0.8333 |
| Medium enclosed | 0.9472 | 0.9744 | 0.3939 | 6/6 | 1 | -0.0315 | -0.5599 | 0.7118 | 0.3333 |
| Small enclosed | 0.9607 | 0.9756 | 0.7097 | 4/6 | 0 | 0.1930 | 0.7792 | 0.2629 | 0.2500 |
| Loop alias stress | 0.8636 | 0.8571 | 0.5227 | 5/6 | 0 | -0.0187 | -4.9362 | 0.1896 | 0.5000 |

The medium selected contact and loop-alias discrimination/progress failure prevent a signal classification. Contact-free behavior alone would not establish mission safety: progress monitoring, stuck detection, recovery, replanning, coverage, and completion remain separate future requirements.

## Historical comparator context

The first three comparators use the prior geometry held-out panel and are not paired with this fresh panel.

| Condition | Contact AUC | Recall | Negative retention | Selected contacts | Route progress (m) |
|---|---:|---:|---:|---:|---:|
| Depth only | 0.7360 | 0.8537 | 0.4242 | 3 | 0.1603 |
| LiDAR only | 0.7148 | 0.9431 | 0.2424 | 1 | 0.1333 |
| Depth + embodied | 0.8927 | 0.9106 | 0.5636 | 2 | 0.1676 |
| Wide geometry + embodied | **0.9315** | 0.9421 | **0.5928** | 1 | 0.0993 |
| FIT-192 contact specialist | 0.8487 | 0.9187 | — | — | — |
| Privileged static-grid guard (historical aggregate-unsafe target) | — | 0.6724 | 0.5789 | — | — |
| Oracle contact, fresh panel | 1.0000 | 1.0000 | 1.0000 | 0 | 0.1447 |

The static-grid guard values are its frozen aggregate-unsafe recall and safe-candidate retention, not paired contact-only metrics. It remains a historical privileged comparator and is not evidence of learned contact-proxy sufficiency.

## Decision, evidence, and runtime

The discrimination and retention improvements constitute `WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY`, but no next model is automatically authorised. `CANDIDATE_CONDITIONED_WIDE_GEOMETRY_CONTACT_PROXY_PREDICTOR_V1` would be the reserved next experiment only after an explicit decision; this pass did not open it.

- Branch collection: 1,589.65 aggregate compute seconds; 410.49 seconds wall time.
- Static geometry materialisation: 12.37 compute seconds; zero simulator steps.
- Training plus smoke upper bound from geometry-index completion to checkpoint: 654.58 seconds. Exact isolated training time was not recoverable after the post-training serializer fault.
- New enhanced sensor shards: 2,300,471 bytes.
- New geometry shards: 25,974,647 bytes.
- Checkpoint: 1,678,995 bytes.
- Generated result tree: approximately 6.8 MiB; high-capacity cache: approximately 34 MiB.
- Row-level ledger: 3,456 rows, SHA-256 `ab47eb7848b980947ced6ee6f10493ef12578ab7871ef8ebdb97b46122617e9c`, content digest `055d9c5581e3980919fbba6ec17aae288759557691891028691fd6f23eae83be`.

Exactly one new model seed was trained. No JEPA predictor, RGB distillation, memory, novelty, routing, beacon, or navigation model was opened or trained.
