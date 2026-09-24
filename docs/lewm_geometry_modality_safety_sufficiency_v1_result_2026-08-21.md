# GEOMETRY_MODALITY_SAFETY_SUFFICIENCY_V1

Date: 2026-08-21  
Source commit: `20dd1b8dbdd52db5f3b55217ed2f6601ec4ec4c0`  
Preserved terminal: `FACTORISED_MICRO_SAFETY_DATA_SCALING_NO_SIGNAL`  
Final classification: **`GEOMETRY_MODALITY_POSITIVE_TENDENCY`**

## Result

None of the three prospectively fixed geometry conditions passed the complete hard-contact safety–mobility gate. Depth plus enhanced embodied state was substantially better than depth alone—aggregate contact AUC increased from 0.7360 to 0.8927 and contact-negative retention increased from 0.4242 to 0.5636—but its recall was 0.9106 rather than the required 0.95. It admitted eleven contact-positive branches, selected two of them, had normalized route regret 0.2205, and recovered the best safe candidate in its top three only 0.6250 of the time.

This is positive development evidence for explicit geometry–embodied fusion, not a qualified safety filter. Neither an RGB-only privileged-geometry successor nor deployment of depth/LiDAR is authorized by this result.

## Frozen data and ontology

The run reused the complete data-scaling lineage without changing a state, branch, action, label, or split:

| Role | States | Branches | Construction |
|---|---:|---:|---|
| FIT-192 | 192 | 2,304 | Original 48, former fresh 48, and scaling-extra 96 |
| Calibration | 24 | 288 | Frozen scaling calibration states |
| Held-out | 24 | 288 | Frozen scaling held-out states |

Each branch has 15 future policy ticks and belongs to one of four equally represented maze families. The hard target is the existing frozen **disallowed robot/environment collision or body-contact** label. It excludes ordinary foot–ground contact and robot self-contact. Stuck is retained only as a recoverability/task-performance diagnostic and never rejects a candidate in these conditions. Fall and unsafe termination remain descriptive and degenerate; they were not learned.

Calibration contained 136 contact-positive and 152 contact-negative branches; held-out contained 123 contact-positive and 165 contact-negative branches. Held-out had 133 stuck-positive branches, including 70 overlapping contact.

## Geometry materialization and verification

No physics replay was required. All 240 states already had verified tick-pose evidence: 48 used the dense replay receipts and 192 used frozen pose/safety shards. Geometry was materialized at those poses against the immutable scene manifests. The receipts bind candidate action, pose, contact, stuck, and aggregate outcomes. There were zero identity or verification failures, zero new state/candidate identities, and zero simulator steps.

The resulting index digest is `453e0e692fb9430cc7373c477e2766ae676b147dbe116f8a9ba3a716f3b9b007`.

### Depth contract

- Ideal, noiseless metric pinhole depth at 64×48.
- Go2 RGB-camera mount `(0.326, 0, 0.043) m`, including frozen per-scene camera-extrinsic jitter.
- Horizontal field of view 78.323°.
- Hard range clipping `[0.05, 10.0] m` with a separate validity mask.
- Current, true future, signed difference, and validity channels are supplied to the model.
- The scene manifest is used only by the offline sensor producer; it is not a model input.

### LiDAR contract

- Ideal, noiseless 360° range scan, explicitly a **`CHANGED_DEPLOYMENT_SENSOR_CONTRACT`**.
- Body mount `(0, 0, 0.25) m`.
- 180 azimuth bins at 2° resolution and four vertical channels at −15°, −5°, 5°, and 15°.
- Hard range clipping `[0.05, 10.0] m` with a validity mask.
- Current, true future, signed difference, and validity channels are supplied.

This idealized scan is an observability diagnostic, not a claim about an already deployed physical LiDAR.

## Pre-training geometry audit

The 0.35 m proximity threshold was frozen and evaluated before any model was trained or scored.

| Audit quantity | Front depth | 360° LiDAR |
|---|---:|---:|
| Contact-positive branches | 1,252 | 1,252 |
| Contact visible before onset | 0.6701 | 0.8299 |
| Visible only at event tick | 0.0998 | 0.1478 |
| Visible at event tick | 0.7093 | 0.9704 |
| Never visible before or at event | 0.2300 | 0.0224 |
| Median first crossing relative to onset | −4 ticks | −4 ticks |
| Pre-event standardized range effect | 0.9183 | 1.0390 |
| Event-tick standardized range effect | 0.7699 | 0.9649 |
| Post-event standardized range effect | 0.9591 | 1.0285 |

The audit confirms wider range coverage but does not itself establish learnable candidate filtering. Link-resolved front/side/rear/body-contact location was not present in the frozen boolean traces, so no post-outcome contact relabeling was attempted.

## Evaluator and models

The common fixture passed all eleven cases: perfect/reversed ranking, transient and persistent contact, contact-negative branch, all-positive and one-negative candidate sets, no admission, exact threshold tie rejection, deterministic kinematic selection, and deterministic JSON serialization.

All conditions use balanced per-tick BCE for contact-active and cumulative-contact outputs, AdamW (`lr=1e-3`, weight decay `1e-4`), 60 final-epoch-only epochs, a one-layer causal GRU of width 128, and no route, utility, stuck, aggregate-unsafe, motion, goal, RGB, or ViT-L output.

| Condition | Derived seed | Parameters | Final loss | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| DEPTH_ONLY | 1,368,795,840 | 203,442 | 0.540560 | `6edd8b4c754f631759e343f40bf88502bada9c4b8923584696de87235d7ea4b0` |
| LIDAR_ONLY | 1,230,547,711 | 153,330 | 0.498315 | `e225ac245e8e625750a59cd17cf8608e507b44a4d693946548e03f7db807d26b` |
| DEPTH_PLUS_EMBODIED | 1,082,109,145 | 270,738 | 0.002033 | `8c51342d431c20496a60a69675851005cd9cc0d88f1440c9a583f1ae6d465204` |

Every fit-only smoke verified the allow-list, ordering, action and sensor sensitivity, finite/nonzero gradients, checkpoint reload, deterministic inference, and row-level persistence before calibration or held-out scoring.

## Calibration

One scalar temperature and one cumulative-H3 contact threshold were fit per condition on the same 24 calibration states. Thresholds were selected at calibration recall ≥0.95 by state availability, contact-negative retention, selected route progress, false abstention, then conservatism.

| Condition | Temperature | Threshold | Calibration recall | Negative retention | States retaining negative | Selected contacts |
|---|---:|---:|---:|---:|---:|---:|
| DEPTH_ONLY | 3.431524 | 0.257489 | 0.9632 | 0.2237 | 8/24 | 2 |
| LIDAR_ONLY | 3.734700 | 0.183621 | 0.9559 | 0.2697 | 9/24 | 2 |
| DEPTH_PLUS_EMBODIED | 3.777697 | 0.077567 | 0.9559 | 0.4934 | 21/24 | 2 |

Threshold ties reject. Held-out values were not used for calibration or threshold changes.

## Held-out branch and event results

| Metric | DEPTH_ONLY | LIDAR_ONLY | DEPTH_PLUS_EMBODIED |
|---|---:|---:|---:|
| Contact AUC | 0.7360 | 0.7148 | **0.8927** |
| Average precision | 0.6833 | 0.6714 | **0.8956** |
| Recall | 0.8537 | 0.9431 | 0.9106 |
| False-negative rate | 0.1463 | 0.0569 | 0.0894 |
| Contact-negative retention | 0.4242 | 0.2424 | **0.5636** |
| ECE | 0.0476 | 0.0497 | 0.0921 |
| Brier | 0.2032 | 0.2102 | **0.1301** |
| Event-tick recall | 0.8059 | 0.8270 | **0.8354** |
| Branch event detection | 0.9024 | **0.9350** | 0.9187 |
| Median detection delay | 0 ticks | 0 ticks | 0 ticks |
| Missed transient-contact rate | 0.2727 | **0.2182** | 0.2909 |

The static geometry audit’s high raw coverage therefore did not translate into adequate LiDAR-only score discrimination. Fusion provided the strongest branch-level discrimination, but still missed eleven of 123 contact-positive branches.

## Candidate filtering and route behavior

| Metric | DEPTH_ONLY | LIDAR_ONLY | DEPTH_PLUS_EMBODIED |
|---|---:|---:|---:|
| Admitted contact-negative | 70/165 | 40/165 | **93/165** |
| Admitted contact-positive | 18/123 | 7/123 | 11/123 |
| States retaining a negative | 14/24 | 11/24 | **23/24** |
| States admitting only positives | 0 | 0 | 0 |
| False abstentions | 10 | 13 | **1** |
| Selected contacts | 3 | **1** | 2 |
| Selected stuck rate (diagnostic) | 0.2857 | 0.4545 | 0.3478 |
| Selected progress | 0.1603 m | 0.1333 m | **0.1676 m** |
| Oracle-contact progress | 0.2067 m | 0.2067 m | 0.2067 m |
| Oracle progress retained | 0.7754 | 0.6449 | **0.8108** |
| Normalized route regret | 0.2814 | 0.2732 | **0.2205** |
| Best-safe top-1 | 0.2500 | 0.1667 | **0.5417** |
| Best-safe top-3 | 0.3750 | 0.1667 | **0.6250** |

Operational safety and task performance are reported separately. A filter is not successful merely because it rejects candidates: it must avoid disallowed contact while leaving enough useful actions for route progress. Fusion restored action availability and progress, but selected two contact-positive candidates and missed the recall, regret, and top-three gates.

## Per-family held-out results

| Condition / family | AUC | Recall | Negative retention | States retaining | Selected contacts | Progress (m) |
|---|---:|---:|---:|---:|---:|---:|
| Depth / large | 0.6760 | 0.8276 | 0.3953 | 4/6 | 0 | 0.0804 |
| Depth / medium | 0.6680 | 0.8438 | 0.2500 | 2/6 | 1 | 0.2381 |
| Depth / small | 0.6911 | 0.6800 | 0.5745 | 5/6 | 2 | 0.2525 |
| Depth / loop-alias | 0.9120 | 1.0000 | 0.4571 | 3/6 | 0 | 0.0614 |
| LiDAR / large | 0.7538 | 1.0000 | 0.2326 | 3/6 | 0 | 0.1060 |
| LiDAR / medium | 0.7781 | 1.0000 | 0.1750 | 2/6 | 0 | 0.1213 |
| LiDAR / small | 0.6877 | 1.0000 | 0.1702 | 2/6 | 0 | 0.1879 |
| LiDAR / loop-alias | 0.6903 | 0.8108 | 0.4286 | 4/6 | 1 | 0.1326 |
| Fusion / large | 0.8540 | 0.8276 | 0.6744 | 6/6 | 2 | 0.2408 |
| Fusion / medium | 0.9195 | 0.8750 | 0.7500 | 6/6 | 0 | 0.2361 |
| Fusion / small | 0.9396 | 1.0000 | 0.3830 | 6/6 | 0 | 0.1255 |
| Fusion / loop-alias | 0.9019 | 0.9459 | 0.4571 | 5/6 | 0 | 0.0481 |

The fusion condition had no availability collapse, but its large-maze contact selection and sub-threshold recall constitute a family-level safety failure.

## Frozen comparator context

Comparator values below are preserved published results. Only FIT-192 used this same held-out-24 panel; the other learned baselines and static-grid guard used the historical held-out-eight panel and are not paired comparisons.

| Frozen comparator | Contact AUC | Relevant operating-point result |
|---|---:|---|
| FIT-192 contact specialist | 0.8487 | Recall 0.9187; FNR 0.0813 |
| Enhanced embodied contact | 0.7549 | Historical contact-negative retention 0.5278 at component operating point |
| Raw RGB contact | 0.6719 | Historical retention 0.7917 with recall only 0.5417 |
| Final-layer ViT-L contact | 0.6175 | Historical retention 0.7778 with recall only 0.4167 |
| Privileged static-grid guard | — | Historical aggregate unsafe recall 0.6724; safe retention 0.5789 |
| Oracle contact | 1.0000 | Retention 1.0; selected contacts 0 |

## Gate and decision

No condition passed. Depth and LiDAR failed discrimination, recall/FNR, state availability, unsafe selection, regret, top-three recovery, and family requirements. Fusion passed AUC, negative retention, calibration, event recall, detection delay, state availability, abstention, and route-progress fraction. It failed:

- contact recall (`0.9106 < 0.95`);
- FNR (`0.0894 > 0.05`);
- zero selected contact (`2` selected);
- normalized regret (`0.2205 > 0.20`);
- best-safe top-3 (`0.6250 < 0.75`);
- no family collapse under the frozen safety definition.

The exploratory fusion-minus-depth effects were +0.1567 AUC and +0.1394 retention, satisfying the frozen substantial-improvement criterion but not the complete gate. Thus the terminal is **`GEOMETRY_MODALITY_POSITIVE_TENDENCY`**.

The next decision should review contact severity and body-region labels or narrow the learned hard-contact claim. `RGB_TO_GEOMETRY_SAFETY_STATE_V1` is not yet justified because true-future depth alone failed, and explicit LiDAR deployment is not justified because LiDAR alone also failed. A future geometry–embodied experiment would require a prospectively frozen architecture and independent panel; it must not be treated as an automatic continuation.

## Runtime, storage, and custody

- Geometry materialization: 69.07 aggregate compute seconds; 77.1 seconds wall time.
- Training/evaluation: 578.46 seconds total; condition training 198.04 / 164.42 / 197.67 seconds.
- Peak VRAM: 308,826,112 bytes.
- Geometry cache: 133,629,615 bytes (129 MiB on disk).
- Tracked/generated result directory: 4.2 MiB.
- Row-level ledgers persist raw logits, calibrated probabilities, labels, threshold decisions, admitted sets, selections, and route outcomes for all 576 calibration-plus-held-out rows per condition.

Exactly one independently keyed seed was used per condition from seed family `2026082012`. No JEPA predictor was opened or trained. No RGB/ViT-L, memory, novelty, routing, beacon, or navigation model was trained. No state or candidate identity was added. No process remained running when the result was finalized.
