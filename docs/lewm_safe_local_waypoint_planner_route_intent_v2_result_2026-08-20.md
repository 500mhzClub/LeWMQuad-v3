# SAFE_LOCAL_WAYPOINT_PLANNER_ROUTE_INTENT_V2 result

Status: `POST_OUTCOME_DEVELOPMENT_SUCCESSOR`

Terminal classification: `TRUE_FUTURE_ROUTE_INTENT_PLANNER_NO_GO`

The V1 terminal `PURPOSE_BUILT_LOCAL_WAYPOINT_DATA_NO_GO` remains unchanged: 34/48 frozen states met its safe positive-distance-progress criterion, below the prospectively frozen 75% threshold. V2 reused all 48 states and all 576 outcomes; it did not replace states, candidates, splits, or safety labels.

## Route-intent adequacy

The corrected route labels use distance progress and reduction in absolute heading error to the next route segment, with fixed margins of 0.03 m and 5 degrees. The V2 adequacy gate passed.

| State class | Count |
|---|---:|
| `TRANSLATIONAL_PROGRESS_AVAILABLE` | 31 |
| `ALIGNMENT_PROGRESS_AVAILABLE` | 7 |
| `SAFE_HOLD_OR_ABSTAIN` | 3 |
| `NO_SAFE_CANDIDATE` | 7 |

Fit contained 24 route-improving states; held-out contained seven. Held-out contained one alignment state and one abstention state. All splits contained safe and unsafe candidates and non-degenerate route progress. Every family contained an improving state and an unsafe candidate.

The route-label audit digest is `73381d7dc834813286b52f571a6b5d3370d04582fb5bf27beeee9447e5e4fd92`; the row index digest is `e8d33671502f717426836ec9a1039d445558b81e636ae105a81e2113151a8b69`.

## Safety reconciliation

The frozen V1 ledger contains 407 unsafe and 169 safe H3 branches. Seven states have zero safe candidates, nine have exactly one, and 32 have two or more.

V1 did not persist component fields. Deterministic replay was therefore used only as an attribution sensitivity and never replaced the frozen aggregate label. Replay observed 263 collision/disallowed-contact, 26 clearance-violation, 301 stuck, zero fall, and zero unsafe-termination flags. The replay aggregate agreed on 558/576 rows; 18 frozen-unsafe rows replayed safe, consistent with contact-label sensitivity. Frozen labels remained authoritative.

## Visual evidence

All 576 frozen branches were deterministically replayed to recover exact H1-H3 base poses and action traces. Static rendering then generated 1,728 textured-v03 RGB targets from the verified poses. The frozen V-JEPA 2.1 ViT-L encoder generated 1,728 FP16 `[768,1024]` grids. The latent-index digest is `df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874`.

Replay state runtime summed to 1,161.09 s; static rendering took approximately 245.71 s wall time; encoding took 33.68 s and peaked at 1.77 GB VRAM. RGB storage is 51,704,712 bytes and latent storage is 2,718,130,176 bytes.

## Planner training

The single seed was `2026082001`. The existing factorised development architecture was preserved at 132,551 trainable parameters: a 2,064-to-64 trajectory projection and seven separately readable outputs for motion, distance progress, heading progress, and path unsafe. True H1-H3 trajectories were used; no predictor was opened.

Training used AdamW (`lr=1e-3`, weight decay `1e-4`) for 60 epochs, final epoch only. Total loss fell from 1.6108 to 1.1835; motion loss from 0.1525 to 0.0725; progress loss from 0.1291 to 0.0367; safety loss from 0.6321 to 0.5503; and pairwise ranking loss from 0.6972 to 0.5240. The checkpoint SHA-256 is `6ef052a46632bbe400c1eab0bb4c45d4457b160a9492382c2f2297f095db198a`.

Calibration used only the eight calibration states. Frozen values were: unsafe logit scale 2.19283, bias -1.01432, unsafe threshold 0.85006, support threshold 1.46026, and unchanged abstention margins of 0.03 m and 5 degrees.

## True-future held-out result

| Gate metric | Result | Threshold | Pass |
|---|---:|---:|:---:|
| Displacement-direction cosine | 0.6497 | >=0.70 | No |
| Median endpoint error | 0.1546 m | <=0.0615 m | No |
| Yaw MAE | 36.61 deg | <=30 deg | No |
| Unsafe AUC | 0.6425 | >=0.80 | No |
| Unsafe ECE | 0.1178 | <=0.10 | No |
| Safe-candidate retention | 1.000 | >=0.80 | Yes |
| Pairwise route-preference accuracy | 0.6433 | >=0.70 | No |
| Best-safe top-3 | 0.500 | >=0.75 | No |
| Normalized safe distance regret | 0.5314 | <=0.25 | No |
| Correct abstention | 0.000 | >=0.75 | No |
| Unsafe movement in no-safe states | 0 | 0 | Yes |

The learned planner selected an unsafe branch in 4/7 non-abstained states (`selected_unsafe_rate=0.5714`). Its mean selected distance progress was 0.0404 m, below the action-only candidate prior's 0.1801 m. The action-only baseline also had lower normalized regret (0.2192 versus 0.5314) and higher pairwise accuracy (0.7898 versus 0.6433).

### Held-out states

| State | Family | Class | Selected | Safe | Distance progress (m) | Heading improvement |
|---|---|---|---:|:---:|---:|---:|
| purpose-10 | large | translational | 0 | No | -0.0913 | -14.56 deg |
| purpose-11 | large | translational | 4 | No | 0.0351 | 38.82 deg |
| purpose-22 | medium | translational | 2 | Yes | 0.3339 | -1.75 deg |
| purpose-23 | medium | translational | 5 | No | -0.0072 | -40.95 deg |
| purpose-34 | small | translational | 0 | Yes | 0.2979 | 9.10 deg |
| purpose-35 | small | translational | abstain | — | — | — |
| purpose-46 | loop alias | alignment | 1 | Yes | -0.1538 | -11.68 deg |
| purpose-47 | loop alias | hold/abstain | 4 | No | -0.1322 | 36.60 deg |

Large enclosed maze had no safe selection. The other families each had only one safe selection, so the no-family-collapse gate also failed.

## Decision

The purpose-built true-future planner did not qualify geometry, safety, route ranking, regret, abstention, or family robustness. Predictor evaluation was therefore correctly not reached. This result is development-only and does not alter the established predictor-fidelity or action-sensitivity evidence.

Exactly one planner seed was trained. No predictor checkpoint was opened. No global memory, novelty, beacon-capture, routing, or closed-loop layer was implemented.

Authoritative generated result: `.generated/safe_local_waypoint_route_intent_v2/result.json`, SHA-256 `0dd4e3d7d6f10a7693bc51fcb71faf10e9ea89a881c2914787f1fd64c71a83e9`.
