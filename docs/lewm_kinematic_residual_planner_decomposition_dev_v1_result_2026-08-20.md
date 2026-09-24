# KINEMATIC_RESIDUAL_PLANNER_DECOMPOSITION_AND_DEV_V1

Status: `POST_OUTCOME_DEVELOPMENT_SUCCESSOR`

Final classification: `KINEMATIC_RESIDUAL_PLANNER_NO_SIGNAL`

The frozen V2 terminal `TRUE_FUTURE_ROUTE_INTENT_PLANNER_NO_GO` is preserved. No predicted futures were evaluated with either planner.

## Frozen inputs

This experiment reused all 48 states, 576 branches, frozen 32/8/8 split, H1-H3 target latents, actions, poses, waypoints, route labels, and safety outcomes. It generated no simulation or visual evidence.

Bindings:

- target index: `df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874`;
- V2 checkpoint: `6ef052a46632bbe400c1eab0bb4c45d4457b160a9492382c2f2297f095db198a`;
- V2 result: `0dd4e3d7d6f10a7693bc51fcb71faf10e9ea89a881c2914787f1fd64c71a83e9`.

## Existing-planner decomposition

| Split | Direction cosine | Median endpoint error | Yaw MAE | Unsafe AUC | Unsafe ECE | Unsafe admission | Pairwise route | Top-3 | Normalized regret | Selected safe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.5767 | 0.1309 m | 44.64° | 0.7922 | 0.1006 | 0.8664 | 0.8051 | 0.4615 | 0.1799 | 0.3462 |
| Calibration | 0.4823 | 0.1456 m | 58.51° | 0.6412 | 0.0673 | 0.8056 | 0.7059 | 0.2857 | 0.5944 | 0.3333 |
| Held-out | 0.6497 | 0.1546 m | 36.61° | 0.6425 | 0.1178 | 0.9310 | 0.6433 | 0.5000 | 0.5314 | 0.4286 |

Fit route accuracy was materially higher than held-out (0.8051 versus 0.6433), while absolute geometry was poor on every split. The learned safety threshold admitted 93.1% of unsafe held-out candidates. Calibration reduced calibration-set ECE but did not transfer to held-out.

## Held-out component substitution

| Condition | Abstain | Selected safe | Selected distance progress | Heading improvement | Normalized regret | Best-safe top-1/top-3 |
|---|---:|---:|---:|---:|---:|---:|
| A: oracle safety + oracle route | 0.125 | 1.000 | 0.2669 m | 8.98° | 0.0000 | 0.875 / 1.000 |
| B: oracle safety + action-only route | 0.125 | 1.000 | 0.2508 m | 8.57° | 0.1086 | 0.750 / 1.000 |
| C: oracle safety + learned route | 0.250 | 1.000 | 0.1424 m | 0.45° | 0.4025 | 0.250 / 0.750 |
| D: learned safety + oracle route | 0.000 | 0.625 | 0.2295 m | 17.01° | 0.0000* | 0.625 / 0.750 |
| E: learned safety + learned route | 0.125 | 0.429 | 0.0404 m | 2.23° | 0.5314 | 0.000 / 0.375 |

`*` Normalized regret is defined only for selected safe candidates; D still selected unsafe candidates in 37.5% of states.

Conditions F/G were unavailable. The stored panel has neither candidate-level planning-time runtime-guard verdicts nor the current local-obstacle observation required to reconstruct the deployed guard. Realised safety labels were not substituted as a deployable guard.

The matrix shows that both learned components contributed to failure. Action-only route scoring with oracle safety was far stronger than the existing learned route score. Oracle route scoring did not repair unsafe admission under learned safety.

## Bottleneck classification

- `ABSOLUTE_MOTION_DECODING_FAILURE`: geometry failed on fit and held-out.
- `PROGRESS_RANKING_FAILURE`: learned held-out pairwise accuracy was 0.6433; oracle-safety learned-route regret was 0.4025.
- `SAFETY_DISCRIMINATION_FAILURE`: held-out unsafe AUC was 0.6425.
- `CALIBRATION_OR_THRESHOLD_FAILURE`: held-out ECE was 0.1178 and unsafe admission was 0.9310.
- `FIT_TO_HELDOUT_GENERALISATION_FAILURE`: route accuracy fell from 0.8051 on fit to 0.6433 held-out.

The diagnostic did not identify the deterministic lexicographic selector itself as the principal failure: oracle safety plus action-only scoring obtained 0.1086 regret and perfect top-3 recovery.

## Deterministic kinematic prior

The prior integrated the stored post-slew body twist at the frozen 0.10 s command interval for H1-H3. It used no learned model.

| Split | Mean endpoint error | Direction cosine | Yaw MAE | Distance-progress MAE | Heading-progress MAE | Pairwise route accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 0.1223 m | 0.5889 | 49.62° | 0.0744 m | 9.43° | 0.8544 |
| Calibration | 0.1111 m | 0.6228 | 57.70° | 0.0643 m | 11.25° | 0.7317 |
| Held-out | 0.0976 m | 0.6677 | 33.55° | 0.0664 m | 7.82° | 0.8854 |

Fit residual standard deviations were nonzero: 0.1383 m in x, 0.0700 m in y, 0.0948 m distance progress, and 0.2153 rad heading progress. Both continuation conditions therefore passed: the action-only/oracle-safety decision was meaningfully better than V2, and residual targets were nondegenerate.

## Residual planner

`KINEMATIC_RESIDUAL_LISTWISE_RANKER_V1` used seed `2026082002`, 2,070 inputs, a 16-wide latent layer, and seven outputs. It has 33,239 trainable parameters. It predicts residual x/y/yaw, distance progress, heading progress, and a separate route score. It has no learned safety, completion, uncertainty, or absolute-motion head.

Training used AdamW (`lr=1e-3`, weight decay `1e-4`) for 60 epochs, final epoch only. Total loss changed from 0.8453 to 0.7877; residual motion from 0.1276 to 0.0844; residual progress from 0.0239 to 0.0138; route-ranking loss from 0.6938 to 0.6896. The near-chance final ranking loss was an early indication of weak discrimination.

Checkpoint SHA-256: `d926d7a544ca39f015b369653b3a8e0cde7f318da98a1c20b01521f66629f124`.

## True-future residual result

| Metric | Kinematic baseline | Residual ranker | Required improvement |
|---|---:|---:|---:|
| Mean endpoint error | 0.0976 m | 0.1548 m | improve |
| Direction cosine | 0.6677 | 0.7893 | descriptive |
| Yaw MAE | 33.55° | 33.59° | descriptive |
| Route pairwise accuracy | 0.8854 | 0.5417 | +0.05 |
| Normalized safe regret | 0.1086 | 0.3452 | -0.05 |
| Best-safe top-1/top-3 | 0.750 / 1.000 | 0.250 / 0.625 | improve |
| Selected distance progress | 0.2508 m | 0.2210 m | improve |
| Abstention | 0.125 | 0.250 | descriptive |

Oracle safety was required for both decision rows because no valid planning-time guard was available. Thus the deployable-safety condition also necessarily failed.

### Residual decisions by held-out state

| State | Family | Selected | Distance progress | Heading improvement |
|---|---|---:|---:|---:|
| purpose-10 | large enclosed | 3 | 0.0381 m | 24.04° |
| purpose-11 | large enclosed | abstain | — | — |
| purpose-22 | medium enclosed | 0 | 0.5338 m | 11.35° |
| purpose-23 | medium enclosed | 4 | 0.1721 m | -23.91° |
| purpose-34 | small enclosed | 0 | 0.2979 m | 9.10° |
| purpose-35 | small enclosed | 1 | 0.3772 m | -0.75° |
| purpose-46 | loop alias | 9 | -0.0933 m | 3.92° |
| purpose-47 | loop alias | abstain | — | — |

No family improved mean selected distance progress over the kinematic baseline. All six residual selections were safe only because oracle safety constrained the candidate set; this is not a deployable safety result.

## Decision

All six residual-development requirements failed except none: endpoint error regressed, pairwise accuracy regressed, regret increased, selected progress fell, no deployable safety condition existed, and improvement was not present in three families. The classification is therefore `KINEMATIC_RESIDUAL_PLANNER_NO_SIGNAL`.

This closes local planner readouts on the current final-layer latent contract. Predictor evaluation was not reached.

No simulation, rendering, or target encoding occurred. Exactly one new planner seed was trained and no predictor seed was opened. No memory, novelty, beacon-discovery, or closed-loop layer was implemented.

Generated result SHA-256: `56346f02bedad1dd9f83208a9efa89e061c77e600501c9f718a331788b903c40`.

The new output directory occupies 217,730 bytes: 139,044-byte checkpoint plus result JSON. Initial train-and-evaluate execution took approximately 6.6 seconds; checkpoint-reload final evaluation took 1.1 seconds.
