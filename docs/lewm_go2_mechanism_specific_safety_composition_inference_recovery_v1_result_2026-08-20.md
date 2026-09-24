# Mechanism-specific safety composition inference recovery V1

Date: 2026-08-20

Starting source: `6982980178748cb1ca6eb42bd94245981593077a`

Status: `POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC`

Preserved results:

- `ENHANCED_EMBODIED_SAFETY_POSITIVE_TENDENCY`;
- `MECHANISM_SPECIFIC_SAFETY_COMPOSITION_V1 / ROW_ALIGNED_COMPONENT_PREDICTIONS_UNAVAILABLE`.

## Classification

`MECHANISM_SPECIFIC_SAFETY_COMPOSITION_NO_SIGNAL`

Frozen checkpoint inference recovered the omitted row evidence and exactly reproduced both published evaluations. The physically specified specialist combination achieved excellent unsafe recall but rejected too many safe candidates and did not preserve route ranking. Calibration-only specialist selection chose the same two specialists, so the secondary diagnostic is identical and also fails.

This is post-outcome development evidence on a previously examined held-out set. It is not an independent scientific claim.

Machine-readable result SHA-256: `3470e8d059c91b7160ae377df55f4230ff69b9234013b0c2ed33552b1ee7ae16`.

## Row alignment and recovered ledger

All 576 branch rows aligned exactly across the two completed experiments:

- 48 states and twelve candidates per state;
- 384/96/96 fit/calibration/held-out rows;
- state, candidate, split, family, and fifteen-tick identity;
- action/control sequence;
- contact, stuck, and aggregate unsafe targets;
- enhanced sensor-index digest `d8b9721a2397961912e604b41b9b4eaea49ee34fc2c4735eba6f6e1edbe0933d`.

The original action model's four channels reproduced the `vx/yaw` projection of the enhanced six-channel sequence exactly; both enhanced `vy` channels were zero as frozen.

Reusable ledger:

- schema: `row_level_component_predictions_v1`;
- decoded-array content digest: `e4e7ae1b494b171dd8a623a5368045a07f315e4ff05a85921b7e004c7d55e9de`;
- serialized SHA-256: `a28be7a1254a77b553730c3024fb6ef24ed914a64ebf8bae3458142e3b0f8a08`;
- index SHA-256: `8223d226209c27860237145daaf0870c713d512ad0b2841ec8b90e4d96c32a03`;
- bytes: 193,003;
- raw-logit dtype: FP32;
- raw tensors: `[576,15,5]` for both frozen models.

The ledger also retains row/state/candidate/split/family identity, labels, both action/control tensors, calibrated specialist probabilities, per-component rejection decisions, and the final OR admission. It is sufficient for later reduction without checkpoint execution.

## Frozen checkpoint reproduction

Checkpoint hashes matched the authorised bindings:

- `ACTION_CONTROL_ONLY`: `bc80ad410f83ab8503976a2cca850c833e05759af9e0cb85c46b406644eb8dcf`;
- `ENHANCED_EMBODIED`: `82b2704c770e2332a4a1e25b83fc6d0e8277877bee522e393f57cca3b5382a77`.

The reducer compared 355 published scalar/count fields per condition, including every aggregate, component, temporal, and per-family field. All 710 fields met `atol=1e-8`, `rtol=1e-7`; selected candidates matched exactly.

| Frozen condition | Metric | Published | Recovered |
|---|---|---:|---:|
| Action/control | aggregate AUC | 0.7792649728 | 0.7792649728 |
| Action/control | contact AUC | 0.5746527778 | 0.5746527778 |
| Action/control | stuck AUC | 0.8653846154 | 0.8653846154 |
| Action/control | safe retention | 0 | 0 |
| Enhanced embodied | aggregate AUC | 0.8373411978 | 0.8373411978 |
| Enhanced embodied | contact AUC | 0.7549189815 | 0.7549189815 |
| Enhanced embodied | stuck AUC | 0.6210664336 | 0.6210664336 |
| Enhanced embodied | safe retention | 0.3947368421 | 0.3947368421 |
| Enhanced embodied | contact tick recall | 0.7021276596 | 0.7021276596 |
| Enhanced embodied | missed transient contact | 0.3170731707 | 0.3170731707 |

No reproduction failure was observed.

## Primary specialist binding and calibration

The assignment was frozen before recovery:

- contact: `ENHANCED_EMBODIED`;
- stuck: `ACTION_CONTROL_ONLY`.

Each scalar temperature and threshold used only the eight calibration states and its own component labels.

| Specialist | Temperature | Threshold | Calibration recall | Negative retention | AP | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Enhanced contact | 20.000000 | 0.506640 | 0.9000 | 0.1957 | 0.6488 | 0.5880 |
| Action/control stuck | 1.088569 | 0.490694 | 0.9057 | 0.3721 | 0.8192 | 0.7802 |

Equality rejects. No aggregate unsafe label influenced either threshold.

## Specialist component performance

| Held-out metric | Enhanced contact | Action/control stuck |
|---|---:|---:|
| AUC | 0.7549 | 0.8654 |
| Average precision | 0.4492 | 0.8568 |
| Recall | 0.9167 | 0.9773 |
| False-negative rate | 0.0833 | 0.0227 |
| Negative specificity/retention | 0.5278 | 0.3269 |
| ECE | 0.2678 | 0.1799 |
| Brier | 0.2419 | 0.1936 |
| Event-tick recall | 0.5745 | 0.7778 |
| Median first-event delay | 0 ticks | 0 ticks |
| Missed transient-event rate | 0.4390 | 0.1912 |

Per-family cumulative recall:

| Family | Contact recall | Contact AUC | Stuck recall | Stuck AUC |
|---|---:|---:|---:|---:|
| large enclosed maze | 1.0000 | 0.6641 | 1.0000 | 0.9036 |
| medium enclosed maze | 0.7500 | 0.7750 | 1.0000 | 0.9141 |
| small enclosed maze | 0.8000 | 0.6421 | 1.0000 | 0.9406 |
| loop alias stress | 1.0000 | 0.9118 | 0.9333 | 0.7481 |

The specialists therefore retain mechanism differences but neither is uniformly calibrated across families.

## Primary OR composition

A branch was admitted only when both component probabilities were below their independent thresholds.

- unsafe recall: `0.9828` (57/58);
- unsafe false-negative rate: `0.0172` (one admitted unsafe branch);
- safe retention: `0.2895` (11/38);
- total admitted: `12` — 11 safe, one unsafe;
- states retaining a safe candidate: `6/8`;
- states admitting only unsafe candidates: `0`;
- states admitting no candidate: `2`;
- false abstentions: `2`;
- descriptive max-risk AUC/AP: `0.8475 / 0.8910`;
- descriptive union-risk AUC/AP: `0.8607 / 0.9022`.

The deterministic kinematic ranker selected six candidates, all safe:

- selected unsafe rate: `0`;
- mean distance progress: `0.19435 m`;
- oracle-safety kinematic progress fraction: `0.8900`;
- normalized safe-progress regret: `0.3788`;
- best-safe top-1/top-3: `0.125 / 0.125`;
- abstention rate: `0.25`;
- states holding despite a safe positive-progress candidate: `1`.

Safety recall and selected-branch safety improved, but safe retention and route-choice quality were inadequate.

## Per-state result

| State | Family | Admitted safe/unsafe | Selected | Safe | Distance progress (m) | Heading improvement | Abstain |
|---|---|---:|---:|---|---:|---:|---|
| purpose-10 | large | 0/0 | — | — | — | — | yes |
| purpose-11 | large | 1/0 | 11 | yes | 0.0258 | 1.79° | no |
| purpose-22 | medium | 2/0 | 2 | yes | 0.3339 | −1.75° | no |
| purpose-23 | medium | 2/0 | 2 | yes | 0.1727 | 4.29° | no |
| purpose-34 | small | 1/1 | 2 | yes | 0.2369 | 14.24° | no |
| purpose-35 | small | 3/0 | 1 | yes | 0.3772 | −0.75° | no |
| purpose-46 | loop alias | 2/0 | 11 | yes | 0.0198 | 6.80° | no |
| purpose-47 | loop alias | 0/0 | — | — | — | — | yes |

## Per-family composition

| Family | Unsafe recall | Safe retention | Retained states | Progress (m) | Normalized regret | Top-3 | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|
| large enclosed maze | 1.0000 | 0.2500 | 1/2 | 0.0258 | 0.3167 | 0.0000 | 0.50 |
| medium enclosed maze | 1.0000 | 0.2667 | 2/2 | 0.2533 | 0.5254 | 0.0000 | 0.00 |
| small enclosed maze | 0.9286 | 0.4000 | 2/2 | 0.3070 | 0.4525 | 0.0000 | 0.00 |
| loop alias stress | 1.0000 | 0.2222 | 1/2 | 0.0198 | 0.0000 | 0.5000 | 0.50 |

No family was completely rejected, but route recovery was poor in every family.

## Frozen comparators

| Condition | Unsafe recall | Safe retention | Safe-retaining states | Selected progress (m) | Normalized regret | Top-3 | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action/control unified | 1.0000 | 0.0000 | 0 | 0.0000 | — | 0.0000 | 1.000 |
| Original proprioception | 0.9828 | 0.0526 | 2 | 0.3159 | 0.2970 | 0.0000 | 0.750 |
| RGB + proprioception | 1.0000 | 0.1316 | 3 | 0.4024 | 0.0626 | 0.2500 | 0.625 |
| Enhanced embodied unified | 0.9310 | 0.3947 | 5 | 0.3276 | 0.1330 | 0.5000 | 0.375 |
| Primary specialists | 0.9828 | 0.2895 | 6 | 0.1944 | 0.3788 | 0.1250 | 0.250 |
| Oracle safety + kinematics | 1.0000 | 1.0000 | 8 | 0.2184 | 0.0000 | 1.0000 | 0.000 |

The privileged static-grid guard retained `0.5789` of safe candidates but recalled only `0.6724` of unsafe candidates (FNR `0.3276`); it remains unqualified.

## Development gate

Passed:

- aggregate recall and FNR;
- both component recalls;
- six states retaining a safe action;
- no only-unsafe-admitted state;
- zero selected unsafe;
- at least 80% of oracle progress;
- no complete family rejection.

Failed:

- safe retention (`0.2895 < 0.40`);
- false abstentions (`2 > 1`);
- normalized regret (`0.3788 > 0.20`);
- best-safe top-3 (`0.125 < 0.75`).

## Calibration-selected secondary composition

The secondary candidate set was restricted to the two lawfully regenerated conditions. Original proprioception and RGB-plus-proprioception were not executed.

Calibration contact AP selected enhanced embodied (`0.6488` versus action/control `0.5315`). Calibration stuck AP selected action/control (`0.8192` versus enhanced `0.7552`). The secondary assignment is therefore identical to the primary physical assignment, its metrics are identical, and it also fails. There is no separate post-hoc tendency.

## Decision

Specialist recombination does not rescue the current learned safety interface. It improves unsafe recall relative to the unified enhanced model but loses useful safe actions and route-ranking quality. Do not automatically train another head on this panel.

The one recommended next architecture is `FACTORISED_MICRO_SAFETY_WORLD_MODEL_V1`, designed jointly from the start with independently trained contact/impact and stuck/motion-shortfall states and evaluated on a fresh frozen panel. This is an architecture-design recommendation, not authority to train it automatically.

This diagnostic does not justify treating an unreported geometric veto as learned safety evidence.

## Prospective evidence persistence

The project-wide `ROW_LEVEL_EVIDENCE_PERSISTENCE` requirement is established in `docs/lewm_row_level_evidence_persistence_requirement_2026-08-20.md`. Future evaluations must persist identities, raw logits, calibrated probabilities, labels, component decisions, candidate inputs, thresholds, and content digests before aggregate reduction. Historical aggregate-only results remain unchanged.

## Runtime and custody

- deterministic inference: `0.61 s`;
- total inference/reduction/serialization: `2.84 s` before the reproduction-report refinement;
- ledger: `193,003` bytes;
- result JSON: approximately `188 kB`;
- models trained or fine-tuned: zero;
- frozen safety checkpoints executed: two;
- simulator, rendering, encoding, and JEPA predictor access: none;
- states, candidates, labels, splits, and kinematic ranking changed: none.
