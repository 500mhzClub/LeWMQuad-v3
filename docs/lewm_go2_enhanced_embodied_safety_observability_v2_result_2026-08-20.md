# Enhanced embodied safety observability V2 — development result

Date: 2026-08-20

Experiment: `ENHANCED_EMBODIED_SAFETY_OBSERVABILITY_V2`

Starting source: `19525d6ca2061924007377ea5fe3255dda85364b`

Preserved predecessor terminal: `CURRENT_DEPLOYMENT_SENSOR_CONTRACT_SAFETY_NO_GO`

## Terminal classification

`ENHANCED_EMBODIED_SAFETY_POSITIVE_TENDENCY`

The enhanced embodied trajectory improved aggregate unsafe discrimination and made a useful safety–retention operating point substantially less pessimistic. It did **not** pass the frozen common gate: held-out recall, false-negative rate, safe retention, calibration, contact discrimination, stuck discrimination, temporal event recall, retained-state count, top-3 recovery, and false-abstention limits failed. This is event-observability evidence from true future sensors, not evidence that risk can be forecast before executing a candidate.

The exact machine-readable result is `.generated/enhanced_embodied_safety_observability_v2/result.json` (SHA-256 `60f53118056fd140c8a6641b75806d54c2ad0dc6d2bd15a3f860bb61b1a65693`).

## Frozen panel and replay

The experiment reused all 48 states and 576 branch identities: 32/8/8 fit/calibration/held-out states, four maze families, twelve candidates per state, and fifteen policy ticks through H3. No state, candidate, action plan, split, route outcome, or frozen safety label was replaced.

The registered branches were replayed only to capture the omitted deployment-valid channels. Every replay matched its registered post-slew action trace, H1–H3 poses, contact trace, and stuck trace. There were zero such mismatches over 576 branches. Twenty-one reconstructed snapshots differed in serialization digest but reproduced every checked tick-level field. The known right-censoring distinction was preserved: 558/576 raw replay H3 aggregates equal the authoritative ledger directly; the remaining 18 retain the frozen authoritative aggregate instead of being relabelled.

Sensor index:

- content digest: `d8b9721a2397961912e604b41b9b4eaea49ee34fc2c4735eba6f6e1edbe0933d`;
- file SHA-256: `87770bad75aee18f10c790b7bb2efb1bd90c7850053a17e2db24e1a6d3f56d20`;
- sensor shards: 2,096,282 bytes;
- replay compute: 360.42 s;
- replay wall time: 391.27 s.

## Enhanced sensor contract

The per-tick embodied state has 73 continuous channels:

- body-frame classical accelerometer, gyroscope, and projected gravity: 3 + 3 + 3;
- joint position relative to the deployed default, velocity, and causal acceleration: 12 + 12 + 12;
- Genesis PD control force as deployment-equivalent actuator torque: 12;
- measured calf-link net contact-force magnitude: 4;
- previous deployed locomotion-policy action: 12.

The action/control vector has six channels: registered post-slew `vx`, `vy`, and yaw command plus the previous applied `vx`, `vy`, and yaw command. Joint acceleration is the causal 10 Hz finite difference of encoder velocity; the pre-action boundary uses a zero reference because a preceding dense velocity sample was not registered.

Jacobian-estimated foot force was unavailable because the frozen Genesis robot was not built with Jacobian/IK support. No surrogate was fabricated. No enhanced channel was degenerate over the corpus.

Excluded inputs were global position or yaw, body linear velocity, scene graph, occupancy grid, RGB, safety labels, and privileged collision geometry.

## Input–label circularity

No input channel was mathematically identical to any of the five tick-level targets. Measured calf force includes ordinary ground support and is not the disallowed-body-contact label. PD actuator force is a continuous controller quantity, not a collision verdict. Stuck labels depend on privileged displacement/window logic, while global or body translation was excluded from the model.

## Event-aligned audit

The audit froze fit-negative 95th-percentile thresholds before examining calibration or held-out event detectability. “Before” means the tick immediately preceding first label onset; event-tick and post-event evidence are detection or aftermath, not preventive evidence.

| Split | Component | Positive branches | Positive ticks | detectable before | first at event | only after | no measured signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit | contact | 189 | 415 | 0.1693 | 0.3651 | 0.0952 | 0.3704 |
| fit | stuck | 204 | 570 | 0.1225 | 0.3578 | 0.0588 | 0.4608 |
| calibration | contact | 50 | 102 | 0.0800 | 0.3000 | 0.0800 | 0.5400 |
| calibration | stuck | 53 | 142 | 0.0377 | 0.3396 | 0.1132 | 0.5094 |
| held-out | contact | 24 | 47 | 0.1667 | 0.4583 | 0.0000 | 0.3750 |
| held-out | stuck | 44 | 99 | 0.0682 | 0.4318 | 0.0227 | 0.4773 |

Held-out standardized positive-versus-negative tick effects were modest. For contact, accelerometer norm was the largest positive effect (`0.3811`), followed by calf force (`0.2364`) and actuator torque (`0.1885`); joint acceleration was `0.0248`. For stuck, all absolute effects were small: calf force `0.1388`, accelerometer `0.1057`, joint acceleration `-0.2164`, and torque `-0.0158`.

At held-out contact onset, mean accelerometer norm rose from 11.80 at the preceding tick to 16.91, mean peak torque from 21.44 to 24.29 N·m, and mean peak calf force from 145.44 to 170.53 N. These are event-aligned signals; only 16.67% of held-out contact events had a threshold exceedance on the preceding tick.

## Evaluator fixture

The common evaluator passed deterministic fixtures for one-tick contact, persistent contact, delayed stuck, a safe branch, all-unsafe candidates, exactly one safe candidate, no admitted candidate, an exact threshold tie, perfect/reversed rankings, deterministic selection, and JSON serialization/reload.

## Model and training

`ENHANCED_PROPRIO_ACTION_SAFETY_HEAD_V1` contains 146,645 trainable parameters:

1. standardized current, true-future, and future-minus-current embodied state (`219 → 128`, GELU);
2. action/control MLP (`6 → 48`, GELU);
3. one-layer causal GRU (`176 → 128`);
4. five per-tick logits: active contact, active stuck, cumulative contact, cumulative stuck, and cumulative aggregate unsafe.

One seed (`2026082009`) was trained for 60 final-epoch-only AdamW epochs (`lr=1e-3`, weight decay `1e-4`). The real-data smoke verified allow-list, ordering, action and temporal sensitivity, finite nonzero gradients, save/reload, and deterministic inference without opening calibration or held-out rows.

- initial balanced BCE: `1.010765`;
- final balanced BCE: `0.085501`;
- training runtime: 5.20 s;
- peak allocated VRAM: 78,187,520 bytes;
- checkpoint SHA-256: `82b2704c770e2332a4a1e25b83fc6d0e8277877bee522e393f57cca3b5382a77`;
- checkpoint bytes: 594,485.

## Calibration

The eight calibration states produced:

- scalar temperature: `5.1851999035`;
- admission threshold: `0.4403937459` (admit only below threshold; equality rejects);
- calibration unsafe recall: `0.972222`;
- calibration safe retention: `0.208333`;
- criterion satisfied: yes.

The threshold maximised safe retention subject to calibration recall at least 0.95, with the more conservative threshold used for ties. It was frozen before held-out scoring.

## Held-out branch and event metrics

| Metric | Enhanced model | Frozen original proprio | Frozen RGB+proprio | Frozen action/control | Frozen final ViT-L |
|---|---:|---:|---:|---:|---:|
| Aggregate unsafe AUC | **0.8373** | 0.7679 | 0.7670 | 0.7793 | 0.7255 |
| Contact AUC | 0.7549 | **0.7671** | 0.7613 | 0.5747 | 0.6175 |
| Stuck AUC | 0.6211 | 0.6541 | 0.7008 | **0.8654** | 0.7745 |
| Safe retention | **0.3947** | 0.0526 | 0.1316 | 0.0000 | 0.0000 |
| Missed transient contact | **0.3171** | 0.7561 | 0.7561 | 0.3902 | 0.9756 |
| States retaining a safe action | **5** | 2 | 3 | 0 | 0 |

Additional enhanced metrics:

- aggregate AP `0.8720`, recall `0.9310`, FNR `0.0690`, ECE `0.1199`, Brier `0.1847`;
- contact AP `0.4492`, cumulative recall `0.9583`, tick recall `0.7021`, median delay `0` ticks;
- stuck AP `0.5025`, cumulative recall `0.7273`, tick recall `0.1414`, median delay `1` tick;
- unsafe branches detected before or at first unsafe tick `0.7759`.

Thus the enhanced channels substantially improved retention and transient-contact detection relative to the original 42-channel proprioceptive contract, but did not qualify contact or stuck observability under the frozen gate.

## Candidate filtering and kinematic planning

The frozen filter admitted 19/96 candidates: 15 safe and four unsafe. Five of eight states retained at least one safe candidate; no state retained only unsafe candidates. The unchanged kinematic ranker selected five candidates and abstained in three states. All five selected candidates were safe.

- selected unsafe rate: `0`;
- mean selected distance progress: `0.32761 m`;
- oracle-safety kinematic mean progress: `0.21838 m`;
- normalized safe-progress regret: `0.1330` over five eligible states;
- best-safe top-1/top-3: `0.50 / 0.50`;
- false abstentions: `3`;
- states holding despite a safe positive-progress action: `2`.

The progress fraction exceeds one because the filtered subset happened to select higher-progress actions than the oracle-safety kinematic tie rule on this eight-state development set; it is not evidence that the filter exceeds an oracle safety decision generally.

## Per-family held-out result

| Family | Aggregate AUC | Contact AUC | Stuck AUC | Safe retention | retained states | selected progress (m) | selected unsafe | abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large enclosed maze | 0.7500 | 0.6641 | 0.5714 | 0.0000 | 0/2 | 0.0000 | 0 | 1.00 |
| medium enclosed maze | 0.7704 | 0.7750 | 0.6484 | 0.4667 | 2/2 | 0.4546 | 0 | 0.00 |
| small enclosed maze | 0.8357 | 0.6421 | 0.8531 | 0.6000 | 2/2 | 0.3545 | 0 | 0.00 |
| loop alias stress | 0.8741 | 0.9118 | 0.5037 | 0.2222 | 1/2 | 0.0198 | 0 | 0.50 |

Large-maze filtering collapsed completely. Small-maze normalized regret was `0.3325`; the other selected-family regrets were zero where defined.

## Frozen gate outcome

Passed: aggregate AUC; median detection delay; no only-unsafe-admitted state; zero selected unsafe; route progress fraction; normalized regret.

Failed: aggregate recall and FNR; safe retention (`0.3947 < 0.40`); ECE; contact AUC; stuck AUC; contact and stuck tick recall; six-state safe retention; best-safe top-3; and no-more-than-one false abstention.

The incremental-value gate passed because safe retention improved by `0.3421`, missed transient contact fell by `0.4390`, and three additional states retained a safe candidate relative to original proprioception. Stuck AUC fell by `0.0330`, within the predeclared `0.05` material-regression tolerance. The common gate did not pass, so the result remains a positive tendency only.

## Decision

The omitted physical channels contain useful true-future event evidence, especially for retaining safe candidates and detecting transient contact. They do not yet provide a qualified safety decision: preventive evidence is sparse, stuck generalisation is weak, calibration misses the frozen operating point, and one family rejects everything.

Do not train the action-conditioned micro-safety predictor from this result. The next sensor decision should either add a clearly changed contract—dedicated body-contact sensing or environment geometry such as depth/LiDAR—or narrow the learned safety claim to the event modes demonstrably observable. A fresh evaluation must separate pre-event forecasting from event detection.

## Custody and stop

- Exactly one new safety-head seed was trained.
- No JEPA/world-model predictor was opened or trained.
- No RGB, ViT-L, memory, novelty, beacon, route, or navigation model was trained.
- No state or candidate identity was created or replaced.
- The simulator was replayed only for deployment-valid sensor instrumentation and frozen-trace verification.
- Nothing remains running.
