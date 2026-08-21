# EXPLICIT_PER_LINK_GEOMETRIC_MICRO_STATE_UPPER_BOUND_V1 result

Date: 2026-08-21
Source commit: `10b3a190d506830e6a87e04a0f1c832b92295bd7`

## Outcome

Primary classification: `SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO`.

Exact Genesis-congruent structured geometry passed the development upper-bound gate. None of front depth, ideal LiDAR, or their union passed the frozen sensor gate. The two-ply micro-viability query is therefore geometrically expressible with the actual successor state, but the represented deployment-sensor surfaces do not cover it sufficiently. This result does not authorise a successor-state predictor or another contact classifier.

The predecessor scope remains unchanged: the repaired learned contact evaluator memorised development-training transitions but did not generalise scene-disjointly even with the actual successor observation. `TWO_PLY_SUCCESSOR_EVIDENCE_RECONSTRUCTION_BLOCKER`, `TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL`, `DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL`, `TRUE_SUCCESSOR_SET_COMPUTE_SIGNAL`, `REPLANNING_INTERFACE_UNRESOLVED`, and `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING` remain preserved.

## Corpus and action contract

The experiment binds repaired-corpus logical digest `e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223`, corpus-index SHA-256 `c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0`, action-contract SHA-256 `cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06`, and predecessor-ledger SHA-256 `63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94`.

Both qualified lateral actions are deployable under the repaired simulator contract: applied `vy=+0.20 m/s` and `vy=-0.20 m/s` remain nonzero through the adapter and controller. State-dependent route-action slew creates duplicate applied commands. Across 176 states, 2,464 historical current entries reduce to 1,594 unique controller/applied-command entries; unique counts are 8 actions in 51 states, 9 in 70, 10 in 49, and 11 in six. For successor sets reached through unique prefixes, 17,360 historical entries reduce to 11,791 unique applied actions.

No held-out state changes zero-versus-nonzero viability after deduplication. The action audit therefore classifies `DEPLOYABLE_MICRO_ACTION_CONTRACT_ALIGNED`; `DEPLOYABLE_MICRO_ACTION_CONTRACT_VIABILITY_NO_GO` is not supported. Historical fourteen-entry results are retained descriptively.

## Geometry contract and materialisation

Every registered 100 ms transition was replayed for 50 physics steps at 2 ms. The persisted contract contains the 13 articulated link transforms and all 27 Go2 collision-shape transforms, scene collision primitives, requested and applied commands, previous-command/controller state, pair filters, and permitted support-ground exclusions. Self-contact and ordinary foot/calf-ground support remain excluded.

At every post-step realised configuration, the history-free exact query restores `qpos`, invokes the Genesis 0.3.14 collision path, and reads native broadphase plus MPR/GJK fallback and manifold generation without advancing the solver. Exact positive distance is not exposed; analytical primitive separation is stored only as a positive-clearance diagnostic and is never substituted for the exact binary verdict.

The structured per-link record contains minimum signed clearance, time to minimum, first threshold crossing, obstacle sector, continuous approach direction where a manifold is available, relative normal approach speed, exact link contact verdict, responsible link/object, and penetration. Sensor conditions apply the same articulated-body query to current and actual next-boundary point clouds. The scene map is used only to render the ideal sensor evidence during authorised replay; it is not an input to the point-cloud-to-body query.

Materialised evidence:

- 176 states;
- 29,470 current/successor transitions;
- 1,473,500 physics steps;
- 29,379/29,470 frozen-label/history-free exact agreements;
- 29,441/29,470 frozen-label/replay-native agreements;
- 1,811,769,485 bytes of compressed geometry shards.

On the development-held-out role, exact agreement is 3,827/3,836 (`0.997654`). Nine discrepancies remain: seven history-free manifold positives absent from the repaired force-threshold label (`CONTACT_MARGIN_OR_MANIFOLD_EFFECT`) and two frozen current contacts not reproduced by the history-free query (`DYNAMIC_OR_CONSTRAINT_SOLVER_DEPENDENCE`). The latter do not make dynamics the bottleneck because the complete exact gate still passes. All discrepancies remain individually inventoried.

## Calibration

Only the repaired internal-calibration role selected sensor clearance thresholds:

| Condition | Threshold | Frontier points | Eligible points | Calibration recall | Calibration FNR |
|---|---:|---:|---:|---:|---:|
| Front depth | 0.675946832 m | 984 | 70 | 0.951242 | 0.048758 |
| Ideal LiDAR | 0.105573244 m | 981 | 52 | 0.950322 | 0.049678 |
| Depth + LiDAR | 0.105573244 m | 982 | 47 | 0.950322 | 0.049678 |

Exact Genesis uses its native binary narrowphase/manifold convention and has no fitted clearance threshold.

## Development-held-out contact result

| Condition / level | Rows | Positive | AUC | AP | Recall | FNR | Negative retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact / current | 336 | 86 | 0.988372 | 0.995098 | 0.976744 | 0.023256 | 1.000000 |
| Exact / successor | 3,500 | 502 | 0.998833 | 0.988933 | 1.000000 | 0.000000 | 0.997665 |
| Front depth / current | 336 | 86 | 0.789209 | 0.559438 | 0.988372 | 0.011628 | 0.108000 |
| Front depth / successor | 3,500 | 502 | 0.829443 | 0.463316 | 1.000000 | 0.000000 | 0.164443 |
| LiDAR / current | 336 | 86 | 0.805628 | 0.528792 | 0.732558 | 0.267442 | 0.748000 |
| LiDAR / successor | 3,500 | 502 | 0.811691 | 0.383476 | 0.770916 | 0.229084 | 0.768512 |
| Fused / current | 336 | 86 | 0.810512 | 0.560963 | 0.732558 | 0.267442 | 0.716000 |
| Fused / successor | 3,500 | 502 | 0.824992 | 0.464263 | 0.784861 | 0.215139 | 0.746164 |

The 588 held-out positive transition rows attribute to 315 front-limb, 114 rear-limb, 157 trunk, and two unresolved contacts. The most frequent links are `FR_calf` (206), base (157), `FR_hip` (95), and `RR_calf` (75). Under the fused threshold, missed contacts include 58 front-limb and 72 rear-limb rows, directly supporting a body/vertical sensor-coverage diagnosis.

## Safe-action count and state decisions

Unique-deployable-action results against repaired oracle counts:

| Condition | Count MAE | Spearman | Exact count | Zero/nonzero | False zero | False nonzero |
|---|---:|---:|---:|---:|---:|---:|
| Exact | 0.018868 | 0.999711 | 0.987421 | 1.000000 | 0.000000 | 0.000000 |
| Front depth | 6.937107 | 0.134136 | 0.201258 | 0.220126 | 0.849315 | 0.000000 |
| LiDAR | 2.213836 | 0.573245 | 0.691824 | 0.761006 | 0.239726 | 0.230769 |
| Fused | 2.345912 | 0.567573 | 0.679245 | 0.742138 | 0.260274 | 0.230769 |

There are 20 oracle-viable and four oracle-nonviable held-out states under both contracts.

| Condition | Viable retained | Correct nonviable abstentions | Selected contact | Selected nonviable successor | False abstention | Progress / exact oracle | Normalized regret | Top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact | 20/20 | 4/4 | 0 | 0 | 0 | 1.000000 | 0.000000 | 1.000000 |
| Front depth | 2/20 | 4/4 | 0 | 0 | 18 | 0.113522 | 0.886478 | 0.100000 |
| LiDAR | 14/20 | 4/4 | 0 | 1 | 6 | 0.822889 | 0.222662 | 0.700000 |
| Fused | 14/20 | 4/4 | 0 | 1 | 6 | 0.788807 | 0.256744 | 0.650000 |

Exact unique-action selected/oracle H3 progress is `4.249981 m`. Under the historical fourteen-entry contract it is `4.432583 m`; both are exactly oracle-optimal inside their respective contracts. The different totals reflect removal of duplicate applied commands, not a viability loss. No exact family collapses. Front depth collapses in three families; LiDAR and fused avoid complete family collapse but fail multiple contact, viability, regret, and ranking gates.

## Compute

The CPU float32 benchmark reduces already materialised per-link geometry for all unique current actions and every actual-successor action set, performs thresholding, safe-action counting, and unchanged H3 selection:

- P50/P90/P95/P99: `0.414932 / 0.417993 / 0.418793 / 0.429105 ms`;
- maximum: `0.609253 ms`;
- 50/80/100 ms misses: `0 / 0 / 0`;
- peak RSS: `757,518,336` bytes;
- peak VRAM: `0`.

Classification: `STRUCTURED_GEOMETRY_SET_REDUCTION_COMPUTE_SIGNAL`.

This benchmark excludes simulator trajectory generation and actual-successor acquisition. It is a reduction upper bound, not a qualified prospective 100 ms observation/replacement interface. `REPLANNING_INTERFACE_UNRESOLVED` remains unchanged.

## Decision

Exact articulated geometry passes all contact and two-ply viability gates, so `EXACT_GEOMETRIC_MICRO_STATE_NO_GO` is not supported and articulated contact dynamics are not the next bottleneck. The deployable applied-action audit is aligned, so the current result is also not an action-contract no-go.

No sensor-derived condition passes; therefore `PER_LINK_CLEARANCE_PREDICTOR_V1` is not yet authorised. The exact next step is a prospective **body-centric 360-degree, vertically denser range-coverage qualification** covering trunk, hip, calf, rear, and side swept volumes. The same deterministic per-link gate must pass under that sensor contract before any per-link clearance model is trained. If equivalent physical sensor coverage is not intended, the alternative is to narrow the simulated contact-avoidance scope transparently rather than train another scalar classifier.

## Runtime, persistence, and claims boundary

Collection wall time was `2,028.164 s` with five ordered workers plus non-overlapping pre-materialisation workers; summed state runtime was `10,880.823 s`. Calibration, evaluation, ledger persistence, and the benchmark took `10.112 s`. The result checker independently reproduced 171 clauses without replay or learned inference.

This is development-only. The target is simulated disallowed contact; it establishes no material-hazard, human-safety, learned closed-loop, emergency-stop, or physical Go2 claim. Actual successor states are used. Correct abstention remains desirable in oracle-nonviable states, but repeated abstention still fails mission-progress requirements. Platform stopping parity and command-replacement acknowledgement remain unresolved.

No model training, experimental learned inference, fresh-panel collection, JEPA access, successor-predictor work, memory, novelty, routing, beacon capture, or navigation occurred. Frozen low-level route/lateral policies executed only as the authorised simulator plant during registered transition replay.
