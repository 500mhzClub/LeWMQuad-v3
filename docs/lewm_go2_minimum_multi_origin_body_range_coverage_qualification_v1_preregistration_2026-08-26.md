# Minimum multi-origin body range coverage qualification V1

Date: 2026-08-26
Experiment: `MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1`
Policy: evaluation-first, deterministic no-training execution, row-level evidence persistence, development-mode end-to-end execution.

Frozen executable contract: `docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_contract_2026-08-26.json`, 46,354 bytes, canonical LF-terminated file SHA-256 `94baafb6bb51ba2b444213b979e2fba17e3bf137c3201ea8b7c5a07a18811aee`, content digest `03e4dfb1f479017bf6ece8d1543216840effa2f675062bb04cb6afbd7f47d591`.

Frozen output schema: `docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_output_schema_2026-08-26.json`, 20,554 bytes, canonical LF-terminated file SHA-256 `7606d41346050fc2b60994b12ce57ddcf81f768fbab9d244b5306a8631c55444`, content digest `e2a9d1e0d50de1d3de2552ea4ed34c942d8951f6ab036766e3c50fcffaa83b1f`.

## Question and claim boundary

The completed `BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1` established `SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO`: neither the realistic assumed platform scan, dense nominal-FOV platform mount, nor one dense spherical body-centric origin retained usable two-ply actions. This experiment asks for the minimum prospectively selected two- or three-origin layout that can observe enough protected geometry to reproduce the exact per-link, two-ply micro-viability decision.

The target remains `H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT`, a simulated robot–environment contact/separation proxy over one committed 100 ms tick. Actual transition geometry is used only as a true-future observability upper bound. A pass would not show pre-action future prediction, deployment safety, emergency stopping, material-impact safety, injury or damage prevention, or learned closed-loop navigation.

No model is trained. No JEPA predictor, checkpoint, memory, novelty, navigation, routing, beacon-capture system, or untouched G2 evaluation is opened.

## Immutable predecessor and corpus

The direct predecessor is frozen at source commit `6fb55dec810b8fb8337d4519096f17a294c78425` and completed result commit `d9748abe0fad0a25face56801f6b0c5e699db92f`. Its result content digest is `98699ac43046a4d1f425998217d5527637209ca233a872cf667e484123a967ac`; its result-file SHA-256 is `8844c0ee5a8bcdd28f505d64a670b5dee595933af05a00290f327cb8f3702019`.

The repaired corpus remains byte-bound by:

- logical digest `e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223`;
- corpus-index SHA-256 `c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0`;
- action-contract SHA-256 `cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06`;
- repaired-row-ledger SHA-256 `63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94`;
- 176 frozen states, 29,470 current/successor transitions, 1,473,500 physics frames, 13 protected links, and 27 protected collision shapes;
- 128 training, 24 internal-calibration, and 24 development-held-out states.

State, role, transition, action, contact-label, protected-link, protected-shape, and H3 identities are immutable. Training data may choose mounts and layouts only through label-free geometric support. Calibration selects thresholds only. Development-held-out outcomes are opened only after layouts and thresholds are frozen. Untouched G2 is forbidden.

The predecessor's `EXACT_SENSOR_MATERIALIZATION_MAP` contains 13,584 exact-geometry representatives and may copy deterministic dense-condition materialisation only. It does not authorize realistic-scan reuse: every realistic condition independently renders every one of the 29,470 transition UIDs for every installed origin and evidence mode, because scan phase is UID-bound even when transition geometry is byte-identical. The decision/action map remains a separate 13,385-representative authority for action identity and set reduction.

Each state-phase receipt freezes the unique canonical transition UID at every transition index and a canonical SHA-256 digest of that ordered list. Every finite-scan receipt must match both the frozen index and UID before its deterministic scan phase is recomputed; reuse and terminal custody revalidate the same binding.

The predecessor `REALISTIC_PLATFORM_SCAN` and `DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT` conditions are regression references. Their bound scans, row evidence, thresholds, and metrics are reused without rematerialisation unless integrity validation proves reuse impossible; they are never reinterpreted.

## Sensor and housing assumption

No exact platform sensor is selected in the local BOM, so every origin uses the explicit development assumption `ASSUMED_GO2_HEAD_LIDAR_L2`, with `APPROXIMATED_REALISTIC_PLATFORM_SCAN` and required secondary classification `ASSUMED_SENSOR_CONTRACT`. Per origin the realistic approximation has 6,400 rays over 100 ms, 64,000 effective points/s, 360° azimuth, inferred −6° to +90° elevation, 0.05 m blind region, 30 m qualification range, exact deterministic point timestamps, and no noise or random dropout.

The official manual binds a 75×75×65 mm, 230 g L2 package. For conservative mechanical checks the complete housing is an axis-aligned box centred on the ray origin. Every supplemental origin keeps that full box outside the static trunk envelope by 0.010 m on a separating axis. This is an explicit development assumption, not a CAD-qualified deployment design.

For mechanical checks and ray tracing, each supplemental 75×75×65 mm envelope remains body-axis-aligned in the trunk frame, independent of the selected optical orientation; the optical ray frame rotates inside that frozen envelope approximation. Each origin's own envelope is ray-exempt as an optimistic emitting-aperture approximation. Every other installed supplemental envelope is a body-axis-aligned rigid trunk-frame box self-occluder. The head origin alone retains the predecessor geom-1/geom-2 emitting-host exemption; those head robot shapes remain occluders for every supplemental origin. Sensor housings never change the 13 protected links, 27 protected collision shapes, contact labels, or attribution authority. This mixes an optimistic own-aperture approximation with conservative other-envelope occlusion and requires exact CAD/aperture confirmation; it is not approved hardware.

Every realistic origin receives an independent deterministic phase:

`SHA256(namespace || 0x00 || raw contract digest || 0x00 || transition UID UTF-8 || 0x00 || mount ID ASCII)`.

The first two big-endian 64-bit words define azimuth and elevation phases. Phase derivation accepts no label, clearance, route, or outcome field. World-frame fusion uses each exact point timestamp and articulated pose.

No realistic scan is reused across transition UIDs or through a shared planning-boundary digest. Thus, for each realistic condition/mode, `sensor_scans = unique_rendered_scans = 29,470 × installed-origin count`, every scan contains 6,400 rays, and `render_cache_reused` is false. Dense conditions report analytic witness/acquisition support and may use the exact-geometry map; they do not claim a finite realistic-scan count.

The predecessor's witness-estimator compatibility rule is also frozen prospectively. Every realistic per-origin witness row is enriched with the label-free matched dense-L2-FOV target's nominal-FOV, direct-visibility, self-occlusion, environment-occlusion, near-blind, and support fields. A valid finite realistic same-object return within the frozen 0.10 m witness radius is inherited into that origin's matched dense-L2-FOV evidence. The nesting `REALISTIC ⊆ DENSE_L2_FOV ⊆ DENSE_SPHERICAL` must hold separately for every origin and after fusion, in both evidence modes; for every supported lower-representation query, the matched upper-bound clearance must also be no greater than lower clearance plus exactly `1e-9` m. Matched dense spherical evidence inherits any dense-L2-FOV support needed to enforce the second continuum inclusion. Because realistic phase is transition-UID-bound while dense evidence is copied through the exact-geometry map, the dense row for one exact-geometry copy group receives the union of finite inherited support from every member UID before the enriched dense row is copied consistently to every member. Matched dense evidence persists inherited flags and counts; every source realistic row retains its UID/mount/ray/time/range/object provenance. Every dual/three state receipt and phase index requires exactly one expected complete three-level chain with exactly both evidence modes and positive per-origin, fused, and clearance-pair counts. Each row persists support query/violation counts, clearance-monotonicity pairs/violations, and maximum clearance excess; any violation or missing row fails closed. The spherical-only diagnostic requires an empty dominance object. UID-group inheritance and both dominance assertions complete before threshold calibration, any gate or conditional decision, and classification. This is label-free estimator compatibility, not angular-grid tuning.

## Frozen origins

Body-frame transforms use metres and right-handed body-from-sensor rotation.

- `HEAD_STOCK`: the frozen stock `base→radar` transform, translation `[0.28945,0,-0.046825]`, RPY `[0,2.8782,0]`. It alone retains the predecessor's optimistic ray-only coarse head-housing exemption; protected contact geometry is unchanged.
- `REAR_TOP_TRUNK`: translation `[-0.1254,0,0.0995]`, the centre of the rear third of the trunk, with housing half-height 0.0325 m and 0.010 m clearance above trunk top `z=0.057`.
- `LEFT_UPPER_FLANK`: translation `[0,0.09425,0.0285]`, trunk longitudinal centre and upper-quarter height, with housing half-width 0.0375 m and 0.010 m clearance outside `y=0.04675`.
- `RIGHT_UPPER_FLANK`: translation `[0,-0.09425,0.0285]`, the exact sagittal mirror.

Leg mounts and any search outside these four origins are forbidden. Mounting tolerances, detailed mechanical integration, power, and final cable design remain unresolved. Hardware accounting must report sensor count, aggregate rate/rays, bandwidth, payload mass, power, envelope, and cable assumptions.

## Static orientation library and receipt

Each supplemental mount has exactly four candidate orientation IDs:

- `LEVEL`: sensor local +Z is body +Z;
- `INVERTED`: sensor local +Z is body −Z;
- `OUTWARD_DOWNWARD`: +Z is the normalized sum of the mount's horizontal outward vector and body −Z;
- `INWARD_DOWNWARD`: +Z is the normalized sum of the negative horizontal outward vector and body −Z.

The outward vectors are rear `[-1,0,0]`, left `[0,1,0]`, and right `[0,-1,0]`. Sensor +X is body +X projected orthogonal to the selected pole, with projected body +Y as the deterministic degeneracy fallback; +Y is `Z×X`. Canonical body-from-sensor rotation, scalar-first wxyz quaternion, and intrinsic XYZ RPY are persisted.

Before contract freeze, a payload-free static fixture selects one orientation per supplemental mount using only nominal URDF protected surfaces. It maximizes direct visible fraction, minimizes the self-occluded fraction among that origin/orientation's nominally FOV-and-range-eligible witnesses, then maximizes calf and rear-limb visibility, with fixed orientation-ID order as final tie-break. Zero nominally eligible witnesses fails closed. The same receipt verifies each complete body-axis-aligned supplemental housing against all 27 nominal protected primitives; each supplemental minimum is 0.010 m (within deterministic floating-point representation) against `base:00`, and every validation row passes. It reads no transition, contact, safe-action, H3, calibration, or held-out outcome. The exact selected transforms, scores, envelope frame, and clearance validations are frozen in `docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_mount_library_2026-08-26.json`; its canonical LF-terminated file is 16,462 bytes with SHA-256 `47028f5ca82e983995aac6dea989acba1dd049fcffb553bbbee464b788e5d7b8`, content digest `85683ac020d9fff0ec245ca9d2bf99a3942ae506f2c8c8bce7a49028a24392e6`, and static-witness digest `663de93b5cf840b7fe05788a1460702349094b24a882f7500c481e30d5a3f264` over 842 witnesses.

The frozen selections are stock head RPY `[0,2.8782,0]`, quaternion wxyz `[0.13131596830945733,0,0.9913405653290648,0]`; rear-top `INVERTED`, RPY `[pi,-0,0]`, quaternion `[0,1,0,0]`; left-flank `INWARD_DOWNWARD`, RPY `[2.356194490192345,-0,0]`, quaternion `[0.3826834323650898,0.9238795325112867,0,0]`; and right-flank `INWARD_DOWNWARD`, RPY `[-2.356194490192345,-0,0]`, quaternion `[0.3826834323650898,-0.9238795325112867,-0,-0]`.

## Candidate layouts and label-free selection

Every layout contains `HEAD_STOCK`. Pair candidates are exactly:

1. `HEAD_STOCK__REAR_TOP_TRUNK`;
2. `HEAD_STOCK__LEFT_UPPER_FLANK`;
3. `HEAD_STOCK__RIGHT_UPPER_FLANK`.

Three-origin candidates are exactly:

1. `HEAD_STOCK__REAR_TOP_TRUNK__LEFT_UPPER_FLANK`;
2. `HEAD_STOCK__REAR_TOP_TRUNK__RIGHT_UPPER_FLANK`;
3. `HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK`.

Pair and three-origin layouts are selected independently using all 128 training states and dense nominal-L2-FOV true-future raycasting support, before calibration or held-out access. One support unit is a protected-link/physics-step closest-environment witness. It is supported when at least one layout origin directly supports it.

Selection uses a separate `LABEL_FREE_LAYOUT_GEOMETRY_REUSE_MAP`, never the contact-validated `EXACT_SENSOR_MATERIALIZATION_MAP`. Reuse is permitted only within one deployable applied-action copy partition when `qpos`, `link_transform`, `geom_transform`, and the boundary-snapshot digest match exactly at zero tolerance. The map covers each transition in all 128 training states exactly once. Neither this reuse helper nor the layout objective consumes a contact/outcome field. The frozen corpus adapter necessarily reads and validates outcome arrays and metadata for custody, but they have no data path into selection. The receipt therefore records `outcome_fields_used_by_layout_objective=[]`, `contact_labels_used_for_layout_selection=false`, and `frozen_outcomes_read_only_for_corpus_custody_validation=true`, plus the reuse fields and aggregate representative count.

Candidate layouts are ranked lexicographically by:

1. highest minimum support over exactly `TRUNK`, `FRONT_LIMBS`, `REAR_LIMBS`, `HIPS_AND_THIGHS`, and `CALVES`;
2. highest fifth percentile of per-transition support over all 13×50 witnesses;
3. highest rear-limb support;
4. highest calf support;
5. highest overall mean support;
6. lowest layout self-occlusion;
7. earlier fixed layout ID order.

Layout self-occlusion is measured only over witnesses with at least one nominally FOV/range-eligible origin. Its numerator is unsupported witnesses for which every nominally eligible origin is robot-self-blocked. A zero nominal-support denominator fails closed. The selection objective uses no contact, safe-action, H3, calibration, or held-out outcome; frozen outcome arrays are read only by the corpus custody validator as disclosed above. A self-digesting selection receipt is persisted before calibration.

The complete training selection population is persisted at `layout_selection/training_layout_evidence.jsonl.gz` as one deterministic gzip JSONL row per training transition. Every row carries all six candidate layouts with mount IDs, supported/total witnesses, support and unsupported fractions, nominally eligible and all-origin-self-occluded counts, self-occlusion fraction, exact five-region counts/fractions, and protected-link counts/fractions for each of the 13 frozen links (50 witness steps per link per transition). The aggregate `candidate_metrics` likewise reports per-link supported/total counts and support fractions for all 13 links. `contact_label`, `safe_action_count`, and `route_outcome` are explicitly null. The selection receipt self-digestingly binds the ledger path, SHA-256, bytes, row count, training role, layout and region IDs, and empty objective outcome-field list.

## Evidence modes and multi-origin fusion

Every primary selected layout condition reports:

- `PLANNING_TIME_CAUSAL_CLOUD`: acquisitions at the current planning boundary only;
- `TRUE_FUTURE_OBSERVABILITY_CLOUD`: actual timestamped acquisitions throughout the 100 ms transition.

Only true-future results determine qualification and classification. A later learned system would have to predict future per-link geometry before execution.

Robot self-occlusion is evaluated separately at each origin against all exact articulated collision geometry. Self returns keep origin/link/shape identity but are excluded from environmental clearance. Geometry behind robot or environment occlusion is unsupported, never free. After exact motion compensation, fusion is the deterministic union of valid environmental points. It cannot average away a nearer return. Per-link clearance is the nearest supported clearance across origins. A witness is unsupported only if no origin supports it.

Every per-link row persists all supporting origin IDs and acquisition times, per-origin FOV/visibility/point counts/nearest points/ages, and the deterministic responsible origin and point.

## Conditions and prospective conditional flow

The two predecessor regression IDs are followed by these exact new conditions:

- `DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND`;
- `DUAL_DENSE_L2_FOV_UPPER_BOUND`;
- `DUAL_REALISTIC_L2_SCAN`;
- `THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND`;
- `THREE_DENSE_L2_FOV_UPPER_BOUND`;
- `THREE_REALISTIC_L2_SCAN`;
- conditional diagnostic `ALL_FOUR_DENSE_SPHERICAL_DIAGNOSTIC`.

Both regression references and all three selected-dual conditions are evaluated first in both evidence modes. If and only if `DUAL_REALISTIC_L2_SCAN` passes every true-future gate, all three-origin and all-four conditions are prospectively not run. Otherwise all three selected-three conditions run in both modes. If the selected `THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND` then fails, the all-four frozen layout runs as dense spherical true-future only.

The all-four diagnostic never runs a realistic scan, cannot replace the selected pair or triple, cannot become a primary condition, and cannot alter the primary classification. Its only possible classification effect is the secondary `FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED`.

## Calibration, two-ply decision, and gate

One threshold per executed condition/mode is selected only on the 24 internal-calibration states. Candidates are every unique finite calibration score plus exterior sentinels. Eligibility requires combined current/successor recall ≥0.95 and FNR ≤0.05. Eligible thresholds rank by negative retention, viable states retaining an action, correct nonviable abstentions, H3 progress, lower normalized regret, top-3, and finally the numerically larger conservative threshold. A threshold tie is contact-positive.

Two-ply admission requires a predicted contact-free current tick and at least one predicted contact-free unique deployable next action from the actual successor. Safe-next-action margins 1/2/3 are reported. Use of the actual successor is an observability assay, not a prediction claim.

The immutable true-future development-held-out gate is:

- current and successor contact AUC each ≥0.90;
- combined recall ≥0.95 and FNR ≤0.05;
- zero/nonzero safe-action accuracy ≥0.90 and false-nonzero rate ≤0.05;
- at least 18/20 oracle-viable states retain an action;
- all 4/4 oracle-nonviable states abstain;
- zero selected immediate contacts and zero selected nonviable successors;
- H3 progress ≥80% of exact geometry;
- normalized regret ≤0.20;
- best-admissible top-3 ≥0.75;
- no family collapse.

The gate cannot be weakened. Current/successor contact, safe counts, viability, route metrics, per-link, per-region, per-family, per-origin, fused coverage, and planning-time/true-future comparisons are persisted.

## Error attribution

Every frozen-threshold coverage error receives exactly one of:

`INSUFFICIENT_ORIGIN_COUNT`, `VERTICAL_FOV_LIMITATION`, `SCAN_PATTERN_SPARSITY`, `SCAN_TIMING_LIMITATION`, `ROBOT_SELF_OCCLUSION`, `NEAR_BLIND_REGION`, `MOUNT_POSITION_LIMITATION`, `POINT_FUSION_ERROR`, or `UNRESOLVED`.

The prospective evidence hierarchy is fusion loss; near-blind exclusion; all eligible origins self-occluded; matched-layout spherical support without nominal-FOV support; an existing realistic spatial ray available only after the event; dense nominal-FOV support without any realistic spatial ray; support only from a larger allowed layout; residual mount-position nonvisibility; unresolved. Ambiguity at any causal comparison level is conservatively `UNRESOLVED` rather than forced into an earlier class.

## Primary and secondary classifications

Exactly one primary classification is selected in this precedence:

1. dual realistic pass → `DUAL_ORIGIN_REALISTIC_RANGE_SIGNAL`;
2. otherwise triple realistic pass → `THREE_ORIGIN_REALISTIC_RANGE_SIGNAL`;
3. otherwise either selected dual dense condition passes → `DUAL_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK`;
4. otherwise either selected triple dense condition passes → `THREE_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK`;
5. otherwise → `MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO`.

The all-four secondary rule is exactly as bounded above. Other supported secondaries include planning-time limitation, self-occlusion, vertical-FOV, scan-density/timing, assumed-sensor, and `REPLANNING_INTERFACE_UNRESOLVED`.

## Compute and storage

The strongest executed selected dual or three-origin condition is benchmarked on CPU float32 over representative development-held-out states containing every family. Timed inputs are the already materialized float32 physics-step/protected-link clearance and observation-support witnesses, not raw ray clouds. The timed scope reduces those step/link witnesses into structured per-link state and the transition contact decision, then evaluates the complete current and successor action sets, safe-action counting, two-ply admission, threshold decisions, and H3 selection. After 30 warmups, at least 1,000 timed samples report P50/P90/P95/P99/max, 50/80/100 ms misses, RSS, and VRAM. Ray generation and future trajectory acquisition are excluded.

- P99 ≤50 ms and max ≤80 ms: `MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL`;
- otherwise P99 ≤80 ms and max ≤100 ms: `MULTI_ORIGIN_SET_REDUCTION_COMPUTE_POSITIVE_TENDENCY`;
- otherwise: `MULTI_ORIGIN_SET_REDUCTION_COMPUTE_NO_GO`.

The replanning interface remains unresolved regardless of this microbenchmark.

Output is rooted at `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/minimum_multi_origin_body_range_coverage_qualification_v1` on independent ext4 storage. Preflight requires at least 100 GB free there and 20 GB in the workspace. Temporary and final ceilings are 50 GB and 25 GB. Evaluation streams. New row ledgers are deterministic gzip JSONL; state shards are per condition. Regression evidence is bound by reference. Raw clouds are retained only for fixtures and the frozen hash-ranked audit subset.

The preserved `genesis_render_vulkan` environment is used without upgrade: Python 3.12.3, Genesis 0.3.14, NumPy 2.4.6, SciPy 1.17.1, CPU execution. TinyQuadJEPA and training packages are not required.

The runtime environment receipt must augment the predecessor closure with this experiment's contract, geometry core, metrics module, predecessor evaluator, new evaluator entrypoint, and `scripts` namespace, recording exact source paths and SHA-256 values. It fails closed if the required first-party modules are not loaded when preflight records the receipt.

## Fixtures, custody, and stop

Before scientific materialisation, deterministic fixtures must cover: clear full-body sweep; front, side, and rear trunk contacts; front-limb, rear-limb, and calf contacts; one-origin occlusion with another observing; complementary left/right and head/rear coverage; near-blind contact; between-scan contact; synchronized overlap; independent phases; one/zero safe successor; exact threshold tie; correct abstention; deterministic H3 ranking; and byte-identical regeneration.

The tracked fixture receipt embeds `fixture.core.raw_fixture_evidence` under schema `minimum_multi_origin_body_range_coverage_fixture_raw_evidence_v1`, with its own content digest. It retains every finite ray query used by the near-blind, between-scan-samples, and occluded-versus-observed fixtures, including origin, direction, timestamp, environment and robot primitives, physical first hit, and range-filter status. It also retains complete reduced per-origin evidence for every multi-origin union fixture and complete reconstructible inputs for the clear/contact/H3/safe/threshold-tie/scan-phase fixtures. This evidence is canonical JSON inside the tracked receipt, not a separate NPZ, and the fixture gate requires byte-identical regeneration of the complete receipt before scientific materialisation.

The run persists the static mount receipt, layout-selection receipt, conditional execution receipt, condition thresholds/metrics/errors, per-transition and per-link provenance, raw-audit manifest, hardware accounting, benchmark, result, report, and custody counters. Every per-state materialisation receipt, including its complete UID-bound scan receipts and inheritance counts, is byte/SHA-bound from the phase index and revalidated at reuse and terminal check. It stops after the conditional qualification and result commit.

It does not train, collect a panel, read G2, open JEPA or checkpoints, alter identities/labels/links/shapes, implement a predictor, or execute memory, novelty, navigation, routing, or beacon capture.
