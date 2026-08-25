# Body-centric range coverage qualification V1

Date: 2026-08-25
Experiment: `BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1`
Policy: evaluation-first, single deterministic no-training execution, row-level evidence persistence, development-mode end-to-end execution.

This is a new prospectively frozen experiment after `REPOSITORY_BACKUP_SPACE_AND_ENVIRONMENT_RECOVERY_V1`. The preserved recovery terminal remains `NOT_RUN_FAIL_CLOSED_NO_FROZEN_EXECUTABLE_CONTRACT`: recovery completed successfully, and the earlier proposed qualification did not run because no executable experiment contract existed.

## Question and boundary

This experiment asks whether a physically plausible body-relevant range sensor can observe enough scene geometry to reproduce the exact per-link two-ply micro-viability decision on the already frozen repaired corpus. The target is `H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT`, a simulated robot–environment contact/separation proxy over one committed 100 ms tick. It is not a material-impact, injury, property-damage, human-safety, platform-stopping, learned-navigation, or deployment-safety claim.

Current and actual-successor geometry are used as a true-future observability upper bound. A passing true-future result would not show that future geometry can be predicted before action execution. No model is trained and no JEPA predictor is opened.

## Immutable scientific inputs

- Source lineage: `10b3a190d506830e6a87e04a0f1c832b92295bd7`.
- Completed predecessor result: `034c2fb902997ac29e2742fc4ddc2c28ad1706b6`.
- Repaired corpus logical digest: `e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223`.
- Corpus-index SHA-256: `c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0`.
- Action-contract SHA-256: `cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06`.
- Repaired predecessor-ledger SHA-256: `63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94`.
- Geometry-index SHA-256: `67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f`.
- 176 states, 29,470 current/successor transitions, 1,473,500 physics frames; frozen 128/24/24 training/internal-calibration/development-held-out roles.

State, role, action, transition, contact, and H3 identities are immutable. The contact authority is the repaired `frozen_contact_label` bound by the repaired row ledger: the predecessor's authoritative physics-rate H1 target. Native-replay and history-free exact-query verdicts are reconstruction diagnostics and never replace that target. The frozen first-contact step is authoritative; link/object attribution uses native replay only when both verdict and first-contact step agree, then history-free exact query only under the same two agreements, and is otherwise `UNRESOLVED`. Training and held-out outcomes cannot select a sensor, mount, threshold, or implementation choice. No fresh panel is collected.

## Sensor and mounting assumptions

No exact range sensor is selected by the local platform BOM. The pinned third-party Gazebo xacro does include a generic L1 visual and synthetic `gpu_lidar` at `[0.25,-0.038,-0.03]`, but that backend is marked `pinned_pending_license_and_build_audit`, the platform `sensors_required` list does not select it, and the README explicitly drops LiDAR from the v3 corpus; it is simulation scaffolding, not an intended hardware selection. The realistic condition is therefore `ASSUMED_GO2_HEAD_LIDAR_L2`, with secondary classification `ASSUMED_SENSOR_CONTRACT`. Official Unitree L2 documentation binds 360°×96° wide/negative-angle coverage, a 0.05 m blind region, 30 m maximum range at 90% reflectivity, 64,000 effective points/s, 5.55 Hz circumferential frequency, 216 Hz vertical frequency, per-point relative timestamps, and point-cloud/IMU interfaces.

The proprietary non-repetitive phase sequence is unavailable. The realistic scan is prospectively classified `APPROXIMATED_REALISTIC_PLATFORM_SCAN`: 6,400 ideal/noiseless rays per 100 ms, deterministic 5.55 Hz azimuth and 216 Hz triangular elevation phase, inferred elevation interval −6° to +90°, and world-frame motion compensation using exact point timestamps. Horizontal and vertical phases are deterministically derived from the exact planning-boundary snapshot SHA-256 under a frozen namespace. Every candidate at one boundary shares the same phase; candidate action and successor endpoint are excluded. The trailing causal and forward true-future windows are phase-continuous at the boundary.

The nominal platform transform is the matching local Genesis and official Unitree `base→radar` joint: translation `[0.28945, 0, -0.046825]` m and RPY `[0, 2.8782, 0]` rad. It is a development nominal; physical calibration and installation tolerance are unquantified.

That nominal optical origin is inside the frozen coarse `base:01`/`base:02` head-housing collision primitives (URDF lineage `Head_upper`/`Head_lower`). For B and C only, geometry indices 1 and 2 are prospectively excluded from self-ray intersection as an `OPTIMISTIC_RAY_ONLY_COARSE_HOUSING_EXEMPTION`; both remain unchanged protected contact/clearance geometry, and every other robot primitive remains a self-occluder. This whole-coarse-housing approximation substitutes for unavailable aperture CAD and makes B/C optimistic upper bounds; it is not tuned from contact outcomes.

No central payload frame exists in the bound asset. The body-centric mount is selected without outcomes at the trunk collision-envelope top centre plus a prospectively fixed 0.010 m mechanical clearance: parent `base`, translation `[0, 0, 0.067]` m, level RPY `[0, 0, 0]`. This is an explicit development assumption, not a deployment design.

## Frozen conditions

Exactly four primary conditions are evaluated:

1. `CURRENT_SPARSE_RANGE_BASELINE`: 180 endpoint-exclusive azimuth bins, elevations −15/−5/+5/+15°, mount `[0,0,0.25]` m, range 0.05–10 m. Planning-time uses one deterministic sweep at the current boundary; true-future uses the deterministic union of current-boundary and actual-terminal sweeps, reproducing the predecessor timing contract. Primary results apply articulated self-occlusion; a non-primary legacy-no-self-occlusion regression is retained.
2. `REALISTIC_PLATFORM_SCAN`: the assumed L2 contract, nominal platform mount, 6,400 timed rays/tick, exact articulated self-occlusion and environment occlusion, ideal range on existing rays, no stochastic noise/dropout.
3. `DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT`: analytic continuum target-ray visibility with the identical L2 FOV, range, blind region, platform mount, and occlusion rules. It removes scan sparsity/timing without enumerating a finite angular grid.
4. `DENSE_BODY_CENTRIC_SINGLE_ORIGIN`: analytic continuum target-ray visibility from the single frozen body-centric origin, 360° azimuth and −90° to +90° elevation, identical range/blind/self-occlusion rules.

For dense continuum conditions, exact closest-environment surface witnesses are derived from scene and articulated geometry without contact labels. A witness is supported only when it lies in nominal FOV/range, outside the blind region, and an exact target-directed ray reaches it before any non-exempt robot or environment occluder. Planning-time uses the boundary acquisition only; A's dense attribution diagnostic preserves its two boundary acquisitions; C/D true-future use the accumulated union at `t={0,.002,...,.100}`. Full support is any acquisition; event-time support is separately limited to acquisitions no later than the witness event. Provenance selects the latest supporting acquisition at/before the event, else the earliest later acquisition. This is the deterministic infinite-angular-density limit at the prospectively frozen 50×13 witness estimator—not a complete visible-surface cloud and not proof that hidden geometry could not be inferred.

Condition C is additionally required to dominate B witness-by-witness: a valid B same-object return within the frozen 0.10 m target radius is itself a ray inside C's identical mount/FOV/range continuum, so it is inherited as flagged C local-surface support. Any B-support/C-failure inconsistency is fatal before evaluation.

## Evidence modes and occlusion

`planning_time_causal` generates observations only at the current planning boundary. The realistic condition uses a frozen preceding 100 ms scan phase with the boundary pose/configuration held fixed; this optimistic static-boundary accumulation avoids inventing an unavailable past trajectory. For offline scoring only, that causal cloud is reduced against the frozen actual 50-step candidate sweep and labels to ask whether current evidence covers the later protected body volume. Future geometry never enters causal point generation and is not deployable planning-time information. `true_future_observability` uses point timestamps and actual articulated poses during the committed 100 ms transition. Later points cannot establish pre-action prediction.

Robot rays cannot pass through frozen robot collision shapes except the exact B/C emitting-housing exemption above. Robot-self hit identity is retained for coverage auditing, but self points are excluded from environment clearance and geometry behind a self return is unsupported rather than free. The analytic ground plane remains a first-hit occluder and its counts are retained, but ground points are excluded from disallowed-environment clearance/support, matching the predecessor baseline's removal of rendered ground-plane points at `z <= 0.025 m`. Environment occlusion, range limits, timestamp age, and motion compensation are deterministic. Returns and the articulated sweep are reduced in the immutable world frame, which is rigid-transform equivalent to expressing both in the current-boundary body frame. Unsupported evidence is treated as risk, never as free space. Transition support is outcome-blind and fail-closed over the full protected sweep: if any protected-link/physics-step closest-surface witness is unsupported, the transition is unsupported risk independently of threshold. It may not be scoped after the fact to the oracle contact link.

Per transition and protected link the persisted reduction includes observed clearance, time to minimum, first threshold crossing, sector, support, nominal-FOV/direct-visibility fractions, unsupported swept-volume fraction, object/link attribution, nearest ray/point, and point age. The unsupported fraction is prospectively defined as the equal-weight fraction of the 50 exact physics-step closest-environment surface witnesses lacking condition-valid support; it is a witness-weighted swept-volume coverage estimator, not an exact volume integral. Complete row identities and global scores are persisted in JSONL; compact per-link arrays are stored by state. Raw clouds are limited to deterministic fixtures and the single lowest SHA-256-ranked transition within each `(role, family, transition_kind)` stratum, with the frozen namespace and canonical rank input recorded in the contract.

## Calibration, decisions, and gate

Each condition/mode receives one global-clearance threshold selected only on the 24 internal-calibration states. Candidate thresholds comprise both exterior sentinels and every unique finite calibration score. Eligibility requires combined current/successor recall ≥0.95 and FNR ≤0.05. Eligible thresholds are ranked lexicographically by negative retention, viable states retaining an action, nonviable abstentions, H3 progress, lower normalized regret, best-admissible top-3, then the larger/more conservative threshold. A threshold tie is contact and rejects the action.

Two-ply admission requires predicted current-tick contact-free status and at least one predicted contact-free next action from the actual successor. Unique deployable applied actions and frozen H3 ordering are used. The development-held-out gate is immutable:

- current AUC ≥0.90 and successor AUC ≥0.90;
- combined recall ≥0.95 and FNR ≤0.05;
- zero/nonzero safe-action accuracy ≥0.90 and false-nonzero rate ≤0.05;
- at least 18/20 oracle-viable states retain an action;
- all 4 oracle-nonviable states abstain;
- zero selected immediate contacts and zero selected nonviable successors;
- H3 progress ≥80% of exact geometry, normalized regret ≤0.20, best-admissible top-3 ≥0.75;
- no family collapse.

Only true-future modes determine the primary classification. Causal results are reported separately.

## Classification and stop rule

Primary classification is exactly one of `PLATFORM_RANGE_COVERAGE_SIGNAL`, `SCAN_DENSITY_OR_TIMING_BOTTLENECK`, `BODY_CENTRIC_MOUNT_REQUIRED`, `SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO`, or—only if the platform sensor cannot be specified—`RANGE_SENSOR_CONTRACT_UNRESOLVED`. Required secondary visibility and causal-limit classifications are reported where supported.

If and only if a condition passes, the next design may specify, but not train, `PER_LINK_CLEARANCE_PREDICTOR_V1`. It must predict per-link minimum clearance, first violation time, sector, observation support, and an uncertainty/lower-confidence bound from planning-time range, articulated state, one-tick action, and control history; contact and two-ply viability remain deterministic reductions.

The run stops after materialisation, per-link evaluation, attribution, benchmark, result persistence, and result commit. It does not train, access G2, open JEPA, implement a predictor, or execute memory/navigation/routing/beacon capture.

## Compute and storage

The output root is `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/body_centric_range_coverage_qualification_v1` on the verified independent ext4 filesystem. Preflight requires 40 GB output and 20 GB workspace free. Temporary and final ceilings are 30 GB and 20 GB. Evaluation streams states and reuses exact duplicate applied actions. The prospectively strongest condition, `DENSE_BODY_CENTRIC_SINGLE_ORIGIN` true-future, is benchmarked only after ray/future materialisation. Every timed sample reduces the complete 24-state held-out set, including all current actions and all associated next-action sets; the benchmark uses 30 warmups and 1,000 timed samples. P50/P90/P95/P99/max, 50/80/100 ms misses, peak RSS, and CPU-only zero VRAM are reported.

Execution uses the preserved `genesis_render_vulkan` environment at the compatibility path `.generated/venvs/genesis_render_vulkan` (resolved onto RecoveryStorage): Python 3.12.3, NumPy 2.4.6, SciPy 1.17.1, installed Genesis 0.3.14 and Torch 2.12.0+cu130. The evaluator imports neither Genesis nor Torch and performs no simulator step or training; their installed versions are custody bindings only. TinyQuadJEPA is not required.
