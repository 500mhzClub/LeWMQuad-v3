# Go2 JEPA Navigation Generalization Execution Contract

Date: 2026-07-09

Status: active

First-principles correction notice (2026-07-11): N32 qualification sequencing,
post-G2 model immutability, deployment-valid pose inputs, asymmetric physical
morphology, exact-path equivalence, cold-start/view-diversity semantics,
frontier viewpoints, reversible target beliefs, the heading-aware claim
contract, and the JEPA ablation requirement are consolidated in
`docs/lewm_go2_first_principles_plan_corrections_2026-07-11.md`, SHA-256
`b1c5e6087e4956a71cf048cccdd8408384305761a64d9405e08906fd84cc8042`.
The corresponding code-level G3-G5 gap audit and dependency order are recorded
in `docs/lewm_go2_g3_g4_g5_first_principles_gap_audit_2026-07-11.md`, SHA-256
`a6fd3d6c4c51c57b60470b0b6ef15e2e8554654c7b3777e70b34088a10054329`.

Superseding integrity notice (2026-07-10): the V4 sealed manifest was
accidentally byte-read by a read-only review search. It was not evaluated and no
sealed model output was produced, but its opacity contract is broken. V4 is
therefore development-only and permanently ineligible for G8. See
`docs/lewm_go2_v4_sealed_invalidation_2026-07-10.md`. A fresh opaque sealed
generation namespace and access-guarded one-shot launcher are required before
final evaluation. Historical statements below that V4 remained unopened are
retained as time-local evidence and are superseded by this notice.

## Objective

Build a scene-general navigation system whose deployed information flow is:

```text
RGB + odometry + proprioception
    -> spatial action-conditioned JEPA
    -> calibrated traversability and beacon observation heads
    -> persistent online belief map
    -> learned frontier-value exploration head
    -> deterministic graph search and local follower
    -> physically verified beacon claims
```

The primary objective is task completion on novel mazes. Safety is a constraint,
not a substitute objective: standing still is not a successful policy.

The final system must generalize coverage and claim all four beacons. A result is
not promoted because it looks better on one scene, because a proxy claim fires,
or because repeated deterministic seeds inflate the sample count.

## Execution Resource Policy

The current host has 32 logical CPU threads and two ROCm-visible devices with
strongly asymmetric memory capacity. GPU 0 has approximately 34 GB VRAM and is
the only device authorized for substantial JEPA or learned-head training. GPU 1
is the approximately 2 GB Raphael integrated GPU; it is not a training or model
inference lane. The existing N32 V4 smoke already failed on that adapter with
`hipErrorInvalidDeviceFunction` before model output, while the identical smoke
passed on GPU 0. GPU 1 may therefore be used only for non-data hardware
identification or an explicitly isolated backend-compatibility diagnostic.

Independent shard, scene, and synthetic-test work should use bounded CPU pools
where outputs can be merged in a frozen canonical order. Each worker must cap
BLAS/OpenMP-style numerical libraries to one thread so process-level
parallelism does not oversubscribe the host. Hashing, access-ledger mutation,
manifest finalization, immutable publication, and the one-shot sealed
evaluation remain ordered operations. Parallel execution must never change
sample order, seed identity, access scope, result bytes, or the number of
evaluation attempts.

## Non-Negotiable Runtime Contract

Allowed at deployment:

- ego RGB;
- measured or simulated deployment-equivalent odometry;
- proprioception and executed command history;
- learned model outputs;
- online state derived only from allowed observations and executed actions;
- a fixed platform and camera calibration;
- deterministic planning over the online belief state.

Forbidden for the learned-system claim:

- simulator occupancy, depth, object coordinates, or contact geometry;
- scene IDs, scene-specific routes, or held-out labels;
- oracle frontier choices or oracle target locations;
- fixed RGB color-mask geometry presented as learned perception;
- exact simulator pose unless the result is explicitly labeled an exact-odometry
  ablation;
- selecting a configuration from the sealed test set.

Privileged geometry is allowed only for offline labels, audits, and named oracle
positive controls.

## Current Evidence And Corrections

All existing `test_id` results, including `phase4_full18`, are development
evidence. Those scenes were used to diagnose and choose mechanisms and cannot be
the final test.

The independently rescored current development baseline is:

| Arm | Distance+LOS diagnostic events | Controller sightings | Median final coverage | Mean coverage AUC | 4/4 scenes |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 8/72 | 30/72 | 8.30% | 5.67% | 0/18 |
| novelty stack | 9/72 | 26/72 | 5.46% | 5.54% | 0/18 |

The old proxy figures were `16/72` and `13/72`. A distance-only recount reduced
them to `13/72` and `12/72`; the distance-plus-LOS scorer reduces them again to
`8/72` and `9/72`. These remain historical diagnostics because the final
canonical evaluator also requires the registered heading gate. Baseline
rejected events comprise five LOS failures
and three outside-radius events; novelty has three LOS failures and one
outside-radius event. Sightings remain controller confidence events, not
ground-truth visibility. The baseline-versus-novelty comparison also changed
primitive heads and routing together, so it does not identify a novelty-routing
treatment effect.

The ray-depth v0 branch has predictive signal but failed its closed-loop gate:
three deterministic probe scenes changed driven cells from 38 to 34, sightings
from 2 to 4, and claims remained 0. It also admits unsafe false-free geometry,
uses point-ray labels, and feeds a hand-written frontier override rather than a
learned exploration head. It remains an archived negative/intermediate result,
not the production design.

## Evaluation Units And Metrics

The independent unit is a scene. Repeated deterministic seeds are diagnostics,
not additional samples.

Primary metrics, in order:

1. fraction of scenes with all four beacons physically claimed;
2. physically verified beacon claim rate;
3. completion ticks for solved scenes;
4. reachable-area-normalized coverage AUC over the rollout;
5. sight-to-physical-claim conversion;
6. collisions, body-clearance violations, falls, stalls, and path failures.

A canonical physical claim requires all of the following at the claim tick:

- the requested target identity resolves to the claimed beacon;
- robot-to-beacon distance is inclusively within the fixed `1.20 m` radius;
- the zero-inflation physical scene has unobstructed line of sight to that
  beacon; and
- absolute robot-frame bearing to that beacon is inclusively at most `0.25`
  radians.

One shared evaluator supplies G0 tests, G5 labels, oracle acceptance, runtime
trace verification, development scoring, and sealed scoring. Controller or
proxy state-machine accepts are diagnostics only. Historical distance-plus-LOS
figures are not retroactively promoted without the heading-aware rescore.

Coverage is computed on one canonical configuration-space grid and divided by
the reachable free cells in the fixed-spawn component. Final coverage and AUC
are both reported. Euclidean path length and raw coarse-cell counts remain
diagnostics.

Results report scene-clustered uncertainty. Paired arms must differ by one
factor and use the same scenes, spawn, budget, target order, platform contract,
and non-treatment checkpoints.

## Split Policy

Three disjoint scene roles are required:

- `train`: may provide trajectories and privileged offline supervision;
- `development`: may be evaluated repeatedly for design and threshold choices;
- `sealed_test`: manifests are frozen before final model selection and evaluated
  once after all promotion gates pass.

The current 18-scene panel becomes development data. A fresh plan seed must
produce the sealed test. Scene IDs, topology seeds, and rendered frames must be
disjoint across all roles. Head validation is scene-disjoint, never a row-random
split of the same trajectories.

The sealed manifest may be hashed and audited at creation, but per-scene
performance and images must not be inspected until the final run. If the final
run exposes an implementation bug rather than a model failure, the correction
and rerun must be declared; it is not silently replaced by another test set.

### Authoritative benchmark freeze

The initial v1 freeze was invalidated before any sealed evaluation: development
materialization showed that `large_enclosed_maze` creates six landmarks by
duplicating red and blue, while `loop_alias_stress` creates two red and two blue
landmarks. Neither implements the deployed one-red/one-yellow/one-blue/one-green
task. No v1 sealed scene or performance was inspected. A v2 creation attempt
then failed closed on the same task check before writing a commitment.

The authoritative v3 freeze was created from candidate plan seed `2026070905`
and split seed `2026070906`, excluding 302 scene IDs already present in training
or closed-loop result artifacts. Every selected scene is a fresh
`medium_enclosed_maze` topology with exactly one landmark per task color.

| Role | Scenes | Task contract |
| --- | ---: | --- |
| train | 138 | four unique colors, exact-claim valid |
| development | 24 | four unique colors, exact-claim valid |
| sealed test | 30 | four unique colors, exact-claim valid |

The v3 sealed scene list has not been inspected. Its commitment is
`2bff51d6a6754316d23de31414b1428822eeafdba2f8bcf4131455448a90fa04`.
Swept-footprint calibration changed scene eligibility, so v3 is now frozen as
a simulation-only development benchmark under its 0.20 m circular proxy. Its
sealed commitment is invalid for the final platform claim and will never be
evaluated. This decision was made from development geometry only: the sealed
scene list and performance remain unopened.

The observed-max directional gait envelope plus 0.03 m margin has a maximum
vertex radius of 0.461771 m. On the 24 visible v3 development scenes, only
14/24 fixed spawn poses are clear and only 12/24 scenes make all four beacons
reachable under the exact polygon. This is a benchmark-contract failure, not a
learned-policy result. A fresh v4 split therefore uses a new deployment maze
family and new plan seed; it must pass both a yaw-invariant 0.47 m disc audit
and exact actual-yaw polygon/SE(2) eligibility before freezing.

The authoritative v4 freeze now satisfies that contract. It was created from
unprobed candidate plan seed `2026070923` and split seed `2026070924`. All 192
candidates passed both the 0.47 m disc audit and exact actual-yaw polygon SE(2)
audit; none were filtered. The resulting roles are 138 train, 24 development,
and 30 sealed test scenes. Their opaque scene-hash set has zero intersection
with the separate 24-scene physical-authoring probe. The v4 sealed manifest has
not been opened or evaluated.

## Canonical Geometry Contract

One versioned contract must define:

- swept robot footprint and safety margin;
- base and camera frames, camera extrinsics, FOV, and render jitter policy;
- planning cell size and connectivity;
- fixed spawn semantics;
- body-inflated free space and reachable component;
- line-of-sight and beacon standoff geometry;
- physical claim radius;
- kinematic collision and progress semantics.

The initial values must be derived from the platform manifest and an empirical
swept-footprint audit. Existing incompatible values (0.12 m planning inflation,
0.20 m validation inflation, 0.24 m label half-width, and zero-inflation ray
labels) are not allowed to coexist in promoted artifacts.

Geometry v2 makes the orientation abstraction explicit:

- strict collision and final feasibility use the 24-plane
  `go2-directional-observed-max-margin-v1` polygon at actual yaw;
- local occupancy labels and global planning use a 0.47 m yaw-invariant disc
  that encloses the polygon, so a 2D confirmed-free cell is safe at every yaw;
- q99 is diagnostic only because it discards the upper one-percent gait tail
  without improving v3 task eligibility;
- hardware measurement/controller validation remains required for G7, even
  though simulation labels and benchmark authoring are now unambiguous.

## Target Architecture

### Spatial predictive backbone

Use `SpatialLeWorldModel` or a successor with ordered patch tokens,
action-conditioned future prediction, stop-gradient or EMA targets, and explicit
anti-collapse regularization. A randomly initialized encoder trained only on
privileged geometry classification is a supervised perception encoder, not the
JEPA result.

After G2, auxiliary navigation heads consume detached shared features and must
not fine-tune the qualified encoder or traversability head. Before G2, any
joint fine-tuning retains the predictive and anti-collapse objectives. A
post-G2 encoder or traversability change is a new perception candidate and
requires preregistered selection/calibration plus a fresh eligible untouched-G2
role; the original one-shot G2 result cannot be inherited. Every checkpoint
stores the backbone hash, training scene IDs, corpus-plan hash, image geometry,
and loss configuration.

The predictive and physical heads are parallel consumers of one encoder; the
runtime physical head may use only current/past deployment observations. A
matched development ablation with JEPA losses disabled, and all other
architecture, data, initialization, and budget choices fixed, is required
before attributing a generalization improvement specifically to JEPA training.

### Traversability perception

Predict calibrated observable physical FREE/OCCUPIED/UNKNOWN evidence in an
egocentric local grid from spatial tokens. Labels use the recorded camera pose,
full rectilinear frustum, and first rendered-surface intersections. They never
inject a hidden configuration-space obstacle into a per-frame target. Fuse this
physical evidence across views in `OnlineBeliefMap`, then apply the fixed 0.47 m
configuration-space morphology for planning. This separation lets two partial
views jointly certify a footprint and keeps robot geometry out of the learned
visual target. A conservative depth quantile is allowed as an interim ablation,
but hard scalar mean-depth rays are not the final map input.

### Online belief map

Use one shared training/runtime implementation containing:

- reversible occupancy evidence and uncertainty;
- physical traversal and collision evidence;
- visit time/age and coverage state;
- pose and pose uncertainty;
- per-color target belief distributions and claim state;
- versioned serialization and feature export.

Contradictory evidence must be recoverable. Physical traversal clears false
obstacles. A single predicted free observation cannot permanently admit a path.
Planning is restricted to the connected confirmed-free component; exploration
must not fall back to optimistic traversal through unknown cells.

Learned physical evidence, verified swept traversal, and body-center
execution/contact blocks are separate semantic layers with independent
provenance. Near-identical camera views do not count as independent FREE
evidence. Raw physical cells cannot be planned over: every connected component,
frontier, and route consumes an immutable configuration snapshot derived by the
registered asymmetric post-fusion morphology. Stable initial stance and
successful motion certify only their measured actual body support/sweep, never
the larger yaw-invariant planning disc. A frozen yaw scan is the only
perception fallback when no safe route exists; otherwise a separately recorded
deployment reset-clearance certificate is required. FREE projection must remain
fully supported for every transform in the admitted pose/camera uncertainty
set, while OCCUPIED uses the union supercover; over-uncertain frames fail closed.

### Exploration head

Generate reachable frontier-viewpoint candidates deterministically, then learn
a map-conditioned value for each `(configuration_cell, yaw)` candidate or
frozen scan sequence:

```text
expected new coverage + expected beacon discovery
    - path cost - turn cost - collision risk - uncertainty cost
```

Train first on oracle future gain labels across scene-disjoint mazes, then use
DAgger on states visited by the learned perception and exploration stack.
Deterministic A* (or equivalent) and the local follower remain outside the head.

### Beacon observation and claiming

Predict per-color presence, bearing, range distribution, and uncertainty from
spatial features. Fuse positive and negative observations into reversible,
multimodal per-color spatial beliefs with competing hypotheses, age, and
uncertainty. Train the claim head on scene-disjoint data with the canonical
correct-target, distance, physical-LOS, and bearing acceptance contract.
Controller-declared and evaluator-verified claim state remain separate. Remove
analytic RGB masks and exact-vector propagation from the final learned claim.

## Immutable Promotion Gates

Threshold changes require a dated amendment before looking at the candidate
result. A failed gate leads to diagnosis and iteration, not post-hoc metric
replacement.

### G0: benchmark integrity

- fixed-spawn/body-inflated/LOS/claim-radius audit passes every selected scene;
- train, development, and sealed-test scene hashes are disjoint;
- the shared correct-target/distance/LOS/bearing physical claim evaluator and
  normalized coverage are unit tested at inclusive boundaries;
- every result records the full geometry, split, code, checkpoint, and corpus
  provenance.

### G1: oracle end-to-end ceiling

With exact occupancy and pose passed through the proposed planner/follower and
true claim contract, at least 17/18 current development scenes must finish 4/4
within 2,400 ticks. Target is 18/18. Failure blocks learned perception work on
that planner because it identifies a planning, following, or claim defect.

### G2: traversability perception

On untouched scene-disjoint validation scenes:

- admitted observable physical-FREE precision >= 99%;
- directly observable physical-obstacle recall within 2 m >= 95%;
- useful observable physical-FREE recall >= 90%;
- UNKNOWN/KNOWN and FREE/OCCUPIED-given-KNOWN calibration metrics and threshold
  selection are stored;
- deterministic post-memory 0.47 m morphology is unit-tested against exact
  physical evidence. Route collision and fused configuration recall are scored
  at G3 after multi-view fusion, not against a single-frame target.

#### G2 dataset-v2 preregistration

The v1 pilot is invalidated before model selection. Its replacement is frozen
before inspecting any new labels or model outputs:

- 96 source scenes, exactly 12 from each of eight rendered navigation/stress
  families, with v4 development and sealed commitments excluded before source
  artifacts are opened;
- within each family, a label-independent hash rank assigns nine training,
  one checkpoint-selection, one probability-calibration, and one untouched G2
  scene;
- at most 64 complete 0.5-second primitive windows per scene, selected only
  after all six recorded poses and every adjacent segment pass geometry-v2
  0.47 m configuration-space validity;
- RGB visibility is raycast through zero-inflation physical occupancy, while
  FREE/OCCUPIED targets come from the 0.47 m body-inflated configuration space;
- selection and calibration roles must each contain all three classes;
  calibration requires at least 10,000 FREE and 1,000 OCCUPIED supervised
  cells, and at least 90% of rows must contain a nonempty next-observed mask;
- the untouched G2 role is not loaded for checkpoint selection, probability
  calibration, threshold selection, or data-remediation decisions;
- occupancy optimization uses equal-capacity UNKNOWN-vs-KNOWN and
  FREE-vs-OCCUPIED-given-KNOWN terms. Natural class priors are retained for
  held-out probability calibration; promotion calibration permits no synthetic
  balancing and no rare-class backfill.

The first run is a two-epoch development-only wiring smoke. If it passes data,
calibration, and serialization checks, the first candidate is a 20-epoch
development-only run. Row count increases to 128 per scene only if the
checkpoint-selection learning curves remain data-limited.

#### 2026-07-10 implementation-conformance amendment

The split, dataset, thresholds, and gates above remain unchanged. The original
two-, 20-, and 60-epoch executions are invalidated because the trainer selected
its three-class fallback instead of the registered equal-capacity hierarchical
objective. Execution therefore restarts at the two-epoch wiring smoke; the old
60-epoch debug curve cannot trigger the 128-row rebuild.

Every subsequent candidate checkpoint and report must record:

- `occupancy_training_objective.mode == hierarchical_equal_capacity_v1`;
- resolved counts, inverse-count weights, and the 0.5 coefficient for each of
  UNKNOWN-vs-KNOWN and FREE-vs-OCCUPIED-given-KNOWN;
- `three_class_weights: null`;
- requested and effective deterministic-execution state; and
- role-scoped provenance proving that development runs opened only train,
  checkpoint-selection, and probability-calibration artifacts.

The corrected strict-determinism probe failed closed because ROCm/PyTorch has
no deterministic `grid_sampler_2d_backward_cuda`. The corrected seeded smoke
then passed wiring twice without G2 contact; it is explicitly best-effort and
not bitwise deterministic.

#### 2026-07-10 camera, renderer, and observability amendment

Dataset v2 and its projective smoke are retained as negative/debugging evidence,
but are invalid for promotion. This amendment was made before a 20-epoch
projective run and before any untouched-G2 model output was read. The numerical
99/95/90 perception thresholds and label-independent 96-scene family roles stay
fixed; their target semantics are corrected from hidden configuration space to
observable physical evidence as stated in G2 above.

The first-principles audit found three independent impossibility defects:

- v03 rendered square 224x224 RGB while passing horizontal 78.323 degrees to
  Genesis' vertical-FOV API. The actual source had H=V=78.323, not the platform
  H=78.323/V=62.8370386364 pinhole;
- v03 omitted `visual_randomization.distractor_objects`, although they are real
  collision objects in geometry-v2, so visual-stress labels could change while
  RGB stayed identical;
- v2 raycast only to cell centers, then injected globally known 0.47 m inflated
  occupancy. On selection scenes, 34.15% of FREE and 49.09% of OCCUPIED support
  extended outside the horizontal frustum. A per-frame configuration target also
  prevents memory from combining complementary partial views.

The correction keeps the exact 96 scenes, 72/8/8/8 roles, and 5,641 transition
identities, but sparsely rerenders only their 10,311 endpoint frames. V04 uses
224x168 H=78.323/V=62.8370386364 RGB with no runtime crop, renders walls,
obstacles, landmarks, and distractors, applies full roll/pitch/yaw object
transforms, and content-addresses every image, frame set, object set, camera,
manifest, and renderer source. Training target
`observable_physical_occupancy_v3` uses visible ground evidence and conservative
first-surface obstacle witnesses; hidden or geometry-disputed support is UNKNOWN.
The fixed 0.47 m morphology is applied only after physical evidence is fused in
memory.

The primary smoke therefore uses the center projective-column prior with the
corrected camera. The footprint-projective implementation remains a tested,
parameter-neutral ablation but is not the physical-target default. Checkpoint
selection mode `physical_occupancy_ceiling_v1` may rank only physical evidence;
legacy planner/configuration threshold sweeps are forbidden for this smoke. No
longer run, row-count increase, or G2 read is licensed until the new two-epoch
development smoke passes data, provenance, calibration, and learning wiring.

### G3: fast closed-loop coverage

G3 is not executable until development-candidate checkpoint-v5 loading, fused physical evidence,
post-fusion 0.47 m configuration morphology, and the V4 closed-loop action
trace are implemented in the shared runtime. Before evaluating a learned arm,
rerun the privileged-target G1 oracle through the exact physical-to-configuration
path at 2,400 ticks, then freeze a separate no-beacon-anchor 600-tick exact-map
coverage reference, a hash-selected scene-disjoint V4 panel, and the corrected
learned-perception baseline. On that panel:

- an exact zero-inflation physical-evidence backend must pass through the same
  origin-aware fusion, separate verified-traversal/execution layers, asymmetric
  FREE/OCCUPIED morphology, immutable configuration snapshot, frontier, and A*
  used by the learned arm;
- that path must match canonical configuration geometry scene-wide on all 24
  development scenes. The separate G1 regression must retain 96/96 claims;
  the no-anchor G3 reference is scored only on coverage and visibility
  opportunities;
- spawn connectivity must come from a recorded 0.47 m trusted-reset clearance
  certificate or complete physical evidence after bootstrap scanning. Stable
  stance/traversal alone certifies only its measured actual footprint, never the
  larger planning disc;

- a dated pre-output binding freezes the visibility-opportunity evaluator and
  four-beacon denominator, corrected baseline/config/source hashes, confidence
  level, paired scene-cluster interval method/resample count/seed, and numeric
  collision/fall/stall/route-failure tolerances;

- median normalized coverage is at least 2.0x the corrected baseline and at
  least 70% of the 600-tick exact-map oracle median;
- median ground-truth beacon-visibility-opportunity coverage is at least 70% of
  the 600-tick exact-map reference, and no scene with a nonzero exact reference
  may collapse to zero opportunities;
- the lower scene-clustered confidence bound is above no improvement;
- ground-truth beacon-visibility opportunity coverage and final/AUC coverage are
  both reported, with no topology/area/density stratum collapse;
- collisions, falls, and stalled-route fraction do not regress beyond the
  declared safety constraint;
- no scene-specific thresholds or optimistic unknown routing are used;
- current-frame-only versus persistent-fusion and perception-disabled versus
  exact-map engineering-reference ablations isolate the memory contribution.

### G4: learned frontier value

- before learned output, freeze the deterministic baseline's candidate
  generator, yaw/scan set, reachability filter, gain horizon/observation model,
  value weights/normalization, and lexicographic tie breaks; learned and
  baseline arms use the same candidate set;
- candidates are reachable `(configuration_cell, yaw)` viewing poses or frozen
  scan sequences, never unorientated cells or unknown-space goals;
- held-out frontier-viewpoint ranking beats distance-only, random reachable,
  and deterministic information-gain baselines on oracle future
  coverage/discovery labels;
- the learned head improves 600-tick normalized coverage over deterministic
  information-gain frontier selection;
- DAgger closes the model-visited-state gap without scene leakage.
- at 2,400 ticks on all 24 V4 development scenes, ground-truth evaluation
  records at least one valid visibility opportunity for 96/96 beacons and all
  four opportunities in 24/24 scenes before target conversion is promoted.

### G5: target conversion

On scene-disjoint development rollouts:

- learned per-color observation precision >= 99% and recall >= 95% over
  ground-truth visibility opportunities, with bearing/range uncertainty
  calibration reported;
- per-color memory retains reversible positive and negative evidence, competing
  spatial hypotheses, age, and uncertainty; false-track creation, persistence,
  and recovery are reported per episode;
- confirmed-target-to-valid-claim conversion >= 90%;
- false physical claim accepts < 1%;
- the learned observation/belief stack replaces fixed RGB masks and privileged
  target geometry;
- labels, controller decisions, oracle acceptance, and evaluation use the same
  inclusive `1.20 m` distance, physical LOS, and `0.25 rad` absolute-bearing
  contract, with correct requested target identity;
- internal controller-declared and ground-truth-verified claim states remain
  separate in memory and traces;
- oracle-coverage runs isolate the target stack and finish 4/4 on every V4
  development scene before joint scoring. Suppressing sightings cannot improve
  the denominator.

### G6: full simulation promotion

Across at least 24 development scenes at 2,400 ticks:

- the promotion target is 96/96 physically verified beacon claims and 24/24
  scenes finishing 4/4; a lower result is diagnostic, not a pass;
- zero falls, zero accepted false claims, and no collision regression beyond
  the preregistered physical tolerance;
- no topology, area, obstacle-density, or spawn-to-beacon-distance stratum has
  a hidden collapse masked by the aggregate;
- scene-clustered intervals, normalized coverage, completion time, collisions,
  stalls, and oracle gap are reported.

### G7: deployment robustness

- under preregistered calibrated odometry noise and deployment-equivalent
  locomotion, claim rate remains >=95%, 4/4 completion remains >=90%, and the
  paired drop from exact-pose G6 is no more than five percentage points;
- simulator-only geometry guards are disabled;
- action-source tracing proves the runtime contract on every tick;
- a physical smoke protocol passes before multi-maze physical evaluation.

### G8: sealed final evaluation

Freeze code, geometry, thresholds, and all model hashes after G7. An atomic
launcher binds that freeze, records access, and permits one sealed execution.
The target is 120/120 physically verified claims and 30/30 scenes finishing
4/4; anything lower is reported exactly and does not satisfy this plan. All
development results remain labeled as development evidence.

## Ordered Execution

1. Check in this contract and an evidence ledger.
2. Repair fixed-spawn validation, strict success accounting, coverage metrics,
   scene-disjoint manifest generation, and artifact provenance.
3. Establish the canonical geometry contract.
4. Build the oracle end-to-end positive control and pass G1.
5. Extract and test the shared online belief map.
6. Train the genuine spatial predictive JEPA and traversability head; pass G2.
7. Fuse perception conservatively and pass G3.
8. Train the map-conditioned exploration head and pass G4.
9. Train and integrate beacon observation, target belief, and claim heads; pass
   G5.
10. Iterate jointly until G6 passes.
11. Add localization and locomotion realism, then pass G7.
12. Freeze and run G8 exactly once.

No full 18+ scene learned sweep is justified between failed offline/fast gates.
The next experiment must target the measured failure class.

## Evidence Ledger

| Gate | Status | Evidence | Next blocking action |
| --- | --- | --- | --- |
| G0 benchmark integrity | PASSED FOR DEVELOPMENT CANONICAL ACCOUNTING / FINAL SEALED STILL PENDING | Type-exact canonical bytes, external scene-manifest recomputation, five manifest-bound promotion checkers, three patched wrappers, full semantic directional-policy preload binding, and independent finalization pass 394 synthetic tests. The immutable development artifact `canonical_physical_claim_v1_report.json` (file SHA-256 `4093461d842d926d4d351d84dec3bd8dff8a828f8730ef3b78c4a11aadfaee03`) finalizes exactly 96 raw attempts/evaluations/credits plus 96 eligibility witnesses with zero evaluator feedback or forbidden payload opens | Preserve the frozen evaluator/source/result identities through G2-G7; the remaining G0 scope is the one-shot sealed launcher/finalizer at G8 |
| G1 oracle ceiling | PASSED: 24/24 SCENES AND 96/96 CANONICAL PHYSICAL CLAIMS | The canonical development regression finalized 24/24 scenes, 96/96 accepted and credited oracle task objects, 96/96 accepted and credited eligibility witnesses, 24/24 `OnlineBeliefMap` routed scenes, zero rejected/unverifiable/duplicate credits, zero stalls/collisions, and zero actual-yaw directional-polygon collision segments. Evidence: `lewm_go2_canonical_physical_claim_oracle_regression_result_2026-07-11.md` | Treat the planner/follower/claim ceiling as closed; do not change it to compensate for later learned perception, memory, exploration, or target failures |
| G2 traversability | IN PROGRESS / DYNAMIC GEOMETRY PASSED / LEARNED HEAD PENDING | The independently finalized full-label audit scored all 1,310,720 cells and all 129,021 known occurrences. Level-center missed 373, static cell-square missed 4 OCCUPIED cells, and full-quaternion/yaw-aligned dynamic cell-square supported 129,021/129,021 with zero misses in every family. Evidence: `lewm_go2_n32_dynamic_cell_square_geometry_result_2026-07-11.md`, result SHA-256 `ace9b39c4be31fad84eb7bc2aa65c584acec04febb638672fbcead0db4b6b4fe` | Freeze the deployment-valid attitude sidecar/runtime input contract, implement the dynamic Cartesian JEPA lift, run the two-epoch development smoke and registered training/selection/calibration sequence on GPU0, then evaluate the untouched G2 role once against the unchanged 99/95/90 gates |
| G3 fast coverage | NOT YET RUNNABLE ON PHYSICAL V4 | No checkpoint-v5 runtime, two-layer physical/configuration map, asymmetric post-fusion morphology, exact-physical equivalence control, or nonprivileged G3 runner exists. The V3 fast8 result is historical only | Pass G2; prove exact physical evidence -> two-support morphology -> snapshot/frontier/A* equivalence and cold start; build the isolated v5 runtime/runner; then freeze the V4 panel and gates |
| G4 frontier value | NOT STARTED | Current frontier is heuristic and cell-only | Build viewing-pose candidates, oracle coverage/discovery labels, deterministic information-gain baseline, and learned scorer |
| G5 target conversion | NOT STARTED ON THE PROMOTED ARCHITECTURE | Legacy fixed RGB masks, irreversible point target state, and the row-random distance-only claim head cannot satisfy the learned, scene-disjoint physical contract. The shared physical evaluator implementation exists, but caller requalification is reopened by the G0 canonical-byte equality defect and the learned observation/belief stack does not exist | Finish G0 caller requalification; build reversible per-color belief distributions plus presence/bearing/range/uncertainty heads; score them through the canonical evaluator; and pass visibility-opportunity and oracle-coverage isolation gates |
| G6 full simulation | FAILED | 0/18 scenes solved and 8/72 historical distance+LOS diagnostic claims; 13/72 was distance-only. Neither is the final heading-aware claim score | Complete G0-G5 and rerun under V4 canonical accounting |
| G7 deployment robustness | NOT STARTED | Kinematic exact-pose evidence only | Calibrated noise, locomotion, physical smoke |
| G8 sealed final | BLOCKED / V4 INVALIDATED | V4 manifest opacity was broken by an accidental metadata search before model evaluation; no sealed model output exists | Create a fresh opaque, disjoint sealed commitment with an enforced access guard after G7, then freeze and execute once |

## Required Artifact Provenance

Each generated dataset, checkpoint, and result must contain or reference:

- schema/version and creation timestamp;
- git commit plus dirty diff hash;
- command/config with defaults resolved;
- corpus plan and split-manifest hash;
- exact train/validation/development scene IDs;
- geometry-contract hash;
- source-renderer camera audit hash and exact training/runtime projection and
  preprocessing contracts;
- observable-physical label schema, rendered-object parity, and post-memory
  configuration morphology support;
- encoder and parent-checkpoint hashes;
- random seed and whether it affects execution;
- resolved occupancy-objective mode, term coefficients, class counts, and
  class weights;
- requested and effective deterministic-execution settings;
- runtime input contract and action-source counts;
- raw per-scene metrics sufficient to recompute every aggregate.

Artifacts lacking this contract may guide debugging but cannot satisfy a gate.

Current benchmark identity:

- geometry contract SHA-256:
  `551b0938cab9018aca34fccad21e42b9e8cad93e7dc3d8904c77e9d9e7368345`;
- candidate plan SHA-256:
  `f7a350fd4238091a36a2a1ac93fce2182448ab2102510bb408687c4615d4677c`;
- development manifest file SHA-256:
  `92349a8a93e627b3cc3cb193fd9176eb4f290427c8c5889dc3849b91846fb806`;
- sealed manifest file SHA-256:
  `d952f15f1b35a8e052662fb2004728a71b42b1c738f00afe92c49ba8f15548aa`.

These v3 identities are retained for simulation evidence only. Geometry v2
identity and physical-policy evidence are:

- geometry v2 semantic SHA-256:
  `e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca`;
- geometry v2 file SHA-256:
  `e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52`;
- directional policy content ID:
  `c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc`;
- directional policy file SHA-256:
  `750d8afe47ee3edd5988cdea443f19703efad7a3266218932671b9fdfbe43828`;
- v3 development directional audit file SHA-256:
  `7d1c682d166f26e0d3f901e34e9b20309e571c608c5ea735e5eb5c9a236d3851`;
- observed-max result: 14/24 spawn-valid, 12/24 fully claimable,
  48/96 claimable beacons, 2,688/2,688 feasible straight topology segments,
  and 3,100/3,636 feasible endpoint-valid staged turns.

Authoritative v4 benchmark identity:

- candidate plan seed / SHA-256: `2026070923` /
  `0d39c62bb6c70b5143f341f82d45ada5c4ef0f4733878ac07f3d6518f64cd4b1`;
- split seed: `2026070924`;
- development manifest file SHA-256:
  `563f240a023309af42a05a9a8f29008f02a0629dee9f77f03568f779d1166d41`;
- role commitment content / file SHA-256:
  `12f203f6dc03dd2f0ed76067075baaac026d9b2843d9471962a9364514bf0cc7` /
  `82c4a9a382452031febb712faaa90bb52c8fc5e2fab2a33d8a7ea2d447413b75`;
- development screening file SHA-256:
  `15e3b30cd10902664f2a207fc93084dd8b474df3d5028ee3b07b2e986408b344`;
- sealed screening file SHA-256:
  `db118ca435877b06dfe2666b681e3313d0c5c120ed2a316853413469a5a0f103`;
- sealed test commitment:
  `d2dcbcc5444f0046be41311c0127943b63c2485c39b69844ed662f79aa13fef7`;
- creation report file SHA-256:
  `f8cfc19fe97eb1f46a3fa514b13d2b69ac50f34ebade1afbe78dd4b08222702d`;
- physical eligibility: 192/192 disc, 192/192 exact polygon, zero
  exclusions; the sealed manifest remains unopened.

Oracle G1 evidence:

- artifact: `.generated/oracle_positive_control/development/report.json`;
- SHA-256:
  `1aef98620f62a089ebd18b5424db843a8947bf47170854f527f92ee3c8b62aff`;
- result: 72/72 true claims, 18/18 all-four scenes, 17/18 at or above
  90% reachable coverage, zero collisions and zero stalls.

V3 oracle evidence:

- artifact:
  `.generated/oracle_positive_control/generalization_v3_development/report.json`;
- SHA-256:
  `0935e1ad2d8628cf88d5c37eed1a9dfaf1b01009a0a6d787743a29ec78cc2877`;
- result: 96/96 true claims, 24/24 all-four scenes, 19/24 at or above
  90% reachable coverage, zero collisions and zero stalls.

Authoritative v4 oracle evidence:

- artifact:
  `.generated/oracle_positive_control/go2_generalization_v4_development/report.json`;
- SHA-256:
  `7c0a63bb0548fee81918df22b227adec43d4bdc824875ef447793ef4f99d97a5`
  (the final rerun of the identical command re-emitted the report with an
  updated embedded dirty-diff identity; all result numbers are unchanged);
- materialization artifact:
  `.generated/scene_corpus/go2_generalization_v4/materialization_both.json`;
- materialization SHA-256:
  `a52bd82cb501481707d518d1fffd86e5475b440332f7d226586ebda47e6b1415`;
- result: 96/96 true claims, 24/24 all-four scenes, all 24 routed
  through `OnlineBeliefMap.shortest_path`, zero stalls, zero center-grid
  collision attempts, and zero actual-yaw observed-max polygon collision
  segments;
- final normalized coverage: mean 65.2549%, median 61.8546%, minimum
  49.4967%; mean normalized coverage AUC 36.0092%;
- its embedded `lewm_experiment_manifest_v1` binds the development manifest,
  materialization report, geometry, primitive registry, critical planner and
  footprint sources, invocation, environment, commit, and dirty-diff identity;
  re-verification of every recorded file passes;
- only the 138 training and 24 development scenes were materialized. The
  sealed manifest remains unopened and unmaterialized.

Canonical physical-claim v1 requalification evidence (2026-07-11):

- result document:
  `docs/lewm_go2_canonical_physical_claim_oracle_regression_result_2026-07-11.md`;
- immutable artifact:
  `.generated/oracle_positive_control/go2_generalization_v4_development/canonical_physical_claim_v1_report.json`;
- artifact file/content SHA-256:
  `4093461d842d926d4d351d84dec3bd8dff8a828f8730ef3b78c4a11aadfaee03` /
  `1b22227c0a7b8785033dd1c1e6a770a9108cbbda85698ff9dc9dabc5da0c26cc`;
- reviewed implementation-manifest SHA-256:
  `f55656eb303a20a1d2fa99813f2a28d84e822e9240e993422974dd416fa0450b`;
- result: 24/24 scenes, 96/96 unconditional oracle attempts accepted and
  credited, 96/96 independent eligibility witnesses accepted and credited,
  zero rejected/unverifiable/duplicate credits, zero stalls/collisions, and
  zero actual-yaw directional-polygon collision segments;
- one verified policy payload read, zero worker input-file opens, zero
  evaluator feedback, and zero prior-comparator, held-out, sealed, G2, label,
  image, or model-output payload opens.

Geometry-v2 paired-navigation pilot evidence:

- dataset manifest:
  `.generated/go2_paired_navigation/geometry_v2_pilot_v1/dataset/dataset_manifest.json`;
- SHA-256:
  `d2979f2fe778c1a8b06b5a457c1012b82900784954fa2efe674f70b035616e55`;
- 24 scenes across six navigation families, 384 exact 0.5-second primitive
  pairs, 17 training scenes and seven validation scenes, all nine primitives,
  with complete source/shard/RGB provenance;
- label distribution: 97.239685% UNKNOWN, 2.682686% FREE, and 0.077629%
  OCCUPIED. The first two-epoch development-only smoke reached calibration
  and then failed. Audit showed that the probability-calibration role itself
  contained `[131070 UNKNOWN, 0 FREE, 2 OCCUPIED]`; no sampler can fit a valid
  three-class calibration from it. The sampler now preserves the empirical
  distribution, minimally backfills a rare class missed by stride, and fails
  with explicit source counts if a class is truly absent. This pilot is a
  data/protocol failure, not a G2 model result; no untouched G2 model outputs
  were evaluated or used for selection.
- a geometry-v2 audit further invalidated this pilot for learning: only 84/384
  transitions (21.9%) have both endpoints inside the 0.47 m configuration
  space, and only 55/272 training rows have any next-frame observed cell. The
  v1 labeler also used body-inflated occupancy as camera occlusion geometry,
  creating virtual occluders around physical walls. Dataset v1 is retained as
  wiring/negative evidence only and is forbidden for model promotion.

G2 dataset-v2 evidence (2026-07-09, per the preregistration above):

- source index:
  `go2_navigation_sources_09991d78f2e2b483a43b7157a0301987308f958b6a9570c99670b1fb60dfd6b9.jsonl`
  (96 scenes, 12 per family, zero forbidden overlaps);
- dataset manifest:
  `.generated/go2_paired_navigation/geometry_v2_dev_v2/dataset/dataset_manifest.json`,
  SHA-256 `e474fce5c6ca520728a94fdaada9edc7d86beb69387e14a9cd882e4240530b0c`;
  5,641 rows; roles 72/8/8/8; assignments SHA-256
  `016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02`;
  untouched-G2 set commitment
  `0c9d5cfb6fdeec9be17a1afa8aed13fb62848a06594782c98933e1db8a2e1402`;
- adequacy report (G2 shards never opened):
  `.generated/go2_paired_navigation/geometry_v2_dev_v2/adequacy_report.json`,
  SHA-256 `24a0a64aa2a3d69e447289c0de82ea8628c6330841008fdefe855fb66109920a`;
  calibration role 82,388 FREE / 46,080 OCCUPIED supervised cells; combined
  nonempty next-observed row fraction 98.51%;
- run log and per-candidate detail:
  `docs/lewm_go2_g2_dataset_v2_build_2026-07-09.md`.

G2 execution correction (2026-07-10):

- old 2/20/60 results are invalid debugging history because they used the
  three-class fallback instead of the registered hierarchical loss; they
  cannot support G2 or a training-vs-data decision;
- strict deterministic execution failed closed before artifact creation on
  ROCm `grid_sampler_2d_backward_cuda`;
- corrected hierarchical smoke checkpoint/report SHA-256 values are
  `5eefe3ec1a75ac75fb37388ca5ad3dc73ed7376c326ade367d9579113771f345`
  and `ba91c45e94ea2290bb8686034937bdb9acaf8e496754504e202521bea010a4e6`;
- fixed-seed replay checkpoint/report SHA-256 values are
  `54a5b08b4dce138acc346875b6210b1abb7465413af99f489d0464f67a50a338`
  and `824aabc32d1fb979218831de4d8ab5221ebd6af96acc8de5ce74985bc0059338`;
- both corrected smoke reports record `g2_evaluated: false`, zero G2 rows, and
  no G2 provenance-verification scope;
- the corrected 20-epoch checkpoint/report SHA-256 values are
  `385a2380fe4170f9fa89b94a493662662f335f790f432f9b3eba76f663cccd58`
  and `9b036c8dea74226dd9602590810a202a7b163b598711614a9f89dff57af4b97d`;
  it failed offline with 0/288 threshold candidates and no G2 evaluation;
- the final four-view train-vs-selection diagnostic SHA-256 is
  `824d1cdfd597992966ae0e53250288e63e87d575d8c288f43a3196a3438aefff`.
  Both roles fail the frozen runtime contract. Raw selection has safety
  candidates only at near-zero useful recall; raw train has none. No
  data-limited verdict or 128-row rebuild is licensed.
- occupancy-only checkpoint/report/diagnostic SHA-256 values are
  `bbd9b496219792c146571a5628257388a9bb24bcb6c9d1a6fb44c0a5080d9d17`,
  `32b4b45cd977483deac81e664d16c11c94ad31b234c94db51966fe00ba35f9f3`,
  and `c75751c618792a19dcfbef44b452f3f4d34dbd5048f07e1d6c2a179d73886d11`.
  The ceiling rules out joint-loss interference as the primary occupancy
  failure and directs the next iteration to projective lifting and hierarchical
  calibration.

G2 camera/observability correction (2026-07-10, before a projective 20-epoch
run and with G2 model outputs still unread):

- source-camera audit artifact:
  `.generated/go2_paired_navigation/geometry_v2_dev_v2/source_camera_contract.json`;
  file SHA-256
  `f8f24d43768c2d5ddbccb85b91d26a3de790fed27d5b5d34d803f08824b6c80d`,
  content SHA-256
  `1d55b8a7ad7169f0790f793d72bc57625b411cf5e1b6fbcf9c4e87e7d46fc6d6`,
  96 source scenes, and `g2_images_opened: false`;
- non-circular source-bound v2 contract, consumed before corrected dataset
  construction:
  `.generated/go2_paired_navigation/geometry_v2_dev_v2/source_camera_contract_v2.json`;
  file SHA-256
  `b31fd8afdf1f4ec05589677d8c39b90521769501cb7f9e1c161fc5ca779a54e4`
  and content SHA-256
  `06013aaf471e83b8da3ca3806a7072c73050764d9ae46e3a80564ee7c21bc4ea`;
- those camera-only audits were superseded when renderer-object parity found
  that v03 omitted collision distractors; cropping alone is not admissible;
- sparse v04 render plan:
  `.generated/go2_render_selected_v04/render_plan.json`, file SHA-256
  `d93b17d45dd51f7bad4c442e8d434105997c2be4198f86dc498ded955c56a34c`,
  content SHA-256
  `1fbedb84ca584e1ffba7cfa1ae22e4e379deb10d5b85ca7f7cb2dda1a369f7e3`;
- v04 source index:
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/source_index/go2_navigation_sources_v04.jsonl`,
  SHA-256
  `11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c`;
- combined camera/renderer/object audit:
  `.generated/go2_render_selected_v04/audit_report.json`, file SHA-256
  `9a045dff82fb82adbbb89d10cb4dc0063297805038b000e5f6cd53816e995a9a`,
  content SHA-256
  `c9280ed4cab9ff54f7d8684835b8448886209a8cc50eba3588519c34572a6358`;
  96 scenes, 10,311 content-hashed endpoint frames, 4,087 rendered object
  instances, H=78.323/V=62.8370386364, full RPY, and roles 72/8/8/8;
- the audit read committed G2 row metadata and hashed rendered G2 bytes for
  integrity, but decoded/inspected no G2 image and opened no label shard or
  model output. Dataset v2 direct
  configuration labels and its projective smoke remain non-promotable;
- live Genesis now converts horizontal manifest FOV to renderer vertical FOV.
  The replacement physical labels and center-projective smoke must pass before
  any 20-epoch candidate.

Observable-physical dataset-v3 evidence (2026-07-10):

- dataset manifest:
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json`,
  SHA-256
  `ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180`;
- rows index SHA-256:
  `187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac`;
  5,641 rows with dataset/row schemas v3, role rows
  4,262/495/415/469, and role assignment/G2 commitments unchanged at
  `016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02` /
  `0c9d5cfb6fdeec9be17a1afa8aed13fb62848a06594782c98933e1db8a2e1402`;
- the normalized transition SHA-256 is unchanged at
  `51aca982da3c5a5e86ffaa959d10e6a5354d781a84824bae0e3076097fcf93d5`,
  and the sorted scene-set SHA-256 is unchanged at
  `2ab65f3511b7b2405ea0c3df062077771582c1c1b045c98ca2477b6226d7aa5d`;
- every frame target is observable physical FREE/OCCUPIED/UNKNOWN evidence.
  The v04 render summary remains bound in per-scene and global provenance, while
  the 0.47 m configuration morphology is evaluation-only after multi-view map
  fusion;
- the manifest embeds the resolved six-worker invocation, environment, commit
  `617d119172a6f49caf31a678e0fa7d05d5a3f4e9`, dirty-diff SHA-256
  `173c1b2f5cd5a25f8b3816c93e8f9ced24b675808a5d2ca00a26a965d6880c34`,
  source/render/geometry/exclusion hashes, and creation timestamp;
- adequacy report:
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/adequacy_report.json`,
  SHA-256
  `6fa6a667af2729ea1cf717a19997777a4fa227cd8e01551d21f2fb42d2e00e4d`;
  calibration has 100,403 FREE and 11,219 OCCUPIED cells, combined loaded-role
  nonempty-next-observation fraction is 97.757%, and all gates pass;
- the adequacy audit opened 88 train/selection/calibration shards and zero G2
  shards. A separate role-scoped provenance pass verified those same 88 scenes,
  88 shards, and 9,460 RGB files. It did not open the eight G2 shards/images or
  any G2 model output.

Physical occupied-detection threshold amendment (2026-07-10, after the bounded
two-epoch wiring smoke and before any longer physical-target run):

- the earlier v3 execution record had already registered occupied-detection
  threshold selection as the next calibration follow-up; the smoke proved the
  implementation still fixed this independent operating point at 0.50;
- physical target-space selection now searches
  `[0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]` on the calibration role;
- legacy configuration-space evaluation remains fixed at 0.50;
- admission and detection probability intervals must be disjoint, all candidate
  counts and the chosen threshold are persisted, and equal-performing choices
  prefer the highest detection threshold;
- this changes no 99/95/90 G2 threshold, scene role, row, calibration fit, or
  model output. G2 remains unread.

Frozen smoke rescore evidence (2026-07-10, before the matched 20-epoch run):

- artifact:
  `.generated/go2_egomotion_bev_jepa/dev_v4_physical_projective_smoke_v1/threshold_rescore_v1.json`;
- file/content SHA-256:
  `768a3ba2e4fb5a105c71a6b2237df42cfdd99dfeadda6b95d3cf6d961632ae9d` /
  `81959f1c4d7bd74ab6c95c8364e415ab0ce66eeefd06994a588c475c5c6c2932`;
- the rescore bound the exact parent artifacts and row subsets, preserved the
  model-state and calibration identities, and performed no training,
  calibration refit, checkpoint mutation, or G2 evaluation;
- its access ledger records 469 G2 metadata rows but zero G2 label-shard/image
  byte opens and zero G2 model-output rows;
- all 2,016 corrected tuples were evaluated and zero passed. The compatible
  fallback remained `(free_min=0.50, unknown_max=0.35,
  occupied_max=0.35, occupied_detection_min=0.50)` because lower detection
  points require stricter disjoint FREE admission and lose under the registered
  fallback ordering;
- calibration precision / useful-FREE recall / <=2 m OCCUPIED recall were
  0.73667 / 0.11034 / 0.000735; selection values were
  0.68800 / 0.16497 / 0.000857. This confirms that the bounded checkpoint is
  undertrained while closing the selector-wiring prerequisite for the matched
  20-epoch development candidate. It does not pass or evaluate G2.

First matched 20-epoch physical candidate (2026-07-10):

- checkpoint/report SHA-256:
  `c56fe958a841d8e7f89b4df0cecf63546abeae1a6a40044c4a1499f0486f0256` /
  `f40df099087d5e37e14366713b766963573e5638ecc9412ddfedfde90ad65fe6`;
- best epoch 10; train loss continued from 0.80392 to 0.31906 by epoch 20;
- calibration improved joint NLL 0.23554 -> 0.10231 and selected
  `(free_min=0.50, unknown_max=0.05, occupied_max=0.01,
  occupied_detection_min=0.02)`;
- calibration-role head metrics were 0.99209 admitted-FREE precision, 0.97703
  directly observable <=2 m OCCUPIED recall, and 0.01998 useful-FREE recall.
  The candidate therefore fails the unchanged 99/95/90 gate by usefulness,
  despite having safety-admissible thresholds;
- train-vs-selection diagnostic SHA-256:
  `24bfa8a38f1769d5fed2eeb435909acd8bb9ec5e543b9b97d737ed693c671bb1`.
  It verified only train/selection artifacts and concluded
  `train_role_physical_head_failure_blocks_generalization_attribution`;
- the candidate and diagnostic record or verify zero G2 label-shard/image-byte
  opens and zero G2 model outputs. G2 remains unevaluated.

Frozen spatial-grounding evidence (2026-07-10):

- artifact file/content SHA-256:
  `f8fc7c529197b3ba08574cba409695f564f54401d487c8eb48c1aa9cfdb4e3da` /
  `bff482f9036dc1549eedc676ca9944205c28c0b4581f03cea16f4a26f2dc817e`;
- exact pre-deserialization artifact hashes, training-source compatibility,
  deterministic execution, equal paired support, regenerated projective
  geometry, and start/end git state all validate;
- every train/selection frame was paired to a different image, scene, and
  transition. Shuffle worsens balanced NLL in 72/72 train and 8/8 selection
  scenes; selection micro delta is +1.03732. Mean RGB delta is +1.66031;
- identity is the best of 51 transforms. This rejects RGB-insensitive and gross
  alignment failures;
- selection <=2 m OCCUPIED recall is 98.44% in center-visible cells and 83.46%
  in exterior ring one, confirming the finite-cell versus center-point support
  mismatch;
- safe selection FREE recall is 8.11% at 1-2 m, 0.76% at 2-3 m, and zero
  beyond 3 m. The next performance hypothesis is spatial token resolution,
  after the parameter-neutral cell-square support correction;
- the artifact records 469 G2 metadata rows but zero G2 image/shard-byte opens
  and zero G2 outputs. G2 remains unevaluated.

Cell-square support / frozen counterfactual (2026-07-10):

- query-support / physical-aggregation contract SHA-256:
  `904ec5892f789bab55dda93431a0de167333f3887ff6d07f51ccfc79cd0b4107` /
  `db288979e7c389df2c4ca846f3309e395bcb6ec7bcf40cb8db6a3107f7e9f717`;
- support is center plus four fixed 0.05 m cell corners, records no body
  footprint, adds no learned parameter/state key, and is validated by lift type
  throughout checkpoint/report/training/development consumers;
- frozen counterfactual file/content SHA-256:
  `c88131efa8ef28b7db30f1105bafad14224d43ced8875f518c82491ee7f92eda` /
  `1e71339a66cce24aec5414575317292cf314fa18a3be2a8a8490253bb2ce77ab`;
- visible queries increased 1,990 -> 2,062, but unchanged-weight selection NLL
  worsened 0.22805 -> 0.23929, FREE precision fell 0.98591 -> 0.97545, and
  useful-FREE recall stayed near 0.02. The boundary correction alone is not the
  utility mechanism;
- the counterfactual trained nothing, mutated/emitted no checkpoint, and opened
  zero G2 images/shards/outputs. The next gate is the train-only matched token-
  resolution micro-fit, not a full run or memory-fusion bypass.

Patch/tokenization-resolution bundle micro-fit preregistration (2026-07-10,
corrected before authoritative GPU output):

- the exact protocol is
  `docs/lewm_go2_physical_micro_overfit_protocol_2026-07-10.md`; it supersedes
  the earlier temporal-quartile, early-stop, and single-seed wording for this
  diagnostic;
- the first eight-transition/single-scene metadata-only pilot was aborted before
  an authoritative panel or GPU output because medium-maze fit and cross had
  zero FREE support beyond 2 m and other family/bin supports also missed the
  frozen minimum; no threshold or row was adapted from that failure;
- the superseding seed-fixed, label-independent panel hash-ranks all nine train
  scenes per family into a four-scene fit/same pool and a five-scene cross pool.
  Even/odd stream ranks split fit from same-pool holdout, every pool scene
  contributes at least two rows, and fixed metadata-only prefixes select 32
  transitions per family/panel: 160 transitions and 320 frames per panel, 480
  globally disjoint rows and 960 unique endpoint hashes in total;
- post-selection support must include at least 1,000 aggregate true-FREE cells
  and at least 100 per-family true-FREE cells in each of 1-2 m, 2-3 m, and
  >=3 m on every panel; failure aborts without reselection;
- a non-authoritative metadata-only N=32 reproduction produced 160/160/160
  rows, 960 unique endpoint hashes, minimum aggregate gated-bin support 20,551,
  and minimum family/bin support 512. It opened train labels only for the
  frozen post-selection assertion and produced no model output;
- matched arms are 112px/patch14/8x8/sigma1 versus
  112px/patch7/16x16/sigma2 under the same center-projective lift, samples,
  minibatch order, fixed update count, and copied shared initialization; the
  causal claim is the patch/tokenization-resolution bundle, including changed
  patch-embedding tensors, parameter count, and attention compute, not token
  resolution alone;
- the faithful stage always consumes 2,000 updates; if either faithful arm
  fails, both restart from their initial states and always consume the
  registered 3,000-update ceiling stage. There is no independent early stop;
- alternate budgets are accepted only under a distinct non-authoritative smoke
  schema that is explicitly non-promotable and rejected by the finalizer;
- stage pass requires the aggregate gate and all five family gates at each of
  the final three evaluations (1,800/1,900/2,000 or
  2,800/2,900/3,000). Earlier first/three-pass steps are diagnostic only;
- every fit gate requires balanced NLL <=0.03, both hierarchical balanced
  accuracies >=0.99, every class recall >=0.98, FREE recall >=0.95 in each
  gated distance bin, and both cross-scene and same-scene wrong-view minus
  correct-RGB NLL >=0.25;
- per-arm expressivity is faithful OR ceiling. Holdouts are compared only at a
  common passing stage, faithful preferred, using equal-weight family macros;
- on both holdouts patch7 must have macro NLL ratio <=0.80, macro far-FREE
  delta >=+0.10, every macro class-recall delta >=-0.01, and no family/class
  delta below -0.01. All 5/5 cross-scene families and at least 4/5 same-scene
  families must strictly improve both NLL and far-FREE; ties fail and the
  cross-scene directional result is one-sided exact `p=1/32`;
- a single seed may report only provisional support and must always emit
  `patch7_full_train_candidate_licensed=false`. The only license is a pure
  aggregation of immutable result artifacts for seeds `20260710` and
  `20260711`, through `scripts/finalize_go2_physical_micro_overfit.py`, with
  precommitted input file hashes, exact authoritative settings, recomputed
  stored decisions, matching panel/contract/source hashes, and the same
  favorable branch, classification, qualifying optimizer stage, and support
  mechanism;
- the global JSONL parser materializes full non-train row metadata/path strings
  temporarily, but no non-train path is emitted or dereferenced. Source scene
  NPZ archives include unselected train rows and materialize archive-level
  arrays; the optimizer indexes only selected fit rows. Checkpoint-selection,
  calibration, and G2 artifacts remain unopened. This diagnostic cannot pass
  G2 by itself.

Patch/token and categorical-radial micro-fit results (2026-07-10):

- the authoritative patch/token result is
  `.generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json`,
  file SHA-256
  `6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c`;
- neither patch14/8x8 nor patch7/16x16 passed its fit gate. Patch7 improved
  OCCUPIED recall but did not remove the projective boundary/occlusion ceiling,
  so the registered second seed was correctly not run;
- the replacement categorical-radial factorization uses 64 range bins and 256
  bearing bins. Across all 960 frozen panel frames it has zero supervised cells
  outside support, zero mapping collisions, and exact integer-label
  scatter/gather roundtrip;
- categorical-radial V1 result file/content SHA-256 values are
  `72e4ecbe6b9e9024bb910e5231deb42e2d73f3187babd2a9af518251cbb7c2a2` /
  `02c627eb01e42a5b7e8ea57e5bd4bde3d1fc2ca0667abdd9dd1cf8162beacd52`.
  N=1 passed. N=4 passed every evaluation from step 300 through 1,400 but
  failed at fixed step 1,500 after a constant-rate optimizer excursion;
- the V2 optimizer amendment was frozen before output at SHA-256
  `58f994a639c8e5a733d92c6da1fad63fa654e1f57aa7be0a8373e3eaa47b3f46`.
  It changes only the N=1/4/16 ladder to a stage-local no-warmup cosine from
  `2e-4` to `1e-5`, with the final update itself at `1e-5`;
- categorical-radial V2 result file/content SHA-256 values are
  `06517e2c6641495a6262aa9f8a5cb45648912c575f1c3663df899c50a2867daa` /
  `8528ae02d6faaf25eb666d591e15180e82f74c9cf4d798c8322f9d5c50c910bc`.
  N=1 and N=4 pass their fixed terminal gates. N=16 fails smoothly at NLL
  0.01151 and UNKNOWN/FREE/OCCUPIED recall 0.98747/0.99828/0.97007, while its
  wrong-view separation is 3.53510. This is a decoder structure/capacity
  ceiling under the frozen budget, not the V1 optimizer failure;
- the sole full-ray intervention was preregistered in
  `docs/lewm_go2_categorical_radial_ladder_v3_full_ray_amendment_2026-07-10.md`,
  SHA-256
  `921fc48cf2a41924c720654c2d08fbd09ca6ce3ccc7c94ccb6600096a434fcbf`.
  It replaces only the five-bin radial block with a full-ray six-block dilated
  stack and leaves the encoder, factorization, V2 optimizer, data, budgets,
  controls, and gates fixed;
- the authoritative V3 ladder result file/content SHA-256 values are
  `7a5f67bacb2e3df67421bcff13b15d1fa3e00d99f3b2af52c52b0b6ce14617a8` /
  `517313139077027176c471f829f57148684d3df0def6096ce7702d3bbba46ce1`.
  N=1, N=4, and N=16 all pass fixed terminal gates. N=16 reaches NLL 0.00285,
  UNKNOWN/FREE/OCCUPIED recall 0.99644/0.99984/1.00000, and wrong-view delta
  4.00051; it passes every evaluation from step 900 through 2,000;
- this licenses only construction of the original N32 fit/holdout diagnostic.
  It does not license a second seed, full training, calibration, or G2;
- the exact N32 frame batching, optimizer branches, terminal-three gates,
  conditional holdout access, immutable failed-faithful patch7 comparator, and
  two-seed authorization/finalization rules are frozen before N32 output in
  `docs/lewm_go2_categorical_radial_n32_execution_binding_2026-07-10.md`,
  file SHA-256
  `42c2ce88ac78f045b92fdd2b33ad5b77a0801de0af2e05c79d3bb518ca188241`;
- the authoritative N32 V1 seed-20260710 result file/content SHA-256 values are
  `2f079925000ebbcd06843c413f4dcfd07fce93358482dd05512735af69cbc946` /
  `ef023faff0e49888ca673cfab5fca0c1110852e49312ce339ecb7f03ab3a8d5b`.
  Both fixed branches failed the fit gate, so the runner opened neither
  train-role holdout and seed 20260711 is forbidden. The best ceiling
  evaluation reached NLL 0.04954 and UNKNOWN/FREE/OCCUPIED recall
  0.95222/0.99017/0.93674 with large wrong-view separation;
- N=16 received 500 effective epochs, while the original N32 ceiling received
  only 62.5 after a 20x frame-count jump. The exposure-matched V2 retry is
  frozen before output in
  `docs/lewm_go2_categorical_radial_n32_v2_exposure_binding_2026-07-11.md`,
  file SHA-256
  `4164ec011910cb2d1d2fbea5beaad81eb13ea6b506e063ebf13a66a41e14fb6f`.
  It keeps the model/data/gates fixed and transfers the V3 25%-panel batch,
  500 epochs, evaluation cadence, and cosine schedule to N32;
- all patch/token and categorical-radial diagnostics are train-role-only. Their
  recorded checkpoint-selection, calibration, non-train, and G2 byte/model
  access counts are zero. None can pass G2 or license runtime promotion.

Exposure-matched N32 V2 result (2026-07-11):

- the authoritative seed-20260710 result file/content SHA-256 values are
  `0a5f8a822d7fec8287a30103125fca1a4927f0413e2f0906db431cef54ec2265` /
  `e070cc96d69b76e1f85f533fa1d94221225963a2b66a491f0c2a867c008b97ef`;
- aggregate fit passed at terminal steps 1,800, 1,900, and 2,000. At step
  2,000, NLL was 0.01105, UNKNOWN/known balanced accuracy was 0.99254,
  FREE/OCCUPIED balanced accuracy was 0.99987, and UNKNOWN/FREE/OCCUPIED
  recall was 0.98632/0.99959/0.98778;
- the mandatory all-family rule failed consistently. Open-field OCCUPIED
  recall was 0.97872 (598/611 cells), while rough-terrain UNKNOWN recall and
  UNKNOWN/known balanced accuracy were 0.97452 and 0.98693;
- the strict finalizer independently accepted the result and access ledger.
  Both train-role holdouts, seed 20260711, full training, calibration, G2, and
  runtime remained unopened or unauthorized;
- the immutable result report is
  `docs/lewm_go2_categorical_radial_n32_v2_result_2026-07-11.md`. The frozen V2
  contract now licenses only a preregistered representation/capacity
  intervention at N32.

N32 V3 token-width binding (2026-07-11, before dataset-backed V3 output):

- the sole intervention widens the task-token projection from 24 to 32 while
  leaving the encoder, 64-channel context, projective anchors, full-ray
  dilations, factorization, data, loss, controls, 500-epoch schedule, and gates
  unchanged;
- the candidate has 2,891,171 parameters, exactly 4,104 more than V2. Only the
  token-projection weight/bias and context-stem input weight change shape; all
  other initial state entries must be bit-identical to the corresponding V2
  seed initialization;
- a dataset-free direct batch-80 backward check was finite and peaked at
  15,409,975,296 allocated bytes. It produced no research artifact;
- the frozen binding is
  `docs/lewm_go2_categorical_radial_n32_v3_token_width_binding_2026-07-11.md`,
  SHA-256
  `a9898d349d82f65ce35443192b555aac4386136032c8fe70c115eda5a788a5ad`;
- holdouts remain conditional on exact aggregate-plus-all-family passes at
  steps 1,800, 1,900, and 2,000. Seed 20260711 remains conditional on a fully
  favorable seed 20260710. This diagnostic cannot pass G2 or license runtime.
- the pre-run implementation manifest is
  `docs/lewm_go2_categorical_radial_n32_v3_implementation_manifest_2026-07-11.md`,
  SHA-256
  `200c1d9d8944fb0252828b659f1c32c6176cf7d32d2e4e89cf10abb5d2ca1877`.
  It commits the identical 32-entry runner/finalizer source map, both initial
  states and schedules, 128-test regression result, adversarial access fixes,
  and fit-only three-step smoke;
- the smoke file/content SHA-256 values are
  `e8bbd920610c68be9b82d109037745d27a2ebcccc8cb13c6e2de25c7f6b5a2ac` /
  `301b0a9ac486ae4f69d0f34cb46734922ad08a410bdc092b9355acf49aa8ac41`.
  It opened 320 fit images and 20 fit shards, and zero holdout, selection,
  calibration, non-train, G2, or sealed payload/model output.

N32 V3 token-width result (2026-07-11):

- the authoritative seed-20260710 result file/content SHA-256 values are
  `0f3eb212afe54a38d7a81a1fc51ca544dfab667a94a836be742d3ea3e2298d85` /
  `ec8dd8450fb34bee3a5ba1c5a5b532339d281241560c8ed9ac07a48d2c2bea4e`;
- aggregate fit passed at terminal steps 1,800, 1,900, and 2,000, but only the
  medium-enclosed family passed every gate at step 2,000. Final aggregate NLL
  was 0.01214 and UNKNOWN/FREE/OCCUPIED recall was
  0.98506/0.99951/0.98211;
- compared with width-24 V2, rough-terrain UNKNOWN recall fell
  0.97452->0.97208 and open-field OCCUPIED recall fell 0.97872->0.96072.
  The token-compression hypothesis is rejected;
- the strict finalizer independently accepted the artifact and access ledger.
  Both holdouts, seed 20260711, full training, calibration, G2, and runtime
  remained unopened or unauthorized;
- the immutable result report is
  `docs/lewm_go2_categorical_radial_n32_v3_result_2026-07-11.md`, SHA-256
  `a346ecb9b909d897f839067e409bccd906f61223cec8e746da6b54c531f44fca`.

N32 residual diagnosis (2026-07-11):

- V2's conditional FREE/OCCUPIED decision is effectively solved: all 10,228
  occupied targets rank OCCUPIED over FREE once conditioned on KNOWN, and the
  conditional confusion matrix is `[[118763, 30], [0, 10228]]`. The remaining
  joint errors are principally KNOWN-versus-UNKNOWN errors;
- adding one common bias to the FREE and OCCUPIED logits preserves that
  conditional decision, monotonically lowers UNKNOWN recall as it raises
  OCCUPIED recall, and has no value that satisfies the registered rough-family
  UNKNOWN and open-field OCCUPIED gates simultaneously. The proof is frozen in
  `docs/lewm_go2_n32_known_bias_impossibility_2026-07-11.md`, SHA-256
  `e214bb80bcccf9ae5051231d90f7a5d8c2bfa33ca799e7db3eb969698fa2108a`;
- physical-v3 visibility labels are not a monotonic horizontal depth profile.
  FREE requires direct visibility of all intersecting ground cells, OCCUPIED
  comes from a sampled 3-D camera ray hit, and UNKNOWN includes partial,
  outside-frustum, occluded, or vetoed support. A simple one-depth-per-horizontal
  ray head is therefore not a faithful replacement;
- the next intervention is ordered by a metadata-only camera-pose projection
  audit frozen in
  `docs/lewm_go2_n32_pose_projection_audit_binding_2026-07-11.md`, SHA-256
  `c959c45737b9242ef667772af4c7b72effcbb39ae687f5ee28226e38cd63854a`.
  A material rough-terrain projection mismatch licenses a fixed-versus-recorded
  pose A/B first; otherwise the smaller explicit KNOWN plus OCCUPIED-given-KNOWN
  output factorization runs first;
- the audit's monolithic panel input was corrected before any audit result. A
  fit-only 160-transition/320-frame metadata artifact was extracted with file
  SHA-256
  `77d84e242d75b81fd2b96f086e9cf5df72f0a907e1fe7ce24fc48bbc5d514037`
  and content SHA-256
  `8e44dd0238077120e97fd06b4550d6504627066c7e8ddfdfbd138fd7504ee7a8`.
  Its result report is
  `docs/lewm_go2_n32_pose_projection_fit_panel_result_2026-07-11.md`, SHA-256
  `f41b28edb2b8ef23306f2d2bec7be9e10ea308240e9dcfa4bf791c81ef85b33d`.
  The authoritative audit runner must not open the monolithic panel, images,
  labels, model outputs, G2 payload, non-train payload, or sealed data.
- the first authorized audit attempt failed closed with no result before source
  frame scanning because current physical-dataset role and legacy rollout
  `split` were ambiguously named. The unexecuted 15-scene filter proposal,
  SHA-256
  `35c0de28a795d6b5c246548f5d773326b3f137310c0ec9a840b3e7bf1d302e1d`,
  was superseded before output because it would discard current training data
  and unbalance families;
- the governing role-namespace amendment, SHA-256
  `ae17eb856c5329e8c5dfa5e4339306ef19e60c53c5f67d43746b268be9cc3370`,
  retained all 320 current physical-training records. The reviewed V2
  implementation manifest is
  `docs/lewm_go2_n32_pose_projection_audit_v2_implementation_manifest_2026-07-11.md`,
  SHA-256
  `62375f9116843418e3812078ea23a8ed870a6bec0e4fe42580427c411d5df3bf`;
- the authoritative pose result file/content SHA-256 values are
  `2c7efba897054ea0067db58f020e70dc5f3c5804785c74cbda4a8b76e0210b9d` /
  `6a9d05a0fb92289334cf39bb6947a2022a05a7c1892e8bb1c5a7156f9ca227f4`.
  Rough median per-frame p50 displacement was `0.25533` token, pooled
  non-rough was `0.28144`, and their difference was `-0.02611`; both frozen
  dynamic-pose thresholds failed. The next intervention is therefore explicit
  hierarchical output. Exact access reconciliation passed with zero image,
  label, model, G2, physical-nontrain, monolithic-panel, or sealed opens;
- the immutable audit result report is
  `docs/lewm_go2_n32_pose_projection_audit_result_2026-07-11.md`, SHA-256
  `e1a0c7e8c161827c5d8a1e2088135d8d986cbce9f9f7c02aa43d78d37a0be5e8`.
- before V4 implementation or dataset-backed output, the explicit hierarchy
  experiment was frozen in
  `docs/lewm_go2_categorical_radial_n32_v4_hierarchical_binding_2026-07-11.md`,
  SHA-256
  `bb691c787af0b90f813ced4e5e521f1b15b70b75c836147cd69275c50df6b5d3`.
  Its sole model change is a width-24 two-factor polar head reconstructed to
  normalized UNKNOWN/FREE/OCCUPIED log probabilities before the unchanged
  Cartesian gather; V2 loss, schedule, data, controls, gates, and conditional
  access remain fixed;
- the pre-output V4 implementation manifest is
  `docs/lewm_go2_categorical_radial_n32_v4_implementation_manifest_2026-07-11.md`,
  SHA-256
  `6f1f936efeca1e684e394e2a1680002b5ba719d4d24c27694c01821455926ffc`.
  It freezes a 41-entry transitive source map, both seed initial states and
  schedules, 131 bit-identical common state entries, 42 focused tests, and 144
  selected V1-V4 regressions. The distinct fit-only smoke report is
  `docs/lewm_go2_categorical_radial_n32_v4_smoke_result_2026-07-11.md`,
  SHA-256
  `c5d2f17f44f528be9e7dad30c4d7e4aff1a6896251ddb08e241ded44848718ac`;
- the authoritative V4 seed-20260710 result file/content SHA-256 values are
  `d4736b76e354c63268ee7698cacc0ae1834b888407c32095f22b562ce1726789` /
  `719841ac72d09f6240be59a26fdcab059ed070bc4b7cccf3fa79ddbfa2be5103`.
  Aggregate fit passed, but only large and medium enclosed mazes passed every
  family gate. Open-field OCCUPIED recall was 586/611, rough-terrain UNKNOWN
  recall was 200,880/207,086 with UNKNOWN/KNOWN balanced accuracy 0.98449, and
  small-maze OCCUPIED recall was 2,365/2,420. All failed OCCUPIED cells still
  ranked OCCUPIED over FREE and were rejected as UNKNOWN, so last-head
  factorization is rejected as sufficient;
- the strict torch-free adjudication accepted the result and its zero-holdout
  access ledger. Seed 20260711, shared-JEPA construction, G2, runtime, and
  promotion remain forbidden. The immutable result report is
  `docs/lewm_go2_categorical_radial_n32_v4_result_2026-07-11.md`, SHA-256
  `dd0842d1c59b42a985eaf0843f0d6f6adc41286a2a1a2b4b1f95111a9c0efa50`;
- the preregistered next fault is the body-centered context lattice: because
  the camera is 0.326 m forward, its columns do not follow physical camera
  rays. The fit-only geometry/label-observability audit is frozen before
  implementation or further label access in
  `docs/lewm_go2_n32_camera_frustum_observability_audit_binding_2026-07-11.md`,
  SHA-256
  `c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9`.
  It allows only the 320 fit targets and committed source geometry, forbids
  RGB/holdout/G2/sealed/model access, and must prove injective camera-centered
  mapping, complete known-label support, bit-exact label reconstruction, and
  rendered/collision provenance before successor model output;
- the independently finalized v2 camera-frustum audit passes source,
  provenance, reconstruction, camera-mapping, camera-mount, ambiguity, and
  three-phase access gates, but fails known-target coverage. The old
  whole-body current-frame target contains 1 FREE and 372 OCCUPIED supervised
  cells outside proposed current-camera support across 320 fit frames, with a
  failure in every registered family. Camera-frustum representation
  implementation, training, G2, and runtime remain forbidden. The immutable
  result report is
  `docs/lewm_go2_n32_camera_frustum_observability_audit_v2_result_2026-07-11.md`,
  SHA-256
  `8bfb4c9a8b69f67b3b9e4d6e3b21e9ff89ecaff89a2bab3eb83d759ca4fe6d22`;
- the ordered next intervention is a pre-output geometry/decoder amendment.
  Observable-physical-v3 already separates current zero-inflation evidence
  from persistent memory and the audit found no rendered/collision ambiguity.
  The failed center-point-to-one-polar-bin support must be replaced by a
  source-grounded cell-footprint or ray-witness construction that supports
  every visible FREE/OCCUPIED target. Freeze its camera/output support,
  aggregation, coverage proof, N32 ladder, and access boundary before model
  output; do not tune or relabel targets to V4 errors.

Historical distance-plus-LOS learned-baseline diagnostic evidence:

- artifact: `.generated/strict_scores/phase4_full18_v1.json`;
- SHA-256:
  `eb796aea694c260f0da1a2e36404e86631008eac107f0adcc1061c9f11abfe2c`;
- baseline: 8/72 diagnostic claims, 0/18 all-four, 8.30% median final coverage,
  5.67% mean normalized AUC;
- novelty stack: 9/72 diagnostic claims, 0/18 all-four, 5.46% median final
  coverage, 5.54% mean normalized AUC.

Preregistered G3 fast baseline:

- panel: `config/go2_generalization_v3/fast_development_v1.json`;
- run summary SHA-256:
  `aff4a8a81c84a1c3e3ed547b227b9a21f1b0be0f34036954d55e9fbc01cc1333`;
- strict score artifact:
  `.generated/go2_fast_development/v1/baseline_strict.json`;
- strict score SHA-256:
  `50245316b2f7c1249b00cbb606db0b8406caf571f926ca8319deeaa204a4fe8c`;
- result: 2.6316% median final reachable coverage, 1.7505% mean normalized
  AUC, 4/32 distance-plus-LOS diagnostic claims, 0/8 all-four scenes;
- canonical geometry diagnostic: 2,002 occupied-space crossing intervals out
  of 4,800 logged intervals. G3 cannot pass by increasing this count.

## 2026-07-12 Observable-Target Correction And Current Execution State

The old whole-body per-frame target is now rejected from first principles.
Independent perfect-camera and mismatch-decomposition audits showed that
98,472 of 98,473 physical-prior errors required inference about finite world
boundaries invisible in the current RGB frame. That is not a legitimate
single-frame perception target. Current-frame perception must predict only
camera-observable evidence; persistent physical memory and the registered
0.47 m configuration morphology perform the later spatial and body-clearance
reasoning.

The replacement observable camera-evidence V4 dataset is complete and audited:

- dataset manifest:
  `.generated/go2_observable_camera_ray_fit_v4/v1/manifest.json`;
- manifest file/content SHA-256:
  `2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85` /
  `9be0c1539897bd731d4dfaf96e03b5d5c1d31d8cb8c723a2b77ffde57baf2812`;
- audit result:
  `.generated/go2_observable_camera_ray_fit_v4/v1/audit_result.json`;
- audit file/content SHA-256:
  `2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c` /
  `a922114b7e42552043a487bae527c35fb511804d4e8683c5a3f64a2bf499cf76`;
- audited support: 320 train-role frames, 1,310,720 output cells, 20/20
  label shards, exact deterministic reconstruction, and zero forbidden, G2,
  held-out, or RGB payload opens during label construction/audit;
- class counts: 1,072,012 UNKNOWN, 228,477 FREE, and 10,231 OCCUPIED;
- relative to the old target, all old FREE/OCCUPIED labels are retained and
  109,684 formerly UNKNOWN cells become camera-witnessed FREE.

The exact result and target meaning are recorded in
`docs/lewm_go2_observable_camera_ray_fit_v4_result_2026-07-12.md`. The V4
trainer, preauthorization launcher, immutable attempt registry, and two-seed
ladder finalizer remain under adversarial source review. Their authorization
flags remain false. No V4 RGB has yet been decoded for model training and no V4
checkpoint/model output exists.

The independent trainer/ladder review subsequently returned `BLOCK` and is
recorded in
`docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_2026-07-12.md`.
It reproduced a fabricated predecessor gate, aggregate metrics unbound to
checkpoint inference, artifact opens before canonical-path/authorization
checks, a same-process forgeable launcher receipt, and a hash-to-import source
replacement interval. N=1 remains forbidden until canonical artifacts are
reopened, checkpoint inference independently reproduces every metric, and the
trainer executes from verified content-addressed bytes inside one preflighted
launcher process.

The required target-partition freeze then exposed an independent structural
failure in the old `(1,4,16,32,320)` ladder: N=1 contains zero represented
depth hits and N=4 cannot contain all five families, while their gates required
both. Before any V4 RGB decode or model output, the dated amendment
`docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v2_partition_amendment_2026-07-12.md`
replaced those two impossible rungs with N=5, yielding the ladder
`(5,16,32,320)`. The exact four-rung target counts and ordered target-byte
commitments are frozen in
`docs/lewm_go2_observable_camera_ray_fit_v4_target_partitions_2026-07-12.json`
and independently reproduce from all 180 committed evidence files without
opening RGB. Source/auth integration and a new independent review remain
required; no training license follows from the correction itself.

The final registered legacy dynamic-radial compatibility diagnostic also
completed on 2026-07-12. Its result file/content SHA-256 values are
`bc374656c8a871bd111cba916553eb128249aa31f03031e85741206e3c5c0959` /
`5c1255df0f36bb5b1053dd546aa2323e3ea083be85237812b18138dc7b633086`.
The production branch passed its aggregate fit gate but failed open-field and
rough-dynamics family gates; the conditional 5,000-update ceiling branch also
failed. It opened no non-fit or G2 payload and grants no checkpoint, G2,
shared-JEPA, runtime, or held-out license. The immutable negative-result report
is `docs/lewm_go2_dynamic_categorical_radial_fit_result_2026-07-12.md`.

Additive shared-JEPA V5, revisioned physical/configuration memory, and G4
frontier-viewpoint foundations now exist as unpromoted source scaffolds. They
remain blocked on independent source review and exact equivalence/runtime gates;
no learned G3/G4 output is licensed by their presence. In particular, the first
G4 deterministic-baseline source review rejected an unissued caller-supplied
view state, occupancy-frontier-only candidates, conflated coverage/entropy/
discovery terms, substitutable path starts, and nonconservative ray geometry.
Those findings must be closed before any G4 baseline output.

The first G5 reversible-target-belief foundation also received an independent
`BLOCK` verdict on 2026-07-12. The reviewed implementation/test SHA-256 values
were `0e660a6e94b1483cde1e56267f503e3b5e6c35e81a2feab4e4371d4718bd6d10`
and `ae87eb7fe7595c61452713ec65f96eb91ec384d3c855f1bcd81c99bcee47a525`.
Adversarial reproductions showed circular task-set authority from the raw
trace, substitutable physical manifests, caller-asserted negative visibility,
duplicate-payload replay, exact-zero posterior underflow, missing rejected
evaluation records, same-episode evaluator feedback, content-clone snapshot
issuance, mutable ledger aliases, non-atomic commits, and absent strict
whole-memory serialization. No G5 rollout or learned-head output is licensed
until a successor closes those defects and passes a different independent
review.

The G5 stage-1 successor subsequently passed independent source and scaling
review at implementation/test SHA-256 values
`366dfa92e8178d45b01e36552d67b52da0ad2805a7b0cc1abfee62527918459f` /
`b4493a690cbaa6ea85eb9200eecbdd40d1a4cf305a42f868ef68d13287edde67`.
It passed 43 focused and 319 canonical-claim tests, sustains about 0.2 ms online
updates without history replay, and confines exhaustive replay to audit and
serialization. This closes only the reversible-memory foundation; the exact
adapter/router, learned observation head, and all 96/96 conversion gates remain
open.

The first G3 revisioned-memory source review is recorded in
`docs/lewm_go2_g3_revisioned_memory_source_review_2026-07-12.md` and also
returned `BLOCK`. A promoted memory accepted arbitrary caller-built learned
FREE cells, a fabricated traversal polygon that certified 10,000 cells FREE,
and the public exact adapter without privileged taint. The morphology and
planner tests remain useful, but no G3 output is licensed until learned,
traversal/reset, and exact evidence each have distinct instance-issued
admission paths and conservative pose-uncertainty projection.

## 2026-07-13 fail-closed G3/G4 reviews

The successor G3 boundary independently passed the four prohibited promoted
input probes: exact, direct learned labels/UNKNOWN, caller traversal, and caller
execution blocks are rejected atomically, while all exact development evidence
remains privileged-tainted through snapshots and serialization. This is an
authority-boundary PASS only; learned projection, executor/reset issuance,
exact equivalence, cold start, and closed-loop G3 remain open.

The independent G4 source review is recorded in
`docs/lewm_go2_g4_frontier_viewpoint_source_review_2026-07-13.md`. Candidate
generation, conservative camera-ground visibility, information-gain ranking,
and stale-route rejection are retained. The review found that caller-authored
visual swept cells could enter `PhysicalViewStateIssuer.record_view`; promoted
runtime now rejects that path until a qualified camera-view adapter issues the
observation. This is a fail-closed source-foundation PASS only and does not
license G4 output.

The successor exact-physical audit is preregistered in
`docs/lewm_go2_g3_exact_physical_equivalence_audit_plan_2026-07-13.md`. It
separates exact 89/69 implementation agreement, analytic no-unsafe-FREE
dominance, 96/96 claim-endpoint preservation, and the unchanged historical
strict binary-grid equality condition. A one-scene design diagnostic rejected
centre-sampled physical labels; no 24-scene output or contract amendment has
yet been made.

The second independent V4 trainer review remains `BLOCK`: a schema-only caller
mapping could mint the same-process verified launch context and import the
trainer/Torch while authorization was pending, and the metric verifier's
ordinary imports remained separable from the disk bytes it rehashed. The
corrected partitions and metric reconstruction are retained, but no GPU fit is
licensed until canonical context issuance and captured-byte verifier/finalizer
loading pass another different-agent review.

The independent shared-JEPA V5 authority review also remains `BLOCK` despite
31 passing synthetic tests. It reproduced perfect G2 from caller records with
no model/raw outcomes, copied-root registry duplication, omitted decision-core
source authority, registry namespace traversal, and mutable global capability
forgery. All production constants remain `None`; the successor must bind raw
runner outcomes and close all five paths before any G2 access.

The first promoted executor/reset admission candidate also returned `BLOCK`
despite 102 passing dependency tests. Public raw-pose/reset issuance and
caller-selected contracts allowed a synthetic 10 m traversal, a reset at cell
`(1000,1000)`, a 5 m body/step contract, copy-cloned authority, and importable
capability globals; the largest reproduction certified 1,000,000 FREE cells.
The successor must consume fixed runner-owned outcomes, bind the exact frozen
geometry and sampling cadence, reject copy/clone authority, and remain
structurally unavailable while hardware geometry promotion is false.

The immediate fail-closed remediation is recorded in
`docs/lewm_go2_g3_executor_reset_evidence_admission_implementation_2026-07-13.md`.
It withdraws the public bind/issue/build/fuse surface, removes all execution
admission/binding/replay capability globals and mutable issuance tables, makes
memory and the unavailable adapter non-copyable, freezes the future canonical
geometry/timing envelope, and unconditionally rejects promoted executor/reset
evidence while runner identities are unset and hardware promotion is false.
The definitive exploits are permanent regressions. This is a candidate
remediation of the blocked source surface only; the `BLOCK` remains until a
different reviewer accepts it, and no executor/reset or G3 output is licensed.

The qualified learned camera-to-memory boundary is preregistered in
`docs/lewm_go2_g3_qualified_learned_projection_plan_2026-07-13.md`. It admits
only runner-issued, checkpoint/G2/calibration/frame/revision-bound outcomes;
FREE requires complete destination-square support under every allowed pose
transform and OCCUPIED uses the uncertainty union supercover. Numeric view
diversity thresholds remain unselected until legitimate G2 calibration.

The G5 stage-1 authority portion is reopened under the same threat model. Its
sparse posterior/reversibility/scaling result remains valid, but public
hash-parameterized `bind_g3` and producer APIs plus importable global
capabilities allow caller-authored target distributions and visibility
certificates. G5 evidence authority is therefore `BLOCK` until it consumes the
fixed runner-owned raw outcomes; no G5 rollout had been run or promoted.

The governing scientific execution threat model is now explicit in
`docs/lewm_go2_scientific_execution_authority_threat_model_2026-07-13.md`.
Python tokens/closures are not treated as secrets from arbitrary same-process
code. Authoritative stages instead use one-shot reviewed CLI processes,
captured fresh source graphs, fixed no-follow role paths, raw per-instance
outcomes, independent finalization, and no dynamic controller plugins. Stronger
adversarial isolation requires an OS-level boundary rather than another Python
capability object.

## 2026-07-13 navigation-readiness closure

The active milestone is frozen in
`docs/lewm_go2_navigation_work_readiness_goal_2026-07-13.md`. It requires a
qualified learned observation to feed the revisioned physical memory,
viewpoint explorer, target belief, router, and claim evaluator in development
simulation. It does not itself claim closed-loop or held-out success.

The shared-JEPA V5 one-shot source remediation passed different-agent review.
The runner performs exactly one captured inference per raw instance; the
finalizer reconstructs every per-scene count, ledger event, source,
checkpoint, and outcome binding; and the publisher independently reconstructs
and byte-compares both G2 and G3 reports. The focused suite passed `50/50`.
The reviewed core SHA-256 is
`32ddaa83a1120c6b4610863020b4ff4d6dda94b1f8d37dafa2eb5b7740781a2f`.
All six production identities remain `None`, so this is a source-boundary PASS
and not a G2/G3 result or publication license.

A subsequent integration audit reopened the V5 model contract, without
reopening the passed one-shot execution findings. The model advertised pixel
ray tensors as `(B,H,W,D)` while emitting `(B,D,H,W)`, and its joint objective
could omit the three raw V4 source/ray losses. The additive correction is
recorded in
`docs/lewm_go2_shared_jepa_v5_output_loss_correction_candidate_2026-07-13.md`.
It versions the output contract and makes the exact four-equal V4 objective
mandatory inside joint training. The final model/one-shot bytes passed
different-agent review with `51/51` CPU-capped tests, exact loss and tensor-axis
reproduction, malformed-input probes, and GPU visibility disabled. This closes
V5 source readiness only; all production identities remain unset and no
checkpoint, data, gate, runtime, or promotion license was created.

A subsequent stage-order audit superseded only the prior V5 one-shot lifecycle
PASS. The all-at-once authority required hashes of runner/finalizer outputs
before those outputs could exist, and the publisher required G3 before it could
create the passed-G2 candidate needed to enter G3. The staged source candidate
now uses separate G2/G3 runner-input and finalizer-evidence authorities, a
G2-candidate publisher with G3 explicitly pending, and a distinct G2+G3 full
promotion authority. G3 reopens and reconstructs the exact G2 candidate chain
before its own inputs. This candidate passes `20/20` focused and `60/60`
combined CPU-only tests, but remains pending different-agent review with all six
production authority identities unset. Exact identities are in
`docs/lewm_go2_shared_jepa_v5_staged_lifecycle_candidate_2026-07-13.md`.

The reviewed G3 V1 audit was executed once and is recorded in
`docs/lewm_go2_g3_exact_physical_equivalence_v1_result_2026-07-13.md`. The
memory exactly matched the independent 89/69 implementation in all 24 scenes,
admitted zero unsafe FREE cells, and matched all 192 A* probes. It nevertheless
retained only `90/96` claim endpoints: `2/4` in scene `3e28c26ef602` and `0/4`
in scene `5689fb82c098`. The registered 0.10 m full-square raster plus
morphology disconnected the spawn component even though the analytic 0.47 m
disc-clearance component retained all endpoints. Both candidate and legacy
gates therefore remain failed; the V1 artifact is immutable.

A post-result development diagnostic changed only the cell size to 0.05 m,
retaining full-square evidence, the 0.47 m radius, occupied-first morphology,
four-connectivity, and exact LOS. The independently derived 313/277 morphology
retained `96/96` endpoints across the same 24 scenes with zero unsafe FREE.
This resolution is also the already-frozen V4 observable-evidence source grid.
An additive, versioned G3 V2 profile and audit are now required; the diagnostic
does not self-amend V1 or authorize learned output.

The first G3 V2 source candidate was independently reviewed and is `BLOCK`, as
recorded in
`docs/lewm_go2_g3_exact_physical_equivalence_v2_source_review_2026-07-13.md`.
Its offset geometry was numerically correct, but it replaced the preregistered
canonical `6fa138...` / `a18c08...` / `2b00cb...` identities, omitted the
distinct configuration-frame identity/revision, admitted forged and stale
snapshots, reused mutable production supports in its independent oracle,
omitted the governing design document from the captured graph, and exposed an
imported launcher substitution path. The `71/71` passing source tests therefore
do not license execution. The authoritative V2 audit remains unrun, its output
path remains absent, and V1 remains immutable.

## Iteration Rule

For every iteration:

1. name the failed gate and failure class;
2. state one falsifiable mechanism change;
3. choose the smallest scene-disjoint test that can reject it;
4. run offline gates before closed loop and fast gates before full sweeps;
5. record positive and negative results in this ledger;
6. promote only when the preregistered threshold passes.

This document is the governing plan until superseded by a dated amendment that
records why a contract or threshold changed.
