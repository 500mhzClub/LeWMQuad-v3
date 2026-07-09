# Go2 JEPA Navigation Generalization Execution Contract

Date: 2026-07-09

Status: active

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

| Arm | Strict claim+LOS events | Controller sightings | Median final coverage | Mean coverage AUC | 4/4 scenes |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 8/72 | 30/72 | 8.30% | 5.67% | 0/18 |
| novelty stack | 9/72 | 26/72 | 5.46% | 5.54% | 0/18 |

The old proxy figures were `16/72` and `13/72`. A distance-only recount reduced
them to `13/72` and `12/72`; the canonical distance-plus-LOS scorer reduces them
again to `8/72` and `9/72`. Baseline rejected events comprise five LOS failures
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

A physical claim requires the robot reference pose to be inside the fixed claim
radius of the requested beacon at the claim tick. Proxy state-machine accepts
are diagnostics only.

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

If auxiliary navigation heads fine-tune the encoder, retain the predictive and
anti-collapse objectives. Every checkpoint stores the backbone hash, training
scene IDs, corpus-plan hash, image geometry, and loss configuration.

### Traversability perception

Predict calibrated body-inflated free/occupied/unknown probabilities in an
egocentric local grid from spatial tokens. Labels use the actual camera/base
transform and canonical footprint. A conservative depth quantile is allowed as
an interim ablation, but hard scalar mean-depth rays are not the final map input.

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

### Exploration head

Generate reachable frontier candidates deterministically, then learn a
map-conditioned value for each candidate:

```text
expected new coverage + expected beacon discovery
    - path cost - turn cost - collision risk - uncertainty cost
```

Train first on oracle future gain labels across scene-disjoint mazes, then use
DAgger on states visited by the learned perception and exploration stack.
Deterministic A* (or equivalent) and the local follower remain outside the head.

### Beacon observation and claiming

Predict per-color presence, bearing, range distribution, and uncertainty from
spatial features. Fuse observations into the same online belief map. Train the
claim head on scene-disjoint data with the true physical radius as its target.
Remove analytic RGB masks and exact-vector propagation from the final learned
claim.

## Immutable Promotion Gates

Threshold changes require a dated amendment before looking at the candidate
result. A failed gate leads to diagnosis and iteration, not post-hoc metric
replacement.

### G0: benchmark integrity

- fixed-spawn/body-inflated/LOS/claim-radius audit passes every selected scene;
- train, development, and sealed-test scene hashes are disjoint;
- strict physical claims and normalized coverage are unit tested;
- every result records the full geometry, split, code, checkpoint, and corpus
  provenance.

### G1: oracle end-to-end ceiling

With exact occupancy and pose passed through the proposed planner/follower and
true claim contract, at least 17/18 current development scenes must finish 4/4
within 2,400 ticks. Target is 18/18. Failure blocks learned perception work on
that planner because it identifies a planning, following, or claim defect.

### G2: traversability perception

On untouched scene-disjoint validation scenes:

- planner-admitted free precision >= 99%;
- obstacle recall within 2 m >= 95%;
- useful traversable-space recall >= 90%;
- calibration metrics and threshold selection are stored;
- routes planned on predicted maps do not exceed the oracle-map collision gate.

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

### G3: fast closed-loop coverage

On a preregistered scene-disjoint development panel at 600 ticks:

- median normalized coverage is at least 2.0x the corrected baseline;
- the lower scene-clustered confidence bound is above no improvement;
- collisions, falls, and stalled-route fraction do not regress beyond the
  declared safety constraint;
- no scene-specific thresholds or optimistic unknown routing are used.

### G4: learned frontier value

- held-out frontier ranking beats distance-only and random reachable-frontier
  baselines on oracle future coverage/discovery labels;
- the learned head improves 600-tick normalized coverage over deterministic
  information-gain frontier selection;
- DAgger closes the model-visited-state gap without scene leakage.

### G5: target conversion

On scene-disjoint development rollouts:

- sight-to-valid-claim conversion >= 90%;
- false physical claim accepts < 1%;
- the learned observation/belief stack replaces fixed RGB masks and privileged
  target geometry;
- oracle-coverage runs isolate and pass the target stack before joint scoring.

### G6: full simulation promotion

Across at least 24 development scenes at 2,400 ticks:

- physically verified beacon claim rate >= 90%;
- at least 75% of scenes finish 4/4;
- no family has a hidden collapse masked by the aggregate;
- scene-clustered intervals, normalized coverage, completion time, collisions,
  stalls, and oracle gap are reported.

### G7: deployment robustness

- performance remains above the declared promotion floor under calibrated
  odometry noise and deployment-equivalent locomotion;
- simulator-only geometry guards are disabled;
- action-source tracing proves the runtime contract on every tick;
- a physical smoke protocol passes before multi-maze physical evaluation.

### G8: sealed final evaluation

Freeze code, geometry, thresholds, and all model hashes after G7. Run the
sealed test once. The paper's primary generalization result is sealed-test 4/4
scene completion with physically verified claims; all development results are
labeled as such.

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
| G0 benchmark integrity | PASSED (V4) | V3 physical invalidation recorded before sealed use. V4: 192/192 candidates pass disc and exact polygon SE(2), 138/24/30 disjoint roles, opaque screening/role commitments, strict scorer and provenance tests | Preserve the freeze; materialize development only and pass the v4 oracle |
| G1 oracle ceiling | PASSED (V4) | Geometry-v2 v4 dev: 96/96 claims, 24/24 all-claim through the shared `OnlineBeliefMap`; zero stalls, center collision attempts, or actual-yaw polygon collision segments | Preserve as the planner ceiling; proceed to G2 |
| G2 traversability | IN PROGRESS / dataset v2 ADEQUATE, first candidate FAILED offline | Dataset v2 built per preregistration (96 scenes, 5,641 rows, role commitments persisted, 16 recorded at-most-64 shortfalls); adequacy floors pass 8-46x over with zero G2 contact; 2-epoch smoke passed data/calibration/serialization. 20-epoch dev-only candidate: 0/288 threshold candidates pass (free precision 0.64 vs 0.99, obstacle recall 0.59 vs 0.95, traversable recall 0.31 vs 0.90, ECE 0.30); untouched G2 never read | 60-epoch probe separates training- vs data-limited; only a data-limited verdict triggers the preregistered 128-row rebuild |
| G3 fast coverage | BASELINE LOCKED / v0 FAILED | V3 fast8 at 600 ticks: 2.63% median coverage, 1.75% mean AUC, 4/32 claims, 0/8 solves; ray-v0 previously reduced raw cells | Pass G2, then exceed 5.263% median with CI and collision constraints |
| G4 frontier value | NOT STARTED | Current frontier is heuristic | Build oracle labels and candidate scorer |
| G5 target conversion | FAILED | Strict conversion 43-46% on full18 development data | Scene-disjoint observation and claim stack |
| G6 full simulation | FAILED | 0/18 scenes solved, 13/72 strict baseline claims | Complete G1-G5 |
| G7 deployment robustness | NOT STARTED | Kinematic exact-pose evidence only | Calibrated noise, locomotion, physical smoke |
| G8 sealed final | FROZEN / NOT OPENED | V3 sealed commitment invalidated unopened. V4 has 30 opaque committed scenes under final geometry v2 | Do not open or evaluate until G7 passes and all model/code hashes are frozen |

## Required Artifact Provenance

Each generated dataset, checkpoint, and result must contain or reference:

- schema/version and creation timestamp;
- git commit plus dirty diff hash;
- command/config with defaults resolved;
- corpus plan and split-manifest hash;
- exact train/validation/development scene IDs;
- geometry-contract hash;
- encoder and parent-checkpoint hashes;
- random seed and whether it affects execution;
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

Strict learned-baseline evidence:

- artifact: `.generated/strict_scores/phase4_full18_v1.json`;
- SHA-256:
  `eb796aea694c260f0da1a2e36404e86631008eac107f0adcc1061c9f11abfe2c`;
- baseline: 8/72 strict claims, 0/18 all-four, 8.30% median final coverage,
  5.67% mean normalized AUC;
- novelty stack: 9/72 strict claims, 0/18 all-four, 5.46% median final
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
  AUC, 4/32 strict claims, 0/8 all-four scenes;
- canonical geometry diagnostic: 2,002 occupied-space crossing intervals out
  of 4,800 logged intervals. G3 cannot pass by increasing this count.

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
