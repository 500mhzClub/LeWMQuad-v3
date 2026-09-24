# Go2 shared-JEPA G2/G3 implementation plan

Date: 2026-07-11

Status: active implementation plan; promotion thresholds remain governed by
`docs/lewm_go2_generalization_execution_contract_2026-07-09.md`

## First-principles information flow

The promoted system must implement one deployment-valid chain:

```text
current RGB
  -> one online JEPA vision encoder
       |-> action-conditioned predictive BEV branch
       `-> categorical observable-physical-evidence head
              -> calibrated multi-view physical evidence in OnlineBeliefMap
              -> deterministic 0.47 m configuration morphology
              -> connected configuration-free frontier and graph search
              -> learned frontier value, target observation, and claim heads
```

The encoder may have an EMA target copy for JEPA training. It must not have a
second independently trained runtime perception encoder. Simulator geometry,
depth, target coordinates, scene IDs, and optimistic unknown routing remain
forbidden at deployment.

## Current reusable pieces

- `EgomotionBevJepa` already has one online `VisionEncoder`, an EMA target
  encoder/BEV decoder, action-conditioned prediction, equivariance,
  counterfactual action separation, and anti-collapse losses.
- `CategoricalRadialPerceptionFullRay` proves the physical spatial mechanism
  only through N=16. Both exposure-matched N32 candidates failed the registered
  all-family fit gate. It currently owns a separate encoder and is not a JEPA
  head.
- `OnlineBeliefMap` already stores reversible occupancy evidence, traversal,
  pose, targets, conservative frontiers, and routes.
- `PerceptionToBeliefMapAdapter` and `EgomotionBevJepaRuntime` currently assume
  each frame already predicts body-inflated configuration occupancy. That is
  incompatible with observable physical labels and multi-view-first fusion.
- The runtime accepts legacy checkpoint-v2 and intentionally rejects physical
  checkpoint-v4. A promoted shared model requires a new strict schema.

## G2 shared model

Dataset-backed shared-JEPA training is blocked until the ordered pose/hierarchy
intervention passes the complete N32 ladder, including both registered seeds
and both train-role holdouts. Until then, the following width-24 V2 mechanism is
the controlled N32 base, not a promoted or N32-proven model:

- input 112x112, patch 7, 16x16 ordered tokens;
- encoder width/depth/heads 192/6/6;
- token projection width 24;
- five distinct registered vertical anchors;
- frozen polar/Cartesian factorization and support mask;
- context width 64;
- full-ray radial dilations 1, 2, 4, 8, 16, and 32;
- existing angular context and UNKNOWN/FREE/OCCUPIED class order.

Implementation order:

1. Extract a token-consuming `CategoricalRadialFullRayTokenHead`. Keep the
   standalone model as a compatibility wrapper and prove bit-exact migration.
2. Add token/BEV hooks to `EgomotionBevJepa` without changing its default
   outputs or state loading.
3. Add `SharedCategoricalRadialJepa`: one online encoder supplies both the JEPA
   BEV branch and the categorical head. Encode each current/next frame once.
4. After the complete N32 ladder passes, train from a fresh registered
   initialization. N32 weights license the architecture only and must not
   initialize the full candidate.
5. Keep JEPA prediction, equivariance, action contrast, variance, and balanced
   hierarchical physical-evidence losses active jointly.

The corrected 224x168 camera frames need no rerender. The loader may resize
them to the registered 112x112 tensor; normalized horizontal and vertical
coordinates retain the separately registered camera FOVs. Patch 7 preserves
the 16x16 token lattice validated through N=16 and retained by both controlled
N32 candidates. Moving to 128 or patch 14/16 would change that evidence bundle.

If the pose audit licenses dynamic projection, the model input is the camera
origin and basis in the yaw-aligned body frame reconstructed from calibrated
camera extrinsics plus deployment-valid IMU/proprioceptive attitude and height.
Recorded renderer world camera pose may construct training/audit inputs but is
forbidden at promoted runtime. The checkpoint binds the sensor fields,
calibration, transform, units, synchronization, and missing-data behavior.

Required model tests:

- standalone wrapper and extracted token head are bit-exact after state copy;
- only one online encoder is registered;
- JEPA and categorical gradients both reach that encoder;
- one encoder call per frame supplies all online consumers;
- factorization, anchors, support, tensor shapes, and dilations fail closed;
- legacy `EgomotionBevJepa` forward/state behavior remains unchanged.

## Training and one-shot G2

Use a new checkpoint schema, `lewm_go2_shared_categorical_jepa_checkpoint_v5`.
It must record the model family, tokenization, factorization/anchor/dilation
hashes, proof of shared encoder consumers, the prerequisite passing N32
two-seed/holdout license, every JEPA
loss/control, selected state hash, calibration, thresholds, dataset/access
provenance, and `runtime_ready=false` until G3 is implemented.

Before dataset-backed output, write a dated full-training amendment fixing
seeds, schedule, selection, escalation, JEPA-health gates, and G2 execution.
Then:

1. Run a development-only wiring smoke with zero G2 payload access.
2. Train only on the train role.
3. Rank checkpoints only on checkpoint-selection physical evidence, subject to
   mandatory JEPA-health eligibility: noncollapsed target variance/rank,
   predictor improvement over persistence, and action/counterfactual
   sensitivity.
4. Freeze the selected state hash.
5. Fit vector calibration and select thresholds only on the calibration role.
6. Require aggregate and per-family development gates before G2.
7. Commit the final checkpoint and execute G2 once. Do not retrain, refit,
   retune, or substitute after contact.

Before the one-shot G2 execution, train one development-only matched ablation
from the same registered initialization, data order, architecture, supervised
losses, and budget with JEPA prediction/anti-collapse losses disabled. It cannot
select or replace the promoted checkpoint. Report scene-clustered physical
evidence and JEPA-health deltas so any claim that predictive representation
learning improves generalization is causally supported rather than inferred
from shared wiring alone.

The promoted arm alone selects its checkpoint under the registered physical
evidence plus mandatory JEPA-health rule. The ablation is evaluated at that
exact preselected update; its curve and outcomes cannot choose another step.
Both arms then use the same preregistered calibration algorithm on the same
probability-calibration role. This keeps the comparison matched without asking
the no-JEPA arm to satisfy an inapplicable JEPA-health eligibility rule.

Replace the filename-local G2 attempt marker with an atomic role-global
registry keyed only by dataset-manifest SHA, G2 role commitment, and immutable
evaluation-protocol generation. Model family, source-checkpoint/state hashes,
and code hashes belong inside the one immutable attempt record; changing them
must not create another authorization namespace for the same role. The registry
must reserve the role before G2 byte access, reject every later checkpoint or
renamed copy, and survive process crashes.

## G3 physical memory and morphology

The map needs two explicit semantic layers.

### Physical evidence layer

- Perception predicts `observable_physical_occupancy_v3` with zero target
  inflation.
- Persistent and current-frame-only fusion modes are construction-time,
  provenance-bound choices.
- Learned free/occupied evidence, traversal evidence, and execution/contact
  blocks remain separately reversible.
- Rotated FREE projection is conservative: a destination physical cell is
  admitted only when the union of admitted source FREE squares completely
  covers its closed square. Center containment alone is insufficient.
  OCCUPIED projection may use conservative supercover.
- Projection must respect `BeliefMapConfig.origin_xy_m`.
- One observation fuses atomically and increments one map revision.
- Observation identity and calibrated translation/rotation diversity prevent
  repeated near-identical frames from accumulating as independent FREE proof.
- A stable initial stance and successful swept traversal add separate verified
  physical FREE evidence only over the actual measured support/swept body
  polygon. They never certify the larger yaw-invariant 0.47 m disc. Traversal
  may override contradicted learned evidence inside that measured support but
  never erases contact/execution blocks or their provenance.
- Every observation binds pose source, covariance, camera transform, timestamp,
  and synchronization. Promoted mode rejects exact simulator world pose;
  `exact_sim_odometry_ablation` is a distinct non-promoted mode.
- FREE projection must completely cover a destination physical cell for every
  transform in the admitted pose/camera uncertainty set; OCCUPIED projection is
  the supercover union over that set. Frames above the frozen uncertainty bound
  are rejected rather than fused.

Select learned-evidence admission thresholds, translation/rotation diversity,
pose-uncertainty limits, and contradiction recovery only from current physical
train sequences plus the registered probability-calibration role. The
bootstrap yaw sequence is derived from the frozen camera FOV and primitive
contract and may be checked only on physical-training scenes. Freeze all values
and hashes before any V4 G3 closed-loop output; checkpoint-selection, untouched
G2, V4 development outcomes, and legacy held-out results cannot tune fusion.

Cold start uses one of two explicit deployment contracts. A trusted reset may
seed a 0.47 m configuration-free region only when an external reset protocol
guarantees and records that clearance in both simulation and hardware. Without
that certificate, stance evidence covers only the actual body polygon and the
bootstrap scan must supply the remaining FREE support; failure leaves the
configuration cell unresolved and the run fails closed.

### Configuration layer

Apply deterministic asymmetric closed-disc morphology only after physical
evidence is fused. Freeze and hash two sorted offset sets. For physical-cell
size `s`, footprint radius `r=0.47`, and offset `(dx,dy)`, FREE support includes
the closed physical squares intersecting the disc:

```text
max(abs(dx)*s - s/2, 0)^2 + max(abs(dy)*s - s/2, 0)^2 <= r^2
```

OCCUPIED support uses the stronger center witness:

```text
(dx*s)^2 + (dy*s)^2 <= r^2
```

The radius, cell size, inclusive boundary rules, both offset lists, and both
kernel hashes are provenance. The dense
`derive_configuration_labels_from_fused_physical_raster` implementation is the
brute-force oracle.
A configuration cell is:

- blocked if any OCCUPIED-support cell is confirmed occupied, or if its body
  center has a separate execution block;
- free only if every FREE-support cell is confirmed free;
- unresolved otherwise.

Execution blocks are body-center constraints and must not be dilated a second
time. Successful swept-footprint traversal may clear contradicted learned
physical evidence under an explicit provenance record.

Raw physical maps reject frontier and path queries. Planning accepts only an
immutable `ConfigurationSnapshot` bound to a physical-map revision and both
morphology hashes. Current-frame-only mode replaces only learned physical
evidence; verified traversal and contact/execution layers persist.

Add configuration snapshots, connected-free components, frontier cells, and a
deterministic weighted A* path. For eight-connectivity use cardinal cost 1,
diagonal cost sqrt(2), an octile heuristic, no corner cutting, and stable
lexicographic tie breaking. Every frontier/path result binds the physical map
revision and morphology hash.

Required memory/runtime tests:

- both exact 0.47 m support boundaries versus the dense brute-force helper;
- complementary views certify configuration free only after persistent fusion;
- current-frame-only never inherits evidence from a prior frame;
- contradictory evidence recovery and no double dilation of execution blocks;
- repeated-view evidence cannot fake independent confirmation;
- rotated FREE projection proves full destination-cell coverage with no
  supercover or center-containment leakage;
- nonzero map origin registers correctly;
- serialization, revision/hash determinism, and duplicate-observation guards;
- connected configuration frontiers and deterministic A* costs/paths;
- exact zero-inflation scene evidence reproduces the canonical configuration
  grid, spawn connectivity, and oracle routes through this same path;
- checkpoint-v5 fails closed on wrong semantics, nonzero label inflation,
  failed G2, stale hashes, or malformed output geometry.

Use a separate checkpoint lifecycle. `load_g3_candidate` is development-only
and requires a passed immutable G2 report while `runtime_ready=false`.
`load_promoted` requires a derived immutable checkpoint/report binding the G3
pass and `runtime_ready=true`. Neither mode may infer readiness from a filename.

## G3 fast evaluation

Build a distinct G3 runtime and runner rather than extending the legacy
closed-loop benchmark. The controller boundary exposes only RGB,
deployment-valid odometry/IMU/proprioception, its memory, and execution
outcomes. Privileged geometry and exact simulator pose may execute physics and
score results but cannot choose actions. It must start coverage at tick zero;
privileged beacon anchors cannot choose actions. Claims remain opportunistic
diagnostics.

When the connected configuration component contains no safe multi-cell route,
run a frozen deterministic yaw-scan sequence, fuse admitted observations, and
recompute morphology. Failure after the full scan is an explicit cold-start or
perception failure, never a license for optimistic unknown routing.

Before learned output, freeze a hash-selected eight-scene V4 development panel
stratified by topology, reachable area, and obstacle density. V4 remains
development-only; the invalidated V4 sealed role is never used. Freeze the
panel, checkpoint, calibration/thresholds, morphology, planner/follower, pose
mode, and bootstrap seed.

A dated G3 execution binding must also freeze, before any panel output: the
corrected baseline artifact/configuration/source hashes on the new runner; the
visibility-opportunity evaluator and four-beacon-per-scene denominator; camera
frustum, physical LOS, range, and inclusive-boundary rules; confidence level,
paired scene-cluster interval method, resample/exact-enumeration count and seed;
and numeric collision, fall, stall, and route-failure tolerances. Until that
binding exists, the qualitative master thresholds are not executable.

Before any learned arm, feed exact zero-inflation physical evidence through the
same fusion, asymmetric morphology, snapshot, frontier, and A* implementation.
It must match canonical configuration geometry across all 24 development
scenes and pass cold start. Separately, rerun the privileged-target G1 oracle
through that mechanism and retain 96/96 claims; this is a non-G3 regression.
The exact-map G3 reference uses the same no-beacon-anchor controller interface
as learned arms and is scored on coverage and visibility opportunities, not
privileged claim completion.

Run paired arms with identical non-treatment settings:

- persistent learned physical fusion;
- current-frame-only learned perception;
- perception disabled;
- exact-map engineering reference.

Every step trace must bind observation, pose registration, physical map,
configuration snapshot, connected component, frontiers, selected target, A*
path/cost, action source, primitive, and execution outcome. Report final/AUC
coverage, ground-truth beacon-visibility opportunity coverage, stalls,
collisions, and per-stratum results. Apply the preregistered relative/absolute
G3 gate and paired scene-clustered confidence bound from the master contract.

## Downstream order

G4 trains only after the deterministic information-gain frontier-viewpoint
baseline is working through the promoted physical memory. Candidates are
reachable `(configuration_cell, yaw)` viewing poses or frozen scan sequences.
Use oracle future coverage/discovery labels, scene-disjoint ranking, and DAgger
on model-visited states. G5 then adds learned per-color presence, bearing,
range, uncertainty, reversible multimodal target beliefs, and LOS-valid claim
heads. Positive and negative observations update competing hypotheses; an
internal controller claim never mutates the ground-truth verified-claim field.
Joint G6 scoring begins only after G2 through G5 pass their isolated gates.

Before any learned G4 output, preregister the deterministic baseline's candidate
generator, yaw lattice/scan set, reachability filter, information-gain horizon
and observation model, coverage/discovery/path/turn/risk weights, normalization,
and lexicographic tie breaks. Learned candidates must be the same frozen set so
an apparent value-head gain cannot come from changing the action space.

After the one-shot G2 pass, the shared encoder, traversability head,
calibration, and admission thresholds are immutable for G3-G8. G4/G5 heads may
consume detached shared features but must not fine-tune that stack. Any proposed
fine-tune requires a dated preregistration and complete new selection,
calibration, development, and one-shot untouched-role qualification on a fresh
eligible G2 role before it can replace the frozen model.

## 2026-07-13 V5 production-authority review

Status: **BLOCK; production remains structurally disabled**.

The independent review confirmed that current `None` constants fail before
production file access, but reproduced five successor-boundary defects:

1. canonical-shaped caller mappings with no model or raw outcomes produced a
   perfect G2 pass; the finalizer also trusted a caller ledger's repeated scene
   SHA instead of reopening the bound file;
2. an unchanged copied authority module derived a new canonical root and fresh
   registry from its copied `__file__`;
3. the bound source inventory omitted the shared finalizer core and authority
   policy, even though those modules implement the actual decision;
4. a `../../escaped` registry namespace wrote a reservation outside the
   registry root; and
5. module-global token/issuance objects allowed a forged context constructed
   with `object.__new__` to pass when future authority was simulated.

The reviewed CPU suite passed 31 tests, but those tests blessed synthetic
self-reporting rather than model-bound raw outcomes. Required remediation is a
fixed external repository/registry authority, exact SHA namespace validation,
complete decision-source closure, non-caller-mutable factory issuance, and a
canonical runner whose instance-issued raw scene outcomes are independently
reopened and reduced by the G2/G3 finalizers. No G2 or runtime access is
licensed by this source work.

### Unreviewed successor candidate

The remediation candidate removes mapping-based production finalization.
Production G2/G3 accept only closure-issued immutable runner batches; the
runner opens the fixed-root checkpoint/model, role manifest, per-instance raw
outcomes, and ledger, regenerates actual-open events, and the finalizer derives
counts only from per-instance booleans. Synthetic issuance is test-only,
tamper-committed, and explicitly production-ineligible.

The installed root is hard-anchored, registry namespaces require an exact
lowercase SHA-256, the authority inventory now covers authority/main/registry/
runner/core/distinct wrappers, legacy caller-root APIs are tombstones, and
context issuance uses closure-held weak membership rather than a module-global
token/table. All six production constants remain `None`.

Candidate source SHA-256 values are:

- main: `5eeac43cb141b9d82b8b1e2ac504ca9dfe9c3a39a6f4ca61a77291a780e9c688`;
- authority: `a024b9e28c2572bfefa6cd630bf4f7798cae946b353e41033e884529bb0ca20c`;
- registry: `da9d7a8ed8e4ae21adf8efb01950c2e080559c2e92e89bac14e906db4efcdfdc`;
- runner: `99b712c713eee1dae466b85bd550c9967fb93f026bb129a039df7922195ecfa6`;
- finalizer core:
  `194daf3d86b0c351218d6c7bcddde5e96ac460a377170429ca4d78fdb598ad92`;
- G2/G3 wrappers: `a4db5ae914238da88c05ca281095aeccce7f9a054f46d07299b5f8ad7c7f01a7`
  / `c49adcc69fe1cb5d0b67b614bb40bcf6984f935720dbc62a9d6f7d5481a26ec9`;
- tests: `aac92de5d41564d3641b88247f5fd63c30f777d2f7001260409fca248b0ff863`.

The capped CPU suite passes 38 tests and compiles cleanly. This is not a PASS;
a different reviewer must rerun every prior bypass before any constants or
role access can be enabled.

## 2026-07-13 V5 one-shot execution candidate

Status: **candidate only; production execution remains disabled**.

The in-process successor described above was replaced after it was shown to
reopen caller-precomputed booleans rather than run the model, and after live
module substitution could turn synthetic batches into production-eligible
reports.  V5 now has three ordinary process entrypoints:

- `scripts/run_go2_shared_jepa_v5_gate.py {g2,g3}` reserves the fixed role,
  captures and rehashes the authority-listed runtime module graph, opens the
  fixed checkpoint and raw scene inputs, invokes `infer_one` exactly once per
  sorted raw instance, derives boolean metric outcomes from fixed rules, and
  exclusively writes per-scene outcomes plus a complete open ledger;
- `scripts/finalize_go2_shared_jepa_v5_gate.py {g2,g3}` independently reopens
  the fixed role manifest, ledger, raw inputs, checkpoint binding, and every
  per-scene outcome, reconstructs the open sequence and inference cardinality,
  then derives family counts, aggregate fractions, and the decision;
- `scripts/publish_go2_shared_jepa_v5_checkpoint.py` independently reopens the
  fixed checkpoint and passing G2/G3 reports and exclusively writes their
  publication binding.

The production model module no longer exports a context factory, checkpoint
builder/validator, or roundtrip helper.  The runner policy no longer exports a
canonical batch issuer/validator, the finalizer wrappers no longer export a
production mapping API, and the registry policy no longer exports a mutator.
Synthetic batch helpers remain available only for unit tests and always stamp
`synthetic_only=true` and `production_authority_eligible=false`.

The one-shot core is stdlib-only before captured runtime execution.  Its
synthetic-authority environment is explicitly test-only, rejects the canonical
repository root, and can never emit eligible evidence.  CPU subprocess tests
cover exact inference cardinality, precomputed-metric rejection, exclusive
one-shot reservation, caller path rejection, shadow-import substitution, the
two reproduced live-global substitutions, registry mutation, independent
finalization, and permanently ineligible publication.

All six production identities remain `None`; no dataset, checkpoint, G2/G3
role, Torch model, or GPU was opened by this source-only remediation.  A fresh
independent authority review is required before any identity may be frozen.

### 2026-07-13 staged lifecycle correction candidate

Status: **candidate only; independent review and production execution remain
blocked**.

A first-principles integration audit found that the preceding one-shot design
was not executable as a production sequence. Its single authority preflight
required the G2/G3 runner-ledger and final-report file hashes before the runner
could create those files. Its only publication transition also required both G2
and G3, although a passed G2 publication is the artifact that must qualify the
checkpoint for G3 evaluation. This was an actual dependency cycle, not merely a
missing configuration.

The successor uses six immutable, stage-specific authority revisions:

1. G2 and G3 runner-input revisions contain only already-existing role,
   checkpoint, captured-source, raw-scene, and predecessor-publication inputs,
   plus exclusive per-scene outcome and ledger paths. They cannot contain
   runner-ledger, outcome-file, or final-report hashes.
2. G2 and G3 finalizer revisions link the exact runner authority file, exact
   runner ledger, every exact per-scene outcome, and one exclusive final-report
   path. Reports commit both runner and finalizer authority identities.
3. The G2-candidate publisher revision links only the exact passed G2 finalizer
   authority and report. It emits `publication_kind=g2_candidate`, records G3
   as pending, and can exist before any G3 runner artifact.
4. The G3 runner revision links both that candidate publication and its exact
   publisher authority. Before opening G3 runtime inputs it reconstructs the G2
   report and candidate bytes from the full predecessor chain; every predecessor
   artifact opened by this check is included in the G3 runner ledger.
5. The full-promotion publisher revision requires exact passed G2 and G3
   finalizer authorities and reports, a shared checkpoint identity, and a
   distinct exclusive output path. A G2-only or candidate revision cannot be
   reused as full promotion.

The six source-bound production identities now name those six authority
revisions and all remain `None`. Synthetic revisions still require a foreign
temporary root and every synthetic runner, report, candidate, and promotion
artifact is stamped `synthetic_only=true` and
`production_authority_eligible=false`.

Focused CPU-only adversarial verification passes `20/20`. It covers the
complete staged synthetic workflow, future-output-field rejection, exclusive
path aliasing, post-freeze ledger/outcome tamper, candidate creation with G3
absent, wrong-revision reuse, G2-only full-promotion rejection, exact G2
candidate binding, reconstruction rather than trust of frozen candidate claims,
captured-inference substitution, independent finalization, and skeletal report
rejection. No repository dataset, real checkpoint, G2/G3 role, production model
inference/training, or GPU was opened.

Exact candidate identities and the independent-review rule are recorded in
[`lewm_go2_shared_jepa_v5_staged_lifecycle_candidate_2026-07-13.md`](lewm_go2_shared_jepa_v5_staged_lifecycle_candidate_2026-07-13.md).
