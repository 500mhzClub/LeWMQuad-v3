# Go2 G5 reversible target-belief implementation plan

Date: 2026-07-12

Status: pre-implementation, no learned G5 output authorized

## Purpose

G5 must convert visibility opportunities into physically verified beacon
claims without analytic RGB masks, privileged beacon vectors, irreversible
point averaging, or controller-authored ground truth. It begins only after the
shared encoder/perception and revisioned physical/configuration memory have
qualified at G2/G3 and the viewpoint policy has produced the required 96/96
development visibility opportunities at G4.

The learned and deterministic chain is:

```text
detached shared-JEPA spatial features
  -> per-color presence, bearing, range, and uncertainty distributions
  -> reversible sparse world-cell posterior + unlocalized mass
  -> competing target hypotheses
  -> confirmed-free reacquisition or claim viewing pose
  -> deterministic safe path/follower
  -> controller claim attempt
  -> canonical physical evaluator
  -> separate verified-claim ledger
```

## Rejected legacy behavior

`lewm.planning.online_belief_map.TargetBelief` represents one Gaussian and
`fuse_target_observation` precision-averages every observation into it. That is
useful historical infrastructure but cannot satisfy G5 because incompatible
modes collapse while covariance shrinks. Its single `claimed` flag also cannot
prove whether a controller declaration was accepted by the physical evaluator.
It must not be adapted into the promotion path.

The final G5 path also excludes:

- fixed or thresholded RGB color masks;
- a second image encoder;
- exact beacon coordinates or simulator visibility at inference;
- negative updates to cells not predicted visible by the current physical map;
- routing through UNKNOWN configuration cells or into the target cell;
- irreversible target deletion after a rejected claim;
- evaluator feedback used to train or steer the same episode;
- row-random train/validation splits.

## Learned observation head

`TargetObservationHeadV1` consumes detached frozen shared-JEPA patch/BEV
features once per tick. The qualified shared encoder and V4 evidence head remain
immutable. For each of the four task colors it emits:

- calibrated presence probability;
- a categorical bearing distribution over a frozen camera-frame angular grid;
- a categorical range distribution with an explicit beyond-range/unlocalized
  bin;
- aleatoric concentration or log-variance for bearing and range;
- an observation-quality probability used only as fusion strength.

The head may share a small spatial decoder across colors, but color outputs and
the complete parameter count are frozen before training. Presence, bearing,
range, uncertainty objective weights, optimizer, schedules, seeds, early-stop
rule, and calibration method are preregistered before any development output.
The head cannot update shared-JEPA or V4 parameters.

## Label contract

Labels are constructed only from scene-disjoint train roles using the corrected
camera calibration and first-surface visibility contract. A positive example
requires all of:

- correct task-object identity and color;
- projection inside the actual camera frustum;
- first-surface physical visibility after walls, obstacles, distractors, and
  other beacons;
- minimum visible support frozen before labels are built;
- finite camera-frame bearing and range derived from the same full-precision
  pose/calibration contract used by runtime projection.

Hard negatives include in-frustum occlusion, same-color distractors, wrong-color
task objects, near-boundary partial support below threshold, and visually
similar materials. Scene IDs, beacon coordinates, segmentation IDs, and
visibility labels are never model inputs.

Dataset construction is CPU-parallel with at most six spawned workers and all
BLAS/OpenMP thread counts fixed to one. It records exact source hashes, selected
scene IDs, object identities, image hashes, calibration hashes, first-surface
geometry hashes, and zero non-train/G2/sealed access. Neural training may use
only GPU0. GPU1/Raphael is rejected.

## Reversible target memory

The runtime owns one normalized posterior for each color:

```text
TargetPosterior(
    physical_revision,
    configuration_snapshot_sha256,
    sparse_world_cell_mass,
    unlocalized_mass,
    positive_evidence_records,
    negative_evidence_records,
    competing_components,
)
```

Mass is never represented solely by a mean. Every update is transaction-bound
to the current physical revision, map frame, pose provenance, camera
calibration, model checkpoint, and observation payload. Duplicate observation
IDs and semantically duplicate payloads are rejected. Stale or mutated
snapshots are rejected before mutation.

Positive observations project bearing/range distributions through the current
pose uncertainty into sparse world-cell likelihoods. They transfer bounded mass
from the unlocalized pool and strengthen compatible existing cells. They do not
average separated supports into an intermediate location.

Negative observations downweight only cells satisfying all of:

- currently in the camera frustum under registered pose/calibration
  uncertainty;
- expected visible under the current zero-inflation physical map;
- not occluded by confirmed physical OCCUPIED evidence;
- inside the learned head's qualified range/support region.

Cells outside that set and unlocalized mass are unchanged before
renormalization. A later positive observation can restore a downweighted mode.
No negative update can set a cell exactly to zero.

Four-connected posterior components remain separate hypotheses. Each records
mass, weighted mean/covariance for reporting only, positive/negative evidence
counts, evidence-source diversity, age, and uncertainty. Component selection
never replaces the underlying sparse posterior.

Strict canonical serialization includes the entire posterior, unlocalized
mass, every evidence transaction and duplicate key, map/pose/calibration/model
bindings, controller-attempt ledger, verified-claim ledger, revision, and
taint/ablation state. Round-trip reconstruction must reproduce the exact
content hash.

## Target routing

The deterministic router consumes an immutable configuration snapshot and
candidate target components. It generates `(configuration_cell, yaw)` options
that:

- are in the robot's connected confirmed-FREE component;
- never occupy the hypothesized target cell;
- have a complete revision-bound A* path through confirmed FREE;
- face the hypothesis with the registered claim-bearing tolerance;
- are either conservative reacquisition views or candidate claim poses;
- retain clearance, route cost, pose uncertainty, target mass, expected
  visibility, and heading margin in their score and receipt.

Every execution receipt binds the target posterior revision, configuration
snapshot, morphology hashes, candidate-set hash, actual start cell/yaw, exact
path, terminal yaw, and target identity. Memory or target revision changes make
the option stale.

`ClaimReadinessHeadV1`, if needed after deterministic controls, may rank these
same candidates or schedule an attempt. It may not change the candidate set,
claim tolerances, route safety, or evaluator decision.

## Claim-state separation

Three namespaces are distinct:

1. `controller_claim_attempt`: what the policy requested;
2. `physical_claim_evaluation`: the immutable canonical evaluator result;
3. `verified_claim_credit`: a first accepted and credited evaluator event.

Controller attempts never mutate verified state. The verified ledger is written
only by calling `evaluate_physical_claim_trace` with the full raw attempt trace
and immutable physical manifest, then validating the returned evaluator,
manifest, task-set, trace, summary, and event content hashes. A fabricated
`accepted=true` mapping or a duplicate accepted-but-not-credited event is
rejected.

The evaluator contract remains exactly:

- requested and claimed target identities resolve to the same task object;
- inclusive distance `<=1.20 m`;
- zero-inflation physical line of sight;
- inclusive wrapped absolute bearing `<=0.25 rad`;
- full-precision `(x,y,yaw)` with canonical binary64 commitment.

Evaluator feedback remains absent from controller input in the evaluated
episode.

## Ordered implementation and isolation gates

1. Implement and adversarially review pure sparse-posterior transactions,
   duplicate rejection, positive/negative reversibility, components, strict
   serialization, and controller/verified claim separation.
2. Implement the exact-observation adapter and safe target router. With oracle
   observations and the canonical G1 planner/follower/evaluator, finish 96/96
   development claims.
3. Build and independently audit the train-only target-observation dataset.
4. Train the detached observation head with two preregistered seeds and
   scene-clustered selection/calibration; do not access G2 or sealed roles.
5. On scene-disjoint development visibility opportunities, require per-color
   and aggregate precision `>=99%`, recall `>=95%`, and report bearing/range
   calibration, false-track creation/persistence/recovery, and every family.
6. On exact-map no-beacon-anchor rollouts, require 96/96 development claims
   with learned observation, belief, interruptions, routing, and canonical
   verification.
7. With the qualified G2-G4 stack, require confirmed-target-to-valid-claim
   conversion `>=90%`, false physical accepts `<1%`, and no scene/family
   collapse before joint G6 promotion.

Required controls use the same scene clusters and opportunity denominator:

- oracle observation + learned belief/router;
- learned observation + oracle sparse posterior;
- positive-only memory;
- irreversible single Gaussian;
- no negative evidence;
- global negative evidence;
- current-frame-only;
- map-only and current-frame-only features;
- exact map and learned physical map;
- deterministic readiness versus learned readiness;
- random reachable, nearest-mode, and highest-mass routing;
- evaluator-disabled dry attempts that cannot create credit.

No joint learned rollout can satisfy G5 until every isolated gate above passes.
No G5 artifact licenses G2 retraining or a new sealed evaluation.

## 2026-07-12 Stage-1 Independent Source Review

The first foundation implementation was independently reviewed at these
immutable source identities:

- plan SHA-256:
  `12430c21563c0bcc96744a22a65ca70780427b1cea0ee12b749ac4a175bca4e9`;
- implementation SHA-256:
  `0e660a6e94b1483cde1e56267f503e3b5e6c35e81a2feab4e4371d4718bd6d10`;
- test SHA-256:
  `ae87eb7fe7595c61452713ec65f96eb91ec384d3c855f1bcd81c99bcee47a525`.

Verdict: **BLOCK**. The existing 12 synthetic tests passed, but adversarial
checks demonstrated that the source could:

- derive the expected task set from the raw claim trace being judged and
  credit a one-task trace against a two-landmark manifest;
- evaluate against a caller-substituted physical manifest not bound to the
  episode or memory;
- trust caller-constructed physical contexts and negative-visibility cells;
- accept the same evidence payload again under a new tick and identity;
- exponentially erode separated positive modes and eventually underflow a
  repeatedly downweighted mode to exact zero;
- omit rejected and unverifiable evaluations from an immutable physical
  evaluation ledger and expose evaluator feedback to same-episode control;
- accept a separately constructed snapshot clone by content hash, return
  mutable ledger aliases, and mutate state before all commit checks passed;
- omit strict whole-memory serialization and underbind configuration, task,
  issuer, manifest, taint, and pose advancement semantics.

These are foundation defects, not failures of a learned head. No exact-router,
learned-observation, development rollout, GPU, G2, held-out, or sealed output
is authorized from this version. The successor must add authoritative G3-issued
context/visibility capabilities, manifest/task-set-bound observer-only claim
evaluation, separate immutable attempt/evaluation/credit ledgers, payload-level
duplicate rejection, numerically nonzero reversible posteriors, atomic
transactions, instance-issued snapshots, defensive returns, and strict
whole-memory round-trip commitments before another independent review.

## 2026-07-13 Stage-1 Remediation Result

The successor foundation closed the review defects and the later online-scaling
failure. A separate reviewer returned **PASS** at these exact identities:

- implementation SHA-256:
  `366dfa92e8178d45b01e36552d67b52da0ad2805a7b0cc1abfee62527918459f`;
- focused-test SHA-256:
  `b4493a690cbaa6ea85eb9200eecbdd40d1a4cf305a42f868ef68d13287edde67`.

Verification passed 43 focused G5 tests and 319 canonical claim tests. Direct
adversarial probes confirmed manifest/task authority, instance-issued G3
positive/negative evidence, payload replay rejection, nonzero reversible
posteriors, three immutable claim namespaces, observer-only finalization,
atomic state changes, clone/alias rejection, strict round trips, pose-only
context advancement, and stale-G3 rejection.

Online mutation now uses revision leases, rolling evidence commitments, and
cached component metadata. It performs no history replay or canonical
whole-state hash. Measured CPU cost was 209.8 microseconds per update for 300
updates and 183.3 microseconds per update over the final 700 of 1,000 cumulative
updates; a hypothesis read was about 61 microseconds. The exhaustive audit path
remains separate: serialization at 1,000 observations took 42.3 ms and replayed
exactly 1,000 chain entries plus 1,000 posterior entries with one canonical
hash. A cross-layer G3 full-state hash leak found during review was removed and
retested across apply, snapshot, hypotheses, serialization, and restore.

This passes only ordered implementation step 1. It creates no learned target
head and grants no rollout or promotion license. The exact-observation adapter,
safe target router, 96/96 oracle-observation gate, train-only observation
dataset, detached learned head, calibration, and all remaining G5 isolation
gates are still required.

## 2026-07-13 Evidence-authority reopening

Status: **BLOCK for G3/G5 evidence authority; posterior mathematics and scaling
result retained**.

The later executor/reset and V5 reviews established that importable global
tokens and public factories accepting hash-shaped producer identities are not
independent authority. Reapplying that standard exposes the same pattern in
the stage-1 G5 source:

- `_G3_ISSUER_BINDING_CAPABILITY`, `_CANONICAL_EVALUATION_CAPABILITY`, restore,
  and state-commit capabilities are importable module globals;
- public `TargetMemoryContextIssuer.bind_g3(...)` accepts caller-selected
  issuer, frustum, physical-LOS, positive/negative producer, calibration, and
  checkpoint hashes;
- its returned public `G3TargetMemoryContextProducer` accepts caller-created
  candidate domains, localized target distributions, and negative visibility
  probabilities; no canonical G3 runner outcome is reopened;
- issuer capability objects and issuance registries are accessible object
  attributes and therefore require copy/clone/global-token adversarial review.

The prior 43-test/performance result still proves the sparse posterior,
reversibility, non-underflow, component cache, and strict audit serialization
implementation. It no longer proves that evidence entering that structure came
from G3 perception or visibility. Before ordered step 2, a successor must bind
contexts and positive/negative observations to the same fixed runner-owned raw
outcomes required by the V5 and learned-projection plans, remove importable
admission tokens, and pass copy/clone/caller-distribution reproductions under a
different reviewer. No prior G5 rollout existed, so no scientific result is
invalidated; the source-foundation promotion status is corrected before use.
