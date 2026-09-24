# Go2 G3-G5 first-principles gap audit

Date: 2026-07-11

Status: reviewed implementation plan. This document records development gaps
and dependency order; it does not pass or relax any gate.

## System chain

The required causal chain is:

```text
RGB -> observable physical evidence -> persistent physical memory
    -> robot-configuration snapshot -> safe route
    -> useful viewing pose -> beacon observation
    -> reversible target belief -> approach pose -> physical claim
```

The exact-map oracle proves the environment, primitive executor, and a
privileged planner can finish the development task at 96/96 claims. It does
not prove that the current learned runtime implements the chain above without
hidden geometry.

## G3: physical evidence and configuration memory

The current `OnlineBeliefMap` is still effectively one planning layer. Existing
adapters mix per-frame occupancy, body inflation, learned contradictions,
traversal evidence, and execution blocks. That makes it impossible to tell
whether a cell is physically observed, geometrically safe for the whole body,
or merely traversed once.

The promoted runtime requires:

1. A content-addressed map-frame contract: reset-local odometry frame, map
   origin, 0.10 m cell lattice, body-to-map transform, and pose provenance.
2. Pure physical-evidence projection with asymmetric rules. Rotated FREE
   evidence must completely cover a destination square across the admitted
   pose-uncertainty set; OCCUPIED evidence is a conservative union of surface
   witnesses.
3. Independent stores for learned physical evidence, measured traversal
   evidence, and execution/contact constraints. Traversal may correct learned
   evidence only inside the measured swept polygon and may never erase an
   execution block.
4. An immutable `ConfigurationSnapshot` derived after fusion. FREE requires
   all 89 closed-square-intersection support cells to be confirmed physical
   FREE. OCCUPIED uses the separate 69-center support and execution blocks.
   Everything else is unresolved.
5. Planning, connected components, frontiers, and A* only over a snapshot
   bound to one memory revision and both morphology hashes. Stale paths fail
   closed.
6. A zero-inflation exact-physical adapter that traverses precisely the same
   transaction, morphology, snapshot, frontier, and A* APIs as learned
   evidence.
7. A deployment-valid cold start based on an explicit yaw scan or a measured
   reset certificate. Unknown space is never silently treated as traversable.
8. A nonprivileged local follower that consumes adjacent snapshot path cells
   and estimated pose. It must not simulate candidate actions against hidden
   scene geometry.

The proven G1 follower is not itself eligible for G3: its path-target and
primitive-selection functions query privileged exact geometry. Its environment,
primitive mechanics, traces, and scoring can be reused; its controller cannot.

An initial development-only geometry probe also rejected a superficial adapter
swap. Center-sampled zero-inflation physical occupancy followed by the frozen
89-cell FREE support disagreed with the existing 0.47 m canonical grid on all
24 development scenes, with 603-2,202 false rejects and 14-95 false admits per
scene. Exact-physical morphology equivalence is therefore the first executable
G3 gate. If the precise square/surface implementation still cannot equal the
canonical planning reference, the equivalence definition needs a dated
amendment before learned G3 output.

## G4: viewpoint exploration

No promotable exploration head currently exists. Legacy frontier and novelty
mechanisms choose cells without a terminal viewing direction, use optimistic
unknown traversal in some branches, periodically inject scans, clear visit
state, or rely on privileged scene reachability. They remain negative evidence.

The G4 action is a safe option, not a primitive:

```text
FrontierViewpointCandidate(
    snapshot_revision,
    morphology_hashes,
    reachable_cell,
    yaw_index,
    safe_path,
    path_cost,
    turn_cost,
)
```

Use a frozen 16-heading world-frame lattice. Candidates may include a new
reachable cell, a new yaw at a known cell, or a current-cell scan yaw, but every
route cell must be connected confirmed configuration-FREE. Candidate identity
and tie breaks must be deterministic and bound to the snapshot revision.

First implement a deterministic information-gain baseline from deployment-valid
memory only. Its utility combines expected new swept coverage, expected physical
entropy reduction, uniform-prior beacon-discovery opportunity, route cost, turn
cost, uncertainty, and staleness. Freeze every weight, normalization, range,
candidate cap, and tie rule on train roles before output.

The learned `FrontierViewpointValueHead` ranks the exact same candidate set. It
consumes frozen shared-JEPA/map features, evidence strength/diversity/age,
configuration state, connected component, execution blocks, coverage/view
history, pose uncertainty, candidate yaw, and safe-path features. It does not
predict primitive actions directly and cannot fine-tune the qualified shared
encoder after G2.

Oracle counterfactual labels are train-only and score candidates without adding
them: new swept coverage, newly observable physical cells, new valid beacon
opportunities, option time, terminal yaw error, collision, stall, and clearance.
DAgger then labels states visited by the learned option policy to close the
offline/on-policy gap.

Required gates include:

- deterministic safe candidate generation and stale-snapshot rejection;
- an oracle ranker over the runtime candidate set reaching 96/96 visibility
  opportunities within 2,400 ticks before training;
- positive paired scene-cluster improvement over deterministic information
  gain at 600 ticks, with no opportunity collapse;
- 96/96 opportunities and 24/24 all-four scenes at 2,400 ticks;
- matched random-reachable, distance-only, information-gain, cell-only,
  coverage-only, discovery-only, no-DAgger, map-only, current-frame-only,
  exact-map, and oracle-selector controls.

## G5: target belief and physical claiming

The current target stack is not promotable:

- runtime observation is an analytic RGB color mask;
- target memory monotonically averages detections into one point or Gaussian,
  so contradictory modes collapse while confidence rises;
- the old claim head is trained on distance-only privileged labels with
  row-random splits;
- controller claims, visual proxy claims, and physical verification are mixed;
- existing scorers disagree on identity, LOS, and heading.

One shared fail-closed physical evaluator must take requested target identity,
claimed identity, full-precision `(x,y,yaw)`, and the physical manifest. It
computes and records:

- exact object identity;
- inclusive distance `<=1.20 m`;
- zero-inflation physical line of sight with obstacle, distractor, and other
  beacon occlusion;
- inclusive wrapped absolute bearing `<=0.25 rad`;
- factor decisions, rejection reasons, and final acceptance.

The strict scorer, oracle, eligibility audit, and runtime verification must all
call that evaluator. Controller-declared state and ground-truth-verified state
remain separate in memory and traces.

The learned observation head consumes detached shared-JEPA features once per
tick and emits per-color presence plus calibrated bearing and range
distributions. Scene-disjoint labels require corrected camera frustum,
first-surface object visibility, visible support, identity, and hard occlusion
negatives. An RGB mask or second visual encoder is not a promotion path.

Target memory is a normalized sparse world-cell posterior plus unlocalized mass
for each color. Positive observations create or strengthen modes; a negative
view downweights only cells that the current physical map says should have been
visible. Connected components remain competing hypotheses with mass,
covariance, evidence diversity, age, and uncertainty. Incompatible modes are
never averaged.

The target router selects safe `(cell,yaw)` reacquisition or claim poses around
hypotheses. It never routes into the target or through optimistic unknown
space. A separately calibrated `ClaimReadinessHead` may schedule attempts, but
physical success is scored only by the shared evaluator.

G5 isolation order is:

1. shared evaluator parity and the unchanged 96/96 oracle regression;
2. oracle observation plus learned downstream belief/router at 96/96;
3. exact-map, no-beacon-anchor coverage with learned observation and target
   interruptions;
4. learned observation precision at least 99%, recall at least 95%, confirmed
   target conversion at least 90%, false physical accepts below 1%, and 96/96
   development claims before joint promotion.

## Dependency order

The executable order remains:

1. pass the camera-centered perception/N32/G2 sequence;
2. prove exact physical-to-configuration equivalence;
3. build the two-layer checkpoint-v5 runtime and nonprivileged G3 runner;
4. implement and beat deterministic viewpoint information gain at G4;
5. unify physical claiming, then add learned observation and reversible target
   belief at G5;
6. run joint G6, calibrated robustness G7, and one fresh guarded G8.

Historical 8/72 distance-plus-LOS claims, 13/72 distance-only claims, 30/72
controller sightings, fast8 4/32 claims, and the quoted 53% proxy conversion
remain diagnostics. They are not heading-aware physical completion results.

This audit opened no G2 or sealed payload and produced no held-out model output.
