# Go2 held-out maze goal

Date: 2026-07-14

Status: **active execution goal; no new experiment authority granted by this file**

## Objective

Iterate from the ready-to-benchmark handoff to one authorized evaluation on a
fresh opaque held-out maze set.  Completion requires immutable evidence for the
camera ladder, Shared-JEPA training and G2 qualification, the real learned
two-resolution navigation runner, development and robustness gates, source and
model freeze, held-out custody, and the one-shot held-out result.

No held-out result may be used to tune, retry, select, calibrate, repair, or
replace any frozen input.

## Pre-goal preservation boundary

Before any goal source edit, the dirty worktree at Git HEAD
`617d119172a6f49caf31a678e0fa7d05d5a3f4e9` was preserved locally as three
recovery artifacts:

- tracked binary patch SHA-256:
  `dbc5a1e0510c8955d1ec72b62db4a048844b3a8704cf95a0ae7271d98be8e09e`;
- non-ignored untracked archive SHA-256:
  `43494a5797155c7186b0d947f4e1df623988b358dff0ed9f25ec77401b728d6d`;
- porcelain-v2 status receipt SHA-256:
  `a07cab06262be8a506ae99f1fb2f07882d3a82c1e9f522bffc22f0393352c427`.

These artifacts are recovery evidence only.  They do not approve the partial
Camera V13 bytes and do not include ignored `.generated` payloads.  Reviewed
source must later be committed or placed in a durable content-addressed archive
before exact execution.

## Critical dependency order

1. Complete and independently review Camera V13 source.
2. Run the sole Camera V13 N5 attempt.  A failure stops this branch; there is no
   retry.
3. On PASS, preregister, review, and pass the fresh-init two-seed scaling ladder
   through N320.
4. Independently review Full Training V3, bind only real ladder artifacts, pass
   payload-free preflight, and run exact matched JEPA/no-JEPA training once.
5. Independently verify training and qualify the selected pre-G2 checkpoint
   through the staged G2 runner, finalizer, and candidate publisher.
6. Bind the post-G2 checkpoint to a reviewed real runner that performs exactly
   one shared visual inference per observation tick and supplies learned
   physical and per-target evidence to the revisioned two-resolution stack.
7. Pass synthetic, exact-map/oracle, fast-development, full 24-scene
   development, target-conversion, full-navigation, and robustness gates.
8. Freeze every source, model, calibration, threshold, seed, runner parameter,
   evaluator, reset rule, result schema, and execution environment.
9. Under independent custody, create a fresh opaque held-out namespace that is
   disjoint from every previously visible scene.  Model-facing code may receive
   only its commitment, population counts, and one-shot interface before launch.
10. Execute the complete frozen held-out batch once and publish the result
    exactly, whether it passes or fails.

Source-only work and synthetic tests may run in parallel.  Every `.generated`
mutator, GPU workload, exact experiment, development rollout, and held-out
operation is serialized.

## Development freeze gates

The final readiness receipt must bind evidence that:

- the raw supervision receipt, Camera ladder, Full Training V3 record, G2
  candidate, runner, and every calibration/source identity are exact and
  independently reviewed;
- inference count equals observation-tick count, memory revisions and reset
  identities are valid, forbidden opens are zero, and evaluator feedback is
  absent from controller inputs;
- per-color and aggregate target-observation precision are at least `0.99` and
  recall at least `0.95`;
- confirmed-target-to-valid-claim conversion is at least `0.90`, with false
  physical accepts below `0.01`;
- all 96 development beacons obtain a valid visibility opportunity;
- all 24 development scenes finish four of four physically verified claims,
  with zero falls, zero accepted false claims, and no preregistered collision
  regression; and
- deployment-noise and locomotion robustness satisfy the frozen development
  contract before held-out materialization.

## Held-out protocol decisions required before materialization

The held-out preregistration must freeze the scene count and strata, generation
and execution seeds, tick budget, comparator arms, primary scene-level
aggregation, confidence method, missing/crashed-run treatment, abort semantics,
and no-retry rule.  Repeated seeds are nested within scenes and are not counted
as independent scene samples.

The old sealed manifests, previously opened result files, `phase4_full18`, and
all other development-used scenes are permanently ineligible.  Ignore files are
not an access-control or opacity boundary.
