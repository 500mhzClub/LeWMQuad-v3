# Go2 G3 revisioned-memory source review

Date: 2026-07-12

Status: **BLOCK; no exact-equivalence or learned G3 output authorized**

## Reviewed source

The reviewed candidate hashes inherited from the prior source boundary were:

- revisioned physical/configuration memory:
  `d0ae2a629cf4c031ca1f3a0c9c4571731c416e13963b9c527547ed33cccd1dc9`;
- zero-inflation exact adapter:
  `c7d9bd1b00ed9c7d7ddf33c99acba2342bed28ad5b911fb3a8600335cbfe2d8f`;
- focused tests:
  `3940a7f26238899d88486985d81b7049c40fd415780f9a265ac8e7cbeafb79a0`.

The focused tests establish deterministic 89/69 morphology, sparse/dense
agreement for already-labelled physical cells, revision-bound snapshots,
frontiers, A*, serialization, duplicate rejection, and exact-conflict
handling. They do not establish authority for the evidence entering the map.

## Blocking reproductions

1. A `promoted_runtime=True` memory accepted a caller-constructed
   `LEARNED_PHYSICAL` transaction with an arbitrary producer, payload, and FREE
   cell. No qualified V5 checkpoint, G2 report, calibrated threshold,
   local-to-map projection, or issued inference receipt was required.
2. The same promoted memory accepted a caller-constructed 10 m by 10 m
   `VerifiedTraversalPolygon` with a hash-shaped outcome and certified 10,000
   physical cells FREE, including a far cell the robot never traversed. No
   executor-issued outcome, swept-footprint receipt, or reset certificate was
   required.
3. `ZeroInflationExactPhysicalAdapterV1` could be called directly against a
   promoted memory using deployment-odometry provenance. It admitted an
   arbitrary exact FREE cell, left `exact_sim_tainted=false`, and therefore
   made a privileged planning shortcut indistinguishable from learned runtime
   state.

These paths can manufacture coverage and routes, so a later G3 score would not
prove RGB -> physical evidence -> memory generalization.

## Required successor

- A learned-evidence adapter must be the only promoted admission path. It must
  consume an instance-issued inference receipt bound to the qualified shared
  JEPA/physical head, passed G2 report, output semantics, calibration and
  thresholds, exact observation payload, camera transform, pose covariance,
  map frame, and current memory revision.
- The adapter must implement conservative body-frame-to-map projection. FREE
  requires complete destination-square coverage for every admitted pose
  transform; OCCUPIED uses the registered uncertainty supercover. It must
  enforce view-diversity and contradiction-recovery rules selected only from
  the permitted train/calibration roles.
- Traversal, stance/reset, and execution-block transactions require distinct
  instance-issued executor/reset capabilities. A polygon or outcome hash alone
  is not authority.
- The exact adapter must reject promoted memories unconditionally and mark
  every exact-backed snapshot/result as development-only privileged evidence.
- The cold-start reset-certificate versus bootstrap-scan contract and the
  separate candidate/promoted checkpoint lifecycle remain required.

After remediation, a different reviewer must rerun these three exploits plus
serialization, alias, atomicity, pose-diversity, uncertainty-envelope, and
source-checkpoint substitution tests before the 24-scene exact-equivalence
runner is authorized.

No dataset, RGB, model, GPU, G2, held-out, sealed, or scene result was opened or
created for this source review.

## 2026-07-13 Immediate Fail-Closed Remediation

The three reproduced bypasses are now structurally closed while the real
adapters remain unimplemented:

- `promoted_runtime` rejects every direct learned label/UNKNOWN update until a
  qualified learned-projection adapter exists;
- it rejects direct traversal polygons and execution blocks until an issued
  executor-outcome adapter exists;
- the exact adapter rejects promoted memory at construction and the memory
  repeats that check at transaction admission;
- every development exact transaction now taints its memory and configuration
  snapshot as privileged regardless of which pose-source enum the caller used.

Candidate source/test SHA-256 values are:

- memory: `52c7b4491449d263ba99fa42e6fd67cf3de4b51253d72b2d239c2c9e11174d4a`;
- exact adapter: `2dc1629750a6487740187a1464c3d65f42d9fa78e491e8470a0f0cbfbf5cacad`;
- G3 focused tests:
  `ad64797dee33151b64400b98e39df5187c71542e66606a52859626768937dcfe`.

The G3, G4, and G5 dependent focused set passed 85 tests in 28.09 seconds with
native numerical threads capped to one. This is a fail-closed intermediate
state, not a G3 PASS: promoted runtime intentionally cannot ingest learned or
execution evidence yet. The qualified projection adapter, executor/reset
issuer, independent review, exact-equivalence runner, cold start, checkpoint
lifecycle, and closed-loop G3 gate remain required.

## 2026-07-13 Independent Fail-Closed Review

A different reviewer returned **PASS** for this intermediate authority
boundary. Custom atomicity probes confirmed that promoted runtime rejects
exact evidence, direct learned labels/UNKNOWN, caller traversal, and caller
execution blocks with revision zero; a development adapter-issued exact
transaction transferred to promoted memory was also rejected. Exact privilege
taint persists through snapshots and serialization.

The permanent promoted-runtime regression now explicitly includes both a
caller traversal and a constructed `ExecutionBlock`. Current source/test
SHA-256 values are:

- memory: `52c7b4491449d263ba99fa42e6fd67cf3de4b51253d72b2d239c2c9e11174d4a`;
- exact adapter: `2dc1629750a6487740187a1464c3d65f42d9fa78e491e8470a0f0cbfbf5cacad`;
- G3 tests: `a60c6f21cb0e40966216428c938e82024eafcaded95a874c53b542befb9065d4`;
- dependent G5 tests:
  `33740d7c19127dee18e33eff480b5b51e22016df887ad14d577ea6bc83e78c90`.

The current G3/G4/G5 dependency set passes 86 tests in 28.22 seconds with
native numerical threads capped at one. This remains a fail-closed source
boundary only, not a G3 benchmark PASS.
