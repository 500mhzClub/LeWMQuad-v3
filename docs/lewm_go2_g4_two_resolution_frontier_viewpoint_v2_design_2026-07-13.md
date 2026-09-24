# Go2 G4 V2 two-resolution frontier/viewpoint design

Date: 2026-07-13

Status: **preregistered additive implementation contract; no G4 result or promotion**

## Boundary

This unit is the downstream consumer of the frozen G3 V2 two-resolution API.
It does not alter G3, the legacy same-grid G4 module, learned evidence, G5,
V5, data, checkpoints, or any result. It may consume only a current exact-live
`TwoResolutionConfigurationSnapshotV2`, its issuing projection/planner, and the
bound `RevisionedPhysicalMemory`.

The legacy G4 behavior remains the semantic reference: deterministic safe
viewpoint/yaw candidates, 16 world-frame headings, conservative camera-ground
supercover rays, first-UNKNOWN stopping, separate coverage/entropy/discovery
gain, route and turn costs, uncertainty, view diversity, staleness, and exact
revalidation before execution.

## Frozen lattice ownership

- Routes, connected components, frontier cells, viewpoint cells, and view
  history use the `0.10 m` configuration lattice.
- Physical occupancy, ray supercovers, visible cells, swept coverage, entropy,
  and discovery opportunity use the `0.05 m` physical lattice.
- Both frames must have the exact shared world cell-boundary origin carried by
  the G3 V2 snapshot, distinct frame identities, and shapes in the exact 2:1
  ratio on both axes.
- A configuration cell centre is converted to world coordinates through the
  configuration frame, then to continuous physical-grid coordinates through
  the physical frame. Direct same-index reinterpretation is forbidden.
- Configuration path cost is `path.cost * 0.10 m`. Physical ray traversal uses
  closed `0.05 m` cells.

Every state, candidate, and candidate set binds the physical and configuration
frame identities, physical and configuration revisions, G3 projection source,
memory configuration, profile, FREE support, OCCUPIED support, physical
content, both shapes, and current view-memory identity.

## Authority and invalidation

Physical view states, candidates, candidate sets, and their retained G3 paths
are live execution artifacts. Consumers accept only the exact objects issued
by the current issuer/planner instance. Copy, deep-copy, reconstruction,
serialization replay, foreign-frame use, and use after physical evidence,
view-history, reset, or configuration reprojection are rejected.

The development visual-history method accepts physical observed-cell receipts
only when the underlying memory is not promoted. It records swept cells in the
physical lattice and `(configuration cell, yaw)` history in the configuration
lattice. Promoted use remains fail-closed until a separately reviewed qualified
camera-view receipt exists.

## Candidate and observation rules

Candidate goals come from the complete current confirmed-FREE component and
its deterministic G3 V2 frontier artifact. Every route is a current exact-live
G3 V2 A* path through configuration FREE. Candidate ordering and tie-breaking
are deterministic. The camera origin is derived in world metres from the
configuration viewpoint and then expressed on the physical grid. Missing
physical-domain cells and OCCUPIED cells stop a ray before admission; the
first UNKNOWN group is counted and stops the ray.

No physical visibility cell may become a route cell, and no configuration
route index may be counted as physical sweep/entropy evidence.

## Required synthetic tests

Focused tests must include:

1. high nonzero physical/configuration indices under a translated shared
   origin, proving the configuration centre maps through world coordinates;
2. an explicit `0.10 m` route step and distinct `0.05 m` physical-cell step;
3. deterministic frontier/candidate generation, physical observation, scoring,
   selection, and executable-path validation;
4. physical sweep plus configuration view-history update without index aliasing;
5. rejection of wrong frame, changed origin, wrong support/profile, wrong
   physical or configuration revision, copied/deep-copied/reconstructed state,
   candidate, candidate set, and stale artifacts after view or projection
   revision;
6. unchanged legacy G4 tests and unchanged frozen G3 V2 source identities.

No authoritative G3 audit, scene benchmark, held-out role, model, dataset,
checkpoint, RGB input, GPU path, G5, or V5 operation is authorized by this
document.
