# G3 exact physical equivalence V2 two-resolution amendment

Date: 2026-07-13

Status: **remediated additive candidate; not executed; independent rereview required**

This file records implementation history only. The governing contract remains
`lewm_go2_g3_two_resolution_v2_design_contract_2026-07-13.md`; this amendment
cannot replace or relax its records, hashes, tests, or gate.

## Scope

V1 remains immutable at its existing source and output paths. V2 writes only:

`.generated/go2_g3_exact_physical_equivalence/v2/candidate.json`

The launcher rejects caller-selected output paths, refuses replacement of an
existing V2 result, and never aliases the V1 result.

## Fixed two-resolution profile

The initial same-grid 0.05 m idea was rejected before implementation. It would
have changed planning resolution and would not represent the intended system.
V2 instead freezes:

- physical evidence cells: `0.05 m`, matching observable-camera V4
  `SOURCE_CELL_SIZE_M` and its `128 x 128` source evidence lattice;
- configuration and planning cells: `0.10 m`;
- footprint radius: `0.47 m`;
- planning connectivity: four, with diagonal corner cutting forbidden;
- physical labels: exact closed full-cell squares, never centre samples;
- projection order: OCCUPIED first, then FREE only when every required physical
  support is FREE, otherwise UNKNOWN;
- production promotion: `false`; learned projection is not implemented here.

Physical and configuration grids share the same lower-edge world origin. One
configuration cell spans exactly two physical cells per axis. For configuration
index `c` and cross-grid support offset `o`, the exact integer transform is:

`p = 2*c + o`

The physical-cell centre relative to the configuration-cell centre is
`(o - 0.5) * 0.05 m`. The full physical raster shape is exactly twice the
configuration raster shape on each axis. Out-of-raster physical support is
treated as OCCUPIED.

## Independently derived cross-grid morphology

The FREE support contains every 0.05 m physical closed square intersecting the
closed 0.47 m body disc around a 0.10 m configuration centre. The OCCUPIED
support contains every 0.05 m physical cell whose centre lies in that disc.
Inclusive boundaries use `1e-12 m` numerical tolerance.

- FREE: `316` offsets,
  SHA-256 `6fa138060d3df820d646e241fbc2786daf69ded355682afc8d566467f07acb4e`;
- OCCUPIED: `276` offsets,
  SHA-256 `a18c0872c50ba749ae6737b0eeba428f5421b0d1518bd08251532f24ad6cb42c`;
- canonical projection-contract SHA-256:
  `2b00cbe295ef4d0ef9f66e42b1aa7188751045240cba923392d83fd1bc709314`.

These are the exact canonical JSON records preregistered by the governing
design. V4 source-cell binding, four-connectivity, full-square semantics, and
false promotion status are recorded in a separate profile envelope; they do
not change the three canonical identities above.

## Source-review remediation

The first implementation was blocked by
`lewm_go2_g3_exact_physical_equivalence_v2_source_review_2026-07-13.md`. It used
the correct offset lists but hashed a noncanonical record, omitted a distinct
configuration-frame/revision identity, shared support state with its audit
oracle, omitted the governing design from the source graph, and exposed an
importable launcher substitution surface.

The remediated candidate now:

- freezes the governing `6fa138...`, `a18c08...`, and `2b00cb...` records;
- carries distinct physical and configuration map-frame identities, both
  shapes and cell sizes, physical and configuration revisions, memory and
  snapshot content hashes, support hashes, and projection-source identity;
- binds every planner operation to a live issuing projection and rejects
  forged, unissued, stale, wrong-frame, wrong-origin, or mutated snapshots;
- derives the audit supports independently with exact rational arithmetic;
- asserts profile integrity at projection, snapshot, planner, audit, and
  summary boundaries;
- includes the governing design in the captured source graph;
- exposes probe and audit execution only through the fixed fresh-process CLI;
  importing the launcher exposes no loader, callback, runtime, path, or hash.

These changes remediate implementation defects. They do not amend the
preregistered design or authorize execution.

### Final source-review remediation

The independent rereview then identified four remaining execution-authority
gaps. The final candidate closes them without changing the canonical
`6fa138...`, `a18c08...`, or `2b00cb...` identities:

- publication verifies the fixed canonical path and non-symlink parent chain,
  writes and fsyncs a private file through a held directory descriptor, and
  publishes with an atomic hard-link create; a pre-existing or concurrently
  created candidate wins and is never replaced;
- snapshots, components, frontiers, and paths are accepted only when they are
  the exact live Python objects issued by the current projection/planner;
  copies, deep copies, serialization replay, and value reconstruction retain
  diagnostic content but acquire no execution authority;
- physical-memory execution blocks are projected to exactly one configuration
  centre by `(px//2, py//2)`, precede FREE, are never footprint-dilated again,
  and carry a physical-revision-bound receipt in snapshots and scene records;
- `frontier_cells` now returns a deterministic, revision/frame/support/component
  bound artifact, with a separate validating consumer boundary that rejects
  stale, copied, or reconstructed components and frontier artifacts.

The associated race, parent-path, copy/replay, CONTACT, frontier, stale-state,
and genuine-artifact regressions are part of the focused source closure. This
remains remediation history, not an audit result or execution license.

## Audit gate

The versioned core preserves the V1 checks and adds independent cross-grid
projection equality. A candidate pass still requires exactly:

- `24/24` development scenes with unique frozen identities;
- `96/96` beacon claim endpoints retained;
- zero unsafe configuration-FREE cells against analytic rotated-box geometry;
- exact independent projection and connected-component equality;
- deterministic four-connected A* agreement;
- exact line-of-sight endpoint checks;
- captured source, runner, per-job, per-result, manifest, and geometry receipts.

The V4 binding is source evidence only. This amendment does not license a V4
checkpoint, learned physical projection, G2/G3 execution, runtime use, or
promotion.

## Execution state

Only capped CPU synthetic tests and the captured-runner bootstrap may run while
this candidate is under review. The authoritative 24-scene V2 audit is not
authorized by this amendment and has not been run.
