# Go2 G3 V2 two-resolution design contract

Date: 2026-07-13

Status: **preregistered design only; no implementation or gate result**

## Purpose and boundary

G3 V1 is immutable and remains a failed `90/96` result. This document freezes
the only selected successor: exact physical evidence on a `0.05 m` lattice is
projected conservatively to the existing `0.10 m` configuration/planning
lattice. It does not authorize a G3 V2 result, learned projection, navigation
claim, or held-out access.

This amendment changes representation resolution only. The body remains the
closed yaw-invariant disc of radius `0.47 m`; planning remains four-connected
with diagonal corner cutting disabled; OCCUPIED precedes FREE; unresolved
support remains UNKNOWN; exact line of sight and the claim radius are
unchanged. No endpoint, safety, or legacy condition is relaxed.

## Frozen lattices and alignment

For the V1 configuration origin `o=(ox,oy)` and shape `(nx,ny)`:

- configuration cell size is exactly `c=0.10 m`;
- physical cell size is exactly `p=0.05 m`;
- the two grids share the same continuous cell-boundary origin `o`;
- the physical shape is exactly `(2*nx,2*ny)`;
- configuration cell `(cx,cy)` has centre
  `o + ((cx+1/2)c,(cy+1/2)c)`;
- physical cell `(px,py)` has centre
  `o + ((px+1/2)p,(py+1/2)p)`;
- support offset `(dx,dy)` addresses physical cell
  `(2*cx+dx,2*cy+dy)`.

The configuration centre therefore lies at the common corner of physical
cells `(2*cx,2*cy)`, `(2*cx+1,2*cy)`, `(2*cx,2*cy+1)`, and
`(2*cx+1,2*cy+1)`. An implementation that instead aligns cell centres, rounds
world coordinates, changes either origin, or derives support after clipping is
not this contract.

Out-of-domain physical support is treated as OCCUPIED for configuration
projection. Physical labels retain the exact three classes UNKNOWN, FREE, and
OCCUPIED. Every physical FREE label describes its complete closed `0.05 m`
square, not its centre alone.

## Frozen cross-grid supports

Let `r=0.47`, `h=p/2=0.025`, and for integer offset `k` define

`delta(k) = (k+1/2)p - c/2 = (k-1/2)0.05`.

The FREE support is the lexicographically sorted finite set of all integer
offset pairs `(dx,dy)` satisfying

```text
near_x = max(abs(delta(dx)) - h, 0)
near_y = max(abs(delta(dy)) - h, 0)
near_x**2 + near_y**2 <= r**2 + 1e-12
```

This is exactly the set of closed physical squares intersecting the closed
body disc. The OCCUPIED support is the lexicographically sorted finite set
satisfying

```text
delta(dx)**2 + delta(dy)**2 <= r**2 + 1e-12
```

This is exactly the set of physical cell centres inside the closed body disc.
The resulting frozen counts are:

| Support | Count | Per-axis offset extent |
|---|---:|---:|
| FREE closed-square intersection | 316 | `-9..10` |
| OCCUPIED centre-inside-disc | 276 | `-8..9` |

The hashes below are SHA-256 over ASCII canonical JSON with sorted object keys,
compact separators, JSON booleans, finite numbers, and the complete
lexicographically sorted offset list. Both kernel records contain the common
fields

```json
{"configuration_cell_size_m":0.1,"footprint_radius_m":0.47,"inclusive_boundary":true,"physical_cell_size_m":0.05,"physical_index_rule":"(2*cx+dx,2*cy+dy)","shared_origin_cell_boundaries":true}
```

plus `offsets` and their respective schema:

- FREE schema:
  `lewm_g3_v2_cross_grid_free_closed_square_intersection_kernel_v1`;
- FREE support SHA-256:
  `6fa138060d3df820d646e241fbc2786daf69ded355682afc8d566467f07acb4e`;
- OCCUPIED schema:
  `lewm_g3_v2_cross_grid_occupied_center_inside_disc_kernel_v1`;
- OCCUPIED support SHA-256:
  `a18c0872c50ba749ae6737b0eeba428f5421b0d1518bd08251532f24ad6cb42c`.

The complete projection contract core is:

```json
{"configuration_cell_size_m":0.1,"footprint_radius_m":0.47,"free_support_count":316,"free_support_sha256":"6fa138060d3df820d646e241fbc2786daf69ded355682afc8d566467f07acb4e","occupied_precedes_free":true,"occupied_support_count":276,"occupied_support_sha256":"a18c0872c50ba749ae6737b0eeba428f5421b0d1518bd08251532f24ad6cb42c","otherwise":"unknown","out_of_domain_support":"occupied","physical_cell_size_m":0.05,"physical_shape_per_configuration_cell":[2,2],"schema":"lewm_g3_v2_two_resolution_configuration_projection_v1","shared_origin_cell_boundaries":true}
```

Its SHA-256 is
`2b00cbe295ef4d0ef9f66e42b1aa7188751045240cba923392d83fd1bc709314`.

For each configuration cell, the projected label is OCCUPIED if any of its
276 OCCUPIED-support physical cells is OCCUPIED. Otherwise it is FREE only if
all 316 FREE-support physical cells are FREE. It is UNKNOWN otherwise.

An independent preregistration recomputation used exact rational values
`p=1/20`, `c=1/10`, `r=47/100`, and `h=1/40`, rather than the floating-point
derivation used for the diagnostic. It reproduced both ordered lists and all
three hashes above. The nearest excluded/included squared-distance boundaries
have exact slack `1/625` for FREE and `93/20000` for OCCUPIED, so the frozen
`1e-12` tolerance does not select either count. It also established that the
276 OCCUPIED offsets are a strict subset of the 316 FREE offsets, both sets are
invariant under reflection `k -> 1-k` on either axis, and all fixed alignment
vectors listed under required tests hold.

## V1 differential evidence

The V1 result exactly matched its independent 89/69 implementation on all 24
scenes and admitted zero analytically unsafe FREE cells. Its failure was
representation topology, not an audit mismatch:

- the `0.10 m` 89-cell support covers out to `0.55 m` axially and `0.5701 m`
  at its furthest raster corner for a requested `0.47 m` disc;
- `go2_deployment_medium_maze_3e28c26ef602` collapsed from 4,023 exact-safe
  spawn-component cells to 1,283;
- `go2_deployment_medium_maze_5689fb82c098` collapsed from 3,632 exact-safe
  spawn-component cells to 143;
- required connecting strips were UNKNOWN despite exact clearances from
  `0.480 m` to `0.520 m`;
- the widest exact-safe routes to missed endpoints bottlenecked at
  `0.50955 m` and `0.52000 m`, both above the required radius.

The passing physical oracle uses the same `0.10 m` configuration centres and
same yawless `0.47 m` disc. On the two failures, its traversable masks equal the
audit's analytic masks cell-for-cell. Its exact endpoint inventories also
equal the audit inventories: `[306,305,305,304]` and `[321,317,306,317]`.
This excludes cell-centre undercount, yaw modelling, line of sight, and audit
defect as the V1 differential.

| Scene / beacon | Exact valid centres | V1 FREE valid centres | V1 spawn-connected | V1 FREE island size |
|---|---:|---:|---:|---:|
| `3e28...` red | 306 | 283 | 0 | 671 |
| `3e28...` blue | 305 | 281 | 0 | 670 |
| `3e28...` green | 305 | 281 | 281 | 1,283 spawn component |
| `3e28...` yellow | 304 | 279 | 279 | 1,283 spawn component |
| `5689...` red | 321 | 287 | 0 | 464 |
| `5689...` blue | 317 | 289 | 0 | 517 |
| `5689...` green | 306 | 283 | 0 | 585 |
| `5689...` yellow | 317 | 289 | 0 | 444 |

Relaxing either conservative step is not eligible. Full-square physical
labels with only the 69-cell FREE support admitted 275 and 296 unsafe cells in
the two failures. Centre-sampled physical labels with the 89-cell FREE support
admitted 47 and 41 unsafe cells.

## Diagnostic-only V2 evidence

A read-only, six-process CPU counterfactual used the fixed 24 development
manifests, exact full-square `0.05 m` physical labels, the frozen cross-grid
derivation above, the unchanged `0.10 m` configuration lattice, four-connected
components, analytic safety reference, exact line of sight, and unchanged
claim radius. It wrote no candidate and did not invoke the authoritative audit.

| Quantity | Diagnostic result |
|---|---:|
| Development scenes | 24 |
| Claim endpoints retained | 96/96 |
| Scenes retaining all endpoints | 24/24 |
| Unsafe configuration-FREE cells | 0 |
| Conservative false rejects | 13,260 |
| Configuration component cells | 168,043 |
| Analytic component cells | 181,303 |

The two V1 failures retained 4/4 endpoints with configuration components of
3,103 and 3,030 cells. Their per-beacon connected endpoint counts were
`[299,297,297,295]` and `[303,305,299,305]`.

These values are frozen design evidence, not a G3 V2 PASS. They may not be
used as an implementation result or substituted for the required versioned
audit.

## Rejected alternative

A same-grid `0.05 m` physical and configuration design is not unsafe: its prior
development diagnostic also retained 96/96 endpoints with zero unsafe FREE
cells. It is rejected only as unnecessarily expensive. It doubles planner
resolution on both axes, nominally quadruples configuration states per square
metre and A* search state density, changes the already-registered `0.10 m`
route lattice, and provides no task-usability gain over this two-resolution
design. Its 313/277 kernels are not interchangeable with the frozen 316/276
cross-grid kernels.

## Required implementation and tests

Implementation is deliberately deferred until after this preregistration. A
candidate must add distinct physical and configuration frame identities rather
than changing a single shared cell-size constant. It must bind both origins,
shapes, cell sizes, the exact ratio of two, support hashes, physical evidence,
configuration revision, and projection source into every snapshot and
serialized record. V4-derived evidence must enter the `0.05 m` physical grid;
it may not be upsampled from caller-created `0.10 m` labels.

Before any output, focused tests must independently establish:

1. exact 316/276 recomputation, lexicographic ordering, hashes, support-subset
   relation, and reflection symmetry under `k -> 1-k` on each axis;
2. fixed alignment vectors: FREE includes `(10,3)` and excludes `(10,4)`;
   OCCUPIED includes `(9,4)` and excludes `(9,5)`; `(10,0)` is FREE-support
   only; origin translations do not change offsets;
3. complete closed-square physical rasterization at world boundaries and
   translated/rotated box contacts, including inclusive boundary cases;
4. independent array projection equality for FREE, OCCUPIED, UNKNOWN,
   occupied precedence, incomplete support, and out-of-domain support;
5. rejection of centre alignment, rounded world lookup, wrong origin/shape,
   wrong ratio, same-grid 313/277 kernels, mutated supports, and stale frame or
   revision identities;
6. transaction, retraction, reset, serialization, and source-authority tests
   at both resolutions without caller-created learned evidence;
7. exact snapshot labels/components, deterministic A* distance probes,
   analytic safety dominance, and physical claim endpoints on the fixed
   development scenes.

A different agent must review the complete source closure and test closure
before execution. The V2 runner must retain all V1 captured-source, job/result
binding, input-hash, CPU thread-cap, worker-count, independent finalization,
and access-ledger controls. It may open development manifests and exact
geometry only; no RGB, checkpoint, learned output, G2, runtime, held-out,
sealed, or GPU input is eligible for this exact control.

The new output path is frozen as
`.generated/go2_g3_exact_physical_equivalence/v2/candidate.json`. V1 is never
overwritten or deleted. The V2 gate requires all of the following without
threshold amendment:

- 24/24 fixed development scenes;
- exact production/independent 316/276 labels, components, and route probes;
- zero unsafe configuration-FREE cells in every scene;
- 96/96 physically valid claim endpoints retained;
- source and result bindings complete;
- the historical strict-binary equality reported separately, never silently
  amended into the conservative/usability gate.

Only a new immutable result may state PASS or FAIL. Learned G3 remains blocked
on its separate V4/V5/G2 and qualified-projection requirements even if this
exact control later passes.
