# Perfect camera-ray mismatch decomposition result

Date: 2026-07-12  
Status: complete fit-only causal decomposition; no model or generalization
license

## Purpose

The V2 perfect-ray audit found 100,730 cells where the current physical-v3
target says UNKNOWN but prescribed camera rays say FREE. This audit explains
those cells using only the already authorized 320-frame train fit scope and
source geometry.

The two sequential masks are disjoint:

```text
physical_prior_delta = authoritative UNKNOWN
                     & collision-vetoed ray-only FREE

collision_veto_delta = observable ray-only FREE
                     & collision-vetoed ray-only UNKNOWN
```

They reproduce the immutable V2 totals exactly:

- physical-prior delta: 98,473 cells;
- collision-veto delta: 2,257 cells;
- union: 100,730 cells.

## Immutable evidence

Result:

- path:
  `.generated/go2_perfect_camera_ray_field_mismatch_decomposition/v1/result.json`
- file SHA-256:
  `c9bddb0370dd3cb3dc16c6bde295d9d3374d09674cf55824b2a63874702fc274`
- canonical content SHA-256:
  `eabdd7272a9f6a8b566bd26eab404d1ee0a1892d1242249c4f402ab857289021`

Implementation:

- core SHA-256:
  `da2f22ff8f08ac0506236a34c8711651ccbb5678d5cba0e0ddf665ef57464b2c`
- runner SHA-256:
  `4463da8fea7a516a6efdb7b9f988a7fdaf7e8f632d7439164f132d5fa83c9a25`

Immutable V2 input:

- V2 result file SHA-256:
  `388313d4d01ee7f30107b537504638af20cb580e949645499be0d7a6b292f244`
- V2 result content SHA-256:
  `a1e597dbb57517939800aca6b753e23fd3d89582f378409c817f57c97e1e67a3`

V1 and V2 result hashes were unchanged after this run.

## Physical-prior result

For each of the 98,473 cells, the audit reconstructs the exact rotated output-
square support used by V1. A causal source cell must have all five ground rays
visible while the zero-inflation physical grid withholds FREE.

The exclusive precedence is:

1. source index outside the finite physical grid or its closed 0.05 m square
   crosses declared world bounds;
2. remaining source footprint from a ramp, step, slick patch, or registered
   terrain/surface kind;
3. remaining source footprint from a box with exact rendered/collision parity;
4. residual.

| Cause | Cells | Share of physical-prior delta |
| --- | ---: | ---: |
| Outside domain or world boundary | **98,472** | 99.999% |
| Nonfree terrain or surface | **1** | 0.001% |
| Matched collision/rendered source footprint | 0 | 0% |
| Residual | 0 | 0% |

Raw overlapping evidence confirms the exclusive result:

- outside finite grid: 98,472;
- source square crossing world bounds: 98,472;
- terrain/surface footprint: 1;
- matched rendered footprint: 1, the same terrain cell;
- wall footprint, collision-only geometry, or unattributed nonfree support: 0.

As required by the sequential definition, output-level collision overlap
inside this 98,473-cell set is exactly zero. The source-footprint attribution
must not be confused with the later output collision veto.

The single terrain cell is in rough-local-dynamics scene
`rough_local_dynamics_e112bbdc73e4`, next endpoint of fit row 4241, output
cell `(28, 27)`.

## Collision-veto result

All fit scenes have exact rendered/collision box parity. The audit recasts the
same full-RPY rendered boxes at every native 224x168 pixel centre, preserving
nearest-first-hit occlusion and the camera near plane.

The native stride-1 and registered stride-2 lattices have different pixel
centres; stride-2 is not a subset of stride-1. The positive category therefore
means that the native lattice supplies an in-cell first-surface witness that
the registered stride-2 raster did not.

| Cause | Cells | Share of collision-veto delta |
| --- | ---: | ---: |
| Recovered by native stride-1, absent from registered stride-2 | **176** | 7.80% |
| No native pixel first-surface witness | **2,081** | 92.20% |

The second category is intentionally qualified. It can contain an
outside-frustum or occluded surface, near-plane rejection, subpixel surface,
or conservative 2-D collision-overlap overreach. A finite native pixel lattice
cannot distinguish those causes, so this audit does not call all 2,081 cells
genuinely invisible or occluded.

## Family localization

| Family | Boundary | Terrain | Other/residual physical | Native recovered | No native witness |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open obstacle field | 48,635 | 0 | 0 | 0 | 450 |
| Rough local dynamics | 49,837 | 1 | 0 | 176 | 1,631 |
| Small enclosed maze | 0 | 0 | 0 | 0 | 0 |
| Medium enclosed maze | 0 | 0 | 0 | 0 | 0 |
| Large enclosed maze | 0 | 0 | 0 | 0 | 0 |

The earlier family split is therefore explained almost completely by the
finite world-domain prior. The geometry is consistent with enclosed walls
occluding that outside domain while open and rough cameras see rendered ground
beyond the finite label grid.

## Execution and access

The run used system Python 3.12, NumPy 1.26.4, CPUs 0-5, one numeric thread,
20 primitive scene jobs, and six observed fork workers. Workers received only
in-memory primitive arrays and box records; they performed no source I/O.

The parent fit reader opened the same 20 authorized train label shards and
selected exactly 320 label rows, 320 supervision rows, and 320 source frames.
It scanned the bound 960,000 source JSONL records.

All of these counters are zero:

- RGB opens and decodes;
- model/checkpoint output opens;
- checkpoint-selection and calibration opens;
- physical non-train opens;
- G2, held-out, runtime, seed-20260711, and sealed opens;
- denied and unexpected paths;
- derivative shard or cache writes.

All model, G2, holdout, runtime, and promotion licenses remain false.

## Consequence

The 98,473-cell UNKNOWN/FREE conflict is not a general visual-perception
failure. It is almost entirely a finite benchmark-domain label leaking into a
per-frame target described as camera evidence. That domain boundary belongs
in map ownership and planning state, not in the visual evidence head.

The next target should therefore use a canonical body-local camera-evidence
lattice without the source world's finite physical-free mask. Native pixel
first hits can recover 176 collision-veto cells. The remaining 2,081 cells
must not be silently converted into privileged visual UNKNOWN labels: either
define a stronger source-observable continuous surface-support rule, or leave
their collision/traversability consequence to persistent memory and the
separate learned body-risk/configuration layer.

Before learned output:

1. mechanically generate the new camera-only evidence target from admitted
   ray/calibration inputs;
2. prove its builder contains no physical-grid, world-boundary, collision-veto,
   or configuration-morphology fields;
3. audit its perfect-evidence raster on the same fit panel;
4. then rerun the unchanged N32 visual fit ladder;
5. only after N32 passes, proceed to shared JEPA, persistent memory, and the
   frontier-viewpoint exploration head.
