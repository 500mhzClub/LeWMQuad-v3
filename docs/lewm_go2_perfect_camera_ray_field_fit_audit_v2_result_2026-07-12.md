# Perfect camera-ray field fit audit V2 result

Date: 2026-07-12  
Status: complete, fit-only negative for the prescribed observable-ray field;
no model or generalization license

## Question

Can the current 64x64 physical-v3 target be reconstructed from prescribed
perfect camera rays without either privileged source-geometry mechanism used
by the target?

V1 answered only part of this question. Its `ray_only` arm removed the
zero-inflation physical-free prior but still applied the exact collision-
geometry FREE-to-UNKNOWN veto. V2 preserves the immutable V1 result and adds:

```text
observable_ray_only = perfect prescribed camera rays
                    + frozen output lattice/registration
                    - physical-free prior
                    - collision-geometry veto
```

The prescribed ray field still contains more than an ordinary depth image: it
uses five exact ground-support queries for every 0.05 m source cell plus the
registered stride-2 camera-pixel first-hit lattice.

## Immutable evidence

V2 result:

- path:
  `.generated/go2_perfect_camera_ray_field_fit_audit/v2/result.json`
- file SHA-256:
  `388313d4d01ee7f30107b537504638af20cb580e949645499be0d7a6b292f244`
- canonical content SHA-256:
  `a1e597dbb57517939800aca6b753e23fd3d89582f378409c817f57c97e1e67a3`

Preserved V1 result:

- file SHA-256:
  `bfb159a168cf4284d99934e40c00fdf3aab2a705e545e00159622f22aac616ba`
- canonical content SHA-256:
  `d32cd3ae37b6171dff623cf4a15759264cba288b705064e1ee095c110b6cf174`

V2 implementation:

- core SHA-256:
  `f0fa4189017e41d891b1d597b6cb7c118e1122e61111a4c536977f274f7a6d3b`
- runner SHA-256:
  `df5c330e93e092bf559bc4ab61930068e07e4956b6636b76b23cddc088eb6c7e`

The runner independently reverified the V1 file and content hashes before and
after execution. The V2 contract-assisted and collision-vetoed arms reproduce
the V1 hashes and counts exactly.

## Result

The audit compared every cell in the exact 320-frame train fit panel:
1,310,720 supervised cells total.

| Arm | Privileged physical-free prior | Privileged collision veto | Mismatch frames | Mismatch cells |
| --- | --- | --- | ---: | ---: |
| Contract-assisted | yes | yes | 0/320 | 0 |
| V1 collision-vetoed ray-only | no | yes | 111/320 | 98,473 |
| V2 observable-ray-only | no | no | 128/320 | **100,730** |

The V2 observable arm gets every authoritative FREE and OCCUPIED cell right:

- FREE: 118,793 / 118,793
- OCCUPIED: 10,228 / 10,228

All 100,730 errors have the same direction:

```text
authoritative UNKNOWN -> observable-ray prediction FREE
```

There are no FREE-to-UNKNOWN, FREE-to-OCCUPIED, OCCUPIED-to-FREE, or
OCCUPIED-to-UNKNOWN errors.

The affected cells are 7.69% of the full raster and 8.52% of authoritative
UNKNOWN. The privileged collision veto accounts for 2,257 cells, or 2.24% of
the final observable-arm discrepancy. The physical-free prior accounts for
the other 98,473 cells.

## Family localization

| Family | Frames | Mismatch frames | UNKNOWN -> FREE |
| --- | ---: | ---: | ---: |
| open obstacle field | 64 | 64 | 49,085 |
| rough local dynamics | 64 | 64 | 51,645 |
| small enclosed maze | 64 | 0 | 0 |
| medium enclosed maze | 64 | 0 | 0 |
| large enclosed maze | 64 | 0 | 0 |

The mismatch is therefore systematic, not a few edge cases: every open and
rough frame is affected, while all three enclosed-maze families are exact.

## Interpretation

The mechanical decoder is correct. Exact contract-assisted parity proves the
independent ray casting, rotated-square aggregation, first-hit witnesses, and
veto implementation reproduce the current label construction.

The prescribed observable-ray field does **not** determine the target's
UNKNOWN/FREE distinction. Clear ground support is sufficient to recover every
known FREE cell, but the target withholds FREE for 100,730 cells based on
facts absent from this ray representation. Most of that dependence is the
privileged physical-free grid; a smaller part is the privileged collision
overlap veto.

This result is narrower than saying RGB can never predict the target. The
tested field retains exact audit-time pose/output-lattice registration and its
stride-2 obstacle witnesses may omit continuous image-surface information.
It nevertheless falsifies the proposed claim that the prescribed perfect ray
field alone is a sufficient statistic for the existing target. An ordinary
pixel depth image is not licensed either.

## Access and runtime

The exact run used `/usr/bin/python3.12`, NumPy 1.26.4, CPUs 0-7, and one
numeric-library thread. It opened the same 20 authorized train-fit label
shards and selected exactly 320 label/supervision rows. It scanned the bound
train source geometry and selected exactly 320 source frames.

All of the following counters are zero:

- RGB opens and decodes;
- model/checkpoint output opens;
- checkpoint-selection or calibration opens;
- physical non-train opens;
- G2 and held-out opens;
- runtime, seed-20260711, and sealed opens;
- denied and unexpected path attempts;
- derivative shard or cache writes.

All model, G2, holdout, runtime, and promotion licenses remain false.

## Required next step

Do not train another dense map head against the unchanged target. It would be
penalized for predicting visibly clear ground as FREE wherever the privileged
physical grid says UNKNOWN, reproducing the exact ambiguity now measured.

Use the V2 fit-only mismatch masks to perform a source-grounded decomposition
before changing labels:

1. Partition the 98,473 physical-prior cells into outside-domain/boundary,
   nonfree surface, and other physical-grid causes.
2. Partition the 2,257 collision-veto cells into visible continuous surface
   support missed by the stride-2 witness lattice versus genuinely absent or
   occluded camera evidence.
3. Define a current-frame physical-evidence target using only facts recoverable
   from calibrated RGB/first-surface evidence. Unknown scene extent and body
   configuration constraints belong in persistent memory and post-fusion
   configuration morphology, not in the per-frame visual head.
4. Mechanically prove the amended target is exactly determined by its frozen
   camera representation on the same fit panel before any learned output.
5. Only then rerun the unchanged N32 visual fit ladder and proceed toward the
   shared JEPA, online memory, and frontier-viewpoint exploration head.
