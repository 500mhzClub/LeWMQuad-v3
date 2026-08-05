# Go2 N32 camera-pose projection audit binding

Date: 2026-07-11

Status: frozen before the fit-metadata audit result.

## Purpose

The categorical-radial head uses one registered level-camera projective grid:
camera xyz `(0.326,0,0.043)` in the base frame and camera RPY `(0,0,0)`. It
accepts RGB only. Observable-physical-v3 labels instead use each rendered
frame's recorded camera position, look direction, and up vector while the
output grid remains aligned by base x/y/yaw.

This audit measures that geometric mismatch before choosing between an
explicit hierarchical output and a dynamic-pose projective head. It is not a
model evaluation and cannot pass N32, G2, or any runtime gate.

## Frozen inputs and access

Panel:

- path: `.generated/go2_physical_micro_overfit/patch7_v1/panel.json`;
- file SHA-256:
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c`;
- content SHA-256:
  `f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f`;
- fit rows SHA-256:
  `5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d`;
- fit scope: 160 train-role transitions, both current and next sides, exactly
  320 frame records across the five registered families.

For each selected frame, the audit may read only:

- the V04 scene `summary.json` adjacent to the committed image path;
- the source train-role `frames.jsonl` named and hashed by that summary;
- JSON pose fields: base position, base yaw, and camera position/lookat/up.

It must verify every source-frame file against the summary commitment before
parsing, verify every requested `(frame_index,env_index)` exactly once, and
rehash inputs after parsing. It must not open or hash RGB bytes, label shards,
model checkpoints/outputs, holdouts, G2, non-train roles, or any sealed
manifest/payload.

## Geometry

Use the exact registered radial centers, angular centers, five vertical anchor
heights, FOVs, near plane, and 16 x 16 token grid from the width-24 model.

For each frame:

1. Define the output coordinate frame from base world position and base yaw:
   x is forward, y is left, and z is world up relative to base position.
2. Transform recorded camera position/lookat/up into that yaw-aligned frame and
   orthonormalize forward, left, and up.
3. Project the same `5 x 64 x 256` registered 3D anchor queries through the
   actual camera basis.
4. Compare normalized image coordinates and validity with the immutable fixed
   projective grid.

For queries valid under both poses, token displacement is Euclidean normalized
image displacement multiplied by 8, because a 16-token side spans normalized
width 2. Validity flips are XOR of fixed and actual query validity.

## Required report

Record per frame and aggregate/per-family summaries for:

- camera origin delta from `(0.326,0,0.043)` in yaw-aligned base coordinates;
- camera forward pitch and camera up roll relative to the registered level
  basis;
- valid-in-both query count;
- token displacement p50, p95, maximum, and fraction `>= 0.5` token;
- fixed/actual validity counts and validity-flip count/rate.

All quantiles use deterministic NumPy linear interpolation and float64
accumulators. Family order is the registered N32 order. The result must include
input/source hashes, exact record keys, duplicate/missing counts, an access
ledger, source hashes, and a canonical content hash.

## Ordering decision

The audit establishes a material dynamic-pose mismatch if both hold:

- the rough-local-dynamics median of per-frame p50 token displacement is
  `>= 0.5`;
- that rough-family median exceeds the pooled non-rough median by `>= 0.25`
  token.

If material, the next trained A/B changes only fixed versus recorded-pose
projective sampling; the hierarchical-output intervention waits. If not
material, proceed first with the smaller explicit hierarchical-output
intervention.

Regardless of this decision, a later camera-centered polar factorization
remains distinct: the camera's 0.326 m forward offset means body-polar columns
are not true camera rays even under a level pose.

## Output

The immutable output path is
`.generated/go2_n32_pose_projection_audit/v1/result.json`. The runner must use
exclusive atomic creation. Synthetic tests must prove exact zero displacement
for the registered level pose, nonzero displacement/validity changes under
tilt and translation, strict provenance, and zero forbidden access before the
audit is authorized.
