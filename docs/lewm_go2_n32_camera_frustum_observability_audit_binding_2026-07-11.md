# Go2 N32 camera-frustum geometry and label-observability audit binding

Date: 2026-07-11

Status: amended pre-run binding, superseding the earlier 2026-07-11 freezes
before any parsed fit-panel access, audit output, or successor N32 model
implementation.

## Pre-freeze access incident and accounting boundary

Before this amended freeze, an operator accidentally ran:

```text
rg -n 'render_audit|source_index|frame_selection' \
  .generated/go2_physical_micro_overfit -g '*.json' --files-with-matches \
  2>/dev/null | head -20
```

The command ran shortly before it was recorded at 2026-07-11 14:15:31 BST; no
more precise execution time is claimed. Its glob scope contained exactly two
JSON files. `rg --files-with-matches` opened each and scanned an unknown prefix
or all of its bytes as needed outside the ledger:

- `.generated/go2_physical_micro_overfit/patch7_v1/panel.json`, frozen file
  SHA-256
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c`;
- `.generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json`,
  frozen file SHA-256
  `6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c`.

Standard output named only the existing panel path. This disclosed only that
at least one search token occurs somewhere in the panel bytes. The prior-result
path was not emitted, but because standard error was suppressed no token-
absence or content inference is claimed for that file. The command did not
parse JSON, print a row or value, or access a label shard, source-geometry
payload, RGB/image byte, any other model/checkpoint/output, holdout,
selection/calibration, physical-nontrain, G2, runtime, or sealed byte.

The seed-result scan was an out-of-ledger forbidden prior-model-result byte
access, and the panel scan was an out-of-ledger fit-metadata byte access. The
earlier binding is superseded; this amended protocol accepts the incident only
as fully disclosed historical contamination and does not treat it as evidence,
an input, a gate result, or scientific licensing. Neither the positive panel
match observation nor the prior-result path's non-emission can be used to
select, tune, interpret, or authorize the audit. This binding therefore
withdraws any claim of zero pre-manifest panel or model-result byte access. It
freezes all requirements before the first parsed panel access. The fresh
authoritative runner/finalizer ledger scope begins at the SHA-256 freeze of
this amended document; the incident remains a separate mandatory provenance
record and is never silently represented as a zero authoritative counter.
The immutable detailed incident record is
`docs/lewm_go2_n32_camera_frustum_observability_preflight_access_incident_2026-07-11.md`,
SHA-256
`683fd43e68f9121f3b4937fbebbf01f760d46ccbc90ff4d1d7551b9a251184ca`,
with required status `acknowledged_pre_authoritative_run`. The machine
manifest, result, and strict finalizer must bind that exact path, hash, and
status. The incident record is controlling if any shorter narrative here could
otherwise be read differently.

## In-place amendment: fixed camera-mount composition

This amendment was frozen after synthetic adversarial review and before any
authoritative parsed panel, label-shard, or source-geometry access. It resolves
the earlier ambiguity between the fixed deployment mount and the recorded
world-frame camera pose. The pre-amendment document SHA-256 was
`93f63cdce1d869b49e0aceb2494e75c15e3e137ec3b167c5b48f2e82939efe17`.

For every selected source frame, the recorded camera pose must be the fixed
rigid camera mount composed with the recorded full body attitude. The audit
must retain and validate all of:

- finite `base_quat_world_xyzw` with exactly four values;
- finite base world position and stored base yaw;
- frame `camera_mount_body` and plan `camera.mount_body` records;
- finite recorded camera position, lookat, and up vectors, each with exactly
  three values.

The plan and frame mount records must compare exactly, without numeric
tolerance, to this object:

```json
{"parent_link":"camera_link","rpy_body_rad":[0.0,0.0,0.0],"xyz_body_m":[0.326,0.0,0.043]}
```

Let `q=(qx,qy,qz,qw)` be the recorded XYZW quaternion. Require
`abs(norm(q)-1.0) <= 1e-5`. Construct the standard body-to-world rotation
matrix directly from the recorded float64 components, without renormalizing
`q`. Let `p_base` be the full recorded base position and define:

```text
p_expected       = p_base + R(q) * (0.326, 0.0, 0.043)
forward_expected = R(q) * (1.0, 0.0, 0.0)
up_expected      = R(q) * (0.0, 0.0, 1.0)
lookat_expected  = p_expected + forward_expected
```

The wrapped absolute difference between the stored base yaw and the yaw
derived from `q` must be at most `1e-5 rad`. Using float64 arithmetic and zero
relative tolerance, require inclusive componentwise maximum absolute residual
at most `1e-5 m` for recorded camera position and lookat, at most `1e-5` for
recorded up, and `abs(norm(lookat-position)-1.0) <= 1e-5 m`. The angular error
between normalized recorded and expected forward vectors and between
normalized recorded and expected up vectors must each be at most `1e-5 rad`.
The implementation must clamp only each final dot product to `[-1,1]` before
`acos`; it may not otherwise loosen, scale, or replace these tolerances.

This rule deliberately preserves body roll and pitch. A yaw-only camera pose
for a non-level body fails. Camera safety retraction, manifest extrinsic
jitter, an alternative look distance, a different mount, or any arbitrary
recorded translation/rotation also fails. The geometry contract's runtime
safety-retraction flag does not authorize such a substitution in this
fit-source audit: V04 rerenders the committed recorded pose, while this audit
does not license dynamic camera pose as a learned input. Every one of the 320
selected frames must pass. Any failure fails provenance, sets
`rendered_collision_target_ambiguity=true`, and denies representation
authorization.

## In-place amendment: failed preparation and full-render subset relation

This amendment was frozen after the first metadata-only preparation failed and
before any retry, label-shard byte open, authoritative audit, or result. The
pre-amendment document SHA-256 was
`6b8a243d8ec2d3fa1df386defb761f2defe87d5ed491e371df050d3054e644eb`.

The controlling combined pre-authoritative access record is now
`docs/lewm_go2_n32_camera_frustum_manifest_preparation_failure_2026-07-11.md`,
SHA-256
`5c3fad3b8e296aed239c3573e263af766b52e391fb9fe86e0e31d26c94845db3`,
with required status `acknowledged_pre_authoritative_run`. It incorporates the
earlier search incident by immutable path/hash and records the failed
metadata-preparation command, its missing emitted ledger, the reached code-path
boundary, and the zero label/image/model/G2/held-out/sealed contact claim. This
combined record supersedes the earlier incident document as the direct machine
manifest/result incident input; the earlier document remains transitively
bound and is not erased.

The failed preparation exposed a preregistration error, not a label or model
result. V04 frame selection and rendering commit the larger rendered source
corpus for each scene; the 320 fit-panel endpoints are a selected subset of
that corpus. The frozen provenance relation is therefore:

1. for each selected source scene, frame-selection keys and render-summary
   frame keys are identical complete sets, with exact scene, counts, key-set
   hash, timestamps, image-SHA values, and rendered-set hash reconciliation;
2. the selection artifact binds the global source-row file, while the original
   per-scene render plan and summary bind the larger source frames JSONL and
   exact camera contract; neither larger file is required to have the same key
   set as the selection;
3. every selected render key occurs exactly once in the source JSONL with the
   same timestamp, and every fit-panel endpoint is a unique exact subset member
   matching frame index, environment index, timestamp, and image SHA-256;
4. the 320 fit endpoints are unique globally and no fit endpoint is missing or
   repeated; committed source frames outside the selected-render set are
   strict-parsed and ledgered but never selected, reconstructed, inspected as
   a target, or used in an audit statistic.

Equality between the full rendered set and the 320-frame fit subset is no
longer required and must not be reintroduced. This structural correction does
not change the fit rows, labels, camera geometry, support mapping, audit gates,
or authorization thresholds. The prior human manifest SHA-256
`ef8d1a8a768c430caad82505634ec7e25e703c50c4b4a8d098b7a41267b113e6`
is stale. A fresh reviewed implementation/human manifest, machine companion,
and zero-based preparation ledger are required under this amended binding.

## In-place amendment: v1 finalizer ULP mismatch and immutable v2 retry

This amendment was frozen after the first authoritative fit-only runner result
and its independent finalizer rejection, and before any retry, successor target
implementation, model output, G2, holdout, runtime, or promotion access. The
pre-amendment document SHA-256 was
`96eb4b9eb11b0924056ffb89590ecf13bb20ffcc72c6aca5e6cb51e92bb8132e`.

The immutable v1 runner result and finalizer failure are recorded at
`docs/lewm_go2_n32_camera_frustum_observability_audit_v1_result_2026-07-11.md`,
SHA-256
`9882465826a848be303694efbac1c76468026a8c05ce1152e0089d9b6849a365`.
The v1 runner result remains at
`.generated/go2_n32_camera_frustum_observability_audit/v1/result.json`, file
SHA-256
`6ec4ac68828f6b7b8928489267951f78f30533ca52458ed59526596ce107d223`
and canonical content SHA-256
`41c82454f5328bf19fc82428fbffb5a48bb03a2d11479a015248c51587b4c3c7`.
It is immutable but non-authoritative: it may not be edited, replaced, promoted,
or treated as representation authorization. The v2 implementation must not
open, hash, parse, or otherwise use that generated result. Its dated Markdown
report is historical provenance committed by this binding, not an executable
input or gate value.

The v1 runner completed with clean provenance, exact reconstruction, injective
camera mapping, zero ambiguity, and zero forbidden access. Its representation
decision was false because the proposed camera support omitted 1 supervised
FREE and 372 supervised OCCUPIED cells across the 320 fit frames. The
independent finalizer then rejected frame 13 before completing because its
quaternion norm differed from the runner by one ULP. The runner used the frozen
explicit float64 expression
`sqrt(qx*qx + qy*qy + qz*qz + qw*qw)`; the finalizer used Python 3.12
`sum()`, producing `1.0000000510408722` and
`1.0000000510408724`, respectively. Both pass the unchanged `1e-5`
semantic tolerance, but exact compact-evidence equality correctly failed.

The sole arithmetic correction is that the finalizer must compute quaternion
norm with the same explicit four-term expression. It must retain its existing
independent formulas and `sum()` behavior for three-vector norms whose runner
formulas also use `sum()`. A synthetic regression fixture containing the exact
ULP-sensitive quaternion must prove canonical runner/finalizer evidence
equality. This correction does not alter camera geometry, tolerances, mapping,
label support, reconstruction, coverage gates, authorization logic, or the
scientific interpretation of the v1 negative.

The fresh retry is versioned exclusively at:

- human implementation manifest:
  `docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.md`;
- machine implementation manifest:
  `docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.json`;
- exclusive runner result:
  `.generated/go2_n32_camera_frustum_observability_audit/v2/result.json`.

The v2 preparation, runner, and finalizer ledgers restart from zero under the
new binding and source hashes. Historical v1 fit-label access remains fully
recorded in its immutable result/report and is not silently merged into v2.
The v2 runner and finalizer may reopen only the same exact committed fit-only
inputs under the existing per-phase limits. A fresh reviewed human manifest,
machine companion, metadata-only preparation, runner result, and independent
finalizer pass are required. V2 is expected to confirm the same coverage
negative; no camera-frustum representation or successor model output is
licensed merely by repairing the finalizer arithmetic.

## Purpose and prior evidence

The authoritative N32 V4 explicit-output hierarchy failed the fit gate. The
immutable result and adjudication is
`docs/lewm_go2_categorical_radial_n32_v4_result_2026-07-11.md`, SHA-256
`dd0842d1c59b42a985eaf0843f0d6f6adc41286a2a1a2b4b1f95111a9c0efa50`.
That dated report and its hash are the sole predecessor-adjudication input to
this audit. Any generated V4 result path, payload, file hash, content hash, or
schema is historical provenance quoted by the report, not an audit input. The
runner and finalizer must not stat, resolve through a symlink, open, hash,
parse, or otherwise inspect a generated V4 result artifact.

V4 left the conditional FREE/OCCUPIED decision effectively solved while
failing spatially opposed UNKNOWN/KNOWN family gates. A common KNOWN bias is
analytically impossible; the frozen proof is
`docs/lewm_go2_n32_known_bias_impossibility_2026-07-11.md`, SHA-256
`e214bb80bcccf9ae5051231d90f7a5d8c2bfa33ca799e7db3eb969698fa2108a`.
That proof identifies the body-centered polar decoder as the next
representation-level fault: the camera is 0.326 m forward of the body origin,
so one body-bearing column is not one camera ray.

This audit answers the geometry and observability questions required before a
camera-centered categorical representation can be implemented. It is
fit-only, performs no learning, and cannot pass N32, G2, or any runtime or
promotion gate.

## Frozen fit scope and access boundary

The sole data panel is the N32 fit panel already used by V2, V3, and V4:

- path: `.generated/go2_physical_micro_overfit/patch7_v1/panel.json`;
- file SHA-256:
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c`;
- canonical content SHA-256 after omitting only `content_sha256`:
  `f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f`;
- canonical ordered fit-row SHA-256:
  `5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d`;
- scope: the 160 physical-dataset-role `train` fit transitions, both endpoint
  sides, exactly 320 unique fit frames, 64 frames in each of the five
  registered families;
- governing role: current physical dataset role `train`; any legacy rollout
  split is provenance only and cannot filter, rank, calibrate, or select rows.

The canonical frame order is exactly the tuple `(family_rank, scene_id,
global_row, endpoint_side_rank)`: `family_rank` follows the registered order
`open_obstacle_field`, `rough_local_dynamics`, `small_enclosed_maze`,
`medium_enclosed_maze`, `large_enclosed_maze`; `scene_id` is ascending lexical
Unicode code-point order; `global_row` is ascending integer order; and endpoint
side rank is `current` then `next`. Every ordered identity, label-byte stream,
ray record, violation record, veto identity, and mismatch identity uses this
order before its own trailing row/column/angular/range keys. No filesystem,
JSON-object, shard, or encounter order may affect an artifact.

The audit may open only:

1. the panel JSON, this binding, the incident record, the V4 adjudication
   report, the KNOWN-bias proof, and the reviewed implementation manifest;
2. the 20 fit label shards named and hashed by the selected fit records, each
   at most once in the runner phase and once in the independent finalizer phase
   as an NPZ byte-open, under the storage-boundary exception below, and
   semantically only for the 320 selected endpoint rows;
3. source scene, renderer, collision-geometry, camera, base-pose, and frame
   metadata transitively named by those fit records and their committed source
   summaries;
4. the closed implementation and semantic source graph frozen below and in
   the pre-run implementation manifest.

The implementation source map is closed and must contain exactly these roles,
each with resolved repository-relative path and file SHA-256:

- `binding`: this document;
- `audit_core`: the pure camera-frustum mapping and label-diagnostic module;
- `audit_core_test`: its synthetic unit tests;
- `audit_runner`: the authoritative fit-only runner;
- `audit_runner_test`: its synthetic end-to-end and denial tests;
- `audit_finalizer`: the strict torch-free and NumPy-free finalizer;
- `audit_finalizer_test`: its mutation and rejection tests;
- `label_semantics`: `lewm/datasets/go2_paired_navigation.py`, whose exact
  observable-physical-v3 helpers are reused for reconstruction;
- `geometry_contract_semantics`: `lewm/planning/geometry_contract.py`;
- `scene_manifest_semantics`: `lewm_worlds/lewm_worlds/manifest.py`;
- `planning_grid_semantics`: `lewm_worlds/lewm_worlds/planning_grid.py`.

No other repository Python module may supply an executable definition,
constant, parser, geometry rule, or hashing rule that affects the audit. The
Python standard library and NumPy are runtime dependencies rather than
repository semantic sources; their exact interpreter/package versions must be
recorded in the implementation manifest and result. Discovery of another
transitive repository semantic dependency invalidates the manifest and
requires a dated binding amendment before any authoritative run. Tests may
import test-only helpers, but authoritative execution may not import any
repository module outside this closed source graph.

The runner may fully decompress the four registered arrays in each exact
committed shard (`current_labels`, `current_supervision_mask`, `next_labels`,
and `next_supervision_mask`) because NPZ is the existing storage boundary and no
selected-row derivative is authorized. This exception permits physical array
materialization only. The runner may inspect, validate, hash, copy, aggregate,
or retain only the exact selected rows. It may inspect array-level name,
dtype, rank, and shape to validate the container, but may not inspect a value
from an unselected row. After copying selected rows, it must discard all four
materialized arrays before opening another shard. Creating a derivative shard
or cache is forbidden.

Each production shard archive contains exactly those four registered target
members plus these eight registered auxiliary members:
`current_observed_mask`, `next_observed_mask`,
`relative_se2_current_frame`, `primitive`, `current_image_path`,
`next_image_path`, `current_image_sha256`, and `next_image_sha256`. The runner
and finalizer validate the exact twelve unique archive member names from ZIP
metadata. They must not decompress, decode, inspect, or retain any auxiliary
member. A missing, duplicate, encrypted, path-qualified, or additional archive
member rejects the shard. The 80-array decompression count below therefore
means only the four registered target/supervision arrays across 20 shards.

Every allowed source-geometry file must be allowlisted by resolved path and
expected SHA-256 before the audit runner opens its contents. The runner must
verify the hash before parsing, consume only JSON/numeric geometry fields, and
rehash the file after parsing. Source metadata may contribute scene and frame
identities, base position/yaw/full XYZW quaternion, the exact declared camera
mount, camera position/look/up, ground plane, rendered box geometry, collision
box geometry, occupancy-grid geometry, and local-grid geometry. It may not
contribute simulator depth, a rendered pixel, a target outside the selected
fit label rows, or any model value.

Every JSON and every individual JSONL record in every phase, including panel,
manifest, provenance, source geometry, result, and finalizer evidence, must be
decoded as UTF-8 with a parser that rejects duplicate object keys, NaN,
positive or negative infinity, invalid numeric syntax, trailing non-whitespace
content, and a non-object root where an object schema is required. A permissive
standard `json.loads` call without duplicate-key and nonfinite hooks is not a
valid implementation of this requirement.

The following byte reads and derived model outputs are forbidden and must each
remain exactly zero:

- RGB or other rendered-image decode, pixel read, or image-byte hashing;
- same-scene or cross-scene train-role holdout labels or source geometry;
- checkpoint-selection or probability-calibration payload;
- any physical-nontrain role;
- G2 manifest payload, label, image, checkpoint, or model output;
- development closed-loop or runtime payload;
- sealed manifest, metadata, payload, scene identity, or model output;
- model checkpoint, parameter, activation, logit, probability, or prediction;
- any generated V4 result or other predecessor generated model-result payload;
- seed-20260711 construction or evaluation.

Committed image paths and SHA-256 strings may be copied from panel metadata
only to bind frame identity; their image bytes must not be opened or rehashed.
Any attempted path outside the allowlist fails closed before output.

### Semantic path, role, and modality denial

Authorization is two-stage. Before path resolution, lexical path, requested
semantic role, panel/source-declared dataset role, and modality must be
eligible; an ineligible request is denied without resolving or statting its
target. Only an eligible path may be resolved under the repository root. The
fully resolved path must then equal its exact allowlist entry before any
existence check, content stat, hash, open, import, parse, decode, or write. An
allowlisted hash or innocuous filename cannot override a forbidden semantic
role. A symlink, hard-link alias, renamed copy, archive member, import alias,
or metadata indirection cannot change a forbidden role into an allowed one.
Ambiguous or missing role/provenance is forbidden.

The authoritative runner and strict finalizer must deny all of the following
semantic categories regardless of spelling or location: rendered RGB or any
other image; same-scene or cross-scene holdout; checkpoint/model selection;
probability calibration;
physical nontrain; G2; development/runtime/closed-loop; sealed; checkpoint or
model output; seed 20260711; generated V4 result; and any unregistered dataset
role or unallowlisted source/input. The only allowed generated payloads are
the exact fit panel, its exact 20 fit label shards, its transitively committed
fit source geometry, and, for the finalizer only, the exclusive audit result.
The exact v04 render frame-selection artifact committed by each fit summary is
allowed solely as immutable render-provenance evidence; it cannot filter,
rank, or select audit rows. Any other frame-selection artifact is unallowlisted.

Allowed input modalities are limited to UTF-8 Markdown/Python source,
strict JSON/JSONL numeric metadata, and the exact 20 committed NPZ label
shards. All image modalities, including PNG, JPEG, WebP, BMP, TIFF, EXR, HDR,
GIF, video, raster arrays outside the exact label shards, simulator depth,
point clouds, and renderer buffers are forbidden. All model modalities,
including PT/PTH, checkpoint, safetensors, ONNX, pickle/joblib, parameter
archives, activation/logit/probability arrays, and model logs are forbidden.
An extension is not sufficient authorization: every allowed file must also
have the exact registered semantic role, resolved path, and expected hash.

Every rejected attempt increments `denied_attempts_total` and exactly one
primary-reason counter under this immutable precedence: `sealed`, `g2`,
`seed_20260711`, `generated_v4_result`, `model`, `runtime`,
`physical_nontrain`, `selection_or_calibration`, `holdout`, `image_or_depth`,
`unregistered_role`, `forbidden_modality`, `path_alias_or_escape`,
`unallowlisted`. A separate `denied_modality_attempts` table may additionally
classify modality, but it cannot replace primary reconciliation. Record the
attempted lexical path, resolved path when resolution itself is authorized,
requested role, declared role, modality, and primary reason without reading
the target. The sum of primary counters must equal `denied_attempts_total`.
`unexpected_path_attempts` must equal that same total. Every primary counter,
every modality counter, and both totals must be exactly zero in an
authoritative result and in the finalizer's independent ledger. Deliberate
denial tests run only against synthetic paths and do not share an authoritative
ledger.

## Frozen coordinate systems

The body-local Cartesian target remains unchanged:

- shape `[64,64]`, row then column;
- cell size `0.10 m`;
- forward minimum edge `-1.0 m`;
- left minimum edge `-3.2 m`;
- cell center
  `forward=-1.0+(row+0.5)*0.10`,
  `left=-3.2+(column+0.5)*0.10`.

The fixed deployment camera calibration remains:

- body-frame camera origin `(forward,left,up)=(0.326,0.0,0.043) m`;
- camera RPY `(0,0,0)`;
- horizontal FOV `78.323 degrees` with half angle
  `h=39.1615 degrees`;
- near plane `0.05 m`.

The fixed calibration is a body-relative rigid transform, not a world-level or
yaw-only camera. Recorded source poses are admissible only after the full
quaternion composition check in the in-place amendment passes exactly under
its frozen absolute tolerances.

This audit does not license recorded simulator camera pose as a model input.
The prior pose audit rejected dynamic pose as the ordered next intervention.

## Question 1: old body-column bearing distortion

Quantify how much one old body-centered polar context column changes camera
bearing with range.

Use the old lattice exactly:

- 64 body-radius centers `(i+0.5)*0.10 m`, `i=0..63`;
- 256 body-bearing centers uniformly spaced over `[-h,+h]`, using bin
  centers;
- body point `(f,l)=(r*cos(theta_body),r*sin(theta_body))`;
- camera-relative point `(f_cam,l_cam)=(f-0.326,l)`;
- actual camera bearing `theta_cam=atan2(l_cam,f_cam)`.

A sample participates in the primary span only when `f_cam>=0.05`, its camera
bearing lies in the closed corrected horizontal FOV, and at least one of the
five registered vertical anchors is valid under the existing fixed projective
grid. For every old body-bearing column, report participating range count,
minimum and maximum camera bearing, and
`span=max(theta_cam)-min(theta_cam)` in radians, degrees, and fractions of one
new angular-bin width. Also report aggregate linear-interpolation p50, p95,
maximum, and counts with span at least 1, 2, 4, and 8 new angular bins. Columns
with fewer than two participating samples have null span and are counted
separately. Repeat counts without the vertical-anchor condition as a secondary
horizontal-only diagnostic.

These measurements are diagnostic, not a tunable threshold and not a model
authorization gate.

## Question 2: proposed camera-centered mapping

Freeze one proposed `[64,256]` camera-centered range/bearing lattice before
looking at labels. The formulas below are literal float64 comparisons and bin
assignments, not tolerance targets:

- camera-range bins: 64 left-closed/right-open bins of width `0.10 m` over
  `[0.0,6.4) m`;
- camera-bearing support is the closed interval `[-h,+h]`, partitioned into
  bins 0 through 254 that are left-closed/right-open and bin 255 that is
  closed at `+h`;
- the positive FOV edge belongs to angular bin 255 by the sole explicit
  `min(255,...)` edge rule;
- mapping samples the unchanged Cartesian cell center, not a square corner or
  an inflated footprint.

For Cartesian center `(f,l)`, define:

```text
f_cam = f - 0.326
l_cam = l
range = hypot(f_cam,l_cam)
bearing = atan2(l_cam,f_cam)
```

The cell is supported if and only if `f_cam>=0.05`,
`0.0<=range<6.4`, and `-h<=bearing<=+h`. Its range bin is
`floor(range/0.10)`. Its angular bin is
`min(255,floor((bearing+h)/(2*h)*256))`. Unsupported cells receive the exact
integer pair `[-1,-1]`.

No epsilon, ULP expansion, `nextafter`, approximate comparison, tolerance,
plane-edge widening, or near-plane widening is permitted. `atan2` bearing is
not clamped, snapped, or projected into the FOV. A value outside a literal
support inequality is unsupported even if it is one ULP from the boundary.
The positive-edge `min(255,...)` rule above is the only clamping operation.
Synthetic tests must include exact-boundary values and `nextafter` values one
ULP inside and outside `f_cam=0.05`, `range=0`, `range=6.4`, `bearing=-h`, and
`bearing=+h`, and must reject both plane-edge expansion and bearing clamping.

Build the complete `[64,64,2]` signed-int16 mapping in row-major Cartesian
order. Hash its contiguous little-endian bytes. Separately hash the contiguous
row-major uint8 support mask. Report:

- supported and unsupported Cartesian cell counts;
- unique used polar-bin count and unused polar-bin count;
- every collision multiplicity and Cartesian location;
- partially mapped, out-of-range, or nondeterministic entry counts;
- mapping SHA-256 and support-mask SHA-256.

The proposed mapping passes only if it is deterministic, all mapped indices
are in range, no entry is partially mapped, and every supported Cartesian cell
maps injectively to one unique `(range_bin,bearing_bin)` pair. No bin count,
range, FOV, origin, edge convention, or tie rule may be changed after the audit
to obtain a pass.

## Question 3: fit-label support completeness

For each of the 320 selected fit frames, read only its committed final
UNKNOWN/FREE/OCCUPIED target and supervision mask. Require labels to be exact
integers `0/1/2` in the registered order and the supervision mask to match the
full finite 64 x 64 output grid.

Against the proposed support mask, count per frame, family, and class:

- total supervised labels;
- supported and unsupported labels;
- supported and unsupported FREE labels;
- supported and unsupported OCCUPIED labels;
- supported and unsupported UNKNOWN labels.

The complete-coverage gate passes only when every FREE and OCCUPIED fit label
is inside proposed support and every unsupported target is UNKNOWN. Thus both
`unsupported_FREE_count` and `unsupported_OCCUPIED_count` must equal zero over
all 320 frames and within every family. Store the exact violating frame key and
Cartesian location for any nonzero count; do not change support after seeing a
violation.

## Question 4: scalar-depth incompatibility

Scatter each fit target through the proposed injective mapping. For each frame
and angular bin, form a ray sequence from only the mapped Cartesian cells in
strict increasing range-bin order; unused polar bins are omitted rather than
invented as UNKNOWN. Ties are impossible if injectivity passes. Sequences with
fewer than two mapped cells are counted but excluded from transition rates.

Collapse adjacent equal classes and report:

- sequence count and length histogram;
- class-transition histogram and maximum transitions per sequence;
- all six directed unequal transition counts;
- counts with 0, 1, 2, and at least 3 transitions;
- counts containing known evidence after UNKNOWN;
- counts containing FREE after OCCUPIED;
- counts not matching the scalar first-hit regular form
  `FREE* OCCUPIED* UNKNOWN*`;
- the same quantities by family and by endpoint side.

Hash the canonical row-ordered sequence-summary records and the aggregate
transition tables. Report representative violations only by committed frame
key, angular bin, range-bin sequence, and class sequence; do not open RGB.

This is a measurement, not a gate that can license a depth head. A scalar
first-hit depth remains forbidden under every audit outcome because the target
construction aggregates multiple vertical 3-D rays, visible-floor center and
corner support, sparse obstacle-surface witnesses, partial cell support, and
collision vetoes. Any nonmonotonic or three-plus-transition sequence is
additional empirical evidence of that incompatibility.

## Question 5: collision-veto observability

Before reconstructing a label, validate production render provenance with all
non-image-byte checks of the canonical observable-physical-v3 validator. For
each source scene this requires, without opening or statting an image:

- summary schema `lewm_rendered_vision_v04`, complete status, exact scene ID,
  positive resolution, and native height/width ratio `3/4` within absolute
  tolerance `1e-12`;
- a plan whose camera declares horizontal FOV, and a summary camera declaring
  pinhole projection, renderer FOV axis `vertical`, no runtime rectification,
  plan horizontal FOV and derived vertical FOV within absolute `1e-9`, and
  plan near plane within absolute `1e-12`;
- exact nominal plan/frame camera-mount declarations and, for every selected
  source frame, successful full-quaternion rigid composition of the recorded
  camera position/lookat/up under the fixed-camera amendment above;
- object parity schema `lewm_render_object_parity_v1`, groups in exact order
  `wall, obstacle, landmark, distractor`, collision distractors rendered,
  full box roll/pitch/yaw rendered, unique canonical object IDs, exact object
  count and ordered IDs, and exact canonical object-ID and object-record
  SHA-256 values recomputed from the scene manifest;
- exact resolved path and raw-byte SHA-256 reconciliation for the render plan,
  source frames JSONL, and scene manifest named by the summary;
- a frame-selection artifact with schema
  `lewm_go2_selected_render_frames_v1`, exact scene ID, raw-byte SHA-256,
  ordered unique key list, and exact canonical frame-key-set SHA-256;
- exact equality of the complete frame-selection and summary rendered-frame
  key sets, exact selected-key membership/timestamp reconciliation against the
  larger committed source JSONL, and one-to-one subset membership of all fit
  panel endpoints under the failed-preparation amendment above, including
  exact image-SHA strings and the summary's canonical rendered-frame
  metadata-set hash, while never checking image-file existence or bytes;

Any missing, duplicate, extra, or inconsistent source, plan, selection,
manifest, frame key, timestamp, image-hash string, projection field, camera
mount/composition field, or object parity field fails provenance and sets
target ambiguity. The synthetic end-to-end suite must contain independent
failing mutations for every bullet, not merely a hand-built geometry fixture.
Camera mutations must independently cover translation, rotation, retraction,
mount jitter, look distance, quaternion norm, quaternion/yaw disagreement,
yaw-only substitution under nonzero roll/pitch, and exact/inside/outside
tolerance boundaries. Exact parity with the production v3 validator is
semantic and is why `label_semantics`, manifest semantics, and planning-grid
semantics are part of the frozen source map.

Reconstruct, from allowed fit source geometry only, the observable-physical-v3
label stages used for each selected frame:

1. visible physical FREE support from the center and four corners of each
   0.05 m physical source cell;
2. visible rendered-obstacle first-hit witnesses;
3. the pre-veto 0.10 m UNKNOWN/FREE/OCCUPIED aggregation;
4. exact collision-box overlap used only to downgrade pre-veto FREE to
   UNKNOWN;
5. the final stored target.

The stored target and reconstructed final target must match bit-exactly at
every supervised cell. Report any mismatch before interpreting veto counts.

Define a `collision_veto_only_UNKNOWN` cell as stored UNKNOWN, reconstructed
pre-veto FREE, and overlapping collision geometry. Count these cells per
frame, family, and distance bin, and hash their ordered
`(frame_key,row,column)` identities.

Compare rendered and collision boxes with one-to-one deterministic matching.
Canonical box fields are center xyz, size xyz, and the float64 3 x 3 rotation
matrix. A pair matches only when every corresponding finite value differs by
at most `1e-12`; matching is lexicographic by canonical geometry tuple, with
duplicate multiplicity preserved. Report unmatched rendered boxes, unmatched
collision boxes, and matched multiplicities per source scene.

For each veto-only UNKNOWN cell, separately report whether its overlap can be
attributed to at least one matched collision/rendered box and whether it
depends on any unmatched collision box. Also compare output-cell overlap masks
constructed from all rendered boxes and all collision boxes and report their
XOR counts per frame and family.

Persist the exact ordered identities in a four-way exhaustive, disjoint truth
partition: `(matched=true,unmatched=false)`, `(false,true)`, `(true,true)`, and
`(false,false)`. Hash each partition and the ordered union. Per-frame
`attributed_to_matched_box_count` is the sum of the two `matched=true`
partitions; `depends_on_unmatched_collision_box_count` is the sum of the two
`unmatched=true` partitions. Recompute those counts and identities exactly by
family and aggregate; no cell may be absent from or repeated across the
four-way partition.

Set `rendered_collision_target_ambiguity=true` if any of the following holds:

- a selected frame fails the fixed camera-mount composition contract;
- a final target cannot be reconstructed exactly;
- a collision-veto-only UNKNOWN cell depends on unmatched collision geometry;
- a collision box affecting a selected target cell lacks a rendered match;
- required rendered-versus-collision provenance is missing or nonunique.

Matched rendered geometry may still produce veto-only UNKNOWN and must be
reported, but is not alone classified as missing-observation ambiguity.

## Required counts and hashes

The result must contain aggregate and five-family reports in the immutable
order `open_obstacle_field`, `rough_local_dynamics`, `small_enclosed_maze`,
`medium_enclosed_maze`, `large_enclosed_maze`. Each report includes frame
count, UNKNOWN/FREE/OCCUPIED counts, proposed-support counts by class,
ray-sequence diagnostics, veto-only UNKNOWN counts, geometry-ambiguity counts,
and reconstruction mismatches.

The artifact must also bind:

- all 320 canonical ordered compact frame reports, each containing its full
  identity, family/side, class and support counts, ray-sequence count/hash,
  reconstruction mismatch identities/count, veto identities/count, and
  four-way attribution counts/hashes, plus the recorded base
  position/quaternion/yaw and plan/frame mount, recorded camera pose, all
  fixed-mount composition residuals, and a composition-pass boolean;
- the complete canonical ordered list of all 320 fit-frame identities and its
  SHA-256 over
  `(family,scene_id,global_row,side,image_sha256,label_shard_sha256,label_row)`;
- the ordered allowed label-shard path/hash manifest and its SHA-256 and, for
  each of the exact 20 shards, the canonical selected tuples
  `(family,scene_id,global_row,side,label_row)`, selected-row count, and
  family-by-side counts;
- exact identity reconciliation of 320 unique endpoint frames, 64 per family,
  160 per endpoint side, and 32 per family-by-side cell;
- ordered source-geometry path/hash manifest SHA-256;
- old-column span table SHA-256;
- proposed mapping and support-mask SHA-256 values;
- ordered selected-label bytes SHA-256, with each target encoded contiguous
  row-major uint8 in canonical frame order;
- a required ordered aggregate/five-family UNKNOWN/FREE/OCCUPIED class-count
  table and canonical SHA-256, exactly recomputed from per-frame reports;
- ray-sequence and transition-table SHA-256 values;
- ordered per-frame ray-sequence count/hash records and every
  aggregate/family/side transition table; the finalizer reconstructs the full
  sequences from independently selected labels, so the result need not embed
  tens of megabytes of repeated class sequences;
- complete ordered reconstruction-mismatch identities and veto-cell
  identities, plus per-frame `attributed_to_matched_box_count` and
  `depends_on_unmatched_collision_box_count`, the four truth-partition
  identity lists/hashes, and exact aggregate/family reconciliation;
- complete ordered scene box-parity records and their table SHA-256, with the
  exact `(scene_id,family)` set equal to the distinct set in selected frame
  reports and every source-geometry entry assigned an allowed role and one of
  those exact scenes;
- complete implementation source map and SHA-256 values;
- exact Python implementation/version and NumPy version used by the runner;
- the complete pre-freeze access-incident record, status, and canonical
  SHA-256 defined above;
- canonical JSON content SHA-256 after omitting only the content-hash field.

The strict finalizer independently recomputes, rather than copies: ordered
panel identities and counts; shard commitments and selected indices; selected
label bytes; aggregate/family class counts; support counts; mapping/support
bytes and gates; old-column span table; every ray sequence, transition table,
and associated hash;
every compact-record hash and aggregate/family/side reconciliation; veto,
mismatch, attribution-partition, and box-parity table hashes and internal
reconciliation; every fixed-mount quaternion/composition check and residual;
source-map and source-geometry hashes; incident-record hash; access-ledger
arithmetic; result content hash; and both authorization booleans. Label-stage
reconstruction itself remains runner-computed, but its complete
mismatch/veto/attribution identities are persisted, their partitions must be
exhaustive and disjoint, and the finalizer rejects any count not derivable from
those records. No central hash used by a gate may exist only as a
runner-supplied scalar.

All JSON object hashing uses UTF-8 canonical compact JSON with sorted keys and
no NaN or infinity. Arrays have the explicit byte encodings stated above.

## Machine implementation manifest and pre-run authorization

The reviewed human implementation report is fixed at
`docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.md`.
It must have a canonical machine companion at
`docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.json`
with schema
`lewm_go2_n32_camera_frustum_observability_audit_implementation_manifest_v1`.
The companion contains, at minimum:

- this amended binding's exact path and SHA-256 and the disclosed incident
  record/status/hash;
- exact path, file SHA-256, content SHA-256 where defined, schema, and semantic
  role for the panel, V4 adjudication report, KNOWN-bias proof, all 20 label
  shards, all 20 committed render summaries, and every transitively authorized
  plan, frames JSONL, frame-selection artifact, scene manifest, renderer
  source, and physical geometry contract;
- the exact eleven-role implementation source map and canonical source-map
  SHA-256, with no additional repository semantic source;
- exact Python implementation/version and NumPy version;
- every test, compile, import-isolation, and diff-check command, exit code,
  deterministic result/count, and captured-output SHA-256, including all
  production-provenance and denial mutations required here;
- the exclusive result path/schema, proof that it was absent, a zero-output
  pre-run state, the preparation access ledger, reviewer identity/status, and
  literal `authoritative_fit_audit_authorized=true`.

Because the exact shard and transitive source inventories originate in the
panel metadata, a post-binding, metadata-only manifest-preparation pass may
open and parse the panel and committed JSON/JSONL provenance after this
amended binding is hashed. It may not open a label shard, inspect an image,
reconstruct a target, or compute an audit statistic. Its phase-separated
ledger and exact inventory are embedded in the companion and carried into the
result/finalizer provenance; this is the only pre-run parsed-panel access.
Thus the companion can be complete without pretending those inputs were known
independently of the panel. Any access after this amended freeze that is absent
from that preparation ledger invalidates authorization.

The companion uses canonical compact sorted-key UTF-8 JSON, forbids nonfinite
numbers, includes `content_sha256` computed after omitting only that field, and
is itself raw-byte hashed. The human report binds the reviewed code/test
evidence; the companion binds the human report path/hash. The runner receives
the amended binding SHA-256 and companion file SHA-256 as explicit command-line
authorizations. Before its first panel, shard, summary, source-geometry, or
output-path access, it must parse the companion, verify its schema/content/file
hashes and explicit authorization, rehash every document and all eleven source
files, compare exact paths/roles/hashes/environment/test evidence, verify the
preparation ledger, and fail closed on any difference. Runner constants or a
Markdown table alone are not authorization.

## Access ledger and output

The access ledger must record, at minimum:

- panel metadata byte opens;
- label-shard hash byte opens and NPZ opens;
- exact registered arrays decompressed, materialized label rows, materialized
  supervision rows, selected label rows read, selected supervision rows read,
  unselected-row values inspected, unselected-row metrics computed,
  unselected rows retained, and derivative shard/cache writes;
- source-geometry hash byte opens and JSON parses;
- RGB byte opens and decodes;
- holdout, selection, calibration, physical-nontrain, G2, runtime, sealed, and
  model-output accesses;
- generated-V4-result and seed-20260711 accesses;
- every primary and modality denial counter, denied total, and unexpected path
  attempts defined above.

Every forbidden counter and unexpected-path count must equal zero. Label-shard
NPZ opens must equal the exact distinct selected fit-shard count, with no cache
eviction/reopen. Exactly 80 registered arrays must be decompressed. Define
`materialized_label_rows` as the sum of axis-0 lengths of `current_labels` and
`next_labels` across all 20 shards and `materialized_supervision_rows` likewise
for the two supervision arrays; record the exact per-shard values and require
the two totals to agree. Selected label rows and selected supervision rows must
each equal exactly 320 and reconcile one-to-one with the committed frame
identity set. Unselected-row inspection, metrics, retention, and derivative
writes must each equal zero. The sum of primary denial counters must equal both
`denied_attempts_total` and `unexpected_path_attempts`, all zero. Preparation,
runner, and finalizer ledgers remain distinct. The immutable runner result
contains the measured preparation and runner ledgers plus the frozen expected-
finalizer ledger contract. The finalizer does not mutate that result: its
strict stdout decision contains its measured finalizer ledger and the
independently recomputed three-phase reconciliation. The disclosed pre-freeze
incident is not folded into or erased by those fresh counters.

The exclusive immutable output path is
`.generated/go2_n32_camera_frustum_observability_audit/v2/result.json` with
schema `lewm_go2_n32_camera_frustum_observability_audit_result_v1`. Creation
must be exclusive and atomic. Synthetic tests and a reviewed implementation
manifest must freeze source/input hashes, exact geometry, mutations,
reconstruction, access denial, canonical hashing, and finalizer rejection
before this path may be created.

## Independent strict-finalizer evidence

The finalizer must run under the exact Python implementation/version frozen in
the machine manifest, with torch and NumPy unavailable for import. It may not
import the runner, audit core, or label/geometry semantic modules. It must
independently:

1. hash and validate this binding, the immutable incident record and status,
   the human and machine implementation manifests, the V4 adjudication report,
   the KNOWN-bias proof, and all eleven source-map files;
2. strictly parse the panel, derive the canonical 160 rows and 320 endpoint
   identities in the frozen order, and reconcile the full identity list/hash,
   exact shard commitments/row indices, 64-per-family, 160-per-side, and
   32-per-family-by-side counts;
3. raw-byte open each exact label shard once, verify its SHA-256, and use only
   standard-library `zipfile`, `ast`, and `struct` equivalents to decode the
   four registered current/next label and supervision NPY arrays from the
   in-memory bytes; require the exact twelve-member archive inventory frozen
   above without decompressing the eight auxiliary members, and reject
   extra/missing or duplicate archive names, encryption, duplicate NPY-header keys, object
   dtype, unsupported dtype/order, malformed/trailing headers or payload, and
   decompression sizes inconsistent with shape and dtype;
4. accept full-array materialization only under the same storage-boundary
   exception as the runner, select the exact panel-bound rows, and discard all
   four arrays before opening the next shard; never compute or retain an
   unselected-row value or metric;
5. independently recompute the old-column span table and literal
   mapping/support gates and, from the independently selected labels and
   supervision rows, selected-label bytes, class/support counts, every ray
   sequence and transition table, and every associated compact record/table
   hash;
6. derive the authorized source-geometry set from the companion manifest and
   panel provenance, independently raw-byte hash every source, validate strict
   provenance and exact scene/role membership, and reconcile the result's
   ordered source manifest and box-parity scene set without recomputing labels;
7. recompute all mismatch, veto, matched-box attribution,
   unmatched-collision dependence, ambiguity, family/aggregate partition, and
   table hashes from the complete compact identities persisted in the result;
8. independently recompute result content hash, preparation/runner/finalizer
   ledger arithmetic, all gates, and both authorization booleans.

The finalizer compares `sys.implementation` and the complete Python version
string to both manifest and result, and obtains the installed NumPy
distribution version through standard-library package metadata without
importing NumPy, again requiring exact manifest/result equality. A missing,
extra, or mismatched runtime version rejects the result.

The finalizer's label-shard byte access and standard-library decompression are
explicitly authorized only for these exact 20 fit shards. Its ledger separately
records 20 raw-byte hash opens, 20 NPZ container opens, 80 arrays decompressed,
materialized label/supervision row totals, exactly 320 selected label and 320
selected supervision rows, and zero unselected-row inspection/metrics/retention.
It may not create a derivative or cache and may not hash or stat an image. Each
geometry-source ledger record includes resolved path, semantic role, scene,
expected/observed SHA-256, and open count. Every input path, role, hash,
identity, selected row, family-side count, source scene, evidence partition,
environment version, access count, and decision must reconcile exactly or the
result is rejected.

## Authorization decision

The strict finalizer emits
`camera_frustum_representation_implementation_authorized=true` only if:

1. provenance, source hashes, reconstruction, and access reconciliation pass;
2. the proposed mapping is deterministic, in range, and injective with zero
   collisions;
3. every fit FREE and OCCUPIED label is supported and every unsupported label
   is UNKNOWN.

This authorization permits only implementation of the preregistered
camera-centered categorical representation. It does not authorize trained
model output, holdout access, seed 20260711, G2, runtime, or promotion.

The finalizer separately emits
`target_amendment_required_before_model_output=true` whenever
`rendered_collision_target_ambiguity=true`. Geometry implementation may
proceed if its three gates pass, but no dataset-backed successor model output
may be produced until a dated, source-grounded target amendment is frozen. An
amendment may remove genuinely unobservable privileged dependence or repair
provenance; it may not tune labels, support, or gates to V4 errors. If mapping
injectivity or known-target coverage fails, representation implementation is
not authorized and a new geometry binding is required before any retry.
