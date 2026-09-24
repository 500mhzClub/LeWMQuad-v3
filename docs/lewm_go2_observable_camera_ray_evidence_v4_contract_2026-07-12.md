# Observable camera-ray evidence V4 contract

Date: 2026-07-12  
Status: additive foundation and synthetic dataset/audit closure implemented;
exact 320-frame fit build remains unauthorized and unexecuted pending re-review

## Purpose

V4 defines the perception target that may feed persistent online memory. It
answers only this question:

> What FREE or OCCUPIED ground-plane evidence is supported by the current
> calibrated camera observation?

It does not answer whether the Go2 body can occupy a cell. Body clearance,
collision risk, morphology, route feasibility, and temporal fusion are later
stages with separately measured contracts.

This separation prevents a simulator map or robot footprint from being hidden
inside a target described as visual perception.

## Immutable schemas

- Evidence: `lewm_go2_observable_camera_ray_evidence_v4`
- Raster: `lewm_go2_observable_camera_ray_raster_v4`
- Reference implementation:
  `lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py`

The reference module is a pure NumPy transform. It performs no file I/O and
imports no renderer, simulator, dataset, learned model, or collision geometry.

## Coordinate frames and lattices

All XY coordinates are metres in the current yaw-aligned body frame:

- `forward` is the first array axis;
- `left` is the second array axis;
- source extent is `forward [-1.0, 5.4]`, `left [-3.2, 3.2]`;
- source lattice is `128 x 128` at `0.05 m`;
- registered stride-2 camera-ray lattice is `84 x 112` over the calibrated
  `168 x 224` image;
- output lattice is `64 x 64` at `0.10 m`, with the same closed outer extent.

Each source cell has five fixed ground support points: its centre and four
closed-footprint corners. The helper
`canonical_ground_support_points_body_m()` is the canonical point generator.

## Admitted evidence

An evidence payload has exactly these fields:

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `schema` | string | Exact V4 evidence schema |
| `camera_origin_body_m` | float32 `[3]` | Calibrated camera origin in the yaw-aligned body frame |
| `camera_basis_body_fru` | float32 `[3,3]` | Orthonormal camera forward/right/up basis in the yaw-aligned body frame |
| `ground_plane_z_body_m` | float32 scalar | Ground-plane height relative to the yaw-aligned body origin |
| `ground_support_in_frustum` | bool `[128,128,5]` | Ground support projects inside the calibrated image |
| `ground_support_clear_to_target` | bool `[128,128,5]` | Camera ray reaches that ground support before an opaque first surface |
| `pixel_hit_mask` | bool `[84,112]` | Pixel ray has a finite visible opaque first-surface hit |
| `pixel_first_hit_distance_m` | float32 `[84,112]` | Authoritative first-surface distance along the calibrated pixel ray |

`ground_support_in_frustum` must exactly match projection through the supplied
calibration. `ground_support_clear_to_target` must be false outside that mask.
Pixel distance must be canonical zero where `pixel_hit_mask` is false and
strictly beyond the `0.05 m` near plane where it is true. All numeric input must be
finite. The basis must be orthonormal and use the registered forward/right/up
handedness. Quaternion-derived axes are normalized with the same forward,
`right=cross(forward,up_hint)`, `up=cross(right,forward)` Gram construction as
the frozen source camera contract.

A ground support is clear when the nearest opaque first hit is at least the
target distance minus `1e-9 m`, matching the frozen physical-v3/perfect-ray
source tolerance.

`pixel_hit_xy_body_m` is a read-only derived value:

```text
hit_xyz_body = camera_origin_body + pixel_ray_body * first_hit_distance
```

It may be cached with an equality check but is never label authority. The same
calibration deterministically maps any body-frame ground query to image UV and
target distance, so later heads can consume arbitrary ground-query geometry
without simulator pose or a stored Cartesian answer.

These fields can be generated offline from rendered opaque first surfaces,
camera calibration, and camera/ground pose. At inference, a learned visual head
must predict equivalent evidence from RGB/JEPA state without simulator access.

## Forbidden target sources

The evidence and raster stages must not receive or derive labels from:

- `physical_free` or any complete scene occupancy/free map;
- collision boxes, collision overlap, or collision vetoes;
- body inflation, footprint dilation, or configuration-space morphology;
- route planner state, beacon position, future poses, or held-out answers.

The external mapping parser rejects physical-free, collision, morphology,
body-inflation, and configuration-space field names before accepting a payload.
Generated-data builders must preserve the same rejection at their boundary.

## Deterministic rasterization

The pure transform produces `UNKNOWN=0`, `FREE=1`, `OCCUPIED=2` labels.

### Source FREE

A `0.05 m` source cell is FREE if and only if all five ground supports are both
in-frustum and clear to their target:

```text
source_free[i,j] = AND_k(in_frustum[i,j,k] AND clear[i,j,k])
```

Any missing or occluded support keeps that source cell unknown.

### Output FREE

A `0.10 m` output cell is FREE if and only if all four exactly aligned `0.05 m`
source cells are FREE. There is no interpolation, erosion, dilation, collision
test, physical-free veto, or morphology in this rule.

### Output OCCUPIED

Each valid derived pixel first-surface XY hit is rasterized into every output
cell whose closed square contains the point. A hit strictly inside a cell marks
one cell; a hit on an internal edge marks two; a hit on an internal corner
marks four. A hit on the closed outer boundary belongs to its single edge cell.
Hits outside the closed output extent contribute no label. Because calibration
and first-hit range are authoritative float32 values, closed-edge supercover
uses an absolute `2e-5 m` tolerance.

### Precedence

OCCUPIED overrides FREE. A cell is UNKNOWN only when neither rule supplies
evidence. Raster arrays are immutable and content hashes are deterministic.

## Fit-only dataset and audit boundary

The new implementation is isolated from active dynamic-fit sources:

- builder: `scripts/build_go2_observable_camera_ray_fit_v4.py`;
- auditor: `scripts/audit_go2_observable_camera_ray_fit_v4.py`;
- immutable dataset root:
  `.generated/go2_observable_camera_ray_fit_v4/v1/`.

The builder admits exactly the registered `160` physical-train transitions /
`320` endpoint frames, the existing allowlisted train source geometry, and only
the `train` attitude-sidecar role. It does not open RGB or fit-label bytes.
Checkpoint-selection, calibration, G2, held-out, runtime, sealed, checkpoint,
and model-output payloads are forbidden. Legacy rollout split is recorded
provenance only; current physical `dataset_role=train` governs access.

Twenty scene shards contain canonical index JSONL plus raw little-endian arrays.
Every file has a byte SHA-256, every frame has an evidence/raster content hash,
and the top-level canonical content hash covers the complete shard and input
inventory. The top-level manifest also carries a canonical `rgb_receipt` with
one entry per exact frame key: the RGB path and SHA-256 are copied verbatim from
the frozen fit-panel metadata without opening RGB bytes. This is the sole RGB
path authority for a later trainer; it may not rediscover or guess paths.

Six spawned CPU workers are the maximum; all native numerical thread
counts are one, results merge canonically, and one-worker versus six-worker
synthetic artifacts must be byte-identical. Every exact worker validates the
complete transitive source map before and after computation; the parent
revalidates that map and the frozen source-audit semantic graph before publishing
the manifest. No GPU is used.

Exact-mode imports are neutral until the existing source authorization installs
its pre-hashed semantic namespaces. The reviewed V4 implementation hash and the
respective build/audit authorization boolean are required inside every function
that can open exact geometry, sidecar, or label payloads; CLI ordering is not an
access control. The source closure includes the dynamic projection, attitude
sidecar, source-access runner, and every semantic module it executes.

Before hard-link publication, the parent requires the exact regular-file
inventory and verifies every byte count and hash against the returned shard
manifest. It repeats those checks on the published tree before manifest-last,
confines the exact output beneath the repository root, and fsyncs the output
parent. Staging creation and every publication step are inside one cleanup
scope, so worker, hard-link, inventory, and manifest failures remove all
partial dataset and staging state. The builder owns the destination only after
its exclusive directory creation and removes that entire owned destination on
every exception through post-manifest validation and fsync; only a successful
function return makes publication durable. Denial receipts require the exact complete
primary-reason and modality key sets, strict nonnegative counts, and equality
between both nested totals, the overall total, unexpected-path total, and
record count. Any denied or unexpected access fails the run.

The separate auditor first enumerates the exact dataset-root, `shards/`, shard
directory, and shard-file inventories, rejecting extras, symlinks, and
nonregular entries. Declared shard-file records must also be one-to-one: a
duplicate path is rejected before records can be collapsed into a lookup map.
It then verifies every V4 byte, joins every shard index to
the manifest-bound RGB receipt, and reconstructs every frame without opening
RGB. In exact mode it additionally joins each receipt path and hash back to the
frozen fit-panel metadata before any label-shard payload is opened.
Only its exact-fit mode may then open the 20 registered train-fit label shards.
Before that open it validates the complete dataset build receipt: reviewed
producer/source hashes, zero forbidden build ledger, sidecar role counts,
resource and publication contracts, array layout, false promotion licenses,
and exact 320-frame/20-scene train scope.
It scores all `1,310,720` cells and reports the legacy physical-v3 confusion and
mismatches without adding a physical-free prior or collision veto. The builder
and auditor remain blocked behind a reviewed implementation-manifest hash; the
exact payload run has not occurred.

## Architectural boundary

V4 is the visual evidence layer in this chain:

```text
RGB -> frozen JEPA state -> learned camera evidence head
    -> deterministic V4 raster -> persistent online memory
    -> separately learned body-risk/traversability head -> exploration planner
```

Persistent memory may fuse observations over time using measured ego-motion.
The body-risk head may learn Go2-specific traversability from execution
outcomes, but it must not rewrite the provenance of V4 visual evidence.

## Acceptance gates

Before this target can replace an existing benchmark target, all of these are
required:

1. Synthetic contract tests pass for closed boundaries, four-source FREE AND,
   OCCUPIED precedence, determinism, immutability, and forbidden fields.
2. A source audit proves every generated field comes only from admitted camera
   evidence and pose/calibration inputs.
3. Exact perfect-evidence rasterization is first measured on the registered
   320-frame train-only N32 panel before any visual head is trained.
4. Only after the complete N32 ladder passes may a learned head be evaluated on
   the two registered train-role holdouts, with FREE precision, OCCUPIED
   precision/recall, UNKNOWN calibration, and boundary metrics reported.
5. Closed-loop claims are compared on the unchanged navigable held-out maze
   manifest; current physical safety and claiming benchmarks remain unchanged.

Passing this foundation establishes only the label mechanics. It does not by
itself establish learnability, online-memory correctness, safe traversability,
coverage improvement, or beacon-claim generalisation.
