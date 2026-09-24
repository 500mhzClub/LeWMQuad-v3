# Go2 G3 native learned physical projection V1 implementation handoff

Date: 2026-07-13

Status: **development-only candidate; awaiting different-agent source review**

## Purpose

This additive boundary converts one exact, instance-issued synthetic V4 raw
outcome into retractable learned evidence on the native `0.05 m` physical
lattice. It exercises the intended G2-to-G3 contract without opening a model,
checkpoint, report, held-out scene, GPU path, hardware path, or navigation
path.

It does not enable production learned admission. All production identities and
the production adapter remain `None`.

## Artifacts

- implementation:
  `lewm/planning/native_learned_physical_projection_v1.py`
  - SHA-256:
    `f8b149c685a4320ae938ff367edcf833047016250caae7699cddfe8026cc0634`
- tests:
  `lewm/tests/test_native_learned_physical_projection_v1.py`
  - SHA-256:
    `1f47ee15e46be1e8d5407ffa6f39f753b2dba92d15be67af8217ab4e146b5661`

The hashes above are the candidate bytes before this handoff file was added.

## Public development API

```python
outcome = runner.issue(
    snapshot=configuration_snapshot,
    pose=pose_provenance,
    source_geometry=native_0p05_geometry,
    ground_clear_query_tensor=raw_ground_rows,
    ordered_ray_hit_depth_tensor=raw_ray_rows,
    rgb_frame_id=rgb_frame_id,
    rgb_frame_sha256=rgb_frame_sha256,
    raw_outcome_file_sha256=raw_outcome_file_sha256,
)

package = NativeLearnedPhysicalProjectionAdapterV1.issue(
    configuration_snapshot,
    outcome,
)
receipt = adapter.commit(package)

retraction = adapter.issue_retraction(current_snapshot, package)
retraction_receipt = adapter.commit(retraction)
```

Both constructors require the private synthetic-fixture opt-in. The raw outcome
and transaction package are exact-object, non-copyable, non-serializable,
single-use capabilities. Their issuance registries retain the original issued
digest, so mutating an object and recomputing its own digest still rejects.

## Bound raw authority

`SyntheticNativeV4RawOutcomeV1` carries raw ground-clear logits and query
geometry plus ordered ray hit logits and depths. It carries no caller labels or
aggregate metrics. Its content binds:

- runner, inference, projection, and access-ledger source identities;
- exact checkpoint, passed-G2-report, and frozen-calibration identities;
- RGB frame ID and bytes, raw-outcome bytes, and pose provenance;
- both reset-local map frames, both shapes, both revisions, physical content,
  configuration snapshot, and projection-source identities;
- the native source geometry and complete raw tensor content.

The adapter accepts only the exact live outcome from its runner and consumes it
when it issues the development transaction.

## Conservative projection

- The only thresholds are stored in the exact frozen synthetic calibration
  object.
- Source geometry must be native `128 x 128` at `0.05 m` with the exact frozen
  origin. A `0.10 m` source or an upsampling derivation rejects.
- FREE evidence is computed from closed native source-cell squares. A physical
  destination cell is FREE only when its whole closed square is covered for
  every member of the finite registered pose/camera transform set.
- OCCUPIED evidence is the union closed supercover of thresholded ordered-ray
  hit locations over every transform.
- OCCUPIED overrides FREE. Cells in the projected domain with neither witness
  remain UNKNOWN.
- Projection is origin-aware and requires the exact current physical and
  configuration frames, `2:1` shapes, revisions, and snapshot identities.
- Covariance above the frozen envelope rejects rather than widening authority.

The package never exposes the ordinary `PhysicalEvidenceTransaction`. The
adapter reconstructs that transaction privately at commit and requires its
digest to match the admission. Learned evidence remains observation-scoped and
is retracted only through the exact active committed projection package.

Every receipt, admission, and package serializes and hashes
`development_only=true`, `hardware_execution_authorized=false`, and
`production_promotion_authorized=false`.

## Verification executed

All runs explicitly hid ROCm/HIP/CUDA devices and limited numerical worker
threads to one per process.

```text
test_native_learned_physical_projection_v1.py       34 passed in 63.08s
test_revisioned_physical_configuration_memory.py    32 passed in 0.19s
test_two_resolution_configuration_projection_v2.py  14 passed in 37.40s
```

The focused suite covers full-square FREE, translation, rotation,
boundary-touching OCCUPIED supercover, OCCUPIED precedence, covariance limit
and excess, native-resolution enforcement, query-geometry validation, wrong
origin, both shape bindings, every issued identity class, copied/stale/replayed
objects, foreign adapters, rehashed object mutation, contradiction recovery,
and exact observation retraction.

`py_compile`, `git diff --check`, and the source-surface scan also completed
cleanly. The source-surface test forbids Torch, NumPy, CUDA, ROCm, checkpoint
loading, file opening, and held-out access in this module.

Read-only dependency hashes observed at handoff:

- `revisioned_physical_configuration_memory.py`:
  `13fccc662784c0a7eed75965a9d4154369666f26e804173482b461c55b8b9add`
- `test_revisioned_physical_configuration_memory.py`:
  `a60c6f21cb0e40966216428c938e82024eafcaded95a874c53b542befb9065d4`
- `two_resolution_configuration_projection_v2.py`:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107`
- `test_two_resolution_configuration_projection_v2.py`:
  `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c`

## Explicit exclusions and next gate

This candidate does not claim a real camera-model reprojection, a real V4/V5
runner, a passed immutable G2 identity, source-isolation protection against
runtime file replacement, promoted-runtime admission, cold-start authority,
view-diversity calibration, executor correction, hardware readiness, or
navigation readiness.

A different agent must review the exact implementation and test bytes. Only
after that review may a separately authorized task bind real checkpoint/G2
artifacts and extend the synthetic runner boundary. Production globals must
remain unset until all prerequisites in the preregistered G3 plan are met.
