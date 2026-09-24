# Go2 G3 native learned physical projection V2 handoff

Date: 2026-07-13

Status: **development-only successor candidate; awaiting different-agent review**

## Purpose

V2 is an additive lifecycle successor to the frozen, independently blocked V1
candidate. It closes the two V1 retraction findings without modifying V1 or
claiming any new inference, geometry, production, hardware, or navigation
authority.

V2 reuses V1's frozen raw V4 outcome, conservative native-`0.05 m`
projection, admission, receipt, and hidden physical-transaction contracts. It
adds an exact-identity retraction reservation state machine around a private V1
engine. The public V2 adapter is composition-based, not a V1 subclass, so a
caller cannot bypass the successor lifecycle through ordinary base-method
dispatch.

## Candidate artifacts

- implementation:
  `lewm/planning/native_learned_physical_projection_v2.py`
  - SHA-256:
    `327f3f7ab42ae39b416d54936bba6d39febdf6d85cea46c6acd7075c79716f40`
- focused regression tests:
  `lewm/tests/test_native_learned_physical_projection_v2.py`
  - SHA-256:
    `691e9d8a101044cb4b189f10a272bc5c633bf408724c657d66825c86651ca25b`

These hashes identify the candidate bytes before this handoff file was added.

## Frozen V1 evidence preserved

The successor task did not edit any V1 evidence byte. Hashes at handoff are:

- V1 implementation:
  `f8b149c685a4320ae938ff367edcf833047016250caae7699cddfe8026cc0634`
- V1 candidate tests:
  `1f47ee15e46be1e8d5407ffa6f39f753b2dba92d15be67af8217ab4e146b5661`
- V1 implementation handoff:
  `caccd6204e394bd07e7c1f3d15b35775de20ac6fa2e17027d63efc5c326dbb2a`
- V1 independent adversarial tests:
  `787b6d1ba10f24161ad355aef13a84e9891556d42d40693a02c803779b342ac3`
- V1 independent BLOCK review:
  `5a41793bec15ea72ba89d5ce35e07746c44f3526dc4f16ce4f68a3ca30c9d07e`

The V1 verdict remains BLOCK. V2 does not rewrite that history.

## API

The raw runner contract remains the frozen V1 synthetic contract:

```python
outcome = runner.issue(
    snapshot=snapshot,
    pose=pose,
    source_geometry=native_geometry,
    ground_clear_query_tensor=raw_ground_rows,
    ordered_ray_hit_depth_tensor=raw_ray_rows,
    rgb_frame_id=rgb_frame_id,
    rgb_frame_sha256=rgb_frame_sha256,
    raw_outcome_file_sha256=raw_outcome_file_sha256,
)

package = adapter_v2.issue(snapshot, outcome)
projection_receipt = adapter_v2.commit(package)

current = projection.project()
retraction = adapter_v2.issue_retraction(current, package)
retraction_receipt = adapter_v2.commit(retraction)
```

`NativeLearnedPhysicalProjectionAdapterV2` requires the same explicit
synthetic-fixture opt-in as V1. It assigns a distinct V2 adapter contract hash.

## V2 retraction lifecycle

Each reservation stores the exact target package identity and issuance digest,
the exact retraction package identity and issuance digest, and the exact bound
snapshot identity and issuance digest. Two maps provide the lifecycle:

- exact retraction package identity to its reservation record;
- exact target observation/package identity to its sole LIVE reservation.

The only states are LIVE, STALE, and CONSUMED.

### Issue

Before issuing any retraction, V2 requires:

- the exact current V2 snapshot;
- the exact active committed projection package from this adapter;
- self-consistent package content equal to the immutable digest retained at
  original issuance;
- exact presence in the committed-observation registry and active learned
  memory.

If a LIVE reservation already exists and its snapshot is still current, a
second issuance rejects. If its exact snapshot can no longer commit, the old
record becomes terminal STALE, its live-target reservation is released, and a
fresh exact package may be issued against the current snapshot.

### Commit

Immediately before evidence removal, V2 rechecks both the target and retraction
packages against their immutable original issuance digests. An unreserved,
copied, reloaded, transferred, forged, or mutated retraction cannot enter the
private V1 commit engine.

If the bound snapshot is stale, the commit rejects and the reservation becomes
terminal STALE. A subsequent current-snapshot issue can then create a new LIVE
package while the target evidence remains active. The old package remains
terminal and can never be revived or accepted.

Successful removal transitions the exact reservation to CONSUMED, releases its
live-target slot, and retains normal package replay rejection.

## Preserved contracts

- Raw inputs remain typed runner-issued logits/query geometry and ordered ray
  hit/depth tensors, never caller labels or aggregate metrics.
- FREE remains complete closed destination-square coverage for every frozen
  transform. OCCUPIED remains the closed union supercover. OCCUPIED precedence,
  UNKNOWN fallback, origin-aware geometry, native resolution, and covariance
  rejection are inherited unchanged from the frozen V1 engine.
- Active-observation, exact snapshot/revision, one-use, copy, transfer, replay,
  serialization reload, and `object.__new__` rejection remain enforced.
- The ordinary `PhysicalEvidenceTransaction` remains hidden and is rebuilt and
  digest-checked only inside the private engine.
- Adapter, admission, receipt, and package surfaces remain
  `development_only=true`, `hardware_execution_authorized=false`, and
  `production_promotion_authorized=false`.
- All V2 production runner/checkpoint/G2/calibration/adapter globals are `None`.

## Verification

Every run disabled external pytest plugins, set numerical worker counts to one,
and hid HIP, CUDA, and ROCr devices.

```text
V2 focused successor suite                         12 passed in 40.07s
frozen V1 candidate suite                          34 passed in 65.07s
frozen revisioned physical memory suite            32 passed in 0.20s
frozen two-resolution configuration projection     14 passed in 39.63s
                                                    -------------------
green successor and regression total               92 passed

frozen V1 independent BLOCK suite                   1 passed, 2 failed
```

The two V1 independent failures are the exact frozen mutation-acceptance and
stranded-retraction findings. Their continued failure is expected negative
evidence. The adapted V2 cases pass, including:

- mutation rejection both before retraction issue and immediately before
  evidence removal;
- rejection of a second concurrent LIVE retraction;
- proactive stale detection and stale detection on failed commit;
- successful fresh retry while the target remains active;
- permanent rejection of the old STALE package;
- transition to CONSUMED and successful target removal;
- retraction copy/reload/transfer/replay rejection;
- preserved geometry, resolution, covariance, authority, and hidden-transaction
  behavior.

`py_compile`, `git diff --check`, line-length, and forbidden-source-surface
checks completed cleanly. The V2 implementation contains no Torch, NumPy,
accelerator, file-opening, checkpoint-loading, or held-out access surface.

Read-only adjacent dependency hashes at handoff:

- `revisioned_physical_configuration_memory.py`:
  `13fccc662784c0a7eed75965a9d4154369666f26e804173482b461c55b8b9add`
- `test_revisioned_physical_configuration_memory.py`:
  `a60c6f21cb0e40966216428c938e82024eafcaded95a874c53b542befb9065d4`
- `two_resolution_configuration_projection_v2.py`:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107`
- `test_two_resolution_configuration_projection_v2.py`:
  `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c`

## Explicit exclusions and next gate

No real V4/V5 checkpoint, G2 report, held-out scene, accelerator, hardware, or
navigation input was opened. V2 does not claim real runner/source isolation,
real G2 calibration, view diversity, executor correction, cold-start authority,
promotion, hardware readiness, or navigation readiness.

A different agent must review the exact V2 implementation and test bytes. No
downstream integration or real-artifact binding is authorized before that
review records a source-level PASS.
