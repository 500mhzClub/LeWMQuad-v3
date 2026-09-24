# Shared JEPA V5 development raw-supervision builder V1 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_builder_arch`

Status: **AUTHOR COMPLETE; DIFFERENT-AGENT REVIEW REQUIRED; EXACT BUILD NOT RUN**

## Scope

V1 is an additive, exact-capable builder for the three frozen development
roles. It converts all 9,460 unique planned endpoints into reviewed observable
camera-ray V4 evidence plus the deterministic V4 three-state raster, while
retaining all 10,344 pair endpoint references and all 5,172 parent pairs.

The exact entry point is fail-closed behind a future machine authorization that
must bind PASS reviews of both this builder and a separately implemented
auditor. The absent authorization rejects before metadata or referenced-source
access. No exact build, source frame, source scene manifest, render plan,
render summary, RGB byte, parent label shard, G2 payload, checkpoint, model
output, runtime result, held-out input, hardware, or production artifact was
opened during authorship or verification.

## Frozen parents

| Artifact | SHA-256 |
|---|---|
| `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_preregistration_2026-07-13.md` | `07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_source_inventory_amendment_2026-07-13.md` | `39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3` |
| metadata plan V5 source | `67c4d325ddab3ac3405e231b78681f4b9ef17b4833ca199395f24ed7a8b82921` |
| metadata plan V5 author handoff | `b362d26372f01e670a477dda5e7abb5e55370cc1d8d89052545afa229e7bba66` |
| metadata plan V5 independent QA | `8a50bcf5275d243f06b92264e017f355fd54faaca8f8e73aab1e3cc45dc51298` |
| metadata plan V5 independent review | `7d7344e423492a3cf36d1cd50ca09e6c7eb6eba17c25861c840531465aaf7706` |

The exact loader freezes the reviewed V4 evidence, fit builder, source-access,
dynamic projection, scene-manifest, paired-navigation, geometry-contract, and
planning-grid source hashes before any referenced source can open.

## Candidate identities

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v1.py` | `3bc1559776e2f8471bb6a7a1ddd8808b1f1224687dedf280fd2300820afe25ec` |
| `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v1.py` | `df5fd60b50ba852d44fd6fe0034c7e763fc08030875488be3850e774906ceeb3` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v1.py` | `15767446ba45851a7f5774560db8e8f6f87d831a51fde7585acffa028f3ba2e4` |

## Exact source conversion

The source loader first reproduces the independently passed metadata V5 result:

- 5,172 pairs, 10,344 endpoint uses, 9,460 unique endpoints, and 88 scenes;
- plan content `8004ab0d...921a3`;
- ordered pairs `76810dba...b5ea`;
- ordered endpoints `8130e961...f698`; and
- all five frozen source-inventory hashes.

Each of at most six spawned scene workers opens only its allowlisted frames,
scene manifest, render plan, and render summary through a filesystem-root
anchored `O_NOFOLLOW` descriptor chain. Complete seven-field fingerprints for
every ancestor and leaf are checked before and after each read. The worker then:

1. validates render-plan/summary camera calibration and source commitments;
2. joins every endpoint by scene, episode, environment, step, frame, timestamp,
   and image SHA-256;
3. calls the reviewed V4 camera-mount composition and attitude cross-check;
4. calls the reviewed full-RPY rendered-object parity and box transform;
5. calls `build_frame_evidence_v4()` exactly once per unique endpoint; and
6. calls `rasterize_observable_camera_ray_evidence_v4()` for the derived label.

No RGB path is dereferenced. The legacy physical label-shard path and hash stay
only in the unchanged pair records.

## Published artifact

The sole final directory is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`

It contains `manifest.json`, unchanged canonical `pairs.jsonl`, a directly
joinable `endpoints.jsonl`, and one `shards/<sha256(scene_id)[:16]>/` directory
per scene. Every shard has:

| File | dtype | shape |
|---|---|---|
| `camera_origin_body_m.f4` | `<f4` | `[N,3]` |
| `camera_basis_body_fru.f4` | `<f4` | `[N,3,3]` |
| `ground_plane_z_body_m.f4` | `<f4` | `[N]` |
| `ground_support_in_frustum.u1` | `u1` | `[N,128,128,5]` |
| `ground_support_clear_to_target.u1` | `u1` | `[N,128,128,5]` |
| `pixel_hit_mask.u1` | `u1` | `[N,84,112]` |
| `pixel_first_hit_distance_m.f4` | `<f4` | `[N,84,112]` |
| `raster_labels.u1` | `u1` | `[N,64,64]` |
| `index.jsonl` | canonical JSONL | `[N]` |
| `shard.json` | canonical JSON | scalar manifest |

Endpoint rows directly bind the endpoint identity and original plan-endpoint
hash to the scene shard, shard row, RGB path/hash commitments, V4 evidence
hash, and V4 raster hash. The top manifest inventories every non-self regular
file by path, bytes, and SHA-256; its own canonical `content_sha256` provides
the non-recursive self commitment.

Every license remains false, including independent-audit PASS and dataset-use
authority. The independently implemented auditor must reconstruct all indexes,
arrays, hashes, role boundaries, and raw V4 evidence before use.

## Audit precommit

The manifest deterministically selects one endpoint per observed
`(dataset_role, family)` by minimum:

`SHA256(role || NUL || family || NUL || endpoint_identity_sha256)`

The frozen exact population must produce 24 records. The auditor must
reconstruct those endpoints from original development geometry; the builder's
own checks cannot authorize the result.

## Process and publication

- CPU scene workers: at most six, `spawn` start method.
- Native worker threads: all four thread variables fixed to one.
- GPU visibility: CUDA, HIP, ROCr, and ordinal variables empty.
- Merge: canonical role, scene, and endpoint-identity order.
- Revalidation: metadata V5 is rerun and every one of the 354 source files is
  rehashed again immediately before manifest creation.

Publication retains a canonical parent descriptor reached by a complete
component/no-follow walk. Staging is a private mode-0700 sibling created
relative to that descriptor. Parent and staging identities are rechecked
throughout. After complete file inventory, validation, and fsync, commit is one
`renameat2(..., RENAME_NOREPLACE)` using the same retained parent FD, followed
by parent-FD fsync. An occupied destination is never replaced.

Cleanup removes only a named directory whose current device/inode still equals
the builder-owned staging identity. A replacement is preserved. After a
verified-authority failure, an exclusive external `development_raw_supervision_v1.failed.json`
receipt records terminal no-retry state without overwriting any existing leaf.

## Verification

All pytest commands fixed the four native thread variables to one, hid all
accelerators, disabled external pytest plugins, and used only synthetic source
fixtures plus the already reviewed metadata-only V5 inputs.

```text
Focused builder V1 synthetic suite:              15 passed in 1.43s
Builder + reviewed V4 + metadata V5 matrix:     103 passed, 2 deselected
py_compile (source, CLI, tests):                  PASS
git diff --check (source, CLI, tests):            PASS
ASCII check (source, CLI, tests):                 PASS
```

The two deselections are frozen predecessor tests that require the old V4
implementation manifest to be unauthorized. That manifest is authorized in the
current repository; the unchanged full run reports those two stale assertions
and 103 passes. They do not exercise this V1 implementation.

## Required different-agent review

A reviewer other than `/root/raw_builder_arch` must independently:

1. rehash this source, CLI, test, and handoff plus every frozen parent;
2. replay gate ordering and prove absent/invalid authority reaches no metadata
   or referenced-source opener;
3. probe descriptor continuity, source joins, full-RPY parity, duplicate/orphan
   rejection, and exactly-once endpoint scheduling;
4. reproduce one-worker/six-worker byte identity and the eight-array layout;
5. adversarially test occupied output, parent/staging replacement, cleanup
   ownership, failure-receipt preservation, and true no-replace rename; and
6. issue PASS or BLOCK without exact source construction.

Even a builder PASS does not authorize exact construction until the separately
implemented auditor also passes and one machine authorization binds both exact
reviewed source closures.
