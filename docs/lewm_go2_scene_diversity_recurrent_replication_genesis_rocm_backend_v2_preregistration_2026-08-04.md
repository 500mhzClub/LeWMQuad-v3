# Genesis ROCm backend V2: exact `ld.lld` driver successor

## Status and scope

This document preregisters a fresh infrastructure successor. It is not a V1
retry, resume, refill, overwrite, integrity replacement, or scientific pivot.
It grants no qualification or scientific execution authority.

The V1 qualification was consumed before `rocminfo`, HIP, Genesis import,
Genesis initialization, EGL, VRAM selection, worker launch, scene execution,
or RGB generation. The independently audited terminal review is bound at
SHA-256 `3e35cdb459c18d862e21df676b0a630a0496d1a26f8a97874095c71ab2facb5b`
and 14,742 bytes. V2 binds only that review document as source-level identity
and failure evidence. It does not bind, open, or reuse any V1 runtime,
reservation, terminal, cache, worker, scene, RGB, or result payload.

## Material hypothesis

The bound regular binary `/opt/rocm-7.1.1/lib/llvm/bin/lld` is a generic
multi-driver executable whose dispatch depends on `argv[0]`. V1 verified that
both linker lookup routes resolved to this target and then incorrectly invoked
the resolved target itself. That command exits nonzero and asks the caller to
use `ld.lld`.

V2 invokes the exact unresolved Unix-driver symlink:

```text
/opt/rocm-7.1.1/lib/llvm/bin/ld.lld --version
```

The plan requires that exact lexical path, symlink link text `lld`, resolution
to the separately SHA-256/byte-count-bound regular target, and stdout prefix
`AMD LLD 20.0.0`. Direct production invocation of the regular target is
forbidden. Both the `PATH` lookup and `$ROCM_PATH/lib/llvm/bin/ld.lld` must be
the exact driver path before resolution.

This correction proves only that preflight can select the intended LLD Unix
driver. It is not evidence that HIP, Genesis, rendering, physics,
qualification, representation learning, planning, or navigation will pass.

## Frozen science and fresh custody

V2 is rebuilt from the same frozen Vulkan scientific plan as V1. The 64-scene
panel and roles, four states per scene, two history blocks, nine actions,
horizons, action-conditioned successors, RGB/depth counts, DINOv2 binding,
visual and zero-vision arms, seeds 411/412/413/414, 800 updates, train-only
fitting, live task/action baseline, checkpoint boundary, evaluations,
bootstrap, caps, and decision gates are unchanged.

The fresh scientific identity is
`go2-scene-diversity-recurrent-replication-genesis-rocm-backend-v2`; its root
is `.generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2/attempt_v1`.
The fresh qualification identity appends `-qualification`; its root is
`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_qualification/attempt_v1`.
Each has its own role-local `quadrants_cache`. No V1 payload or cache may cross
either boundary, and qualification output is permanently ineligible for
science.

## Qualification and hard stop

After independent source review and a separate one-shot authority, V2 first
consumes its fresh qualification root and only then runs preflight. Preflight
must execute the asserted `ld.lld` symlink path, observe exact AMD LLD 20,
then pass the unchanged `rocminfo`, single R9700/gfx1201 HIP identity, imported
module-path, EGL Device 1, and DRM identity checks.

Only then may two fresh full-scene workers run in fixed order: scene 12 and
scene 0. Each executes four states, two complete history blocks, nine actions,
36 branches, 12 context frames, 36 target frames, 48 stored RGB renders, and
48 transient depth validations. Each worker has the unchanged 300-second hard
watchdog. Feasibility remains
`64 * max(worker_elapsed_seconds) + 900 <= 7200`.

Any preflight, worker, renderer, physics, VRAM, kernel, count, finite-value,
binding, timing, or inherited scene-gate failure consumes and hard-stops V2.
There is no retry, resume, refill, overwrite, partial reuse, or second V2
attempt. A PASS only permits later consideration of a separately built
scientific authority; it makes no world-model or navigation claim.
