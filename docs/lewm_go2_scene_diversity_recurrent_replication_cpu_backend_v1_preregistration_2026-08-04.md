# Go2 scene-diversity recurrent replication CPU backend V1

Date frozen: 2026-08-04

Status: `PREREGISTERED_QUALIFICATION_NOT_RUN`

This is a materially different backend successor after the final V3 Vulkan
hard stop. It is not V4, an integrity replacement, a retry, or a resume. It
grants no sealed, held-out, scientific-execution, navigation, planner,
promotion, deployment, partial-reuse, refill, overwrite, or retry authority.
Independent source review, a separate one-shot qualification authority, exact
PASS qualification, and a later scientific authority are all mandatory.

## Consumed Vulkan attempts

Original V1 and integrity replacements V1, V2 and V3 are consumed. No scene,
receipt, RGB, mesh, render, reservation, checkpoint, tensor, metric, process
result or other partial output from any of them may be opened, copied, joined,
resumed, refilled, screened or admitted here.

V3's terminal is SHA-256
`0d54d5c733a074098bd6d740d71a3358700e5e608ec019b4cdbbd47e1012ff4c`
and 442 bytes. Its review is SHA-256
`73360d1db0a65a29f2a825f32899337ce7ad53894f3c153dc18c7c973d9243a9`
and 21,073 bytes. V3 repeated V2's AMDGPU ring timeout, recoverable device
wedge/reset and GsTaichi Vulkan semaphore abort at exact worker/scene 12 after
12 completed workers. It published no combined physics result and entered no
DINO, training, checkpoint, evaluation, metric or gate stage. The audited V3
hard stop prohibits another Vulkan identity replacement.

## Exact material change

The scientific CPU plan differs from the original frozen scientific plan in
exactly four fields:

1. `attempt_id` is the new CPU-backend identity.
2. `output_root` is the fresh CPU scientific collection root.
3. `execution_contract.backend` changes from `vulkan` to `cpu`.
4. `execution_contract.environment.GS_BACKEND` changes from `vulkan` to
   `cpu`.

The Python invocation and Genesis 0.3.14 environment remain exactly pinned to
the existing `genesis_render_vulkan` interpreter. `EGL_DEVICE_ID=1`,
`MESA_VK_DEVICE_SELECT=1002:7551!`, `PYOPENGL_PLATFORM=egl`, the exact
Vulkan/EGL graphics preflight, and AMD Radeon AI PRO R9700 rendering remain
unchanged. Genesis physics executes on CPU; raster rendering remains on the
bound EGL/R9700 route. CPU physics numerics may materially differ from Vulkan
and no cross-backend bit-equivalence claim is made.

All other data and science remain frozen: the exact 64-scene panel and role
allocation, four states per scene, history tapes, nine actions, horizons,
3,072 RGB frames, render/depth-validation gates, one fresh process per scene,
plan-first Genesis seed, scene-local caches, DINOv2 binding, visual and exact
zero-vision arms, model seeds 411/412/413, sampler seed 414, 800 updates,
train-only PCA/statistics, live task/action baseline, durable checkpoint
boundary, two exact evaluations, metrics, bootstrap and five decision gates.

The selected-device VRAM cap remains 16,977,405,952 bytes because rendering
still uses the R9700. The full scientific wall cap remains 7,200 seconds.

## Mandatory separate qualification

Qualification has its own exact plan, root, authority, reservation, result and
terminal namespace. Its root is
`.generated/dev/go2_scene_diversity_recurrent_replication_cpu_backend_v1_qualification/attempt_v1`.
The scientific root must remain absent throughout qualification. Qualification
authority is valid only after independent source review and cannot authorize
the full collection.

Exactly two fresh full-scene probes run in this fixed order:

- frozen scene index 12, the repeated Vulkan failure scene;
- frozen scene index 0, a distinct reference scene.

Each probe executes all four frozen states, both full history blocks, all nine
candidate successors per state and the full horizon: 36 branches, 12 context
frames, 36 target frames, exactly 48 stored RGB renders and 48 auxiliary depth
validation renders. Existing receipt, input-binding, physics, render-quality,
depth, count, nonfinite, mesh-cache and output-binding validation applies.

Each worker is a fresh process group under a 180-second hard watchdog. The
selected-device monitor and release barrier remain mandatory. Any nonzero or
incomplete worker, timeout, nonfinite value, render/count/integrity failure,
VRAM-cap breach, or new AMDGPU ring timeout/reset fails qualification.

Timing qualification passes only when
`64 * max(worker_elapsed_seconds) + 900 <= 7200`. The 900 seconds reserve
graphics preflight, final joins, DINO, training, checkpoint and both
evaluations. This is a fixed gate, not an adaptive estimate.

Probe scene outputs are non-scientific and permanently forbidden from reuse
in the 64-scene collection. A PASS result only permits the later scientific
authority builder to consider issuing authority; it makes no representation,
model, planning or navigation claim. The scientific authority must bind and
revalidate the exact PASS result and cannot exist before it.

## Scientific decision and hard stop

If qualification passes and exact scientific authority is subsequently
issued, the scientific run starts all 64 scenes from the fresh root
`.generated/dev/go2_scene_diversity_recurrent_replication_cpu_backend_v1/attempt_v1`.
No probe or Vulkan predecessor artifact is reusable.

The unchanged gates are: integrity/custody pass; visual normalized rank regret
at most 0.13; visual minus live task/action regret at most -0.02 with paired
95% upper bound below zero; visual minus no-vision regret at most -0.01 with
paired upper bound below zero; and visual beats uniform random expectation.
A pass supports only the preregistered incremental visual hypothesis, not
planning utility.

Any qualification or scientific CPU-backend infrastructure failure hard-stops
this CPU backend. There is no CPU retry, resume, partial reuse or second CPU
attempt. A future Genesis 0.4.6 ROCm backend would require separate
qualification, preregistration, source review and authority. This document
grants none.
