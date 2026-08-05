# Go2 scene-diversity recurrent replication Genesis ROCm backend V1

Date frozen: 2026-08-04

Status: `PREREGISTERED_PLANS_ONLY_QUALIFICATION_NOT_AUTHORIZED`

This is the separately named Genesis 0.4.6 ROCm/HIP backend successor required
by the consumed CPU-backend V1 terminal review. It is not a CPU retry, a
Vulkan retry, an integrity replacement, or a resume. This document and its
plan builder grant no qualification, scientific execution, data generation,
GPU, navigation, planner, promotion, deployment, partial-reuse, refill,
overwrite, or retry authority.

## Why this successor is eligible

The CPU qualification terminal review is bound at SHA-256
`9b0c31c05b4fb6064c67116a456d34a6f7e49cfe85ec55ed081599acb18502f0`
and 20,536 bytes. It records
`FAIL_CPU_BACKEND_QUALIFICATION_HARD_STOP`, consumes the one CPU qualification
attempt, and permits only a separately preregistered Genesis 0.4.6 ROCm/HIP
qualification direction.

The CPU result is not evidence against CPU physics or rendering. Worker 12
failed before Genesis import, initialization, scene construction, physics, or
rendering because nested mutable validator overlays redirected validation of
the frozen Vulkan witness through the CPU-specific validator. This successor
does not patch that attempt. Its plan validation is local and immutable: it
does not replace or depend on mutable `pilot.validate_plan` or
`pilot.EXECUTION_ENVIRONMENT` globals.

Original Vulkan V1 and Vulkan integrity replacements V1-V3 remain consumed.
CPU backend V1 remains consumed. No reservation, scene, receipt, RGB, depth,
mesh, checkpoint, tensor, metric, process result, or partial output from any
predecessor may be opened, copied, screened, joined, resumed, refilled, or
admitted here.

## Frozen science and exact material change

The original 64-scene Vulkan V1 exact plan remains the sole scientific-plan
input. Its exact binding is SHA-256
`c34aa23303951d32dd9686a607de7b78df06db026918d868017a6a93c506a040`
and 346,027 bytes. The unexecuted CPU scientific plan is also bound as an
identity witness at SHA-256
`258d6bf004fa3618d492b583c56ea7fbc15b127ade36299fcba11295b147745e`
and 346,045 bytes; the builder proves it is exactly the four-field CPU overlay
of Vulkan V1 before constructing either ROCm plan.

The prospective scientific plan changes only the fields needed for the new
attempt and runtime:

1. `attempt_id` and `output_root` use a fresh Genesis ROCm namespace.
2. `execution_contract.backend` is `amdgpu`, denoting the explicit Genesis API
   symbol `gs.amdgpu` rather than the ambiguous generic `gs.gpu` route.
3. `python_invocation_path` is exactly
   `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python`.
4. `execution_contract.environment` is the exact ROCm/EGL environment below.
5. `execution_contract.graphics_preflight` identifies one R9700/gfx1201 HIP
   device and the matching EGL/DRM renderer without retaining Vulkan fields.
6. `runtime_bindings` replace the Genesis 0.3.14/Vulkan interpreter closure
   with the exact Genesis 0.4.6, Quadrants, ROCm compiler, shared ROCm Torch,
   policy-loader, and EGL closure.
7. `successor_contract` binds the Vulkan plan, CPU plan, CPU terminal review,
   backend versions, non-authority status, and (only for qualification) the
   ordered two-scene probe contract.

All scientific content remains frozen: the exact scene panel and train/eval
roles, four states per scene, two history blocks, nine actions, horizons,
3,072 stored RGB frames, transient depth checks, action-conditioned physical
successors, DINOv2 binding, visual and zero-vision arms, model seeds
411/412/413, sampler seed 414, 800 updates, train-only fitting, live
task/action baseline, durable checkpoint boundary, two evaluations, metrics,
bootstrap, decision gates, 16,977,405,952-byte R9700 cap, and 7,200-second
scientific wall cap. No cross-backend bit-equivalence claim is made.

## Exact runtime contract

The execution environment is exactly:

```text
EGL_DEVICE_ID=1
GS_BACKEND=amdgpu
GS_CACHE_FILE_PATH=<exact role-local attempt root>/quadrants_cache
GS_ENABLE_FASTCACHE=1
GS_ENABLE_NDARRAY=1
GS_ENABLE_ZEROCOPY=1
GS_PARA_LEVEL=0
HIP_VISIBLE_DEVICES=0
PATH=/opt/rocm-7.1.1/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PYOPENGL_PLATFORM=egl
PYTHONDONTWRITEBYTECODE=1
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
PYTHONSAFEPATH=1
ROCM_PATH=/opt/rocm-7.1.1
ROCR_VISIBLE_DEVICES=0
```

The cache path is not a free placeholder in either plan. It is exactly
`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1/attempt_v1/quadrants_cache`
for science and
`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_qualification/attempt_v1/quadrants_cache`
for qualification. Both begin fresh. Qualification may share only its own
attempt-local compiled cache between its two fresh worker processes; no cache
crosses into the scientific attempt.

`HSA_OVERRIDE_GFX_VERSION` must be absent. The R9700 is natively `gfx1201`;
the historical `11.0.0` override for the laptop gfx1103 iGPU would silently
compile for the wrong target. `LD_LIBRARY_PATH` is deliberately not set: the
shared Torch 2.12.0+rocm7.2 wheel must retain its own ROCm 7.2 libraries.
Only `ROCM_PATH=/opt/rocm-7.1.1` and ROCm 7.1.1 LLVM `bin` first on `PATH` are
used to select the known-working Quadrants linker. The system LLD route that
previously reported `unknown abi version` is ineligible. Both `ld.lld` found
through the exact `PATH` and `${ROCM_PATH}/llvm/bin/ld.lld` must resolve to the
bound regular target `/opt/rocm-7.1.1/lib/llvm/bin/lld`.

Preflight must establish exactly one HIP-visible device at index 0 named
`AMD Radeon AI PRO R9700`, architecture `gfx1201`; Genesis must expose and use
`gs.amdgpu`. DRM vendor/device IDs must be `0x1002:0x7551`. EGL device 1 must
report the same R9700, with `PYOPENGL_PLATFORM=egl`. Physics selection and
OpenGL rendering are separate and both must pass.

The runtime closure binds the exact Go2 URDF, checkpoint/config, Python target
and its seven referenced DAE meshes, both venv `pyvenv.cfg` files, Genesis
0.4.6 metadata and installation `RECORD`, the seven runtime-critical Genesis
initialization/constant/misc/scene/camera/rasterizer/EGL source files,
Quadrants 0.6.2 metadata and installation `RECORD`, its four small
initialization/kernel/runtime-helper sources, native `quadrants_python` core,
AMDGPU runtime bitcode, and the exact focused ROCm 7.0
OpenCL/OCKL/OCML/gfx1201/ABI/math-mode bitcode closure.
It also binds `/opt/rocm-7.1.1/bin/rocminfo`, the regular ROCm 7.1.1 LLD
target, EGL probe, and the `.pth` file importing the separately bound
`world_model_rocm_7_2_1_v1` site-packages. Finally, it binds metadata and
installation `RECORD` files for Torch 2.12.0+rocm7.2, torchvision
0.27.0+rocm7.2, tensordict 0.13.0, rsl-rl-lib 5.4.1, NumPy 2.4.6, and Pillow
11.3.0, plus the exact `rsl_rl/runners/on_policy_runner.py` source. The
one-time preflight must also resolve the imported Genesis module to the bound
Genesis venv and the imported Torch, NumPy, and Pillow modules beneath the
bound world-model site-packages; `torch.version.hip` must identify a 7.2
build. This is a focused runtime identity closure, not a claim that every
transitive Python file or large Torch DSO is rehashed by every worker.

Environment preparation reported a clean `pip check`. A no-Genesis-init,
no-GPU readiness check loaded the frozen checkpoint and config through
`GenesisGo2PPOPolicy(device="cpu")` with the expected
45→512→256→128→12 network and 12-joint action output. These are unbound
readiness observations only, not backend qualification or execution evidence.
The transient 5.4.2 package candidate was removed; the final plan binds the
frozen-policy rsl-rl-lib 5.4.1 version. No policy implementation change is
part of this successor.

## Mandatory two-scene qualification first

The qualification plan has its own attempt identity and fresh root:

`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_qualification/attempt_v1`

The prospective scientific root must remain absent throughout qualification.
Independent source review and a separate one-shot qualification authority are
mandatory before either probe. Plan creation, environment installation, a
PASS source review, or this preregistration cannot authorize a workload.

Exactly two fresh full-scene workers run in this fixed order:

1. frozen scene index 12, the repeated Vulkan failure scene;
2. frozen scene index 0, a distinct reference scene.

Each worker executes all four frozen states, both full history blocks, all
nine action successors per state, and the full horizon: 36 branches, 12
context frames, 36 target frames, exactly 48 stored RGB renders, and 48
transient depth-validation renders. This is intentionally the smallest probe
that exercises the actual state-clone/control path and the exact textured V03
static renderer under Genesis 0.4.6; a primitive sphere smoke is insufficient.

Each worker is a fresh process group under a 300-second hard watchdog. The
300 seconds contain a possible cold Quadrants JIT; they are not the scientific
feasibility threshold. Qualification timing passes only when
`64 * max(worker_elapsed_seconds) + 900 <= 7200`, equivalent to at most
98.4375 seconds for the slower observed worker. The 900-second reserve covers
preflight, joins, DINO, training, checkpointing, and both evaluations.

Both workers must exit zero and satisfy all existing plan/receipt/input,
physics, state-clone, requested-versus-executed action, RGB count and quality,
transient depth, nonfinite, mesh-cache, output-binding, selected-device VRAM,
process-release, and freshness checks. In addition, qualification fails on:

- any backend other than exact `gs.amdgpu` on the single R9700/gfx1201;
- any `HSA_OVERRIDE_GFX_VERSION`, changed visibility selector, fallback, or
  use of the system linker;
- failure of the historical V03 scene builder or EGL renderer under Genesis
  0.4.6;
- any AMDGPU ring timeout, reset, device wedge, HSA exception, or leaked
  worker/process group;
- any call to `robot.get_links_net_contact_force()` on this backend. Prior
  local evidence found this exact Genesis 0.4.6/R9700 call reliably faults;
  source review must prove the qualified counterfactual call graph does not
  reach it rather than masking a runtime failure.

Probe data are non-scientific and permanently ineligible for the 64-scene
collection. A PASS qualification can only permit a later independent builder
to consider issuing scientific authority after rebinding the exact PASS
result. It makes no representation, world-model, planning, navigation, or
generalization claim.

## Scientific decision and hard stop

Only after exact qualification PASS and later scientific authority may the
full 64-scene prospective run start from the fresh root
`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1/attempt_v1`.
It must start from scene 0 with no qualification or predecessor payload reuse.

The unchanged scientific gates are: integrity/custody pass; visual normalized
rank regret at most 0.13; visual minus live task/action regret at most -0.02
with paired 95% upper bound below zero; visual minus no-vision regret at most
-0.01 with paired upper bound below zero; and visual beats uniform random
expectation. Passing supports only the preregistered incremental visual
hypothesis, not planning utility.

Any qualification infrastructure or backend failure consumes and hard-stops
this Genesis ROCm backend V1 qualification. There is no retry, resume, refill,
partial reuse, integrity replacement, or second V1 attempt. Any successor
would require a new material hypothesis, fresh preregistration, independent
review, and new authority.
