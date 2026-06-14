# AMD render backend: why the render venv pins genesis-world 0.3.14

_Last updated 2026-05-21._

## TL;DR

The mass-datagen **render** phase runs in a *separate* venv pinned to
**`genesis-world==0.3.14`** because that is the newest release that still
exposes the **`gs.vulkan`** backend. The physics **rollout** phase stays on
`genesis-world==0.4.6` (`gs_madrona`/ROCm) — the rollout corpus is already
generated and unaffected. This split is a workaround, not a permanent state;
see "Path back" below.

## The problem

This box has a discrete **AMD Radeon AI PRO R9700 (34 GB)** + an AMD iGPU, with
working **Vulkan 1.4 / RADV** (`/usr/share/vulkan/icd.d/radeon_icd.json`). It is
fully capable of fast GPU rendering. But on `genesis-world==0.4.6`:

- **`gs.vulkan` was removed.** In the 0.3.x → 0.4.0 refactor the backend enum
  slot 3 changed from `vulkan` to `amdgpu` (ROCm/HIP). 0.4.x enum is
  `cpu/gpu/cuda/amdgpu/metal`; 0.3.14 is `cpu/gpu/cuda/vulkan/metal`.
- **Madrona `BatchRenderer` (the fast GPU rasterizer) requires CUDA.**
  `genesis/vis/batch_renderer.py` raises `"BatchRenderer requires CUDA backend"`
  and `gs_madrona/renderer_gs.py` pulls in `nvidia_cuda_nvrtc_cu12`. NVIDIA only,
  even though `gs_madrona` ships Vulkan source under `src/render/vk/`.
- **The remaining non-CUDA renderer is the OpenGL `Rasterizer` (pyrender)**,
  which on this box falls back to **llvmpipe (CPU software)** — `DISPLAY` unset,
  default EGL device is llvmpipe. The `amdgpu`/ROCm path is also slow for
  rendering (v2 measured ~38 s/env).

### Measured impact (2026-05-21, genesis 0.4.6, OpenGL Rasterizer)

- Software/llvmpipe: **~5–8 frames/s** single worker.
- `EGL_DEVICE_ID=0` engaged the **iGPU** (GPU[1]) to 100% but ran at the *same*
  ~27 s/100f as software → the path is **CPU-per-frame-bound** (GPU→CPU readback
  + 224 resize + depth-validate + write, ×48 envs), not rasterization-bound.
  The discrete R9700 was never touched.
- Depth-validate is ~75% of per-frame cost (40 s→10 s/100f with `--no-depth`).
- Concurrent GL contexts crash beyond ~3–4 workers (matches v2's
  `VULKAN_SAFE_WORKER_LIMIT=4`).
- Full render (1450 scenes × 48 envs × ~1000 frames = **69.6 M frames**) on this
  path: **~20–145 days**. Not viable.

### Reference: what v2 achieved with `gs.vulkan`

LeWMQuad-v2 rendered on the same class of AMD GPU via `--sim_backend vulkan`:
~16 s/env, 4 Vulkan workers, **32k envs in ~35 h**. Extrapolated to v3's
~69.6k env-renders ≈ **~3 days**. That is the "<5 day" target.

## The decision

Pin the **render venv** to `genesis-world==0.3.14` (`gs.vulkan`), render on the
R9700 with `WORKERS≤4` (Vulkan worker limit). Keep the **rollout venv** on
0.4.6. Validate with `rocm-smi` that GPU[0] (R9700) loads during rendering.

Trade-offs accepted: 0.3.14 is older (fewer fixes; possible API drift in the
render scripts vs 0.4.6 — handle in the render wrappers only), and we maintain
two genesis versions until a fast AMD path returns to 0.4.x.

## Path back (re-add a fast AMD path; upgrade the render venv to 0.4.x+)

Re-converge on a single 0.4.x+ venv once **any** of these lands:

1. **`gs_madrona` ships a Vulkan/ROCm backend.** The Vulkan renderer source
   already exists (`gs_madrona/src/render/vk/`); today the Genesis adapter gates
   it behind CUDA + `nvidia_cuda_nvrtc`. When Madrona exposes Vulkan/HIP, drop
   the `gs.backend != gs.cuda` guard path and use `BatchRenderer` on AMD.
2. **Genesis re-adds a Vulkan rasterizer backend** to the 0.4.x enum (or a
   first-class headless EGL-on-discrete-GPU path for the OpenGL Rasterizer).
3. A different **AMD-capable batch renderer** is integrated.

Tracking: watch `gs_madrona` releases for AMD/Vulkan, and Genesis backend-enum
changes (`genesis/constants.py: class backend`). When upgrading, re-run the
`rocm-smi` GPU-load + throughput validation before regenerating renders.

## Validated config (2026-05-21, genesis-world 0.3.14 render venv)

The render is **OpenGL** (`genesis.ext.pyrender` + `JITRenderer`, numba/OpenGL),
*not* Vulkan — `gs.vulkan` only moves the **compute** (gstaichi) onto the R9700.
The render device is chosen by **EGL**, and the R9700 *is* an EGL device:

```
EGL devices (eglinfo): #0 iGPU (radeonsi raphael), #1 AMD R9700 (radeonsi gfx1201), #2 llvmpipe
```

**Target the R9700 for rendering with `EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl`.**
(`MESA_VK_DEVICE_SELECT` / `DRI_PRIME` did *not* move the OpenGL render — they
affect Vulkan / GLX, not pyrender's EGL device pick.)

Single-env (correct per-env cameras) throughput on a maze-class scene:

| workers (R9700) | agg env-fps | note |
| --- | --- | --- |
| 1 | 308 | overhead-bound, GPU mostly idle |
| 4 | 1073 | near-linear |
| **8** | **1339** | **knee — recommended** |
| 16 | 677 | oversubscribed, throughput *drops* |
| 16 R9700 + 16 iGPU | 1492 | marginal, 5/32 workers died |

Each worker is ~1 CPU core (per-frame `set_pose`/readback/encode); ~8 workers
balance CPU-feed against R9700 fill. **Batched `env_separate_rigid` is unusable**
on 0.3.14's Rasterizer ("same camera transform for all envs") — must render
**single-env per env** for correct egocentric cameras.

**Recommended render config:** `EGL_DEVICE_ID=1`, single-env mode, ~8 workers →
**~14–15 h for the full 69.6 M-frame render**. (Try `OMP_NUM_THREADS=1` per
worker to push past 8.)

## Notes

- The discrete GPU is `rocm-smi` GPU[0] (34 GB, R9700); the iGPU is GPU[1].
- EGL device enumeration on this box: device 0 = iGPU (radeonsi), 1/2 = llvmpipe
  (software). The OpenGL path never reached the discrete R9700.
- Render is read-only over the completed rollout output + corpus, so swapping
  the render genesis version cannot corrupt the physics corpus.
