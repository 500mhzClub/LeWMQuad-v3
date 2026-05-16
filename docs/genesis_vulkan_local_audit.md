# Genesis Vulkan Local Audit

Date: 2026-05-16
Status: Current local finding

## Summary

The local machine has a working Vulkan stack on the AMD integrated GPU, and
Genesis can run a small parallel smoke through Vulkan when pinned to the older
0.3.x backend stack. The currently installed `genesis-world==0.4.6` does not
expose `gs.vulkan`, so `backend="vulkan"` cannot work with that package.

The v3 backend resolver must not silently fall back to CPU when Vulkan is
explicitly requested. A CPU fallback hides the exact failure that would make the
bulk-generation pipeline infeasible.

## Local GPU/Vulkan Facts

- GPU: AMD Phoenix1 / Radeon 780M integrated GPU.
- Vulkan driver: Mesa RADV PHOENIX.
- Vulkan instance: 1.3.280.
- ROCm reports gfx1103 with 2 GB VRAM, but the global PyTorch install is
  `2.12.0+cu130`, not ROCm. `torch.cuda.is_available()` is false and
  `torch.version.hip` is `None`.

## Genesis Package Findings

Installed package:

```text
genesis-world 0.4.6
gs.cpu=0
gs.gpu=1
gs.cuda=2
gs.amdgpu=3
gs.metal=4
gs.vulkan=None
```

`gs.amdgpu` under 0.4.6 is not usable on this machine with the current PyTorch
install. Genesis rejects it because it expects a Torch CUDA/HIP device, and the
installed Torch build has neither CUDA access nor HIP support.

Wheel inspection:

- `genesis-world==0.2.1` contains `gs.vulkan`.
- `genesis-world==0.3.10` contains `gs.vulkan`.
- `genesis-world==0.3.14` contains `gs.vulkan`.
- `genesis-world==0.4.0` has no `gs.vulkan` backend symbol.
- `genesis-world==0.4.6` has no `gs.vulkan` backend symbol.

## Vulkan Smoke Result

An isolated probe environment using `genesis-world==0.3.14` and
`gstaichi==4.6.0` successfully initialized Genesis with `gs.vulkan` and stepped
a small parallel scene:

```text
Running on [AMD Ryzen 7 7840U w/ Radeon  780M Graphics] with backend gs.vulkan
genesis_backend_smoke_pass backend=vulkan n_envs=4 steps=5
```

Genesis also reported:

```text
Torch GPU backend not available. Falling back to CPU device.
Zero-copy mode not enabled...
```

Interpretation: Taichi/Genesis simulation kernels are using Vulkan, but Torch
tensor interop is CPU-side on this host until a compatible ROCm/HIP Torch stack
is installed. For the Vulkan path, this is acceptable for a backend smoke but
must be benchmarked before declaring production throughput.

## Reproduction Commands

These commands keep the probe isolated under `.generated/`:

```bash
python3 -m venv --system-site-packages .generated/venvs/genesis_0314_vulkan
.generated/venvs/genesis_0314_vulkan/bin/python -m pip install \
  --no-deps genesis-world==0.3.14
.generated/venvs/genesis_0314_vulkan/bin/python -m pip install gstaichi==4.6.0
TI_ENABLE_VULKAN=1 GS_BACKEND=vulkan \
  .generated/venvs/genesis_0314_vulkan/bin/python \
  scripts/check_genesis_backend_smoke.py --backend vulkan --n-envs 4 --steps 5
```

The project code now treats explicit Vulkan as a hard requirement: if the
active Genesis package has no `gs.vulkan`, backend resolution raises instead of
falling back to CPU.

## v2 Reference

`../LeWMQuad-v2` is still the closest implementation reference:

- `../LeWMQuad-v2/lewm/genesis_utils.py` supports `GS_BACKEND` and explicit
  `vulkan`.
- `../LeWMQuad-v2/scripts/1_physics_rollout.py` accepts
  `--sim_backend vulkan` and uses parallel `scene.build(n_envs=...)`.
- `../LeWMQuad-v2/scripts/2_visual_renderer.py` prefers Vulkan on AMD render
  workers and caps unsafe parallel worker startup.
- `../LeWMQuad-v2/report.md` records Vulkan as the preferred AMD render backend
  when ROCm/HIP rendering is too slow.

## Go2 Locomotion Example

The Genesis source repository has Go2 locomotion training scripts:

- `examples/locomotion/go2_train.py`
- `examples/locomotion/go2_env.py`

The pip package installed locally ships the Go2 URDF assets, not these example
scripts and not a pretrained Go2 checkpoint. Treat the upstream Go2 example as
a training recipe unless a compatible checkpoint is separately supplied.
