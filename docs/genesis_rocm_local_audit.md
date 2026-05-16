# Genesis ROCm Local Audit

Date: 2026-05-16
Status: Current local finding

## Summary

The local ROCm path works with current Genesis when the Python environment uses
a ROCm PyTorch wheel and the RDNA3 override is set for the integrated GPU.
Genesis `0.4.6` successfully initialized `gs.amdgpu`, built batched scenes
with `n_envs=4`, and stepped them. The stronger Go2 smoke loads the
Genesis-bundled Go2 URDF, sets all 12 leg DOFs in every env, issues PD
position controls, and steps the simulation.

This is the current-path replacement for the legacy Vulkan backend. Vulkan was
removed from Genesis `0.4.x`; use ROCm/HIP for current Genesis.

## Local Hardware And Runtime

- GPU: AMD Phoenix1 / Radeon 780M integrated GPU.
- ROCm device target: `gfx1103`.
- VRAM reported by `rocm-smi`: 2 GB.
- Python: 3.12.
- Genesis: `genesis-world==0.4.6`.
- Isolated ROCm Torch venv:
  `.generated/venvs/genesis_rocm`.
- Torch installed in that venv:
  `torch==2.9.1+rocm6.4`.

The global Python environment still has CUDA-only Torch and does not support
Genesis `gs.amdgpu` on this host. Use the ROCm venv for GPU smoke tests.

## Required Local Override

Without an override, Torch sees the GPU but fails on the first HIP kernel:

```text
torch.AcceleratorError: HIP error: invalid device function
```

On this machine, the fix is:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

With that override, Torch HIP succeeds:

```text
torch 2.9.1+rocm6.4
torch.version.hip 6.4.43484-123eb5128
cuda.is_available True
device_count 1
device_name AMD Radeon  780M Graphics
hip_tensor_sum 64.0
```

## Genesis Parallel Smokes

The reusable smoke requires `--n-envs >= 2` so a single-env init cannot pass as
a parallel-generation check.

Generic rigid-body command:

```bash
scripts/check_genesis_rocm_smoke.sh
```

Generic equivalent explicit command:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  .generated/venvs/genesis_rocm/bin/python \
  scripts/check_genesis_backend_smoke.py \
  --backend amdgpu \
  --n-envs 4 \
  --steps 5
```

Observed pass:

```text
Running on [AMD Radeon  780M Graphics] with backend gs.amdgpu
genesis_backend_smoke_pass backend=amdgpu n_envs=4 steps=5
```

Go2 command:

```bash
scripts/check_genesis_go2_rocm_smoke.sh
```

Go2 observed pass:

```text
Running on [AMD Radeon  780M Graphics] with backend gs.amdgpu
genesis_go2_backend_smoke_pass backend=amdgpu n_envs=4 steps=10 n_dofs=18 urdf=...
```

The Go2 smoke is the preferred local gate before touching the bulk rollout
loop, because it exercises the robot URDF and batched leg-control path rather
than only primitive rigid bodies.

Upstream locomotion env command:

```bash
scripts/check_genesis_go2_locomotion_env_rocm_smoke.sh
```

This fetches the upstream Genesis files into
`.generated/upstream_genesis/locomotion` when missing:

- `go2_train.py`
- `go2_env.py`
- `go2_eval.py`

Observed pass:

```text
genesis_upstream_go2_env_smoke_pass backend=amdgpu n_envs=4 steps=3 obs_shape=(4, 45) done_count=0
```

This is the pre-training gate for Genesis's Go2 locomotion recipe. It imports
the upstream `Go2Env`, builds four parallel envs, runs zero-action steps, and
checks finite observations/rewards.

Upstream trainer smoke command:

```bash
scripts/check_genesis_go2_locomotion_train_rocm_smoke.sh
```

Observed one-iteration pass:

```text
Run name: codex-smoke
Total steps: 96
Steps per second: 21
Collection time: 4.114s
Learning time: 0.281s
Mean reward: -0.10
```

The trainer smoke invokes upstream `go2_train.py` with `-B 4` and
`--max_iterations 1`. It is not a useful policy; it only verifies that
`rsl-rl-lib`, the Genesis Go2 env, batched rollout collection, and one PPO
update can execute on the local ROCm path.

## Installation Used For The Probe

The reusable form lives in
`scripts/setup_genesis_rocm_training.sh` (idempotent; safe to re-run).

Manual equivalent:

```bash
sudo apt install lld    # Genesis AMDGPU JIT needs ld.lld
python3 -m venv .generated/venvs/genesis_rocm
.generated/venvs/genesis_rocm/bin/python -m pip install --upgrade pip wheel setuptools
.generated/venvs/genesis_rocm/bin/python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/rocm6.4
.generated/venvs/genesis_rocm/bin/python -m pip install genesis-world==0.4.6
.generated/venvs/genesis_rocm/bin/python -m pip install matplotlib
.generated/venvs/genesis_rocm/bin/python -m pip install \
  tensordict tensorboard GitPython onnx onnxscript
.generated/venvs/genesis_rocm/bin/python -m pip install 'coverage>=7.6'
.generated/venvs/genesis_rocm/bin/python -m pip install --no-deps \
  'rsl-rl-lib>=5.0.0'
```

The venv is intentionally **not** `--system-site-packages`. Earlier iterations
used that flag to inherit a system Genesis install, but the setup script now
installs `genesis-world` explicitly, and the system-site path was leaking
incompatible copies of `coverage` (pre-7.6, breaking numba) and `matplotlib`
(built against numpy 1.x, ABI-incompatible with the venv's numpy 2.x).
Isolated venv kills the whole class of bug.

## Production Tier B Training

`scripts/train_genesis_go2_locomotion.sh` is the production wrapper for the
upstream Genesis Go2 PPO recipe. It uses the ROCm venv above and calls
`go2_train.py` with the upstream defaults (`B=4096`, `max_iterations=101`,
`seed=1`). Override via `LEWM_GO2_NUM_ENVS`, `LEWM_GO2_MAX_ITERS`,
`LEWM_GO2_SEED`, `LEWM_GO2_EXP_NAME`.

Important: do **not** export `HSA_OVERRIDE_GFX_VERSION` on the Radeon AI Pro
9700. RDNA4 (`gfx1201`) is natively supported by ROCm; the
`HSA_OVERRIDE_GFX_VERSION=11.0.0` value documented above is a workaround for
the laptop iGPU's `gfx1103` only. Setting it on the production GPU will
silently mistarget the kernels.

Fresh-clone workflow on the production host:

```bash
git clone <repo-url> LeWMQuad-v3
cd LeWMQuad-v3
scripts/setup_genesis_rocm_training.sh                    # one-time
scripts/check_genesis_go2_locomotion_train_rocm_smoke.sh  # 1 iter wiring
scripts/train_genesis_go2_locomotion.sh                   # PPO run
```

Logs and checkpoints land under
`.generated/upstream_genesis/locomotion/logs/<exp_name>/`. Tail with
`tensorboard --logdir .generated/upstream_genesis/locomotion/logs`.

## Caveats

- This laptop iGPU has only 2 GB VRAM; it proves backend viability, not training
  throughput.
- Production GPU smoke must be repeated on the target Radeon AI Pro 9700
  without assuming the laptop override or throughput characteristics transfer.
- `torch.cuda.*` is still the PyTorch API surface for ROCm/HIP builds; this is
  expected.
