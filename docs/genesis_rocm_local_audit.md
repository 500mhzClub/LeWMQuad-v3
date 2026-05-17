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

### Production R9700 result, 2026-05-16

The production host exposes the Radeon AI Pro R9700 as ROCm device 0:

```text
torch 2.9.1+rocm6.4
torch.version.hip 6.4.43484-123eb5128
cuda.is_available True
device_count 2
device_name AMD Radeon AI PRO R9700
```

Important environment detail: this host installs ROCm under the versioned
prefix `/opt/rocm-7.1.1`, not `/opt/rocm`. Genesis's AMDGPU JIT failed with
Ubuntu `lld-18`:

```text
ld.lld: error: unknown abi version
```

Putting `/opt/rocm-7.1.1/lib/llvm/bin` first on `PATH` and exporting
`ROCM_PATH=/opt/rocm-7.1.1` fixed the JIT linker path. The training and trainer
smoke wrappers now discover that prefix automatically. The laptop-only
`HSA_OVERRIDE_GFX_VERSION=11.0.0` override must remain unset on this GPU.

Patched trainer smoke passed:

```text
HSA_OVERRIDE_GFX_VERSION unset (correct for RDNA4 / gfx1201)
ROCM_PATH=/opt/rocm-7.1.1
ld.lld=/opt/rocm-7.1.1/lib/llvm/bin/ld.lld
Run name: codex-smoke
Total steps: 96
Steps per second: 34
Mean reward: -0.07
```

Full PPO run completed with upstream defaults:

```text
exp_name: lewm-go2-20260516T160610Z
num_envs: 4096
max_iterations: 101
seed: 1
Total steps: 9,928,704
Steps per second: ~113k after warm-up
Final mean reward: 15.14
Final mean episode length: 1001.00
Checkpoint: .generated/upstream_genesis/locomotion/logs/lewm-go2-20260516T160610Z/model_100.pt
```

The final checkpoint also passed a non-interactive load smoke: 16 envs, 20
policy-controlled steps, no resets, and finite rewards.

The same checkpoint does **not** pass the LeWM locomotion command contract. A
contract probe now lives at:

```bash
scripts/check_genesis_go2_policy_contract.sh \
  --exp-name lewm-go2-20260516T160610Z \
  --ckpt 100
```

Observed result:

```text
policy_contract_probe_fail
hold: requested [0.0, 0.0, 0.0], measured vx=0.528, yaw=0.091
backward: requested vx=-0.200, measured vx=0.475
yaw_left: requested yaw=0.450, measured yaw=0.091, measured vx=0.542
yaw_right: requested yaw=-0.450, measured yaw=0.088
arc_left: requested [0.200, 0.0, 0.450], measured yaw=0.066
arc_right: requested yaw=-0.450, measured yaw=0.062
```

Root cause: the upstream Genesis training recipe is command-conditioned in
shape but not in data distribution. The policy observation includes the
3-D command vector, but `go2_train.py` trained only on a fixed command:
`lin_vel_x_range=[0.5, 0.5]`, `lin_vel_y_range=[0, 0]`,
`ang_vel_range=[0, 0]`. The checkpoint is therefore a forward-walk policy, not
a compliant LeWM `[vx_body_mps, vy_body_mps, yaw_rate_radps]` primitive policy.

### LeWM-contract PPO result, 2026-05-16

The follow-up training run used a repo wrapper that samples exactly the
trainable velocity primitives from `config/go2_primitive_registry.yaml`:

```bash
scripts/train_genesis_go2_locomotion_contract.sh
```

Training configuration:

```text
exp_name: lewm-go2-contract-20260516T163413Z
num_envs: 4096
max_iterations: 501
seed: 11
command_bank: hold, forward_slow, forward_medium, forward_fast, backward,
              yaw_left, yaw_right, arc_left, arc_right
tracking_ang_vel reward scale: 0.8
Total steps: 49,250,304
Final mean reward: 33.33
Final mean episode length: 1001.00
Checkpoint: .generated/upstream_genesis/locomotion/logs/lewm-go2-contract-20260516T163413Z/model_500.pt
SHA256: e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff
```

The velocity-primitive contract probe passed:

```bash
scripts/check_genesis_go2_policy_contract.sh \
  --exp-name lewm-go2-contract-20260516T163413Z \
  --ckpt 500
```

Observed measured body-frame response after a 2 s warmup in 5 s rollouts:

```text
hold          request [ 0.00, 0.00,  0.00] -> measured [ 0.001, 0.001, -0.001]
forward_slow  request [ 0.20, 0.00,  0.00] -> measured [ 0.180, 0.010, -0.008]
forward_medium request [0.25, 0.00,  0.00] -> measured [ 0.252, 0.013, -0.007]
forward_fast  request [ 0.30, 0.00,  0.00] -> measured [ 0.303, 0.000, -0.012]
backward      request [-0.20, 0.00,  0.00] -> measured [-0.191, 0.017,  0.012]
yaw_left      request [ 0.00, 0.00,  0.45] -> measured [-0.000,-0.009,  0.452]
yaw_right     request [ 0.00, 0.00, -0.45] -> measured [ 0.011, 0.014, -0.458]
arc_left      request [ 0.20, 0.00,  0.45] -> measured [ 0.197,-0.005,  0.450]
arc_right     request [ 0.20, 0.00, -0.45] -> measured [ 0.199, 0.017, -0.455]
```

All tested velocity primitives had zero resets and passed direction/magnitude
criteria. This validates the blind PPO policy against the LeWM velocity command
contract for the trainable primitive set. Nonzero lateral `vy_body_mps` remains
disabled in the primitive registry and platform safety limits until a lateral
policy pass is trained and validated. This does not validate mode-event
primitives such as recovery stand.

### Bulk-rollout adapter status, 2026-05-16

The contract checkpoint is now wired into `lewm_genesis.rollout.GenesisGo2PPOPolicy`.
The adapter loads `locomotion.policy_artifact` from `config/go2_platform_manifest.yaml`,
reconstructs the upstream 45-D policy observation, applies the trained action
latency semantics, and returns absolute 12-DOF joint position targets in the
rollout's leg-joint order.

The frozen artifact now lives outside `.generated/`:

```text
models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt
models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl
```

Integration details locked by the smoke path:

- Genesis bulk rollout resolves the same Genesis-bundled Go2 URDF used by the
  training environment via `robot.genesis_urdf: genesis_builtin_go2`.
- The adapter validates the robot's policy joint names and entity-local DOF
  order before a rollout starts.
- `RolloutRunner` resolves concrete leg DOF indices from built robot joint
  names, so static scene entities cannot silently shift the policy mapping.
- Static scene boxes are inserted as fixed Genesis entities.
- Genesis scene seeds are clamped to the signed 32-bit range accepted by the
  Quadrants compile config.

A one-env CPU smoke on the generated acceptance corpus built a scene, loaded
the PPO adapter from the manifest, stepped hold/forward/yaw command ticks, and
completed with finite base and joint state. This smoke checks integration
plumbing only; the formal behavior check remains the contract probe above.

### Bulk-rollout CLI status, 2026-05-16

The first scriptable Genesis bulk path is:

```bash
scripts/genesis_bulk_rollout.sh \
  --scene-corpus .generated/scene_corpus/acceptance \
  --split train \
  --family open_obstacle_field \
  --scene-limit 1 \
  --n-envs 1 \
  --n-blocks 1 \
  --backend cpu \
  --out .generated/genesis_bulk_rollouts/ppo_smoke_rgb
```

The RGB CPU smoke wrote one MCAP raw-rollout directory with the expected
per-env topics:

```text
/clock                                      5
/env_00/camera_info                        1
/env_00/imu/data                           5
/env_00/joint_states                       5
/env_00/lewm/episode_info                  5
/env_00/lewm/go2/base_state                5
/env_00/lewm/go2/command_block             1
/env_00/lewm/go2/executed_command_block    1
/env_00/lewm/go2/foot_contacts             5
/env_00/lewm/go2/mode                      5
/env_00/lewm/go2/reset_event               1
/env_00/odom                               5
/env_00/rgb_image                          1
```

`scripts/convert_smoke_bag_to_raw_rollout.py` now recognizes per-env Genesis
topics such as `/env_00/lewm/go2/command_block`, keeps `env_index` in each
JSONL record, and audits them against the canonical command contract topic.
The smoke conversion passed:

```bash
scripts/convert_smoke_bag_to_raw_rollout.sh \
  .generated/genesis_bulk_rollouts/ppo_smoke_rgb_rawpilot_ready/open_obstacle_field_36c57d3baa8d \
  --out .generated/genesis_bulk_rollouts/ppo_smoke_rgb_rawpilot_ready_raw \
  --quality-profile raw_pilot
```

Result:

```text
contract_audit: pass=True command_blocks=1 executed=1 resets=1
data_quality_audit: profile=raw_pilot pass=True issues=0
```

AMDGPU bulk smoke status on the Radeon AI Pro R9700:

```bash
scripts/genesis_bulk_rollout.sh \
  --scene-corpus .generated/scene_corpus/acceptance \
  --split train \
  --family open_obstacle_field \
  --scene-limit 1 \
  --n-envs 1 \
  --n-blocks 1 \
  --backend amdgpu \
  --out .generated/genesis_bulk_rollouts/ppo_smoke_amdgpu_rgb_default_contacts
```

This now passes with RGB enabled and converts under `--quality-profile
raw_pilot` with `contract_audit: pass=True` and `data_quality_audit:
pass=True`. The CLI defaults `foot_contact_source=zero` on `amdgpu`; summaries
record that fact. The reason is a Genesis/R9700 backend fault in
`get_links_net_contact_force()`, which reliably triggers:

```text
HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
```

The same AMDGPU scene, policy stepping, per-tick ROS message construction, and
MCAP writing pass when contact-force reads are skipped. CPU remains on
Genesis-derived contact-force labels by default.

Inline RGB now calls `Camera.set_pose()` from the current base pose plus the
manifest camera mount before each render. One-block CPU and AMDGPU RGB smokes
passed after this change and converted under `--quality-profile raw_pilot`.

The split-phase path for physics-first, render-later production is also
scriptable. A CPU physics-only smoke:

```bash
scripts/genesis_bulk_rollout.sh \
  --scene-corpus .generated/scene_corpus/acceptance \
  --split train \
  --family open_obstacle_field \
  --scene-limit 1 \
  --n-envs 1 \
  --n-blocks 2 \
  --backend cpu \
  --no-rgb \
  --out .generated/genesis_bulk_rollouts/physics_only_cpu_1scene_2block
```

converted with:

```text
contract_audit: pass=True command_blocks=2 executed=2 resets=1
data_quality_audit: profile=raw_pilot pass=True issues=0
```

and planned render replay with:

```bash
scripts/plan_bulk_render_replay.sh \
  --raw-root .generated/genesis_bulk_rollouts/physics_only_cpu_1scene_2block_raw \
  --out-root .generated/rendered_vision_plans/physics_only_cpu_1scene_2block_v2 \
  --camera-hz 10
```

The generated `frames.jsonl` includes base pose, joint state, episode metadata,
camera mount, and computed world-frame camera pose for each frame.

The replay renderer now consumes those plans:

```bash
scripts/render_replay_genesis.sh \
  .generated/rendered_vision_plans/physics_only_cpu_1scene_2block_v2/000000_physics_only_cpu_1scene_2block_raw/render_replay_plan.json \
  --backend amdgpu \
  --max-frames 2 \
  --out .generated/rendered_vision/physics_only_cpu_1scene_2block_amdgpu_smoke \
  --overwrite
```

CPU replay smoke rendered 3 frames with RGB PNG plus depth `.npy` outputs and
`invalid_frame_count=0`. AMDGPU replay smoke rendered 2 frames with the same
validity result. This confirms the physics-first, render-later path is viable
for mass rendering without requiring GPU physics in the same job.

The split-phase path also handles batched source environments. A 16-env,
1-scene, 2-block CPU physics rollout converted with the env-aware reset audit:

```text
contract_audit: pass=True command_blocks=32 executed=32 resets=16
data_quality_audit: profile=raw_pilot pass=True issues=0
```

The render planner emitted 160 frame jobs from those 16 source envs, and the
AMDGPU replay renderer completed all 160 RGB/depth frames with
`invalid_frame_count=0` using `--replay-env-mode single`. This means physics
can remain highly batched while rendering replays one source env at a time,
which avoids coupling render camera binding to the physics batch size.

Current CPU physics scale probes on the production host:

```text
n_envs  writer  blocks  result
16      mcap    2       converted, raw_pilot pass, 160-frame AMDGPU replay pass
128     mcap    1       converted, raw_pilot pass
512     mcap    1       converted, raw_pilot pass, 16-frame AMDGPU replay smoke pass, 0.5 s sim in 3.3 s wall
1024    none    1       pass, 0.5 s sim in 9.1 s wall
2048    none    1       pass, 0.5 s sim in 18.1 s wall
```

Genesis reported the 2048-env CPU probe at roughly 30k-39k aggregate sim FPS
after build. This reaches the old Vulkan-style 2048-scene concurrency shape for
physics. Writer-enabled production should ramp through longer 512-env shards
before treating 1024 or 2048 as the default batch size.

A two-process AMDGPU render smoke also passed for separate scene plans at the
same time: 10 total RGB/depth frames, `invalid_frame_count=0` in both jobs.
Render concurrency should start at 2 and be tuned upward only after longer
jobs confirm stable VRAM and driver behavior.

A slightly broader AMDGPU pilot also passed:

```bash
scripts/genesis_bulk_rollout.sh \
  --scene-corpus .generated/scene_corpus/acceptance \
  --split train \
  --scene-limit 2 \
  --n-envs 1 \
  --n-blocks 2 \
  --backend amdgpu \
  --log-progress-every-blocks 1 \
  --out .generated/genesis_bulk_rollouts/ppo_pilot_amdgpu_rgb_2scene_2block
```

Scenes:

```text
large_enclosed_maze_6edb8f490970
loop_alias_stress_d11d400f506e
```

Both converted with `--quality-profile raw_pilot`:

```text
contract_audit: pass=True command_blocks=2 executed=2 resets=1
data_quality_audit: profile=raw_pilot pass=True issues=0
```

## Caveats

- The laptop iGPU has only 2 GB VRAM; it proves backend viability, not training
  throughput.
- Production GPU training has now passed on the Radeon AI Pro R9700, and the
  contract-trained checkpoint passed velocity-primitive command validation.
  The checkpoint is wired into the Genesis bulk `PolicyInterface`; mode-event
  recovery remains a separate validation item.
- R9700 AMDGPU bulk rollouts currently emit zero-valued foot-contact labels by
  default. This preserves the topic contract for pilot data while avoiding the
  Genesis contact-force hardware exception above.
- `torch.cuda.*` is still the PyTorch API surface for ROCm/HIP builds; this is
  expected.
