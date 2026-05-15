# Decision: Pivot to Genesis for Bulk Data Generation

Date: 2026-05-15
Status: Accepted
Supersedes: `docs/go2_genesis_gazebo_ros2_bringup_plan.md` as the authority on
backend roles. The bringup plan's LeWM interface specs (command blocks, reset
semantics, episode bookkeeping, scene manifest, dataset gates) still apply.

## Decision

1. **Genesis becomes the authoritative bulk-generation backend.** GPU physics +
   GPU rendering, parallel envs scaled to the target hardware.
2. **Gazebo is retained as a small audit oracle.** Not on the bulk path. Used
   for sensor-fidelity and dynamics-parity spot checks against the Genesis
   corpus.
3. **Lidar is dropped from the v3 corpus contract.** It was already marked
   optional in the bringup plan and absent from the data spec; the pivot makes
   that absence explicit.
4. **Locomotion uses Genesis's built-in Go2 quadruped example as the starting
   policy (Tier A).** Tier B/C (porting an external Go2 RL checkpoint, or
   training one from scratch) are escalation paths if the Tier A gait
   distribution proves insufficient.

## Why

### Throughput math forces the pivot

The data spec target is on the order of 154M raw ticks before validation and
test corpora. Single-instance Gazebo on a 9950X3D-class desktop is not a
credible generation path against that target. Multi-instance Gazebo gets a
constant factor; it does not close the gap. GPU-parallel Genesis is an
order-of-magnitude shift in the right direction.

### v2 evidence: ROCm-accelerated Genesis already works at scale

LeWMQuad-v2 generated approximately 1 TB of simulation data in days using
Genesis on AMD hardware. The backend resolver in
`../LeWMQuad-v2/lewm/genesis_utils.py` already handles `gs.amdgpu` / ROCm /
HIP detection; the rollout loop in
`../LeWMQuad-v2/scripts/1_physics_rollout.py` ran at `--n_envs 2048` in
parallel envs. ROCm-Genesis viability on the production GPU (Radeon AI Pro 9700
32 GB) is not a research risk.

### Locomotion replacement is not a research risk either

v2 replaced CHAMP-equivalent kinematic control with a frozen PPO actor-critic
driving `robot.control_dofs_position`. That pattern transfers to Go2. Genesis
ships a Go2 quadruped locomotion example that walks out of the box, and the
broader ecosystem (`unitree_rl_gym`, legged_gym, IsaacLab) provides Go2
checkpoints that can be ported with retuning rather than from-scratch training.

### The split-pipeline detour was a partial answer to the wrong question

The Gazebo-as-oracle + GPU-render-replay design assumed Gazebo was the
authoritative dynamics path and only rendering needed acceleration. That holds
only if Gazebo dynamics throughput is acceptable. The throughput math says it
is not at the spec scale, which collapses the rationale for keeping Gazebo on
the critical generation path.

## What Changes

### Backend roles (replaces bringup plan section 3.2)

1. `scene_manifest -> Genesis scene -> raw_rollout + rendered_vision`:
   authoritative for bulk generation. State, commands, resets, contacts, and
   RGB/depth all originate from Genesis.
2. `scene_manifest -> Gazebo SDF`: audit oracle only. Used for sensor-fidelity
   spot checks (small N), dynamics-parity verification samples, and as a
   fallback debug path when Genesis output is suspect.

The scene manifest remains the source of truth. Both backends consume the
same manifest.

### Lidar removed from corpus

- `/velodyne_points/points` and `/unitree_lidar/points` are not part of the
  v3 training corpus.
- The bringup plan already marked them optional. The audit doc topic list
  retains them only as upstream-available references, not as expected outputs.
- If a future target claim adds lidar back, the implementation path is
  Genesis-side ray casting against the scene mesh (~1 week to wrap and validate
  noise model). Do not rely on the Gazebo lidar path for scale.

### Locomotion policy

- **Tier A (default):** Use Genesis's built-in Go2 quadruped example as the
  locomotion policy for bulk rollout. This is the starting state.
- **Tier B (if Tier A's gait distribution is too narrow or wrong-shaped):**
  Port an existing open Go2 policy. Candidates: `unitree_rl_gym`,
  `legged_gym`, IsaacLab Go2. Effort: ~1 week to retarget obs/action and
  retune.
- **Tier C (if A and B both fail):** Train a Go2 locomotion policy in Genesis.
  Several weeks. Avoid unless forced.

### ROS 2 stack disposition

- LeWM interface nodes ([lewm_go2_control/nodes/](../lewm_go2_control/nodes/) —
  `command_block_adapter`, `base_state_publisher`, `camera_info_publisher`,
  `reset_manager`, `foot_contacts_publisher`, `mode_manager`,
  `feature_check_runner`) remain valid as contracts. Their **runtime form**
  changes: they become Python utilities the Genesis rollout loop calls
  directly, or they front Genesis through a thin in-process ROS bridge when a
  ROS-bag workflow is needed for audit.
- The CHAMP + ros2_control + ros_gz_bridge stack stays alive only for the
  Gazebo audit oracle path. It is not on the bulk generation path.
- Dataset products move toward v2-style on-disk artifacts (HDF5/Zarr/.npz)
  rather than ROS bags as the primary format for bulk generation. ROS bags
  remain for the audit oracle path.

## What Stays Unchanged

- The LeWM action contract: command blocks
  (`[vx_body_mps, vy_body_mps, yaw_rate_radps]`), `CommandBlock` /
  `ExecutedCommandBlock` semantics, primitive registry, mode events.
- Reset/episode bookkeeping semantics: monotonic `episode_id`, `reset_count`,
  `/lewm/episode_info`, audit invariants (every requested `sequence_id` must
  reappear executed; resets advance by exactly one).
- The data quality gates: `contract_audit`, `topic_audit`,
  `data_quality_audit`, the `smoke` / `pilot` / `training` profile ladder.
- The scene manifest model in [lewm_worlds/](../lewm_worlds/) and the split
  planner.
- The raw_rollout → rendered_vision separation as a *concept*. The
  implementation collapses to a single Genesis pass in practice, but the
  schema separation survives for downstream consumers.

## Considerations and Risks

### Gait distribution validation (Tier A risk)

Whatever policy Genesis's example ships with is optimized for benchmark
reward, not for our data-distribution needs. Before committing to a full
generation run, validate the executed-command histogram against the spec's
expected primitive coverage. If the policy ignores yaw, refuses certain
vx ranges, or produces unstable gaits outside narrow conditions, escalate to
Tier B.

### Safety threshold recalibration

[lewm_go2_control/nodes/command_block_adapter:311-369](../lewm_go2_control/nodes/command_block_adapter#L311-L369)
clips against `min/max_vx_mps`, `max_yaw_rate_radps`, and per-tick deltas
calibrated for CHAMP's response curve. A Genesis-side learned policy will
respond differently. The `clipped` / `safety_overridden` flags in
`ExecutedCommandBlock` will trip at different rates. Recalibrate by sampling
the new policy's actual response distribution before locking thresholds.

### No automated parity oracle on the bulk path

Gazebo is no longer in the inner loop. If Genesis dynamics drift weirdly
(unphysical contacts, slip behavior, joint limit violations), there is no
automatic baseline to flag it. Mitigations:

- Keep Gazebo runnable. The audit set must remain executable so spot checks
  remain cheap.
- Periodically diff a small Genesis batch against a Gazebo batch on the same
  scene manifests and command tapes. Cadence: at minimum, once per
  scene-corpus regeneration; ideally per dataset version.
- Real-robot teleop logs are the eventual ground truth. Where available,
  diff Genesis traces against them in preference to Gazebo.

### ROCm coverage on Radeon AI Pro 9700

v2 proved ROCm+Genesis works on prior AMD hardware. The 9700 is RDNA4-era.
Treat as low risk based on v2 precedent but verify with a single
`1_physics_rollout.py` smoke run before committing weeks of pipeline work.

### Lidar absence is a product decision, not a technical one

If a downstream model author wants lidar later, the answer is "we dropped it
in the pivot; adding it back is a scoped Genesis ray-cast feature, not a
backend switch." Avoid soft-reintroducing it through optional flags that
accumulate.

## Rollback Condition

Reverse this decision if **any** of the following hold after the first
end-to-end Genesis bulk-generation pilot:

1. Tier A through Tier B both fail to produce a gait distribution acceptable
   to the data spec. (Tier C would be a different decision — multi-week
   training schedule.)
2. ROCm-Genesis throughput on the production GPU is materially worse than v2
   demonstrated, and no driver/stack remedy is identified within a week.
3. Genesis-Gazebo parity audits show systematic dynamics drift large enough
   that downstream training collapses.

Rollback path: re-elevate Gazebo to the bulk generation backend, keep the
GPU render-replay split, accept the throughput cost, and renegotiate dataset
scale with the spec.

## Related Documents

- [docs/go2_genesis_gazebo_ros2_bringup_plan.md](go2_genesis_gazebo_ros2_bringup_plan.md)
  — superseded as backend authority; LeWM interface specs still apply.
- [docs/fresh_retrain_data_spec.md](fresh_retrain_data_spec.md)
  — dataset schema authority; the dynamics-oracle line is amended by this doc.
- [docs/upstream_go2_sim_audit.md](upstream_go2_sim_audit.md)
  — Gazebo-stack audit; remains valid for the audit oracle path only.
- [docs/go2_ros2_scaffold_runbook.md](go2_ros2_scaffold_runbook.md)
  — scaffold runbook; the ROS 2 launch instructions inside it now describe
  the audit oracle, not the production stack.
- [../LeWMQuad-v2/scripts/1_physics_rollout.py](../../LeWMQuad-v2/scripts/1_physics_rollout.py)
  — pattern reference for the Genesis bulk rollout loop.
