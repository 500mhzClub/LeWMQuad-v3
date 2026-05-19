# Mass data-gen runbook

End-to-end recipe for going from a clean repo to a labelled rendered
dataset. Each step lists the command, where its artifacts land, and what
to check before moving on.

## 0. Prerequisites

- Genesis ROCm venv built (`scripts/setup_genesis_rocm_training.sh`).
- ROS workspace overlay built (`colcon build --packages-select lewm_worlds
  lewm_genesis --symlink-install`) — required because the corpus-generation
  shell wrapper sources the overlay. Rebuild whenever `lewm_worlds/` or
  `lewm_genesis/` source changes.
- Acceptance corpus exists at `.generated/scene_corpus/acceptance/` (used
  by tests and quick probes).

## 1. Generate the scene corpus

```bash
bash scripts/generate_scene_corpus.sh \
  --standard \
  --name production_$(date -u +%Y%m%dT%H%M%SZ) \
  --train 200 --val 50 --test-id 50 --test-hard 50 \
  --plan-seed 1 \
  --out .generated/scene_corpus/production_<timestamp>
```

- Defaults to `validate=True`, so fragmented seeds (no single free
  component contains every beacon) are rerolled via `scene_seed_salt`.
  Salt=0 reproduces the legacy seed exactly; older corpora keep mapping
  to the same scenes.
- Expected runtime at 350 scenes: ~1 minute on CPU. Reroll rate ~9%,
  max salt observed in practice: 3 (budget: 32).
- Output: per-scene `manifest.json`, `world.sdf`, `topology.json`,
  `genesis_scene.json` plus a top-level `corpus.json` listing every
  assignment (including its salt).

## 2. Pre-flight audit

```bash
PYTHONPATH=lewm_genesis:lewm_worlds python3 scripts/audit_scene_corpus.py \
  .generated/scene_corpus/production_<timestamp>
```

- Walks every `manifest.json` and re-runs `audit_scene_reachability`.
- Exits non-zero on any failure — use as a CI gate before launching the
  bulk render.
- Fast (~30 ms per scene). Run any time you change spawn-safety
  constants in `lewm_genesis/rollout.py`, the planning-grid inflation,
  or motif builders — those are the things that can retroactively
  invalidate a previously-valid corpus.

## 3. Quality probe (route-teacher beacon yield)

```bash
# Per-family probe: 3 scenes, 4 envs, 400 blocks, no RGB, no writer.
for FAMILY in large_enclosed_maze local_composite_motifs loop_alias_stress \
              medium_enclosed_maze open_obstacle_field rough_local_dynamics \
              small_enclosed_maze visual_sensor_stress; do
  bash scripts/genesis_bulk_rollout.sh \
    --scene-corpus .generated/scene_corpus/production_<timestamp> \
    --split train --family "$FAMILY" --scene-limit 3 \
    --n-envs 4 --n-blocks 400 --backend cpu --no-rgb --no-writer \
    --collector-mix route_teacher=1.0 \
    --recovery-interlock-clearance-m 0 \
    --log-progress-every-blocks 0 \
    --out .generated/profile_route_teacher/prod_sample_<timestamp>/$FAMILY
done
```

- Per-env metrics under `<out>/<scene_id>/summary.json` →
  `extra.rollout_stats.per_env_metrics[*].beacons_achieved /
  beacons_available`.
- Expected per-family pass rates (from the large-scale benchmark):
  - `loop_alias_stress`, `open_obstacle_field`, `rough_local_dynamics`,
    `small_enclosed_maze`, `medium_enclosed_maze`: ≥80%
  - `large_enclosed_maze`, `local_composite_motifs`: ≥70%
  - `visual_sensor_stress`: ≥50%
- Anything noticeably below these baselines is a regression — investigate
  before launching the full rollout.

## 4. Bulk rollout (data generation)

```bash
bash scripts/genesis_bulk_rollout.sh \
  --scene-corpus .generated/scene_corpus/production_<timestamp> \
  --split train \
  --n-envs 8 --n-blocks 2000 --backend gpu \
  --collector-mix route_teacher=0.7,frontier=0.2,recovery=0.1 \
  --no-rgb \
  --out .generated/bulk_rollout/<run_name>
```

- Writes MCAP per scene under `<out>/<scene_id>/<scene_id>_<env>.mcap`
  plus `summary.json`.
- For mass datagen, keep inline RGB **disabled** and use the render-replay
  pipeline in §6. Inline RGB is useful for single-env QA, but multi-env
  training data must render each env from its own replayed camera pose.
- `--collector-mix` defaults to the §13 production mix when omitted. The
  route-heavy mix above is a probe/expert-rollout shard, not the default
  JEPA corpus mix.

## 5. Raw-rollout conversion

```bash
bash scripts/convert_smoke_bag_to_raw_rollout.sh \
  .generated/bulk_rollout/<run_name>/<scene_id> \
  --out .generated/raw_rollout/<run_name>/<scene_id> \
  --quality-profile raw_training
```

- `raw_training` is the strictest profile (catches stream rate drops,
  not just dropped messages); use `raw_pilot` while still iterating.
- Output: compact `messages.jsonl` + `summary.json` with the
  contract/data-quality audit results. Both `pass=True` is the gate.

## 6. Render-replay (optional GPU vision generation)

When the bulk rollout was run without RGB (saves disk + time), the
render-replay path re-renders camera frames from the raw_rollout:

```bash
# Plan
bash scripts/plan_bulk_render_replay.sh \
  --raw-root .generated/raw_rollout/<run_name> \
  --out-root .generated/render_plan/<run_name> \
  --backend gpu --camera-hz 10

# Render
bash scripts/render_replay_genesis.sh \
  .generated/render_plan/<run_name>/<seq>_raw_rollout/render_replay_plan.json \
  --scene-corpus .generated/scene_corpus/production_<timestamp> \
  --backend gpu \
  --out .generated/rendered_vision/<run_name>/<seq>
```

- Smoke test the path first (`--max-frames 30 --env-index 0 --backend cpu`)
  before launching a full job — keeps the first failure cheap.
- `frames_rendered.jsonl` carries the camera-pose, joint-state, and
  command-context per frame; downstream label derivation pulls from
  here.
- Do not pass `--overlay-target-label` for training renders. That option is
  reserved for manual QA because it draws privileged target labels into RGB.

## 7. Pre-flight checklist before pressing "go" on a full run

- [ ] `scripts/audit_scene_corpus.py` reports `0 failures`.
- [ ] Per-family probe yields are within the expected ranges (§3).
- [ ] `PYTHONPATH=lewm_genesis:lewm_worlds python3 -m pytest
      lewm_genesis/lewm_genesis/tests lewm_worlds/lewm_worlds/tests`
      reports all green.
- [ ] `colcon build --packages-select lewm_worlds lewm_genesis
      --symlink-install` was run after the most recent source edit, so
      the ROS-wrapped scripts pick up your changes.
- [ ] Disk-space estimate: ~50 MB per scene at full RGB+depth+MCAP,
      ~500 KB per scene MCAP-only. 350-scene production corpus =
      ~17 GB full / ~170 MB MCAP-only.

## Failure modes seen historically

| Symptom | Root cause | Where the gate lives |
| --- | --- | --- |
| All envs stuck at spawn, `cells_visited=1`, `path_length_m<0.5` | Beacon on spawn cell (s_bend, slalom) | Motif builders + `canonical_spawn_cells` `grid.is_free` check |
| `beacons_achieved=0/N` despite long path | Spawn yaw faces a wall | `_align_spawn_yaw` in `_select_spawn_pose` |
| `--fixed-spawn` non-deterministic | Manifest spawn not in canonical, falls back to sampler | Audit script will flag if `manifest_spawn ∉ canonical_spawn_cells` |
| Scene with 0 canonical cells | Fragmented topology (no shared free component) | `plan_corpus(validate=True)` reroll + `_build_one` safety net |
| Stuck on render_replay frame 0 | Manifest changed since the rollout, mismatch on reset pose | Use the *same* `--scene-corpus` path across rollout + replay |

## Related code

- `lewm_worlds/scene_validation.py` — `audit_scene_reachability` (the gate)
- `lewm_worlds/planning_grid.py` — inflated occupancy grid + free-ray helpers
- `lewm_worlds/scene_graph.py` — `canonical_spawn_cells`, `sample_spawn_pose`
- `lewm_worlds/splits.py` — `plan_corpus(validate=True)`, salt mechanism
- `lewm_worlds/corpus.py` — `_build_one` safety net
- `lewm_genesis/rollout.py` — `_select_spawn_pose`, `_align_spawn_yaw`
- `lewm_genesis/collectors/route_teacher.py` — `spawn_restriction_cells`,
  `spawn_planning_grid`
- `scripts/audit_scene_corpus.py` — standalone audit CLI
- `scripts/profile_route_teacher.sh` — quality-probe wrapper
