# Collection Policy Reference

This is the implementation reference for the §13 collection mix in
[fresh_retrain_data_spec.md](fresh_retrain_data_spec.md). The privileged
labels these collectors use are scoped by
[v3_hjepa_plan.md §3.4](v3_hjepa_plan.md) — every signal here is allowed at
data generation time, and **none** of it may enter the deployed model's
input vector.

## Architecture

```
            ┌────────────────────────────────────┐
            │       EpisodeScheduler             │
            │ draws collector per env, per       │
            │ episode, from §13 share table      │
            └──────────────────┬─────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
  ┌─────▼──────┐       ┌───────▼──────┐      ┌────────▼─────┐
  │ RouteTeach │       │ FrontierTeach│  ... │ OUExploration│
  │  (BFS→     │       │ (least-visit │      │  (snap to    │
  │   landmark)│       │   reachable) │      │   primitive) │
  └─────┬──────┘       └───────┬──────┘      └────────┬─────┘
        │                      │                      │
        ▼                      ▼                      ▼
   BlockChoice(requested_block, primitive_name,
               command_source, route_target_id, next_waypoint_id)
                              │
                              ▼
              ┌──────────────────────────────┐
              │  RolloutRunner._collect_block│
              │  emits CommandBlock per env  │
              └──────────────────────────────┘
```

`RolloutRunner` calls `EpisodeScheduler.on_episode_reset(env_idx)` whenever
an env resets (initial spawn, fall, or out-of-bounds). That redraws the
collector for that env from the share table so the §13 mix is preserved
across the whole run, not just at episode 0.

## Collector reference

Each collector implements [`CollectorPolicy`](../lewm_genesis/lewm_genesis/collectors/base.py)
(`name`, `on_episode_reset(env_idx)`, `on_block(...)` → `BlockChoice`).

| Name | Share | What it does | Privileged inputs |
| --- | ---: | --- | --- |
| `route_teacher` | 30 % | Picks a goal cell (landmark, or random distant cell) per episode. Each block runs BFS from the current cell, picks the next waypoint, and emits a primitive that steers toward that waypoint's center. Re-picks goal on arrival or when unreachable. | scene graph, current cell, base xy/yaw |
| `frontier` | 20 % | Per-env visit counts over cells; picks the highest-scoring reachable cell where `score = -visits + 0.2 × graph_distance`. Visit counts carry across episodes (intentional — seeds loop-closure). | scene graph, current cell, base xy/yaw, per-env visit history |
| `primitive_curriculum` | 20 % | Uniform random over trainable velocity primitives. The legacy sampler, kept as a collector so its share is auditable. | none |
| `ou_noise` | 10 % | Ornstein–Uhlenbeck process in `(vx, vy, yaw_rate)` snapped to the nearest trainable primitive. Drives smooth drifts through the action space. | none |
| `recovery` | 10 % | FSM: `approach` (steer toward a distant cell, ignoring clearance) → `backout` (`backward` primitive for `backout_blocks` blocks once clearance < `approach_clearance_m`) → `pivot` (`yaw_left`/`yaw_right` for `pivot_blocks` blocks) → next approach. Produces the "wall contact + recovery" examples in §10. | scene graph, current cell, base xy/yaw, **clearance to walls** |
| `loop_revisit` | 10 % | Same as `route_teacher` but with `revisit_after_arrival=True`: re-targets to a previously visited goal cell (across all envs' histories) once any goal exists. Generates loop-closure pairs for [v3_hjepa_plan.md §5.3](v3_hjepa_plan.md). | as `route_teacher`, plus per-env goal history |

`RouteTeacher`, `FrontierTeacher`, and `RecoveryCurriculum` all use the
same bearing-to-primitive mapping:

| heading error | primitive |
| --- | --- |
| ≤ π/8 | `forward_medium` |
| ≤ π/3 | `arc_left` / `arc_right` |
| > π/3 | `yaw_left` / `yaw_right` |

Defined in `primitive_toward_bearing` and overridable per-collector.

## Privileged-label outputs

Every `CommandBlock` message now carries three audit fields:

- `command_source` — name of the collector that produced this block
  (`"route_teacher"`, `"frontier"`, `"primitive_curriculum"`, `"ou_noise"`,
  `"recovery"`, `"loop_revisit"`).
- `route_target_id` — the goal cell the teacher is heading toward, or `-1`
  when the collector doesn't have a goal (`primitive_curriculum`,
  `ou_noise`, `recovery` in approach to a wall).
- `next_waypoint_id` — the next cell on the BFS shortest path, or `-1`.

These are stamped at request time so downstream audits can join by
`sequence_id` against `ExecutedCommandBlock` and replay the teacher's
intent against the executed motion. They are deployment labels only.

## Spawn-pose randomization

`RolloutConfig.randomize_spawn_pose=True` (default) calls
`SceneGraph.sample_spawn_pose(rng, clearance_floor_m=0.20)` for every reset
env. Cells whose center clearance to walls falls below the floor are
rejected; if no cell passes, the rollout falls back to the manifest spawn
so the scene always boots. Per-env spawn poses are cached so the
ResetEvent log, the camera-pose helper, and the next reset all see the
same spawn.

Disable with `RolloutConfig(randomize_spawn_pose=False)` for fixed-spawn
debugging.

## Configuring the mix

```python
from lewm_genesis.collectors import DEFAULT_COLLECTION_MIX
from lewm_genesis.rollout import RolloutConfig

# Use the spec mix (default)
config = RolloutConfig()

# Use a custom mix (must reference registered collectors only)
config = RolloutConfig(
    collector_mix={"route_teacher": 0.5, "recovery": 0.3, "primitive_curriculum": 0.2}
)

# Force the legacy uniform random sampler (empty dict)
config = RolloutConfig(collector_mix={})
```

If a referenced collector isn't registered, its share is redistributed
across the available collectors rather than erroring. If the entire mix
references unregistered collectors, the scheduler falls back to uniform.

## Realized-mix audit

Every rollout's `stats` dict carries `collector_mix_realized`, which the
bulk-rollout script writes into the per-scene `summary.json`:

```json
{
  "rollout_stats": {
    "scene_id": "large_enclosed_maze_be5794389320",
    "collector_mix_realized": {
      "route_teacher": 6,
      "frontier": 4,
      "primitive_curriculum": 5,
      "ou_noise": 2,
      "recovery": 2,
      "loop_revisit": 1
    }
  }
}
```

The counts are episode counts per collector (one per env per reset), so
the realized share is `count / sum(counts)`.

## Adding a new collector

1. Implement the [`CollectorPolicy`](../lewm_genesis/lewm_genesis/collectors/base.py)
   protocol in a new module under `lewm_genesis/collectors/`.
2. Register it in `build_default_policies` in
   [`lewm_genesis/collectors/__init__.py`](../lewm_genesis/lewm_genesis/collectors/__init__.py).
3. Add an entry to `DEFAULT_COLLECTION_MIX` (or pass via
   `RolloutConfig.collector_mix`).
4. Add a unit test in `lewm_genesis/tests/test_collectors.py`.

The collector receives a `SceneGraph` (or `None` if the scene has no
graph) on every `on_block` call. If your collector requires the graph,
guard against `scene is None` and emit a safe fallback (`hold` or a random
primitive) — `EpisodeScheduler` collapses the mix to graph-free policies
when no graph is available, but defensive coding here keeps the rollout
boot path safe.
