# Topological Nav — Stage 4a: first closed-loop run (harness + findings)

Date: 2026-06-10. Follows the Stage 3 wiring
(`docs/lewm_topological_nav_stage3_wiring_2026-06-09.md`).
Harness: `scripts/benchmark_topo_nav_e2e.py` (genesis vulkan venv, CPU torch,
`--apply-textures --backend vulkan`). Artifacts: `.generated/topo_nav/
topo_nav_e2e_smoke.{json,log}`, persisted head `same_yaw_loop_head.pt`.

## Protocol (tour + seek)

- **Tour** (privileged *motion* only — stands in for exploration, deferred;
  perception deployment-valid): dense pose walk along a DFS traversal of the
  scene graph (~0.12 m steps, yaw facing motion); the `TopologicalNavigator`
  builds its memory live from rendered ego frames (held-out
  `medium_enclosed_maze`, 14 cells, 182 steps → 44 nodes).
- **Seek**: goal IMAGE rendered at a tour-visited cell (facing the tour's
  pass-through heading — the registered goal-facing convention); policies on
  identical (start, goal): `topo` (learned stack), `v2` (servo at goal image),
  `bearing` (privileged oracle), `hold`. Ground truth only for eval.

## What WORKS live (first time, end to end)

- Memory built online in the sim loop; **goal image matched to the correct
  node/cell (score 0.989, cell-correct)**.
- **Level-1 routing live: chosen subgoals reduce true goal distance 0.91–1.0
  of replans** — matches the offline probe.
- The plumbing: navigator in the loop at block cadence, filter localization
  feeding subgoal commitment, kinematic veto, scan state machine.

## The iteration ladder (each fix exposed the next layer — all real)

1. Waypoint-follower tour too slow → teleport-walk tour (motion is privileged
   anyway).
2. Goal 9 hops at a 50-block budget → un-winnable for the *oracle* (the §3
   benchmark-budgeting trap, caught again) → bounded goal hops + bigger budget.
3. Per-block replanning → subgoal flapping (yaw_left/right ×59, zero
   translation) → **subgoal commitment** (hold until reached or budget).
4. Raw-cosine "subgoal reached" fooled by same-heading corridor aliasing (66
   false positives) → **spec §6.2 rule: reached = filter localizes to the
   node** (MAP == subgoal or posterior ≥ 0.5).
5. Level-3 arcs grinding into walls (84 blocked arcs) → **kinematic veto**
   (spec's non-learned veto; was never wired — beacon tasks were open-field).
6. 2-step MPC horizon cannot plan large turns → pure yaw oscillation →
   **scan-then-servo** ported from the validated perception demo (scan once at
   start / when lost, not on every subgoal — the first version scanned
   forever).
7. Node-path lengths here (~6–10 nodes for 5-cell routes; commits every
   ~0.5 m) are shorter than the offline banks' (~4 nodes/cell) → lookahead
   k=8 clamped to the goal node itself (topo ≡ v2); k=3 matches this memory's
   node density. **Lookahead must be calibrated to node density, not fixed.**

## The blocking finding (the real Stage 4 result)

After all of the above, the robot still does not translate down corridors:
post-scan, the vetoed `plan_cost` chooser picks hold/yaw (~0.3 m path in 200
blocks). **Diagnosis: `plan_cost` between two views of the same bare corridor
~0.5–1.5 m apart is flat** — the documented distance-concentration weakness.
The beacon results (0.92/0.73) were earned in the *salient-object* regime: a
unique beacon in view gives the latent cost a real gradient; interior corridor
keyframes of a textured maze do not. Level 3's image-servoing contract is
narrower than "visible + goal-facing": it is **visible + goal-facing +
visually salient**.

This does NOT contradict the adopted architecture — it identifies the one
unimplemented spec mechanism designed for exactly this: **`MemoryEdge.action_summary`**
(§5.4) — the actions that produced each edge, recorded during exploration.
SPTM-style traversal then needs no image-servoing between adjacent nodes:
align to the current node's view (scan against its keyframe — view-nodes are
heading-specific, so alignment recovers the tour heading), then **replay the
edge's recorded action** (mostly "forward"), re-localize, repeat; `plan_cost`
servoing is reserved for the final salient approach at the goal. Deployment-
valid (the robot's own actions). The teleport tour records no actions — Stage
4b therefore needs a kinematically *driven* tour (privileged route, real
primitives), which also exercises edge recording for the future exploration
policy.

## Registered next steps (Stage 4b)

1. Driven tour (bearing-follower along the DFS route with the kinematic veto,
   real primitives at block cadence) recording per-edge action summaries on
   `MemoryEdge`.
2. Seek = align-to-node-view (scan vs MAP node keyframe) → replay edge action
   → re-localize; `plan_cost` servo only for the last hop (goal view salient
   by the goal-facing convention).
3. Then the registered Stage 4 gate: success vs v2 on ≥ 8 scenes × 2 goals,
   non-visible goals, §9.4 baselines 0/1/2, per-level diagnostics; exploration
   mode (§6.2 frontier) after that.

## Scope notes

Single scene/goal so far (harness iteration, not an evaluation). The §9.3
decomposition already localizes the work: Level 1 healthy live; Level 2
mechanics (commitment, filter-reached) in place; Level 3 = the blocker, with
the spec's own edge-action mechanism as the registered fix.
