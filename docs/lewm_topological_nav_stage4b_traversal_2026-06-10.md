# Topological Nav — Stage 4b: edge-directed traversal (align → walk → re-localize)

Date: 2026-06-10. Follows Stage 4a
(`docs/lewm_topological_nav_stage4a_e2e_2026-06-10.md`), whose blocking finding
was: `plan_cost` is flat between same-corridor views 0.5–1.5 m apart, so
image-servoing between interior keyframes stalls (hold/yaw attractor). Level-3
image servoing only works on *salient* goal-facing views (the beacon regime).

## Registered design (written before running)

**Key observation:** the tour walks forward only, so the spec §5.4
`action_summary` of every directed edge is ≡ "forward" — all directional
information lives in the node keyframes: a view-node's keyframe faces the
direction the tour was moving when the node was committed. Hence:

1. **Route over DIRECTED memory edges.** The DFS tour (with backtracking)
   traverses tree edges in both directions, so directed connectivity covers
   the toured graph. For a directed edge A→B, B's keyframe faces the A→B
   travel direction by construction.
2. **Traverse node-by-node:** for the next path node B — ALIGN: rotate,
   sampling raw-frame cosine against B's keyframe (early-exit on high match;
   the corridor view self-similarity that *broke* the distance gradient
   actively *helps* heading alignment); WALK: forward (kinematic veto) until
   the filter localizes to B (MAP == B or posterior ≥ 0.5) or a per-edge block
   budget expires (→ replan path from current MAP). No `plan_cost` between
   interior nodes at all.
3. **Final hop:** at the goal node, switch to `plan_cost` servoing on the goal
   image (salient + goal-facing by the registered convention) + perceptual
   arrival (§6.4).
4. Lookahead becomes moot (traversal is node-by-node); the Stage-3 cluster-skip
   question dissolves in the traversal design.
5. Full per-edge action recording (for non-forward motion) is deferred to the
   exploration-mode work, where it becomes necessary; registered as such.

**Smoke gate (registered):** on the Stage-4a failure case
(medium_enclosed_maze_0198ae36dc38, 5-hop / 4.0 m non-visible goal), the
traversal controller must translate (path length ≫ 0.3 m) and reduce final
distance materially below the v2 baseline; success-by-radius is the target but
progress + per-level diagnostics are the decision evidence at smoke N=1.

## Result — FIRST END-TO-END SUCCESS (smoke gate PASSED)

`medium_enclosed_maze_0198ae36dc38` (held-out), goal cell 5 hops / **4.00 m**
away, goal image only, no privileged signals in the policy:

| policy | final dist | path walked | outcome |
|---|---:|---:|---|
| **topo (learned stack)** | **0.36 m** | 4.48 m | **SUCCESS — perceptual stop (§6.4) inside the goal radius** |
| v2 (servo at goal image) | 3.99 m | 0.10 m | never moves (the known failure the hierarchy fixes) |
| bearing (privileged oracle) | ~3.9 m | — | jams on maze walls (greedy heading) |

Three more integration findings were needed beyond the registered design, each
measured before fixed:

1. **Pure-directed routing strands** (fresh seek-nodes have only incoming
   edges) and **pure-undirected routing walks the tour backward** facing away
   from every keyframe (paths were 5/5, 9/12, 17/19 reversed). Fix:
   `_weighted_path` — Dijkstra with reversed edges allowed at 3× cost +
   multi-candidate goal nodes (the same place exists under several view-nodes;
   prefer the directed-reachable one), with the 180° alignment correction for
   residual reversed edges.
2. **Arrival must accept any node at/after the path index** — parallel
   view-nodes split the posterior, so MAP advances along sibling chains.
3. **THE DECISIVE FIX: freeze the belief window during ALIGN.** The H=8
   window assumes smooth locomotion (the training distribution); scan
   rotations poison it for ~8 blocks after every alignment, which silently
   prevented every filter-confirmed arrival. Feeding the filter only during
   WALK (+ clearing the stale window on walk entry) took the run from
   "translates but never arrives" to **success in one step.**

Interesting nuance: intermediate node-arrival confirmations still never fired
(`sg_reached 0`) — the robot succeeded via align → walk → budget-exhaust →
**re-localize-and-replan**, three times, then the perceptual §6.4 stop at the
goal. "Replan when lost" is doing the Level-2 work; per-node confirmation is
optional, which simplifies the controller.

## Demo video

`.generated/topo_nav/topo_nav_e2e_success_demo.mp4` (~76 s, 12 fps, repo HUD
format): TOUR phase (memory nodes accumulating on the minimap trail) → SEEK
phase (the goal *image* shown as an inset — the only task input — with live
goal distance) → **PERCEPTUAL STOP banner at 0.4 m**. Regenerate with
`--demo-video <path>` on the same command (deterministic; the success
reproduced exactly under demo rendering).

## Status + next

This is the first closed-loop validation of the thesis mechanism end to end:
frozen LeWM recognition substrate + BeliefEncoder + calibrated loop-closure
filter memory + view-keyframe goal matching + graph routing + align/walk
traversal + perceptual arrival, in a held-out maze, from a goal image. N=1
smoke — NOT yet the Stage 4 gate. Next: the registered evaluation (≥8 scenes ×
2 goals, non-visible goals, baselines 0/1/2, per-level diagnostics, plus a
goal-selection RNG already made reproducible), then exploration mode (§6.2) to
retire the privileged tour.
