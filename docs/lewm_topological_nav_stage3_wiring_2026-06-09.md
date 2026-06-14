# Topological Nav — Stage 3 wiring: GoalAdapter, TopologicalNavigator, offline Level-1 routing

Date: 2026-06-09. Follows the Stage 3a GO
(`docs/lewm_topological_nav_stage3a_loop_closure_2026-06-09.md`). Everything
here is offline / pure torch (cached banks; no genesis).

## Design decision recorded up front: no learned ReachabilityHead (for now)

A3 already showed latent-pair reachability regression generalizes at the
majority baseline cross-scene, and the spec's own principle #4 says routing is
graph BFS over the learned memory. So Level-1 routing = **BFS over memory
edges** (exact, zero parameters); the learned ReachabilityHead returns only if
routing proves graph-incompleteness-limited. (The §6.2 unknown/unreliable
gating is subsumed by the §6.1 purity rule at evaluation.)

## Unit A — GoalAdapter (§5.2): gate FAILED, with the decisive diagnostic

`lewm/models/goal_adapter.py` + `scripts/train_goal_adapter.py`: MLP mapping a
frozen single-frame latent into v6 belief space, cross-modal supcon
(positives = same (cell, yaw_bin), self-pair excluded), 3 seeds.
Registered gate (recorded relaxation of the spec's pre-evidence ±15 pp):
+5 pp cell-R@5 over the frozen single-frame cosine baseline AND view-R@1
non-regression. Artifact: `goal_adapter_seq4_e9_v6.json`.

| query | cell R@5 | view R@1 |
|---|---:|---:|
| frozen single-frame cosine (baseline) | 0.580 | **0.301** |
| GoalAdapter (3 seeds) | 0.611–0.614 | 0.252–0.263 |
| belief→belief oracle (alignment ceiling) | 0.633 | 0.297 |

The adapter aligns nearly to the oracle ceiling (+3.2 pp, short of +5), but
**view-R@1 regresses** — and the oracle row shows why: belief space itself is
no better than the raw heading-dominated code at view-level matching. Since
memory nodes are *view keyframes*, the strongest goal→node matcher is the raw
frozen frame code with zero parameters. **Decision: Level-1 goal matching =
raw-frame cosine against node representative-observation latents; adapter kept
as artifact only.** (Third instance of the repo pattern: plan_cost >
GoalEnergyHead; v6 > v7 yaw-objective; raw-frame > GoalAdapter.)

## Unit B — TopologicalNavigator (the seam wiring)

`lewm/memory/topological_navigator.py`: `Memory` subclass for the Stage-0
`HierarchicalPlanner` seam. Composes: H=8 frozen-latent window → v6
BeliefEncoder → `OnlineTopologicalMemory` (§5.4 filter, validated at 0.96
replay coherence) → view-keyframe nodes storing representative-observation raw
latents + opaque obs refs → raw-frame goal matching → BFS routing with a
subgoal **lookahead** (see Unit C) → `select_subgoal` returns the next node's
representative observation as a goal-facing sub-goal image; falls back to the
original goal when unmatched/unlocalized (v2 behaviour; exploration mode is
Stage 4).

## Unit C — offline Level-1 routing probe: 2 of 3 criteria pass; bottleneck precisely localized

`scripts/probe_topo_routing_offline.py`: per held-out trajectory (cached
banks), build the memory online, then sample (t_current, t_goal) queries with
true BFS ≥ 2; goal = the *image latent* at t_goal; score the selected sub-goal
node's majority cell against the true scene graph. 25 scenes scored; 7 skipped
(trajectory never produced a valid d≥2 query — insufficient spatial coverage,
reported not averaged). Artifact: `topo_routing_offline_seq4_e9_v6.json`.

Registered gate: progress ≥ 0.70 ∧ margin vs random ≥ +0.25 ∧ goal-match ≥ 0.60.

**Healthy components:** goal-match accuracy **0.879**; plannable 0.996;
disconnections 0.000; localization upstream already at 0.96 coherence.

**The one defect, found and measured:** the literal BFS first hop lands on
another view of the *same place* ~78% of the time (view fragmentation ≈ 4
nodes/place chains same-place views), so progress@k1 = 0.22 while
non-regression = 0.97. Sub-goal selection needs to skip the same-place
cluster:

| hop rule | progress | non-regress | locality (cells) |
|---|---:|---:|---:|
| k=1 (literal next node) | 0.221 | 0.971 | 0.30 |
| k=4 | 0.574 | 0.971 | 0.83 |
| k=6 | 0.677 | 0.974 | 1.09 |
| **k=8 (adopted provisional)** | **0.744** | 0.979 | 1.32 |
| k=10 | 0.795 | 0.977 | 1.57 |
| privileged first-different-cell (ceiling) | **0.912** | — | ~1 |
| adaptive belief-similarity skip (τ 0.7/0.8/0.9) | 0.458/0.392/0.299 | ~0.97 | — |
| random node (teleports scene-wide) | 0.600 | — | unbounded |
| random LOCAL node (≤2 cells; fair) | 0.552 | — | ≤2 |

**Gate verdict (honest): NOT passed as registered.** k=8 clears progress
(0.744 ≥ 0.70) and goal-match (0.879 ≥ 0.60) but the margin vs random-local is
+0.19 < +0.25. No further criterion adjustment — the record stands. The
adaptive belief-similarity skip failed (running-mean node embeddings of
same-cell views are less mutually similar than hoped — v6's yaw invariance is
only partial at node level), so fixed k=8 is the provisional deployable rule,
leaving 0.91 − 0.74 = 0.17 on the table, **entirely in the cluster-skip rule**
— not in goal matching, localization, or graph connectivity.

Scope: one belief seed (20260609); upstream stages were 3/3-seed robust.

## Where this leaves Stage 3 → Stage 4

1. The full Level-1 pipeline runs end-to-end offline on held-out scenes with
   zero privileged signals at query time: goal image → node match (0.88) →
   BFS route → local sub-goal that progresses 0.74 of the time (ceiling 0.91)
   while staying within ~1.3 cells of the agent.
2. **The remaining headroom is one rule** (same-place cluster skipping).
   Candidates for Stage 4, where the closed loop grounds the choice (Level 3
   either reaches the sub-goal or it doesn't): (a) tag commit-chain edges
   (consecutive novelty commits) vs loop-closure edges and skip within-chain;
   (b) collapse view-clusters into place-nodes at routing time via the pair
   scorer; (c) keep k=8 and let Level-2 plateau-escalation absorb the misses.
3. **Stage 4 next:** closed-loop run with the navigator in the
   `HierarchicalPlanner` seam (vulkan venv, `--apply-textures --backend
   vulkan`), exploration mode (§6.2: least-visited-node frontier when the goal
   is unmatched), perceptual arrival (§6.4), baselines 0/1/2 of the §9.4
   ladder, per-level §9.3 diagnostics.

## Reproduce

```bash
~/TinyQuadJEPA/bin/python scripts/train_goal_adapter.py \
  --belief-encoder .generated/topo_nav/belief_encoder_seq4_e9_v6_train32_encoders/belief_encoder_seed20260609.pt \
  --yaw-train-banks .generated/topo_nav/belief_banks_yaw_train32.pt \
  --yaw-eval-banks .generated/topo_nav/belief_banks_yaw_eval.pt \
  --output .generated/topo_nav/goal_adapter_seq4_e9_v6.json --device cuda
~/TinyQuadJEPA/bin/python scripts/probe_topo_routing_offline.py \
  --belief-encoder .generated/topo_nav/belief_encoder_seq4_e9_v6_train32_encoders/belief_encoder_seed20260609.pt \
  --yaw-train-banks .generated/topo_nav/belief_banks_yaw_train32.pt \
  --traj-banks .generated/topo_nav/traj_banks_yaw_eval.pt \
  --gate-lookahead 8 \
  --output .generated/topo_nav/topo_routing_offline_seq4_e9_v6.json --device cuda
```
