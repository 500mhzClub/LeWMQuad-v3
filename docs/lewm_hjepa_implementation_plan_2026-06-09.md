# H-JEPA Implementation Plan (grounded in June findings + literature)

Date: 2026-06-09

This sequences the build of the H-JEPA navigation stack specified in
`docs/v3_hjepa_plan.md` (the authoritative design, written 2026-05-17), updated
with everything learned since — the task-aligned program, the closed-loop
grounding, and the external nav-JEPA literature
(`docs/lewm_pose_aux_literature_and_options_2026-06-06.md`). It does not restate
the v3 spec; it (a) records which decisions the June evidence settles, (b) marks
Phase A as answered, and (c) gives a staged, de-risked build order.

## 1. Convergent evidence: the architecture is decided

Three independent lines agree on a **recognition-based topological** stack, not a
metric one:

- **Our closed loop (2026-06-09).** Pure-perception servoing fails because the
  LeWM latent is a *recognition* code, not a *metric* one — confirmed with two
  cost models: `plan_cost` is flat over distance (recognition only), the
  `RelPoseHead` predicts ~constant distance (only weakly metric ρ≈0.22). The
  scan/recognition half works; the metric approach does not.
  (`docs/lewm_perception_servoing_limit_2026-06-09.md`.)
- **Phase A diagnostics (June).** Frozen LeWM latents are a good place-recognition
  code (retrieval@1 ≈ 0.42, ~21× lift) but a poor metric code (ρ≈0.03; the
  reachability head ≈ baseline). [[project_lewm_aliasing_a2]],
  [[project_lewm_nav_cost_phase0]].
- **Literature.** "Good prediction ≠ good planning" (WDS 2512.24497); L2-in-latent
  is anti-metric and is rescued by SSL-reg or CEM sampling (PLDM, DINO-WM); NWM
  uses metric-free goal-image scoring. The literature review ranks
  **recognition/topological planning** as the #1–2 option precisely because it
  "leverages retrieval@1 ≈ 0.42 and avoids treating the pooled latent as a
  continuous map."

**Conclusion.** v3's H1 ("single-frame latents don't preserve enough topology")
is true; build the belief/memory/reachability stack. The whole design rests on
**recognition + history**, never on metric latent distance.

## 2. Phase A is answered — proceed to Phase B

| v3 Phase A step | status | evidence |
|---|---|---|
| A0 planner refactor | partial | the closed-loop benchmark has proto-`LocalMPC` (`_choose_lewm_primitive` + CEM-ish primitive scan) and a proto-`hierarchical` multi-beacon loop, but not the modular `planning/` + `memory/` contract |
| A1 cell-graph extraction | done | `lewm_worlds.labels.derived` / mined task-aligned decisions carry `cell_id`, `local_graph_type`, `route_target_id`, `oracle_next_cell_id`, BFS targets |
| A2 visual-aliasing audit | done | recognition R@1 ≈ 0.42 (good); latent↔graph ρ≈0.03 (severe metric aliasing) → A4 "insufficient frozen-latent regime" |
| A3 frozen reachability probe | done | reachability head ≈ baseline (nav-cost diagnosis) |
| A4 decision gate | **passed → build Phase B** | the closed-loop recognition-vs-metric result is the strongest confirmation |

The one open A-stage item is **A0 (refactor)**, which is the first build task.

## 3. Decisions the June evidence settles (deltas to the v3 spec)

1. **Base checkpoint = `seq4_e9`, frozen.** It is the better nav base (visible
   beacon 0.73–0.92 vs seq11 0.17–0.25), validated across families. **Re-validate
   place-recognition R@1 on seq4** before building memory (the 0.42 figure
   predates the seq4-vs-seq11 split). `docs/lewm_nav_base_synthesis_2026-06-09.md`.
2. **Local cost = `plan_cost` (latent L2), not the GoalEnergyHead.** The learned
   energy head *degrades* nav (contrastive ranking ≠ servoing cost); plain
   recognition L2 is the better Level-3 cost. `docs/lewm_energy_head_vs_plancost_2026-06-09.md`.
3. **`representative_observation` must be approach/goal-facing.** A goal image
   rotated even 90° off collapses servoing 0.92→0.00 (`--goal-yaw-offset-rad`
   test). So when a memory node is committed (§5.4) its stored representative
   observation, and any GoalAdapter goal image, must be a goal-facing keyframe —
   not an arbitrary-yaw frame. This is a hard constraint on §5.2/§5.4.
4. **No metric components.** Do not build anything that depends on metric latent
   distance: ReachabilityHead is a recognition-based *bucket classifier*
   (the spec already frames it this way); subgoal success is recognition
   (am-I-at-this-node), not a distance threshold; routing is graph BFS over the
   learned memory, not latent L2.
5. **Local servoing (Level 3) already works for VISIBLE subgoals.** seq4 +
   `plan_cost` CEM reaches a visible beacon — so the hierarchy only needs to keep
   each subgoal *visible and goal-facing*; the execution layer is solved.

## 4. The demo's privileged scaffolding == the components to build

The multi-beacon demo faked the H-JEPA stack with ground truth. Each privileged
bit maps to exactly one learned component:

| demo privileged bit | H-JEPA replacement |
|---|---|
| subgoal placed 2 m toward the true beacon | Level 1–2 planner: memory routing + ReachabilityHead picks the next *visible* node along the graph |
| goal keyframe rendered from the true beacon pose | memory node `representative_observation` (committed online) + GoalAdapter |
| "claimed" via true robot↔beacon distance | GoalMatchHead / filter lock (§6.4 perceptual success) |
| nearest-neighbour tour over true positions | shortest path over the learned topological graph |

Building these *is* the project.

## 5. Staged, de-risked build order

Each stage has a registered gate; the next stage runs only if the gate passes.

### Stage 0 — Planner refactor (A0)
Extract the benchmark's planner into the v3 module contract: `planning/local_mpc.py`,
`planning/costs.py`, `planning/primitive_bank.py`, `memory/topological_memory.py`
(abstract `Memory` + a `KeyframeMemory` baseline), `planning/hierarchical_planner.py`.
Seed `LocalMPC` from `_choose_lewm_primitive` (use `plan_cost`). **Gate:** behaviour
identical to the current benchmark within tolerance (v3 §4.1).

### Stage 1 — Minimal topological memory on FROZEN latents (cheap H2 test)
Before the full contrastive BeliefEncoder, test whether frozen seq4 recognition +
a short history is already enough to build a coherent map. This is the
`docs/lewm_phase_b_minimal_retrieval_plan_2026-06-06.md` path, and it directly
re-uses the committed retrieval probes (`probe_lewm_history_retrieval.py`,
`place_retrieval.py`). Build: a frozen-latent place key (raw/proj + last-`H`
history), a cosine `LoopClosure`-style match, and the top-k Bayes filter (§5.4).
**Gate (the H2 question):** on held-out scenes, does frozen-latent + history reach
the §5.5 Same-place Recall and filter-coherence bars? 
- If **yes** → the full BeliefEncoder may be unnecessary; skip to Stage 3.
- If **no** → frozen recognition is insufficient under aliasing → build Stage 2.
(The A2 history-disambiguability audit predicts which: if few aliased pairs are
history-separable at any `H`, even BeliefEncoder can't fix it — see §10.)

### Stage 2 — BeliefEncoder + GoalAdapter + LoopClosureHead (full Phase B)
Only if Stage 1 fails. Train the contrastive history encoder (supcon + VICReg
anti-collapse, body-motion auxiliary), the GoalAdapter (goal image → belief
space, **goal-facing keyframes**), and the calibrated LoopClosureHead (precision
≥ 99%). **Gates:** the full v3 §5.5 acceptance set. **If Same-place Recall@5 does
not beat the frozen-latent baseline by the registered margin, H2 is falsified and
v3's main path stops** — escalate to a substrate change (DINOv2 patch features,
already probed in `probe_dinov2_patch_retrieval.py`) as a separate program.

### Stage 3 — ReachabilityHead + hierarchical planner (Phase C)
Train the recognition-based ReachabilityHead on memory-generated node pairs (6
buckets incl. `unknown/unreliable`, false-loop negatives, purity rule). Wire the
3-level planner: Level 1 memory routing + goal match, Level 2 subgoal pursuit
(goal-facing `representative_observation`), Level 3 = seq4 + `plan_cost` LocalMPC
(validated). **Gate:** v3 §9.3 + the end-to-end gate below.

### Stage 4 — End-to-end evaluation (learned subgoals replace the scaffold)
Re-run the multi-beacon / `landmark` closed-loop task with the *learned* stack
(no privileged subgoals, no ground-truth claim). **References already in hand:**
- ceiling = `bearing` oracle;
- the privileged-scaffold demo (2/2 with ground-truth breadcrumbs) = the
  "perfect subgoals" upper bound for the learned router;
- floor = pure-perception servo+scan (fails, §1).
**Gate:** end-to-end success beats the v2/no-memory baseline by the registered
margin (v3 §10.9); arrival is perceptual (§6.4), not ground-truth.

## 6. Risks / contingencies

- **H2 may be false (Stage 1+2 gate).** If history doesn't disambiguate aliased
  places on this substrate, the topological memory can't be built reliably — the
  bottleneck is the encoder, and the path forks to a substrate change (DINOv2
  patch features / a spatial-token retrieval encoder), which is **out of v3's
  frozen-LeWM scope** and is a separate program. The A2 history audit and Stage 1
  surface this early and cheaply.
- **Data coverage.** Audit the corpus against v3 §8.3 (aliasing pairs, goal-image
  diversity, trajectory diversity) before Stage 2; do not let dataset deficiency
  masquerade as architectural failure. The textured corpus + mined decisions
  already provide cell labels and goal frames.
- **Goal-facing keyframes.** Online node commitment must capture an
  approach-facing representative observation; if the robot commits a node while
  facing away, GoalAdapter/Level-2 servoing will fail (the §3 finding). Add this
  to the §5.4 commit rule.
- **Recognition-not-metric is load-bearing.** Any reintroduction of a metric
  assumption (latent-distance routing, distance-threshold arrival) will silently
  fail; keep everything recognition/graph-based.

## 7. First concrete deliverables

1. Re-validate seq4 place-recognition R@1 (re-run `place_retrieval` /
   `probe_lewm_history_retrieval` on seq4 frozen latents, held-out scenes).
2. Stage 0 planner refactor (the modular `planning/` + `memory/` seam) from the
   benchmark, behaviour-locked.
3. Stage 1 minimal frozen-latent topological memory + the H2 gate decision.

Items 1–2 are low-risk and unblock everything; item 3 is the cheap test that
decides whether the heavy BeliefEncoder build (Stage 2) is even needed.
