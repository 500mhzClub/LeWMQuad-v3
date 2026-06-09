# Topological Neural Navigation — Implementation Plan (grounded in June findings + literature)

Date: 2026-06-09

> **Naming (2026-06-09).** This was "the H-JEPA plan"; the name was a misnomer.
> What we are building is **not** an H-JEPA (a hierarchy of JEPA *predictors* at
> multiple timescales). It is **one flat JEPA predictor (LeWM) + a topological
> recognition memory + a hierarchical planner** — the SPTM / Neural Topological
> SLAM lineage. The hierarchy is in *planning and memory, not prediction*. A
> *true* H-JEPA (a learned high-level predictor over place embeddings) is kept as
> a **deferred research bet**, baselined against the symbolic graph built here —
> see §8. Decision (2026-06-09): pursue option 3 — **build the topological stack
> now; treat the learned high-level predictor as a later bet measured against this
> graph.**

This sequences the build of the topological navigation stack specified in
`docs/v3_topological_nav_plan.md` (the authoritative design, written 2026-05-17,
renamed from `v3_hjepa_plan.md`), updated with everything learned since — the
task-aligned program, the closed-loop grounding, and the external nav-JEPA
literature (`docs/lewm_pose_aux_literature_and_options_2026-06-06.md`). It does
not restate the v3 spec; it (a) records which decisions the June evidence
settles, (b) marks Phase A as answered, and (c) gives a staged, de-risked build
order.

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
| A0 planner refactor | **done (2026-06-09)** | extracted to `lewm/planning/` + `lewm/memory/` (genesis-free); benchmark delegates; behaviour-lock gate passes (`lewm/tests/test_planning_refactor.py`, 6/6, + benchmark-wrapper equivalence). See Stage 0 below. |
| A1 cell-graph extraction | done | `lewm_worlds.labels.derived` / mined task-aligned decisions carry `cell_id`, `local_graph_type`, `route_target_id`, `oracle_next_cell_id`, BFS targets |
| A2 visual-aliasing audit | done | recognition R@1 ≈ 0.42 (good); latent↔graph ρ≈0.03 (severe metric aliasing) → A4 "insufficient frozen-latent regime" |
| A3 frozen reachability probe | done | reachability head ≈ baseline (nav-cost diagnosis) |
| A4 decision gate | **passed → build Phase B** | the closed-loop recognition-vs-metric result is the strongest confirmation |

The one open A-stage item was **A0 (refactor)** — **done 2026-06-09** (Stage 0
below). Phase B now proceeds.

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

The multi-beacon demo faked the topological-nav stack with ground truth. Each
privileged bit maps to exactly one learned component:

| demo privileged bit | topological-stack replacement |
|---|---|
| subgoal placed 2 m toward the true beacon | Level 1–2 planner: memory routing + ReachabilityHead picks the next *visible* node along the graph |
| goal keyframe rendered from the true beacon pose | memory node `representative_observation` (committed online) + GoalAdapter |
| "claimed" via true robot↔beacon distance | GoalMatchHead / filter lock (§6.4 perceptual success) |
| nearest-neighbour tour over true positions | shortest path over the learned topological graph |

Building these *is* the project.

## 5. Staged, de-risked build order

Each stage has a registered gate; the next stage runs only if the gate passes.

### Stage 0 — Planner refactor (A0) — **DONE 2026-06-09**
Extracted the benchmark's planner into the v3 module contract under the `lewm`
package (all genesis-free so they unit-test with a fake model):
- `lewm/planning/primitive_bank.py` — `active_blocks` + `candidate_action_tensor`.
- `lewm/planning/costs.py` — pure `rollout_costs(..., allow_pose_head)` unifying the
  exact math of `_choose_lewm_primitive` (energy/plan_cost) and
  `_lewm_primitive_costs` (pose/energy/plan_cost); the head-selection asymmetry is
  preserved by `allow_pose_head`.
- `lewm/planning/local_mpc.py` — `PlannerState`/`GoalSpec` bundles + `LocalMPC`
  (`.choose`, `.candidate_costs`) seeded from `_choose_lewm_primitive` (plan_cost).
- `lewm/memory/topological_memory.py` — abstract `Memory` + `KeyframeMemory`
  baseline (goal image = sub-goal → v2-identical).
- `lewm/planning/hierarchical_planner.py` — `HierarchicalPlanner` routing
  LocalMPC + KeyframeMemory; the seam where Stages 2–3 plug in.

`scripts/benchmark_lewm_closed_loop_mpc.py` now delegates its four planner
functions to these (signatures unchanged → rest of benchmark untouched).
**Gate PASSED:** `lewm/tests/test_planning_refactor.py` (6/6) asserts the
extracted modules reproduce a verbatim reference of the original math
bit-for-bit (incl. the asymmetry and the hierarchical-planner identity), and a
benchmark-wrapper equivalence check confirms `_choose_lewm_primitive` /
`_lewm_primitive_costs` are unchanged end-to-end. (Full closed-loop re-run on a
real scene is the optional heavier confirmation; the unit gate is the registered
v3 §4.1 acceptance.)

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

1. ~~Re-validate seq4 place-recognition R@1~~ — **DONE**: R@1 0.43, ρ 0.08
   (recognition-not-metric confirmed on seq4).
   `docs/lewm_topological_nav_stage1_retrieval_2026-06-09.md`.
2. ~~Stage 0 planner refactor (modular `planning/` + `memory/` seam),
   behaviour-locked~~ — **DONE** (Stage 0 above; `lewm/tests/test_planning_refactor.py`).
3. ~~Stage 1 minimal frozen-latent topological memory + the H2 gate decision~~ —
   **DONE**: naive frozen-history pooling fails the recall gate, but
   history-disambiguability AUC 0.86 on aliased pairs → **decision: build the
   Stage 2 BeliefEncoder** (the info is present, just not poolable).

**Next deliverable → Stage 2 (BeliefEncoder).** Train the contrastive history
encoder against the registered bars: beat the naive-pooling Recall@5 baseline and
approach the AUC-0.86 history-separation ceiling on aliased pairs; if it cannot
despite the present signal, H2 falsifies → fork to DINOv2 patch features (§6). The
`HierarchicalPlanner` + `Memory` seam from Stage 0 is where the learned memory
plugs in.

## 8. Deferred research bet: a *true* H-JEPA (option 3)

The stack above is hierarchical in **planning/memory**, not in **prediction**.
A genuine H-JEPA would add a second, *learned* predictor at a coarser level —
not a counted graph. Concretely: a **high-level JEPA predictor over place
embeddings** that, given the current place embedding and a coarse action,
predicts the *next place* embedding, and plans by rolling that predictor out at
the abstract level (instead of, or above, BFS over the symbolic graph). Levels:

- **Level 0 (have it):** LeWM, frozen — local dynamics in pixel-latent space.
- **Level 1 (the new bet):** a learned predictor over *place* embeddings —
  `z_place_{t+1} ≈ f(z_place_t, coarse_action)` — trained on the same
  memory-generated node transitions the topological graph already collects
  (`transition_count`, `action_summary` become its supervision).

**Why it is deferred, not chosen now.** A learned high-level predictor only helps
if its abstract latent is *plannable* (smooth/monotone enough to roll out and
score). Our load-bearing finding is **recognition-not-metric** at Level 0
(latent↔graph ρ≈0.03; pose-aux failed to inject actionable geometry). There is no
evidence the *place-level* latent escapes this, so a Level-1 predictor risks
inheriting the same flat-cost failure — while the symbolic graph sidesteps it by
construction (BFS is exact). So the topological stack is the lower-risk path that
fits the substrate we actually have.

**Entry gate (run only after the topological stack works, and only if it has a
ceiling the graph cannot raise).** Before building Level 1, run a cheap
*offline* probe analogous to Phase A: take the place embeddings the topological
memory already produces, fit `z_place_{t+1} ≈ f(z_place_t, coarse_action)`, and
measure whether rolling it out 2–4 abstract steps yields a **plannable** ordering
(does predicted-place distance/order correlate with true graph distance on
held-out scenes?). **Baseline = the symbolic graph + BFS** built here. The bet is
funded only if the learned predictor *beats the graph* on a task the graph
demonstrably caps (e.g. generalizing transitions to unobserved node pairs,
where BFS has no edge but a learned predictor could interpolate). If it does not
clear the graph baseline, the H-JEPA name stays aspirational and the symbolic
stack stands. This keeps "H-JEPA" honest: a measured research direction with a
concrete baseline, not the label for what we ship.
