# H-JEPA / v3 Plan — Simulator- and Robot-Agnostic Specification

## Purpose of this document

This is a self-contained specification of the v3 / Hierarchical-JEPA plan, written to be portable across simulators and robot embodiments. It assumes:

- A vision-based agent operating in a procedurally generated environment containing a navigable graph of locations (rooms, cells, waypoints, etc.) connected by traversable edges (corridors, free space, etc.).
- An action representation that can be discretised into fixed-length **action blocks** (a contiguous chunk of `K` low-level commands treated as one decision unit). The block size is a hyperparameter, not a structural commitment.
- A pretrained pixel-input JEPA-style world model (LeWM, paper-faithful) that produces per-frame latent embeddings and supports rollout in latent space conditioned on action blocks.
- A current-generation primitive-bank cross-entropy method (CEM) planner with a goal-image-conditioned cost as the existing baseline.
- Onboard proprioception sufficient to integrate body-frame motion across an action block (IMU-equivalent). If the platform does not expose this, see §3.4 for the consequent change in framing.

The plan is written so that none of the steps depend on a specific simulator (Genesis, MuJoCo, Isaac, PyBullet, etc.) or a specific robot (quadruped, wheeled, drone, manipulator base). Where a step requires environment-specific scaffolding, that scaffolding is named abstractly and its contract is specified.

---

## 1. Thesis claim

Goal-conditioned navigation from egocentric perception and action history can be made to work without a metric map, without oracle pose at deployment, and without a privileged deployment-time success signal, by combining:

1. A paper-faithful pixel-input JEPA world model (LeWM) for short-horizon local dynamics.
2. A history-conditioned retrieval embedding (BeliefEncoder) for place / belief representation, built only if frozen LeWM latents fail a reachability diagnostic.
3. A goal adapter that maps a single goal image into the same retrieval space as BeliefEncoder, so that a place embedding from history and a place embedding from a single image are directly comparable.
4. A conservative topological memory with a calibrated loop-closure head and a top-k discrete Bayes filter over candidate current nodes.
5. A reachability head trained on memory-generated node pairs with false-loop negatives and an explicit "unreliable" bucket.
6. A hierarchical planner that selects local sub-goals in the memory graph using reachability, and delegates short-horizon execution to the existing primitive-bank CEM controller with a hard kinematic safety veto.

The thesis is **falsifiable**: if a frozen-latent reachability probe — built on top of the existing LeWM latents with no belief encoder, no memory, no temporal context — meets the registered Phase A4 criteria on a held-out scene set, then the entire belief / memory / reachability stack is unnecessary and the v2 navigation failure is elsewhere (planner cost, latent objective, or scene difficulty). Phase A of this plan exists to answer that question before Phase B is built.

---

## 2. Core hypothesis being tested

> **H1.** Single-frame LeWM latents do not preserve enough graph-topological information to support long-horizon goal-conditioned navigation in environments with strong visual aliasing (e.g. mazes with repeated junctions, identical corridor segments, or repeated room geometry).

> **H2.** A short-history belief embedding trained with a contrastive objective whose targets are derived from the ground-truth scene graph **does** preserve that information, and the gap between H1 and H2 is large enough to explain the navigation failure observed under the v2 architecture.

**Phase A tests H1 directly. Phase A does not test H2** — H2 cannot be tested before BeliefEncoder exists. Phase A therefore tests H1 and decides whether testing H2 is worth doing:

- If H1 is false (frozen-latent probe meets criteria), the belief / memory stack is unnecessary; do not build Phase B; investigate planner cost, goal-conditioning, or latent objective instead.
- If H1 is true (frozen-latent probe fails criteria), build Phase B in order to test H2 directly.
- If Phase B fails to test H2 affirmatively (BeliefEncoder does not improve retrieval over the frozen-latent baseline by a registered margin), the bottleneck is at a deeper representational level — predictor action-conditioning, encoder receptive field, or pretext objective — and the v3 spec stops here pending a separate program of work.

The point is to keep the falsification structure clean: each phase is a question with a registered answer threshold, and the next phase only runs if the previous answer says it should.

---

## 3. Design principles

### 3.1 Things v3 keeps from v2

- **Pixel-input LeWM** as the dynamics backbone. No structural changes to the encoder or predictor.
- **Block-action representation.** All planning operates in blocks of `K` commands, not single steps.
- **Primitive command bank** for CEM sampling. Hand-authored library of behaviour primitives (pivot, side-step, drive forward N blocks, etc.).
- **Kinematic safety veto.** A non-learned safety check on candidate trajectories that vetoes physically infeasible or collision-bound rollouts. The learned cost ranks among the survivors.
- **Goal-image conditioning.** The agent is given an image of the destination; success is whether it gets there.
- **Frozen-encoder principle.** Once LeWM is trained, its weights are not retrained for downstream tasks during the diagnostic and main v3 experiments. All downstream heads (BeliefEncoder, GoalAdapter, LoopClosure, Reachability) train on top of a frozen LeWM. See §3.5 for the one allowed exception.

### 3.2 Things v3 changes

- **Replaces the keyframe-cosine memory router** with a multi-hypothesis topological memory and a top-k discrete Bayes filter.
- **Replaces the goal-image cost** at the planner's outermost level with a learned reachability bucket prediction; the goal-image cost survives only inside the local MPC stage, evaluated against the **sub-goal's** representative observation, not the final destination image.
- **Adds a separately-trained retrieval embedding (BeliefEncoder)** so that memory does not key off the same latents that the predictor produces.
- **Adds a goal adapter** so that a single static goal image can be projected into the same retrieval space as a history-conditioned BeliefEncoder embedding, without distribution shift.
- **Adds an explicit decision gate (Phase A4)** before any of the above is built.

### 3.3 Things v3 deliberately does not do

- No metric SLAM. No occupancy grid. No global pose estimator.
- No use of simulator-privileged state as an **input** to any deployed model. Privileged state (true cell ID, BFS distances, oracle pose) appears only as **automatically generated privileged-label supervision** for downstream heads, never at inference. Note that LeWM itself is JEPA-style self-supervised; the downstream heads (BeliefEncoder, GoalAdapter, LoopClosure, Reachability) are not strictly self-supervised — they consume scene-graph-derived privileged labels.
- No reset-conditioned exploration policy. The agent is evaluated on the policy it had at train time; rollouts are not collected adaptively against the eval distribution.
- No reinforcement-learning reward signal. Downstream heads are trained from automatically generated privileged labels (graph distance buckets, co-location, local graph type), not from task reward. These labels are training-time only.
- No re-training of the LeWM weights as part of v3's main path. If LeWM is found inadequate during Phase A, that is a separate program of work.

### 3.4 The privileged-leak rule and the deployment-input claim

Any signal that depends on simulator-privileged state may appear in **labels** for automatically-generated privileged-label supervision of downstream heads, but must not appear in **inputs** at deployment.

- World-frame Δx, Δy, Δyaw between two timesteps is privileged and may not be used as a model input.
- Body-frame integrated motion `(∫v_body_x dt, ∫v_body_y dt, Δyaw)` over an action block is **not** simulator-privileged — it is reproducible on a real robot from IMU and proprioception, and is therefore admissible as a deployment input on platforms that expose it.
- Cell IDs, scene graph adjacency, beacon/landmark presence, and BFS distance to landmarks are privileged labels only.

**Cell IDs are scene-scoped.** All cell-graph labels (`cell_id`, adjacency, BFS distance) are local to the scene that produced them. Throughout this document, the canonical identifier for a place is the pair `(scene_id, cell_id)`. Any co-location, graph-distance, loop-closure, or reachability label is defined over `(scene_id, cell_id)` pairs, not `cell_id` alone. Pairs drawn from different scene graphs have no finite BFS distance: they are not positives, and they are not finite-bucket negatives — they belong in the `unknown/unreliable` bucket where an unknown bucket exists, or are excluded from training where it does not. Failing to scope cell IDs by scene conflates structurally similar but semantically unrelated procedural cells and silently corrupts every contrastive label.

**Consequence for the deployment-input claim.** If body-frame motion is used as an input, the system is not "vision-only"; it is "egocentric perception plus action history plus proprioceptive motion." The thesis claim in §1 is written to that scope. If the target platform does not expose IMU/proprioceptive motion, drop body-frame motion from the BeliefEncoder input and replace it with an **auxiliary prediction target** (i.e. BeliefEncoder predicts integrated body-frame motion as a side head trained on simulator labels) so that the encoder still learns motion-aware geometry without depending on it at inference.

**Default for Phase B.** Train BeliefEncoder with body-frame motion as an **auxiliary prediction target** first. Add it as an input only as a registered ablation, and only if removing it causes a retrieval-accuracy gap on the acceptance set.

### 3.5 Frozen-encoder principle: the one allowed exception

LeWM is frozen during Phase A, Phase B, and Phase C. **One** post-hoc allowance: if the full hierarchy meets all Phase B and Phase C acceptance criteria but exhibits a systematic distribution mismatch at end-to-end deployment, a low-learning-rate alignment finetune of LeWM may be run as a separately reported ablation. It is not part of the main method, and its results must be presented alongside (not in place of) the frozen-LeWM result.

---

## 4. Phase A — Diagnostic sequence (the falsifiable gate)

The next block of work is **not** "build the hierarchy". It is "decide whether the hierarchy is necessary." Phase A exists to answer that decision with quantitative criteria registered before the experiments are run.

### 4.1 Phase A0 — Refactor the planner

Extract the existing inference planner into modular components so that future variants (hierarchical planner, learned cost, alternative memory routers, goal adapter) can be implemented as composable modules rather than as branches inside a monolithic script.

Target module contracts:

- `planning/local_mpc.py` — class `LocalMPC` with the contract `(current_state, goal_spec) -> action_block`, where `current_state` bundles `z_history` (the recent window of LeWM latents required for predictor rollout), `action_history` (recent action blocks for warm-start and predictor conditioning), and any belief/memory metadata the cost function needs (e.g. current sub-goal node ID). The interface accepts a state bundle rather than a single latent so that downstream cost variants (sub-goal cost, multi-step rollout) do not require interface changes. Owns warm-start, primitive sampling, kinematic veto, and the MPPI/CEM optimisation loop.
- `planning/costs.py` — pure functions of `(predicted_latent_trajectory, goal_spec) -> scalar`, plus a `cost_mode` selector. No side effects.
- `planning/primitive_bank.py` — construction of the primitive library from the same DSL used at data collection time. Single source of truth.
- `memory/topological_memory.py` — abstract `Memory` interface with `update(observation, action_block)`, `current_belief()`, `select_subgoal(goal)`, and a v2-compatible concrete subclass `KeyframeMemory` that preserves current behaviour.
- `planning/hierarchical_planner.py` — orchestrator with the contract `(observation, goal) -> action_block`. Initially routes through `LocalMPC` with `KeyframeMemory`, behaviourally identical to v2. Becomes the seam for v3.

**Acceptance criterion:** with fixed RNG seeds and deterministic backend flags, the refactored planner produces **identical primitive index sequences** and **numerically equivalent costs to within a fixed tolerance** (e.g. 1e-5 absolute, 1e-4 relative on cost scalars) compared with the pre-refactor planner. Audit JSON must be **semantically identical** after ignoring timestamps, absolute path strings, and non-deterministic ordering fields. The intent is no behavioural change; the test must not be defeated by harmless float-formatting drift.

### 4.2 Phase A1 — Ground-truth cell-graph extraction

For every existing rollout in the data corpus, recover the ground-truth scene graph signals **offline**, without re-rolling out:

- `cell_id` per step (which graph node the agent occupied — interpreted as `(scene_id, cell_id)` via the rollout-level `scene_id`; see §3.4)
- `yaw_bin` per step (discretised heading)
- `local_graph_type` per step (open cell, T-junction, dead-end, etc.)
- `nearest_cell_distance` (Euclidean distance from agent position to nearest cell centre)
- `bfs_distance_to_landmark` per step (graph distance to each named landmark within the rollout's scene)

This is feasible whenever the scene generator is deterministic from a stored seed: regenerate the scene metadata from `scene_seed`, then map the per-step world position into the regenerated graph.

**Current implementation.** `lewm_worlds.labels.derived` and
`scripts/derive_raw_rollout_labels.py` implement this as an offline
post-pass over compact `messages.jsonl` or per-scene rosbag2 MCAP raw
rollouts. The pass rebuilds the scene from `family + topology_seed` when
available, or consumes an explicit manifest, and writes
`derived_labels/labels.jsonl` plus a join summary.

**Why offline matters.** Re-rolling out the corpus is expensive and changes the data distribution if the generator has been touched since. Offline metadata recovery preserves the existing trajectories exactly.

**Acceptance criterion:** the derived-label artifact keys are populated for ≥99% of steps across all rollouts; the remaining <1% are explained by edge cases (agent precisely on a wall, scene seed mismatch) and are flagged with a sentinel value, not silently filled. HDF5/Parquet packaging can consume the JSONL artifact directly; it should not recompute these labels independently.

### 4.3 Phase A2 — Visual-aliasing audit

With cell IDs in hand, characterise how badly the LeWM latent space confuses graph-distinct locations.

Produce:

1. **Latent-distance vs graph-distance scatter.** For each same-scene pair of (state, state) sampled across the held-out scene set, plot **latent distance** (defined as `1 - cosine_similarity` between LeWM latents) against ground-truth BFS graph distance. Compute Spearman ρ. Higher ρ is better. **Convention used throughout this document:** "latent distance" always means `1 - cosine_similarity`; the sign of every reported ρ is consistent with "higher is better".
2. **Nearest-neighbour cell confusion matrix.** For each step in the eval set, find its top-K nearest neighbours in latent space; tabulate how often the nearest neighbour is in the same cell, an adjacent cell, or graph-distant. Break down by scene type and by local graph type.
3. **Aliasing breakdown.** Compute per-scene-type confusion: corridors vs junctions vs dead-ends vs open rooms. The hypothesis is that aliasing is concentrated in corridors and identical junctions.
4. **History-aliasing audit.** For each visually aliased pair, check whether the preceding `H` action blocks and observations also alias. The "history-disambiguable" subset is the set of pairs that are visually similar but whose recent histories differ; this subset is the training signal for BeliefEncoder. Quantify what fraction of visually aliased pairs are history-disambiguable at each `H ∈ {4, 8, 16, 32}`.

**Why this matters.** If aliasing is mild (Spearman ρ ≥ 0.70 between latent distance and graph distance), the belief encoder is not solving a real problem. If aliasing is severe (ρ ≤ 0.40), the belief encoder is necessary. The middle band is where Phase A3 has to make the decision. Item 4 additionally tells you whether BeliefEncoder *could* succeed if built — if very few visually aliased pairs are history-disambiguable at any reasonable `H`, the partial-observability problem is structural and BeliefEncoder cannot fix it.

### 4.4 Phase A3 — Frozen-latent reachability probe

Train a small head on top of frozen LeWM latents that takes `(z_current, z_goal)` and predicts a bucketed graph distance.

**Splits.** Train on the **training** scene split. Evaluate on a **held-out** scene split. No `scene_seed`, generated topology, or cell-graph instance may appear in both splits. This is a generalisation test, not a memorisation test.

**Negatives.** Include false-loop-closure negatives: same-scene pairs of states that look similar in latent space but are graph-distant. Sample these proportionally to their incidence in the training split.

**Buckets (Phase A3).** Five finite buckets: `same/adjacent`, `2-3`, `4-7`, `8-15`, `>15`. The `unknown/unreliable` bucket is **deferred to Phase C** because Phase A3 has no memory and no node-uncertainty source — every same-scene pair has a finite BFS distance, and cross-scene pairs are excluded from this probe. Reintroduce the `unknown/unreliable` bucket in §6.1 when memory-generated nodes provide a real source of uncertainty.

**Pair scoping.** All pairs are within a single scene (same `scene_id`). Cross-scene pairs are excluded from this probe; they have no defined BFS distance.

**Critically:** the input is just LeWM latents, with no temporal history, no belief encoder, no memory. This is the floor: how well can the dynamics latents alone predict reachability.

**Baselines reported alongside.**
- **Trivial baseline.** Predict the majority bucket on the training split.
- **True-cell oracle.** Compute the bucket directly from ground-truth current cell ID and ground-truth goal cell ID using the regenerated scene graph. This is the ceiling assuming perfect localisation.
- **Local-geometry baseline.** Predict from `local_graph_type` of current and goal alone. This isolates how much of the probe's accuracy comes from coarse geometry vs latent identity.

**Acceptance criterion:** report top-1 bucket accuracy, confusion matrix, and Spearman ρ on the held-out split. The decision gate in §4.5 uses these against registered thresholds.

### 4.5 Phase A4 — Pre-registered decision gate

Register before running A2 and A3. Note that the true-cell oracle is approximately 100% by construction, so "within 10% of oracle" effectively means "≥90% bucket accuracy" — this is too strict. The registered thresholds are absolute, not relative to oracle:

- **Strong frozen-latent regime.** Spearman ρ ≥ 0.70 **and** A3 reachability head top-1 bucket accuracy ≥ 70% **and** near/far confusion (i.e. predicting `same/adjacent` when truth is `>15`, or vice versa) ≤ 15%. Conclusion: do not build BeliefEncoder; the frozen latents are adequate. Investigate planner cost, goal-conditioning, or scene difficulty.
- **Insufficient frozen-latent regime.** Spearman ρ < 0.40 **or** A3 reachability head accuracy ≤ trivial baseline + 15 percentage points. Conclusion: build Phase B.
- **Ambiguous regime.** Anything between the two. Run one round of **probe-only diagnostics** that do **not** modify LeWM (LeWM stays frozen until a deliberate decision in §3.5). Allowed diagnostics:
  1. Probe over alternative latent layers / projection heads of the frozen LeWM (raw encoder latent vs predictor-space latent vs early-layer latent) to check whether some layer of the existing encoder already preserves the geometry.
  2. Probe with a **short frozen-latent history** (e.g. the last 4 latents concatenated) to check whether trivial temporal context recovers performance without training BeliefEncoder.
  3. Probe with and without action-block conditioning at the probe input (frozen LeWM only; no encoder change) to isolate whether the deficit is action-blind vs action-aware.
  4. Per-scene-type / per-local-graph-type breakdown to see whether the deficit is uniform or concentrated in specific topologies.

  If any of these moves the metrics into the strong regime, treat as the strong regime. Otherwise, default to building Phase B. Crucially, **none of these diagnostics involve modifying LeWM weights, encoder receptive field, or pretext objective** — those are out of scope for v3 and would constitute a separate work program. The point of the ambiguous-regime diagnostics is to make the decision quickly without prematurely starting an architectural change.

The gate is **registered**, not adaptive. Move the thresholds before running the experiments, not after.

---

## 5. Phase B — Belief and memory (conditional on A4)

Build only if Phase A4 returns "insufficient" or "ambiguous → diagnostics fail". Each component below has a defined contract, training procedure, and acceptance test.

Before training any of these heads, audit the v2 corpus against §8.3 coverage requirements (trajectory diversity, aliasing pairs, goal-image diversity). If coverage is inadequate, collect a targeted v3 corpus before continuing — do not let dataset deficiency masquerade as architectural failure.

### 5.1 BeliefEncoder

**Input.** A short window of `H` consecutive LeWM latents (`H` is a hyperparameter, expected range 8-32 blocks) plus the action blocks that drove transitions between them. Body-frame integrated motion is an **auxiliary prediction target** by default; it becomes an input only as a registered ablation (see §3.4).

**Output.** Two embeddings via a projection-head pattern from the contrastive / self-supervised representation literature:
- `z_proj` (unnormalised, dimension `d_proj` ≈ 256-512) — used **only** at training time for the contrastive and anti-collapse losses.
- `z_retrieval` (L2-normalised, dimension `d_retrieval` ≈ 128-256) — used **only** at deployment for memory key lookup.

**Why two outputs.** Contrastive losses and VICReg-family anti-collapse losses behave poorly on L2-normalised embeddings because normalisation discards the variance signal those losses depend on. The standard remedy (used by SimCLR, VICReg, and related methods) is to compute the loss on a projection head whose output is unnormalised, and use a normalised retrieval embedding (typically the layer below the projection head, also normalised) for downstream lookup. Both come from the same encoder; only the heads differ.

**Loss.** Three components combined:

1. **Masked supervised contrastive (`L_supcon`).** The primary objective. Pair targets are scoped to `(scene_id, cell_id)` (see §3.4):
   - **Strong positive.** Same `(scene_id, cell_id)` and similar `yaw_bin`. Pull together with full weight.
   - **Weak positive.** Same `(scene_id, cell_id)` but different `yaw_bin` (same place, different approach heading). Pull together with reduced weight `λ_yaw_weak ∈ [0.3, 0.7]`. **Why a weak positive, not a negative:** treating same-place-different-yaw as negative would fragment the memory at every junction by approach direction, since the topological filter would commit a separate node per heading. Treating it as a strong positive would discard yaw information entirely. The weak-positive compromise keeps places clustered while preserving orientation information in the embedding's variance.
   - **Valid hard negative.** Different `(scene_id, cell_id)` with graph distance ≥ 2 (within the same `scene_id`), or any pair across different `scene_id`s. Includes visually aliased pairs whose **histories differ** under window `H` (the history-disambiguable set from §4.3 item 4). Push apart.
   - **Ambiguous-ignore.** Adjacent / boundary cells within the same scene, **and** pairs that are both visually aliased and history-aliased under `H`. Mask out of the loss; do not push or pull.

   Yaw handling is a **registered design choice**: report `λ_yaw_weak` and an ablation that drops weak positives entirely. If memory fragmentation by approach heading is observed at deployment, increase `λ_yaw_weak`; if junction-disambiguation degrades, decrease it.
2. **VICReg variance hinge (`L_var`).** Each dimension of `z_proj` is required to have batch standard deviation above a threshold (the standard VICReg variance term).
3. **VICReg covariance decorrelation (`L_cov`).** Off-diagonal covariance of `z_proj` is penalised toward zero.
4. **Body-frame motion auxiliary (`L_motion`)** *(when body-frame motion is not used as input).* Predict `(∫v_body_x dt, ∫v_body_y dt, Δyaw)` over the input window from the encoder output, regressed against ground-truth proprioception integrated over the same window. Forces the embedding to be motion-aware.

Total: `L_belief = L_supcon + λ_var L_var + λ_cov L_cov + λ_motion L_motion`. VICReg on its own does not push specific negative pairs apart — it is the anti-collapse companion to the supervised contrastive term, not a substitute for it.

**Why supervised contrastive plus VICReg, not SIGReg.** SIGReg (Cramér-Wold + Epps-Pulley normality) is appropriate for **predictive** latents where the predictor benefits from Gaussian-shaped target distributions. For **retrieval** embeddings, the relevant property is anti-collapse and decorrelation, which VICReg targets directly. SIGReg does not provide guarantees about retrieval geometry.

**Why the alias mask carves out history-aliased pairs only.** If two states are visually aliased *and their histories differ*, BeliefEncoder should be able to distinguish them — these are exactly the cases the history window exists to solve, and they are the most informative hard negatives. Excluding all visually aliased negatives would discard precisely the training signal the encoder needs. Excluding only visually-aliased-AND-history-aliased pairs (genuine partial observability under window `H`) keeps the hard negatives in and removes the impossible ones.

**Cross-scene negative capping.** Cross-scene negatives are valid for preventing accidental global cell-ID conflation (without them, two structurally similar cells from different procedural scenes could collapse to the same retrieval embedding), but they are typically too easy: scenes differ in coarse visual statistics, so cross-scene pairs are pushed apart by features unrelated to the topology BeliefEncoder is supposed to learn. They should therefore be **capped or downweighted** so that same-scene graph-distinct hard negatives dominate the retrieval geometry. Default scheme: per batch, sample same-scene hard negatives first up to a target count, fill the remainder with cross-scene negatives, and apply a weight `λ_cross_scene ∈ [0.1, 0.3]` to the cross-scene contributions in `L_supcon`. Report the ratio of same-scene to cross-scene negatives used in each batch as a training-time diagnostic. The same scheme applies to GoalAdapter training in §5.2 (cross-scene goal-belief pairs follow the same cap and weight).

### 5.2 GoalEncoder / GoalAdapter

**Why this module exists.** BeliefEncoder consumes a window of `H` latents and action blocks. A single static goal image is **not** from the same distribution. Passing a goal image through BeliefEncoder by either (a) padding with zero-action history or (b) duplicating the goal frame `H` times produces an embedding that lives in a different region of `z_retrieval` space than any belief embedding the agent ever produces during a rollout. This is a silent train/deploy distribution mismatch that would corrupt every Level-1 routing decision.

**Module.** A small adapter that maps a single LeWM latent into the same retrieval space:

```
GoalAdapter: z_LeWM(goal_image) -> z_goal_retrieval ∈ same space as BeliefEncoder.z_retrieval
```

**Training.** Pair each goal image (taken from any rollout's terminal frame, or any frame designated as a candidate goal) with belief windows from rollouts whose terminal cell matches the goal's cell. Treat as a supervised contrastive problem:
- Strong positive: goal image and belief window with matching terminal `(scene_id, cell_id)` and similar `yaw_bin`.
- Weak positive: same `(scene_id, cell_id)` but different `yaw_bin` (same place from a different approach), with reduced weight per the §5.1 yaw scheme.
- Valid hard negative: different `(scene_id, cell_id)` with graph distance ≥ 2 within the same `scene_id`, or any pair across different `scene_id`s.
- Ambiguous-ignore: as in §5.1.

The GoalAdapter is trained on top of frozen BeliefEncoder. Its output must be in the same metric space as `z_retrieval` so that cosine similarity between `z_goal_retrieval` and any node's `z_retrieval` is meaningful.

**Goal-match confidence head.** In addition to the embedding, train a small classifier `GoalMatchHead(z_goal_retrieval, node.z_retrieval) -> probability of same place`. This is structurally similar to LoopClosureHead but conditioned on the goal-vs-belief asymmetry. It is used at planning time to decide whether the agent has actually arrived at the goal and to gate confident vs unconfident matches between goal embedding and memory nodes.

**Offline acceptance criterion (used for training/release gating).** Given a held-out scene set where every belief window has a known `(scene_id, cell_id)`:

- **Query:** a single goal image whose ground-truth `(scene_id, cell_id)` is known.
- **Database:** held-out belief windows from the same scene set.
- **Metric:** Recall@5 for retrieving belief windows whose terminal `(scene_id, cell_id)` matches the goal's.
- **Bar:** must exceed a frozen-LeWM baseline (cosine similarity between goal `z_LeWM` and rollout `z_LeWM` per step) by a registered margin (e.g. ≥15 percentage points).

The point is to verify that the adapter actually projects useful geometry, not just that it produces an embedding. This is an **offline** retrieval test against a static database of held-out windows; it does not assume any online memory exists.

**Online use (separate concern, not part of acceptance).** At inference, the planner queries `GoalMatchHead(z_goal_retrieval, n.z_retrieval)` against memory nodes built up *during* the current episode. If the goal location has never been visited in the current episode, no memory node will match it; that case is handled by the planner's exploration mode (§6.2 Level 1). Online behaviour is evaluated end-to-end in Phase C, not in the GoalAdapter offline acceptance test.

### 5.3 LoopClosureHead

**Input.** A pair of belief embeddings `(z_retrieval_a, z_retrieval_b)` plus optionally the body-frame integrated motion between them (or a small action history if body-frame motion is unavailable).

**Output.** A scalar probability that the two states correspond to the same memory node.

**Loss.** Binary cross-entropy with the same three-bucket label scheme as BeliefEncoder, treating ambiguous pairs as masked.

**Calibration.** After training, run isotonic or Platt scaling on a held-out set so that the output probability is well-calibrated. The downstream filter uses the probability directly as an observation likelihood, so calibration matters more than discriminative score.

**Acceptance criterion (precision-first).** A false loop closure corrupts the memory graph; a missed loop closure merely creates a duplicate node. Therefore precision is the primary gate, not F1.

- **Primary:** loop-closure **precision ≥ 99%** at the deployment threshold on held-out scenes.
- **Primary:** ECE ≤ 5% after calibration.
- **Reported but not gated:** recall at the same threshold, F1, threshold-precision and threshold-recall curves.

### 5.4 TopologicalMemory and the top-k filter

**Memory data model.**
- `MemoryNode`: fields `node_id`, `z_retrieval` (the committed embedding), `creation_step`, `n_visits`, `representative_observation` (image kept for sub-goal pursuit and visualisation), optional metadata.
- `MemoryEdge`: fields `(node_from, node_to)`, `transition_count` (posterior-weighted), `action_summary` (running mean of action blocks that produced this transition), `traversal_belief_change` (mean Δ in belief embedding across this transition), `last_traversed_step`.

**Top-k discrete Bayes filter.**

State: a posterior over the current memory node, represented as a top-k list of `(node_id, weight)` pairs with weights summing to 1.

Per-step update:

1. **Prediction.** For each hypothesis `(n_i, w_i)` in the current posterior, distribute its weight across outgoing edges of `n_i` according to a kernel that mixes (a) the empirical edge transition probability `p(n_j | n_i)` and (b) an action-edge compatibility term `exp(cosine(action_block, edge.action_summary) / τ)` with `τ ∈ [0.2, 0.5]`. Normalise.
2. **Observation update.** For each predicted hypothesis, compute the LoopClosureHead probability `p(observe | n_j) = LoopClosure(z_retrieval_now, n_j.z_retrieval)`. Multiply weights by these likelihoods.
3. **Top-k truncation.** Keep the `k` highest-weight hypotheses (`k` typically 5-10). Renormalise.
4. **Branch death.** Drop any hypothesis whose weight stays below `0.02` for `M` consecutive steps (`M ≈ 5-10`).

**Three named thresholds (no symbol overloading).**
- `τ_new = 0.70` — **new-node proposal.** If `max LoopClosure(z_retrieval_now, n.z_retrieval) over all existing memory nodes n` (not only the predicted top-k) `< τ_new` for `N` consecutive steps, propose a new node and commit it (the committed `z_retrieval` is the running mean of the past `N` belief embeddings). **Critically: the maximum is taken over the entire memory, not the prediction-step top-k.** If novelty were checked against the top-k only, a wrong prediction step (e.g. a transient transition prior pointing the wrong way) would create a duplicate node for a place that already exists somewhere else in memory but didn't survive into the current top-k. Global novelty checking prevents that failure mode at trivial extra cost (one cosine + LoopClosure call per existing node).
- `τ_commit_high = 0.95` — **commit-to-existing.** If a single hypothesis's posterior weight exceeds `τ_commit_high` for `N` consecutive steps, lock onto that node as the unique current location.
- `τ_commit_low = 0.70` — **relapse to multi-hypothesis.** If a locked node's posterior falls below `τ_commit_low`, return to top-k mode.

The gap between `τ_commit_high` (lock in) and `τ_commit_low` (relapse) is the standard control-engineering hysteresis remedy for boundary flicker.

**Edge update on transitions.** Edge counts and summaries are updated using **posterior-weighted contributions across consecutive steps**, not by the moment a lock changes:

- For consecutive posteriors `P_{t-1}` and `P_t`, every pair `(n_i, n_j)` contributes weight `P_{t-1}(n_i) × P_t(n_j)` to `transition_count(n_i, n_j)`.
- `action_summary(n_i, n_j)` updates as a running mean of the executed action block, weighted by `P_{t-1}(n_i) × P_t(n_j)`.
- `traversal_belief_change(n_i, n_j)` updates similarly with the per-step Δ in `z_retrieval`.

This formulation handles uncertainty correctly under top-k beliefs and avoids the previous (incorrect) phrasing about "outer products over action-block dimensions" — the outer product is across **node identities**, not action dimensions.

**Why top-k, not particle filter.** With k = 5-10 and a discrete graph, the full top-k is computationally negligible and avoids the resampling artefacts of a true particle filter. The "particles" here are graph nodes, not continuous states; each has integer identity.

**Why hysteresis, not single threshold.** A single threshold induces flicker around boundary states (the agent's belief oscillates between two adjacent cells, and the memory commits and uncommits the same node repeatedly). Hysteresis is the standard control-engineering remedy.

### 5.5 Acceptance criteria for Phase B

These replace the earlier single "cell ID retrieval ≥ 80%" gate, which would have been over-simplified for an aliased environment.

> **Forward-reference.** `node_majority_label` is defined formally in §6.1 as the majority `(scene_id, cell_id)` assigned to a memory node during replay over training rollouts; a node is **reliable** when its `node_purity` exceeds `τ_purity` (default 0.8). The criteria below use that definition.

- **Same-place Recall@5** on held-out scenes improves over the frozen-latent baseline (from §4.3) by ≥ 15 percentage points.
- **Graph-distance rank correlation** (Spearman ρ between `z_retrieval` distance, defined as `1 - cosine_similarity`, and graph distance) improves over frozen-latent baseline by ≥ 0.15.
- **Exact cell retrieval top-1**, computed only on **non-boundary, non-ambiguous** frames (those not in the §5.1 ambiguous-ignore set), ≥ 80%.
- **LoopClosureHead** precision ≥ 99%, ECE ≤ 5% per §5.3.
- **GoalAdapter Recall@5** improves over frozen-latent goal baseline by ≥ 15 percentage points per §5.2.
- **Filter trajectory coherence:** on a held-out trajectory, the locked-in node sequence matches the ground-truth `(scene_id, cell_id)` sequence ≥ 90% of the time on non-boundary frames, after mapping each memory node to its `node_majority_label` (§6.1). Duplicate nodes for the same true cell (fragmentation) are **not** counted as localisation errors as long as they all map to the same `node_majority_label`. Only false merges (a single node mapped to multiple true cells, or `node_purity < τ_purity`) and outright wrong locks count against coherence. Report fragmentation rate (`#reliable_nodes / #unique_true_cells_visited`) separately as a memory-quality diagnostic, not as part of the coherence gate. This reflects the navigation-relevant ordering: false merges are fatal to routing; fragmentation is at worst a minor inefficiency.

If any of these fail, do not proceed to Phase C; debug Phase B first. If `Same-place Recall@5` does not improve over the frozen-latent baseline by the registered margin, H2 is **falsified** and v3's main path is over: the bottleneck is at a deeper representational level (see §2).

---

## 6. Phase C — Reachability and hierarchical planning (conditional)

### 6.1 ReachabilityHead

**Critical training/deployment alignment.** Train ReachabilityHead on **memory-generated node pairs**, not on raw rollout pairs. The pairs that ReachabilityHead sees at deployment time are pairs of committed memory nodes; training on raw pairs creates a distribution mismatch that silently degrades deployment performance.

**Input.** `(z_retrieval_current, z_retrieval_goal)` plus optional auxiliary features (memory edge density around current node, time since last commit, etc.).

**Output.** Bucketed graph distance: `same/adjacent`, `2-3`, `4-7`, `8-15`, `>15`, `unknown/unreliable`. The "unknown/unreliable" bucket is **explicit**, not a fallback. The head is trained to predict it on out-of-distribution pairs (e.g. nodes from disjoint memory components).

**Loss.** Cross-entropy over the 6-way categorical, with class-balanced weighting.

**False-loop-closure negatives.** Construct training pairs where two memory nodes have high LoopClosureHead score but are graph-distant (i.e. visual aliases). These are the hardest cases and must be present in training, or the head will fail in exactly the conditions where it matters.

**Memory-node label assignment (handles fragmentation and false merges correctly).** During training, the topological memory is run on training rollouts and each memory node accumulates a set of frames that contributed to it. Assign a training label to each node as follows:

```
node_cell_distribution[n] = histogram over (scene_id, cell_id) of the frames assigned to n
node_purity[n]            = max probability in node_cell_distribution[n]
node_majority_label[n]    = argmax (scene_id, cell_id) of node_cell_distribution[n]
```

Node purity classification (with registered threshold `τ_purity`, default 0.8):

- `node_purity[n] ≥ τ_purity` → node `n` is **reliable**; its label is `node_majority_label[n]`.
- `node_purity[n] < τ_purity` → node `n` is **unreliable** (a false merge across multiple true cells, a boundary-state node, or other corrupted aggregate). It has no clean cell label.

Pair-level reachability targets:

- Both nodes reliable, same `scene_id`: label = BFS bucket between `node_majority_label[n_a]` and `node_majority_label[n_b]` in that scene's graph.
- Either node unreliable, **or** different `scene_id`s: label = `unknown/unreliable`.

**Why this matters.** Without an explicit purity rule, ReachabilityHead is silently trained on corrupted pairs (a node that aggregates two distinct true cells gets assigned one BFS distance, contradicting itself). The purity rule routes those pairs to `unknown/unreliable` rather than feeding the head garbage. It also turns memory-quality problems into a measurable training-set characteristic: report `% of memory nodes reliable` and `% of training pairs labeled unknown` as memory-quality diagnostics. Fragmentation (one true cell split into multiple reliable nodes) is harmless to this rule; only false merges are filtered out.

**Bucket-distribution audit (mandatory before training).** Before training any reachability probe or head — both the Phase A3 frozen-latent probe (§4.4) and the Phase C ReachabilityHead — report the bucket distribution for the train and held-out splits. If any finite bucket has fewer than a registered minimum number of examples (default: ≥ 500 train, ≥ 100 held-out per bucket), merge adjacent buckets or re-register bucket boundaries before training. Severely unbalanced buckets produce a head that is silently a near-majority-class predictor; class-balanced weighting alone does not fix this when a bucket is below the minimum-evidence floor. Report the per-split histogram in the same artefact as the head's accuracy results.

**Cache key.** Reachability training is the most expensive step; cache aggressively. The cache key must include **all** of:
- LeWM weights hash
- BeliefEncoder weights hash + projection-head config
- GoalAdapter weights hash
- LoopClosureHead weights hash + calibration parameters + `τ_new`, `τ_commit_high`, `τ_commit_low`
- Memory filter parameters (`k`, `M`, branch-death weight threshold, action-edge temperature)
- Scene split seed
- Cache version integer

A stale cache that omits any of the above will produce silent training-deployment mismatches that are essentially impossible to debug from the training loss alone.

### 6.2 Hierarchical planner

Three-level decision loop:

**Level 1 — Memory routing and goal matching.** Given current top-k memory belief and a goal image:

1. Embed the goal image with **GoalAdapter**, producing `z_goal_retrieval`. (Do **not** pass the goal image through BeliefEncoder.)
2. For each memory node `n`, compute `GoalMatchHead(z_goal_retrieval, n.z_retrieval)`. Find the highest-scoring node and call it `n_goal_candidate`; let `p_match` be its score.
3. **Goal-match decision.**
   - If `p_match ≥ τ_goal_high` (e.g. 0.95): treat `n_goal_candidate` as the goal node. Plan through the memory graph toward it.
   - If `p_match < τ_goal_low` (e.g. 0.70): the goal is not confidently in memory. Enter **exploration mode**: pick a frontier sub-goal (least-visited node within reachable bucket; or unexplored direction inferred from the memory graph) and route there. Periodically re-check `GoalMatchHead` against new observations.
   - In between: continue current sub-goal but increase the rate of frontier checking.
4. **Reachability routing (when in plan mode).** For each top-k current hypothesis `n_i`, query `ReachabilityHead(n_i, n_goal_candidate)` and weight by hypothesis weight. The expected bucket is the routing distance.
5. If routing distance is `same/adjacent`, drop into Level 2 with `n_goal_candidate` as the sub-goal.
6. If routing distance is `>15` or `unknown/unreliable`, fall back to exploration mode regardless of `p_match`.
7. Otherwise, pick the next sub-goal node along the shortest path in the memory graph from the locked-in current node to `n_goal_candidate`.

**Level 2 — Sub-goal pursuit.** Given a sub-goal memory node:

1. Use the sub-goal's `representative_observation` as the goal image for the local cost. **Critically:** Level 3 chases the **sub-goal's** image, not the final destination image. This is what prevents the v2 failure mode in which long-horizon goals dominate the local cost and cause the agent to ignore the corridor it is currently in.
2. Hand off to LocalMPC with that sub-goal image.
3. **Sub-goal success (probabilistic).** Declare local sub-goal success when **either** (a) the posterior probability of the sub-goal node `P(sub_goal_node)` exceeds `τ_subgoal_success` (e.g. 0.85) for `N_success` consecutive steps, **or** (b) the locked-in node equals the sub-goal node under the §5.4 hysteresis rule. Condition (a) lets the planner advance under top-k mode without waiting for a hard lock; condition (b) is the legacy clean case. Either triggers return to Level 1 for the next sub-goal.
4. Plateau detection: if `P(sub_goal_node)` does not increase by a registered margin within a budget of action blocks, escalate back to Level 1 (re-route or explore).

**Level 3 — LocalMPC.** Unchanged from v2 in structure: warm-started CEM over the primitive bank with kinematic safety veto. The cost function inside LocalMPC is the goal-image cost evaluated against the **current sub-goal's representative observation**. ReachabilityHead operates only at Levels 1 and 2.

### 6.3 Hybrid safety framing

Do not replace the kinematic safety veto with a learned cost. Keep the veto as a hard constraint over candidate trajectories, and use the learned components (BeliefEncoder, GoalAdapter, LoopClosure, Reachability, sub-goal-image cost) only to rank trajectories that pass the veto. This decoupling makes failure modes legible: if the agent stops moving, you can tell whether the veto is excluding all candidates or whether the cost is degenerate.

### 6.4 Perceptual success detection

The agent must decide on its own when it has reached the final goal — there is no privileged success signal at deployment. Final-goal success is declared when **either**:

- `GoalMatchHead(z_goal_retrieval, z_retrieval_now) ≥ τ_goal_arrived` for `N_arrived` consecutive steps, **or**
- the locked-in current node equals `n_goal_candidate` and `GoalMatchHead(z_goal_retrieval, n_goal_candidate.z_retrieval) ≥ τ_goal_arrived`.

`τ_goal_arrived` is registered (default 0.95) and may differ from `τ_goal_high` used for routing in §6.2 — arrival is a higher bar than committing-to-route, because a false arrival ends the episode.

**Privileged signals at evaluation, not at deployment.** Ground-truth `(scene_id, cell_id)` of the agent and goal are logged for evaluation purposes (reporting success rate, path-length efficiency, and per-level diagnostics in §9.3). They do not enter the success-detection or stop-condition logic.

This closes the loop on the §1 thesis claim "without a privileged deployment-time success signal": the agent stops on its own perceptual judgment; the experimenter only sees ground truth in the evaluation log.

---

## 7. Training regime

### 7.1 LeWM (pre-existing)

Frozen for the entirety of v3's main path. See §3.5 for the one allowed post-hoc alignment exception.

### 7.2 BeliefEncoder

- Single-stage training on rollout corpus with three-bucket pair sampling and the history-aware alias mask (§5.1).
- Frozen LeWM as the per-frame backbone.
- Loss: masked supervised contrastive + VICReg variance/covariance + body-frame motion auxiliary (when motion is not used as input).
- Track each loss term separately. Validate by retrieval accuracy on a held-out scene set, not by total loss value.

### 7.3 GoalAdapter

- Trained on top of frozen BeliefEncoder.
- Supervised contrastive between single-image goal latents and belief-window embeddings, with the same ambiguous-ignore mask.
- GoalMatchHead trained jointly or in a follow-up stage.

### 7.4 LoopClosureHead

- Trained on top of frozen BeliefEncoder.
- BCE on positive/valid-negative pairs with ambiguous masking.
- Calibrated post-hoc with isotonic or Platt scaling on a held-out set.

### 7.5 ReachabilityHead

- Trained on top of frozen BeliefEncoder + GoalAdapter + LoopClosure + memory.
- The training loop runs the topological memory and filter on training rollouts to produce node pairs, then trains the head on those pairs.
- This is **expensive**; cache the (memory state, training-pair) artefacts under the full cache key from §6.1.

### 7.6 Phasing

Train sequentially: BeliefEncoder → GoalAdapter → LoopClosureHead → (memory dry-run on val set) → ReachabilityHead. Do not co-train. Joint optimisation across these stages destroys the diagnostic interpretability of each component.

---

## 8. Data requirements

### 8.1 What v2 data already gives you

- Pixel observations
- Action blocks (commands chunked)
- Body-frame proprioception sufficient to integrate motion
- Scene seed (deterministic regeneration of scene metadata)

### 8.2 What Phase A1 adds (offline)

- Per-step `cell_id`
- Per-step `yaw_bin`
- Per-step `local_graph_type`
- Per-step `nearest_cell_distance`
- Per-step `bfs_distance_to_landmark` (per landmark)

### 8.3 Phase B coverage requirements (audit before training)

These properties are required for Phase B to be trainable. Audit the v2 corpus against them at the start of Phase B; if any fails, collect a targeted v3 corpus before continuing.

- **Trajectory diversity.** Each scene must be visited from multiple starting positions and headings, producing belief windows that overlap on the same cells but with different recent histories. Without this, BeliefEncoder cannot learn invariance to history. **Audit:** for each cell visited in the corpus, count distinct (entry-direction, recent-history-cluster) combinations. Require a registered minimum (e.g. ≥ 4 per cell on average).
- **Aliasing pairs.** The corpus must contain visually similar locations from graph-distinct cells, and the **history-disambiguable** subset (§4.3 item 4) must be non-trivial. **Audit:** report the §4.3 item 4 statistics; require a minimum fraction of visually aliased pairs to be history-disambiguable at `H = 16`.
- **Goal-image diversity.** Each goal location should appear in the corpus as both an observation (during some rollout) and as a candidate goal across multiple scenes. **Audit:** for the goal-set used in evaluation, verify that each goal cell has ≥ N belief windows in training that terminate at that cell with diverse approach directions.

If any audit fails, the dataset is the bottleneck, not the architecture. Collect more data with the appropriate diversification before attributing failure to the model.

### 8.4 What v3 does **not** need

- Adaptive exploration policy. v3 trains on the fixed data distribution (v2 corpus or targeted v3 corpus per §8.3). If that distribution is inadequate after the §8.3 audit, that is a data-collection problem, not a Phase B/C debug.
- Reinforcement-learning reward labels.
- Privileged pose at deployment.

---

## 9. Evaluation metrics

### 9.1 Phase A metrics

- Spearman ρ between latent distance (`1 - cosine_similarity`) and graph distance, per scene type.
- Top-K nearest-neighbour cell confusion, per scene type and per local graph type.
- History-disambiguable fraction at multiple `H` (§4.3 item 4).
- Frozen-latent reachability probe top-1 bucket accuracy, confusion matrix, near/far confusion rate.

### 9.2 Phase B metrics

- BeliefEncoder Same-place Recall@K on held-out scenes.
- BeliefEncoder graph-distance rank correlation.
- BeliefEncoder exact cell top-1 on non-boundary frames.
- GoalAdapter Recall@K vs frozen-latent goal baseline.
- LoopClosureHead precision at deployment threshold + recall + ECE.
- Filter trajectory coherence (locked-in node vs ground-truth cell, with allowance for boundary frames).

### 9.3 Phase C metrics

- ReachabilityHead bucket accuracy on memory-generated pairs.
- ReachabilityHead reliability of the "unknown/unreliable" bucket on out-of-distribution pairs.
- End-to-end navigation success rate, broken down by scene difficulty (graph diameter, aliasing density, dead-end count).
- Path-length efficiency: ratio of agent path length to ground-truth shortest path.
- Per-level diagnostics: Level-1 routing accuracy (chosen sub-goal lies on the true shortest path), Level-2 sub-goal-success rate, Level-3 local-MPC success rate. The bottleneck level is the one with the lowest accuracy.

### 9.4 Comparison baselines

The oracle baselines are defined as a **nested ladder**, each strictly stronger than the previous, so that consecutive gaps causally attribute capacity to a single sub-system. Treat the ladder as a specification; you do not need to run all five rungs simultaneously, but the decomposition is only valid if higher rungs are supersets of lower rungs.

- **Baseline 0 — v2 keyframe-cosine planner.** Unmodified pre-v3 system; the regression gate. Reported alongside the ladder but not nested into it.
- **Baseline 1 — v3 full system (learned everything).** BeliefEncoder, GoalAdapter, LoopClosureHead, topological memory + filter, ReachabilityHead, and LocalMPC are all learned and run end-to-end. No privileged signals at deployment.
- **Baseline 2 — Oracle localisation.** Replace the topological-memory filter with the true current `(scene_id, cell_id)`. The agent still uses **learned** ReachabilityHead, **learned** sub-goal routing, and **learned** LocalMPC. The gap from Baseline 1 to Baseline 2 isolates the cost of imperfect localisation.
- **Baseline 3 — Oracle localisation + oracle graph/reachability.** In addition to Baseline 2, replace ReachabilityHead with the true scene-graph BFS distance buckets between memory-equivalent nodes. Sub-goal routing and LocalMPC remain learned. The gap from Baseline 2 to Baseline 3 isolates the cost of learned reachability prediction given perfect localisation.
- **Baseline 4 — Oracle localisation + oracle graph + oracle subgoal routing.** In addition to Baseline 3, replace the Level-1/Level-2 sub-goal selection with the optimal next sub-goal along the true shortest path in the true graph. Only LocalMPC remains learned. The gap from Baseline 3 to Baseline 4 isolates the cost of learned routing given perfect localisation and perfect reachability.
- **Baseline 5 — Oracle everything except local execution.** Same as Baseline 4 but additionally provides LocalMPC with the true relative pose to the next sub-goal (when that information would be observable to a perfect planner). Only the primitive-level execution is left to the learned controller. The gap from Baseline 4 to Baseline 5 isolates the cost of imperfect local-MPC cost shaping given perfect everything upstream. The gap from Baseline 5 to 100% is the floor imposed by the primitive bank, kinematic veto, and CEM optimisation itself — improving anything upstream cannot beat this.

**Decomposition narrative.** Because each rung is a strict relaxation of the rung below, consecutive gaps causally attribute capacity to one sub-system at a time: Δ(0→1) is the v3-vs-v2 effect; Δ(1→2) is the cost of imperfect localisation; Δ(2→3) is the cost of imperfect reachability; Δ(3→4) is the cost of imperfect routing; Δ(4→5) is the cost of imperfect local-MPC cost; Δ(5→100%) is the structural floor of the primitive bank and CEM. Reporting non-nested oracle conditions (e.g. an "oracle graph" without "oracle localisation") confounds these effects and is **not** a substitute for the nested ladder.

- **v3 ablations** (reported separately from the ladder): no LoopClosureHead (use cosine threshold); no hysteresis; no false-loop negatives; single-hypothesis filter (k=1); no GoalAdapter (goal image passed through BeliefEncoder with zero-action history); no body-frame motion auxiliary; no global-novelty check (novelty over predicted top-k only); no node-purity filter (reachability trained on all pairs including impure ones); reachability trained on raw rollout pairs instead of memory-generated pairs.

---

## 10. Risks and contingencies

### 10.1 Phase A returns "strong frozen-latent regime"

The frozen latents are adequate. The v2 navigation failure is somewhere else: planner cost, goal conditioning, action-block granularity, or scene difficulty. **Do not** rationalise into building Phase B anyway. Investigate the specific axis indicated by the diagnostic and report findings.

### 10.2 Phase A returns "ambiguous"

Run the **probe-only diagnostics** from §4.5: alternative frozen latent layers, short frozen-latent history, probe-side action conditioning, and per-scene-type / per-local-graph-type breakdown. Do **not** modify LeWM weights, encoder receptive field, or pretext objective during Phase A — those are out of scope for the v3 main path (§3.5). If these diagnostics move the metrics into the strong frozen-latent regime, treat as case 10.1. Otherwise default to building Phase B.

### 10.3 BeliefEncoder fails the Phase B Same-place Recall@5 acceptance criterion

H2 is falsified at the registered margin. Before declaring v3 dead, check in this order: (a) §8.3 dataset audit results — was coverage actually adequate? (b) Window length `H` — try longer history. (c) Ambiguous-mask boundary — relax the boundary definition. (d) Auxiliary supervision on `local_graph_type` as a multi-task head. If none of these recover, the bottleneck is at a deeper representational level (predictor action-conditioning, encoder receptive field, pretext objective) and v3's main path stops here.

### 10.4 GoalAdapter fails its acceptance criterion

The goal image distribution differs too much from belief-window terminal embeddings. Try: (a) mining hard goal-belief pairs explicitly; (b) including a small action-history token on the goal side (treating "goal observed at end of trajectory" as a degenerate history); (c) training GoalAdapter and BeliefEncoder jointly with a shared retrieval head, rather than in sequence.

### 10.5 LoopClosureHead is uncalibrated

Re-run isotonic regression on a larger held-out set. If still uncalibrated, the head's input features are insufficient: add body-frame integrated motion or a small action history as explicit input.

### 10.6 LoopClosureHead cannot reach 99% precision

The retrieval embedding is not separable enough; do not deploy. Either revisit BeliefEncoder (longer `H`, better negatives) or accept a more conservative deployment threshold and report degraded recall — but do **not** lower the precision target, since it directly governs memory-graph integrity.

### 10.7 Filter oscillates despite hysteresis

The threshold gap is too small. Widen `τ_commit_low`/`τ_commit_high` and re-run. If oscillation persists in specific scene topologies (long symmetric corridors), add a boredom term: penalise revisits to nodes already in the top-k.

### 10.8 ReachabilityHead is dominated by the "unknown/unreliable" bucket

The training distribution does not cover the deployment distribution. Increase memory dry-run coverage in training, and verify that the cache key (§6.1) is invalidating correctly.

### 10.9 End-to-end success rate does not improve over v2

This is a real possibility and the most useful negative outcome. The hierarchical planner adds capacity to **route** through known graph structure, but if the underlying difficulty is **local** (the agent cannot execute a single sub-goal traversal reliably), routing capacity does not help. Diagnose by separately reporting Level-1 routing accuracy, Level-2 sub-goal success rate, and Level-3 local-MPC success rate. The level whose accuracy is lowest is the bottleneck.

---

## 11. Portability notes (for simulator/robot transition)

When porting this plan to a new simulator and robot:

1. **Rederive the action-block representation** from scratch. Block size `K`, command dimensionality, and the primitive bank are all robot-specific. The plan's structure does not depend on their values; only the LocalMPC implementation does.
2. **Rederive the scene generator's metadata interface.** You need a function `(scene_seed) -> (cell_centres, adjacency, local_graph_types, beacon/landmark positions, world_bounds)`. In this repo that interface is `SceneManifest` + `SceneGraph` + `local_graph_type_per_node`; without an equivalent you cannot run Phase A1, and without Phase A1 the entire Phase A diagnostic gate is uncomputable.
3. **Re-train LeWM on the new pixel distribution.** The frozen-encoder principle assumes LeWM has been trained on rollouts from the new simulator/robot. Using a LeWM trained on a different simulator is a domain-transfer experiment, which is out of scope for v3.
4. **Reverify body-frame motion availability.** If the new platform does not expose IMU-equivalent integrated body-frame velocity cleanly, drop body-frame motion as a BeliefEncoder input and keep it as an auxiliary prediction target only (per §3.4 default). The thesis claim in §1 then reads as "egocentric perception and action history" without the proprioceptive-motion qualifier.
5. **The privileged-leak rule (§3.4) carries over unchanged.** Whatever signals the new simulator exposes, they remain labels-only at deployment.
6. **The phased acceptance criteria carry over unchanged in structure**, but the absolute thresholds (Spearman ρ ≥ 0.70, Recall@5 margins, etc.) may need re-calibration: a more visually distinct simulator may push the floor up; a more aliased one may push it down. The decision-gate **structure** is portable; the **numbers** should be re-registered against the new platform's frozen-latent baseline before running the gate.
7. **Re-audit dataset coverage (§8.3) against the new platform.** Coverage requirements are properties of the corpus, not the simulator; verify each before assuming v2-corpus-style data is adequate.

---

## 12. Reading-order summary

For a collaborator picking this up cold:

1. §1-§3 to understand thesis claim, hypotheses, and design principles.
2. §4 (Phase A) to understand the diagnostic gate that decides whether the rest of the plan runs.
3. §5 (Phase B) and §6 (Phase C) only after agreeing with Phase A's outcome. Pay particular attention to §5.2 (GoalAdapter) — it is the module whose absence would silently break Phase C routing.
4. §10 for what to do when each component fails.
5. §11 for porting concerns.

The shortest summary of the plan: **build a falsifiable diagnostic before building a hierarchy. If the diagnostic says the hierarchy is unnecessary, don't build it.**
