# LeWMQuad-v3 — Flat LeWM Capabilities, Limits & Roadmap (2026-06-04)

## How to read this

This consolidates the state diagnostic and the Phase A decision gate (A2 + A3)
into a capability-and-roadmap view: what the current flat LeWM **can** do, what
it **cannot**, why, what to do about it, and what it implies for the H-JEPA
hierarchy. Every key point has a **Plain:** line in non-specialist language.

**Later evidence, 2026-06-06:** the scaled SIGReg/source-mix ablation did not
restore metric geometry, while rollout-objective/data scaling substantially
improved forward prediction. Treat the SIGReg root-cause language below as the
hypothesis at this document's 2026-06-04 snapshot, not the current decision.
Current representation evidence and the controlled pose-aux plan are in
`docs/lewm_nav_cost_diagnosis_2026-06-06.md` and
`docs/lewm_pose_aux_experiment_design_2026-06-06.md`.

- Detailed chronological evidence + tables: `docs/lewm_state_diagnostic_2026-06-04.md` §1–§8.
- Registered plan being tested: `docs/v3_topological_nav_plan.md`.
- Probe scripts: `scripts/probe_lewm_latent_aliasing.py` (A2),
  `scripts/probe_lewm_reachability_a3.py` (A3),
  `scripts/probe_lewm_rollout_horizons.py`, `scripts/probe_lewm_receding_mpc_proxy.py`,
  `scripts/benchmark_lewm_closed_loop_mpc.py`.
- Checkpoint evaluated throughout: `lewm_seq4_e9_b050000.pt`.

## 0. TL;DR

The flat LeWM is a **good single-frame place-recognition encoder and a weak
short-horizon dynamics/cost model.** It recognizes places it has seen (~21× above
chance) but has almost no internal sense of *distance* between places, its
"imagination" of the future collapses after ~2 steps, and its goal-seeking cost
is flat enough to make it stop early. None of the *global navigation* failures
are the flat model's job — that is the hierarchy's — and crucially the gate shows
the flat model hands the hierarchy a **usable recognition substrate** to build on.
The real bottleneck is **metric geometry**, most likely an artifact of the
anti-collapse regularizer, and it is testable and possibly cheap to fix.

**Plain:** Our model is like someone with a good memory for *what places look
like* but a poor sense of *how far apart* they are, who also can't picture more
than a second ahead. That's fixable, and the parts it fails at (route-planning
through a maze) were always meant to be handled by a separate planning layer we
haven't built yet.

## 1. What the system is

Single-encoder LeWM (paper-faithful), `lewm/models/lewm.py`: a ViT-Tiny image
encoder (192-dim, 12 layers, 224px) + a 6-layer action-conditioned Transformer
predictor + BatchNorm projectors. ~16M parameters. It is trained to **predict the
next latent (not pixels)** under an action, with a SIGReg anti-collapse term;
planning scores candidate moves by L2 distance between the predicted latent and a
goal-image latent. Macro-step = 0.5 s; training context = 4 frames.

**Plain:** It watches through the robot's camera and learns to predict "what the
next half-second will look like (in a compressed code) if I take this action,"
rather than drawing the actual picture. To navigate, it imagines a few candidate
moves and picks the one that lands closest to a photo of the goal.

## 2. Evidence base (what we ran)

| probe | what it measures | headline |
|---|---|---|
| in-training eval (`metrics.jsonl`) | 1-step prediction, isotropy, action-sensitivity | converged; eval geometry under-generalizes |
| rollout-horizon (e7) | multi-step latent error vs baselines | beats "do nothing" at 1 step, worse by step 3 |
| receding-MPC proxy (e8) | does it rank the true action well | 0.68→0.52 top-1 as horizon grows |
| closed-loop MPC (e9) | actual goal progress | 7/9 local beacon; stress benchmark invalid |
| **A2 aliasing** (e9) | latent distance vs graph distance | ρ ≈ 0.03 (no metric structure) |
| **A3 reachability + history** (e9) | trained readouts; recognition; aliasing | retrieval ~21× lift; reachability ≈ baseline; aliasing mild |

## 3. What the flat LeWM CAN do

**3.1 Predict one step ahead, conditioned on the action.** Action-sensitivity
deltas are positive and grew through training; the 1-step recorded-action rollout
beats a persistence ("freeze the frame") baseline (ratio 0.40 at horizon 1).

> **Plain:** Given a command, it can roughly picture the very next moment, and it
> genuinely *uses* the command (it isn't ignoring the steering wheel).

**3.2 Recognize places from a single frame.** Same-cell nearest-neighbor
retrieval@1 ≈ 0.42 (raw 0.43) vs ~0.02 chance — a **~21–24× lift**; retrieval@5
≈ 0.63; a linear readout decodes the position of *seen* cells at R² ≈ 0.22 (≈30%
better than guessing the mean). This is the memory graph's core requirement.

> **Plain:** From a single camera image it can often say "I've been right here
> before" / which spot this is — far better than chance. This is the single most
> important positive result: it's the foundation the navigation layer needs.

**3.3 Approach a visible, roughly-aligned goal over short range.** On held-out
open scenes with the beacon visible and ~1.2 m away, the LeWM planner reaches it
7/9, beating hold (0/9) and random (0/9) and approaching a privileged
bearing oracle (9/9).

> **Plain:** If the target is in view and more or less straight ahead within
> about a metre, it can usually walk over to it.

**3.4 Tell even look-alike places apart, mostly.** Among the most visually
similar pairs (bottom-decile latent distance), single-frame distance still
separates different-cell from same-cell at AUC ≈ 0.78; a history window adds only
~0.01–0.03.

> **Plain:** Even similar-looking corridors are mostly distinguishable from one
> snapshot. The world confuses it *less* than the "all hallways look the same"
> worry assumed.

## 4. What the flat LeWM CANNOT do

**4.1 Predict more than ~2 macro-steps (~1 s) ahead.** The learned rollout is
worse than persistence by horizon 3 (ratio 1.45), and as an action-ranker it
loses to persistence by horizon 2. This is structural: training (`seq_len=4`)
only supervises 3 transitions.

> **Plain:** Its imagination of the future falls apart after about a second, so
> it cannot plan ahead by simulating a long sequence of moves.

**4.2 Represent *distance* between places.** Latent distance does not track graph
(BFS) distance: Spearman ρ ≈ 0.03; normalized latent distance is ~1.0 for *every*
graph distance, including adjacent cells (distance concentration).

> **Plain:** It knows places by their look but has almost no internal sense of
> "how far" one place is from another — near and far feel about the same to it.

**4.3 Provide a usable goal-seeking gradient → the "hold-attractor."** Because
the goal-image cost is L2 *magnitude* in a distance-concentrated space, moving
toward the goal barely lowers the cost, so "stay still" scores as well as
progressing. In closed loop the planner over-selects `hold` and stops early.

> **Plain:** Since far and near look equally far, its "get closer to the goal"
> signal is weak, so it tends to freeze partway there.

**4.4 Solve global navigation / mazes from a single frame.** Confirmed
(and expected): no route-finding through topology from one snapshot.

> **Plain:** It can't work out a path through a maze from a single photo — but it
> was never supposed to; that's the planning layer's job.

**4.5 Predict cross-scene reachability from latent pairs.** A trained MLP
*memorizes* training pairs (top-1 0.94) but generalizes to held-out scenes at the
majority-class baseline (0.27 vs 0.25). The relationship is scene-specific, not
transferable.

> **Plain:** Show it two photos from a brand-new maze and it can't say how many
> steps apart they are. (In the real system that comes from the explicit map, not
> from comparing two images — so this isn't fatal.)

**4.6 Place brand-new spots on a metric map.** Localization to *unseen* cells has
R² < 0 (worse than guessing).

> **Plain:** It can recognize places it has visited but can't position a never-seen
> spot on a coordinate map.

## 5. Root-cause analysis (why)

**5.1 SIGReg isotropy likely flattens the metric.** The anti-collapse term pushes
embeddings toward an isotropic Gaussian, where pairwise distances mathematically
concentrate to a near-constant — exactly the flat A2 bucket profile. So the cure
for collapse may be eroding the distance structure planning needs.

> **Plain:** The training trick that stops the model from cheating (mapping
> everything to one point) also squashes its sense of distance. Recognition
> survives; metric doesn't.

**5.2 Eval geometry under-generalizes; the LeJEPA de-risk was never applied.**
Held-out per-dim std is 0.75 vs 1.0 on train, and the SIGReg term dominates eval
loss without improving. The trainer uses a uniform sampler over the full mix
(teacher + recovery + exploration); the 2026-05-28 review's source-weighted
sampling and full-mix-vs-exploratory ablation had not yet been wired into
`scripts/train_lewm.py` when this diagnostic was written.

> **Implementation update, 2026-06-04:** source-aware sampling is now wired into
> `scripts/train_lewm.py`, with `--source-allow`, `--source-cap`, and
> `--source-weight`. The ablation plan and launch commands are recorded in
> `docs/lewm_scaled_ablation_decisions_2026-06-04.md`.

> **Plain:** The model's "map sense" doesn't transfer cleanly to new scenes, and
> a planned experiment to fix the training data mix was never actually run.

**5.3 Short horizon is baked in by `seq_len=4`.** Only 3 supervised transitions →
no signal to learn longer rollouts.

> **Plain:** We only ever taught it to predict ~3 steps, so it can't imagine
> further. Teaching longer requires longer training sequences (more compute).

**5.4 Capacity.** ~16M params (ViT-Tiny + 6-layer predictor) is small for this
visual+dynamics task.

> **Plain:** It's a small model; some ceiling is just size.

## 6. Next steps — fixing the flat-LeWM failures

Ordered by value/cost. The measured failure is **metric geometry + the L2 cost**,
not recognition or aliasing, so the fixes target those.

**6.1 (Cheapest, highest information) SIGReg / sampling de-risk ablation.**
Retrain a *small* LeWM with (a) a lower SIGReg λ and/or (b) exploratory-balanced
sampling (the never-run LeJEPA P1/P2), then re-run A2/A3 (ρ + retrieval). If
metric structure recovers, it lifts retrieval *and* the local cost at once and
tells us whether the frozen substrate can be made sound **before** spending
`seq_len=8` compute.

> **Implementation update, 2026-06-04:** the scaled ablation runner now exists:
> `scripts/run_lewm_scaled_factorial_ablation.sh`. It supports the scaled control
> (`λ=0.09`), lambda dose response (`0.03`, `0.01`), exploratory-only arm, and
> combined arms with size-matched `--source-cap auto`.

> **Plain:** Dial back the anti-collapse knob and/or rebalance the training data,
> retrain a small version, and re-measure. This could fix the "no sense of
> distance" problem at the source.

**6.2 Learned cost / reachability head on frozen latents.** Replace raw L2 goal
cost with a learned metric/place head (metric learning). Directly attacks the
hold-attractor and pushes retrieval above 0.42 without re-training the encoder.

> **Plain:** Instead of using crude distance, train a small add-on that learns
> "am I getting closer?" — cheap, no full retrain, fixes the early-stopping.

**6.3 Fix the benchmarks + reconcile the gate doc.** Add yaw-jitter/occlusion to
the beacon task; fix-or-delete the under-budgeted stress benchmark (its oracle
also scores 0/6, so it can't discriminate); correct the planning-gate doc's
stress framing.

> **Plain:** Make our tests actually able to tell a good model from a bad one.

**6.4 (Deferred) `seq_len=8` retrain for horizon** — only after 6.1 shows the
substrate is sound. Longer context is the only way to extend the ~2-step horizon,
but it's expensive and pointless on a broken metric geometry.

> **Plain:** Teaching it to imagine further is worth it only once the basics are
> fixed.

## 7. H-JEPA next steps

**7.1 The gate is answered: build a hierarchy, but it has a foundation.** A2 (ρ
0.03) + A3 (reachability head ≈ baseline) = the registered "insufficient
frozen-latent regime" → Phase B is justified. But recognition works (~21× lift),
so this is "build with a foundation," not the §2 "deeper representational
failure / stop."

> **Plain:** The flat model isn't good enough on its own (as expected), but it's
> good enough to build the navigation layer on top of.

**7.2 Phase B should be a learned retrieval/place head + explicit graph — not
necessarily the heavy history BeliefEncoder.** A3 showed aliasing is mild (history
adds ~0.01–0.03 over single frames), so the BeliefEncoder's *disambiguation* role
is small; its value is robuster retrieval. Reachability/distance comes from BFS
over the explicit memory graph (the plan already does this; A3 confirms latent
regression won't).

> **Plain:** The next layer is mainly a better "have I been here?" matcher plus a
> map the robot fills in as it goes; the elaborate history-memory module the plan
> imagined looks less necessary than thought.

**7.3 Re-calibrate the A4 thresholds.** The plan's gate (ρ ≥ 0.70 AND head ≥ 70%)
is arguably mis-specified for this setting: cross-scene pairwise reachability from
two frames is partly ill-posed (hop-count depends on unseen walls), so it can't
reach 70% regardless of encoder quality. **Recognition retrieval lift is the
better substrate metric**, and it passes decisively.

> **Plain:** One of our pass/fail bars was testing something that's nearly
> impossible by design; the recognition test is the fair one, and we pass it.

**7.4 Decide the long-horizon mechanism explicitly.** The current plan handles
long horizons via graph + hierarchical sub-goaling, **not** a learned
long-horizon predictor (a "true" hierarchical-JEPA higher level). These are
different bets; pick one before building, because it changes whether topology is
*constructed in a map* or *learned in a predictor*.

> **Plain:** There are two ways to do long-range planning — fill in a map, or
> train a model to predict far ahead. We should decide which on purpose.

## 8. Recommended sequencing

1. **6.1 SIGReg/sampling ablation** (small retrain) — the one move that could fix
   the substrate *and* the local cost; gates everything else.
2. In parallel, **6.3 benchmark fixes** (cheap, makes future numbers valid).
3. If 6.1 lifts ρ/retrieval → proceed to **7.2 learned retrieval head + explicit
   graph** (mini Phase B) and **6.2 learned cost**.
4. If 6.1 does *not* help → the metric weakness is deeper (encoder receptive
   field / objective); reconsider scope per plan §2 before committing
   `seq_len=8` or the full belief/memory stack.

> **Plain:** First try the cheap fix that could solve the core problem; fix the
> tests alongside; then build the navigation layer if the fix works, or step back
> and rethink the encoder if it doesn't.

## 9. Reproduce

All probes run on CPU (no GPU contention with training) against the frozen
checkpoint and the rendered/labelled corpus. Setup:

```bash
VENV=.generated/venvs/genesis_render_vulkan/bin/python
CKPT=models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9_b050000.pt
OUT=models/checkpoints_textured_v03_full_20260531/sweep_seq4
```

**A2 — aliasing audit** (§7; ~3 min; fully deterministic):

```bash
$VENV scripts/probe_lewm_latent_aliasing.py --checkpoint $CKPT \
  --split test_id --scenes-per-family 4 --frames-per-scene 160 --max-per-cell 4 \
  --max-pairs-per-scene 40000 --batch-size 64 --device cpu --seed 20260604 \
  --output $OUT/latent_aliasing_e9_b050000_testid.json
```

**A3 — reachability head + history** (§8; ~35 min):

```bash
$VENV scripts/probe_lewm_reachability_a3.py --checkpoint $CKPT \
  --train-scenes-per-family 4 --eval-scenes-per-family 4 \
  --eval-frames-per-scene 240 --train-frames-per-scene 100 \
  --eval-max-per-cell 8 --train-max-per-cell 3 \
  --per-bucket-per-scene 150 --pca-k 48 --ridge-alpha 10.0 \
  --mlp-hidden 256 --mlp-epochs 50 \
  --history-scenes 6 --history-envs 3 --history-seg-len 200 --history-H 4,8 \
  --device cpu --seed 20260604 \
  --output $OUT/reachability_a3_e9_b050000_testid.json
```

Notes:

- Both default `--manifest-corpus` to
  `.generated/scene_corpus/minimum_tex_20260520T211541Z` — the corpus the
  rollouts were generated from (its scene-ids differ from `minimum_*`).
- A2 and the A3 recognition/retrieval/localization metrics are fully
  deterministic (seeded `random`/numpy). The A3 reachability **head** decimals
  may differ ≈±0.02 from the stored JSON: the original run's torch training was
  unseeded; the script now calls `torch.manual_seed`, so re-runs are mutually
  reproducible. The ≈-baseline conclusion is stable regardless.
- Closed-loop MPC (§3, 7/9 beacon), the rollout-horizon diagnostic, and the
  receding-MPC proxy are documented in `docs/lewm_planning_readiness_gate.md`
  (the `probe_lewm_rollout_horizons.py` and `benchmark_lewm_closed_loop_mpc.py`
  blocks).
- Every result JSON embeds its full parameter block (top-level or `config`), so
  the exact flags behind any artifact are recoverable from the file itself.
