# LeWMQuad-v3 — State & Efficacy Diagnostic (2026-06-04)

## Scope

A whole-project assessment of where v3 stands: pipeline state, the trained LeWM
checkpoint, all three evaluation layers, benchmark validity, and how the current
activity maps onto the registered v3 plan. All numbers below were read directly
from the artifacts, not from prior summaries.

Checkpoint under evaluation: `lewm_seq4_e9_b050000.pt`
(`models/checkpoints_textured_v03_full_20260531/sweep_seq4/`).

Artifacts referenced:

- `models/checkpoints_textured_v03_full_20260531/sweep_seq4/metrics.jsonl`
- `.../closed_loop_mpc_visible_beacon_e9_b050000_testid_open_obstacle_field.json`
- `.../closed_loop_mpc_kinematic_e9_b050000_testid_visual_sensor_stress.json`
- `.../cpu_mini_rollout_probe_lewm_seq4_e7.json`
- `.../cpu_receding_mpc_proxy_lewm_seq4_e8.json`
- `docs/v3_topological_nav_plan.md`, `docs/fresh_retrain_data_spec.md`,
  `docs/lejepa_strategy_review_2026-05-28.md`,
  `docs/lewm_planning_readiness_gate.md`
- `lewm/models/lewm.py`, `scripts/train_lewm.py`,
  `scripts/benchmark_lewm_closed_loop_mpc.py`
- Post-diagnostic implementation note:
  `docs/lewm_scaled_ablation_decisions_2026-06-04.md`

## Verdict up front

The **engineering pipeline is a genuine success** — datagen → 404 GB textured
render → a trained-to-convergence LeWM → a working evaluation harness. The
**science is at a much earlier and more fragile point than the activity
suggests.** What exists is a vision-only latent cost model with a *real but
shallow* one-step action-conditioned signal that decays to useless by horizon 3,
plus a degenerate bias toward "hold."

Dropping the maze/stress benchmark was the right call — but more strongly than
the planning-gate doc states: **the stress benchmark is invalid as configured
(the privileged oracle also scores 0/6), and the beacon benchmark sits at the
opposite extreme (so easy a trivial bearing oracle aces it).** Neither yet
discriminates model quality well.

Separately, the project has **drifted off its own registered plan**: the v3
H-JEPA plan assumes a *frozen, adequate* LeWM and defines a Phase A decision
gate that has never been run, while the de-risking the LeJEPA review demanded
(source-aware sampling) was never wired into the trainer at the time of this
diagnostic snapshot. Same-day follow-up work has since wired it; see
`docs/lewm_scaled_ablation_decisions_2026-06-04.md`.

## 1. Where we are — pipeline state

| Stage | State | Evidence |
|---|---|---|
| Scene corpus | ✅ Complete — §7.2 MINIMUM tier, held-out `test_id`/`test_hard` | `minimum_20260520T080420Z`, `open_obstacle_field` in `test_id` |
| Datagen (rollout+labels) | ✅ Complete, NaN family handled | `dataset_sequences` ~153k indexed |
| Render (textured v03) | ✅ Complete — 1450/1450 scenes, 69.6M PNGs, 404 GB, 0 fails | render manifest |
| LeWM training (seq_len=4) | 🟡 **Converged, ~70% through final epoch** — epoch 9, batch 50k/72k | `metrics.jsonl`, tmux `lewm_seq4_gate_20260602` live |
| Eval harness | ✅ Three layers: offline horizon probe, receding-MPC proxy, closed-loop MPC | scripts + JSON outputs |
| Source-aware LeWM sampling | ✅ **Implemented after this diagnostic snapshot** — default still uniform; ablation flags now available | `train_lewm.py` parses `command_source`; see decision log |
| v3 Phase A diagnostic gate | ✅ **Run after this diagnostic snapshot** — A2/A3 artifacts recorded below | plan §4.3–4.5; §7–§8 |
| seq_len=8 / 16 sweep | ⏸ Gated behind manual approval of e9 | gate doc §"Current Main Run" |

**Model** (`lewm/models/lewm.py`): paper-faithful single-encoder LeWM — ViT-Tiny
encoder (192-d, depth 12, patch 14 @ 224 px) + 6-layer AdaLN-conditioned
Transformer predictor (192-d, 16×64 heads), BatchNorm projectors. ~16M params,
`cmd_dim=15`, `max_seq_len=4`, `stride=5` (0.5 s/macro). Loss = teacher-forced
latent MSE + 0.09·SIGReg. Planner cost = terminal **L2² to the goal-image
latent** (`plan_cost`, `lewm.py:237`).

Training is **not under-trained** — ~10 epochs × ~72k batches × 128 ≈ 90M
sample-presentations; eval loss plateaued by epoch ~3. The ceiling is capacity /
data-geometry / objective, **not** more steps.

## 2. The results, read honestly (three layers)

### Layer 1 — Offline latent metrics (`metrics.jsonl`, epoch-end)

| ep | eval_pred (1-step MSE) | eval_rollout (≤3-step) | eval_std | eval_sig | action_zero_Δ | action_shuf_Δ |
|---|---|---|---|---|---|---|
| 0 | 0.050 | 0.139 | 0.651 | 44.8 | 0.026 | 0.025 |
| 7 (best) | **0.040** | 0.149 | 0.726 | 37.7 | 0.043 | 0.037 |
| 8 | 0.042 | 0.159 | 0.747 | 38.6 | 0.042 | 0.035 |

**Good:** action-sensitivity deltas are positive and *growing* — the predictor
genuinely conditions on actions (it is not ignoring control). 1-step MSE is low
and improving.

**Concerning:** `eval_rollout_pred` is flat-to-worsening (rollout error ~3.5× the
1-step error, not improving). The headline: **`eval_loss` ≈ 3.43 is 99% SIGReg**
(0.040 pred + 0.09·37.7). On *train*, `train_sig` ≈ 0.74; on held-out scenes
`eval_sig` ≈ 38, with `eval_std` 0.75 vs `train_std` 1.00. The std gap
(normalization-robust) shows held-out embeddings are **under-dispersed** — the
encoder's isotropic geometry does **not** generalize to held-out scene
topologies. (The ~50× sig gap is partly an eval-batch-size/normalization effect
and should be confirmed with a matched-N check, but `eval_std` corroborates the
direction independently.)

### Layer 2 — Planning proxies (`probe_*` JSONs)

Rollout horizon probe (e7) — learned rollout vs persistence (lower = better):

| horizon | rollout/persistence | reads as |
|---|---|---|
| 1 | **0.40** | clearly beats "do nothing" |
| 2 | 0.87 | marginal |
| 3 | **1.45** | **worse than persistence — diverges** |

Receding-MPC proxy (e8) — does the model rank the *true* recorded action well
among candidates?

| horizon | top-1 rate | win-rate vs persistence |
|---|---|---|
| 1 | 0.68 | 0.61 |
| 2 | 0.63 | **0.37** |
| 3 | 0.52 | **0.32** |

**Takeaway:** the model is a usable **1-step** cost, marginal at 2 steps, and
**anti-useful by 3**. As a *ranker* it loses to persistence beyond h1. This is
why the closed-loop benchmark had to be pinned to `--horizon 2`. This is not "a
world model you can plan in" — it is a short-range local cost.

### Layer 3 — Closed-loop MPC (`e9_b050000`, kinematic, vision-only LeWM vs baselines)

**Visible-beacon, `test_id/open_obstacle_field`, start 1.2 m, goal radius 0.35 m, 12 blocks:**

| policy | success | progress (m) | final dist (m) | path eff |
|---|---|---|---|---|
| bearing *(privileged: knows goal xy)* | 9/9 | 0.875 | 0.325 | 1.000 |
| **lewm** *(vision-only)* | **7/9** | 0.781 | 0.419 | 0.951 |
| random | 0/9 | 0.547 | 0.653 | 0.643 |
| hold | 0/9 | 0.000 | 1.200 | 0.000 |

lewm primitive counts: `forward_fast 27, hold 19, arc_left 12, arc_right 9,
forward_slow 8, forward_medium 4, yaw_right 3`.

**Stress, `test_id/visual_sensor_stress`, start 4.36 m, 8 blocks:**

| policy | success | progress (m) |
|---|---|---|
| bearing *(oracle)* | **0/6** | 0.394 |
| random | 0/6 | 0.112 |
| lewm | 0/6 | 0.037 |
| hold | 0/6 | 0.000 |

lewm primitive counts: `hold 25, yaw_right 10, yaw_left 9, arc_left 2,
forward_medium 1, arc_right 1`.

## 3. The benchmark-validity problem (important)

**The stress benchmark is mis-specified, not just "hard."** The *privileged
bearing oracle* scores 0/6. Across both runs the effective closing rate is
~0.1–0.13 m per block (bearing nets 0.875 m in 7 blocks on the easy task). At
that rate, 8 blocks closes at most ~1 m — but the stress start is **4.36 m**
away. **No controller, perfect or otherwise, can win that config.** The gate
doc's framing — *"evidence that the flat LeWM local cost is insufficient for
harder navigation"* — is therefore an over-reading. Failure cannot be attributed
to the model when the oracle fails identically; the correct conclusion is "the
benchmark is under-budgeted (block cap ≪ distance)."

**The beacon benchmark is at the opposite extreme.** It guarantees
line-of-sight, zero yaw jitter (robot starts *facing* the beacon), open space,
and a block budget (12) calibrated to be *just* winnable (~1.2 m ÷ ~0.1 m/block).
That is why a trivial bearing oracle gets a perfect 9/9. It is the right *kind*
of task, but the ceiling is so low it mostly tests "can you walk forward toward
the thing you are already pointed at." Honest read of lewm 7/9: a vision-only
cost approaching a privileged oracle on the easiest possible task — encouraging
that the goal-image latent is *not degenerate*, but not yet evidence of a
planning capability.

**The lewm failure signature is consistent and diagnostic.** On the easy task it
still spends 19/84 primitive-steps on `hold` and the 2 failures are premature
stops; on the (unwinnable) stress task it collapses to `hold 25, yaw 19`. The
cost has a **"hold" attractor**: once easy initial progress is made, candidate
rollouts that command motion score *worse* (further from goal latent) than
standing still. Mechanistically this lines up with the rollout divergence (the
predictor's multi-step motion predictions are unreliable, so "do nothing" looks
safe in latent space). The mechanism is *plausible, not proven* — the two probes
are small-sample (16 and 64) and disagree in detail — but the **behavioral**
hold-bias is robust across both runs.

## 4. What's achieved vs. what isn't

**Achieved**

- End-to-end reproducible pipeline: corpus → 404 GB textured render → train →
  eval, resumable, 0 render fails.
- A converged, paper-faithful LeWM that is genuinely **action-conditional**
  (action-sensitivity deltas positive/growing).
- A **non-degenerate goal-image cost** at short range: beats hold/random
  decisively, approaches a privileged oracle on an open LoS approach (7/9).
- A three-tier eval harness (offline horizon, MPC proxy, closed-loop) and an
  explicit, checkpoint-specific approval gate — good experimental hygiene.

**Not achieved**

- **Any planning horizon.** Useful signal dies by h3 (rollout worse than
  persistence; ranker loses to persistence by h2).
- **Held-out representation geometry.** Eval embeddings under-dispersed (std 0.75
  vs 1.0); SIGReg term dominates eval loss and does not improve.
- **A discriminating benchmark.** One is unwinnable for the oracle; the other is
  acelable by a trivial oracle.
- **The registered v3 decision.** Phase A (frozen-latent reachability probe:
  ρ ≥ 0.70, bucket acc ≥ 70%, near/far confusion ≤ 15%) — the *actual* gate that
  decides whether to build the belief/memory stack — has not been run.
- **The LeJEPA-review de-risking.** Source-aware sampling (P1) and the full-mix
  vs exploratory ablation (P2) were never implemented; this run trained on the
  uniform marginal the review explicitly warned against.

## 5. Evaluation of the approach & efficacy

**The architecture choice is sound and current.** Single-encoder LeWM + SIGReg
is defensible and matches the cited papers. Freezing LeWM and testing whether a
cheap reachability probe suffices (the v3 thesis) is a *good, falsifiable*
research design — arguably the strongest thing in the repo.

**Three structural issues are limiting efficacy:**

1. **Methodological drift.** The registered plan (`v3_topological_nav_plan.md`) says LeWM
   is *frozen* and the next work is the Phase A *diagnostic*, not "retrain LeWM
   and invent closed-loop gates." The current activity — a 10-epoch retrain plus
   bespoke MPC benchmarks with hand-tuned distances/block caps — is
   *relitigating LeWM adequacy* with un-pre-registered, mis-calibrated tests.
   That is how a doc narrative ("much less bleak") can swing on benchmark knobs
   rather than on the registered ρ/bucket criteria. Decide which game is being
   played.

2. **The LeJEPA review was right and was not acted on.** `train_lewm.py` samples
   uniformly over teacher + recovery + exploration. The review predicted this
   concentrates the marginal and hurts representation geometry — and the eval
   metrics show exactly that symptom (under-dispersed held-out embeddings,
   dominant non-improving eval SIGReg). The fix (P1 source-weighted sampler, P2
   exploratory-vs-full ablation) is cheap relative to a 10-epoch retrain and was
   skipped.

3. **The capability ceiling is structural for a single-frame, seq_len=4 model.**
   `max_seq_len=4` ⇒ teacher-forced loss covers only 3 transitions; the model
   cannot be expected to plan beyond ~2 steps, and the data confirms it does
   not. The hold-attractor + horizon-3 divergence are the predictable
   consequences. This is *consistent with the plan's own framing* ("LeWM is a
   local backbone, not a router") — which is precisely why pouring more effort
   into closed-loop MPC tuning on a flat LeWM has low marginal value.

**Efficacy summary:** high engineering efficacy, low-but-nonzero scientific
efficacy so far, and **declining marginal returns on the current activity**
(benchmark-knob-tuning a converged, capacity-limited local model) versus the
high-information, low-cost step the plan already specifies (Phase A diagnostic).

## 6. Recommendations (prioritized)

> **Update (2026-06-04, post-gate):** the §6 list below is the *pre-gate*
> snapshot. Recommendation 1 is now **done** — A2 and A3 were built and run (see
> §7–§8). The re-evaluated, superseding plan (capabilities, fixes, H-JEPA next
> steps, with plain-language explanations) is
> `docs/lewm_capabilities_and_roadmap_2026-06-04.md` §6–§8. A later same-day
> implementation note (`docs/lewm_scaled_ablation_decisions_2026-06-04.md`)
> records the source-sampler wiring, scaled ablation runner, and benchmark
> configuration decisions.

1. **Run the registered Phase A diagnostic on `e9` — highest-information next
   step, already designed.** Frozen-latent reachability probe + visual-aliasing
   audit → Spearman ρ between latent distance and graph distance, bucket
   accuracy, near/far confusion. This *decides* whether the belief/memory stack
   is needed and tells you far more than another MPC run. It is the gate already
   written but never used (plan §4.3–4.5).

2. **Fix the two benchmarks so they discriminate.** Stress: raise `--max-blocks`
   to match distance (≥ ~40 for 4.36 m at the observed ~0.1 m/block) *or*
   shorten the start — otherwise delete it, since an oracle-failing test is
   noise. Beacon: add `--beacon-start-yaw-jitter-rad` (e.g. 0.5–1.0) and a
   partial-occlusion variant so a privileged bearing oracle no longer gets 9/9;
   that creates headroom to actually measure the model.

3. **Before any seq_len=8 retrain, run the LeJEPA review's P1+P2**
   (source-weighted/exploratory sampler vs full-mix, ~3 small checkpoints). If
   exploratory-biased sampling lifts eval_std toward 1.0 and the reachability ρ,
   that is a cheap win and avoids baking the narrow-marginal defect into the more
   expensive sweep.

4. **Investigate the hold-attractor directly** — it is the concrete blocker.
   Check whether `plan_cost` (terminal L2² to goal latent) systematically
   under-rewards motion (e.g., compare predicted-latent displacement under
   `forward_*` vs the true encoder displacement). A cost-shaping or a short
   multi-step cost may matter more than more training.

5. **Update the gate doc's stress-test framing** — replace "evidence the local
   cost is insufficient" with "benchmark under-budgeted; oracle also 0/6; not a
   valid discriminator." As written it will mislead a future reader.

The training run is near the end of epoch 9 and will stop itself; nothing is on
fire. The right move is not "more LeWM/MPC tuning" — it is **run the Phase A gate
already registered**, and **act on the source-sampling de-risk** before spending
the next sweep's compute.

## 7. Phase A2 result — frozen-latent aliasing audit (run 2026-06-04)

First registered Phase A artifact, built and run on `lewm_seq4_e9_b050000.pt`:
`scripts/probe_lewm_latent_aliasing.py`, 32 held-out `test_id` scenes (4 per
family), 160 frames/scene, CPU. Each rendered frame is mapped to its
ground-truth `cell_id` (`labels.jsonl`), encoded with the frozen encoder, and
pairwise **latent L2 distance** is correlated against pairwise **BFS graph
distance** on the scene's `SceneGraph`. Cross-scene pairs are never formed.
Artifact: `.../sweep_seq4/latent_aliasing_e9_b050000_testid.json`.

### Numbers

Spearman ρ (latent distance vs graph distance), per-scene, aggregated:

| representation | mean ρ | median ρ | max ρ |
|---|---|---|---|
| projected `z_proj` (goal-matching cost space) | 0.029 | 0.027 | 0.21 |
| raw backbone `z_raw` | 0.080 | 0.072 | 0.24 |
| projected, yaw-matched (same heading bin) | 0.242 | 0.187 | 0.59 |

Per-scene-median-normalized latent distance, by graph-distance bucket:

| graph dist | 0–1 | 2–3 | 4–7 | 8–15 | 16+ |
|---|---|---|---|---|---|
| norm. latent dist | 0.94 | 1.00 | 1.01 | 1.00 | 0.99 |

Near/far confusion: of the closest-10%-in-latent pairs, ~22% (median) to ~29%
(mean) are actually ≥8 graph-hops apart. Per-family ρ_proj: open_obstacle_field
highest (~0.1–0.21), rough/loop ~0.06–0.12, mazes ≈0 or slightly negative
(medium mazes −0.05 to −0.08).

### A4 verdict

Decisively the **insufficient frozen-latent regime** (`v3_topological_nav_plan.md` §4.5):
ρ_proj median 0.03 ≪ 0.40; confusion ≫ 15%. Per the registered gate,
"insufficient" → build Phase B (BeliefEncoder) to test H2.

### Three findings

1. **Distance concentration.** Latent distance is flat (~1.0) across every graph
   distance — cells 16 hops apart are no farther than adjacent ones. Classic
   high-dimensional **isotropic** embedding signature (what SIGReg pushes
   toward). Corroborates the eval_std 0.75 / eval_sig 38 story in §2.
2. **Heading dominates; place is weak-but-present.** Yaw-matching lifts ρ from
   0.03 → 0.19 (max 0.59). The single-frame latent is largely a *heading* code;
   place structure exists but is faint and masked. The internal gradients prove
   the pipeline is sound, not buggy: yaw-matching helps, open-field beats mazes,
   and adjacent cells (0.94) are the only sub-median bucket.
3. **Mechanistically explains the closed-loop hold-attractor (§3).** A
   near-constant latent distance to the goal image means the L2² `plan_cost` has
   almost no gradient toward the goal, so "hold" scores as well as moving. The
   beacon task worked (7/9) only because start and goal share heading over ~1 m
   — the narrow yaw-matched regime where the faint signal survives. A2 and the
   MPC results are one coherent story.

### Is this just expected for a flat JEPA, not the H-JEPA? (interpretive guardrail)

Partly yes, and the distinction matters:

- **Global multi-hop routing is *not* the flat model's job.** No one should
  expect a single-frame encoder to make latent distance track global BFS
  distance, especially in the far buckets (8–15, 16+). A low ρ there is expected
  and is *precisely the plan's trigger* to build Phase B — not a verdict that
  "LeWM failed."
- **But the H-JEPA is built *on top of* these frozen latents.** Per
  `v3_topological_nav_plan.md` §3, LeWM is frozen through Phases A/B/C; BeliefEncoder,
  memory, GoalAdapter, and ReachabilityHead all consume LeWM latents and never
  re-encode pixels. Crucially, in this plan **topology is represented by an
  explicit memory graph + a trained reachability head, not by a future
  long-horizon latent predictor.** Long horizon is handled by hierarchical
  sub-goaling over the graph; the predictor stays the short-horizon LeWM. So the
  load-bearing question is whether the frozen latent carries enough *local,
  place-discriminative information* for (a) node retrieval (BeliefEncoder) and
  (b) reachability prediction (head) to recover. That is a *substrate-quality*
  question, not a "missing long-horizon predictor" question.
- **A2 measures geometry, which conflates two very different cases.** Raw L2
  distance being flat is consistent with *both* "info present but not in
  L2-metric form" (fixable by a learned head / history — fine, build the H-JEPA)
  *and* "info destroyed at encode time" (NOT fixable by any downstream head or
  memory — the plan's §2 "deeper representational level" failure, out of scope
  for v3). A2 alone cannot separate them.
- **Steelman of the flat-model defense (granted):** place may be inherently a
  *temporal/belief* construct — two identical corridors cannot be told apart
  from one frame regardless of encoder quality, so single-frame ρ is partly an
  impossible ask. **Counter:** even within a fixed heading (aliasing reduced),
  ρ is only ~0.19, and even *adjacent* cells are barely closer than random —
  that is thinner local discriminability than "place needs history" alone
  predicts, and it is also why the flat model's *own* in-scope job (the local
  goal-image cost, e.g. the 1 m beacon) is shaky.

Net: A2 does not condemn the approach; it correctly returns "insufficient regime
→ build the stack," with a **yellow flag on substrate quality**. The user's
question ("isn't topology the H-JEPA's job?") is the strongest argument *for*
running A3 next, not for dismissing A2.

### What A2 does not settle → run A3

A2 is distance-based. The **Phase A3 frozen-latent reachability head** (small
linear/MLP probe predicting the BFS bucket) is the disambiguator and completes
the A4 gate:

- If the head **also** can't beat trivial+15pp → info isn't recoverable →
  deeper representational problem (encoder receptive field / SIGReg objective /
  narrow marginal). The H-JEPA would be building on sand; the plan says stop and
  fix the encoder/objective (out of v3 scope).
- If the head **can** recover buckets → the latents hold place info that L2
  can't use → the belief stack is justified and the fix is a *learned*
  reachability cost (Phase C ReachabilityHead). A2's low ρ was then just "not the
  flat model's job + SIGReg geometry."

The plan's A2 also specifies a **history-disambiguability** check (§4.2 item 4)
that directly tests the steelman above; it has not been run yet and should
accompany A3.

## 8. Phase A3 result — reachability head + history (run 2026-06-04)

Built and ran `scripts/probe_lewm_reachability_a3.py` on `lewm_seq4_e9_b050000.pt`,
32 held-out `test_id` scenes (eval) + 32 `train` scenes (head training), CPU.
Artifact: `.../sweep_seq4/reachability_a3_e9_b050000_testid.json`. This answers
the question §7 left open — *info present-but-not-L2-usable* vs *info destroyed* —
and it **refines §7's "barely encodes topology" into a sharper recognition-vs-metric
split.**

### Numbers

Place **recognition** (training-free same-cell nearest-neighbor; the memory
graph's actual job):

| metric (proj / raw) | median | chance | lift |
|---|---|---|---|
| retrieval@1 | 0.41 / 0.43 | ~0.02 | **~21× / 24×** |
| retrieval@5 | 0.63 / 0.64 | — | — |
| localization R² (recognition, *seen* cells) | 0.22 / 0.24 | — | err ratio 0.72/0.69 (beats baseline ~30%) |
| localization R² (metric, *unseen* cells) | −0.11 / −0.18 | — | fails |

Reachability bucket head (cross-scene; trained on `train` pairs, eval on
`test_id`; balanced buckets, ~19.5k train / ~19.5k eval pairs):

| head | train top-1 | eval top-1 | majority baseline | gain | near/far conf |
|---|---|---|---|---|---|
| linear `|z_a−z_b|` | 0.36 | 0.26 | 0.246 | +0.014 | 0.18 |
| MLP concat | **0.94** | **0.27** | 0.246 | **+0.025** | 0.09 |

History-disambiguability (AUC separating aliased *different-cell* from same-cell
pairs; mean-pooled window = a weak lower bound for a real BeliefEncoder):

| H | single-frame AUC | history AUC | n diff-cell |
|---|---|---|---|
| 4 | 0.76 | 0.79 | 1383 |
| 8 | 0.78 | 0.79 | 1287 |

### A4 verdict (gate now complete)

Formally the **insufficient frozen-latent regime** (`v3_topological_nav_plan.md` §4.5):
A2 ρ ≈ 0.03 (< 0.40) **and** A3 reachability head ≈ majority baseline (+0.025 ≪
+15 pp). Per the registered gate → **build Phase B.** But the binary verdict
hides the important part: *why* it's insufficient.

### The real finding — recognition works, metric does not

The frozen latents are a **good place-*identity* code but a poor *metric/relational*
code**:

- **Recognition is recoverable** — a single frame's nearest neighbor is the same
  cell ~42% of the time vs ~2% chance (**~21× lift**), and seen-cell position
  decodes at R² ≈ 0.22. **The information the H-JEPA needs is present, not
  destroyed at encode time.** This is the case that justifies building the belief
  stack *with a foundation*, not the §2 "deeper representational failure" case.
- **Metric/relational structure is not** — latent distance ≠ graph distance (A2),
  unseen-cell metric generalization fails (R² < 0), and the MLP *memorizes*
  training pairs (0.94) but generalizes at baseline (0.27). Cross-scene
  reachability is **not regressible from latent pairs** — but in the v3 plan
  reachability comes from BFS over the explicit memory graph, so this failure is
  not fatal to the actual mechanism.

This resolves the user's question definitively: **global metric topology is the
hierarchy's job, the flat model correctly does not solve it, and A3 shows the
flat model nonetheless supplies a usable recognition substrate for the hierarchy
to build on.** §7 read as bleak only because it measured L2 geometry, which is
exactly the part that is broken.

### Surprise: aliasing is milder than the maze framing assumed

Even among the most look-alike pairs (bottom-decile single-frame distance),
single-frame distance already separates different-cell from same-cell at AUC
~0.78, and a (crude) history window adds only ~0.01–0.03. So genuine hard
aliasing (truly indistinguishable different places) is **relatively rare** here —
partial observability is **not** the dominant bottleneck. The §4.2 test
therefore comes back the *opposite* of the "structural aliasing" failure mode:
aliased pairs are mostly disambiguable from a single frame already.

### What this implies (strategic refinement)

The bottleneck is **metric/relational recoverability + the L2 cost geometry**, not
partial observability. Consequences:

1. **The hold-attractor / weak local MPC (§3) stands and is the same root cause.**
   The flat planner's `plan_cost` uses L2 *magnitude* — the one thing that is
   broken (distance concentration). Recognition working does not rescue a metric
   cost. A learned cost likely matters more than more training.
2. **Phase B's value is a better *retrieval head*, not history-disambiguation.**
   Recognition at 0.42 retrieval@1 is workable-not-great; a learned metric/place
   head should push it higher. The heavy history-conditioned BeliefEncoder is
   *less* clearly necessary than the plan assumed, because aliasing is mild — its
   payoff is robuster retrieval, not rescuing structural ambiguity.
3. **Reachability should come from the explicit graph, not latent regression**
   (A3 showed the latter fails cross-scene). The plan already does this; A3
   confirms it's the right call.
4. **The distance-concentration is plausibly a SIGReg-isotropy artifact** (§2,
   §7). Whether a less-aggressive isotropy weight or the exploratory-balanced
   sampling de-risk (§6 rec 3) restores metric structure — and lifts retrieval
   and the local cost — is a cheap, high-value ablation worth running before
   committing to the full Phase B stack.

### Recommended next step

The gate is answered: there *is* a foundation, so a hierarchy is worth building,
but the cheapest high-value move is **not** the full belief/memory stack. It is
to (a) add a **learned place/reachability head** (metric learning on frozen
latents → push retrieval up; explicit graph for distance) and (b) test whether
the **objective/sampling de-risk** lifts the broken metric geometry. Both
directly target the measured failure (metric, not recognition) and both are far
cheaper than Phase B + Phase C in full.
