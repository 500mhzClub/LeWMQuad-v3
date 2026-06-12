# Nav-cost diagnosis & fix plan — chasing energy-head navigation

**Date:** 2026-06-06
**Decision:** Chase navigation. The world-model prediction metrics scale cleanly
with 4× data, but the energy-head nav demo is sub-random at every checkpoint. We
commit to fixing the **planning cost geometry** rather than treating nav as a
documented open problem.
**Status:** Phase 0 (diagnose) in progress.

Related: `models/.../seq11_rollout_lam0p25_h10_warm2_sess8k_ep12/scaling_report.md`
(per-checkpoint suite), `docs/lewm_state_diagnostic_2026-06-04.md` §7–§8 (Phase-A
recognition-vs-metric gate), `docs/lewm_rollout_prediction_experiment_log_2026-06-05.md`.

---

## 1. What works vs what's stuck

**Works (scaling run, e0→e3 of the 8k-session model):** action-conditioned
prediction improves monotonically — `zero−free`@h10 +0.136→+0.201, beats-persistence
0.39, MPC win-rate vs-zero ~0.66. The substrate predicts forward dynamics and is
action-conditioned.

**Stuck:** the `GoalEnergyHead` pure-perception nav (`navH`) is **below the random
baseline at every epoch** (mean progress 0.06–0.14 m vs random 0.382 m; success
0–25% on N=8). Per-scene analysis shows it is **argmax churn on a flat cost
surface** — the same scenes do *not* win across epochs; outcomes snap between +1.0
(drives to goal) and −1.2 (drives away). It never had a real nav signal; the swings
are not learning/forgetting.

The decisive observation: the head's **static** ranking accuracy is 0.98, but that
score has no usable margin for the **selection** question MPC actually asks.

## 2. Root-cause hypotheses (from reading the harness)

The MPC picks `argmin_cost` over 343 candidate primitive sequences (7 primitives,
horizon 3). `_choose_lewm_primitive` rolls each candidate forward
(`plan_rollout`), then scores the endpoint with the head against the goal latent.
Three mechanisms can flatten that cost surface:

1. **Easy negatives (head objective).** `train_lewm_energy_head.py` trains the head
   contrastively against **in-batch negatives = other windows' goals = entirely
   different places**. So 0.98 acc means "right place vs a totally different place"
   — pure *place discrimination*. It was never trained to separate the 343
   *nearby* candidate endpoints that all aim at the **same** goal. A `hard_offsets`
   path (same-trajectory temporal neighbors) is plumbed but the production head was
   selected by `ranking_acc` (not `hard_acc`) → cross-scene negatives only.

2. **Heading-domination (representation).** The LeJEPA latent is heading-dominated
   (yaw R² 0.81 vs pos 0.16; `lewm_viewcode_rollout_diag`). A goal latent encodes
   mostly *"from what heading am I viewing the beacon,"* not *"where is the
   beacon."* So "close to goal" partly means "predicted view has the goal's
   heading," not "predicted position is near the goal."

3. **Single-view goal, deployed (CORRECTED).** The benchmark *supports* multi-view
   image goals (`--goal-views N`: render the beacon from N approach headings, take
   `min` energy over views — explicitly to be robust to arrival heading given (2)).
   **But the driver `watch_cpu_nav_track.sh` never passes `--goal-views`**, so every
   `navH` run used a **single front-view goal** (default 0). This is not a
   train/eval mismatch — training is single-view too — it is an **unused
   robustification**. With a heading-dominated latent, a single goal photo only
   matches one arrival heading, so a candidate that reaches the right *position* but
   wrong *heading* is penalized. Prime suspect, cheap to test.

**Going-in bet:** nav failure is **(1)+(3)** — head-objective + the unused
multi-view lever — *before* it is (2), the representation wall. Phase 0 measures
which.

## 3. Plan

### Phase 0 — Candidate-level probe (CPU, no retrain, on existing e3)
`scripts/diagnose_nav_cost.py`. For each of the 8 deployed scenes, at the **start
pose** (reproduces navH's first decision exactly — same seed/scene/goal), score all
343 candidates three ways — **head single-view** (as deployed), **head multi-view
min** (the unused lever), **bare L2** (as navL2) — against the **true** final
distance each candidate reaches under the kinematic model. Report per scene +
aggregate:

| Metric | Tells us |
|---|---|
| Spearman ρ(cost, true_distance) | Is the cost monotone in reality at all? |
| **regret** = true_dist(argmin cost) − true_dist(oracle-best candidate) | Is *ranking* the failure vs the candidate set? |
| oracle-best vs random-pick vs chosen distance | Can the 343 candidates even reach the goal? (planner/horizon vs cost) |

### Phase 1 — Decision gate on the probe
- **oracle-best can't approach goal** → candidate/horizon problem → benchmark-only
  fixes (finer forward primitive, horizon 1–2 to cut compounding, bigger candidate
  set). No retrain.
- **oracle reaches but head-regret high, ρ≈0; multi-view ρ ≫ single-view ρ** →
  the unused multi-view lever + head objective → **Phase 2A**.
- **even bare-L2 ρ≈0 AND distinct endpoints have near-identical latents** → genuine
  representation wall → **Phase 2C** (expensive, gated behind this evidence).

### Phase 2A — Metric-aware head retrain (minutes/iteration on cached latents)
1. Enable **temporal hard negatives** (`hard_offsets`): rank the goal step below its
   own neighbor steps → fine closer-vs-farther gradient.
2. Add a **distance-monotonicity target**: we have true poses in datagen, so train
   the scalar to be monotone in `distance(pred_pose, goal_pose)` — a metric cost,
   not a place classifier.
3. Train the head **under the multi-view min** it will be evaluated with, and turn
   on `--goal-views` in the nav benchmark.
   Iterate against the Phase-0 regret/ρ probe (seconds); promote to nav only when
   regret drops.

### Phase 2C — Representation fix (only if 2A fails)
Auxiliary metric head (pose/odometry or reachability-distance regression) as an aux
loss during **backbone** training, so the latent carries metric geometry. Expensive
(backbone retrain) — gated behind Phase-0 evidence.

### Phase 3 — Make the eval trustworthy
N=8×1 can't show a trend. Scale nav to ~32–64 scenes × 3 trials with yaw jitter,
report **progress** not success-rate, and add the **cost-regret metric** to the
per-checkpoint suite so every future checkpoint directly reports "is the cost field
metric."

The 8k scaling run continues untouched on GPU throughout; all of Phase 0/2A is CPU.

## 4. Probe invocation

```bash
PY=.generated/venvs/genesis_render_vulkan/bin/python
DIR=models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12
$PY scripts/diagnose_nav_cost.py \
  --checkpoint $DIR/lewm_seq11_e3.pt --head-ckpt $DIR/head_lewm_seq11_e3.pt \
  --family open_obstacle_field --split test_id --scene-limit 8 --horizon 3 \
  --goal-views 8 --device cpu --backend cpu --output $DIR/navcost_e3.json
```

## 5. Results — Phase 0 on e3 (2026-06-06)

`navcost_e3.json`. 8 scenes, horizon 3, 343 candidates, start-pose decision (=
navH's first decision, exact same scenes/goals). Lower distance = closer; ρ is
rank-correlation of cost vs true endpoint distance (want strongly positive);
regret = chosen − oracle-best (want → 0). Random-pick regret ≈ 1.41 − 1.12 = 0.29 m.

```
init=1.50m  oracle-best=1.12m  random-pick=1.41m
  head_single  rho=+0.223  regret=0.294m  chosen=1.419m    <- AS DEPLOYED
  head_multi   rho=+0.377  regret=0.280m  chosen=1.405m    <- unused --goal-views lever
  l2           rho=-0.312  regret=0.392m  chosen=1.517m    <- navL2 path
```

**Findings (branch = Phase 2A, not 2C):**

1. **The deployed head ranks no better than random in outcome.** head_single regret
   0.294 m ≈ random-pick regret 0.29 m; chosen endpoint 1.419 m ≈ random-pick
   1.41 m. Despite a *positive* ρ (+0.22), the argmin pick isn't the closest
   candidate. This is the direct cause of sub-random navH.

2. **The latent carries exploitable metric signal — it is not a hard wall.**
   Multi-view min nearly **doubles** the correlation (ρ +0.22 → +0.38), confirming
   the heading-domination hypothesis: a single goal photo only matches one arrival
   heading, and averaging over headings recovers position signal. The unused
   `--goal-views` lever is real — but not sufficient alone (regret only 0.294 →
   0.280; the pick barely moves). Necessary, not sufficient.

3. **Bare L2 is actively anti-metric** (ρ −0.31; chosen 1.517 m > initial 1.50 m =
   moves *away*). Confirms why the head was introduced and why navL2 ≤ 0 progress.

4. **Candidate set is capable, just small-stepped.** oracle-best closes only 0.38 m
   in 3 steps (small primitives), but MPC replans over 12 blocks and the `bearing`
   oracle reaches goal 100% with the same primitives — so per-step ranking, not the
   candidate set, is the blocker. A longer horizon / finer forward primitive would
   raise the per-replan ceiling.

**Verdict:** weak-but-positive metric signal in the latent, left on the table by
(a) the easy cross-scene-negative head objective and (b) single-view deployment.
Goes to **Phase 2A**, not the representation wall (2C).

**Next (Phase 2A), iterate against this probe (seconds):**
1. Turn on `--goal-views` in the nav benchmark (free ρ +0.22 → +0.38).
2. Retrain the head with temporal **hard negatives** + a **distance-monotonicity**
   target, trained **under the multi-view min**. Goal: push ρ up and regret toward 0.
3. Try horizon 4–5 / a longer forward primitive to raise oracle-best (the ceiling).
Re-run the probe; promote to the nav benchmark only when regret drops materially.

**Caveats:** start-pose single decision, N=8, per-scene variance (e.g. af52 multi
*hurt*: single ρ +0.39 → multi +0.08). Phase 3 (32–64 scenes × 3 trials) firms it up.

## 6. Phase 2A results — metric-aware head sweep on e3 (2026-06-06)

Added cmd-integrated **distance-monotone margins** + temporal **hard negatives** to
`train_lewm_energy_head.py` (`--hard-neg-offsets`, `--dist-margin-scale`). Retrained
the head 4 ways on a fresh e3 cache (2048/512 windows, horizons 3,5,8,10, 30 ep) and
scored each with the probe (single + multi-view). `repro` = deployed recipe (no hard
negs), the apples-to-apples baseline.

```
variant     acc   | single ρ / regret / chosen | multi ρ / regret / chosen
repro      0.985  |  +0.137 / 0.330 / 1.455    |  +0.357 / 0.217 / 1.342
hardonly   0.475  |  +0.378 / 0.271 / 1.396    |  +0.022 / 0.331 / 1.456
mono2      0.480  |  +0.275 / 0.291 / 1.416    |  +0.179 / 0.313 / 1.438
mono4      0.478  |  +0.224 / 0.341 / 1.466    |  +0.331 / 0.288 / 1.413
context: init=1.50m  oracle-best=1.12m  random-pick=1.41m   (acc: repro=easy, others=hard)
```

**Findings — the head-objective hypothesis did NOT pan out; multi-view is the win:**

1. **Multi-view eval is the robust lever** — on the `repro` head it cuts regret
   0.330 → 0.217 and lifts ρ +0.14 → +0.36. This is the single biggest, cheapest
   improvement and the deployed navH never used it.

2. **Hard negatives sharply help single-view ρ (+0.137 → +0.378) but COLLAPSE
   multi-view (ρ +0.357 → +0.022).** Strong interaction: a head trained to split
   adjacent temporal steps stops being min-able across approach headings.
   Distance-monotone margins (mono2/mono4) only trade one for the other.

3. **Decisive: no retrained variant beats `repro` + multi-view on the metric that
   matters.** Every variant's *chosen* distance stays ≈ random-pick (1.40–1.47 m)
   vs oracle-best 1.12 m; the best pick remains `repro`+multi (1.342 m, regret
   0.217). Improving global rank-correlation ρ did **not** improve the argmin pick —
   the low-cost tail stays mis-ordered.

4. **`hard_acc` saturates at ~0.47 (chance).** Even trained explicitly to, the head
   **cannot separate adjacent temporal steps** — fine-grained metric structure is
   not in the latent. That is a partial **representation** limit (2C), not a head-
   objective gap. Multi-view works because it exploits the *coarser* "which approach
   heading" signal that *is* present.

### 6.1 Payoff test — does multi-view move actual NAV? NO.

Ran the full closed-loop nav benchmark on the deployed e3 head with `--goal-views 8`
vs the deployed single-view navH (same head, 8 scenes):

```
                 lewm progress   success   vs random 0.382m
single-view      +0.058 m        0%        below
multi-view (8)   -0.040 m        0%        below (slightly WORSE)
```

**Multi-view's start-pose regret win did NOT carry to closed-loop nav** — it got
marginally worse, still 0% success, still sub-random. So the start-pose cost-regret
probe is a useful *cost-quality* diagnostic but **does not predict closed-loop nav**.

Two reasons the probe over-promised:
- **First-primitive vs full-horizon mismatch.** Nav executes only the *first*
  primitive of the argmin candidate then replans; the probe scores the 3-step
  *endpoint*. The head's argmin optimizes the endpoint, which need not start with a
  progress-ward first move.
- **Compounding over 12 replans** (free/TF ≈ 4×) on a heading-dominated latent.

### 6.2 Phase 2A verdict — exhausted, negative on nav

Nothing in Phase 2A moved closed-loop nav above random: not the head retrain (probe
said it never beats `repro`+multi), not multi-view (nav got slightly worse). Combined
with **hard_acc ≈ chance** (the head cannot separate adjacent steps), the evidence now
says the per-step cost has a **representation-level ceiling**, and head-only tuning is
done.

**Fork (decision needed):**
- **(A) Cheap, first:** confirm the *first-primitive mismatch* and horizon ceiling —
  extend the probe to score the argmin's first-primitive progress, and test horizon
  4–5 / a finer forward primitive / committing >1 block per plan. May recover nav
  without retraining; will at least pinpoint the execution-vs-cost split.
- **(B) Phase 2C (expensive):** representation-level metric — aux pose/odometry or
  reachability-distance head during *backbone* training so the latent carries the
  fine metric structure (hard_acc≈chance says it currently doesn't). Backbone retrain.
- **(C) Re-scope:** lead the paper with the prediction-scaling result; treat
  closed-loop nav as documented future work backed by this diagnosis.

**Caveats:** N=8; the robust reads are nav ≈/below random across *all* configs,
hard_acc ≈ chance, and the probe→nav disconnect.

### 6.3 Path A diagnostic — CLOSED (cheap fixes rejected, 2026-06-06)

Extended the probe with **first-primitive** scoring (`navcost_h3_firststep.json`) and ran
nav across horizons:

```
first-step probe (h3):  endpoint rho/regret    first-primitive rho/regret/chosen
  head_single             +0.223 / 0.294          +0.079 / 0.085 / 1.460
  head_multi              +0.377 / 0.280          +0.067 / 0.125 / 1.500
  (first-step oracle=1.37, random=1.47 — one primitive moves only ~0.13 m)

nav horizon sweep (deployed head):  h1 -0.125 | h2 -0.036 | h3 +0.058 | h5 +0.106 | random +0.382
```

- **first_rho ≈ 0**: the cost cannot rank *first moves*; its chosen first move (~0.04 m
  closed) ≈ random (0.03 m). The endpoint regret only *looked* tolerable because the
  endpoint oracle is far (1.12 m).
- **Horizon 1 — greedy, NO first-primitive mismatch, model's most reliable 1-step
  prediction — is the WORST (-0.125 m), not the best.** So the execution mismatch is
  not the killer; longer horizon only helps marginally and stays sub-random.

**Conclusion: Path A is closed.** Three independent angles now agree the per-step cost
is representation-limited, not mis-executed or mis-tuned: h1 sub-random, first_rho ≈ 0,
hard_acc ≈ chance. The latent's metric resolution is coarser (heading-level) than the
per-step motion (~0.13 m), so no planner/horizon/head change recovers nav.

### 6.4 Where this leaves nav — the live decision

Metric-cost MPC nav from this latent is **exhausted**. The latent is a strong place-
*recognition* code (Phase A retrieval@1 ~0.42, ~21× lift) but not a *metric* one. Three
real paths:

- **D — recognition/topological planner.** Stop using the latent as a continuous metric
  cost; build nav on what it's good at — recognize places + plan over a landmark/place
  graph (subgoal recognition, not distance regression). Plays to the representation's
  strength; medium effort; no backbone retrain.
- **2C — fix the representation.** Aux pose/odometry/reachability-distance head during
  *backbone* training so the latent carries fine metric geometry. Highest-uncertainty,
  multi-day; risks degrading the prediction story. A cheap **decodability-ceiling probe**
  (regress true distance from frozen latents) should gate this before committing.
- **C — re-scope.** Lead the paper with the (strong, scaling) action-conditioned
  prediction result; ship this diagnosis as the basis for nav future-work.

## 7. "If it can't navigate, is it a good world model?" — the split

A fair objection: the world-model thesis is *propose actions → roll them out in the
predictor → pick the one that best satisfies the goal*. If nav fails, is the model
any good? The answer is that this loop has **two separable halves**, and our results
indict only the second:

1. **Forward model** — "if I take action a, what is my next latent state?" (imagine).
2. **Goal-satisfaction score** — "how well does that predicted state satisfy the
   goal?" (pick the best). This is the **value/cost function**, not the world model.

**Half 1 works and scales** — that is the actual world-model claim, and it is intact:
action-conditioned prediction beats zero/persistence, MPC win-rate ~0.66, improving
with data. As a *forward* model it is good.

**Half 2 is what's broken — and it is not the world model.** This component is always
learned separately in practice: MuZero learns a value network, Dreamer learns a
reward/value head — *nobody navigates by raw L2 distance in the dynamics latent*,
because a representation optimized for prediction/invariance is not automatically one
where Euclidean distance equals distance-to-goal. LeJEPA gives predictive,
recognition-strong features; it does **not** promise a metric. So the failure is that
**we never built a proper goal-conditioned value function**: the energy head was a
*recognition ranker* (goal-place vs other-place, contrastive), not a distance-to-goal
value *regressed against true geometry*. That is exactly why it has no far-field
gradient (first_rho ≈ 0): from far away, every candidate is equally "not yet the goal
place."

**This also reframes Path D.** D is not "graph search instead of prediction." When the
value gradient only exists locally (recognition range), you chain recognizable
subgoals, and *at each hop you still run the loop* — roll out candidate actions in the
predictor, pick the one whose predicted endpoint reaches the next subgoal. The world
model keeps doing the imagining; the subgoals just keep the goal-score in a regime
where it has gradient. So D honors the principle rather than abandoning it.

### 7.1 The decisive test — value-function decodability ceiling
`scripts/probe_latent_metric_decodability.py` trains a supervised value head
`V(z_state, z_goal) → true distance` on the **frozen** e3 latents (true geometry from
cmd integration), measuring how well distance is decodable from the latent SPACE
(independent of predictor error), overall and at fine resolution (within-band):
- **Decodable with fine resolution** → planning-by-prediction is rescued with **no
  backbone change** — we just had the wrong scorer (recognition instead of value).
- **Only coarse / not at the ~0.13 m per-step scale** → the metric isn't in the latent
  → backbone change (2C) or subgoal chaining (D).

### 7.2 Result — distance is NOT decodable from the projected latent (e3)

`latent_metric_decodability_e3.json`, eval pairs (z_proj, within-window, true dist
0.08–1.08 m):

```
value head : pearson +0.054  spearman +0.083  MAE 0.257 m   (predictions collapse to the ~0.3 m mean)
bare L2    : pearson +0.095  spearman +0.082
within-band Spearman:  [0,0.2]+0.015  [0.2,0.5]+0.000  [0.5,1.0]+0.002  [1.0,3.0]+0.059
```

A value head trained *directly* to regress true distance from frozen `z_proj` pairs
gets **~zero** correlation (pearson 0.05, no better than raw L2), with no resolution at
any band — it just predicts the mean distance. So the projected latent does **not**
encode pairwise distance, even coarsely. This reconciles the earlier multi-view
endpoint ρ≈+0.38: that was heading/appearance matching weakly correlated with endpoint
distance in one candidate set, **not** true position-metric signal — which is why it
never converted to a usable pick or to nav.

**`z_raw` (pre-projection encoder features):** better than `z_proj` but still far too
weak, and it **overfits** (eval pearson peaks ~0.22 at ep30, decays to 0.177 while
train Huber keeps dropping):

```
value head (z_raw): pearson +0.177 (peak ~0.22)  spearman +0.177  MAE 0.216 m
bare L2 (z_raw)   : pearson +0.114
within-band Spearman:  [0,0.2]+0.018  [0.2,0.5]+0.073  [0.5,1.0]+0.041  [1.0,3.0]+0.074
```

So the projector *does* discard metric info (z_proj 0.05 → z_raw 0.18), but the **raw
encoder features only weakly encode COARSE distance and have essentially NO fine
resolution** (within-band ≈ 0 everywhere) — nowhere near the per-step ~0.13 m
discrimination nav needs, and it overfits from a tiny true signal.

### 7.3 Verdict — the metric is not in the representation

A value head on the frozen latent **cannot** regress distance: z_proj ≈ 0, z_raw ~0.18
coarse-only with no fine resolution. The cheap "we just needed a value function instead
of recognition" fix is **closed** — the information isn't there to decode. Combined
with hard_acc ≈ chance, first_rho ≈ 0, and nav ≤ random at every horizon, the
conclusion is firm: **the forward model (half 1) is good and scales; the goal/value
half cannot be built on this latent because LeJEPA's encoder doesn't capture metric
geometry.**

That leaves exactly the three paths, now evidence-locked:
- **2C** — change the *backbone* objective to inject metric structure (aux
  pose/odometry/reachability head into encoder training). The only route to *pure*
  latent value-planning; expensive, uncertain, risks the prediction story.
- **D** — recognition/subgoal planner. With the metric absent, this is **not** a
  compromise of the principle but the *only* way to run propose→roll-out→score on a
  recognition-strong / metric-poor latent: keep each subgoal within recognition range
  (the latent's proven strength) so the per-hop score has gradient.
- **C** — ship the prediction-scaling result; this diagnosis becomes the precise,
  defensible statement of what the representation needs next.

## 8. Decision: no-privileged-runtime-geometry navigation (Path 2C) — 2026-06-06

Chosen: make the **latent itself** metric so the propose→roll-out→score loop works
with no privileged geometry. (Audit finding: the TinyQuadJEPA-v2 reference
`5_genesis_eval.py` never navigated on the latent either — its cost was
`0.75·true_distance + 0.30·true_heading + energy_weight·latent_energy`, i.e.
privileged geometry with the world model as a faded garnish, on a salient red-pillar
beacon. Our pure-latent test was the honest, harder one.)

**Objective:** auxiliary relative-pose head `RelPoseHead: P(z_a, z_b) → (dx, dy, dyaw)`
in a's body frame (`lewm/models/pose_head.py`), smooth-L1 vs true geometry; the loss
backprops into the **encoder**, forcing it to retain the distance/bearing it currently
discards (keeps yaw R²0.81, loses position decodability ≈0.05). Nav cost becomes
`‖predicted dxy(z_state, z_goal)‖` — the model's own distance-to-goal.

**Primary labels:** aligned physical replay pose from `frames.jsonl`, verified
against rendered global frame indices. Command-integrated relative pose remains
an explicit ablation because commanded and realized quadruped motion differ.

**Status:** the training, checkpoint-resume, pose-cost policy, physical-pose cache,
encoded/predicted geometry probe, candidate-level pose-cost diagnostic, and
failure-safe per-checkpoint watcher are wired and compile-checked. Focused pose and
checkpoint tests pass. No GPU fine-tune is approved until the controlled proxy
ladder selects the auxiliary loss scale.

**Plan:** run the fixed e3 frozen-head, continuation, encoded-only, and
encoded-plus-predicted controls from
`docs/lewm_pose_aux_experiment_design_2026-06-06.md`; promote only a cell that
improves physical geometry and first-action ranking without regressing prediction.

### 8.1 Design hardening update — 2026-06-06

The command-integrated, encoded-pair-only, `lambda_pose=1.0` launch above is
superseded by `docs/lewm_pose_aux_experiment_design_2026-06-06.md`.

The primary experiment now uses aligned physical replay poses, bidirectional and
horizon-balanced pairs, wrapped yaw loss, and an additional
predictor-endpoint-to-encoded-goal objective matching the deployed MPC contract.
Command integration remains a label-source ablation. Loss weights must be selected
from measured gradient scale on a controlled proxy; they are not assumed to be 1.0.

No fine-tune is promoted unless it improves physical-pose geometry, including
within-distance-band and predictor-to-goal metrics, while preserving the 8k e3
forward-model gates. The runtime claim is explicitly limited to **no privileged
geometry at deployment** because physical pose is used as a training auxiliary.
