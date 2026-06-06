# LeWM: Diagnosing a LeJEPA World Model as a Planning Substrate — Story, Evidence, Roadmap (2026-06-04)

This is the consolidated paper narrative. It records what we built, what failed,
the ablation that ruled out the obvious fix, the two mechanistic diagnostics that
localized the real failures, and the recommended (logged) plan. It is written so
the experiments and figures map directly onto paper sections.

## Thesis (abstract-in-one-paragraph)

A LeJEPA-style latent world model (ViT-Tiny encoder + transformer predictor +
SIGReg anti-collapse) trained on textured quadruped maze rollouts learns a useful
**place-recognition** code but fails as a **planning substrate**: its latent
distance is not metric (does not track topological distance) and its rollout is
useful only to ~1 s. We first show the obvious representation knobs do **not**
fix this — a scaled λ (SIGReg-strength) × source-mix factorial leaves the metric
correlation flat (rho ≈ 0.03–0.05, vs a 0.40 "usable" bar). We then localize the
failure with two targeted probes: (1) the latent is **heading-dominated** — yaw is
decodable at R²≈0.81 while position is at R²≈0.16, and a place at a new heading is
≈ as far in latent space as a *different* place; (2) the short prediction horizon
is **autoregressive compounding**, not unpredictable targets — single steps from
the true latent beat persistence 3–20× out to 5 s, while free-running rollout
drifts to 23× the per-step error. These are **separable** failures with separate
fixes: a rollout-stability training objective (horizon) and an
orientation-disentangled encoder (place code). Contribution: a reusable diagnostic
methodology that says *which part* of a JEPA world model is broken, plus a
cautionary negative ablation (tuning the wrong knob).

## 1. System under test

- Encoder: ViT-Tiny; Predictor: 6-layer causal transformer; ~16M params total.
- Objective: JEPA latent prediction loss + λ·SIGReg (isotropic-Gaussian
  anti-collapse), λ=0.09 (confirmed from the e9 `model_config`).
- Training window: `max_seq_len=4`, `stride=5` (≈0.5 s/macro-step); the loss
  supervises **1-step-ahead** prediction inside that 4-step window.
- Data: textured v03 renders, single forward camera, partial observability +
  visual aliasing; 1450 scenes; ~48k env-sessions; production run = 9.24M
  windows/epoch × 10 epochs.
- Use case: latent state for receding-horizon (MPC/CEM) planning to a goal.

## 2. Phase-A diagnosis — what works, what doesn't

(Artifacts: `lewm_state_diagnostic_2026-06-04.md`,
`lewm_final_checkpoint_benchmark_2026-06-04.md`.)

- **Place recognition: good.** A3 retrieval@1 ≈ 0.40, lift ≈ 21× over chance.
- **Metric geometry: poor.** A2 projected Spearman rho ≈ 0.029; latent distance is
  flat across graph-distance buckets; a trained reachability head is ≈ majority
  baseline (+1–2 pp) cross-scene.
- **Prediction: short.** Rollout beats persistence at h=1–2 (≤1 s), loses from
  h=3 onward.

Conclusion: the model recognizes places but the embedding is not a map, and its
imagined future is stable only for ~1 s.

## 3. Hypothesis 1 (representation knobs) — RULED OUT

We hypothesized SIGReg isotropy pressure was flattening metric structure, and/or
the route-teacher-heavy source mix biased the marginal. We ran a scaled
factorial (λ ∈ {0.09, 0.03, 0.01} × {uniform, exploratory source mix}),
size-matched (19,061 windows/cell), 3 epochs / 300 sessions.

**Proxy validity:** the `uniform:0.09` control reproduces the production
diagnosis (rho_proj 0.0297 vs production 0.0288; retrieval lift 11× vs 21×, weaker
but well above chance) at ~1600× less training — so the metric failure is not a
training-budget artifact and the sweep is interpretable.

| cell | rho_proj | retr@1 | lift | eval_std | train_sig | action-sens |
|---|---:|---:|---:|---:|---:|---:|
| PROD e9 (λ0.09, full) | 0.0288 | 0.396 | 20.97× | 0.757 | 0.74 | 0.0377 |
| uniform λ0.09 (control) | 0.0297 | 0.276 | 11.18× | 0.695 | 4.39 | 0.0033 |
| uniform λ0.03 | 0.0251 | 0.238 | 10.46× | 0.533 | 6.81 | 0.0003 |
| uniform λ0.01 | 0.0254 | 0.172 | 7.50× | 0.371 | 12.85 | 0.0000 |
| exploratory λ0.09 | 0.0453 | 0.244 | 9.75× | 0.639 | 4.80 | 0.0017 |
| exploratory λ0.03 | 0.0467 | 0.228 | 11.09× | 0.562 | 7.23 | 0.0001 |
| exploratory λ0.01 | 0.0321 | 0.175 | 7.93× | 0.366 | 13.74 | 0.0000 |

**Verdict:** λ is **orthogonal** to the metric axis — lowering it leaves rho flat
(~0.025–0.03) and *degrades* recognition (lift 11→7.5×) and dispersion
(eval_std 0.70→0.37). The exploratory source mix is a *mild* positive lever on rho
(~0.030→~0.046) but still ~10× short of 0.40 and within per-scene noise (SE≈0.015).
No cell — alone or combined — approaches a usable metric code. The fix is not a
hyperparameter; it is architectural/objective. (This is the paper's cautionary
subplot: a plausible knob, cleanly falsified.)

## 4. Mechanistic diagnostics — localizing the real failures (key contribution)

Two targeted probes on the production e9 checkpoint (`test_id`, 32 held-out
scenes), reusing the A2/A3 latent banks.

### 4.1 View-vs-place probe (`scripts/probe_lewm_yaw_invariance.py`)

Question: is the latent a *place* code or a *view* code? Two readouts, on
held-out frames, in the projected (planning) space:

- **Same-place-across-yaw vs across-place latent distance.** `yaw_place_ratio` =
  median(within-cell, different-heading L2) / median(across-cell L2).
- **Decode yaw vs position** from the latent (held-out ridge, identical
  frames/split): yaw as [sin,cos] of the bin angle (8 bins), position as
  cell-centre xy.

| measure | value | reading |
|---|---:|---|
| decode yaw R² | **0.81** | heading almost fully recoverable |
| decode position R² | **0.16** | location barely recoverable |
| yaw_place_ratio | **0.96** | new heading ≈ as far as a new place |

**Conclusion: the latent is heading-dominated.** Position is a weak secondary
axis. The A3 "place recognition" was substantially *view* recognition: the same
(x,y) at a different yaw lands almost as far away as a different cell. Artifact:
`models/checkpoints_textured_v03_full_20260531/sweep_seq4/yaw_invariance_e9_final_testid.json`.

### 4.2 Rollout decomposition (`scripts/probe_lewm_rollout_horizons.py`, extended)

Question: does the horizon die because (a) the predictor compounds error, or
(b) the target latent is intrinsically unpredictable? We compare **teacher-forced**
(predict one step from the TRUE latent) vs **free-running** (autoregressive).

| horizon | teacher-forced 1-step | free-running | persistence | free-run / teacher-forced |
|---:|---:|---:|---:|---:|
| 1 (0.5s) | 0.050 | 0.050 | 0.091 | 1.0 |
| 2 (1.0s) | 0.031 | 0.137 | 0.188 | 4.4× |
| 3 (1.5s) | 0.043 | 0.302 | 0.289 | 7.1× |
| 5 (2.5s) | 0.033 | 0.476 | 0.460 | 14.5× |
| 8 (4.0s) | 0.032 | 0.733 | 0.643 | 23× |
| 10 (5.0s) | 0.040 | 0.876 | 0.743 | 22× |

Teacher-forced 1-step error is **flat and low** (~0.03–0.05) at every horizon and
**beats persistence 3–20×** out to 5 s. Free-running explodes to **23×** the
per-step error. (Sanity: at h=1 teacher-forced == free-running exactly.)

**Conclusion: the per-step dynamics head is competent; the horizon dies purely
from autoregressive compounding** — expected, since the model was only ever
trained to predict 1 step ahead in a 4-step window, never to roll out. This is a
training-objective defect, **not** a representation defect. Artifact:
`models/checkpoints_textured_v03_full_20260531/sweep_seq4/rollout_decomp_e9_final.json`.

## 5. Synthesis — two separable failures

The two stated priorities have **different root causes** (this refined an earlier
"single root cause" guess — the decomposition shows the per-step model predicts
the view-latent fine, so the short horizon is *not* caused by the view-code):

1. **Place/metric code** is broken because the encoder is heading-dominated →
   **representation** fix.
2. **Prediction horizon** is short because of compounding → **training-objective**
   fix; the dynamics are already good per-step.

They can be fixed independently and in parallel.

## 6. The fixes

### 6.1 Prediction horizon — rollout-stability objective (highest ROI, low risk)

The per-step model is good; train it to *roll out*:

- **Temporal-pyramid / multi-step loss:** supervise predictions at h=1 AND
  h=2,3,5 (then 8,10), downweighting longer horizons. Keep h=1 for control
  sensitivity.
- And/or **scheduled sampling / latent noise:** during training, feed the model's
  *own* predicted latents (with probability ramping up) so it learns to recover
  from its own errors — directly attacks compounding.
- Likely needs `max_seq_len` > 4 so longer targets exist in-window.
- **Success metric:** re-run §4.2 (free-run/teacher-forced ratio should flatten)
  and the rollout gate (rollout beats persistence past h=3, target multi-second).

### 6.2 Yaw-robust encoder — orientation-disentangled place code (representation)

Goal: make the *place* component invariant to heading while keeping heading
available for control/prediction. Split the embedding:

  z = [ z_place , z_pose ]

- **z_place** — orientation-invariant "where am I." Trained with a
  **place-consistency** term: pull together latents of the same ground-truth cell
  at different yaw bins (we have cell+yaw labels in `labels.jsonl`), via an L2
  invariance loss or, better, an **InfoNCE contrastive** loss (positives = same
  cell any heading; negatives = other cells). This *directly* targets the measured
  defect (yaw_place_ratio 0.96 → want ≪1; pos-decode R² 0.16 → want ↑).
- **z_pose** — carries heading. Trained to predict yaw (small supervised head on
  yaw_bin, or self-supervised by integrating the commanded yaw action).
- **Dynamics predictor** runs on the full [z_place, z_pose] (so it can still
  predict view changes); the **planning cost** and **place recognition** use only
  z_place (the stable code).
- Keep SIGReg (anti-collapse) on the full latent; invariance/contrastive shapes
  *which* directions carry place vs pose.
- Cheaper first cut (no architectural split): add the place-invariance term to the
  existing single latent — but risks erasing heading the predictor needs, so the
  split is the principled version.
- **Design honesty for the paper:** using cell labels makes z_place training
  partly *supervised* (a privileged auxiliary). Note self-supervised alternatives
  — define "same place" via action/odometry integration or tight temporal
  proximity — and ideally report both.
- **Success metric:** re-run §4.1 (yaw_place_ratio ↓, pos-decode R² ↑, yaw−pos R²
  gap ↓) and A2/A3 (rho_proj ↑, retrieval preserved).

### 6.3 Topology/metric (original Phase B) — LAST

Explicit latent-distance supervision (e.g., regress latent distance toward
graph/odometry distance, or a spatial-neighbor contrastive). Some of it may follow
once z_place is orientation-stable; revisit after 6.1/6.2.

## 7. Recommended plan (LOGGED — what I think is best)

1. **Start the rollout-stability objective first** (§6.1). Highest ROI, low risk,
   directly answers the "predict >1 s" requirement, and the diagnostics say it
   should work because per-step prediction is already strong. Run as a
   training-objective ablation on the validated scaled proxy (cheap), measured by
   §4.2 + the rollout gate.
2. **In parallel, prototype the place⊥pose encoder** (§6.2) on the proxy, measured
   by §4.1 + A2/A3.
3. **Then topology** (§6.3), only if §6.2 leaves rho short.
4. For the **paper**, the publishable core is already in hand even before the
   fixes land: the diagnostic methodology (§4) + the falsified knob (§3) +
   the separable-failures framing (§5). The fixes can be "results" or
   "future work" depending on how far we get.

## 8. Reusable diagnostic methodology (a contribution in itself)

- **View-vs-place probe**: decode yaw vs position from a frozen latent +
  same-place-across-yaw distance ratio → tells you if a "place code" is really a
  view code. Generalizes to any egocentric visual world model.
- **Teacher-forced vs free-running rollout decomposition**: separates per-step
  dynamics quality from autoregressive compounding → tells you whether to fix the
  predictor's *training* or the *target representation*. Avoids the trap of
  redesigning the encoder when the dynamics head is fine (or vice versa).

## 8.1 Related-work calibration: H-JEPA / DINO-WM and the substrate-vs-hierarchy lesson

We considered moving to a hierarchical JEPA (H-JEPA), citing HWM (Hierarchical
Planning with Latent World Models) which reaches long-horizon Push-T. Key reading
of that line of work, calibrated against our diagnostics:

- **DINO-WM** (the substrate HWM builds on) gets long-horizon Push-T from the
  *representation*, not hierarchy: a **frozen DINOv2 encoder** producing **spatial
  patch-token** latents, an action-conditioned ViT predictor (causal,
  frame-level autoregressive — i.e. flat, not hierarchical), MPC planning with an
  MSE latent cost. It rolls out far because the targets are strong, stable, and
  spatially grounded, and Push-T is **fully observed and low-aliasing**.
- **HWM** adds temporal abstraction on top: a high-level planner optimizes
  macro-actions (sequences of primitives via an action encoder) over a
  long-horizon model → first predicted latent = subgoal → low-level short-horizon
  model fills in primitives. Result: ~3× cheaper planning, matched/greater
  success, longer effective horizon. No pose/topology/place supervision (zero-shot,
  goal-only).

**Implication for LeWM.** Our substrate is the *opposite* of DINO-WM's:
from-scratch ViT-Tiny, a **single pooled global vector** (not patch tokens),
**egocentric, partially observed, highly aliased**. That is exactly why our flat
autoregressive rollout compounds to 23× by h=10 while DINO-WM's flat rollout stays
usable far longer on Push-T. Therefore:

1. **H-JEPA hierarchy will not, by itself, reproduce the Push-T result** — that
   result is substrate-driven (frozen strong patch features + full observability),
   with hierarchy as an efficiency/horizon multiplier on top.
2. **It won't "break" anything either**: hierarchy is orthogonal to the view-code,
   and (like DINO-WM/HWM) needs **no place supervision** — consistent with the
   decision to skip the supervised place loss.
3. **Highest-leverage, in order, for the horizon (our #1 problem):**
   (a) **train for rollout** on the current model (multi-step / scheduled sampling)
   — cheapest, and justified because §4.2 shows per-step dynamics are already good;
   (b) **substrate upgrade** à la DINO-WM — predict **spatial patch tokens** instead
   of a single global vector, and/or use **stronger/frozen pretrained features** —
   more predictable targets, less whole-vector swing under rotation;
   (c) **history/belief** to handle partial observability + aliasing (Push-T has
   neither; our maze has both) — this is also the real route to topology if ever
   wanted, not a distance loss on single-frame latents;
   (d) **then hierarchy (H-JEPA/HWM)** for very-long horizons + cheap planning.
4. **View-code × hierarchy interaction:** heading is the *fast* dominant variable,
   which is backwards for clean temporal abstraction (higher levels want *slow*
   variables). But an unsupervised temporal-abstraction objective may learn to
   abstract heading away at the top — a possible **label-free** path toward a more
   place-like code, aligned with skipping supervised place.

Topology de-prioritized (deliberate): in an aliased POMDP a metric from single
views is close to ill-posed (a human shown two aliased views also can't give a
distance) — it is a **belief/memory** problem, not an encoder-tuning one.

## 9. Artifact index

- Diagnostics: `…/sweep_seq4/yaw_invariance_e9_final_testid.json`,
  `…/sweep_seq4/rollout_decomp_e9_final.json`,
  `…/sweep_seq4/latent_aliasing_e9_final_testid.json`,
  `…/sweep_seq4/reachability_a3_e9_final_testid.json`,
  `…/sweep_seq4/planning_gate_lewm_seq4_e9_final.json`.
- Scaled factorial: `models/checkpoints_textured_v03_scaled_ablation_20260604/seed0/{uniform,exploratory}_sig0p0{9,3,1}_maxsess300/` (+ `probes/`).
- Rollout objective fast path:
  `models/checkpoints_textured_v03_full_20260531/sweep_seq4/predictor_only_stage1/`
  for cached-latent Stage 0/1; image-proxy Stage 2 seq4 in
  `models/checkpoints_textured_v03_rollout_stage2_20260604/seq4_rollout_lam0p5_h3_maxsess300/`
  and seq11 in
  `models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p5_h10_maxsess300/`.
- Scripts: `scripts/probe_lewm_yaw_invariance.py` (new),
  `scripts/probe_lewm_rollout_horizons.py` (teacher-forced extension),
  `scripts/cache_lewm_latents.py`, `scripts/train_lewm_predictor_only.py`,
  `scripts/run_lewm_scaled_factorial_ablation.sh`, `scripts/eval_lewm_ablation_probes.sh`.
- Prior docs: `lewm_state_diagnostic_2026-06-04.md`,
  `lewm_final_checkpoint_benchmark_2026-06-04.md`,
  `lewm_scaled_ablation_decisions_2026-06-04.md`.

## 10. Paper narrative arc (five beats)

1. We want a latent world model for visual maze planning; LeJEPA is the natural
   self-supervised recipe.
2. It learns to *recognize* places but not to *measure* them, and predicts only
   ~1 s — neither is enough for planning.
3. The obvious representation knobs (SIGReg λ, source mix) **don't** fix the metric
   code — a clean negative.
4. Two diagnostics localize the real causes: a **heading-dominated** code (place is
   washed out by orientation) and **autoregressive compounding** (good per-step
   dynamics, no rollout training) — *separable*.
5. Targeted fixes follow from the diagnosis: orientation-disentangled encoder +
   rollout-stability objective; topology last. The methodology transfers to other
   egocentric world models.

## 11. Option 1 implementation scope — rollout-training objective

Diagnosis recap: training optimizes only the 1-step teacher-forced loss
(`lewm/models/lewm.py:157-187`), so the predictor never sees its own predictions
→ autoregressive compounding (free-run/teacher-forced ≈ 22× at h=10, §4.2). The
free-running rollout loss we want is *already computed at eval*
(`scripts/train_lewm.py:532-535`) and is differentiable — we just add it to the
training objective.

Code changes:
- **`LeWorldModel.forward()`** — add a horizon-weighted free-running rollout term
  beside the existing 1-step loss (keep 1-step for immediate control sensitivity):
  ```
  H = min(rollout_horizon or T-1, T-1)
  roll_proj = self.plan_rollout(z_raw[:,0], cmd_seq[:, :H])      # (B,H,D)
  per_step  = (roll_proj - z_proj[:, 1:H+1]).square().mean(-1)   # targets non-detached (matches 1-step)
  rollout_loss = (per_step * w).sum(-1).mean() / w.sum()         # w = γ^(t-1) downweight
  total = pred_loss + sigreg_lambda*sig_loss + rollout_lambda*rollout_loss
  ```
- **`LeWorldModel.__init__`** — `rollout_lambda` (default 0.0 = current behavior),
  `rollout_horizon`, `rollout_gamma`. Scheduled sampling remains Tier B, not
  implemented in the first pass.
- **`scripts/train_lewm.py`** — `--rollout-lambda/-horizon/-gamma` flags; log
  `out["rollout_loss"]`; record in `model_config` (line ~646) + resume-compat check
  (~693); `--device cuda` hard-fails if CUDA is unavailable (no silent CPU
  fallback). Default rollout objective off → all existing training behavior is
  preserved when the new flags are omitted.
- **`max_seq_len`** — to reach multi-second, train at `max_seq_len≈11` (stride 5 →
  5 s windows → 10-step rollout). Resizes `predictor.pos_embed` ⇒ FRESH run (no
  resume from seq4); ~11 frames/sample vs 4.

Tiers: **A (minimal, first):** the sketch above + `rollout_lambda` warmup (0→target
after epoch 1) + grad-clip. **B (robust, if A wobbles):** step loop with scheduled
sampling (feed true latent w.p. p, ramp 1→0.3) + horizon curriculum (2→K).

Success metrics (reuse `probe_lewm_rollout_horizons.py`): free-run/teacher-forced
ratio at h=10 drops from ~22 toward ~1–3; rollout point-MSE beats persistence at
h≥3. Guardrails: 1-step `pred_loss` and action-sensitivity don't regress;
`eval_std` stays sane. Note: fixes horizon only; does not touch the view-code.

## 12. Fast-validation ladder — assess efficacy in minutes, not days

The point is to spend the expensive (image, end-to-end, multi-day) run only after
a cheap screen says the objective works. The compounding fix is a *predictor*
question, so we can isolate the predictor and skip the encoder + image I/O.

- **Stage 0 — cache frozen-e9 latents (one-time, minutes).** Run the frozen e9
  encoder over a subset of windows at `max_seq_len≈11`, save `(z_raw, z_proj, cmd)`
  tensors. Latents are tiny vs PNGs.
- **Stage 1 — predictor-only rollout ablation on cached latents (MINUTES for a
  whole sweep).** Train a fresh predictor (+pred_projector) on cached latents,
  with/without the rollout loss, sweeping `rollout_lambda`, `gamma`, `max_seq_len`,
  Tier A/B. No images, no encoder fwd/bwd → seconds–minutes per config. Measure the
  decomposition on held-out cached latents. **This doubles as a preview of the
  DINO-WM-style "freeze encoder, predict cached features" path.**
  - Decision gate (asymmetric): Stage-1 **positive** (ratio flattens, beats
    persistence at h≥3) = strong go. Stage-1 **negative** is *inconclusive* —
    end-to-end may still help because the encoder can co-adapt to be more
    roll-out-able — so fall through to Stage 2.
- **Stage 2 — image proxy, end-to-end (~5–15 min/cell).** The validated scaled
  proxy (max-sessions 300, 3 epochs; eval-load already ~30 s after the
  `_load_corpus` fix) with the rollout loss, `max_seq_len` 4 then 11. Confirms the
  cached-latent result holds when the encoder co-adapts.
- **Stage 3 — full run (days).** Only after Stages 1–2 are positive.

Note: the scaled proxy is itself already minutes/cell (not days), so even without
Stage 1 the efficacy read is fast; Stage 1 is the *fastest* tier and the cleanest
isolation of the predictor/objective effect.

## 13. Stage 0/1 result — cached-latent rollout objective screen

Completed on GPU only (ROCm venv `genesis_render_vulkan`, no CPU fallback). The
thread-cap environment variables `OMP_NUM_THREADS` / `MKL_NUM_THREADS` made
`torch.cuda.is_available()` report false in this environment, so GPU runs must
avoid those env assignments.

Artifacts:
- Caches:
  `models/checkpoints_textured_v03_full_20260531/sweep_seq4/predictor_only_stage1/train_seq11_n8192_gpu.pt`
  (8,192 train windows from the max-sessions-300 train slice) and
  `.../eval_seq11_n2048_gpu.pt` (2,048 windows from the 2% eval scene holdout).
- Stage-1 JSONs:
  `.../e9init_evalonly_seq4ctx_n8192_eval2048.json`,
  `.../e9init_onestep_ft20_seq4ctx_n8192_eval2048.json`,
  `.../e9init_rollout_lam0p5_ft20_seq4ctx_n8192_eval2048.json`.

Matched cached-latent screen (all initialized from production
`lewm_seq4_e9.pt`, seq4 predictor context, frozen e9 encoder latents):

| cell | h | teacher-forced | free-running | persistence | free / TF | free / persistence |
|---|---:|---:|---:|---:|---:|---:|
| e9 baseline | 1 | 0.039 | 0.039 | 0.075 | 1.00 | 0.52 |
| e9 baseline | 3 | 0.042 | 0.296 | 0.258 | 7.05 | 1.15 |
| e9 baseline | 10 | 0.040 | 0.868 | 0.733 | 21.96 | 1.18 |
| one-step fine-tune | 1 | 0.074 | 0.074 | 0.075 | 1.00 | 0.98 |
| one-step fine-tune | 3 | 0.077 | 0.322 | 0.258 | 4.19 | 1.25 |
| one-step fine-tune | 10 | 0.073 | 1.137 | 0.733 | 15.51 | 1.55 |
| rollout-loss fine-tune | 1 | 0.056 | 0.056 | 0.075 | 1.00 | 0.74 |
| rollout-loss fine-tune | 3 | 0.059 | 0.171 | 0.258 | 2.90 | 0.66 |
| rollout-loss fine-tune | 10 | 0.057 | 0.535 | 0.733 | 9.41 | 0.73 |

Interpretation:
- **Positive for the rollout objective.** A 20-epoch predictor-only rollout-loss
  fine-tune cuts h=10 free-running MSE from `0.868` to `0.535` and makes rollout
  beat persistence at h≥3. The free/teacher-forced compounding ratio drops from
  ~22× to ~9× at h=10.
- **One-step loss alone is the wrong fix.** The matched one-step-only fine-tune
  lowers train one-step loss but worsens held-out rollout (`0.868` → `1.137` at
  h=10) and loses harder to persistence. This directly supports adding the
  free-running rollout term, not merely training longer on one-step prediction.
- **Guardrail/caveat.** Rollout-loss fine-tuning slightly worsens held-out
  teacher-forced h=1 (`0.039` → `0.056`), though it remains better than
  persistence. Stage 2 should therefore use a mixed objective, warmup, horizon
  weights, and monitor h=1 teacher-forced loss; consider lower rollout λ or
  scheduled sampling if h=1 regresses in the image proxy.

Decision: Stage 1 is green. Proceed to Stage 2 image-proxy implementation of
Tier A rollout loss (`max_sessions=300`, first seq4 to verify objective, then
fresh seq11 for the real 5 s horizon).

## 14. Stage 2 result — image proxy seq4 rollout objective

Completed on GPU only. Also deleted the stale CPU fallback cache artifact
`models/checkpoints_textured_v03_full_20260531/sweep_seq4/predictor_only_stage1/train_seq11_n2048.pt`
so only explicitly GPU-generated Stage 1 caches remain in the active record.

Implementation landed:
- `lewm/models/lewm.py`: Tier A horizon-weighted free-running rollout loss in
  `LeWorldModel.forward()` plus returned `rollout_loss` / effective
  `rollout_lambda`.
- `scripts/train_lewm.py`: direct repo-root import fix; rollout objective CLI
  flags; model-config / resume guards; metrics logging; eval logging; explicit
  `--device cuda` hard fail if CUDA is not visible.

Seq4 image-proxy command:
```
.generated/venvs/genesis_render_vulkan/bin/python scripts/train_lewm.py \
  --data-root .generated/datagen_full \
  --render-root .generated/datagen_full/render_textured_v03 \
  --allow-material-color-render \
  --out-dir models/checkpoints_textured_v03_rollout_stage2_20260604/seq4_rollout_lam0p5_h3_maxsess300 \
  --max-seq-len 4 --stride 5 --max-sessions 300 --epochs 3 --batch-size 128 \
  --sigreg-lambda 0.09 --rollout-lambda 0.5 --rollout-horizon 3 \
  --rollout-gamma 0.9 --rollout-warmup-epochs 1 \
  --eval-max-batches 32 --device cuda
```

Trainer eval trend:

| epoch | eval pred | eval rollout loss | eval rollout pred | train pred | train rollout |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.236 | 0.376 | 0.386 | 0.185 | 0.257 |
| 1 | 0.213 | 0.335 | 0.344 | 0.130 | 0.203 |
| 2 | 0.185 | 0.295 | 0.303 | 0.116 | 0.183 |

Matched decomposition probe, same 2% eval scene holdout, 256 samples:

| model | h | teacher-forced | free-running | persistence | free / TF | free / persistence |
|---|---:|---:|---:|---:|---:|---:|
| no-rollout seq4 proxy | 1 | 0.188 | 0.188 | 0.179 | 1.00 | 1.05 |
| no-rollout seq4 proxy | 3 | 0.209 | 0.531 | 0.426 | 2.54 | 1.25 |
| no-rollout seq4 proxy | 10 | 0.258 | 1.321 | 0.815 | 5.13 | 1.62 |
| rollout-loss seq4 proxy | 1 | 0.159 | 0.159 | 0.163 | 1.00 | 0.98 |
| rollout-loss seq4 proxy | 3 | 0.137 | 0.316 | 0.357 | 2.30 | 0.89 |
| rollout-loss seq4 proxy | 10 | 0.127 | 0.825 | 0.669 | 6.50 | 1.23 |

Interpretation:
- **Stage 2 seq4 is positive.** The image-proxy result matches the Stage 1
  direction: free-running MSE drops strongly (`h3: 0.531 → 0.316`,
  `h10: 1.321 → 0.825`) and rollout beats persistence through `h=3` (1.5 s).
- **It is not solved at seq4.** The model was trained with only 3 transitions of
  rollout supervision, so it still loses to persistence at `h=8/10` and behaves
  close to zero-action/shuffled-action at the far horizon. This is exactly why the
  next proxy must use `max_seq_len=11` and a 10-step rollout horizon.
- **Guardrail passed.** Unlike the cached-latent predictor-only screen, the
  image-proxy rollout objective did *not* hurt the teacher-forced one-step column
  on this matched holdout (`0.188 → 0.159` at `h=1`; `0.258 → 0.127` at `h=10`).

Decision after seq4: proceed to fresh seq11 Stage 2 proxy (`max_seq_len=11`,
`rollout_horizon=10`, same λ=0.5/gamma=0.9 warmup recipe, smaller batch if
needed for memory). Success criterion: free-running point MSE beats persistence
through at least `h=8`, ideally `h=10`, and h10 free/teacher-forced moves toward
the 1-3× band.

## 15. Stage 2 result — seq11 5 s rollout objective

Completed on GPU only. This is the first image-proxy run trained on the real
10-transition / 5-second objective instead of asking a seq4 model to extrapolate
beyond its training window.

Seq11 image-proxy command:
```
.generated/venvs/genesis_render_vulkan/bin/python scripts/train_lewm.py \
  --data-root .generated/datagen_full \
  --render-root .generated/datagen_full/render_textured_v03 \
  --allow-material-color-render \
  --out-dir models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p5_h10_maxsess300 \
  --max-seq-len 11 --stride 5 --max-sessions 300 --epochs 3 --batch-size 64 \
  --sigreg-lambda 0.09 --rollout-lambda 0.5 --rollout-horizon 10 \
  --rollout-gamma 0.9 --rollout-warmup-epochs 1 \
  --eval-max-batches 32 --device cuda
```

Trainer eval trend:

| epoch | eval pred | eval rollout loss | eval rollout pred | train pred | train rollout | eval zero delta | eval shuffle delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.066 | 0.185 | 0.207 | 0.109 | 0.224 | 0.000 | 0.000 |
| 1 | 0.074 | 0.196 | 0.217 | 0.069 | 0.175 | 0.003 | 0.003 |
| 2 | 0.102 | 0.279 | 0.310 | 0.060 | 0.154 | 0.005 | 0.005 |

The trainer metric overfits after epoch 0/1 even while train loss keeps
improving, so checkpoint choice should use the horizon decomposition probe.

Matched decomposition probe, same 2% eval scene holdout, 256 samples:

| checkpoint | h | teacher-forced | free-running | persistence | free / persistence | free / TF | zero - free | shuffled - free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seq11 e0 | 1 | 0.081 | 0.081 | 0.067 | 1.21 | 1.00 | 0.001 | 0.001 |
| seq11 e0 | 3 | 0.091 | 0.194 | 0.196 | 0.99 | 2.13 | 0.000 | 0.000 |
| seq11 e0 | 8 | 0.109 | 0.329 | 0.373 | 0.88 | 3.02 | 0.001 | 0.002 |
| seq11 e0 | 10 | 0.072 | 0.362 | 0.416 | 0.87 | 5.04 | -0.000 | 0.002 |
| seq11 e1 | 1 | 0.085 | 0.085 | 0.075 | 1.13 | 1.00 | -0.000 | 0.005 |
| seq11 e1 | 3 | 0.091 | 0.190 | 0.192 | 0.99 | 2.08 | 0.003 | 0.018 |
| seq11 e1 | 8 | 0.089 | 0.297 | 0.314 | 0.95 | 3.35 | -0.007 | 0.032 |
| seq11 e1 | 10 | 0.096 | 0.337 | 0.365 | 0.92 | 3.52 | -0.004 | 0.032 |
| seq11 e2 | 1 | 0.124 | 0.124 | 0.120 | 1.04 | 1.00 | 0.007 | 0.011 |
| seq11 e2 | 3 | 0.104 | 0.223 | 0.240 | 0.93 | 2.14 | 0.021 | 0.047 |
| seq11 e2 | 8 | 0.117 | 0.374 | 0.404 | 0.93 | 3.21 | 0.020 | 0.054 |
| seq11 e2 | 10 | 0.112 | 0.400 | 0.437 | 0.92 | 3.55 | 0.016 | 0.050 |

Interpretation:
- **Seq11 fixes the main seq4 failure mode.** The seq4 rollout-loss proxy still
  lost to persistence at `h=8/10`; seq11 beats persistence at the far horizons
  (`e1 h8: 0.297 < 0.314`, `e1 h10: 0.337 < 0.365`) and cuts h10
  free-running MSE versus seq4 rollout (`0.825 → 0.337`).
- **The strict all-horizon gate is not fully passed.** The best long-horizon
  checkpoint (`e1`) is worse than persistence at `h=1/2` and only crosses below
  persistence from `h=3` onward. Treat the seq11 result as green for the
  multi-second objective, not as a fully balanced planner loss.
- **Action conditioning is still the weak point.** `e1` has a real shuffled-action
  gap at long horizon (`h10 shuffled - free = 0.032`), but zero-action is still
  slightly better than commanded rollout at `h8/10`. `e2` improves the zero-action
  gap but gives up raw long-horizon MSE. Future selection should include an
  action-sensitivity guardrail, not only free/persistence.
- **Checkpoint choice for reporting:** use `lewm_seq11_e1.pt` for the seq11
  scoreline: it has the best h10 free-running MSE and the best h10
  free/teacher-forced ratio among the seq11 checkpoints (`3.52×`, still above
  the desired 1-3× band).

Decision: Stage 2 seq11 is positive but not final. Next runs should tune the
mixed objective around the h1/h2 and zero-action regressions: lower rollout
λ or a longer one-step warmup, checkpoint selection on decomposition probes, and
an explicit zero/shuffle-action guardrail before promoting the model to a
production planning run.

## 16. Stage 2 result — seq11 action-sensitivity tuning sweep

Follow-up to §15. The seq11 λ=0.5 baseline beat persistence at the far horizons
but still took the "generic smooth future" shortcut (zero-action ≥ commanded
rollout at h8/10). Two nearby variants were run to keep the long-range gains
while forcing the model to condition on the action, then every epoch checkpoint
was scored on the **same** 2% eval-scene holdout, 256 samples, horizons
`1,2,3,5,8,10`:

- **Variant A** — weaker rollout pressure: `--rollout-lambda 0.25
  --rollout-warmup-epochs 1` (dir `seq11_rollout_lam0p25_h10_warm1_maxsess300`).
- **Variant B** — weaker pressure **+ longer warmup**: `--rollout-lambda 0.25
  --rollout-warmup-epochs 2` (dir `seq11_rollout_lam0p25_h10_warm2_maxsess300`).

(The "longer-warmup-alone" cell, λ=0.5 warm2, was scoped but not run — A/B already
isolate the lever; see below.)

Matched decomposition probe (`scripts/probe_lewm_rollout_horizons.py`,
`free-running` = commanded rollout):

| checkpoint | h | teacher-forced | free-running | persistence | free / persistence | free / TF | zero − free | shuffled − free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A e1 | 1 | 0.099 | 0.099 | 0.089 | 1.11 | 1.00 | +0.002 | +0.006 |
| A e1 | 3 | 0.111 | 0.234 | 0.239 | 0.98 | 2.11 | +0.008 | +0.036 |
| A e1 | 8 | 0.091 | 0.384 | 0.418 | 0.92 | 4.20 | −0.009 | +0.040 |
| A e1 | 10 | 0.104 | 0.420 | 0.450 | 0.93 | 4.03 | −0.013 | +0.040 |
| A e2 | 1 | 0.128 | 0.128 | 0.127 | 1.01 | 1.00 | +0.008 | +0.012 |
| A e2 | 8 | 0.122 | 0.438 | 0.498 | 0.88 | 3.59 | +0.008 | +0.031 |
| A e2 | 10 | 0.118 | 0.484 | 0.554 | 0.87 | 4.11 | +0.001 | +0.021 |
| **B e1** | **1** | **0.108** | **0.108** | **0.103** | **1.05** | **1.00** | **+0.004** | **+0.012** |
| **B e1** | **3** | **0.103** | **0.219** | **0.226** | **0.97** | **2.14** | **+0.005** | **+0.027** |
| **B e1** | **8** | **0.106** | **0.398** | **0.467** | **0.85** | **3.77** | **+0.012** | **+0.048** |
| **B e1** | **10** | **0.112** | **0.446** | **0.527** | **0.85** | **3.98** | **+0.010** | **+0.042** |
| B e2 | 1 | 0.134 | 0.134 | 0.136 | 0.99 | 1.00 | +0.007 | +0.011 |
| B e2 | 8 | 0.115 | 0.392 | 0.434 | 0.90 | 3.42 | +0.012 | +0.046 |
| B e2 | 10 | 0.101 | 0.423 | 0.485 | 0.87 | 4.16 | +0.019 | +0.050 |

Interpretation:
- **Lower λ alone (Variant A) does NOT fix the action shortcut** — confirms the
  §3 "representation/loss-weight is not the lever" finding extends to λ. A e1 still
  has `zero − free` negative at h8/10 (−0.009/−0.013, *worse* than the λ=0.5
  baseline e1's −0.007/−0.004). The shortcut is unchanged by simply weakening the
  rollout term.
- **The longer warmup (Variant B) is the lever.** B e1 is the only `e1`-class
  checkpoint that is *both* low-error *and* action-conditioned: it flips
  `zero − free` positive at every horizon (h8/10 = +0.012/+0.010) while posting the
  best far-horizon win over persistence of any checkpoint in the whole sweep
  (`free / persistence` = **0.85 / 0.85** at h8/10, vs 0.95/0.92 for baseline e1).
  Story: ramping rollout pressure in over 2 epochs lets the one-step predictor
  settle first, so when the multi-step term turns on the model learns
  action-conditioned dynamics instead of a generic motion prior.
- **The only cost is a small short-horizon penalty.** B e1 h1/h3 free-running is
  0.108/0.219 vs baseline e1's 0.085/0.190 (+27% / +8% at h1/h3) — but it is the
  *lowest* short-horizon error among the action-sensitive checkpoints (baseline e2
  0.124/0.223, A e2 0.128/0.269, B e2 0.134/0.235). Fixing the shortcut costs a
  little short-range sharpness; B e1 pays the least for it.

Gate scorecard (h8/10 ≈ 4–5 s):

| checkpoint | beats persistence @ h8/10 | real-action > zero @ h8/10 | real > shuffled @ h8/10 | short-horizon (h1) |
|---|:--:|:--:|:--:|---:|
| baseline e1 (incumbent) | ✓ 0.95/0.92 | ✗ −0.007/−0.004 | ✓ | 0.085 (best) |
| **B e1 (new candidate)** | **✓ 0.85/0.85** | **✓ +0.012/+0.010** | **✓** | **0.108** |

Decision: **promote `seq11_rollout_lam0p25_h10_warm2/lewm_seq11_e1.pt`
(Variant B, epoch 1) as the planner candidate.** It clears all three gates the
incumbent baseline e1 failed on action-sensitivity, with the strongest
long-horizon margin in the sweep and only a minor short-horizon cost. Before
locking it as the *final* planner model, validate on (a) a larger holdout than
256 samples — the `zero − free` margins are real but small (~+0.01) — and (b) the
receding-horizon MPC proxy (`scripts/probe_lewm_receding_mpc_proxy.py`), since
action-sensitivity in open-loop MSE is necessary but not sufficient for
closed-loop planning quality.

### 16.1 Larger-N confirmation (N=2048) and the e1-vs-e2 trade

The matched probe samples only 256 of the 153,415 eval-holdout windows, and the
deciding `zero − free` margins were ~+0.01. Re-scored B e1, baseline e1, and B e2
at **2048 samples** (`--batch-size 32 --max-batches 64`, same seeds), written to
`*_n2048.json`:

| checkpoint | `zero − free` h8/10 @ N=256 | `zero − free` h8/10 @ N=2048 | `free/persistence` h8/10 @ N=2048 | h1 free/pers |
|---|---:|---:|---:|---:|
| **B e1** | +0.012 / +0.010 | **+0.019 / +0.016** | 0.842 / 0.841 | 1.032 |
| **B e2** | +0.012 / +0.019 | **+0.030 / +0.031** | 0.840 / 0.833 | 1.020 |
| base e1 (incumbent) | −0.007 / −0.004 | **−0.001 / −0.001** | 0.920 / 0.905 | 1.096 |

- **The Variant-B action flip is real**, not a 256-sample artifact: both B
  checkpoints' `zero − free` stays positive and *grows* at 8× the data, while the
  incumbent baseline e1 stays ≤ 0 (no benefit from the commanded action). This
  confirms the warmup, not λ, fixed the shortcut.
- **Larger N changes the within-B ranking.** At 2048, B e2 has ~2× the
  action-conditioning of B e1 (+0.030 vs +0.019 at h8) with an *equal* far-horizon
  persistence margin (~0.84/0.84). B e2's only deficit is short-horizon sharpness
  (h1 free-running abs 0.134 vs B e1 0.108). This is a genuine e1-vs-e2 trade
  (short-range fidelity vs long-range action-conditioning) that open-loop MSE
  cannot resolve.

Decision (updated): **both B e1 and B e2 clear all three gates the incumbent
fails; B e1 is the conservative pick (least short-horizon cost), B e2 the
aggressive pick (strongest action-conditioning).** Break the tie with the
receding-horizon MPC proxy, which measures closed-loop planning quality directly —
the metric we actually care about — rather than open-loop reconstruction.

### 16.2 Closed-loop tiebreaker — receding-horizon MPC proxy

`scripts/probe_lewm_receding_mpc_proxy.py`, horizons 1,2,3,5, 4 replan steps,
512 sequences → **2048 decisions/horizon**. At each replanned step it scores the
terminal latent cost of the *recorded* action vs zero / shuffled / persistence;
`win-rate vs zero > 0.5` means the model prefers the true action over doing
nothing more often than not. (Same eval-scene holdout / seeds.)

| checkpoint | vs-zero win-rate h1/3/5 | vs-shuffled h1/3/5 | recorded/zero cost h1/3/5 |
|---|---:|---:|---:|
| **B e2** | **0.55 / 0.57 / 0.57** | 0.58 / 0.61 / 0.60 | **0.948 / 0.914 / 0.921** |
| B e1 | 0.50 / 0.55 / 0.54 | 0.55 / 0.60 / 0.61 | 0.975 / 0.960 / 0.963 |
| base e1 (incumbent) | 0.46 / 0.49 / 0.50 | 0.51 / 0.57 / 0.58 | 0.995 / 1.006 / 1.021 |

- **The incumbent fails closed-loop too:** vs-zero win-rate ≤ 0.50 and
  recorded/zero ≥ 1.0 at h3/5 — in an MPC loop it plans no better than ignoring
  the commanded action. Consistent with its open-loop `zero − free ≤ 0`.
- **B e2 wins the tiebreaker:** highest vs-zero win-rate at every horizon, lowest
  recorded/zero cost (strongest action preference), and its weaker short-horizon
  *reconstruction* did not cost it closed-loop action preference (it leads at h1
  too). The open-loop and closed-loop evidence agree.
- **Calibration:** the absolute signal is modest — vs-zero win-rate ~0.55–0.57,
  i.e. the model prefers the true action only slightly more than half the time.
  This is a real, significant lift over the incumbent's ~0.50 (≈3–4 SE at this N),
  but it confirms the original worry that much of seq11's gain is a smooth-motion
  prior. Warmup tuning bought a genuine but small closed-loop action signal; larger
  gains will need the §6 fixes (rollout-stability scaling, capacity/data, or an
  explicit action-contrastive term), not more λ/warmup tuning.

**FINAL DECISION:** promote **`seq11_rollout_lam0p25_h10_warm2/lewm_seq11_e2.pt`
(Variant B, epoch 2)** as the planner candidate — it is the best of the sweep on
the closed-loop metric and clears all three gates the incumbent fails. Keep B e1
as the fallback if short-horizon reconstruction sharpness later proves to matter.
The candidate is validated as *action-conditioned in closed loop*, but its action
signal is modest; do not treat it as a finished planner — it is the substrate for
the §6 work, not a replacement for it.
