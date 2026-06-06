# LeWM Multi-Step Rollout Prediction — Full Experiment Log & Next Steps (2026-06-05)

**Scope.** This document is the end-to-end record of the *rollout-prediction* track:
from the first full LeWM training run, through the diagnosis that the model could
only imagine ~1 s ahead, to the rollout-training objective and the seq11 tuning +
validation that produced the current planner candidate. It is written to answer one
question directly: **are we now likely to have better predictions over a multi-step
rollout?**

**Deliberate de-scoping.** We are *not* chasing a metric/topological map in latent
space here. The latent-distance-as-map problem (Phase-B topology, the
heading-dominated place code) is handed to the **H-JEPA** layer — hierarchy on top
of a good flat substrate, which (per the DINO-WM/HWM reading in the roadmap §8.1)
needs no place supervision. The world model's job in this track is narrow and
concrete: **predict the action-conditioned future latent accurately over multiple
seconds.** Everything below is measured against that.

Companion doc (representation/topology track, full diagnosis, paper narrative):
`docs/lewm_paper_narrative_and_roadmap_2026-06-04.md`. This log duplicates the
rollout-relevant data so it stands alone.

---

## 0. The one-screen answer (scoreboard)

The scale-invariant metrics are **free-running / persistence** (rollout vs "assume
nothing changes"; **< 1 means we beat doing nothing**) and **free-running /
teacher-forced** (the autoregressive *compounding* multiplier; **want 1–3×**).
Absolute MSE is *not* comparable across rows because the encoders and training
budgets differ — read the ratios.

| # | stage | objective change | h10 free/pers | h10 free/TF | action-conditioned @ long horizon | run scale |
|---|---|---|---:|---:|---|---|
| E0 | Production e9 | 1-step only, seq4 | **1.18** (loses) | **22×** | no (shortcut) | FULL: 10 ep, 9.24M win/ep |
| E3 | (diagnosis) | — localizes the cause — | — | — | — | — |
| E4 | Stage-1 cached | +rollout loss, predictor-only | **0.73** | 9.4× | n/a (screen) | frozen-e9 latents, 20-ep ft |
| E5 | Stage-2 seq4 | +rollout loss, h=3 | 1.23 (loses@10) | 6.5× | ~no | proxy: 3 ep / 300 sess |
| E6 | Stage-2 seq11 (λ0.5) | +rollout loss, h=10 | 0.92 (wins) | 3.5× | no (zero ≥ real) | proxy |
| E9 | Variant B e2 (proxy best) | λ0.25 + 2-epoch warmup | 0.87 / 0.83\* | ~4× | yes, modest (win-rate 0.57) | proxy: 0.4% data |
| E10 | **2.8% scaling, e3 (NEW BEST)** | **same recipe, 6.5× data + 5 ep** | **0.65**\* | ~4× | **yes, clear** (win-rate 0.61, rec/0 0.74) | **2.8% data: 1,952 sess / 5 ep** |

\* N=2048 re-score (E9: 0.833; E10: 0.65).

**Read of the scoreboard:** we moved from a model that was *worse than assuming the
world is frozen* and compounded its own error 22× over 5 s (E0), to one that beats
persistence out to 5 s and compounds only ~4× (E9), to — once we stopped starving it
of data (E10) — a planner that beats persistence by **35% at 5 s** and conditions on
the action *clearly*, not modestly. The big remaining lever is now **data scale**
(E10 used only 2.8% of the textured set) and **compounding** (still ~4×, untouched by
data — that's scheduled-sampling's job, §11 Step 2). Full discussion in §10.

---

## 1. System under test

- Encoder ViT-Tiny; predictor 6-layer causal transformer; **~16M params total**.
- Objective: JEPA latent-prediction loss + λ·SIGReg (isotropic-Gaussian
  anti-collapse), λ_sig = 0.09.
- Data: textured v03 renders, single forward camera, **partial observability +
  visual aliasing** (this matters — it is why flat rollout is hard here and easy on
  fully-observed Push-T); 1450 scenes.
- Macro-step ≈ 0.5 s (`stride=5`). "h10" = 10 steps ≈ **5 s**.
- Intended use: latent state for receding-horizon (MPC/CEM) planning.

The original training window was `max_seq_len=4` and the loss supervised **only
1-step-ahead** prediction. That single fact is the origin of the whole story.

---

## 2. E0 — First full training run (production e9)

The production model: seq4, `stride=5`, λ_sig 0.09, **10 epochs × 9.24M windows**,
1450 scenes. Checkpoint `lewm_seq4_e9.pt`. This is the model all Phase-A diagnosis
ran on.

What it learned (Phase-A, companion doc §2): a usable **place-recognition** code
(retrieval@1 ≈ 0.40, ≈21× chance) but **not a map** (latent distance ≈ flat across
graph-distance buckets), and — the part that concerns this track — **a rollout that
is useful only to ~1 s**: it beats persistence at h1–2 and loses from h3 onward.

For a planner that needs to imagine several seconds ahead, E0 is not usable.

---

## 3. E1–E2 — What we ruled out before touching the objective

- **E1 (Phase-A rollout finding):** rollout beats persistence only at h≤2. Logged;
  motivates everything after.
- **E2 (representation-knob ablation, companion §3):** a scaled λ_sig × source-mix
  factorial. Relevant negative result for *this* track: lowering λ_sig drove the
  per-step **action-sensitivity toward zero** (0.0377 → 0.0003 → 0.0000 as λ_sig
  0.09→0.03→0.01). Takeaway carried forward: **SIGReg λ is not a lever for
  action-conditioning** — if anything, weakening it erases action signal. (The
  factorial's main purpose was the metric/topology axis, now H-JEPA's problem.)

---

## 4. E3 — The pivot: why the horizon dies (rollout decomposition)

The decisive diagnostic (`scripts/probe_lewm_rollout_horizons.py`, teacher-forced
extension) on production e9. It separates two hypotheses: does the horizon die
because (a) the predictor **compounds** its own error, or (b) the **target latent is
intrinsically unpredictable**? Compare **teacher-forced** (one step from the *true*
latent) vs **free-running** (autoregressive).

| horizon | teacher-forced 1-step | free-running | persistence | free / TF |
|---:|---:|---:|---:|---:|
| 1 (0.5 s) | 0.050 | 0.050 | 0.091 | 1.0× |
| 3 (1.5 s) | 0.043 | 0.302 | 0.289 | 7.1× |
| 5 (2.5 s) | 0.033 | 0.476 | 0.460 | 14.5× |
| 10 (5.0 s) | 0.040 | 0.876 | 0.743 | **22×** |

**Teacher-forced 1-step error is flat and low (~0.03–0.05) at every horizon and
beats persistence 3–20×.** Free-running explodes to 22×. The per-step dynamics head
is **competent**; the horizon dies *purely* from autoregressive compounding —
expected, because the model was only ever trained to predict 1 step ahead and never
to roll out.

**This is a training-objective defect, not a representation defect** — and it is the
single most important finding in this track, because it says the fix is cheap: train
the existing model to roll out.

---

## 5. E4 — Stage-1 cached-latent screen (does a rollout loss help at all?)

Before spending an expensive image run, isolate the *predictor*: freeze the e9
encoder, cache its latents at seq11, train a fresh predictor with vs without a
free-running rollout term. Minutes per config.

| variant | h | teacher-forced | free-running | persistence | free / TF | free / pers |
|---|---:|---:|---:|---:|---:|---:|
| e9 baseline | 10 | 0.040 | 0.868 | 0.733 | 22.0× | 1.18 |
| one-step fine-tune | 10 | 0.073 | 1.137 | 0.733 | 15.5× | 1.55 |
| **rollout-loss fine-tune** | 10 | 0.057 | **0.535** | 0.733 | **9.4×** | **0.73** |

Two clean results:
1. **The rollout loss works:** h10 free-running 0.868 → 0.535, and rollout now beats
   persistence at h≥3 (free/pers 0.73).
2. **Training *longer* on the 1-step loss is the wrong fix** — it *worsens* held-out
   rollout (0.868 → 1.137). It must be the free-running term specifically.

Caveat noted for later: rollout fine-tuning slightly worsened h1 teacher-forced
(0.039 → 0.056) → use a *mixed* objective with warmup in the real run.

---

## 6. E5 — Stage-2 seq4 image proxy (does it survive a co-adapting encoder?)

Implemented the Tier-A horizon-weighted free-running rollout loss in
`LeWorldModel.forward()` + CLI flags in `scripts/train_lewm.py`. First image-proxy
run still at seq4 (3 transitions of rollout supervision), λ=0.5, 3 epochs, 300
sessions.

| model | h | free-running | persistence | free / pers |
|---|---:|---:|---:|---:|
| no-rollout seq4 proxy | 3 | 0.531 | 0.426 | 1.25 |
| no-rollout seq4 proxy | 10 | 1.321 | 0.815 | 1.62 |
| **rollout-loss seq4 proxy** | 3 | 0.316 | 0.357 | **0.89** |
| rollout-loss seq4 proxy | 10 | 0.825 | 0.669 | 1.23 |

Positive and end-to-end: rollout now beats persistence through h3 (1.5 s), and the
h1 guardrail held (teacher-forced did **not** regress in pixels, unlike the cached
screen). But with only 3 steps of supervision it still loses at h8/10 — exactly why
the next run must train the full 10-step / 5-second horizon.

---

## 7. E6 — Stage-2 seq11 image proxy (the real 5-second objective, λ=0.5)

First model trained on the true 10-transition / 5-second objective
(`--max-seq-len 11 --rollout-horizon 10 --rollout-lambda 0.5 --rollout-warmup-epochs 1`).
Matched decomposition probe, 2% eval-scene holdout, 256 samples; best checkpoint = e1.

| seq11 e1 | h | teacher-forced | free-running | persistence | free / pers | free / TF | zero−free | shuf−free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | 1 | 0.085 | 0.085 | 0.075 | 1.13 | 1.00 | −0.000 | +0.005 |
| | 3 | 0.091 | 0.190 | 0.192 | 0.99 | 2.08 | +0.003 | +0.018 |
| | 8 | 0.089 | 0.297 | 0.314 | 0.95 | 3.35 | −0.007 | +0.032 |
| | 10 | 0.096 | 0.337 | 0.365 | **0.92** | **3.52** | **−0.004** | +0.032 |

**seq11 fixes the main E0 failure:** it beats persistence at the far horizons and
cuts h10 free-running from the seq4-proxy's 0.825 to 0.337, with compounding down
from 22× (E0) to 3.5×. **But it takes the lazy shortcut:** `zero − free` is ≤ 0 at
h8/10, i.e. feeding *zero* action predicts the future about as well as feeding the
*real* action. The rollout is largely a generic smooth-motion prior, not genuinely
action-conditioned. That is the problem the tuning sweep set out to fix.

---

## 8. E7 — seq11 action-sensitivity tuning sweep

Goal: keep the long-horizon win while forcing the model to condition on the action.
Two nearby variants, each epoch scored on the **same** 256-sample holdout:

- **Variant A** — weaker rollout pressure: `--rollout-lambda 0.25 --rollout-warmup-epochs 1`.
- **Variant B** — weaker pressure **+ longer warmup**: `--rollout-lambda 0.25 --rollout-warmup-epochs 2`.

Key rows (h8/10), `zero−free > 0` = real action beats zero (the gate E6 fails):

| checkpoint | h8 free/pers | h8 zero−free | h10 free/pers | h10 zero−free | h1 free (abs) |
|---|---:|---:|---:|---:|---:|
| base e1 (E6) | 0.95 | −0.007 | 0.92 | −0.004 | 0.085 |
| A e1 (λ0.25, w1) | 0.92 | −0.009 | 0.93 | −0.013 | 0.099 |
| A e2 | 0.88 | +0.008 | 0.87 | +0.001 | 0.128 |
| **B e1 (λ0.25, w2)** | 0.85 | +0.012 | 0.85 | +0.010 | 0.108 |
| **B e2** | 0.90 | +0.012 | 0.87 | +0.019 | 0.134 |

Findings:
- **Lower λ alone (A) does NOT fix the shortcut** — A e1's `zero−free` is *worse*
  than the λ=0.5 baseline. Confirms (consistent with E2) that λ is not the lever.
- **The longer warmup (B) is the lever.** Ramping rollout pressure over 2 epochs lets
  the one-step predictor settle first, so when the multi-step term turns on the model
  learns *action-conditioned* dynamics instead of a motion prior. Both B checkpoints
  flip `zero−free` positive while posting the best far-horizon margins in the sweep.
- The cost is a small short-horizon reconstruction penalty (h1 abs error up from
  0.085 to ~0.11–0.13).

*(Operational note: the original run of this sweep was interrupted when its driver
process died mid-Variant-B; B was re-run fresh to completion and all six new
checkpoints were probed on the identical holdout. The seq11-baseline 256-sample
files were left untouched.)*

---

## 9. E8–E9 — Validation: larger-N re-score and the closed-loop tiebreaker

The deciding `zero−free` margins were ~+0.01 on only 256 of 153,415 eval windows, so
both validations below were run before believing the result.

**E8 — N=2048 re-score** (8× the data, same seeds):

| checkpoint | zero−free h8/10 @ N=256 | zero−free h8/10 @ N=2048 |
|---|---:|---:|
| B e2 | +0.012 / +0.019 | **+0.030 / +0.031** |
| B e1 | +0.012 / +0.010 | +0.019 / +0.016 |
| base e1 (E6) | −0.007 / −0.004 | −0.001 / −0.001 |

The action flip is **real and strengthens** at larger N for both B checkpoints; the
incumbent stays ≤ 0. Larger N also flipped the within-B ranking: B e2 now shows ~2×
the action-conditioning of B e1 at equal persistence margin.

**E9 — receding-horizon MPC proxy** (`scripts/probe_lewm_receding_mpc_proxy.py`,
horizons 1/2/3/5, 4 replan steps, **2048 decisions/horizon**). Closed-loop: at each
replanned step, does the model assign **lower terminal latent cost to the recorded
action** than to zero/shuffled? `win-rate vs zero > 0.5` = prefers the true action.

| checkpoint | vs-zero win-rate h1/3/5 | recorded/zero cost h1/3/5 |
|---|---:|---:|
| **B e2** | **0.55 / 0.57 / 0.57** | 0.948 / 0.914 / 0.921 |
| B e1 | 0.50 / 0.55 / 0.54 | 0.975 / 0.960 / 0.963 |
| base e1 (E6) | 0.46 / 0.49 / 0.50 | 0.995 / 1.006 / 1.021 |

- **The incumbent fails closed-loop too:** win-rate ≤ 0.50 and recorded/zero ≥ 1.0 at
  h5 — in an MPC loop it would plan no better than ignoring the command.
- **B e2 wins the tiebreaker** at every horizon, and its weaker short-horizon
  *reconstruction* did not cost it closed-loop action preference (it leads at h1 too).
  Open-loop and closed-loop agree.

**PROXY PICK (E9): `seq11_rollout_lam0p25_h10_warm2/lewm_seq11_e2.pt` (Variant B, epoch 2)**
— superseded by E10 below once we trained on more data.

---

## 9.5 E10 — Data scaling: the action signal grows (and was being starved)

The E0–E9 runs all used `--max-sessions 300` (proxy) — which is only ~0.4% of the
textured set. **Crucial discovery:** `render_textured_v03` is 1450 scenes × **48
parallel envs** × 1000 steps ≈ **69,600 environment-sessions**, *not* 1450. The
proxies were data-starved by ~250×. Production e9 used the *full* set (same render
root) — that is why it took days.

A first scaling step — same Variant-B recipe, arch fixed, `--max-sessions 2000` →
**1,952 sessions / 369,912 windows (~2.8% of the data)**, 5 epochs — already
transforms the planner. Matched probes (N=2048) + closed-loop MPC, vs the proxy best:

| checkpoint (N=2048) | h10 free/pers | h10 zero−free | h10 shuf−free | MPC vs-zero h1/3/5 | MPC rec/0 h5 |
|---|---:|---:|---:|---:|---:|
| proxy B e2 (E9) | 0.83 | +0.031 | +0.063 | 0.55 / 0.57 / 0.57 | 0.921 |
| 2.8% e2 | 0.70 | +0.059 | +0.091 | — | — |
| **2.8% e3 (best)** | **0.65** | **+0.085** | **+0.134** | **0.57 / 0.60 / 0.61** | **0.736** |
| 2.8% e4 | 0.65 | +0.080 | +0.144 | — | — |

- **Every gate jumps with ~6.5× more data.** Open-loop action-conditioning ~2.7×
  stronger (zero−free +0.031 → +0.085); persistence margin at 5 s goes 17% → **35%**;
  h10 rollout MSE roughly halves (0.42 → 0.22).
- **Closed-loop confirms it:** e3's recorded-action vs-zero win-rate beats the proxy
  at every horizon (0.61 vs 0.57 at h5) and the cost preference is far stronger —
  the recorded action's terminal cost is **26% below zero-action** (rec/0 = 0.74) vs
  only 8% for the proxy (0.92). The action signal is no longer "modest."
- **Honesty checks:** e3's rosy N=256 read (free/pers 0.59, zero−free +0.106) softened
  at N=2048 (0.65, +0.085) — small-sample optimism, conclusion unchanged. And
  **compounding is still ~4×** (free/TF) — data scale did *not* fix it; that remains
  scheduled-sampling's job (§11 Step 2). e3 and e4 are ~tied on decomposition; e3 wins
  the one MPC run.

**NEW BEST: `…_warm2_allsess_ep5/lewm_seq11_e3.pt`.** And this is still only 2.8% of
the data — the scaling curve is pointing up, which is the direct evidence that the
full-data run (§11 Step 1) is worth its multi-day cost.

---

## 10. Direct answer — are we likely to have better multi-step rollout predictions now?

**Yes — clearly better than the original model, on exactly the metric that defines
this track, and the improvement is validated open-loop *and* closed-loop.** Concretely,
versus the first full run (E0):

1. **We crossed the "better than doing nothing" line out to 5 s.** h10
   free/persistence went 1.18 (E0, *loses*) → 0.87/0.83 (E9, *wins*). At 5 s the
   model's imagined future is now meaningfully closer to the truth than freezing the
   last frame; at E0 it was worse than freezing.
2. **We cut compounding from 22× to ~4×.** h10 free/teacher-forced 22× → ~4×. The
   rollout no longer diverges from its own competent per-step dynamics nearly as fast.
3. **The rollout is now genuinely action-conditioned — for the first time.** Every
   prior checkpoint (E0, E6) treated a zero action as ≈ the real action at long
   horizon; B e2 prefers the real action in closed loop (win-rate ~0.57 vs the
   incumbent's ~0.50, ≈3–4 SE).

**Update (E10): the "modest" qualifier is already lifting with data.** The E9
"~57% / modest" read was at 0.4% of the data. At 2.8% (E10) the closed-loop vs-zero
win-rate rises to **0.61** and the recorded-action cost preference jumps from 8% to
**26%** below zero-action — clearly action-driven, not modest. Long-horizon prediction
also improved: free/persistence at 5 s went 0.83 (E9) → **0.65** (E10). So the "degree"
question is resolving favourably: **more data ⇒ stronger action-conditioning and better
rollout**, and we are still at 2.8%.

**One qualifier that did *not* improve with data:** compounding is still ~4×
(free/TF). Data makes each step better but doesn't teach recovery-from-own-error;
that is scheduled-sampling's job (§11 Step 2).

So: **the objective question is settled — training for rollout fixed the horizon and
introduced action-conditioning — and the scaling question is now answered positive in
its first step.** The levers that matter are **data scale** (Step 1) and **compounding
via scheduled sampling** (Step 2); λ and warmup are exhausted.

---

## 11. Recommended next steps (rollout-prediction track; topology = H-JEPA)

Ordered by leverage-per-effort. Explicitly **out of scope here** (handed to H-JEPA /
the representation track): the heading-dominated place code, latent-distance-as-map,
and any place/topology supervision.

### Step 1 — Scale to the full textured dataset (do first; now VALIDATED-POSITIVE at 2.8%)
E10 proved the direction: ~6.5× more data (still only 2.8% of the set) already lifted
every gate. Now go to the **full ~69,600-session textured set** (drop `--max-sessions`),
same B recipe; consider a **larger predictor** too (the ~16M is capacity-limited, and
more data invites more capacity). Cost is real — **~37 h/epoch on the R9700, multi-day
for a few epochs** — but the scaling curve justifies it. Same gates: decomposition +
MPC win-rate. Intermediate rungs (~8k sess overnight, ~20k sess ~1.5 days) de-risk it.
Open sub-question: does the win-rate keep climbing past 0.61 toward a confident planner?

### Step 2 — Scheduled sampling (the unused objective lever; medium effort)
The §6.1 "Tier B" was never implemented. Feed the model its **own** predicted latents
during training (probability ramping 1→~0.3) so it learns to recover from its own
drift — the textbook fix for the residual ~4× compounding, and the natural next move
now that λ/warmup are tapped out. Pair with a horizon curriculum (2→10). Cheap to
prototype on the seq11 proxy first; measure free/TF and MPC win-rate.

### Step 3 — Substrate upgrade, DINO-WM style (highest ceiling; higher effort)
The structural reason the action signal is modest and rotation swings the whole
latent: we predict a **single pooled global vector** from a from-scratch ViT-Tiny.
DINO-WM rolls out far on Push-T because it predicts **spatial patch-token** targets
from **strong/frozen pretrained features**. Predicting patch tokens (and/or stronger
features) gives more predictable, more action-localizable targets. **This is also the
substrate H-JEPA should sit on** — hierarchy multiplies horizon/efficiency on a good
flat substrate, it does not manufacture one.

### Step 4 — Short history / belief for the POMDP (complements H-JEPA)
Our maze is partially observed + aliased; a single-frame latent cannot disambiguate
look-alike views, which caps reliable rollout length. A short history/recurrent belief
is the principled fix and, per the roadmap, the real route to any spatial structure
under aliasing — i.e. it works *with* H-JEPA rather than competing with it.

### Explicitly NOT next
- More **λ / warmup** tuning — exhausted (E2, E7: λ does nothing for action; warmup
  gave its one-time, now-measured gain).
- The **place/metric/topology encoder** work — deliberately deferred to H-JEPA.

**Suggested immediate action:** kick off **Step 1** (full-scale B-recipe run) and,
while it trains, prototype **Step 2** (scheduled sampling) on the cheap seq11 proxy.
Re-probe both with the same decomposition + MPC gates used here.

**Representation-track update, 2026-06-06:** this recommendation is only for the
forward-prediction track. Scaling the same objective is not expected to create the
missing fine metric geometry. Do not treat a stronger rollout checkpoint as
navigation-ready without the separate gates in
`docs/lewm_pose_aux_experiment_design_2026-06-06.md`.

---

## 12. Artifact index (this track)

- Checkpoints + probes: `models/checkpoints_textured_v03_rollout_stage2_20260604/`
  - `seq11_rollout_lam0p5_h10_maxsess300/` — E6 baseline (`rollout_decomp_e{0,1,2}.json`)
  - `seq11_rollout_lam0p25_h10_warm1_maxsess300/` — Variant A
  - `seq11_rollout_lam0p25_h10_warm2_maxsess300/` — **Variant B (final = `lewm_seq11_e2.pt`)**
  - Per-checkpoint: `rollout_decomp_e*.json` (N=256), `rollout_decomp_e*_n2048.json`
    (E8), `mpc_proxy_e*.json` (E9)
- E0/E3 diagnostics: `models/checkpoints_textured_v03_full_20260531/sweep_seq4/`
  (`rollout_decomp_e9_final.json`)
- E4 cached screen:
  `…/sweep_seq4/predictor_only_stage1/e9init_*_seq4ctx_n8192_eval2048.json`
- E5 seq4 proxy: `…/seq4_rollout_lam0p5_h3_maxsess300/`
- Scripts: `scripts/probe_lewm_rollout_horizons.py`,
  `scripts/probe_lewm_receding_mpc_proxy.py`, `scripts/train_lewm.py`,
  `scripts/cache_lewm_latents.py`, `scripts/train_lewm_predictor_only.py`
- Companion (representation/topology, paper narrative):
  `docs/lewm_paper_narrative_and_roadmap_2026-06-04.md` (§16/16.1/16.2 = E6–E9).
