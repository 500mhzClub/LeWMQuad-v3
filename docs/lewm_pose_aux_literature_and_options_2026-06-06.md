# Navigation-JEPA literature review, regularization analysis, and ranked options (2026-06-06)

Companion to `lewm_pose_aux_experiment_design_2026-06-06.md` and
`lewm_pose_aux_ladder_300sess_findings_2026-06-06.md`. Written after the 300-session
pose-aux geometry screen to ground the next decision in the external literature.

**Screen verdict being acted on:** the pose-aux geometry loss injects *encoded*
metric decodability (encρ monotonic F0 +0.067 → C2 +0.188) with the forward model
preserved, but the *decisive* predicted→goal correlation only ties the
plain-continuation control (C2 +0.142 vs C0/posthoc +0.148) and **no cell produces
positive first-action ranking**. I.e. decodable geometry improved; *actionable*
geometry did not. Conclusion: predictor/projector-alignment + action-sensitivity
branch, not a λ problem.

Provenance note: claims attributed to a paper are from that paper (fetched
2026-06-06). Sections marked **[analysis]** are our interpretation, not a published
result.

---

## Part 1 — JEPA-style world models used for navigation/planning

| model | latent or generative | how it plans | supervises pose/position? |
|---|---|---|---|
| **DINO-WM** (Zhou et al., 2411.04983) | JEPA on frozen DINOv2 features | MPC + **CEM**, cost = **L2 in latent** to goal | No — pure self-sup prediction |
| **PLDM** (Sobal, LeCun, 2502.14819) | JEPA latent dynamics, VICReg-regularized | **MPPI**, cost = L2-in-latent + ensemble uncertainty | No |
| **Navigation World Model** (Bar, LeCun, 2412.03572) | **generative** (synthesizes future video) | rolls imagined video, scores **goal-frame similarity** | No |
| **V-JEPA 2-AC** (Meta) | action-conditioned V-JEPA 2 | MPC from a small amount of action data | No |
| **ReL-NWM** (2511.11011) | latent NWM on DINOv3 | latent rollout + goal match | No |
| **Hierarchical Planning w/ Latent WMs** (2604.03208) | survey/combiner of PLDM, V-JEPA2-AC, DINO-WM | hierarchical latent planning | No |
| **LeWM / LeWorldModel** (le-wm.github.io, 2603.19312) | end-to-end JEPA from pixels (our basis) | — | No |

**Key structural fact:** none of the mainstream nav-JEPAs supervise physical pose.
Our pose-aux approach is a deliberate departure (the design doc says so). They obtain
plannability from the *self-supervised* objective + action-conditioning + data
coverage instead.

### Did they hit our problem? Yes — it is a named result.

- **"Good prediction ≠ good planning."** *What Drives Success in Physical Planning
  with JEPA World Models?* (2512.24497) states it explicitly: *"even with models
  which are able to faithfully unroll a large number of actions, success at the
  planning task is not an immediate consequence."* This is our "decodable geometry
  improved, actionable geometry did not."
- **Non-metric / hard-to-rank latent.** Same paper: *"larger embedding spaces make
  it harder for the planning optimization to distinguish nearby states"* — our
  dead within-band fine resolution.
- **Gradient/ranking fails; sampling rescues.** DINO-WM found gradient-based
  planning fails on a non-smooth latent landscape and switched to **CEM**; they
  explicitly *did not* constrain latent smoothness. Maps to our first-action ranking
  ≈ random.
- **Action-insensitivity / slow features** is the central failure the subfield
  engineers against — the same phenomenon as our own seq11 "zero-action shortcut"
  that rollout-warmup fixed.

### How the working methods fix it (none use pose labels)

1. **Inverse-dynamics (IDM) + variance/covariance reg (PLDM).** Latent kept
   action-sensitive by predicting the action from consecutive latents; anti-collapse
   via VICReg variance+covariance + temporal smoothness. Ablations are dramatic:
   removing the variance term drops Two-Rooms success **98% → 13%**; removing
   covariance → 29%. PLDM never supervises distance — it relies on "distance in the
   representation ∝ number of steps" *made roughly true by the regularization*.
2. **Stronger action conditioning (What Drives Success).** Injecting actions at
   **every transformer block (AdaLN + RoPE)** beats concatenation; **multi-step
   rollout loss** beats teacher-forcing; need ≥2-frame context for velocity; **CEM
   L2 > L1 > gradient**.
3. **Action coverage / harder data (PLDM).** Planning quality depends heavily on
   action diversity and coverage of critical regions (random vs directional actions;
   covering the doorway). Mirrors our decision-tree branch "candidate ranking fails →
   inspect coverage, train on harder action-conditioned endpoint pairs."
4. **Sampling-based planning (DINO-WM, WDS).** CEM/MPPI with L2 cost is the
   consensus precisely because the latent isn't cleanly metric for gradients.
5. **Generative escape (NWM).** Avoids a metric latent altogether — synthesizes the
   future and scores goal-image similarity. Connects to our finding that frozen LeWM
   latents are strong at place *recognition* (retrieval@1 ≈ 0.42) but weak at metric.

---

## Part 2 — VICReg vs SIGReg: two orthogonal axes

A JEPA latent must satisfy two unrelated things, regularized by different tools:

- **Axis 1 — the *marginal* embedding distribution:** don't collapse, fill the
  space, make distances meaningful. **VICReg / SIGReg live here.**
- **Axis 2 — the *dynamics* geometry:** how actions *move* you through that space;
  is displacement legible and metric. **Inverse-dynamics / action-conditioning /
  temporal smoothness live here.**

Our failure (good prediction, non-actionable geometry, first-action ranking ≈
random) is an **Axis-2 deficit**. SIGReg is an Axis-1 tool. So this is not a
"VICReg vs SIGReg" decision.

### Axis 1: SIGReg supersedes VICReg

- **VICReg** controls only first/second moments: a per-dimension variance floor
  (threshold γ) + off-diagonal covariance → 0 (linear decorrelation). It says
  nothing about distribution *shape* and removes only *linear* dependence; multiple
  heuristic knobs; no optimality guarantee.
- **SIGReg** (LeJEPA, 2511.08544) pushes the *full* distribution toward an
  **isotropic Gaussian** via random 1-D projections (Cramér–Wold) + a univariate
  Gaussianity test (Epps–Pulley characteristic-function statistic) per slice. It
  pins *all* moments, not just covariance. LeJEPA proves the isotropic Gaussian
  **minimizes worst-case downstream risk** over unknown tasks, uses **one
  hyperparameter**, is linear-time, and removes heuristics (no stop-gradient, no
  EMA teacher-student, no asymmetric predictor) — collapse impossible by
  construction. **Weak-SIGReg** (2603.05924) re-derives covariance regularization as
  a weak slice of the SIGReg idea, positioning VICReg-style covariance reg as a
  special case.

So **for the collapse / marginal-shape job, VICReg would be a step backward**, and
our stack already uses the better tool (`sigreg-lambda 0.09`). We are *not* swapping
SIGReg out.

### Axis 2: what PLDM actually buys — SIGReg can't

- **Inverse-dynamics (IDM):** predict aₜ from (zₜ, zₜ₊₁) → latent becomes
  action-sensitive and action-conditioned displacement is recoverable (a self-sup
  cousin of our encoded relative-pose contract, driven by logged actions, no pose
  labels).
- **Temporal smoothness / step-count metric:** "latent distance ∝ steps" — makes
  L2-in-latent planning work.

These shape the *conditional* p(z′|z,a) — motion through the cloud. **SIGReg only
shapes the marginal p(z)** — the shape of the cloud. An isotropic marginal is
necessary (distances meaningful, no collapse) but **not sufficient**: it can sit
over arbitrary non-metric, action-illegible dynamics — plausibly our model now.

### The tension **[analysis]**

SIGReg and *absolute* metric geometry can fight. A truly metric latent — positions
mirroring physical layout — is structured and non-Gaussian (rooms/corridors aren't
an isotropic blob), while SIGReg pulls the marginal toward featureless isotropy.
This may partly explain why pose-aux's encoded ρ rose only modestly and the
predicted contract didn't carry. The reconciliation: encode geometry in the
**transitions** (IDM, relative-pose pairs, step-count), which coexists with an
isotropic marginal (isotropic cloud, structured motion). This argues the IDM/
action-conditioning route is *more compatible with the SIGReg backbone* than forcing
absolute pose into the embedding.

---

## Part 3 — Mapping the literature to our result

| our observation | external corroboration |
|---|---|
| prediction strong, planning/ranking weak | WDS: prediction fidelity ≠ planning success |
| first-action ranking ≈ random; fine resolution dead | WDS: large embeddings hard to distinguish nearby states; DINO-WM: gradient planning fails on non-smooth latent |
| L2-in-latent anti-metric | DINO-WM/PLDM use L2 but make it work via SSL reg (PLDM) or sampling (DINO-WM CEM) |
| zero-action shortcut killed by rollout warmup | the slow-features/action-insensitivity failure the whole subfield targets |
| pose-λ is not the lever | nobody fixes this with metric supervision; they use IDM + action-conditioning + coverage |
| latent good at recognition (retrieval@1 0.42) | NWM-style goal-image scoring is a viable metric-free route |

---

## Part 4 — Ranked options

Conditional on the 1000-session confirm (in flight). If that *flips* and C2 beats the
control with more data, "scale pose-aux" rises to the top. Base case = screen holds.

| # | option | why ranked here | cost | payoff |
|---|---|---|---|---|
| **1** | **Inverse-dynamics aux + relative/step-count geometry** | the mechanism working nav-JEPAs rely on (PLDM); attacks action-insensitivity; conditional geometry is SIGReg-compatible | Low — IDM head slots into existing ladder/watcher, fine-tunes from e3 | High |
| **2** | **Harder action-conditioned pairs / coverage** | our decision-tree branch *and* PLDM's empirically-critical factor; complements #1 | Low — sampling/curriculum change | Med-High |
| **3** | **Recognition / topological planning (hedge)** | leverages retrieval@1 ≈ 0.42; NWM-style goal-image scoring sidesteps metric; most likely to *actually navigate* | Med — retrieval/graph planner on frozen latents, no retrain | High (different deliverable) |
| **4** | **Stronger action-conditioning architecture (AdaLN+RoPE)** | top driver in WDS; fixes "predictor doesn't carry geometry" | **High** — architecture change ⇒ retrain from scratch, lose e3 lineage | High but expensive |
| **5** | **Finish 1000-sess confirm (+ A1 command-label ablation)** | necessary gate; cheap; low new info beyond settling the screen | ~free (running) | decisional |
| **6** | **More pose-λ / pose-aux tuning** | research + our λ-sweep say wrong knob | Low | Low — stop after confirm |
| **7** | **Planner-only tuning (CEM/MPPI) / generative NWM** | substrate, not planner, is the bottleneck; generative is a heavy departure | Low / Very high | Low |

### Recommended sequence

1. **Do #1 + #2 together, reusing the ladder.** Add an `--idm-lambda` head (predict
   the logged *action* from consecutive latents — fully self-supervised, no pose
   labels) with hard-pair/coverage-aware sampling, fine-tuned from e3 on the same
   harness. Highest expected value per GPU-hour, SIGReg-compatible, directly tests
   the "predictor/projector isn't carrying geometry" diagnosis.
2. **Run #3 in parallel as a hedge** — frozen-latent retrieval/graph planner, no GPU.
   Most likely to ship a working navigator even if metric geometry never lands.
3. **Defer #4** until #1/#2 show whether a cheap aux fix suffices (we already have
   its multi-step-rollout half via warm2; the new piece is per-block conditioning).
4. **Drop #6/#7** as primary directions.

---

## Sources (fetched 2026-06-06)

- LeJEPA: Provable and Scalable SSL Without the Heuristics — arXiv 2511.08544 — https://arxiv.org/html/2511.08544v3
- SIGReg overview (EmergentMind) — https://www.emergentmind.com/topics/sketched-isotropic-gaussian-regularization-sigreg
- LeJEPA explainer (Turing Post) — https://www.turingpost.com/p/lejepa
- Weak-SIGReg: Covariance Regularization for Stable Deep Learning — arXiv 2603.05924 — https://arxiv.org/html/2603.05924v1
- PLDM — Learning from Reward-Free Offline Data: Planning with Latent Dynamics Models — arXiv 2502.14819 — https://arxiv.org/html/2502.14819v3
- What Drives Success in Physical Planning with JEPA World Models? — arXiv 2512.24497 — https://arxiv.org/html/2512.24497v1
- DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning — arXiv 2411.04983 — https://arxiv.org/html/2411.04983v1
- Navigation World Models (Bar, Zhou, Tran, Darrell, LeCun) — arXiv 2412.03572 — https://arxiv.org/html/2412.03572v1
- ReL-NWM: Representative Latent World Model — arXiv 2511.11011 — https://arxiv.org/pdf/2511.11011
- Hierarchical Planning with Latent World Models — arXiv 2604.03208 — https://arxiv.org/html/2604.03208v1
- LeWorldModel (project basis) — https://le-wm.github.io/ — arXiv 2603.19312
- Learning Invariant Visual Representations for Planning with JEPA World Models — arXiv 2602.18639
- V-JEPA 2-AC (action-conditioned V-JEPA 2, Meta) — surveyed in 2604.03208
