# Programme plan: observability ceiling, metric validity, and audit custody

Date: 2026-08-05
Branch: `jepa-spatial-world-model-nav`
HEAD at authorship: `4adcbdad16baa81c93112e9f0f4a6aa643008fad`

Status: **plan and proposal only.** This document is not a preregistration, not
an execution authority, and does not authorize data generation, training,
evaluation, GPU work, artifact mutation, or any access to untouched or sealed
roles. Work packages A and B each require their own preregistration, source
review, and authority before execution, exactly as §9–§13 of the session
handoff required.

Companion documents:

- `docs/lewm_go2_world_model_session_handoff_2026-07-31.md` (§9–§13, terminal)
- `docs/lewm_jepa_navigation_literature_and_experimental_history_critical_review_2026-08-02.md`
- `docs/SAINTS_Year_1_Progression_Document-3.pdf` (§5 Year-2 plan, §9 progress)

---

## 0. Summary

Three work packages, executed in order:

| WP | Name | Cost | Consumes | Decides |
|---|---|---|---|---|
| **A** | Observability-ceiling assay | < 1 GPU-hour, no Genesis | V3 eval successor custody | Whether the `0.13` rank-regret gate is achievable at all, and where the loss lives |
| **B** | Metric-validity study | minutes, kinematic harness | 24 dev scenes (already exposed) | Whether one-step rank regret predicts closed-loop planning utility |
| **C** | Audit-trail custody | minutes | nothing | Removes the single-point-of-failure on ~2 weeks of registered results |

Plus a reconciliation of the SAINTS Year-1 progression document with the
Aug 1–5 evidence block, which it does not currently contain (§4 below).

---

## 1. The finding that motivates A and B

### 1.1 Cross-panel regret table

Every arm ever scored on matched-branch normalized rank regret, grouped by the
panel it was measured on. **The two panels are not commensurable** and must not
be read as one ranking.

**16 train + 16 eval scene panel** (§11, and the frozen V-JEPA interface ceiling):

| arm | regret | privileged? |
|---|---:|---|
| privileged physical oracle | `0.000` | is the label |
| retained physical predecessor | `0.149` | privileged physical state |
| physical-only matched (learned) | `0.164` | learned, non-visual |
| task/action only | `0.174` | non-visual control |
| dense-DINO joint JEPA (learned) | `0.176` | learned visual |
| **true-future frozen V-JEPA tokens** | **`0.207`** | **privileged visual, zero prediction error** |
| current-state V-JEPA | `0.223` | privileged visual |
| relational persistence | `0.229` | — |
| train action-mean innovation | `0.232` | — |
| same-action wrong-scene future | `0.240` | negative control |
| random expectation | `0.495` | — |

**32 train + 32 eval scene panel** (§13):

| arm | regret |
|---|---:|
| visual recurrent direct (learned) | `0.272` |
| no-vision recurrent direct (learned) | `0.303` |
| task/action only | `0.308` |
| random expectation | `0.477` |

### 1.2 Three facts that follow

**(i) The `0.13` absolute gate has never been shown achievable by any function
of observations.** Both §11 and §13 preregistered it; both failed it. The only
arm ever to come within reach is `0.149` — retained *physical predecessor
state*, which is privileged non-observational information. The nearest thing to
an achievability check, the frozen V-JEPA physical-interface ceiling, was run
through a *deliberately* constrained interface: train-only `K=8` PCA into a
245-parameter shared spatial readout. It therefore measured an **interface**
ceiling, not an **information** ceiling. The question "can any readout of the
observations reach `0.13`?" is open.

This matters because the repository already holds itself to exactly this
standard elsewhere. SAINTS §4 and §9.3.1: *"An oracle has passed the physical
perception gate, demonstrating that the gate is achievable rather than
internally inconsistent."* That treatment was given to the perception gate and
never to the rank-regret gate.

**(ii) A fixed absolute threshold is applied across panels of very different
difficulty.** Task/action-only — an unchanged non-visual control — scores
`0.174` on the 16-scene panel and `0.308` on the 32-scene panel. A
panel-invariant absolute gate cannot mean the same thing on both. §13's
"visual regret is more than twice the absolute ceiling" is partly a statement
about panel difficulty.

**(iii) Both "visual beats task/action" gates were statistically unreachable at
the sample size used.** The gate form is *point ≤ −δ **and** CI upper < 0*.
With observed CI half-width `h`, the gate can only fire when the point estimate
is at most `−h`. Whenever `h > δ`, the binding constraint is `h`, not the
registered `δ`:

| comparison | registered δ | observed CI | half-width `h` | effective required effect |
|---|---:|---|---:|---:|
| §11 joint vs task | `0.020` | `[-0.02898,+0.03194]` | `0.0305` | `-0.0305` |
| §11 joint vs matched | `0.010` | `[-0.01891,+0.04152]` | `0.0302` | `-0.0302` |
| §13 visual vs task | `0.020` | `[-0.08920,+0.01334]` | `0.0513` | `-0.0513` |
| §13 visual vs no vision | `0.010` | `[-0.05831,-0.00470]` | `0.0268` | `-0.0268` *(fired: point `-0.0307`)* |

So §11 and §13 each declared a `0.02` effect meaningful while being powered
only to resolve effects 1.5× and 2.6× larger. **Failing an unreachable gate is
not evidence of no effect.** §13 correctly reports "not statistically secure";
the stronger and more accurate statement is that the visual-vs-task comparison
was *inconclusive by construction* on both panels. This does not rescue either
mechanism — the absolute gate failed independently and by a wide margin — but
it does mean the visual-vs-task question has never actually been tested at
adequate power, and no future preregistration should fix a gate without a prior
variance estimate.

**Conclusion.** Queueing the dense action-conditioned JEPA successor against
the same `0.13` absolute gate, on a panel of unmeasured difficulty, with no
power calculation, risks a third pre-doomed experiment at collection cost. A
and B are the two cheap measurements that fix this.

---

## 2. Work package A — observability-ceiling assay

### 2.1 Question

On the exact §13 V3 panel, what is the lowest scene-disjoint normalized rank
regret achievable by a readout of the available observations, when prediction
error is removed by supplying **actual** successor observations and readout
capacity is not artificially constrained?

This is an oracle/achievability assay in the sense SAINTS §4 already endorses.
It is explicitly **not** a world-model, dynamics, prediction, navigation, or
promotion experiment, and no arm in it is eligible for closed-loop use.

### 2.2 Data — no new collection

The immutable V3 collection at
`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3/attempt_v1/collection`:

- 64 scenes: 32 train, 32 evaluation, scene-disjoint;
- 256 states, 2,304 matched nine-action successor branches;
- 3,072 PNGs on disk (768 context + 2,304 successor), 245 scene meshes;
- registered collection SHA-256
  `711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0`.

No Genesis, no rendering, no recollection, no scene filtering.

### 2.3 Custody cost — declare before running

A opens the **evaluation-side successor PNGs** (1,152 frames across the 128
evaluation states) for the first time. The §13 attempt and its complete-tie
diagnostic both opened zero. After A, this panel is spent for
privileged-successor purposes and cannot serve as a fresh successor role again.
This is the real price of A and must appear in its preregistration. It does not
touch untouched, sealed, or V4 material.

### 2.4 Arms

All fit on the 32 train scenes, evaluated on the 32 disjoint evaluation scenes,
under the §13 scorer, one-centimetre tolerance, state/scene panel, bootstrap,
and the §13 complete-tie convention, all unchanged.

| # | arm | input | role |
|---|---|---|---|
| 1 | privileged physical oracle | true physical outcome | integrity anchor, regret `0` by construction |
| 2 | privileged physical successor state | simulator successor pose/contact | bound if physics were predicted perfectly |
| 3 | **privileged true-successor dense DINOv2** | actual successor RGB | **primary: visual information ceiling** |
| 4 | privileged true-successor dense V-JEPA 2.1 | actual successor RGB | representation comparator |
| 5 | context-only | 3 context frames + action, no successor | achievable-without-prediction floor; what a deployed system can see |
| 6 | task/action-only ridge | action + history, no vision | existing non-visual control, refit on this panel |
| 7 | random expectation | — | anchor |

Arms 2–6 share one readout architecture. Capacity is swept over three declared
widths; the width is selected by **train-side** fit (require train regret to
approach zero, confirming the assay is not capacity-limited) and never by
evaluation score. All three widths are reported.

### 2.5 Pre-declared decision rule

Let `R*` be the evaluation regret of arm 3.

- **Outcome I — gate achievable.** `R* ≤ 0.13`. The `0.13` gate stands; the
  §13 residual (`0.272`) is prediction error; the dense action-conditioned JEPA
  successor is justified as specified. Proceed to preregister it.
- **Outcome II — gate too tight, visual information present.** `R* > 0.13`,
  **and** `R*` beats arm 6 with CI upper bound below zero. The absolute gate is
  re-derived ceiling-relative (successor must reach `R* + δ`, with `δ` fixed
  from A's own variance estimate). The successor is justified but re-gated.
- **Outcome III — no visual headroom.** `R* ≥` arm 6, or `R*` materially worse
  than arm 2. No predictor architecture can close this: one-step rank regret
  from observations is not a solvable target on this panel. Stop the
  matched-branch rank-regret program and move the primary endpoint to B's
  metric.
- **Outcome IV — panel degenerate.** Arm 5 ≈ arm 3, i.e. actual successors add
  nothing over context. The branch construction does not produce visually
  distinguishable outcomes; the action grid or step length needs redesign
  before any successor experiment is collected.

Outcome IV is a live possibility, not a formality. §13 reported a scene where
"the robot moved by only millimetres" across all nine actions, and a
class-count histogram of `{1:4, 2:6, 3:4, 4:3, 5:6, 6:12, 7:38, 8:94, 9:89}`.

### 2.6 Mandatory diagnostic

Report evaluation regret **conditioned on per-state branch physical-displacement
spread**, in declared quantile bins. If the achievable ceiling is dominated by
low-spread states, the corrective action is the action grid / step length, not
the architecture. This diagnostic is cheap and would change the design of the
successor collection.

### 2.7 Power output

A also produces the variance estimate that §11 and §13 lacked. Report, for each
paired comparison, the bootstrap CI half-width at this panel size, and the
number of scenes required to resolve a `0.02` effect. **Every subsequent
preregistration on this family must cite that number before fixing a gate.**

### 2.8 Cost and environment

Encoder passes over 3,072 PNGs plus readout fits: well under one GPU-hour.
GPU torch via `~/TinyQuadJEPA/bin/python` (ROCm). Genesis is not required, so no
Vulkan venv. Deterministic-repeat and byte-exact re-evaluation gates as in the
§13 diagnostic.

---

## 3. Work package B — metric-validity study

### 3.1 Question

Does one-step normalized rank regret predict closed-loop planning usefulness on
this stack?

### 3.2 Why it is not rhetorical

Two Aug-4 results already point opposite ways about the same representation:

- **Planner-oracle assay (H1 PASS).** Correct candidate ranking converts to
  materially better control: `+0.4320` m progress over deterministic shuffled
  scores, CI `[0.3773, 0.4833]`, 24/24 scenes, oracle first-action regret
  exactly `0`. The planner seam is *not* the bottleneck.
- **DINO true-successor goal-cost (gate FAILED).** The same frozen-DINO cost
  achieved **14/24 successes — exactly matching the geometric oracle's 14/24** —
  and `+0.29568` m over shuffled, CI `[0.10139, 0.47261]`. Yet it *failed* its
  first-action geometric regret gate: `0.01574` m against a required `0.020` m,
  CI `[-0.04360, +0.01188]` crossing zero.

So one scorer is simultaneously regret-gate-failing and closed-loop
oracle-matching. Either the regret gate is mis-specified, or closed-loop success
on this harness is insensitive. B determines which, before that distinction is
used to close a research line.

### 3.3 Design

Reuse the 24-scene development closed-loop harness that both Aug-4 assays used.
For each scorer, measure **both** endpoints:

1. one-step normalized rank regret on the matched panel;
2. closed-loop progress and success on the 24-scene harness.

Scorers spanning the regret range (n ≈ 8–9): privileged physical successor,
true-successor DINO, true-successor V-JEPA, §13 visual recurrent checkpoint,
§13 no-vision checkpoint, task/action ridge, shuffled scores, persistence,
random. Estimate the rank correlation between the two endpoints across scorers,
with a bootstrap CI.

**Honest limitation, stated up front:** n ≈ 8 scorers gives a wide correlation
CI. B is powered to distinguish "strong monotone relationship" from "no
relationship", not to estimate the correlation precisely. That is sufficient
for its decision rule and B must not be reported as more than that.

### 3.4 Pre-declared decision rule

- **Strong monotone relationship.** Rank regret is a valid cheap proxy; keep it
  as the primary endpoint, governed by A's re-derived threshold.
- **Weak or absent relationship.** The gate is invalid as a promotion criterion.
  Replace the primary endpoint with closed-loop progress against the oracle on
  development scenes, and demote one-step rank regret to a diagnostic.

### 3.5 Relation to the stated research method

SAINTS §10.3: *"Collapse diagnostics are necessary but are not substitutes for
obstacle semantics or planning utility."* B is the operationalization of that
sentence. It also directly targets two of the four recurring methodological
hazards catalogued in SAINTS §9.2.1 — *teacher-forcing confusion* (a one-step
metric looking strong or weak while closed-loop behaviour disagrees) and
*component misattribution*.

---

## 4. Work package C — audit-trail custody

### 4.1 The exposure

- 834 untracked paths: 315 `docs/`, 151 `scripts/`, 330 `lewm/`, plus 2 stray
  result JSONs at repository root.
- 36 modified tracked files, `+12,476 / −1,067`;
  `scripts/train_go2_egomotion_bev_jepa.py` alone is `+2,524` lines.
- Every plan builder, runner, scorer, evaluator, and result document for
  handoff §9–§13 is **untracked**.

The handoff records SHA-256 for artifacts, but the *source that produced them*
is not in version control, and the recorded execution HEADs do not describe the
working tree that actually ran. A `git clean` — or a disk failure — currently
destroys the reproducibility of roughly two weeks of registered results.

This also bears directly on SAINTS §7.2.2 (reproducibility) and §10.1, which
claim configuration hashes and a reduced reproduction path as project outputs.

### 4.2 Action

Commit in coherent, reviewable chunks rather than one mega-commit:

1. world-model experiment infrastructure and its documents;
2. benchmark, dataset, and planning source changes;
3. tests.

`.generated/` remains gitignored; no artifact bytes are committed. Run the
`lewm/tests` suite before committing and report the result honestly rather than
committing a tree of unknown status. Record in the handoff whether HEAD now
matches the code that produced §9–§13, and state precisely where it does not.

### 4.3 Sequencing note

The requested order is A → B → C. C first would protect A's and B's own
artifacts as they are produced, at no scientific cost. Flagged for the decision;
the stated order is otherwise followed.

---

## 5. Reconciliation with the SAINTS Year-1 progression document

The progression document is dated 4 August 2026 — one day before this plan —
but its §9.3/§9.4 describe the repository as bottlenecked on **perception
qualification** (RGB→BEV occupied-space semantics), and its progress narrative
ends at the direct-BEV and signed-boundary experiments. Four issues follow.

### 5.1 The largest recent evidence block is missing

Nothing in §9 records the Aug 1–5 matched-counterfactual action-utility
programme. A new §9.3.4 should record, compactly:

- the three-arm V3 localization/alignment result and its terminal stop;
- the scene-disjoint matched-branch panel and the four-arm × three-seed
  evaluation;
- the frozen V-JEPA 2.1 physical-interface ceiling (privileged actual-future
  tokens failed against retained-physical and task/action controls);
- the grounded dense-DINO joint-JEPA stop;
- and the one genuine positive: on the expanded 64-scene panel, visual context
  beat matched no-vision by `-0.03065`, CI `[-0.05831, -0.00470]`, entirely
  below zero — reversing the §12 result, which crossed zero. This is the first
  matched visual value the programme has demonstrated and it should be reported
  as such, alongside the fact that absolute regret (`0.272`) remains far above
  the registered ceiling.

### 5.2 A dependency-order inconsistency to resolve

SAINTS §5 sequences WP1 (perception qualification, Oct–Dec 2026) strictly
before WP2 (joint JEPA, Nov 2026–Feb 2027), and states that "no dependent
experiment or claim will proceed until the preceding milestone has been met."
The Aug 1–5 programme ran joint dense-DINO JEPA training and recurrent dynamics
experiments using **frozen external encoders** (DINOv2, V-JEPA 2.1) rather than
the repository's unqualified BEV representation.

That is defensible — it asks a different question, on a different substrate,
about whether recoverable action signal exists in observations *at all* — but
the progression document currently implies a linear order the work has not
followed. Two honest repairs:

- **(a), recommended:** declare the matched-branch programme a *parallel
  diagnostic line* whose purpose is to de-risk WP2's premise independently of
  WP1's outcome. This is both true and good practice: if there is no
  recoverable one-step action signal in observations, WP2 fails regardless of
  whether WP1 qualifies, and it is far cheaper to learn that now.
- **(b):** re-order WP1 and WP2 in the Gantt.

Option (a) also gives the document a clean answer to the obvious examiner
question of why joint-JEPA work appears before perception qualification.

### 5.3 A is the standard the document already sets

SAINTS §4: *"An oracle has shown that the gate is achievable, and learned
candidates that fail are blocked rather than promoted as the 'closest' result
or tuned on untouched scenes. This creates a traceable link from a physical
requirement to immutable evidence."*

That is precisely work package A, applied to the rank-regret gate instead of the
perception gate. The strongest available justification for A is the project's
own stated methodology.

### 5.4 The risk register needs one row, and one row is already firing

Existing row — *"Latent rollout adds no value over reactive heads / Medium /
Medium / Treat the reactive baseline as mandatory. If no benefit is observed,
narrow Paper A to the representation–memory interface and report rollout as a
negative result."* The Aug 1–5 results are substantial evidence toward this row
firing. B determines whether that evidence is being read on the right metric
before the narrowing decision is taken.

Proposed new row — *"Primary one-step metric does not predict closed-loop
utility / Medium / High / Validate the proxy against closed-loop outcomes
across scorers spanning its range before using it as a promotion gate; demote to
diagnostic if uncorrelated."*

### 5.5 The statistics self-assessment is over-modest, but names a real gap

At §10.5 the document flags: *"This is a significant weak point for me, I've
done very little statistical analysis, let me know if the above is sensible or
not."*

The practice actually in use is stronger than typical robotics practice:
preregistered gates fixed before observation, 10,000-draw family-balanced
whole-scene bootstrap, paired confidence intervals, scene-clustered resampling,
pre-declared decision rules, independent recomputation, and honest reporting of
partial passes. That is defensible as written and should be stated with more
confidence.

The genuine, concrete gaps are three, and all are fixable:

1. **No power analysis before fixing gates.** As §1.2(iii) above shows, the
   "visual beats task/action" gate was unreachable at the panel size used in
   both §11 and §13. Fix: estimate the paired CI half-width from a pilot or
   from A's output, and require the registered effect size to exceed it before
   the gate is frozen.
2. **Fixed absolute thresholds across panels of different difficulty.** A fixed
   control arm moves `0.174 → 0.308` between panels. Fix: gate relative to a
   measured achievable ceiling (A's output), not an absolute constant.
3. **Multiplicity across many simultaneous gates with no stated family-wise
   policy.** Runs routinely score 5–10 gates. Fix: declare in advance which gate
   is primary and which are supporting, or state an explicit correction.

Adding a short paragraph on power and ceiling-relative gating to §10.5 would
convert the acknowledged weak point into a strength.

---

## 6. What this plan does not authorize

No data generation, rendering, bulk re-render, pool scaling, training of any
navigation or promotion candidate, threshold relaxation, retry or resume of any
stopped mechanism, planner integration, blind rollout, deployment, or any
access to untouched, sealed, held-out, or V4 material. A and B are
development-tier and non-promotional. Each requires its own preregistration,
independent source review, and execution authority before it is run.
