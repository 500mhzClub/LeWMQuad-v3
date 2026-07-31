# First-principles review: JEPA navigation campaign vs. stated aims

Date: 2026-07-31

Scope: independent re-evaluation of the 2026-07-24 → 2026-07-31 campaign against
the repository goal, triggered by the full handoff at commit `0279294`
(`docs/lewm_go2_learned_jepa_navigation_full_handoff_2026-07-31.md`).

Status of this document: **review and proposal. Not a scientific result, not a
qualification, not an authority.** It cites existing authorized scientific artifacts as
evidence and adds one development-tier measurement that is explicitly marked
non-citable.

Authority correction: no July 31 world-model, action-conditioned, or
counterfactual execution-authorization file exists in the durable repository
record. Temporal V1's one-shot authority is consumed, with retry and resume
false. This review grants no execution permission. Until fresh explicit
development authority is granted, its actionable scope is document/source review
and synthetic-only work; references to development outputs below are historical
records, not retrospective authority or a license to continue them.

Post-review source audit invalidated two later exploratory interpretations. The
geometry output used a noncanonical footprint and an actual-spawn fallback. The
counterfactual output used the wrong H6 cadence/actions, unbound provenance, and
kinematic targets. Current hardened sources reject those protocols and have only
synthetic test evidence; no corrected runtime result exists.

---

## 1. Summary of disagreement

The handoff concludes that temporal V1 was a *valid negative result* caused by a
low-rank predictor shortcut, and recommends a rank-preserving V2.

This review finds:

1. Temporal V1 validly failed its preregistered update-50 continuation gate and
   its one-shot attempt remains consumed. A later non-citable development replay
   shows that the statistic **bottomed at update 25, recovered, and crossed the
   0.25 threshold by update 200** — reaching 0.3442 by the 400-update cap while
   the persistence gap closed from 4.64× to 1.15×. This challenges the stronger
   claim that the *mechanism* was closed: the gate fired 150 updates before the
   replay would have passed it. It does not retroactively pass or reopen V1.
2. The recommended V2 remedies (whitened-target loss, variance/covariance floor)
   directly optimize rank without a demonstrated task link and could trade
   total-error accuracy for that rank. The review did not measure those remedies,
   so their effect on persistence remains a hypothesis.
3. Three systemic risks are supported: unstable verdicts at very small training
   scales, promotion-style ceremony applied to exploratory questions, and
   gate/aim decoupling. They do not establish one universal minimum scale or
   license governance-free development; task-coupled bounded pilots and explicit
   authority must decide each next tranche.
4. The campaign never touched navigation, formal G2, or held-out evaluation, so
   it cannot establish the repository objective. A later source audit invalidated
   both the initial "corridors are too narrow" claim and the attempted reversal:
   the historical geometry run used a noncanonical footprint and a spawn
   fallback. Archived coverage correlations are descriptive and do not localize
   the causal limiter.

Item 1's mechanism-level challenge is supported by the non-citable development
measurement in §4; the registered attempt status is unchanged. Items 3 and 4
are the substantive review findings. Three claims made during this review were
tested and **withdrawn**:
that the 0.25 rank threshold could be failed by an optimal predictor (§2.3);
that corpus corridors were too narrow for the robot (§5b.0); and that
guard/veto conservatism was the coverage limiter (§5b.3). Each is recorded where
it was made rather than deleted.

---

## 2. The rank gate: what it does and does not establish

### 2.1 What the gate measures

`lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py:800-848`
computes a position-centered cross-sample covariance over the 192-dim feature
axis and reports its entropy-exponential effective rank. The continuation gate
requires

```
effective_rank(prediction) / effective_rank(EMA target) >= 0.25
```

Observed at the terminal sentinel: `4.0974 / 22.5848 = 0.1814`, shortfall
`1.5488`.

### 2.2 The signature was read incorrectly

The result JSON records:

| quantity | update 0 | update 50 |
|---|---:|---:|
| prediction effective rank | 19.6967 | 4.0974 |
| prediction cross-sample variance | 0.2290 | **0.5020** |
| target cross-sample variance | 0.8010 | 0.8043 |
| mean JEPA loss | 0.6368 (u1) | 0.2154 |

The handoff reads *rank down while variance up* as evidence of a shortcut:
"prediction variance concentrated into very few directions."

Falling rank together with **rising** variance and **falling** loss can occur as
an L2-trained predictor moves away from random initialization; it is not by
itself a distinguishing marker of a shortcut. A controlled simulation with no
shortcut available, converging to its exact closed-form optimum, reproduces it:

```
step     0  loss=1.6523  pred_erank= 53.58  var=0.1578   <- random init
step    50  loss=1.2922  pred_erank= 34.55  var=0.1470   <- gate fires here
step   250  loss=1.0044  pred_erank=  8.42  var=0.4099
step  4000  loss=0.9847  pred_erank=  7.50  var=0.5163   <- correct asymptote
```

including the variance rise from ~0.15 to ~0.52 against V1's 0.229 → 0.502.

Two consequences:

- **Update 0 is not a healthy reference.** The predictor is randomly
  initialized; its effective rank of 19.70 is isotropic noise. The "79.2% rank
  drop from update 0" recorded in the result JSON measures departure from
  noise, not degradation of a learned state.
- **The gate fired inside the transient.** In the simulation the rank at the
  equivalent step is still far above its own asymptote and still falling. No
  measurement in the V1 artifact set establishes where V1's rank settles,
  because the registered full retention panel was scheduled for update 200 and
  the run stopped at 50.

### 2.3 The threshold was not shown impossible; its suitability remains unvalidated

The 0.25 floor was preregistered without deriving what a *correct* predictor's
rank would be. Two reference values were needed and never computed:

- the effective rank of the **persistence baseline's** prediction (the encoded
  current frame), and
- the effective rank of a defensible **conditional-mean reference** for the same
  target. The measurement below supplies only a fitted linear reference, not a
  global optimum.

§4 measures both. A least-squares linear conditional-mean reference, fitted and
evaluated on ordered, unstratified sentinel halves, retains ~0.55–0.60 of target
rank, above 0.25. That restricted result withdraws the claim that an optimal
predictor was shown able to fail the gate; it neither validates 0.25 nor
establishes a nonlinear oracle.

The measured defect is the single-point application: the gate was evaluated at
update 50 against an unmeasured training trajectory, with no rule separating
"below threshold and rising" from "below threshold and falling." §4 shows it
was rising in the later replay. Whether 0.25 is task-appropriate remains open.

### 2.4 Causal trends did not determine the registered continuation decision

At the same sentinel the causal controls were improving and, in the case of
history, unambiguous:

| control (lower better) | update 0 | update 50 | families favouring correct |
|---|---:|---:|---|
| current-only reset | 0.9836 | 0.9574 | 8/8 |
| wrong history | 1.0019 | 0.9684 | 8/8 |
| wrong action | 1.0106 | 0.9921 | 6/8 |
| persistence | 4.5769 | 1.6588 | 0/8 |

The persistence gap closed from 4.58× to 1.66× in 50 updates. History controls
favored the correct ordering across families, while wrong-action evidence was
weaker at 6/8. These were promising causal trends, not proof of deployable
counterfactual action utility. The frozen continuation rule validly terminated
the attempt on rank; the later replay shows that rank point was insufficient for
the broader mechanism-level inference.

### 2.5 The recommended V2 is goal-decoupled as specified

The handoff proposes making the objective "rank-preserving" via whitened
EMA-target coordinates, a dimension-balanced loss, or a prediction
variance/covariance floor.

- **Whitened-target loss** equalizes weight across target directions and may
  upweight low-variance, less-predictable directions. That could buy rank at the
  cost of total-error accuracy, but was not measured here.
- **Variance/covariance floor** directly pressures output spread. Without a
  task-coupled justification, it may add variance not supported by context; its
  actual persistence effect is unmeasured.

Both remedies target the rank statistic rather than deployed action utility.
Neither is justified as the next experiment on the current evidence.

---

## 3. Systemic findings

### 3.1 Scale: the runs were too small for several mechanism-level closures

| | measured |
|---|---|
| Spatial V1 ("qualified") wall-clock | **151 s** (08:55:xx → 08:58:xx) |
| Temporal V1 wall-clock | **59 s** (10:41:55 → 10:42:54) |
| Batch size | 16 presentations (spatial) / 10 sequences (temporal) |
| LR schedule | none — constant LR, no warmup |
| Registered Temporal V1 maximum | 4,000 H6 sequence rows / 16,000 logical RGB presentations |
| Available pool | ~1.81M H6 candidates / 55.2M frame rows / ~2.896 TB |
| Unique-candidate fraction at the registered maximum | **~0.22%** of H6 candidates |
| GPU during this review | 8 W, 3% utilization, 33 °C — idle |

The encoder is ~2.7M parameters (112 px, patch 7, dim 192, depth 6, 256 tokens).
These wall-clock values are end-to-end run durations, not isolated GPU-compute
measurements. A later prepacking exercise suggests input delivery mattered, but
its pre-hardening timing and bit-identity output did not satisfy the current
provenance contract. The defensible scale observation is the registered exposure
count and ~0.22% candidate fraction, not an exact claim about seconds of GPU math
or how the 151 s decomposed.

At batch 10–16, constant LR, and 500–16,000 total samples, some results do not
separate "the mechanism is wrong" from "the model has not reached a stable
regime." This is not a claim that every negative result is invalid or that one
universal sample floor applies. Two closures share V1's exact early low-rank
signature — recurrent-H4 persistence-residual and delay-line V4 — and should be
treated as unresolved pending trend-based, task-coupled evidence.

The main-pool audit's conclusion that data is "an implausible primary
explanation" is defensible for **breadth** — the 4,000 rows do span families,
scenes and actions. It has silently been generalized to volume, which it does
not support.

### 3.2 Process mismatch: promotion ceremony applied to exploratory questions

The review counted 1,127 documents, 134 in a single day, and approximately seven
governance artifacts around many experiments (preregistration, source manifest,
source review, clean-export certification, execution authorization, scientific
result, independent review). Those counts signal process weight, but they do not
measure authoring effort or prove an orders-of-magnitude cost ratio.

One-shot authority, attempt consumption, and no-retry are appropriate for the
repository's protected promotion roles, including formal gates before G8. When
the full promotion ceremony is also used for every exploratory schedule or
wiring question, plumbing failures and hyperparameter mistakes inflate the
number of apparent scientific mechanisms. The handoff itself notes that many
versions were plumbing recovery.

A lighter development process may be scientifically preferable, but this review
cannot create it. It requires an explicit governing decision and operation-level
authority; `.generated/dev/**` and "non-citable" are evidence classifications,
not permission to run.

### 3.3 Gate/aim decoupling

Every gate in the campaign is intrinsic: effective rank, persistence ratio,
action NLL, control ratios, physical-margin scopes. None has a demonstrated
mapping to navigation success, and this repository has twice produced evidence
against such mappings:

- the recognition-not-metric finding (frozen latents with ρ≈0.03 still support
  successful closed-loop beacon navigation), and
- three separate instances of a frozen channel outperforming a learned head
  (GoalAdapter, GoalEnergyHead, raw-frame cosine goal matching).

Beating persistence in total latent MSE is not by itself required for navigation.
Consecutive frames are similar, so persistence is a strong baseline on *total*
error; navigation may instead consume the *action-dependent delta*. V1's own
control table shows a developing delta while total error trailed. Task-coupled
action ranking/regret is therefore needed before deciding how much the total-MSE
comparison matters.

### 3.4 The repository objective was not touched

The 2026-07-24 → 07-31 campaign was perception-only; it executed no navigation,
formal G2, or held-out evaluation. It therefore cannot establish persistent
physical/per-color belief, learned target/frontier/route/action selection,
deployed predictor causality, robustness, or novel-maze four-beacon completion.
The later geometry work exposed the narrow-corridor claim as unsupported, but
its attempted opposite conclusion was also invalidated by source review (§5b).

---

## 4. Development-tier measurement: where does prediction rank settle?

Probe: `scripts/dev_temporal_v1_rank_asymptote_probe.py`.
Output: `.generated/dev/temporal_v1_rank_asymptote/rank_asymptote.json`.

**Not citable as scientific evidence.** It emits no checkpoint, does not touch
the consumed attempt root, and opens no held-out or sealed material. It reuses
the frozen reviewed modules unchanged for the model, data roles, schedule,
masks and the training step; the deliberate deviation is that the continuation
gate is observed but not enforced. The frozen training step hard-caps at 400
updates, so the 4,000-row schedule was not cycled beyond its registered budget.

It adds the three measurements the authorized run lacked:

1. prediction effective rank tracked well past update 50;
2. effective rank of the **persistence** prediction (encoded current frame);
3. effective rank of a **least-squares linear conditional-mean reference**
   E[future | current] and E[future | current, action], fit closed-form on half
   the sentinel and measured on a disjoint evaluation half.

Fidelity check — the probe reproduces the authorized V1 update-0 sentinel exactly:
prediction/target rank ratio `0.8574` = `19.6967 / 22.9730`.

### 4.1 Results

The probe used the frozen `training_update_v1` unmodified, which hard-caps at
`MAXIMUM_UPDATES_V1 = 400`. It therefore ran **exactly the registered budget of
temporal V1** — no cycling was possible without reimplementing the step, which
was not done. Every number below comes from the reviewed modules.

| update | loss | pred erank | pred/target ratio | gate (≥0.25) | persistence ratio |
|---:|---:|---:|---:|:--|---:|
| 0 (random init) | — | 19.70 | 0.8574 | n/a | 4.639 |
| 10 | 0.3445 | 4.43 | 0.1934 | fail | 2.369 |
| 25 | 0.2726 | 3.73 | **0.1641** (minimum) | fail | 1.884 |
| **50** | 0.2154 | 4.10 | 0.1814 | fail ← **authorized run terminated here** | 1.667 |
| 75 | 0.2071 | 4.32 | 0.1918 | fail | 1.568 |
| 100 | 0.2167 | 4.42 | 0.1952 | fail | 1.492 |
| 150 | 0.1683 | 5.61 | 0.2411 | fail | 1.374 |
| 200 | 0.2256 | 6.54 | 0.2694 | **PASS** | 1.308 |
| 300 | 0.1880 | 9.04 | 0.3407 | **PASS** | 1.201 |
| 400 (cap) | 0.1625 | 9.84 | 0.3442 | **PASS** | 1.149 |

Reference values that the authorized run never measured:

- target effective rank: 22.97 at update 0, drifting to **28.59** by update 400
  (the EMA target encoder trains too, so the ratio's denominator *grew* — the
  prediction caught up against a moving, richening target);
- **persistence** prediction effective rank: 29.16, ratio 1.020 — persistence is
  a *sample*, not a conditional mean, so it trivially matches target rank and is
  not a meaningful health reference;
- **fitted linear reference** E[future | current]: effective rank 17.17, ratio **0.601**;
- **fitted linear reference** E[future | current, action]: effective rank 15.77, ratio
  **0.552**.

Fidelity: the probe reproduces the authorized V1 update-50 ratio to nine
significant figures (`0.181424597` vs `0.1814245973`).

### 4.2 What this development measurement supports

1. **The update-50 rank failure was a transient in this replay.** The
   ratio bottomed at update 25 (0.1641), was already recovering when the gate
   fired at update 50, crossed the 0.25 threshold by **update 200**, and reached
   0.3442 by update 400. The authorized attempt terminated 150 updates before
   this replay crossed the same numeric threshold; update 200 was a registered
   observation point. Its original failure status remains valid.
2. **The update-50 observation does not support stable mechanism closure.**
   Effective rank rose monotonically from update 25 onward, roughly 2.6× by the
   cap. This refutes treating the update-50 rank as an established asymptote; it
   does not rule out every shortcut or generalization failure.
3. **The persistence gap continued to close**: 4.639 → 1.149, still falling at
   the cap, i.e. within ~15% of the persistence baseline rather than the 1.66×
   recorded at termination. The cap does not establish whether parity is the
   asymptote.
4. **The 0.25 threshold was not unreasonable relative to the fitted linear
   reference, but it was unanchored.** That reference retains ~0.55–0.60 of
   target rank; it does not establish the rank of a nonlinear oracle. The
   earlier optimal-predictor claim is withdrawn. The measured defect is that the
   gate was evaluated at a single early point against an unmeasured asymptote,
   with no stopping rule distinguishing "below threshold and rising" from
   "below threshold and falling."
5. **The true asymptote remains unmeasured.** At the 400-update cap the rank is
   still rising and still well below the ~17 linear-predictable reference. Where
   it settles remains open. Training past the cap would require fresh explicit
   authority and is justified only if a task-coupled action metric needs that
   scale; total-MSE persistence parity alone is not the repository objective.

### 4.3 Consequence for the recommended next experiment

Temporal V1 validly failed its frozen gate, but the stronger mechanism-level
rejection relied on a measurement that reverses in the development replay. A
rank-preserving V2 aimed only at that transient is therefore not justified.

This does **not** establish that the same mechanism merely needs more scale. The
next decision must measure scene-disjoint counterfactual action utility and task
regret, distinguish training fit from generalization, and only then choose more
data, a revised objective, an architectural successor, or no further work.

---

## 5. Proposed next steps

These are decisions, not execution grants. The current durable boundary is
document/source review and synthetic-only work until fresh explicit development
authority exists.

**0. Do not run the rank-preserving V2 as specified.** Its registered remedy is
aimed at an early rank transient and is not connected to task action utility.

**1. Preserve formal names and evidence classes.** Use `WM-*` for world-model
diagnostics. Reserve G2-G8 for traversability, persistent coverage, learned
frontier value, target conversion, full development, robustness, and the final
one-shot evaluation. A non-citable development result cannot pass or authorize
any formal gate.

**2. Reconcile the current substrate before scaling.** Re-measure the worsened
spatial-control panel, audit the physical interface, and treat place retrieval as
an early diagnostic rather than physical requalification. Keep the recurrent-H4
and delay-line low-rank closures unresolved rather than automatically reopening
or rerunning them.

**3. Size, do not launch, a counterfactual pilot.** Within the current
source/synthetic-only boundary, use reviewed source and recorded counts to
estimate scene/state/action coverage, bytes, render/GPU wall-clock, and the exact
authority needed before any runtime input is opened. Large generation is not
justified by the 94-group overfit diagnostic.

**4. If fresh authority is granted, run a bounded scene-disjoint WM-A pilot.**
Train the unchanged architecture with within-state contrast, then distinguish:
failure to fit training groups; training fit without withheld-scene
generalization; latent fidelity without planner action-regret benefit; or genuine
withheld-scene action utility. Only the last outcome justifies a larger data
tranche. Freeze referenced thresholds before confirmatory use.

**5. Prefer a one-step causal scorer.** Evaluate all candidate actions, compare
action ranking/regret with action-blind/current-state controls, and integrate it
into an existing navigation seam as a development diagnostic. The existing
classical seam is not itself a promotable fully learned controller. Build a
composable full-state head and K-step role only if task evidence localizes the
remaining failure to sequence horizon.

**6. Construct a candidate before formal G2.** Finalize the world-model
interface, run the exactly matched no-JEPA development arm, requalify physical
perception after every encoder change, and freeze the complete source,
checkpoint, calibration, and threshold identities. Passing place/topology or
counterfactual probes is insufficient.

**7. Keep the causal candidates matched.** The archived suite motivates testing
tick budget, forward displacement, escape/yaw churn, guard behavior, map
resolution, perception, and geometry; it ranks none of them causally. Do not
re-render wider corridors or exclude `local_composite_motifs` from the invalid
legacy geometry output. Revisit either only after a corrected, authorized audit.

**8. Follow the existing formal promotion order.** G2 → G3 → G4 → G5 → G6 →
G7 → fresh eligible one-shot G8. Before G6, verify that target,
frontier/viewpoint, route/subgoal, and ordinary-motion selection are learned and
that the predictor is causally present. Retain the fixed matched no-JEPA arm.

---

## 5b. Geometry and coverage audit — second correction

The original review claimed maze corridors were physically too narrow. A first
revision then claimed a development raster audit refuted that hypothesis. Source
review shows that **neither conclusion is established**.

The historical output in `.generated/dev/family_clearance_audit/` swept values
0.12 / 0.20 / 0.24 / 0.27 m as though they were interchangeable body radii. The
repository's applicable yaw-invariant planning contract is
`OBSERVABLE_FOOTPRINT_RADIUS_M = 0.47`. The old implementation also substituted
the largest traversable component when the actual spawn was infeasible and
reported a median connected fraction as a family-level "connected" value. A
raster disc, even when implemented correctly, is only a planning proxy: it does
not model orientation, turning sweeps, dynamics, contact, gait, or rough terrain.

The table that motivated the attempted reversal is retained as audit history:

| family | n | historical fits% | historical connected summary |
|---|---:|---:|---:|
| open_obstacle_field | 130 | 94.4% | 100% |
| rough_local_dynamics | 79 | 93.5% | 100% |
| loop_alias_stress | 175 | 83.5% | 100% |
| medium_enclosed_maze | 356 | 79.4% | 100% |
| large_enclosed_maze | 239 | 78.4% | 100% |
| small_enclosed_maze | 196 | 76.3% | 100% |
| local_composite_motifs | 196 | 91.7% | 9.2% |

Those numbers do not establish physical corridor adequacy, all-scene
connectivity, or a valid exclusion decision for `local_composite_motifs`. The
current source defaults to 0.47 m, fails closed at an infeasible actual spawn,
separates spawn reachability from largest-component size, records input hashes,
and labels its four-neighbour disc result non-physical. It has passed synthetic
tests but has not produced a corrected corpus result.

The archived 115-run closed-loop summary is also descriptive rather than causal:

| marginal correlation with cells visited | r |
|---|---:|
| distance travelled | +0.378 |
| escape blocks | +0.346 |
| tick budget | +0.342 |
| guard block rate | +0.059 |
| contact-like stalls | −0.049 |

Distance is the largest of these five marginal associations, but run variants,
budgets, policies, and environments are confounded. One veto-off run travelled
34.6 m and visited six cells, whereas one guarded run travelled 6.6 m and visited
14; this shows only that guard removal was not sufficient in that comparison.
It does not refute guard effects generally or establish travel throughput as the
causal limiter. The privileged-DFS runs' 11--14 m travel likewise do not isolate
an observability ceiling.

Consequences:

- The original "too narrow" claim and its "corridors are adequate" reversal are
  both withdrawn. Physical navigability remains unresolved.
- A roughly 3 TB wider-corridor re-render is not justified without a corrected,
  explicitly authorized audit and a task-coupled intervention.
- Tick budget, displacement, escape/yaw churn, guard behavior, map resolution,
  perception, and family geometry remain matched factors or hypotheses, not a
  ranked causal diagnosis.
- No family should be excluded or regenerated on the legacy raster output.

---

## 6. Useful task-coupled development evidence

Two historical results are closer to task-coupled evidence than the July
campaign's intrinsic metrics. Under the governing correction, previously visible
test panels are development evidence, not formal held-out or promotion results:

- **Scene-disjoint development safety comparison**: 6 disjoint test mazes × {baseline, +risk},
  contacts per 100 ticks 51.3 → 12.6 (4.1×, every scene improved), 0 falls,
  claims unchanged, frozen risk model AUC 0.74–0.90 on that visible panel.
- **Topological navigation development demonstration**: goal image only, no
  privileged policy signals, 4.00 m over 5 hops to a 0.36 m final error with a
  perceptual stop; wide-maze beacon-goal success at 0.46 m final, 0 falls.

These motivate task-coupled diagnostics. They do not satisfy the corrected fully
learned target/frontier/route/action architecture or any formal G2-G8 gate.
