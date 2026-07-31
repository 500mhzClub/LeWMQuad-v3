# Session handoff: JEPA world-model review, correction, and counterfactual diagnosis

Date: 2026-07-31
Repository HEAD at session start and end: `0279294e5679fa81f845f13c9e47326bab291238`
Branch: `jepa-spatial-world-model-nav`

**Nothing in this session is committed.** All artifacts are untracked (§8).
All experimental outputs are development-tier under `.generated/dev/**` and are
explicitly **not citable as scientific evidence**. No sealed, held-out, or G8
material was opened. No sealed attempt was consumed.

This handoff is a record and proposal, not an authority or preregistration. The
development label does not itself authorize data generation, artifact mutation,
GPU work, training, or evaluation. Formal G2-G8 meanings and authority boundaries
remain unchanged.

Authority audit after the session found no July 31 world-model,
action-conditioned, or counterfactual execution-authorization file. Temporal V1's
only durable authority is one-shot, bound to `attempt_v1`, 400 updates / 16,000
presentations, with retry and resume false; its result records the attempt consumed
at update 50. Therefore the current durable boundary is document/source review and
synthetic-only work until fresh explicit development authority is granted. This
record does not retrospectively invent authority for the development outputs or
license their continuation.

Companion documents produced this session:

- `docs/lewm_go2_campaign_first_principles_review_2026-07-31.md` — review of the
  2026-07-24 → 07-31 campaign against stated aims.
- `docs/lewm_go2_action_conditioned_world_model_plan_2026-07-31.md` — phased
  plan with gates.
- `docs/lewm_go2_world_model_substrate_u700_sched3000_run_manifest_2026-07-31.json`
  — exact bounded substrate DAG; draft, not authority.
- `docs/lewm_go2_world_model_substrate_development_authority_proposal_2026-07-31.json`
  — one-attempt proposal for that DAG; explicitly withheld pending source freeze
  and issue-time binding of the independent review.
- `docs/lewm_go2_world_model_counterfactual_pilot_sizing_decision_2026-07-31.md`
  — source-derived calibration/pilot workload and the source blockers that must
  be removed before counterfactual execution can be authorized.
- `docs/lewm_go2_world_model_next_tranche_independent_source_review_2026-07-31.json`
  — independent PASS of the exact source-safe package; not authority, and still
  conditional on commit/freeze and the proposal's remaining issue gates.

---

## 1. Executive summary

The session began as a review of the campaign handoff at `0279294`, which
reported Temporal JEPA V1 as a valid negative result and recommended a
"rank-preserving V2". Five substantive findings emerged. They challenge several
mechanism-level interpretations in that handoff but do not retroactively change
the consumed attempt's valid registered failure status.

1. **The registered update-50 rank failure was an optimization transient in a
   later development replay.** The statistic bottomed at update 25, was already
   recovering at update 50, and crossed 0.25 by update 200. This shows the
   single-point stopping rule did not support mechanism-level closure; the
   original attempt nevertheless remains a valid failure under its frozen rule.
2. **A materially scaled development protocol produced favorable substrate
   diagnostics.** Persistence ratio went 4.64 → 0.93, place retrieval improved,
   and raw encoder effective rank rose 56 → 70. The protocol also changed batch,
   schedule, learning rate, and budget, so it does not isolate which change
   mattered and is not promotion evidence.
3. **Neither corridor-width conclusion survived source review.** The historical
   raster output used radii 0.12--0.27 m instead of the repository's canonical
   0.47 m yaw-invariant planning radius, substituted a largest component when
   the actual spawn was infeasible, and summarized medians as "connected". It
   therefore establishes neither that corridors are too narrow nor that they
   are adequate. The archived coverage correlations are descriptive, not a
   causal localization of the coverage ceiling.
4. **The current masked output is not directly re-entrant.** Output is 64 masked
   tokens while the predictor input requires a 256-token lattice. This blocks
   unchanged output-to-input chaining, not every possible completion/adapter
   mechanism and not a one-step action scorer; broader composability is
   undetermined and rollout is conditional on task-coupled need.
5. **The historical counterfactual result is protocol-invalid.** Its recovered
   context used a 48-frame delta rather than H6's five-tick/240-frame endpoint
   cadence, did not preserve the two executed historical actions, and evaluated
   kinematic renders as though they were physics-valid outcomes. The 1.13× and
   1.48× numbers and the overfit verdict remain audit history, not evidence.
   H6's lack of within-state action contrast is a verified data property, but
   architecture, objective, optimization, coverage, and generalization remain
   unresolved.

The next bounded world-model decision is to freeze/review the source-safe
package, then implement the missing synchronized physics-branch and WM-A.4
consumer contracts. Source-only sizing now bounds calibration at 160 branches
and the recommended pilot at 2,304 branches, but wall time and numerical
repeatability still require the calibration run. Large data generation is not
justified. The repository critical path remains the formal G2-G8 promotion
sequence and the full learned navigation architecture.

---

## 2. Claims made and withdrawn during this session

Recorded deliberately. The original three mistaken claims remain visible, and
the later goal/authority audit adds two narrowed claims and one withdrawn
critical-path claim rather than silently rewriting the history.

| claim | status | how resolved |
|---|---|---|
| The 0.25 rank threshold could be failed by an optimal predictor | **withdrawn** | a least-squares linear conditional-mean reference, fitted and evaluated on disjoint sentinel halves, retained 0.55–0.60 of target rank; this does not establish a nonlinear oracle |
| Corpus corridors are too narrow for the robot | **withdrawn; opposite also unproven** | the claim incorrectly promoted a planning abstraction to physics; later source review also invalidated the audit used to claim adequacy, so physical corridor suitability remains unresolved |
| Guard/veto conservatism is the coverage limiter | **withdrawn; cause unresolved** | one unmatched veto-off run did not improve coverage and guard rate had little marginal correlation in the archived suite; neither observation identifies the general causal limiter |
| Option C: regenerate corpus with wider corridors | **withdrawn as a recommendation** | no valid measurement currently justifies this expensive intervention; this is not evidence that no corridor change will ever be needed |
| Escalate immediately to an action-factored architecture if fidelity fails | **deferred, not resolved** | the historical fidelity/capacity protocol was invalid; a correctly bound training-fit and scene-disjoint evaluation must precede any architectural conclusion |
| Composable rollout is mandatory for the repository goal | **narrowed** | the governing objective permits an action-conditioned one-step predictor or latent rollout if it causally affects deployed scores; composability is required only for the rollout branch |
| Counterfactual data generation is the repository critical path | **withdrawn as stated** | it is a candidate next world-model diagnostic; formal G2-G8 and the full learned target/frontier/route/action stack remain the repository promotion path |
| The 94-group probe measured valid counterfactual action fidelity and capacity | **withdrawn** | source audit found wrong H6 cadence, missing executed-action provenance, kinematic rather than physics-valid targets, and no valid scene-disjoint train/eval contract; corrected source rejects these conditions |
| The geometry audit established maze connectivity and `local_composite_motifs` fragmentation | **withdrawn pending rerun** | historical radii and spawn fallback did not implement the canonical contract; current source separates spawn reachability from largest-component size and defaults to 0.47 m |

---

## 3. Measurements

### 3.1 Historical Temporal V1 finite trajectory (`scripts/dev_temporal_v1_rank_asymptote_probe.py`)

Re-ran the frozen V1 configuration with the continuation gate disabled, using
the reviewed modules unchanged. Reproduces the authorized V1 update-50 sentinel
to nine significant figures (`0.181424597` vs `0.1814245973`).

| update | loss | pred erank | pred/target ratio | gate ≥0.25 | persistence |
|---:|---:|---:|---:|:--|---:|
| 0 (random init) | — | 19.70 | 0.8574 | n/a | 4.639 |
| 25 | 0.2726 | 3.73 | **0.1641** min | fail | 1.884 |
| **50** | 0.2154 | 4.10 | 0.1814 | fail ← **authorized run terminated here** | 1.667 |
| 200 | 0.2256 | 6.54 | 0.2694 | **PASS** | 1.308 |
| 400 (cap) | 0.1625 | 9.84 | 0.3442 | **PASS** | 1.149 |

Reference values the authorized run never measured:
- persistence prediction erank 29.16 (ratio 1.020) — persistence is a *sample*,
  not a conditional mean, so it trivially matches target rank;
- fitted linear E[future | current] reference erank 17.17 (ratio **0.601**);
- fitted linear E[future | current, action] reference erank 15.77 (ratio 0.552).

The frozen `training_update_v1` hard-caps at `MAXIMUM_UPDATES_V1 = 400`, so this
ran exactly the registered budget with no modified step.

### 3.2 Campaign scale (from artifact timestamps)

| | measured |
|---|---|
| Spatial V1 ("qualified") wall-clock | **151 s** |
| Temporal V1 wall-clock | **59 s** |
| Batch | 16 presentations (spatial) / 10 sequences (temporal) |
| LR schedule | none — constant, no warmup |
| Registered Temporal V1 maximum | 4,000 H6 sequence rows / 16,000 logical RGB presentations |
| Available pool | ~1.81M H6 candidates / 55.2M frame rows / ~2.896 TB |
| Unique-candidate fraction at the registered maximum | **~0.22%** of H6 candidates |
| GPU at session start | 8 W, 3% util — idle |

### 3.3 Scaled temporal training (`scripts/dev_train_temporal_jepa_scaled.py`)

Run `rung1`: batch 256, microbatch 32, warmup 150 + cosine, LR ×4, all 16,000
bound train rows, GPU-resident pre-decoded frames. The script was configured for
3,000 updates and was manually stopped at update 700; it had no implemented
metric-based stopping rule and its final learning-rate fraction was still about
0.91. The run therefore characterizes a development trajectory rather than a
selected optimum.

| update | loss | persistence | wrong-action† | wrong-history | rank ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | — | 4.6395 | 1.0224 | 1.0029 | 0.8574 |
| 100 | 0.1729 | 1.2418 | 0.98214 | 0.9298 | 0.2839 |
| 200 | 0.1450 | 1.0120 | 0.96231 | 0.9461 | 0.3824 |
| 300 | 0.1419 | **0.9579** | 0.94502 | 0.9460 | 0.4290 |
| 700 (final) | 0.1172 | **0.9295** | 0.89632 | 0.9432 | 0.5244 |

† These wrong-action values are retained only as historical outputs. The trainer
excluded action ID 0 rather than canonical HOLD ID 6, so they are not the
registered non-HOLD metric and cannot pass a WM-0 action-control check.

Also at update 700: `current_only_ratio` 0.8279, prediction variance 0.3989,
target erank 36.20.

### 3.4 Spatial retention (`scripts/dev_eval_temporal_retention_and_rollout.py`)

First time the registered predecessor panel has been run on a temporal
checkpoint — it was scheduled for update 200 and never fired. Matched baseline
run through the identical code path rather than trusting reported values.

| metric | predecessor | temporal u600 | |
|---|---:|---:|---|
| place R@5 | 0.4238 | 0.4406 | better |
| chance multiple | 4.690 | **4.887** | better |
| place-key erank | 55.49 | 57.45 | better |
| raw encoder erank | 56.28 | **70.28** | better |
| spatial diversity | 0.7545 | 0.7613 | better |
| ctrl wrong-target | 0.2962 | 0.4240 | **worse** |
| ctrl wrong-context | 0.2967 | 0.4235 | **worse** |
| ctrl position-mean | 0.5279 | 0.8004 | **worse** |

Interpretation (**interpretation, not measurement**): the place/rank diagnostics
improved while all three registered spatial controls worsened. Both the encoder
and predictor parameter groups were trainable during temporal optimization, so
the panel cannot attribute the degradation to either component or their
compatibility. **Unverified — run a component-swap decomposition, re-measure the
exact u700 checkpoint, and audit the physical interface before relying on this
checkpoint.**

### 3.5 Composability (structural, not measured)

`predict_from_encoded_history` requires `(B,S,256,192)` — the full spatial
lattice (`lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py:237`). The
prediction is `(B,64,192)` — the masked target subset. **Output space ≠ input
space; the output cannot be re-fed unchanged.** Mask completion, lattice
assembly, and adapters were not evaluated, so broader composability is
undetermined. Separately, H6 exposes `rgb[0:4]` with positions 4/5/6 forbidden,
so no K-step ground truth exists on this slice.

### 3.6 Historical corpus-geometry output (`scripts/audit_go2_family_clearance_navigability.py`)

**Invalid for geometry conclusions after source audit.** The table below is
retained to explain how the earlier reversal was reached. The run swept radii
0.12--0.27 m rather than the canonical 0.47 m yaw-invariant planning radius,
replaced an infeasible actual-spawn component with the largest component, and
used a median connected fraction that could hide failed scenes. The proxy also
omits orientation, turning sweeps, dynamics, contacts, and rough-terrain
feasibility. No corrected corpus run exists under the hardened source.

All 1,450 scenes, 2 cm raster, Euclidean clearance transform, swept over robot
half-extents 0.12 / 0.20 / 0.24 / 0.27 m. At the runtime inflation actually used
(`--inflation-m 0.12`):

| family | n | fits% | connected | medclr (m) | ray (m) |
|---|---:|---:|---:|---:|---:|
| open_obstacle_field | 130 | 94.4% | 100% | 0.849 | 3.50 |
| rough_local_dynamics | 79 | 93.5% | 100% | 0.780 | 3.22 |
| loop_alias_stress | 175 | 83.5% | 100% | 0.300 | 1.57 |
| medium_enclosed_maze | 356 | 79.4% | 100% | 0.260 | 1.25 |
| large_enclosed_maze | 239 | 78.4% | 100% | 0.240 | 1.22 |
| small_enclosed_maze | 196 | 76.3% | 100% | 0.220 | 1.13 |
| **local_composite_motifs** | 196 | 91.7% | **9.2%** | 0.720 | 0.96 |

The old script happened to match one 2026-07-02 tight-fraction statistic
(0.369 vs 0.36); that scalar agreement did not validate its footprint,
spawn-reachability, or family-summary contract.

These values support no present physical-connectivity, family-exclusion, or
dataset-regeneration decision. The ray values and low historical component
fractions are hypotheses to remeasure under the corrected contract.

### 3.7 Descriptive coverage associations (115 archived closed-loop runs)

These are marginal associations across heterogeneous historical run variants,
not a matched causal study. They cannot rank perception, guard behavior, travel
throughput, budget, or policy as the general coverage limiter.

Suite: `v217_replacement_cap_20260701`.

| correlation with cells visited | r |
|---|---:|
| distance travelled | **+0.378** |
| escape blocks | +0.346 |
| tick budget | +0.342 |
| guard block rate | +0.059 |
| contact-like stalls | −0.049 |

Available veto-off control: `v228_progressoutcome` / `v229_outveto0` ran with
`blocked_forward_requests = 0` (guard fully off), travelled 34.6 m, visited
**6 cells** — fewer than guarded `v225` at 6.6 m and 14 cells.

The archived aggregate reports **~0.71 m of travel per newly visited cell**; median
run travels 9.1 m / 12 cells, best 55.0 m / 69 cells. Mean forward-execution
displacement ~0.11 m, only ~20% of ticks produce a forward execution. The
privileged-DFS control that anchored the 2026-07-02 diagnosis travelled only
11–14 m, so it does not isolate an observability ceiling from a travel-budget
ceiling.

### 3.8 Historical counterfactual action fidelity (`scripts/dev_probe_counterfactual_action_fidelity.py`)

**Protocol-invalid after source audit.** The results below are retained as an
error record, not as a measurement of H6 counterfactual fidelity. The historical
probe inferred context at a 48-frame delta instead of the H6 endpoint cadence of
240 frames, failed to bind the two actual historical actions, accepted embedded
provenance without an H6 identity witness, mixed train/evaluation roles, and
treated kinematic renders as claim-bearing physical outcomes. Corrected source
now fails closed on those conditions; it has not been run on runtime data.

Data: `.generated/jepa_counterfactual/phase2b_{eval,train}_8scene_spatial_v1.jsonl`
— same start state, multiple candidate primitive sequences, each with rendered
futures. 94 groups with ≥3 distinct first actions, 8 families, chance 0.2302.

The historical script reported 3-frame context at **stride 48**. That inference
was wrong: the applicable H6 endpoints are five simulator ticks apart, or 240
interleaved frame indices, and require the two executed actions as provenance.

| arm | fidelity ×chance | discrimination ×chance | diag / off-diag |
|---|---:|---:|---|
| **temporal u700** | **1.129** | **1.477** | 0.1664 / 0.1725 |
| action-blind control | 1.000 | 1.000 | identical, spread 0.0 |
| predecessor | 1.033 | 1.013 | 0.5648 / 0.5655 |

Definitions: `energy[i,j]` = error of prediction conditioned on action `aᵢ`
against the true rendered outcome of `aⱼ`. Fidelity = `argmin_j energy[i,j] == i`.
Discrimination = `argmin_i energy[i,j] == j`.

**Diagnostic A — outcome separation.** Fidelity is flat across a 6.5× range of
true-outcome separation while discrimination rises monotonically:

| separation quartile | n | tgt_sep | fidelity ×chance | discrim ×chance |
|---|---:|---:|---:|---:|
| Q1 | 23 | 0.041 | 1.206 | 1.118 |
| Q2 | 24 | 0.115 | 1.193 | 1.493 |
| Q3 | 24 | 0.183 | 0.985 | 1.524 |
| Q4 | 23 | 0.265 | 1.135 | 1.788 |

Because the context and target protocol was invalid, these quartiles cannot
explain the apparent fidelity result or weaken any horizon/signal hypothesis.

**Diagnostic B — capacity** (`scripts/dev_probe_counterfactual_overfit_capacity.py`).
Trained the same architecture directly on the 94 groups with a within-state
contrastive loss (temperature 0.05 + JEPA MSE), 40 epochs, deliberately
overfitting.

| epoch | fidelity | ×chance | loss |
|---:|---:|---:|---:|
| 0 | 0.2471 | 1.073 | 118.5 |
| 20 | 0.3566 | 1.549 | 68.9 |
| 40 | **0.5338** | **2.319** | 53.1 |

The historical script emitted verdict `REPRESENTATIONAL_CAPACITY_PRESENT`, best
2.43×, still rising at epoch 40. That verdict is withdrawn: it shows numerical
fit to malformed training groups, not validated action-conditional capacity.

**What survives**: each H6 state is paired with exactly one action, so scaling
the same on-policy role creates no within-state action contrast. That data fact
motivates a correctly bound pilot; it does not localize the historical failure.
The malformed 94-group fit does not establish action-conditional capacity.

**Limit**: beyond deliberate overfitting and absent scene-disjoint
generalization, the historical protocol itself was invalid. Architecture,
objective, optimizer, data coverage, and deployed-task relevance are all
undetermined.

---

## 4. Infrastructure built and source-review status

The seven scripts below were hardened after the historical outputs in §3 were
created. The current code has only been exercised on synthetic fixtures; none
of the old runtime outputs satisfies its current provenance and immutability
contracts.

- **`scripts/dev_pack_h6_temporal_frames.py`** — builds a bound uint8 frame pack
  with H6 index identity, per-source byte/hash receipts, ordered raw-RGB identity,
  path-containment checks, immutable output, and sampled bit-exact verification.
- **`scripts/dev_train_temporal_jepa_scaled.py`** — batched, scheduled
  development trainer; the recorded run was configured for 3,000 updates and
  manually stopped at 700. Current source now separates `--updates` from
  `--schedule-updates`, so a complete bounded u700 run can retain the historical
  3,000-update cosine trajectory instead of silently moving its endpoint. Loss
  is asserted equal to the frozen
  `0.5·(pred−target)².sum(-1).mean()` on every microbatch. The update-0 panel
  in the historical run did **not** reproduce the registered wrong-action metric
  because that source used the wrong HOLD ID. Current source uses canonical ID
  6 and has synthetic coverage; runtime panel parity remains unmeasured.
- **`scripts/dev_eval_temporal_retention_and_rollout.py`** — registered spatial
  retention panel against an exact immutable checkpoint identity; reports only
  direct plug-compatibility, leaving broader composability undetermined.
- **`scripts/dev_probe_counterfactual_action_fidelity.py`** — the WM-A probe with
  bound five-tick H6 context/action provenance, scene-disjoint roles,
  physics-valid-by-default targets, correct macro/micro chance, hardest-wrong
  margins, controls, scene-clustered uncertainty, and code/source bindings. It
  is **not pilot-ready**: input JSONLs and the legacy RGB root are hard-coded,
  and it has no physical task-regret consumer. Only its output/checkpoint paths
  are development-confined.
- **`scripts/dev_probe_counterfactual_overfit_capacity.py`** — separates training
  capacity from scene-disjoint evaluation and emits coherent frozen snapshots.
- **`scripts/dev_temporal_v1_rank_asymptote_probe.py`** — a bounded finite
  trajectory with a restricted, explicitly non-oracle linear reference.
- **`scripts/audit_go2_family_clearance_navigability.py`** — a canonical 0.47 m
  raster-disc proxy that retains actual-spawn failures separately from the
  largest component, binds and rechecks its corpus/manifests/source, confines
  CLI input/output roots, and explicitly disclaims physical navigability.

The legacy frame pack, mutable `latest.pt`, retention output, rank trace,
counterfactual reports, and geometry reports have not been migrated or
revalidated. Current tools reject the legacy pack/checkpoint/protocol where
applicable. Any corrected runtime result requires fresh explicit authority.

---

## 5. Plan status

Per `docs/lewm_go2_action_conditioned_world_model_plan_2026-07-31.md`, the target
is a world-model subsystem within the broader fully learned navigation goal:

| ID | property | role | status |
|---|---|---|---|
| WM-A | scene-disjoint action utility for untaken actions | necessary | unmeasured; historical 1.13× probe is protocol-invalid |
| WM-C | output re-entry and multi-step rollout | conditional on deployment path | direct re-entry fails; broader composability unmeasured |
| WM-S | shared spatial/physical substrate retention | necessary | place retrieval favorable; spatial controls worse; physical retention unmeasured |
| WM-D | deployed causal presence | necessary | not integrated |

**A historical scaled run completed, but current WM-0 verification is absent.** At update 700,
persistence was <1.0, wrong-history was <0.95, and rank was ≥0.25 and rising.
The wrong-action calculation used the wrong HOLD ID. Place retrieval was 4.887×
chance at u600 versus the matched predecessor's 4.690×, while all three spatial
controls worsened; no u700 retention panel exists. Because only a mutable legacy
alias was retained, u700 is not an identifiable candidate. A fresh authorized,
immutable run (or an explicitly authorized identity-bound migration) must pass
the corrected panel before any checkpoint can be called a known-good development
substrate, and even that would not physically requalify it.

WM-A counterfactual evaluation correctly precedes optional WM-C/WM-R rollout
work because one-step action utility is independently testable. A full-state
head should be built only if task-coupled integration localizes a remaining
deficit to multi-step sequence evaluation.

---

## 6. Next bounded decision and formal path

The next world-model decision is **not** an automatic large render job:

1. Commit and freeze the exact source-safe tranche, which has now passed an
   independent source/custody review. If separately authorized and bound to that
   frozen identity, the fixed u700/schedule-3000 substrate DAG measures the
   complete retention trajectory; it does not select or physically qualify a
   checkpoint.
2. Implement and synthetically test an online synchronized Genesis branch
   collector, a manifest/hash-bound pilot consumer, and the missing WM-A.4
   physical action-ranking/regret evaluator. Current pose-reset replay is not a
   matched H6 state and the legacy renderer is kinematic.
3. Under a separate calibration-only authority, run at most eight excluded
   scenes, 16 states, and 160 branches. Freeze measured state-repeatability,
   validity-yield, bytes, VRAM, and per-stage/end-to-end timing before choosing
   `STOP`, `LOW`, or `RECOMMENDED`.
4. If separately authorized, train the unchanged architecture on the frozen
   scene-disjoint pilot with
   JEPA MSE plus within-state contrast and evaluate WM-A on withheld development
   scenes, including privileged-planner action ranking/regret.
5. Distinguish training-fit failure, generalization failure, proxy/task
   decoupling, and genuine scene-disjoint action utility before choosing scale
   or a new architecture.
6. Integrate a one-step scorer first. Build a composable head and K-step role
   only if task evidence localizes a sequence-horizon deficit.

No item above is authorized by this document. Exact source-derived workload is
now scoped, but wall-clock throughput, restore/co-branch repeatability, valid
group yield, and GPU memory are unmeasured. The calibration cannot be authorized
until its missing executable source exists and is reviewed.

At repository scope, formal promotion remains G2 traversability perception, G3
persistent physical/configuration belief and coverage, G4 learned
frontier/viewpoint value, G5 reversible per-color target conversion, G6 full
development, G7 robustness, and only then a fresh eligible one-shot G8. `WM-*`
diagnostics cannot replace or authorize those gates.

---

## 7. Standing recommendations from the review

- **Do not** run the rank-preserving V2 as specified: it targets an early
  transient, its remedies may trade total-error accuracy for rank, and it is not
  coupled to action utility.
- **Do not** re-render the corpus merely to widen corridors: neither the original
  too-narrow claim nor its attempted reversal has valid canonical-footprint
  evidence. This does not prohibit a separately justified dataset change after
  a corrected audit.
- Separate exploratory development evidence from formal promotion evidence, but
  do not treat `.generated/dev/**` or "non-citable" as self-authorization. Use a
  lighter development process only when explicitly authorized; preserve the
  repository's formal G2-G8 custody.
- **No single-point gates** — terminate only on below-threshold *and* not
  improving across ≥2 observations.
- Treat as **unresolved, not closed**: the recurrent-H4 persistence-residual
  "correlated low-rank shortcut" and delay-line V4 "effectively two dimensions
  of 64" — both diagnosed from single-point low-rank observations at the same
  scale that produced V1's false negative.
- Do not exclude or regenerate `local_composite_motifs` from the legacy geometry
  output; require a corrected, explicitly authorized audit first.
- Prefer a one-step causal scorer; make composability and rollout conditional on
  measured task need.

---

## 8. Artifacts (all untracked)

Documents:
```
docs/lewm_go2_campaign_first_principles_review_2026-07-31.md
docs/lewm_go2_action_conditioned_world_model_plan_2026-07-31.md
docs/lewm_go2_world_model_session_handoff_2026-07-31.md   (this file)
docs/lewm_go2_world_model_substrate_u700_sched3000_run_manifest_2026-07-31.json
docs/lewm_go2_world_model_substrate_development_authority_proposal_2026-07-31.json
docs/lewm_go2_world_model_counterfactual_pilot_sizing_decision_2026-07-31.md
docs/lewm_go2_world_model_next_tranche_independent_source_review_2026-07-31.json
```

Scripts:
```
scripts/audit_go2_family_clearance_navigability.py
scripts/dev_pack_h6_temporal_frames.py
scripts/dev_train_temporal_jepa_scaled.py
scripts/dev_eval_temporal_retention_and_rollout.py
scripts/dev_temporal_v1_rank_asymptote_probe.py
scripts/dev_probe_counterfactual_action_fidelity.py
scripts/dev_probe_counterfactual_overfit_capacity.py
```

Tests:
```
lewm/tests/test_audit_go2_family_clearance_navigability.py
lewm/tests/test_july31_h6_temporal_pack_provenance.py
lewm/tests/test_july31_temporal_diagnostic_contracts.py
lewm/tests/test_dev_counterfactual_diagnostic_protocol.py
```

Historical pre-hardening development outputs (`.generated/dev/`, reported 2.7
GB, none citable and none validated by the current source):
```
h6_temporal_pack/            2.6G  legacy frame pack + manifest
temporal_jepa_scaled/         50M  legacy trace, mutable latest.pt, retention panels
family_clearance_audit/      7.8M  legacy noncanonical geometry sweeps
temporal_v1_rank_asymptote/   20K  legacy finite-trajectory trace
counterfactual/               20K  legacy protocol-invalid probes
```

The `g2_probe*` strings above are legacy internal filenames. They are WM-A
development diagnostics and are not formal G2 evidence.

Recorded checkpoint alias: `.generated/dev/temporal_jepa_scaled/rung1/latest.pt`
(reported update 700). It is mutable, lacks a retained update-specific binding in
this handoff, and must not be opened or treated as the selected checkpoint without
fresh authority and identity verification. Its historical metrics are non-citable,
its u600 spatial controls worsened, no u700 retention panel exists, and it has no
physical requalification. It does not carry forward, and no full-state successor
is selected.

No sealed material opened. No sealed attempt consumed. V4 held-out role
untouched and remains permanently ineligible.
