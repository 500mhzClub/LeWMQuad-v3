# Plan: action-conditioned world-model subsystem with spatial understanding

Date: 2026-07-31
Status: **development proposal only**. This document is not an authority, a
preregistration, or promotion evidence. It grants no data generation, artifact
mutation, GPU, training, evaluation, G2-G8, held-out, or sealed authority.
Every operation still requires whatever explicit authority governs that operation.

The names `G2` through `G8` are reserved for the repository's existing formal
promotion gates. This document uses only `WM-*` names for world-model subsystem
diagnostics and decisions.

No July 31 world-model, action-conditioned, or counterfactual execution authority
is present in the durable repository record. Temporal V1's one-shot authority is
consumed and forbids retry/resume. Until fresh explicit development authority is
granted, the actionable scope of this plan is document/source review and
synthetic-only work; it does not license continuation of existing development
checkpoints or generation of new runtime artifacts.

Companion: `docs/lewm_go2_campaign_first_principles_review_2026-07-31.md`.

---

## 0. Scope and target

The repository goal is not merely "a JEPA that trains" or even a standalone
world model. The final system must use deployment-available observations,
maintain persistent and reversible physical and per-color beliefs, use an
action-conditioned JEPA predictor **or** latent rollout causally in deployed
candidate/action scores, and learn target, frontier/viewpoint, route/subgoal,
and ordinary-motion selection. Formal G2-G8 promotion remains unchanged.

This plan covers only the world-model subsystem. Its tests are:

| ID | Subsystem property | Role | Current evidence |
|---|---|---|---|
| WM-A | **Action utility** — correct, scene-disjoint predictions or scores for untaken actions improve action ranking/regret | **necessary** for the deployed predictor claim | unmeasured; historical probe invalidated by protocol audit |
| WM-C | **Composability** — predictions can be re-entered to evaluate action sequences | **conditional**: required only if the chosen deployed interface uses latent rollout | direct output-to-input plug fails; broader composability unmeasured |
| WM-S | **Shared-substrate retention** — temporal training does not destroy physical/spatial evidence needed downstream | **necessary**, with full physical requalification before formal G2 | place retrieval retained; spatial controls worsened; physical retention unmeasured |
| WM-D | **Deployed causal presence** — disabling or shuffling the predictor changes candidate/action scores and behavior | **necessary** | not integrated |

Prediction/target rank, total latent MSE versus persistence, place retrieval, and
latent topology are useful diagnostics. They are not substitutes for task-coupled
action utility, physical requalification, or closed-loop success. In particular,
beating persistence is not by itself a repository-goal requirement, and a
composable rollout is not mandatory if a one-step predictor causally supports a
successful receding-horizon controller.

---

## 1. Current development evidence

Historical outputs from the scaled development run (`--tag rung1`, batch 256,
configured for 3,000 updates and manually stopped at 700):

| update | loss | persistence | wrong-action† | wrong-history | rank ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | — | 4.6395 | 1.0224 | 1.0029 | 0.8574 |
| 100 | 0.1729 | 1.2418 | 0.98214 | 0.9298 | 0.2839 |
| 200 | 0.1450 | **1.0120** | 0.96231 | 0.9461 | 0.3824 |
| 300 | 0.1419 | **0.9579** | 0.94502 | 0.9460 | 0.4290 |
| 700 | 0.1172 | **0.9295** | 0.89632 | 0.9432 | 0.5244 |

† Historical output only: the scaled trainer excluded action ID 0 rather than
canonical HOLD ID 6. This column is not the registered non-HOLD metric and must
be recomputed before it supports any action-control conclusion.

The table is an audit record from a legacy mutable checkpoint/output contract;
the current hardened tools do not accept it as a selected checkpoint and no
current-schema runtime output exists. The authorized Temporal V1 attempt validly terminated at its preregistered
update-50 continuation gate. The later non-citable development runs show that
the rank statistic recovers under continued training and that a materially
scaled protocol can beat persistence. They challenge the earlier mechanism-level
closure; they do not retroactively pass the consumed attempt or establish that
architecture, supervision, optimization, and generalization are all adequate.

Two structural facts were found while building the evaluation:

- **Direct self-feeding path blocked**: `predict_from_encoded_history` requires `(B,S,256,192)` (full
  spatial lattice); the prediction is `(B,64,192)` (masked subset). Output space
  ≠ input space, so it cannot be re-fed unchanged. Completion/adapter paths were
  not evaluated. Confirmed at
  `lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py:237`.
- **No K-step ground truth** on the current slice: H6 exposes `rgb[0:4]` with
  positions 4/5/6 forbidden by the V1 contract.

Neither fact blocks a one-step candidate/action scorer. They matter only if
task-coupled evaluation selects the rollout path.

---

## 2. Standing rules

1. **No self-authorization.** Development outputs under `.generated/dev/**` are
   non-citable, but that label grants no permission to create them. Rendering,
   artifact mutation, GPU work, training, and evaluation require explicit
   in-scope authority. Formal G2-G8 retain their existing custody and one-shot
   requirements.
2. **No single-point gates.** A statistic terminates a run only when it is below
   threshold **and** has not improved across ≥2 consecutive observations. This
   is the rule whose absence made Temporal V1's early gate scientifically
   misleading even though the registered attempt status remains valid.
3. **Every threshold anchored to a measured reference** — an oracle, the
   predecessor, or chance — never a round number. Where no reference exists,
   measure it first.
4. **Bounded pilot before scale.** First establish scene-disjoint action utility
   and task regret on the smallest adequate pilot. Scale only after the metric
   is coupled to the deployed goal and the learning curve remains data-limited.
5. **Retention is two-level.** Re-run place/spatial diagnostics during development,
   but treat them only as early warnings. Any encoder update requires complete
   physical selection/calibration requalification before formal G2.
6. **Namespace custody.** `WM-*` diagnostics cannot pass, replace, or authorize
   any formal G2-G8 gate.

---

## 3. World-model workstreams and decisions

### WM-0 — Substrate characterization *(historical run; current verification absent)*

Rung 1 characterizes whether the encoder + patch-GRU substrate can survive a
larger temporal-training protocol. These are diagnostics, not qualification.

| gate | metric | threshold | basis |
|---|---|---|---|
| WM-0.1 | persistence ratio | report; <1.0 is favorable, not a promotion gate | copy-frame diagnostic |
| WM-0.2 | corrected non-HOLD wrong-action ratio | trend-down, <0.95 | pending; historical calculation used the wrong HOLD ID |
| WM-0.3 | wrong-history ratio | <0.95 | causal diagnostic |
| WM-0.4 | prediction rank ratio | ≥0.25 and rising | measured linear reference ~0.60 |
| WM-S.place | place retrieval | ≥0.90 × predecessor's 4.690× matched-code-path value | early retention warning |

**Result**: the historical persistence, wrong-history, and rank values were
favorable diagnostics. Wrong-action is invalid, the retention panel was run at
u600 rather than u700, and all three spatial controls worsened. The only
recorded u700 alias is mutable and lacks a retained update-specific binding.
WM-0 is not verified; no checkpoint is selected, physically requalified, or
promotable.

**Decision rule**: if causal controls remain flat and retention degrades across
multiple observations, test a frozen-encoder temporal head. Do not reject or
promote a mechanism from persistence ratio alone.

---

### WM-C — Composable full-state head *(conditional; currently deferred)*

Build this only if WM-A and WM-I show that one-step action scoring is useful but
that sequence horizon is the remaining task-level limitation. If selected,
change exactly one thing: emit the **full 256-token next-state encoding** instead
of 64 masked queries. Keep encoder, GRU, action/time embeddings, EMA target,
data, and schedule fixed within the comparison.

Rationale: a masked subset cannot be re-fed unchanged as history. That is a
structural defect for the direct self-feeding rollout interface, not proof that
all adapters are impossible and not a defect for a one-step action scorer.

| gate | metric | threshold | basis |
|---|---|---|---|
| WM-C.1 | composability | `predict(predict(x))` executes; output shape == encoder output shape | structural |
| WM-C.2 | one-step action utility | no regression from WM-A best | do not buy rollout by losing the action signal |
| WM-C.3 | masked-subset parity | energy on the 64 masked positions no worse than matched predecessor | like-for-like diagnostic |
| WM-S.place | place retrieval | ≥0.90 × matched predecessor | early retention warning |

**Risk**: predicting 256 tokens is a harder objective and may be dominated by
the easy (visible) tokens. Mitigation: report energy separately on masked vs
visible positions; WM-C.2 is the task-relevant comparison.

**Decision rule**: if a bounded scale ladder cannot retain WM-A action utility,
stop the full-state branch. A pooled state plus decoder is another mechanism and
requires its own comparison; it is not an automatic fallback.

---

### WM-A — Counterfactual action utility *(next bounded decision)*

Evaluate this **before** optional rollout work. Counterfactual fidelity is
testable at one step, but latent fidelity alone is not enough: also measure
privileged-planner action ranking and regret on scene-disjoint development data.

**WM-A0. Source/synthetic-only sizing and source readiness.** Source audit now
shows that the existing pieces cannot produce or consume the required pilot.
The render planner is kinematic; pose-reset physics replay discards matched
velocity, joint, PPO-latency, and previous-command state and emits no RGB; the
probe hard-codes legacy JSONLs/RGB; and no source computes WM-A.4 task regret.
Do not infer physics validity by composing those tools.

The source-only sizing contract is
`docs/lewm_go2_world_model_counterfactual_pilot_sizing_decision_2026-07-31.md`.
It caps a future calibration at eight excluded scenes, 16 states, and 160
branches, and sizes a recommended pilot at 32 disjoint scenes, 256 states, and
2,304 one-step branches. These are workload counts, not authority or measured
wall time. First implement and review a synchronized physical branch collector,
bound consumer, and task-regret evaluator. Then obtain explicit calibration
authority before opening a runtime input or starting Genesis/render/GPU work.

**WM-A1. Pilot diagnostics.**

| gate | metric | threshold | basis |
|---|---|---|---|
| WM-A.1 | executed vs **hardest** wrong action | report family/scene distribution; freeze a referenced threshold before confirmatory use | historical failure mode |
| WM-A.2 | counterfactual energy | predicted-for-`a′` beats predicted-for-`a` on `a′`'s true outcome above chance with scene-clustered uncertainty | fidelity, not discrimination |
| WM-A.3 | HOLD non-dominance | per-action macro result and class distribution | documented HOLD-collapse failure |
| WM-A.4 | oracle action ranking/regret | improve over action-blind/current-state controls on withheld development scenes | task-coupled reference |
| WM-S | substrate retention | place diagnostics plus complete physical requalification before formal G2 | downstream contract |

The historical 94-group overfit result does not establish even a clean capacity
bound: source review found incorrect H6 cadence/actions, unbound embedded
provenance, kinematic targets, and an invalid role contract. A corrected pilot
must distinguish:

- cannot fit training groups: architecture/loss/optimization remains suspect;
- fits training but not withheld scenes: generalization, inductive bias, or data
  coverage remains suspect; do not call the cause "supervision only";
- passes latent fidelity but not action regret: the proxy is task-decoupled;
- improves withheld-scene action regret without substrate regression: justify a
  larger, separately authorized data/training tranche.

An action-factored predictor is considered only after this diagnosis and as a
new mechanism, not as an automatic escalation.

---

### WM-R — Multi-step rollout *(conditional on the WM-I decision)*

If selected, this requires a new, separately authorized K-step role. H6 exposes
`rgb[0:4]`; positions 4/5/6 remain forbidden by the V1 contract and must not be
opened through that role. A longer-horizon index must be independently defined
and authorized.

Train with rollout warmup / scheduled sampling. Precedent exists in this repo:
the seq11 finding that *warmup*, not the loss weight, is what kills the
zero-action shortcut (winner: `lam0.25 warm2 e2`).

| gate | metric | threshold | basis |
|---|---|---|---|
| WM-R.1 | free-running K=4 action regret | improves over one-step and persistence controls | task-coupled horizon test |
| WM-R.2 | divergence | latent norm and effective rank bounded across K | prior free-running failure mode |
| WM-R.3 | action-sequence separation | rollouts under different action sequences separate above noise | planning discriminability |
| WM-R.4 | teacher-forced vs free gap | report against prior reference; freeze threshold before use | exposure-bias diagnostic |
| WM-S | substrate retention | no regression plus later physical requalification | downstream contract |

---

### WM-T — Place and topology probes *(optional diagnostics)*

Place recognition and topology may explain failures or support a rollout
manifold check, but they are not formal promotion gates and do not replace
physical, target-belief, or navigation evidence.

| gate | metric | threshold | basis |
|---|---|---|---|
| WM-T.1 | topology | latent k-NN respects `graph_edges` adjacency above chance | corpus `topology.json` |
| WM-T.2 | loop-alias separation | aliased corridors separable with history | precedent AUC 0.81→0.86 |
| WM-T.3 | rollout manifold | if WM-R is selected, predicted latents retrieve plausible places | hallucination diagnostic |
| WM-T.4 | metric honesty | report ρ(latent distance, spatial distance) | reported, never gated |

WM-T.4 is deliberately not a gate. Prior development evidence suggests that
recognition-like rather than metric latents can support navigation, but that
evidence is not formal promotion evidence.

---

### WM-I — Development integration and deployment-path decision

First integrate a one-step action scorer into the existing development seams
(`lewm/planning/`, `benchmark_topo_nav_e2e.py`,
`benchmark_go2_memory_closed_loop.py`). This is a diagnostic harness only. The
existing classical target/frontier/route/action choices are not a promotable
fully learned controller, and inserting a predictor does not change that.

| gate | metric | threshold | basis |
|---|---|---|---|
| WM-I.1 | causal presence | predictor-off and action-shuffle change candidate/action scores and behavior | binding architecture requirement |
| WM-I.2 | action regret and navigation | improve over action-blind/current-state controls and compare with `seq4_e9 + plan_cost` on scene-disjoint development scenes | task-coupled evidence |
| WM-I.3 | matched no-JEPA arm | paired treatment comparison under the same selection rule | required for a JEPA causal claim |
| WM-I.4 | safety | no development safety regression | constraint, not substitute objective |
| WM-I.5 | architecture conformance | learned target, frontier/viewpoint, route/subgoal, and ordinary-motion choices before promotion | governing correction |

Coverage note: distance has the largest of five reported marginal correlations
with coverage in a heterogeneous historical suite; that is not causal
localization. Treat tick budget, forward displacement, escape/yaw churn, guard
behavior, map resolution, perception, and geometry as explicit matched harness
factors between treatment arms.

**Decision**: if the one-step scorer improves task evidence and is causally used,
retain it and skip WM-C/WM-R. Enter WM-C/WM-R only if evidence localizes the
remaining deficit to multi-step sequence evaluation.

---

### Formal promotion boundary — existing G2-G8 only

Passing every `WM-*` diagnostic still grants no promotion or execution authority.
Before formal G2, finalize the world-model interface, run the matched JEPA/no-JEPA
development treatment, requalify physical perception after every encoder change,
and freeze the complete candidate under the repository's reviewed process.

The existing promotion meanings remain:

1. **G2 — traversability perception** on its eligible untouched role. A pass
   freezes the encoder, physical head, checkpoint, calibration, and thresholds.
2. **G3 — persistent physical/configuration belief and coverage**, including
   deployment-valid pose, cold-start, exact-path, visibility, and causal traces.
3. **G4 — learned map-conditioned frontier/viewpoint value** over matched
   reachable candidates.
4. **G5 — reversible multimodal per-color target belief and physical claim
   conversion**.
5. **G6 — full development simulation**, including architecture conformance,
   task completion, safety, stratum reporting, and the fixed matched no-JEPA arm.
6. **G7 — deployment robustness**, locomotion/odometry/noise, physical smoke,
   and action-source tracing.
7. **G8 — one frozen final evaluation** using a fresh eligible role under
   independent custody. No eligible G8 role currently exists, and this plan does
   not authorize creating one.

---

## 4. Risk register (from the campaign's own failures)

| risk | signature | mitigation |
|---|---|---|
| Low-rank shortcut misdiagnosis | rank falls early while loss falls | trend-based gates; oracle-anchored thresholds |
| Action shortcut / HOLD dominance | action NLL → log(9), HOLD class dominates | WM-A.3; action-shuffle and per-action controls |
| Frozen channel beats learned head | a learned head underperforms a fixed one | always report the fixed-channel baseline (3 prior instances) |
| Temporal objective damages downstream evidence | prediction improves while spatial/physical controls regress | WM-S diagnostics plus complete pre-G2 physical requalification |
| Malformed training fit mistaken for capacity/generalization | legacy overfit fidelity rises on 94 protocol-invalid groups | bound H6 provenance, physics-valid outcomes, scene/state-disjoint pilot, and clustered uncertainty |
| Proxy/aim decoupling | latent fidelity improves but action regret does not | task-coupled WM-A.4 and WM-I.2 |
| Undertrained verdicts | mechanism closed from an unstable curve | trend observations and a referenced bounded scale ladder |
| Development label mistaken for authority | `.generated/dev/**` operation proceeds without a governing grant | explicit operation authority; non-citable status is not permission |

---

## 5. Decision order and cost boundary

| order | decision | evidence | next branch |
|---:|---|---|---|
| 0 | freeze/review the source-safe tranche; optionally reconcile retention under the separate bounded proposal | exact u700/schedule-3000 manifest, full fixed retention trajectory, no automatic checkpoint selection | retain only as diagnostic substrate evidence or stop |
| 1 | implement the missing physics-branch, bound-input, and WM-A.4 source contracts | synthetic state/provenance/receipt tests and independent source review | stop or request calibration authority |
| 2 | calibrate, not pilot | at most 8 excluded scenes, 16 states, 160 branches; repeatability, yield, bytes, VRAM and per-stage wall time | `STOP`, `LOW`, or freeze a recommended-pilot contract |
| 3 | test WM-A on a separately authorized scene-disjoint pilot | train fit, withheld-scene fidelity, paired action regret, controls, WM-S | stop, revise mechanism, or justify scale |
| 4 | integrate the one-step scorer | WM-I causal/action/navigation evidence | keep one-step or localize a horizon deficit |
| 5 | optional rollout branch | WM-C and WM-R | abandon rollout or retain it |
| 6 | construct the pre-G2 candidate | matched no-JEPA, physical requalification, source/checkpoint freeze | enter the existing formal promotion process |

Branch counts, simulated work, and raw RGB bytes are now source-derived. Actual
end-to-end/render/GPU wall time, VRAM, validity yield, and numerical repeatability
remain calibration outputs. No estimate or table in this document authorizes
launch, and no large generation job is the default next step.

## 6. What we do not do

- No rank-preserving V2 (treats a transient as a defect; §2.5 of the review).
- No corpus re-render merely to widen corridors: neither the original
  too-narrow claim nor its historical reversal has valid canonical-footprint
  evidence. This does not prohibit a separately justified and authorized
  dataset change after a corrected audit.
- No large counterfactual generation before sizing, explicit authority, and a
  bounded scene-disjoint pilot.
- No composable head or rollout solely to satisfy a definition; task evidence
  must select that interface.
- No `WM-*` result represented as formal G2-G8 evidence or authority.
- No direct transition from WM-I to G8; formal G2 through G7 remain mandatory.
- No re-litigation of metric latent structure; WM-T.4 reports but never gates.
