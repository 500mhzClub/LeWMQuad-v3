# Preregistration: observability-ceiling assay V1

Date: 2026-08-05
Branch: `jepa-spatial-world-model-nav`
Repository HEAD at preregistration: `4adcbdad16baa81c93112e9f0f4a6aa643008fad`
Attempt identity: `go2_observability_ceiling_assay_v1_attempt_v1`

Status: **development-tier, evaluation-only, non-confirmatory achievability
assay.** It is not a world-model, dynamics, prediction, navigation, promotion,
safety, or held-out experiment. No arm produced here is eligible for closed-loop
use, planner integration, or any navigation claim. It opens no untouched,
sealed, held-out, or V4 material.

This document is frozen before any model is fit and before any evaluation score
is observed. Sections 5 and 6 are the registered decision rule.

---

## 1. Question

On the exact CPU-flat V3 64-scene panel, what is the lowest scene-disjoint
normalized rank regret achievable by a readout of the available observations,
when prediction error is removed by supplying **actual** successor observations
and readout capacity is not artificially constrained?

## 2. Why this assay, and why now

Handoff §11 and §13 both preregistered an absolute gate of
`normalized rank regret <= 0.13` and both failed it. The nearest approach by any
arm ever measured is `0.14896763392857143` — the *retained physical
predecessor* control, which is privileged non-observational information. The one
prior attempt at an achievability ceiling, the frozen dense V-JEPA 2.1
physical-interface ceiling, deliberately constrained its interface to a
train-only `K=8` PCA and a 245-parameter shared spatial readout. It therefore
measured an **interface** ceiling, not an **information** ceiling.

Consequently the question "can any readout of these observations reach `0.13`?"
is open, and the next mechanism-changing experiment is currently queued against
a gate of unknown achievability.

This assay applies the standard the repository already applies elsewhere. The
physical perception gate was validated by an oracle "demonstrating that the gate
is achievable rather than internally inconsistent." The rank-regret gate has
never received that treatment.

## 3. Immutable inputs

Collection root:
`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3/attempt_v1/collection`

- registered collection result SHA-256
  `711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0`;
- registered V3 scientific plan SHA-256
  `0ad79cc46cead469d6532cd0be04c5d7623fffe18ddafc737c32855d6c9a8f29`;
- 64 scenes: 32 train, 32 evaluation, scene-disjoint, 8 families × 4 scenes per
  role;
- 256 states (128 per role), 4 states per scene, 9 branches per state;
- 3,072 PNGs: per role 384 context and 1,152 successor, 224×224 RGB;
- `stored_rgb_bytes` `120372020`.

Frozen encoders, both already cached locally, loaded offline with no network
access:

- `dinov2_vits14`, weights `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`,
  224×224 input, 16×16 = 256 patch tokens, 384 dimensions, per-patch L2
  normalization;
- V-JEPA 2.1 `vjepa2_1_vitb_dist_vitG_384`, weights
  `~/.cache/vjepa2_1_vitb_dist_vitG_384.pt`, used exactly as the frozen
  physical-interface ceiling used it, 256 tokens, 768 dimensions.

No Genesis, no rendering, no recollection, no scene filtering, no new scene,
action, trajectory, or RGB frame is generated. Nothing in the collection is
mutated.

## 4. Custody declaration

This assay **opens the evaluation-side successor PNGs for the first time**:
1,152 evaluation successor frames, which the V3 scientific attempt and its
complete-tie diagnostic both left unopened. It also opens all 1,536 train
frames.

After this assay the V3 panel is **spent for privileged-successor purposes** and
cannot serve as a fresh successor role again. This is the declared, accepted,
one-way cost of the assay. It touches no untouched, sealed, held-out, or V4
material, and consumes no sealed attempt.

The runner maintains an explicit access ledger and reports exact per-role,
per-kind RGB open counts. Expected: `train_context` 384, `train_successor`
1,152, `eval_context` 384, `eval_successor` 1,152.

## 5. Arms, capacity ladder, and fitting

### 5.1 Arms

All arms are fit on the 32 train scenes and evaluated on the 32 disjoint
evaluation scenes, under the unchanged §13 scorer, one-centimetre tolerance,
state/scene panel, and complete-tie convention.

| # | arm | input | role |
|---|---|---|---|
| 1 | `physical_oracle` | true dense rank | integrity anchor; regret `0` by construction |
| 2 | `privileged_physical_successor` | successor kinematic state relative to predecessor: body-frame `dx`, `dy`, `dyaw`, path length, `fell`, `tipped` | **capacity/identifiability control** (§5.4) |
| 3 | `dinov2_true_successor` | actual successor RGB, dense DINOv2 | **primary: visual information ceiling** |
| 4 | `vjepa2_1_true_successor` | actual successor RGB, dense V-JEPA 2.1 | representation comparator |
| 5 | `context_only` | 3 context frames + action, no successor | achievable-without-prediction floor |
| 6 | `task_action_only` | action and history, no vision | non-visual control, refit on this panel |
| 7 | `random_expected` | — | anchor |

### 5.2 Objective and readout

The readout family is the dense action-and-goal-conditioned attention-pooled
scalar readout of
`lewm/models/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py`,
generalized so that PCA width `K` and hidden/value width `H = V` are
configurable. Its parameter count is `3K(H+V) + 8H + 5V + 1`.

Consistent with the frozen ceiling, each visual arm is fit as a **residual on
the task/action-only ridge**: the ridge is fit first on the train role, and the
readout regresses `normalized_rank_target − task_score`. Arm scores are
`task_score + readout_residual`. Arm 6 is the ridge alone.

Three model seeds per rung, scores averaged as an ensemble, matching prior
practice.

### 5.3 Capacity ladder and rung selection

| rung | `K` | `H = V` | parameters |
|---:|---:|---:|---:|
| 0 | 8 | 4 | 245 |
| 1 | 32 | 32 | 6,561 |
| 2 | 128 | 128 | 99,969 |

Rung 0 reproduces the exact frozen ceiling readout and serves as the replication
anchor.

**Rung selection is by inner scene-disjoint validation on the train role only,
never by evaluation score.** The 32 train scenes are split 24 fit / 8
inner-validation, stratified to 3 fit and 1 validation scene per family, by the
registered split seed. PCA, the task ridge, and the readout are fit on the 24
fit scenes only, and each rung is scored on the 8 inner-validation scenes. The
rung with the lowest inner-validation regret is selected.

The selected rung is then **refit on all 32 train scenes** — PCA, ridge, and
readout — and `R*` is that refit model's evaluation regret. Evaluation regret
for **all** rungs, each refit on all 32 train scenes, is reported as a
sensitivity table, but the registered `R*` is the inner-selected one. Selecting
on evaluation would bias the ceiling optimistically and thereby bias toward
greenlighting the successor.

### 5.4 Assay-validity control (arm 2)

Arm 2 receives the privileged successor physical state, from which the dense
rank is in principle recoverable. It is therefore expected to reach near-zero
evaluation regret, and it is the control that establishes the readout family and
capacity ladder are adequate to express the target function at all.

**Registered validity threshold: arm 2 evaluation regret `<= 0.05` at the
selected rung.** If arm 2 exceeds `0.05`, the readout family is too weak, the
assay is declared invalid, `FAIL_ASSAY_CAPACITY_CONTROL` is emitted, and **no
Outcome in §6 may be claimed**. The assay fails closed.

### 5.5 Scoring and uncertainty

- Normalized rank regret with denominator `max(1, max_dense_rank)`; every action
  in a complete tie is oracle-equivalent; selected and random expected regret are
  both zero for a complete tie. Identical to the registered §13 diagnostic
  convention.
- All 128 evaluation states retained. No state is excluded.
- Paired family-balanced whole-scene cluster bootstrap over the 32 evaluation
  scenes, 10,000 resamples, bootstrap seed `2026080502`.
- Model seeds `(2026080511, 2026080512, 2026080513)`. Split seed `2026080503`.

## 6. Registered decision rule

Let `R*` be the evaluation regret of arm 3 at the inner-selected rung. Evaluated
only if §5.4 passes.

- **Outcome I — gate achievable.** `R* <= 0.13`.
  The `0.13` gate stands; the §13 residual of `0.272` is prediction error; the
  dense action-conditioned JEPA successor is justified as already specified.
- **Outcome II — gate too tight, visual information present.** `R* > 0.13`, and
  arm 3 beats arm 6 with the paired bootstrap upper bound below zero.
  The absolute gate is re-derived ceiling-relative: a successor must reach
  `R* + δ`, with `δ` fixed from this assay's own variance estimate. The
  successor remains justified but is re-gated.
- **Outcome III — no visual headroom.** `R* >=` arm 6, or `R*` worse than arm 2
  by a margin whose paired bootstrap interval excludes zero.
  No predictor architecture can close this. Stop the matched-branch rank-regret
  programme and move the primary endpoint to the closed-loop metric.
- **Outcome IV — panel degenerate.** The paired bootstrap interval for
  arm 5 minus arm 3 includes zero, i.e. actual successors add nothing over
  context.
  The branch construction does not produce visually distinguishable outcomes;
  the action grid or step length must be redesigned before any successor
  collection.

Outcomes are evaluated in order I, IV, III, II; the first that holds is the
registered terminal. If none holds the terminal is
`INCONCLUSIVE_NO_REGISTERED_OUTCOME` and no interpretation is claimed.

No threshold in this document may be relaxed, re-derived, or reinterpreted after
any score is observed.

## 7. Mandatory diagnostics

These are reported regardless of outcome and are not gates.

1. **Displacement-spread conditioning.** Evaluation regret for arm 3 and arm 5
   conditioned on per-state branch physical-displacement spread, in registered
   quartile bins over `physical_target_progress_m` range within a state. If the
   ceiling is dominated by low-spread states, the corrective action is the action
   grid or step length, not the architecture.
2. **Power output.** For each paired comparison, the bootstrap CI half-width at
   this panel size, and the number of evaluation scenes required to resolve a
   `0.02` effect at this observed variance. Every subsequent preregistration on
   this family must cite that number before fixing a gate.
3. **Prediction-value decomposition.** `arm 5 − arm 3`, the maximum possible
   value of perfect one-step visual prediction on this panel, with interval.
4. **Rung sensitivity.** Evaluation regret of every arm at every rung.
5. **Train-side fit.** Train and inner-validation regret per arm per rung,
   confirming the ladder reaches the capacity regime.

## 8. Integrity gates

All must pass or the run fails closed with no scientific decision:

- collection, plan, and predecessor result rehash to their registered SHA-256;
- role disjointness: no scene ID appears in both roles;
- role balance: 8 families × 4 scenes × 4 states per role;
- exact RGB open counts as declared in §4;
- arm 1 evaluation regret exactly `0.0`;
- arm 3 beats arm 7 (`random_expected`);
- byte-exact deterministic repeat of the complete evaluation in the same
  process;
- an independent replay process recomputes the registered result from the
  feature caches with zero RGB access and zero encoder execution, and reproduces
  every registered field exactly;
- exclusive output write; no overwrite, refill, retry, or resume is available.

## 9. What this does not authorize

No data generation, rendering, bulk re-render, pool scaling, training of any
navigation or promotion candidate, threshold relaxation, retry or resume of any
stopped mechanism, planner integration, blind rollout, deployment, or any access
to untouched, sealed, held-out, or V4 material. A pass on any Outcome does not
promote any checkpoint and does not authorize the successor experiment; it
determines only what gate that separately preregistered successor must face.
