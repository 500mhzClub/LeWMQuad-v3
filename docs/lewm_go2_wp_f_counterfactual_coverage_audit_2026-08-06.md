# WP-F coverage audit: same-state counterfactual action preservation

Date: 2026-08-06
Status: **AUDIT ONLY. No training launched.** `DEVELOPMENT_ONLY_NOT_CLAIM_BEARING`.
WP-E is untouched — not modified, not rerun.

---

## 0. Question

Does any corpus we hold contain *exact* counterfactual branch groups — one
predecessor state, two or more recorded actions, everything else held identical
— in sufficient quantity to train and evaluate an action-preservation objective?

Unrelated states carrying different actions are **not** counterfactuals and are
excluded throughout.

## 1. The WP-E corpus does not support the question

`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`

```
total pairs (all roles)            5172
distinct current endpoints         5172
branches-per-current-endpoint      {1: 5172}
predecessor states with >=2 actions   0
```

Every current endpoint appears exactly once. This corpus has **zero** same-state
counterfactual groups. The differing-action pairs in it are unrelated states and
were not treated as counterfactuals.

**The corpus WP-E trained on cannot answer this question at all.**

## 2. The V3 matched-branch collection does support it

`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3/attempt_v1/collection`

Purpose-built for exactly this structure.

| item | train | eval |
|---|---:|---:|
| scenes | 32 | 32 |
| families | 8 | 8 |
| states (counterfactual groups) | 128 | 128 |
| branches per state | 9 (all groups) | 9 (all groups) |
| total branches | 1,152 | 1,152 |

**Requested items 1–6:**

1. **Predecessor states with ≥2 recorded actions:** 256 (128 train + 128 eval).
   Branches-per-state histogram is `{9: 128}` in both roles — no partial groups.
2. **Bit-identical current RGB, pose and history:** yes, structurally. The
   context is stored **once per state**, not once per branch: all 9 branches
   resolve to the same three context PNG artifacts (`context:0/1/2`), the same
   `prebranch_base_pose_world`, and the same `history_action_ids`. 128/128 groups
   in each role. Identity is by construction, not by comparison tolerance.
3. **Same prediction horizon:** yes. `requested_block` length is `{5}` across
   every branch of every group.
4. **Actions per group:** all 9, always the full grid —
   `arc_left, arc_right, backward, forward_fast, forward_medium, forward_slow,
   hold, yaw_left, yaw_right`. 0 branches flagged `duplicates_candidate_action_id`.
5. **Role counts:** above. 
6. **Groups wholly inside one role:** yes. A state belongs to one scene and a
   scene to one role; raw scene-id overlap between roles is 0.

**Item 7 — pairwise distances between true successor token states**
(EMA-space convention: `LN(forward_tokens(successor)[:,1:,:])`, flattened cosine):

| | train | eval |
|---|---:|---:|
| within-group branch–branch cosine, mean | `0.8057` | `0.7949` |
| median | `0.8385` | `0.8531` |
| p10 / p90 | `0.6756` / `0.9497` | `0.5725` / `0.9584` |
| min-pair per group, median | `0.6694` | `0.6690` |

Calibration against a cross-state reference (**same action id, different
state**, train role): mean cosine `0.5563`, median `0.5775`, p10 `0.3466`.

The ordering is the correct one: successors of different actions at the *same*
state (`0.806`) are more similar to each other than successors of the *same*
action at different states (`0.556`). Actions move the representation
materially, and by less than scene identity does. There is a real
within-state action signal to preserve, and it is not swamped by scene identity.

**Item 8 — fraction of groups with meaningfully different successors:**

| threshold on the group's *closest* branch pair | train | eval |
|---|---:|---:|
| min-pair cosine < 0.95 | `0.992` | `1.000` |
| min-pair cosine < 0.90 | `0.961` | `0.953` |
| min-pair cosine < 0.80 | `0.766` | `0.711` |
| fraction of *all* branch pairs < 0.95 | `0.901` | `0.874` |

Essentially every group contains at least one genuinely separated action pair,
and ~90% of individual pairs are separated at the 0.95 level.

## 3. Where coverage is weak — stated plainly

**Physical branch separation is small.** Endpoint poses within a group:

| | train | eval |
|---|---:|---:|
| median per-branch displacement | `0.060` m | `0.044` m |
| median within-group displacement spread | `0.097` m | `0.090` m |
| median pairwise endpoint separation | `0.052` m | `0.044` m |
| **pairs ending within 5 cm** | **`0.481`** | **`0.558`** |
| pairs ending within 10 cm | `0.828` | `0.870` |
| branches `clipped` | 768 / 1152 | 768 / 1152 |
| branches `physical_fell` | 0 | 0 |
| groups with 9 distinct endpoints | 127 / 128 | 124 / 128 |

Roughly half of all action pairs put the robot within 5 cm of the same place,
and two thirds of branches were velocity-clipped. This is the same action grid
that produced `OUTCOME_IV_PANEL_DEGENERATE`. **A large minority of these
"counterfactuals" are physically near-degenerate**, and any objective built here
must not be penalised for failing to separate them.

For scale, the branch horizon is physically comparable to WP-E's h=1 (median
branch displacement `0.060` m against WP-E's h=1 translation median `0.0799` m),
so the two corpora are roughly commensurate in step size — though the horizon
definitions differ (5-step block versus fixed frame-index delta 240).

**There is no clean held-out role.** V3 has only `train` and `eval`, and the
eval-role successor RGB was already consumed by the observability-ceiling assay
(`attempt_v1`/`attempt_v2`) — the panel is spent for privileged-successor
purposes. A selection role must be carved out of the 32 V3 *train* scenes.

**Disclosure:** this audit read V3 **eval**-role successor RGB in order to report
items 7 and 8 for both roles. That role was already spent by the ceiling assay,
but the read is recorded here rather than left implicit. If a strictly untouched
V3 eval role was wanted, it no longer exists and did not before this audit.

**Scale is modest.** 128 train groups across 32 scenes.

## 4. Combinability with the WP-E corpus

Raw scene-id overlap between V3 (either role) and the WP-E corpus (both roles)
is **0**. Both cover the same 8 families
(`large_enclosed_maze, local_composite_motifs, loop_alias_stress,
medium_enclosed_maze, open_obstacle_field, rough_local_dynamics,
small_enclosed_maze, visual_sensor_stress`). The corpora are scene-disjoint, so
the main normalised-state JEPA loss can keep running on the WP-E corpus while
the counterfactual term draws on V3 without cross-contaminating either split.

## 5. Verdict

**Coverage is adequate to pose the question, with a bounded scope.**

Adequate: exact structure (256 groups × 9 actions, shared context by
construction, one horizon, groups whole within roles), a real and correctly
ordered successor signal, and scene-disjointness from the main corpus.

Bounded by: 128 usable training groups, ~48% physically near-degenerate branch
pairs, and no untouched held-out role. This supports a **development-tier
mechanism test**, not a claim-bearing result.

## 6. Recommended objective — target-grounded soft matching

One term, added to the existing normalised-state JEPA loss, which remains the
main objective.

For a group with shared current state `s`, actions `a_1..a_9`, EMA-target
successor states `t_1..t_9` and predictions
`p_i = LN(predictor(LN(z_s), a_i))`:

```
S_ij = cos(p_i, t_j) / tau_p          # prediction -> successor, 9x9
Q_ij = cos(t_i, t_j) / tau_t          # successor  -> successor, 9x9  (no grad)

L_match = (1/9) * sum_i  CE( softmax_j S_ij ,  softmax_j Q_ij )
```

Negatives are drawn **only from within the same group** — the same current
state, different action. That is what makes it a counterfactual objective rather
than a generic contrastive one.

**Why the target-side soft label, not a one-hot identity target.** With ~48% of
branch pairs physically within 5 cm, a hard `p_i ↔ t_i` InfoNCE would demand
separation that the true successors do not exhibit, and would be minimised by
fabricating differences the world does not contain. Using
`softmax_j Q_ij` as the label makes the loss **target-grounded**: where the true
successors are distinct, `Q_i` collapses toward one-hot at `i` and the term
reduces exactly to 9-way InfoNCE; where two actions genuinely lead to the same
place, `Q_i` spreads mass and the loss stops asking for separation. The floor is
set by the corpus, not by the objective.

This satisfies the constraint directly: it matches `p_i` to `t_i` rather than
`p_i` to `t_j`, and it never merely pushes predictions for different actions
apart — repulsion appears only where the *targets* are themselves apart.

**Composition.** `L = L_jepa(normalised state) + lambda * L_match + L_bev`,
with the BEV auxiliary unchanged and `lambda` small (start `0.1`). `Q` is
computed under `no_grad` from the EMA encoder; gradients flow only through `p`.

**Gate.** The metric already exists and is the one every encoder-moving recipe
so far has lost on: correct − shuffled changed-cosine on the designated
selection role, together with the three raw health diagnostics (token variance,
effective rank, temporal delta). The objective earns its place only if action
discrimination rises **without** the WP-E collapse signature.

**Calibration to record before interpreting any run:** chance is `1/9 = 0.111`,
and the achievable ceiling is not 1.0 — it is bounded by the target degeneracy
in `Q`. That ceiling must be computed from the successor states themselves and
reported alongside any accuracy, or the number is uninterpretable.

**Not launched.** No training run has been started.
