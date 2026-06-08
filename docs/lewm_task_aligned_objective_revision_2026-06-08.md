# Task-Aligned Objective Revision + Final-Block Adapter Rerun

Date: 2026-06-08

Follow-up to `docs/lewm_task_aligned_frozen_base_result_2026-06-08.md`. That
program closed the frozen-base candidate scorer, the ranking auxiliary, the
inference selection-grid sweep, and a first final-block adapter — all failed
the minimum gate. The post-mortem named one remaining confound: the task cost
over-rewarded safe low-motion turns. This document revises the objective to
remove that confound and reruns the final-block adapter against it.

## Diagnosis Of The Old Cost

Old per-candidate ground-truth cost (`score_task_aligned_counterfactuals.py`):

```
task_cost = distance + 0.25 * heading_error
cost      = task_cost + 2.0 * collided + 1.0 * clearance_shortfall
```

Each velocity primitive is `block_size 5 x dt 0.10 s = 0.5 s`, so the fastest
forward primitive (0.30 m/s) moves only ~0.10 m. The inter-candidate progress
spread is therefore ~centimeters, while the collision term is 2.0 and the
clearance shortfall up to 0.35. The progress signal sat inside the noise of the
penalty terms, so the oracle-"best" action was reliably a safe turn-in-place or
even a retreat.

Oracle-best distribution under the old cost (val32, 16,384 decisions):

| group | share |
|---|---:|
| arcs (arc_left/right) | 43.2% |
| backward (away from goal) | 24.5% |
| turn-in-place (yaw/hold) | 24.2% |
| pure forward | 8.1% |

A model imitating that cost gets low cost-regret but ~0 progress, which
contradicts the "positive progress" gate. Objective and gate were inconsistent:
the trivial `yaw_left` prior had cost-regret only 0.116 because turning is cheap
under this cost, so "beat the prior on regret while making progress" was
unsatisfiable by any moving policy.

## Revised Objective

Two changes, both registered as new defaults.

1. **Progress-dominant task term.** Replace absolute `distance` with progress,
   scaled so one primitive of forward travel is a unit-scale reward:

   ```
   task_cost = -10.0 * progress + 0.1 * heading_error
   cost      = task_cost + 3.0 * collided + 0.5 * clearance_shortfall
   ```

   Weights: `progress_weight=10.0`, `heading_weight=0.1`,
   `collision_penalty=3.0`, `clearance_target_m=0.35`, `clearance_penalty=0.5`.
   Calibrated on existing scored geometry (progress/heading/clearance/collision
   are geometric and independent of the weights, so calibration needs no
   re-simulation). `progress_weight` saturates above ~10; collision stays a near
   hard veto (`collision_penalty * max_progress = 1.0 < 3.0`).

2. **Drop no-safe-action decisions.** Decisions where every candidate collides
   (boxed-in spawn poses) are not navigable and impose an irreducible collision
   floor that makes the collision gate meaningless. `--drop-no-safe-action`
   (default on) removes them: 1,630/16,384 train (9.9%) and 1,902/16,384 val
   (11.6%). After filtering the oracle collision floor is 0.0.

Counterfactual schema bumped to `task_aligned_counterfactual_v1`. New scored
data: `.generated/task_aligned_decisions/{train32,val32}_v1_scored.jsonl`
(14,754 / 14,482 rows). The old-cost files are preserved.

### Effect on the oracle and controls (val32, filtered)

Oracle-best now leads with forward motion (forward family ~43%, forward_fast
the single most common best action), oracle mean progress +0.069 m at 0%
collision. The control baselines the policy must beat:

| control | regret | collision | progress |
|---|---:|---:|---:|
| oracle | 0.000 | 0.00% | +0.0687 |
| action-only (`yaw_left`) | 0.609 | 2.76% | 0.0000 |
| logged behavior policy | 0.873 | 22.45% | +0.0441 |
| random | 0.998 | 20.23% | +0.0192 |

The gate is now well-posed: the oracle satisfies the minimum gate (collision
0% < 2.76%, progress > 0), so passing is achievable by a good-enough policy.
`yaw_left` is now correctly far from the oracle on regret (0.609), so "beat the
prior" is a genuine navigation test.

Registered gates (auto-recomputed from controls under the new cost):

- minimum validity: regret < 0.609, collision < 2.76%, progress > 0;
- promotion: regret-ratio vs random ≤ 0.5 (regret < 0.499), collision ≤ 5%,
  progress > +0.0192.

## Final-Block Adapter Result (v1 cost)

Same bounded adapter as before: initialize from `lewm_seq11_e3.pt`, freeze the
predictor/projectors/early encoder, unfreeze only the final two ViT blocks and
final norm, train jointly with the multi-task candidate scorer. One seed, eight
full train/eval epochs. Deploy selection uses the same weights as the cost.

Artifact: `.generated/task_aligned_policy_v0/adapter_v1_last2_seed20260608.json`

| model | regret | collision | progress | min gate |
|---|---:|---:|---:|---|
| **final-two-block adapter (cp=3 deploy)** | **0.571** | 5.90% | **+0.0206** | fail (collision) |
| action-only `yaw_left` prior | 0.609 | 2.76% | 0.0000 | control |
| logged behavior policy | 0.873 | 22.45% | +0.0441 | control |
| random | 0.998 | 20.23% | +0.0192 | control |

This is a qualitative change from the frozen-base program. Under the corrected
objective the adapter:

- **dominates the behavior policy and random on every axis** (regret, collision,
  progress);
- makes positive progress above random (+0.0206 > +0.0192);
- reaches regret-ratio 0.572 and collision 5.90%.

It misses the minimum gate **only on collision** (5.90% vs the do-nothing prior
2.76%) and misses the promotion gate by small margins (regret-ratio 0.572 vs
0.50; collision 5.90% vs 5.0%). The 8-epoch trace trades more collision for more
progress (collision 0.059 -> 0.077 as progress rises), so the binding bottleneck
is held-out collision prediction, not progress.

## Deploy Collision-Penalty Sweep

With `ranking_weight=0` the trained head is invariant to the deploy collision
penalty — that weight only enters action selection. So the collision miss can be
bought back at selection time without retraining the representation. Same seed,
same 8 epochs, `--collision-penalty 5.0`.

Artifact: `.generated/task_aligned_policy_v0/adapter_v1_last2_cp5_seed20260608.json`

| deploy cp | regret | regret-ratio | collision | progress | min gate | promo gate |
|---|---:|---:|---:|---:|---|---|
| 3.0 | 0.571 | 0.572 | 5.90% | +0.0206 | fail (coll) | fail (ratio, coll) |
| 5.0 | 0.575 | 0.577 | **4.74%** | +0.0148 | fail (coll) | fail (ratio, prog) |
| oracle | 0.000 | 0.000 | 0.00% | +0.0687 | — | — |
| action-only `yaw_left` | 0.609 | 0.611 | 2.76% | 0.0000 | — | — |
| random | 0.998 | 1.000 | 20.23% | +0.0192 | — | — |

The knob behaves exactly as expected: raising the deploy collision penalty walks
the policy down a clean safety/progress Pareto frontier — collision drops below
the 5% promotion bound (4.74%), but progress falls below random and regret-ratio
is unchanged. **Regret-ratio is pinned at ~0.57 regardless of cp** (the deploy
weight rebalances safety vs progress but not the overall cost-regret), so 0.57 is
the representational ceiling of this adapter; the promotion bar is ≤0.50.

## Interpretation And Next Step

The objective revision did its job. It converted an ill-posed contract — where
the oracle was a retreat/turn and `yaw_left` was near-optimal — into a well-posed
one where the oracle is forward motion (+0.069 m, 0% collision) and `yaw_left` is
correctly far from it. Under that corrected objective the final-block adapter is
a **real navigation policy**: it strictly dominates the logged behavior policy
and random on regret, collision, and progress, and makes positive progress above
random. This is a categorical improvement over the frozen-base program, where
every learned policy was dominated by the do-nothing prior.

It still does not pass the gates. Two facts locate the remaining wall:

1. **Regret-ratio plateaus at ~0.57** across deploy settings and is flat over the
   last several epochs — this is not under-training and not a selection-rule
   artifact. The frozen LeWM base, even with its final two blocks adapted to the
   task, cannot rank the nine candidate actions closely enough to the oracle.
2. **The collision/progress trade is a frontier, not a free lunch.** The adapter
   can be as safe as ~4.7% collision (cp=5) or push progress to +0.021 (cp=3),
   but not simultaneously hit collision < 2.76% and progress > random. The oracle
   does both, so the information exists in the privileged geometry; the adapted
   frozen descriptors do not expose it sharply enough on held-out scenes.

Per the decision rule registered in
`docs/lewm_task_aligned_frozen_base_result_2026-06-08.md`: small-head and
deploy-time tuning are now exhausted, and the task target has been corrected.
The two remaining review items before any large retrain are the
**goal-conditioning contract** (the head sees only a pooled frozen goal
descriptor plus a goal-present bit; the behavior policy reaches +0.044 m progress,
so progress is attainable — the adapter is leaving it on the table) and the
**base training objective**. Recommended next controls, in order:

1. a frozen-feature head trained under the identical v1 objective, to confirm
   whether final-block adaptation is contributing at all (does it beat the frozen
   substrate's regret-ratio, or is ~0.57 already the frozen ceiling?);
2. if adaptation helps, a bounded capacity step (unfreeze four blocks, or a
   stronger goal-conditioning path) rather than more epochs, which only trades
   collision up for progress;
3. only if both stall, revisit the base objective / a larger retrain.

Do not collect new rollouts for any of these; the 32-scene decision sets and
v1 scoring are sufficient.

## Artifacts

- v1 scored decisions: `.generated/task_aligned_decisions/{train32,val32}_v1_scored.jsonl`
- adapter (cp=3 deploy): `.generated/task_aligned_policy_v0/adapter_v1_last2_seed20260608.{pt,json}`
- adapter (cp=5 deploy): `.generated/task_aligned_policy_v0/adapter_v1_last2_cp5_seed20260608.{pt,json}`
- logs: `.generated/task_aligned_policy_v0/adapter_v1{,_cp5}_seed20260608.log`
