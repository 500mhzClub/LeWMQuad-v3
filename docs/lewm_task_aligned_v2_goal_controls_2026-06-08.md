# Task-Aligned v2 Contract Repair + Goal-Source Controls

Date: 2026-06-08

Executes Phase 1 (contract repair) and most of Phase 2 (frozen goal-source
controls) from `docs/lewm_task_aligned_contract_review_2026-06-08.md`. No new
rollouts; this is an index/label repair plus diagnostic controls on the existing
32-scene decision sets.

## Contract repair (Phase 1)

`mine_task_aligned_decisions.py` now attaches, per decision:

- `route_target_frame` — representative image of `route_target_id` (final goal),
  the only goal image v1 used;
- `local_target_frame` — representative image of `oracle_next_cell_id`, the
  **local next cell that scoring actually targets**;
- `target_present` (a scored target exists) separated from
  `route_target_image_present` / `local_target_image_present`.

Re-mined to `{train32,val32}_v2.jsonl` (deterministic: 16,384 rows/split, same
as v1). Re-scored with the unchanged progress-dominant v1 cost but
`--no-drop-no-safe-action`, so recovery rows are retained
(`{train32,val32}_v2_scored.jsonl`).

Index facts (train32_v2): `target_present` 77.6%, `route_target_frame` 73.9%,
`local_target_frame` 68.5%. Where both frames exist, the **local image differs
from the route image in 78.2% of rows** — the v1 contract bug was substantial,
and the matched-local control is well powered.

## Goal-source controls (Phase 2)

Frozen 2x2-spatial head (same architecture/seed/objective as the v1 control),
three goal sources via `build_task_aligned_feature_dataset.py --goal-frame-field`:

- `none` — start image + action only (goal zeroed);
- `route` — final-route image (reproduces the v1 input contract);
- `local` — matched local-subgoal image (the repaired contract).

Metrics are sliced by decision type with
`scripts/evaluate_task_aligned_goal_controls.py`, because regret-ratio averaged
over all retained rows is diluted by goal-less recovery rows. `branch` rows are
the goal-seeking decisions; `recovery` rows have no goal task term.

### Regret-ratio vs random (lower is better), val32_v2

| subset | none | route | local | rows |
|---|---:|---:|---:|---:|
| all (diluted) | 0.697 | 0.620 | 0.604 | 16,384 |
| goal-present | — | 0.722 | 0.679 | 11,491 (route) |
| **branch** | **0.904** | 0.762 | **0.734** | 9,141 |
| **branch ∩ goal-present** | — | 0.749 | **0.689** | 8,469 (route) |
| recovery | 0.529 | 0.547 | 0.557 | 12,426 |

Selected target progress (branch): route +0.0226 m, local **+0.0305 m** (+35%).
(`none` has no goal term so its branch∩goal / goal-present slices are empty.)

### Reading

1. **Goal conditioning works.** Adding any goal image collapses branch
   regret-ratio from 0.904 (no goal) to ~0.76. The head genuinely uses the goal.
2. **The v1 contract bug was real and cost real signal.** The matched local
   image beats the final-route image by 0.028 (branch) / 0.060 (branch∩goal)
   regret-ratio and +35% progress, and is neutral on recovery rows where the
   goal is irrelevant — the signature of a correct fix, not noise.
3. **Necessary but not sufficient.** Even with the correct local image,
   branch∩goal regret-ratio is ~0.69 — still far from the 0.50 promotion bar.
   Fixing the goal contract does not by itself unlock the local-action task.

The earlier "~0.57 representation ceiling" is therefore doubly retired: it was
measured on a confounded contract, and on the repaired contract the goal-seeking
gap is real but its cause is not yet localized.

### Caveats

- Recovery rows are retained, so selected collision is ~18% (the inflated-grid
  all-collide floor). Regret-ratios are **not** comparable to the v1 filtered
  numbers; compare only within this table.
- `local_target_frame` is available on 5.4 points fewer rows than
  `route_target_frame` (some oracle-next cells were never visited, so no
  representative frame). The local control is thus mildly handicapped on
  coverage yet still wins.
- `collided` remains an unvalidated 0.20 m inflated-grid proxy (rename/validate
  to `grid_unsafe` is still open from the contract review).

## Next: privileged relative-goal vector (Phase 2 control #4)

The remaining open question is whether the branch∩goal ~0.69 gap is **visual
goal-matching** (the spatial image descriptor cannot localize the subgoal
precisely) or the **base representation / safety prediction**. The decisive
control is a privileged start→target bearing/distance vector as the goal input:

- if it materially beats the local image, visual goal-matching is the bottleneck
  (pursue richer spatial goal cross-attention, deployable goal encodings);
- if it does not, the limit is representational/safety, and only then is a base
  change justified.

This requires adding start→target-cell geometry to the feature builder; it is
the recommended next step and was deliberately checkpointed before building.

## Artifacts

- v2 index: `.generated/task_aligned_decisions/{train32,val32}_v2{,_scored}.jsonl`
- features: `.generated/task_aligned_policy_v0/{train32,val32}_v2_{none,route,local}_spatial2.npz`
- heads: `.generated/task_aligned_policy_v0/head_v2_{none,route,local}_spatial2_seed20260608.{pt,json}`
- sliced controls: `.generated/task_aligned_policy_v0/v2_goal_controls.json`
