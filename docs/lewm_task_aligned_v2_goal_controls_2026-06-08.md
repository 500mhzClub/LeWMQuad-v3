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
four goal sources:

- `none` — start image + action only (goal zeroed);
- `route` — final-route image (reproduces the v1 input contract);
- `local` — matched local-subgoal image (the repaired contract);
- `privileged` — diagnostic ceiling: the body-frame start->target offset
  (`[cos bearing, sin bearing, tanh distance]`) injected into the goal slot,
  available on every scored-target row. Not deployable (uses ground-truth
  geometry); it isolates whether the limit is reading the goal or using it.

Image goals come from `build_task_aligned_feature_dataset.py --goal-frame-field
{none,route_target_frame,local_target_frame}`; the privileged goal is derived
from the no-goal features by injecting the `relative_goal_vector_body` that
`score_task_aligned_counterfactuals.py` now records.

Metrics are sliced by decision type with
`scripts/evaluate_task_aligned_goal_controls.py`, because regret-ratio averaged
over all retained rows is diluted by goal-less recovery rows. `branch` rows are
the goal-seeking decisions; `recovery` rows have no goal task term.

### Regret-ratio vs random (lower is better), val32_v2

| subset | none | route | local | privileged | rows |
|---|---:|---:|---:|---:|---:|
| all (diluted) | 0.697 | 0.620 | 0.604 | 0.426 | 16,384 |
| **branch** | **0.904** | 0.762 | 0.734 | **0.517** | 9,141 |
| branch ∩ goal-present | — | 0.749 | 0.689 | 0.517 | 8,017–9,141 |
| recovery | 0.529 | 0.547 | 0.557 | 0.429 | 12,426 |

Selected target progress (branch): route +0.0226 m, local +0.0305 m (+35%),
**privileged +0.0456 m (2x route)**. Goal coverage on branch rows: route 92.6%,
local 87.7%, privileged 100% (geometry is always available). `none` has no goal
term so its goal-present slices are empty.

### Reading

1. **Goal conditioning works.** Adding any goal image collapses branch
   regret-ratio from 0.904 (no goal) to ~0.76. The head genuinely uses the goal.
2. **The v1 contract bug was real and cost real signal.** The matched local
   image beats the final-route image by 0.028 (branch) / 0.060 (branch∩goal)
   regret-ratio and +35% progress, and is neutral on recovery rows where the
   goal is irrelevant — the signature of a correct fix, not noise.
3. **Image goals are necessary but not sufficient.** Even with the correct local
   image, branch regret-ratio is ~0.73 — still far from the 0.50 bar.
4. **The bottleneck is visual goal-matching, not the base representation.** The
   privileged relative-goal vector — the same frozen substrate and head, handed
   the goal as body-frame geometry — drops branch regret-ratio to **0.517** and
   doubles progress (**+0.0456 m**). The image-vs-privileged gap (0.22) dwarfs
   the local-vs-route gap (0.03). So the representation and the action/safety
   head are *not* the limit: given a usable goal signal they nearly pass the
   gate. The binding constraint is recovering the subgoal's relative position
   from a goal *image*. (Privileged's ~12% coverage edge cannot explain a 0.22
   regret-ratio gap.)

The earlier "~0.57 representation ceiling" is therefore retired three times over:
it was measured on a confounded contract; on the repaired contract the gap is
real; and a privileged goal signal nearly closes it without touching the base
encoder — so the base representation was never the binding limit.

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

## Conclusion and next step

All four controls are run. The diagnostic resolves cleanly: the local-action
gap is **visual goal-matching**, not the frozen base representation and not the
action/safety head. A privileged goal vector nearly passes the gate on the same
substrate; a pooled goal *image* does not.

Recommended next work, in order:

1. **Goal-localization path.** Add a head that recovers relative goal geometry
   (bearing/distance, or a goal-relative spatial attention map) from the goal
   image, and feed that to the candidate scorer. Target: close the image→
   privileged gap (branch regret-ratio 0.73 → ~0.52). This is deployable; the
   privileged vector is only the diagnostic ceiling.
2. **Validate `grid_unsafe`** (still the open contract-review item): the retained
   recovery rows hold collision at ~18%; rename/validate the inflated-grid label
   before reading absolute collision/promotion numbers.
3. Re-run the final-two-block adapter for three seeds **only after** the goal
   path lands, on the v2 contract. Do not retrain or further unfreeze the base
   encoder: these controls show it is not the binding limit.

Do not collect new rollouts.

## Artifacts

- v2 index: `.generated/task_aligned_decisions/{train32,val32}_v2{,_scored}.jsonl`
- features: `.generated/task_aligned_policy_v0/{train32,val32}_v2_{none,route,local,privileged}_spatial2.npz`
- heads: `.generated/task_aligned_policy_v0/head_v2_{none,route,local,privileged}_spatial2_seed20260608.{pt,json}`
- sliced controls: `.generated/task_aligned_policy_v0/v2_goal_controls.json`
