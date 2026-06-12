# Task-Aligned v1 Contract Review And Next Plan

Date: 2026-06-08

This review follows the progress-dominant v1 objective and the frozen-v1
control documented in `docs/lewm_task_aligned_objective_revision_2026-06-08.md`.
The objective rebalance was necessary and useful: it removed the old incentive
to choose low-motion turns. However, the resulting adapter metrics must not yet
be interpreted as a frozen-base representational ceiling. The v1 input, target,
and safety-label contracts still contain material inconsistencies.

## Decision

Do not run a larger adapter, goal cross-attention model, or base retrain yet.
First build and validate a task-aligned v2 decision contract. Existing rollout
data remains sufficient; this is a label/index repair, not a collection task.

## Findings

### 1. v1 mixes final-route-goal and local-subgoal contracts

`mine_task_aligned_decisions.py` attaches `target_frame` from `route_target_id`.
`score_task_aligned_counterfactuals.py` instead scores progress toward
`oracle_next_cell_id` when it exists.

On the filtered v1 validation index:

- 10,260 rows have a target frame;
- 6,388 of those rows (62.26%) score a different target cell from the cell
  depicted by the target frame;
- 767 rows (5.30% of all validation rows) have a scored target but no target
  frame.

The trainer defines `goal_present` from target-frame availability. Therefore,
those 767 targeted rows are trained and selected as though no goal exists, even
though their ground-truth cost includes progress and heading. The model is
commonly asked to infer local-next-cell progress from a final-route-target image
and is sometimes given no goal task term at all.

A final route-goal image paired with a local action label can define a valid
image-goal navigation task, but it requires routing/topological inference. It is
not the registered local-action contract: `docs/v3_topological_nav_plan.md` explicitly
states that local control chases the selected subgoal's representative image,
not the final destination image. v1 therefore mixes the routing and local-action
problems and cannot establish which one causes the remaining gap. This prevents
the current result from establishing that pooled goal conditioning or the
frozen representation is the binding local-action limit.

### 2. No-safe-action filtering removes recovery data using unvalidated labels

The scorer marks a candidate `collided` when its point trajectory enters the
0.20 m inflated occupancy grid. If the recorded start pose is already inside
the inflated region, stationary `hold`, `yaw_left`, and `yaw_right` are all
marked collided even though they do not translate.

On filtered v1 validation:

- `hold`, `yaw_left`, and `yaw_right` each have exactly 2.76% collision;
- their collision labels are identical on 100% of rows.

On the 1,902 validation rows dropped as having no safe action:

- 1,886 (99.16%) are recovery or branch+recovery rows;
- 98.79% recorded positive actual translational motion;
- only 27.02% ended stuck.

These rows are not established as unnavigable. Dropping them makes the gate
easier by removing difficult recovery states and should not be treated as
removing a proven irreducible physical-collision floor.

### 3. The reported v1 result remains useful but narrower than claimed

The corrected progress scale demonstrates that the old cost was ill-posed and
that final-block adaptation helps relative to the frozen control:

| model | regret-ratio | grid-collision label | progress |
|---|---:|---:|---:|
| frozen-v1 spatial2 | 0.593 | 7.40% | +0.0241 m |
| final-two-block adapter | 0.572 | 5.90% | +0.0206 m |

This is evidence that task adaptation exposes useful signal. It is not yet
evidence of a base-representation ceiling because most goal-conditioned rows
use a mismatched input target and the safety gate uses an unvalidated proxy.

## Next Bounded Program

### Phase 1: repair and audit the v2 contract

1. Attach both a local-subgoal frame for `counterfactual_target_cell_id` and a
   final-route-target frame. Use the local frame for the registered local-action
   benchmark; report final-route-goal navigation as a separate harder task.
2. Separate `target_present` from `target_image_present`. For the
   goal-image-conditioned benchmark, either require a matching local-target
   frame or explicitly report the missing-target subset; never silently convert
   it to a targetless row.
3. Split reporting by `branch`, `recovery`, and `branch+recovery`. Branch rows
   require local-target progress. Targetless recovery rows require clearance
   gain / stuck-exit metrics, not progress.
4. Rename the current `collided` field to `grid_unsafe` until it is validated.
   Audit start-in-inflation cases and do not discard them by default.
5. Validate the grid proxy on logged actions and, if existing labels cannot
   establish physical contact, run a bounded physics replay of sampled existing
   start states. This does not require new rollout collection.

Stop Phase 1 unless:

- the local-action benchmark's input subgoal equals its scored target;
- final-route-goal and local-subgoal results are reported separately;
- no scored target is silently treated as absent;
- safety labels have defined behavior for starts already inside the inflated
  region;
- filtered/excluded coverage is reported by task type.

### Phase 2: cheap diagnostic ceilings on v2

Run four frozen-head controls before adapting the encoder:

1. **No-goal control:** start image + candidate action only.
2. **Final-route-target image:** preserve the current v1 input contract as a
   routing-plus-control comparison.
3. **Matched local-target image:** start image + frame from the scored local
   target cell.
4. **Privileged relative-goal vector ceiling:** start image + candidate action +
   target bearing/distance derived from geometry.

The relative-goal vector is a diagnostic ceiling unless equivalent geometry is
available at deployment. It decides whether failures come from visual goal
matching or from safety/action prediction.

Decision rule:

- local-target image materially beats no-goal: then test richer spatial
  goal-conditioning;
- local-target image materially beats final-route-target image: the current v1
  gap is primarily target-contract/topological ambiguity, not local control;
- privileged vector beats local-target image materially: visual goal matching
  is the bottleneck;
- neither improves regret while safety prediction remains weak: repair the
  safety target/model before touching the base encoder.

### Phase 3: only then rerun bounded adaptation

Rerun the frozen spatial head and final-two-block adapter on the identical v2
contract for three seeds. Promote only when the result passes task-specific
gates on held-out scenes and reports full-distribution coverage, including the
recovery subset.

Do not unfreeze more blocks or retrain the base model unless the corrected v2
contract and matched-goal controls still show a stable representation-limited
gap.

## Reproducible Audit

```bash
python3 scripts/audit_task_aligned_contract.py \
  --scored .generated/task_aligned_decisions/val32_v1_scored.jsonl \
  --unfiltered-scored .generated/task_aligned_decisions/val32_scored.jsonl \
  --output .generated/task_aligned_decisions/val32_v1_contract_audit.json
```
