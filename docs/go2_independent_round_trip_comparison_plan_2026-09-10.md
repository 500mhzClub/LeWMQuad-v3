# Next independent-layout comparison: implementation plan

Use the complete existing eight-layout inventory with a matched four-arm
development study. This is preparation for a future protocol and launcher,
not a native launch, frozen execution authority or a completed comparison.
The original six-model maze2 waiter retains ownership of the next native run.

| Arm | Controller | Assigned primary model |
| --- | --- | --- |
| Persistent JEPA planner | ResidualAnchoredContinuationController | seed_2026091001_full_jepa |
| Persistent supervised planner | ResidualAnchoredContinuationController | seed_2026091001_full_supervised_rollout |
| Reactive, nonpredictive route rule | ReactiveFloorTransportController | No high-level world model |
| Current-pair JEPA planning map | ResidualCurrentObservationPlanningController | Same model as persistent JEPA |

Use the already assigned expanded-data models and training-only corrections;
do not select a seed or checkpoint using the transfer prediction readout or the
new layouts. The original primary-seed assignment predates the corrected score
inspection. Bind the exact corrected model states from the completed correction
admission, not a filename or a freshly refitted substitute.

The JEPA/supervised comparison addresses the training objective under matching
data, schedule and native planner. The two JEPA arms address planning-grid
persistence while retaining contact and localization histories. The reactive
arm compares the whole predictive-planning method against an observed-route
rule without learned forecasts or residuals. Its future-pose feasibility gates
necessarily differ: do not present it as an isolated prediction-ranking
ablation or as two separate baseline arms merely because it is both reactive
and nonpredictive. Additional controls remain possible if the results require
a narrower attribution. No advantage is guaranteed or required for retaining
an outcome.

All four arms must execute all eight declared layouts, with fresh controller,
memory and model instances. Reuse each layout's fixed physics and appearance
seeds across its arms. Freeze the complete case order before the first scene,
retain every scientific failure and prohibit outcome-based layout replacement.
There are 32 episodes but only eight distinct layout units for paired analysis.
Do not count treatment repetitions as independent maze replications.

The existing 10GiB collection allowance plus 1GiB persistence headroom per case,
over the unchanged 40GiB reserve, requires 392GiB free for the full 32-case
population. The last measured 612GiB artifact capacity would still cover this
after the queued six-case 66GiB allowance. This is an estimate, not a reservation:
refresh actual storage, RAM and competing processes before admission and each
case. Keep one native scene; available CPU cores alone do not establish a safe
multi-scene renderer configuration.

Implementation still required before launch:

1. A fixed arm/model factory and complete case manifest. Extend the separately
   checked independent-layout collector/auditor integration to reactive and
   current-pair treatments without modifying the original live sources.
2. Full original inventory, model/correction and completed six-case input
   admission. The memory-prefix result
   `f802f1c14263878520e4d9221128bd037442c2ed9fdae62de2ba589eb1efaf49` establishes
   the source intervention on the old assigned model; it is not a forecast or
   physical-prefix claim for the expanded models.
3. Exact matched per-layout startup evidence, native geometry/setup/renderer
   checks, complete raw sensor and controller-command replay, strict visibility,
   contact, distinct-edge, outbound/return and timing accounting for every case.
4. A final paired readout over the complete planned population. Passing source
   checks or lower prediction error cannot substitute for verified navigation.

Inspect the queued expanded-model outcomes before spending the independent
population. If they expose a correctable implementation failure, diagnose and
fix it on existing development evidence, then revise the still-unexecuted
population protocol prospectively. Do not use new-layout results to select a
replacement policy and retain a claim that those layouts are unseen.

This comparison remains development simulation. The full goal additionally
requires reliable end-to-end navigation evidence, realistic sensing and timing,
and bounded real-platform evidence before deployment claims.
