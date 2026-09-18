# Matched longer-budget comparator components prepared

The older measured-plane reactive, nominal and current-observation planning
controls retain the earlier tracker and 4,000-step mission. They are not
already matched to the prepared 8,000-step chained single-pass controller.
Separate new components now provide that common perception and mission
configuration without changing the old controls, active replay, prepared
native launcher or existing independent-layout study definitions.

## Comparison scope

| Component | Forecast and selection treatment | Retained state |
| --- | --- | --- |
| `ExtendedReturnBudgetChainedController` | Assigned frozen learned forecasts and predictive selection | Complete existing perception, routing, contact, residual and mission state |
| `ExtendedReturnBudgetForecastSourceController`, `frozen_world_model` | Explicit learned-source reference through the comparator selector | Same retained state and optimized footprint dispatch |
| `ExtendedReturnBudgetForecastSourceController`, `nominal_requested_twist` | Nominal requested-velocity forecasts through predictive selection; no model forward | Same retained state, including observed residual correction |
| `ExtendedReturnBudgetReactiveController` | Current measured geometry and observed-route reactive rule; no future candidate forecasts | Same perception, routing, contact and mission state; no model or forecast residual |
| `ExtendedReturnBudgetCurrentPlanningController` | Learned predictive selection queries only the current paired-observation routing cells | Persistent contact, tracking, floor anchors, model history, residual and mission state remain |

The reactive control remains a whole-method comparison, not an isolated
prediction-ranking ablation. The nominal control remains predictive. The
current-planning control isolates accumulated routing cells and is not a
fully memoryless controller. None of these source components by itself
establishes JEPA, prediction or memory benefit.

Reactive construction obtains the exact fresh tracker, registration, mapper,
memory and mission objects from the longer learned controller's constructor
with `model=None`. That constructor processes no observation and creates no
world model. The reactive object retains neither a model nor a residual;
all four compared instances have independent mutable perception objects and
independent single-pass indices. Its advance method uses the larger bounded
measured-pose accessor, with its original reactive selection and terminal
behavior otherwise retained.

The forecast-source selector composes the existing explicit forecast provider
with the longer optimized selector. The learned-source reference preserves
complete ordinary decisions after removing only explicit source provenance
and root comparator labels. The nominal provider makes no model forward,
retains eight-step predictive planning and does not label nominal outputs as
corrected model forecasts.

The current-planning map keeps the extended body-projected, tiled geometry
observation method and the original checked current-view capture. It keeps
the complete accumulated map and contact evidence; only the view passed to
the full optimized selection chain restricts routing cells. Views retain
their two camera witnesses, current timestamp and immutable routing-cell
mappings. A view from an earlier observation cannot be queried at a later
observation time.

## Observed tests and limitation

All tests used the deterministic single-thread existing Genesis Python
environment, `-B`, `PYTHONDONTWRITEBYTECODE=1` and pytest
`-q -p no:cacheprovider` on only the new named files. These short tests
overlapped the original live full-history comparison, whose timing remains
nonisolated development timing.

- Comparator controllers: session 94789, **13 passed in 18.36 s**, first
  observed invocation, exit zero. Checks use actual image inputs and fixed
  corrected synthetic models for full-RGB JEPA and no-RGB direct heads.
  Across the first four observations, the four arms match pose evidence,
  complete measured memory/mission receipts, map cells and retained routes.
  The learned and explicit learned-reference arms make one forward each;
  the nominal arm makes zero. Complete learned-reference decisions match
  after only the stated provenance normalization. All three predictive arms
  enter the extended optimized footprint scope. Duplicate observations stop
  all arms without extending the map.
- Current-planning controller: first invocation session 85001 produced
  **2 failed, 1 passed in 10.72 s**. Both failures were the final test
  assumption that duplicate-input tracking failure invalidates an untouched
  map's last valid view. Actual controller behavior had stopped in tracking
  before calling the mapper; its last valid evidence remained intact.
  No implementation or tolerance was changed. The test now verifies that
  the last valid view remains readable at its original timestamp, is rejected
  at a later timestamp, and terminal control issues zero command without
  another model forward. Revised invocation session 33209 produced
  **3 passed in 10.88 s**, exit zero.

The current-planning tests also execute the real image-to-action pipeline for
both head/input variants. They compare complete observation/mission receipts,
all retained route and map cells, original contact queries at two poses per
frame, model forecasts, immutable current views, and stable tracking/memory
object identities. Both arms retain the extended footprint optimization.
They do not establish long-run physical behavior, 8,000 real image observations,
timing improvement, independent-layout reliability or navigation success.
Longer population/accessor tests remain the separately recorded component
evidence; these controls need their own integration and prospective execution.

## Source identity and custody

A source-only discovery/check, session 13582, exited zero with a 2,679-path
union: the prepared native source/test ancestry plus these four new paths.
It reverified the active replay's 2,639 bound sources, all unchanged. The
prospective budget-prefix and native attempt directories remained absent.
No scene, training, replay of a full recorded episode, source export or sealed
material was accessed by this preparation.

| Exact path | SHA-256 |
| --- | --- |
| `lewm/extended_return_budget_comparator_controllers_development.py` | `d45e667824c0ba144811167a288797a1e14138a1f2d5b2a031049b1b4630f384` |
| `lewm/tests/test_extended_return_budget_comparator_controllers_development.py` | `5f441be6e3fd8e9ab1ee950115145c936d19db38c732f6d721645a9225d8dad8` |
| `lewm/extended_return_budget_current_planning_development.py` | `93ea16afae35d8f2293aa0ed1a91755c75759b5784c9e2a112a95b4412051b5c` |
| `lewm/tests/test_extended_return_budget_current_planning_development.py` | `c0acf3851bf6c853d33b08cf848293d20ffcf146af7fb99f0bfa932d6e8f2369` |

These four paths are outside the active replay's frozen roster. The prepared
single-case native launcher still uses its original learned controller and
2,675-path source union; this document does not add these controls to that
experiment or admit an independent-layout population.

## Remaining work

Complete and authenticate the exact live chained/single-pass comparison,
then the prepared prospective budget-prefix check, then the prepared fresh
longer native trial if admitted. Preserve negative results at every stage.
After reviewing the actual longer native result, integrate appropriate
matched controls into collection and full raw auditing and define the
independent-layout, appearance and training-seed roster prospectively. The
old 32-case roster remains unexecuted and is not silently updated here.

Real-time sensing/control, useful prediction and memory on independent
layouts, matched training comparisons and bounded physical-robot evidence
remain requirements of the active larger goal.
