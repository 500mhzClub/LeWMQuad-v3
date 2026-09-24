# Fixed independent-layout comparison assignments V1

This implements the arm factory and full roster from
`go2_independent_round_trip_comparison_plan_2026-09-10.md`.
It is preparation, not a frozen execution protocol or native launch. The
original six-model maze2 waiter keeps ownership of the next native scene.

All eight layouts receive persistent JEPA, persistent supervised rollout,
reactive observed-route control, and current-pair JEPA planning. The assigned
expanded-data primary seed is 2026091001. The factory binds the exact corrected
tensor states: JEPA `35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`
and supervised `755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.
The two JEPA arms use the same state identity but load separate model instances.
Reactive control has no high-level model or learned residual. Every factory
call constructs a new controller, map, contact memory, mission and tracker.
The future launcher must also enforce a fresh worker process for each case.

The complete 32-case roster is layout-major. Starting from the listed arm order,
layout index i rotates the arm order left by i modulo four. Thus every arm
occupies every within-layout execution position twice. All arms use the same
fixed physics and appearance seeds within a layout. This order is specified
before any new-layout execution and uses no navigation outcome. There are eight
layout units for paired analysis, not 32 independent maze replications.

The full population requires 392GiB free artifact storage and 32GiB available
RAM at admission: a 40GiB reserve plus 11GiB per remaining episode. The resource
helper accepts only an exact ordered completed-name prefix when reducing this
allowance. It does not authenticate completion receipts or reserve OS resources;
the launcher must verify those receipts and refresh actual resources before
each case. Scientific failures remain in the population.

The factory receives only public mission coordinates from the layout source.
No graph, route or native pose is passed into the controller. It reloads through
the original corrected-model loader, checks the fixed treatment and actual state
digest, and verifies construction did not change that state. It accepts no
caller-selected model, controller implementation or navigation budget.

The current-pair intervention restricts planning-grid queries and retains
contact, localization, temporal model, residual and mission history. It is not
fully memoryless. Reactive control is a whole-method comparator with different
future-pose feasibility gates, not an isolated prediction-ranking ablation.

Source files:

- `lewm/independent_round_trip_comparison_study_development.py`
- `scripts/independent_round_trip_controller_factory_development.py`
- `lewm/tests/test_independent_round_trip_comparison_study_development.py`

Focused tests completed: 20 passed in 4.37s, process handle 63225, exit 0.
They cover the full matched roster, balanced order, resource boundaries and
invalid measurements, real controller construction with synthetic model-loader
results, fresh state, reactive model exclusion, incorrect assignments and
construction-time model mutation. These tests do not run model inference,
create a scene or establish navigation evidence.

Still required before execution: collector/auditor integration for all four
arms; complete original inventory, fit/correction and completed six-case
development-result admission; matched startup evidence; whole-population raw
audits and paired outcomes; and a frozen prospective launcher/protocol. Inspect
the queued expanded-model results and resolve correctable implementation
failures using existing development evidence before consuming these layouts.
No new-layout outcome, navigation qualification or deployment claim follows
from these assignments or source checks.
