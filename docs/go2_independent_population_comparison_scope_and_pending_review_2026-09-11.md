# Independent-population comparison scope and remaining review

This is a source review of the prepared, unexecuted independent-maze population.
It changes no case, model assignment, controller, launch gate or running process.
No independent-layout sensor data or sealed material was consumed.

## Questions the existing four arms address

The fixed definition in
`lewm/independent_round_trip_comparison_study_development.py` contains eight
layouts and 32 episodes, with layout-major cyclic arm order. The three contrasts
in `lewm/independent_round_trip_population_readout_development.py` are:

| Contrast | Common elements | Actual interpretation |
| --- | --- | --- |
| Persistent full-RGB JEPA versus persistent full-RGB supervised rollout | Controller class, persistent planning map, assigned primary seed, completed expanded-data training/correction lineage | Training-objective comparison for these fixed models; no advantage has been established. |
| Persistent JEPA versus current-pair JEPA | Exact JEPA model, residual selector chain, localization/contact history, learned temporal history and mission | Effect of accumulated planning cells; the current-pair controller retains other memory and is not fully memoryless. |
| Persistent JEPA versus reactive | Dual-camera visual motion, measured floor registration, measured transport map and mission classes | Whole predictive-planning method versus an observed-route rule. Future feasibility, learned scoring and residual use differ. This does not isolate prediction-based ranking. |

The reactive implementation explicitly reports the absence of learned forecasts,
predicted surface/path gates and learned residuals. It is one reactive,
nonpredictive treatment, not two distinct baseline arms. A narrower claim about
ranking alone would require another prospectively defined intervention and
actual matched execution. It should not be inferred from this method contrast.

The current-pair implementation exposes a current paired planning view while
retaining contact, tracking, floor-anchor, learned temporal, residual, selector
scan and settling state. An eventual advantage in the persistent arm would
support the specified planning-grid contrast, not a claim that all memory was
removed in its comparator.

These conclusions follow from the actual controller factory and implementations:
`scripts/independent_round_trip_adapter_controller_factory_development.py`,
`lewm/reactive_floor_transport_controller_development.py`, and
`lewm/residual_current_observation_planning_controller_development.py`.
They are source-scope conclusions, not completed navigation evidence.

## Review dependency that needs updating before population launch

`scripts/independent_round_trip_final_admission_development.py` currently binds
the original five diagnostic outcomes: frontier, hold, contact, tracking and
extended budget. Its final review schema is explicitly a five-diagnostic review
and permits only the original fixed 32-case definition. It does not yet include
the subsequently queued sustained-turn, contact-plus-flow and chained-anchor
native outcomes in that review evidence.

Those three outcomes remain pending. Before spending the independent layout
population, extend the review evidence through a checked successor that binds
their completed waiter/native results and retains all negative outcomes. If
that review calls for policy, model-arm or budget changes, implement and check a
new population definition before any independent-layout execution. The existing
five-diagnostic review must not be represented as reviewing all eight diagnostics.

The no-RGB direct model's single development outbound arrival is not an assigned
arm of the current four-arm population. It is development evidence to consider
in that final review, not authority to silently substitute a model in an arm or
select an arm using independent-layout outcomes.

## Work currently resolving the dependencies

The extended-budget launcher and the three downstream waiters were confirmed
live during this review. The separate density-routed controller timing replay
is also live. Its completed 854-observation registration result shows an 18.37%
component speed reduction with identical registration outputs and states;
complete-controller timing remains pending. No queued navigation controller has
adopted that performance candidate.

The next substantive decisions depend on completed native outcomes and the full
timing replay. The independent study, verified round trips, causal outcome
comparisons, 100 ms timing and bounded real-platform evidence remain outstanding.
