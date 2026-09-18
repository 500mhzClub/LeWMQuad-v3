# Forecasting the delay before a command starts

The development planner now has an explicit candidate bank with two committed
100 ms commands, one candidate command, and five zero-command intervals. The
caller must obtain the first two commands from its execution ledger; measured
past commands do not establish that future prefix. Scoring uses predicted
progress from 200 to 300 ms and contact by 300 ms, with coefficient 1.2.
All eight model horizons remain recorded. A scheduled command must be prepared
by observation time plus 200 ms; it expires at plus 300 ms. A missed start is
discarded rather than retimed.

`route_from_current_pose` in `lewm/delayed_action_planning_development.py`
separately takes the latest measured pose when querying the older routing
snapshot. Its output retains both timestamps and map age. Three focused tests
pass (2.01 s), covering the exact known command prefix/tail, delayed scoring,
missed deadlines, expiry and veto behavior.

The three existing seed-2026091001 full-input models were probed on ten actual
causal sensor histories from the verified independent layout-0 JEPA run. The
two zero commands in this probe are a proposed counterfactual protocol, not a
claim about the recorded future. Direct, supervised rollout and JEPA all
returned complete forecasts, had exactly zero candidate-dependent spread over
the shared first two command intervals, and selected forward in all ten
observations. Median model inference times were 7.20, 6.62 and 6.62 ms,
respectively. All model parameters remained unchanged. This establishes input
compatibility and common-prefix consistency on these histories; it does not
establish forecast accuracy or physical performance with delayed execution.

Result: `docs/go2_delayed_action_model_probe_2026-09-13.json` (session 65371,
exit zero). It retains all forecasts and candidate scores.

`lewm/fresh_obstacle_dispatch_development.py` adds a separate current paired
depth observation of sampled obstacle cells and a 0.45 m nominal-disk veto.
It uses the current measured pose with the map's fixed gravity/floor reference.
Missing, future or more-than-200-ms-old observations request zero, as do expired
plans and observed nominal intersections. Two focused tests pass (1.76 s).
The disk connector uses requested speed over the remaining command interval;
it is explicitly not a measured future pose, whole-body collision check,
ground-support test, or clearance certificate. Absent returns do not prove
that unobserved space is free.

This is not yet a continuous native controller. Integration must maintain the
actual command ledger, run acquisition/pose, map updates and planning without
blocking actuation, and use already-completed fresh-observation results at
dispatch time. The 200 ms allowance includes the actual upstream computation
and delivery delay; the 7 ms model-only probe does not establish that budget.
The next experiment should connect those components on a paced recorded stream,
then run a short prospective simulation with continuously advancing physics
and measure actual contacts, visibility failures, deadline misses and progress.
