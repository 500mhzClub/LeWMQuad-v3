# Observed outbound and return controller

`lewm/observed_round_trip_controller_development.py` connects the explicit
coordinate mission state to the existing observed-floor, eight-step learned
planner and causal final-goal residual scoring. This is development source;
it has not executed a native round trip or demonstrated maze backtracking.

The caller supplies exactly an outbound initial-frame XY instruction, a return
XY instruction, a boolean return requirement, and an explicit global navigation
budget of 1–4,000 observation intervals. Scene walls, routes and native state
are not accepted mission fields. The maximum is a source bound, not an adopted
experiment budget. Constructor-owned coordinate copies prevent caller mutation.

Each live decision admits the current joint visual pose, exact consecutive
100 ms clock and full four-frame model history. It then admits the previous
executed-request residual, evaluates mission state and, when movement is needed,
evaluates all original candidate actions through the variable-target selector.
Only the first command interval is requested. All original surface, observed
floor contact, nominal path, full-plan contact score and causal correction rules
remain in that selector. The inherited observer still requires primary RGB-D,
fast sensors and the calibrated 45-degree auxiliary depth stream.

Both outbound and return require the original 4 cm observed region and ten
completed zero-request quiet intervals. Outbound completion changes the active
instruction to home and requests zero on the transition frame. It resets only
target-specific scan state. The same mapper, floor/surface memory, visual tracker,
learned history deque and residual estimator persist. Residual samples naturally
expire through the existing eight-tick window during a long quiet dwell; none
are carried past expiry merely to make a retention claim. No mission budget is
reset. Final completion is an observed candidate needing independent verification.

Ten consecutive valid infeasible forecasts may request zero and reobserve;
the eleventh stops. A feasible action resets that wait count. Sensor/model
failure, view exhaustion and mission deadline remain latched stops. Terminal
drain calls issue zero and do not resume tracker, map, history or residual updates.

## Checks and remaining integration

Five mission-state tests passed, covering separate dwells, deadline crossing,
dwell resets, bad clocks and owned coordinate-only instructions. Three selector
tests passed, covering the two variable coordinate substitutions, unchanged
eight-step planning and goal-specific scan reset. Six controller tests passed:
synthetic outbound/return transitions retain object identities and history,
wait/recovery limits and deadline holds are enforced, and bad pose/history/sensor
inputs latch zero stops. The first controller test run had a fixture error that
treated the map's floor dictionary as a set; after fixing the fixture, all six
passed. Synthetic controller tests use admitted synthetic joint-pose witnesses
and stub policy tensors/action selections; they do not demonstrate learned
navigation. Actual recorded sensor/model replay is the next integration check.

Native integration still needs an explicit longer-run camera/session contract,
exclusive collector, storage/resource allowances, command and raw sensor audits,
and evaluator-only verification of both arrival windows and physically traversed
return edges. The prospective four-maze source is separately documented in
`docs/go2_novel_maze_round_trip_scene_source_2026-09-08.md`. Neither component tests
nor replay substitute for those new native executions or matched baselines.
