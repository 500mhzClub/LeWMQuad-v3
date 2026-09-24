# Observed-route reactive baseline source and prefix V1

This implements a non-predictive action baseline for the requested comparisons.
It uses the same public primary RGB-D/body/gyro and auxiliary depth observer,
persistent floor/surface map, source-defined geometric route, instructed mission
coordinates, 100 ms command interval, three warm-up ticks, shared navigation
budget, ten-interval arrival dwell and outbound/return obligation. It has no
world model, future-outcome bank, online forecast residual or command-integrated
pose. RGB still contributes to observed motion estimation. This is a reactive
local action rule with geometric route planning and persistent memory; it is
not a memory-free baseline or a learned-planning advantage result.

`ReactiveObservedRouteSelector` takes the same route proposal, 0.35 m lookahead
and observed exact-goal connector helper. It turns toward the observed target
when absolute heading error exceeds 0.1 rad; otherwise it requests the original
forward primitive only if the straight current-to-target connector contains
observed floor throughout and has nominal 0.45 m clearance. When a local target
is within 0.04 m or no route exists, it uses the original ordered bounded view
headings and floor-side scan preference. Goal changes reset scan state only.

Every nonzero request requires current nominal clearance and the original
observed foot/ground contact policy applied to the actual current footprint
(zero displacement and zero relative yaw). These are measured geometry gates,
not candidate future-pose evaluations, guaranteed motion clearance or articulated
certification. The predictive controller instead evaluates learned future
surface/path forecasts, and may apply nominal reentry. These gate differences
are explicit parts of the baseline treatment: this comparison can assess whole
controller behavior, not isolate predictive ranking from every other gate.
No learned feasibility filter is reused and called non-predictive.

`ReactiveObservedRoundTripController` independently composes the same observer,
map and mission classes. It still admits all four-frame policy histories before
decisions, though no learned inference consumes the resulting tensors. Ten valid
geometry-infeasible commands permit reobservation; the eleventh latches a stop.
Sensor/history errors stop before selection. Observed arrival remains a candidate
requiring separate actual native pose/command/backtracking/visibility audit.
Eleven focused tests passed in 2.07 s, covering heading direction, no future
footprint calls, unknown/occupied/current-surface vetoes, bounded scan, retained
mission return state, ten waits, latched stop and invalid sensor/history admission.

Run `scripts/replay_go2_reactive_observed_route_prefix_v1.py` once in exclusive
`go2_reactive_observed_route_prefix_v1_attempt_001`. Bind the completed nominal
reentry native result `b48f3c79d19889c67438073a3a7305d0eafcbc5f8b7286ea014e2eb1fd711739`
and readout `0ff22595c9a49fb855212952a11f02915a6269c22db3011180c03dbfda0a20f7`,
their artifacts and frozen source/native dependencies before/after replay.
No model state is deserialized for baseline inference. Replay public packets in
order, asserting identical observer/map/mission and command outputs before the
first changed command or terminal policy. Stop before consuming any outcome of
the changed request. Absence of forecast fields is explicit. Do not compare
unexecuted alternative outcomes or claim navigation from this prefix.

One independent ordered CPU replay may run alongside the existing separately
owned waypoint native attempt after hardware assessment. Require 8 GiB available
RAM and 256 MiB above the unchanged 40 GiB artifact reserve; record concurrency
and hardware. No second native scene, GPU, training or large collection is
launched. Readout writes can conservatively count toward the live collector's
volume-consumption limit. Future native baseline collection needs its own bounded
source and raw audit; no hardware or real-time qualification is granted here.
