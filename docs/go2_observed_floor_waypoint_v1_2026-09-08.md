# Observed floor/waypoint V1 — fixed source and replay protocol

Develop waypoint proposals from public depth and current joint visual pose.
Retain the complete verified surface-memory interface. Initialize a map frame
from the mean quiet, valid public specific-force history and projected initial
body-forward axis. Infer a fixed candidate floor height from the median of
initial measured ground-mesh points (at least 100 eligible cells). No native
attitude, body height, environment geometry, semantic floor label or goal outcome
is an input. Quiet initial acceleration, a static flat floor and uncalibrated
visual transforms remain explicit development assumptions.

Use a fixed 5-cm grid, 200 by 200 cells spanning [-5,5) m on each map axis. For
every candidate square, project its four corners on the hypothesized plane into
the current camera. Require the entire conservative image rectangle to be in
public range/FOV and composed of valid measured ground-mesh cells under the
existing normal/planarity rules. Every measured corner must also lie within
10 mm of the chosen plane. A plane alone, endpoints alone or a missing pixel
cannot establish coverage. Retain each covered square and its first witness.

Separately retain sampled surface obstacle cells for public returns between
30 mm and 650 mm above the plane, sampled every fourth pixel in each dimension.
This bounded height slab is a proposal-model assumption, not a complete robot
clearance test. Inflate occupied squares by a nominal 450-mm disk using closed
cell enclosures. Do not use occupied or unknown cells as floor-route cells.
Route on the four-neighbour observed-floor component. If the mission goal cell
is reachable, propose that route; otherwise choose the reachable frontier
minimizing goal distance plus 0.1 times path length, with stable grid tie-breaking.

The camera cannot cover floor beneath itself. Select the nearest observed-floor
entry within 1.25 m whose closed supercover connector crosses no inflated occupied
cell. Explicitly return every unknown connector cell without marking it observed.
Such a proposal has incomplete route-floor coverage and cannot be called a
validated route from the robot. If no entry exists, request another view. No
proposal grants motion or complete footprint/volume/support qualification.

Validate missing-pixel/plane/FOV rejection and an actual graph detour around an
inflated wall. Then freeze sources and replay both complete traces from native
result `eb70bb999f78f063031575b2773e2871651f095cb583e0bd6fb4b94ec84bfa4e`
under `go2_family_transition_goal_probe_v1_attempt_001`, admitted with the complete
surface-memory replay result
`d19b8254779c7aafb7ef4d0bee612a0d266da5236f1bea831f376d775c79dcfe`.
Create `go2_observed_floor_waypoint_v1_attempt_001` exclusively. Reconstruct the
original corner observer and require exact evidence equality. Retain any new
floor-map failure, original visual failure and all terminal-drain frames. Report
coverage, obstacle witnesses, waypoint/scan statuses and separate update/planning
timing. Verify every inherited source/input/artifact identity before and after.

This small CPU replay performs no native simulation, fitting, model selection,
closed-loop waypoint execution or physical backtracking. It preserves every prior
goal/contact/depth failure and grants no independent-maze or hardware claim.
