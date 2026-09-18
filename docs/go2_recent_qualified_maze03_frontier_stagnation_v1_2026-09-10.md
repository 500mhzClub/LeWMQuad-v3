# Completed maze3 frontier stagnation diagnosis V1

Read only the exact completed episode31 result
`330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723` and its
bound collection, command tape, decision stream, native trace and audit receipts.
Authenticate the consumed files and original source map before and after the
diagnosis. Do not create a native scene, load/refit a model, rescore a policy or
infer outcomes for commands not executed. Preserve the original six-model queue.

The initial native trajectory inspection found extensive rotation with less
than one cell of displacement. Sampled saved decisions remain in WAYPOINT mode
while targeting a single frontier cell nearby. Examine the complete 2,805-row
stream to determine the prevalence of each mode, frontier status, exact target
and selected action. Count forward-action eligibility from the original saved
phase, surface and eight-step nominal-path gates; do not confuse a lower saved
utility with infeasibility.

Save a compact summary for every original decision, fixed snapshots at frames
3, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 2793 and 2794, the ten longest
unchanged-target runs, waypoint distance distributions, and native rotation and
translation measures. A one-cell route is a descriptive category, not a new
runtime arrival threshold. Both 2ms trace path length and command-endpoint path
length include gait/body oscillation and are not claims of useful route progress.
Native pose is used only by this offline diagnosis, never passed to a controller.

Source inspection: `continuous_connector_waypoint_development.propose` picks
the reachable frontier minimizing goal distance plus graph distance. It has
no reached/retired frontier state. `MissionTargetWaypointSelector.choose`
enters WAYPOINT whenever route_cells is nonempty; view-scan budgeting applies
to the empty-route branch. The diagnosis will quantify the observed pattern
without claiming an unexecuted frontier policy would solve navigation.

Keep this older corrected model/controller result distinct from the queued
expanded-data residual-continuation comparison. Use the completed evidence to
design a prospective correction on existing development layouts before spending
the new independent-layout inventory. Passing audits and reduced tracking
failures are not navigation, real-time or hardware qualification.
