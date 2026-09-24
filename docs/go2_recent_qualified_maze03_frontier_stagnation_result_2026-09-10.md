# Maze3's repeated nearby-frontier target explains the next intervention

The complete episode31 decision stream shows a persistent local frontier target
without a transition to another exploration target or a bounded view phase.
This is a concrete navigation failure before the later floor-registration
terminal. It is not evidence that a replacement policy already works.

Diagnosis process 26042 exited 0. Artifact root:
`go2_recent_qualified_maze03_frontier_stagnation_v1_attempt_001` under BASE.

- Result SHA-256:
  `bfff88c111488032383d5e3b73efc8fb75f5e09be6ee35c2d09ed88ef28f22dc`.
- Launch SHA-256:
  `99f5acbf53f4000876914fa7990127ca65ebb4307263f47bd451fdcd60698c0a`.
- Complete 2,805-frame compact stream SHA-256:
  `29874f830e8f52f2f464abc65fa8034739d483b82d4342f93dfaf12f8ec01814`.

All 1,846 source bindings and nine consumed original artifact bindings were
checked before and after analysis. The nine consumed bindings include the exact
completed result, launch, raw audit, prefix, worker terminal, collection, native
physics trace, decision stream and tape as enumerated by the original result.
The prior turn authenticated the
full 16,870-file original completion. This diagnosis did not repeat raw model
inference or read every original raster. Final check 27430 exited 0 and verified
the result, both outputs, all sources and all 1,890 original queued sources.
Focused arithmetic/gate tests: handle 40136, exit 0, three passed in 1.70s.

All 2,791 active selections were WAYPOINT selections with
`OBSERVED_FLOOR_ROUTE_TO_FRONTIER`; none entered VIEW_ACQUISITION. The waypoint
`[0.475, -0.075]` in the observed floor-map coordinates was used 2,654 times,
including an uninterrupted run from frame 142 through 2793: 2,652 observations
over 265.1 simulated seconds between their timestamps. That run made 18.68 net
yaw revolutions with no cell crossing.

There were 2,336 single-cell routes. Their median target distance was 22.29mm,
p95 34.65mm and maximum 51.48mm. Over all active routes, the median target
distance was 23.94mm. The mission goal remained unreached; these are
distances to a local observed frontier waypoint, not mission progress.

The saved forward-action gates were phase-allowed and surface-clear on all
2,791 selection frames, and eight-step nominal-path clear on 2,051. A turn was
selected on 1,723 frames where forward passed all three gates. Thus the long
rotation cannot be attributed solely to a prohibition on forward commands.
The saved utilities and their distance/alignment/contact components are retained
per frame; no counterfactual policy was rescored and no forward outcome inferred.

Completed commands: 2,278 left turns, 89 right turns, 406 left arcs, two right
arcs and 29 zero commands (including warmup/hold/drain). No straight-forward
command was executed. Native net displacement from the first observation was
0.4555m; maximum displacement 0.5220m. Net yaw rotation was 18.54 revolutions.
Sampled XY path length was 9.306m at 2ms and 8.334m at command endpoints; both
include gait/body oscillation and are not useful route-distance claims.
Native cell crossings and observed arrivals were zero.

Source explanation: `continuous_connector_waypoint_development.propose` uses
current reachable cells and goal distance to select a frontier. It has no
reached/retired frontier state. `MissionTargetWaypointSelector.choose` stays
in WAYPOINT whenever route_cells is nonempty, even for a nearby singleton.
The view scan's exhaustion logic belongs to the empty-route branch. This
explains why its view budget does not bound the recorded repeated target.
Learned utility errors and persistent-map details may also matter; this
diagnosis does not establish a sole cause or a successful remedy.

The first terminal remains frame 2794: floor registration exceeds the fixed
development correction gates. Three causal-history warmups and eleven terminal
observations account for the remaining frames. All original failures remain
preserved, including the earlier registration failures recovered by this run's
qualified-reference intervention.

Next intervention: add an explicit observation-based transition for reached or
exhausted local frontiers, with bounded observation/revisit behavior and selection
of another reachable frontier where justified. Keep model, mission, sensor,
clearance and action constraints fixed for a prospective prefix comparison on
existing development evidence. Measure the first actual changed command, then
require a fresh physical continuation before claiming navigation improvement.
Do not spend the independent-layout population to tune this behavior.

The expanded-model six-case child 2653445 and original waiter 2641948 remained
live at the final process check. Input admission had reported successful contact
and recent-qualified predecessor verifiers; the first native scene had not yet
started. Their controller and source identities remain unchanged. Its results
must still be reviewed before preparing an independent population launch.

Overall: 31 completed audited native episodes, zero verified round trips.
This diagnostic used no new native scene, neural inference, fitting, independent
layout outcomes or hardware execution.
