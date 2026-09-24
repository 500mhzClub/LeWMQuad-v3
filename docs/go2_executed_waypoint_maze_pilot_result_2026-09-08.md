# Executed-waypoint native pilot: three edges, no arrival

The reused maze-0 attempt `go2_executed_waypoint_maze_pilot_v1_attempt_001`
completed with result SHA-256
`d5ba1136c969cd6591028f435a93f0e35a4dae7b9cbf217524887b459271ace8`.
Its launch SHA-256 is `f7fec194358f3821c036df413c360f355cd758ec7d1ca5c1b7466b99ceea9f4e`,
binding 1,436 sources. Session 58173 completed and worker 2345120 terminated.
Worker wall time was 2,017.955 s, with maximum RSS 5,872,214,016 bytes.

The controller stopped at decision 1155 for no phase-admissible surface/nominal
candidate, followed by ten zero drain commands. There were 1,165 complete commands,
1,166 paired observations and 59,000 physics samples. No physical/acquisition
stop or arrival occurred. Raw sensor/model/command replay, unchanged model state,
physical visibility and all measurement gates passed. The first four physical/
public observations matched the predecessor through changed command 3.

Native traversal was `[-1,0] -> [0,0] -> [0,1] -> [1,1]`, three valid open edges,
with no invalid crossings or out-of-maze positions. Path length was 4.909109277 m.
Minimum outbound-goal distance was 2.850807224 m; terminal distance 2.852544232 m,
at native initial-frame XY [2.243516183, 1.022298421] m. There was no outbound
arrival window or return. This adds zero independent layouts and zero verified
arrivals/round trips; route progress alone does not achieve the navigation goal.

Readout `go2_executed_waypoint_maze_readout_v1_attempt_001` completed with result
SHA-256 `1bb9ae412452fd68a99fa827ad304d9201f5dfdc2b7bb24f57e003390ba7799c`.
Its launch SHA-256 is `715ee54331e8feecb270dfe4f4d239fee761ad6f285e2240c23746a49f652c87`,
with 1,440 sources. Session 41463 completed. Of 1,120 complete selected waypoint
intervals, 1,066 local rankings differed from the old 800 ms utility on those same
current states. This is not an alternative executed-controller trajectory.
Mean native XY error was 7.719 mm for the original corrected-model first-step
forecast and 6.154 mm for causal scoring XY. No incomplete interval was imputed.

Selected actions: 355 left arcs, ten right arcs, 466 right turns, 152 left turns,
148 holds and 22 infeasible selections. No forward primitive was selected.
Eleven nominal-reentry left turns occurred between 1127 and 1144; none restored
an ordinary nominal-gate selection. Their native first-step XY errors ranged
from 5.286 to 14.961 mm. At the final infeasible decision all six surface checks
passed but all nominal paths failed, with current clearance 0.431087914 m to
occupied cell [38,13], below 0.45 m. The three translations were also excluded by
the existing view phase. This identifies continuing clearance/reentry limitations,
without claiming an unexecuted action would have succeeded.

Maximum observed-pose XY error was 7.280 mm. Median observation/control took
825.112 ms; command-inclusive iteration 855.995 ms; receipt-inclusive iteration
861.631 ms. Every cycle exceeded 100 ms while physics paused. No real-time or
hardware qualification is established. Reactive native comparison, other layouts,
JEPA/RGB/planning/memory attribution and verified physical return remain required.
