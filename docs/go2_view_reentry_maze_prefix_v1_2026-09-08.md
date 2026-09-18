# Translating view-recovery development prefix V1

This separately named policy follows the failed executed-waypoint native attempt
on development maze 0. At its terminal observation 1155, the view phase admitted
only hold and turns. Both turn forecasts worsened the already violated nominal
clearance; all three translating forecasts had nonworsening eight-segment
clearance, positive first-endpoint clearance witnesses, and clear first-step
surface receipts. Those are model predictions, not executed recovery outcomes.

The new selector starts from the same RoundTripMissionSelector, applies nominal
reentry with an explicit phase exception, then applies the frozen execution-time
waypoint scoring. In VIEW_ACQUISITION only, when the original selection has no
action and the current positive clearance violates the 0.45 m radius, reentry may
consider all six original primitives. Hold remains ineligible. Every eligible
nonzero command must pass the original surface check, all eight raw predicted
segments must be no worse than current clearance within the original numerical
tolerance, and its raw first endpoint must strictly improve clearance. Ranking
remains first-endpoint clearance gain minus 1.2 times the full-plan contact score.
The first command is still committed for only 100 ms.

The frozen reentry implementation performs all geometry reconstruction and
ranking. The wrapper retains the original phase allowance and nominal veto,
records the expanded recovery allowance separately, identifies any selected
translation as a phase exception, and separates original-phase eligible counts
from all recovery eligible counts. An exhausted view budget, existing valid
action, nonpositive current clearance, or currently clear nominal radius cannot
activate the new exception. Other phases use the frozen predecessor recovery.
Observation, persistent memory, mission, dwell, shared 3000-tick budget, residual
state and latched failure handling are inherited unchanged. No training occurs.

Replay reads the completed executed-waypoint attempt only after binding result
`d5ba1136c969cd6591028f435a93f0e35a4dae7b9cbf217524887b459271ace8`,
its artifacts and completed readout
`1bb9ae412452fd68a99fa827ad304d9201f5dfdc2b7bb24f57e003390ba7799c`.
It uses the same assigned corrected JEPA model and public primary/auxiliary RGB-D
and body packets, with no native pose used by the policy. For every frame through
the first changed command, causal observation, map, mission, raw forecasts and
constraint receipts must match. In a view-reentry frame, the predecessor's
transformation is inverted and then fully reconstructed with its frozen function
before the new transformation is checked exactly. Other selection receipts must
be identical. Model state and absent gradients are checked after replay.

Stop immediately at the first changed requested command or terminal decision;
do not consume its future outcome. No new native trajectory, arrival, backtrack,
clearance guarantee or navigation success can be inferred from this prefix.
This is further development on one reused maze, not an independent layout test.
The original failure and all launched source identities remain unchanged.

Launch only after focused policy and predecessor-comparison tests pass. One CPU
replay worker requires at least 8 GiB available RAM and 256 MiB artifact headroom
above the existing 40 GiB reserve. Hardware is measured before launch and after
completion. No native scene, GPU, cache retirement or model training is involved.
Sources and completed inputs are bound before and after replay; output is the
exclusive `go2_view_reentry_maze_prefix_v1_attempt_001` directory. Preserve any
partial output and terminal failure. A new native experiment is still necessary
to test the changed command and remains subject to its full resource allowance.
