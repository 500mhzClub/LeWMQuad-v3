# Provisional return hold selection diagnosis

This is bounded inspection of completed decision receipts in the still-running
go2_measured_floor_transport_maze_pilot_v1_attempt_001, case
full_jepa_novel_maze_00. It is not a completed native audit or an alternative
command experiment. The current run and the fixed independent-layout cohort
remain unchanged. Original failures remain preserved.

Inspection session16462 completed successfully. It captured a count of2891
complete timing rows, then read at most that many gzip decision lines, parsing
only ticks2410 and2890. It did not read an unclosed gzip EOF or use native
positions to choose commands. Session20080 was an earlier, less compact
inspection of the same selection masks through2869.

Both inspected decisions have phaseRETURN, selected actionhold, no terminal or
controller failure, no mission-required hold, and no infeasible-action wait.
Observed distance home changed from3.8194315376076102m at2410 to
3.8188673679927185m at2890: only0.00056417m net change across48 simulated
seconds. These endpoints do not establish every intervening action or actual
native displacement. All six actions are phase-allowed. Five are admissible
under the full predicted-path checks at each inspected decision. All six pass
the measured-surface conflict checks at2410 and2890.

At2410:

| Action | Final utility (m) | Full-path admissible | Minimum observed-cell distance (m) |
| --- | ---: | --- | ---: |
| hold | -0.001131481584354003 | yes | 0.458658530655482 |
| forward | -0.002904553300068991 | yes | 0.4505327434003355 |
| left_arc | 0.00409197623675799 | no | 0.44199864633368763 |
| right_arc | -0.0016807403378531031 | yes | 0.459725732005587 |
| left_turn | -0.0015084037132607974 | yes | 0.45778954949733297 |
| right_turn | -0.0025325800437166713 | yes | 0.45968201409164333 |

The left arc fails the nominal0.45m-radius checks on predicted segments ending
at600,700 and800ms. Its first100ms segment passes. Therefore its higher
unmasked utility cannot make it the selected action. Forward remains admissible
across all eight segments, narrowly at the last segment.

The actual source lewm/executed_waypoint_score_development.py computes
100ms causal residual-corrected waypoint distance-plus-alignment improvement,
minus1.2m times the800ms cumulative contact score, then chooses the highest
utility among phase/surface/full-path-admissible actions. At2410, forward's
distance benefit is0.012198779320863218m, alignment benefit
-0.000023881979325812702m, and contact score0.012566208868005349.
Hold's corresponding values are0.000017776369902466893m,
0.000083611621893942m and0.001027391313458695. Those numbers reproduce the
recorded ranking: hold has the largest admissible utility. The contact score
is explicitly uncalibrated; it is not an established physical risk probability.

At2890 the same ranking mechanism persists: hold utility-0.001146731251508482,
forward-0.0029269531300988427, and excluded left arc+0.004043700446269146.
The left arc fails segments ending500–800ms; its minimum distance is
0.4415008982948727m. Forward still passes, with minimum0.4500849797664651m.

The target inspected at2410 was approximately0.385m ahead, so a reached-waypoint
explanation is unsupported. The evidence explains the selected hold at these
two observations through the combination of long-horizon path exclusion and
short-progress/full-horizon-contact scoring. It does not prove the predicted
contact scores or paths accurate, nor that forward or an arc would succeed.

Next: finish the unchanged native collection and full raw audit; bind any
subsequent full-interval diagnosis to completed artifact identities. Preserve
the already fixed independent-layout and reactive experiments. A future
scoring successor would require a separately defined policy, causal prefix
comparison and fresh physical outcomes after its first changed command.
Do not weaken clearance or suppress failures based on this inspection.
