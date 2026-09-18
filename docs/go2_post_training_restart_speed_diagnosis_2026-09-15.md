# JEPA return-phase speed stops: command-transition evidence

The two speed-stop failures are retained in the four-controller comparison.
This analysis uses saved physics only; it changes no controller or guard.

Both stops follow forward, left arc, left turn, hold, then forward. The arc,
turn and hold each last 0.4 s; the final requested/applied forward speed is
0.2 m/s. The full-body 0.3-m/s limit is crossed 0.492 s into the restart on
layout 2 and 0.412 s into it on layout 3. The final commands were ordinary
`distance_and_heading_progress_minus_contact` selections with clear candidates;
neither reserve-recovery mode was selected, and the memory filter did not
change the selected action. This is not a forced recovery sequence.

The aligned recorded traces are in
`go2_post_training_restart_speed_stops_2026-09-15.png` and `.svg`; both plots
were visually inspected. Each failed root also retains its exact terminal
command segments in `terminal_command_transition_diagnostic_v1.json`.

Across the ten evaluated assignments, the forward-command exposure summary
is `go2_post_training_forward_speed_exposure_2026-09-15.json`. It selects
navigation-phase 2-ms samples requesting `[0.2, 0, 0]` (absolute matching
tolerance 1e-7). In the four JEPA trajectories, median body speed under that
request is 0.167–0.176 m/s and the 99th percentile 0.243–0.246 m/s. Only about
0.003–0.023% of forward samples exceed 0.29 m/s. Thus the two guard crossings
are unusual transients in the observed population, not persistent overspeed.
The small subset after at least one uninterrupted second of forward request
never exceeds 0.255 m/s in any of these ten recordings. That subset has limited
exposure and does not establish a bound.

`go2_post_training_restart_pattern_matches_2026-09-15.json` searches those
same ten recordings for the exact three 0.4-s arc/turn/hold segments followed
by forward. Seven matches occur, all in the four JEPA recordings. Five do not
stop during the observed ensuing forward segment; their peak speeds range
from 0.265 to 0.294 m/s. Four have a complete 0.8-s forward window; one switches
command after 0.4 s. The other two matches are the failed restarts, truncated
by their guard crossings. The pattern was chosen after observing failures,
and trajectories, exposure and preceding robot state differ. These counts
are descriptive and do not establish causation or a method-level risk rate.

A command-transition/gait-state interaction is now a specific hypothesis for
the return failures. A controlled physical replay or a separately declared
restart treatment could test it after the fixed comparison. Merely raising
the speed limit would not test the hypothesis. The current XY/yaw/contact
forecast does not certify instantaneous full 3-D body speed; small XY forecast
error cannot establish compliance with this guard. The live XY-source ablation
remains a distinct test of learned prediction, not a remedy already shown to
prevent these stops.
