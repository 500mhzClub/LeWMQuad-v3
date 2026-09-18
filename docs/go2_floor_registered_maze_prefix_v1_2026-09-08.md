# Floor-registered maze prefix V1

One CPU process reconstructs the complete confirmed-floor predecessor and the
new floor-registered controller from identical public packets, using the same
frozen learned model and initial-frame mission. This is a development prefix
inspection on reused maze 0, with no native execution or training.

The required inputs are the completed confirmed-floor native result and its
completed readout, supplied by exact SHA-256 arguments to
`scripts/replay_go2_floor_registered_maze_prefix_v1.py`. Collection fragments
are insufficient. Existing source and artifact bindings are verified before and
after. Output is exclusive:
`go2_floor_registered_maze_prefix_v1_attempt_001`. No retry or overwrite occurs.

Every predecessor decision must reconstruct exactly, including original raw
sensor processing, maps, forecasts, mission state and command. The candidate
must preserve the entire original visual witness and match raw forecasts when
both controllers plan. Its maps, contact checks, residuals and mission distances
may differ because they use the declared corrected pose. The new pose schema
must not be relabeled as an original joint-fit witness.

The replay stops immediately at the first changed requested command or terminal
decision, before reading any resulting unexecuted transition. The final changed
command is a proposal; the predecessor's following outcome cannot be attached
to it. Candidate admission failure is a terminal diagnostic failure, not a
successful policy intervention. No physical improvement or visibility pass is
inferred from replay. All original failures remain intact.

Use one CPU worker, two controller instances, at least 16 GiB available RAM,
and 512 MiB artifact headroom over the existing 40 GiB reserve. The native
scene is not instantiated. Assess current resources before executing; the
existing live native audit must retain its frozen sources and process.

Focused tests cover measured-plane correction, shared consumer pose and
evidence tampering, exact predecessor/forecast comparison, and intervention
stop detection. The complete fresh native sensor/command audit, independent
layout runs, matched comparisons, realistic timing and hardware evidence remain
required for the full navigation goal.
