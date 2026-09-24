# Auxiliary depth native probe: valid sensing, zero verified arrivals

Both fixed corrected models completed fresh closed-loop missions and exact raw
controller replays. All primary and auxiliary measurement gates passed. Neither
model reached the goal. The auxiliary sensor changes the observed map and
executed route; it has not established improved goal-reaching.

| Model | Terminal decision tick | Commands including ten zero drain ticks | Minimum / terminal goal distance |
| --- | ---: | ---: | ---: |
| Full JEPA, seed 2026091001 | 38 | 48 | 1.111330307 m |
| Full direct, seed 2026091001 | 35 | 45 | 1.098843367 m |

Both stopped with `NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`.
There was no physical or acquisition stop. At each stopping decision all six
candidates passed the articulated surface check, including both primary and
auxiliary evidence. Every first 100-ms nominal path instead failed against
observed occupied cell [11, -2]. JEPA candidate clearances range from
0.444913951 to 0.449593591 m; direct candidate clearances range from
0.442825501 to 0.444808123 m, below the unchanged 0.45-m criterion. This terminal
failure differs from the preceding JEPA run's uncovered-front-foot veto.

All 49 JEPA and 46 direct paired frames passed both-camera audits. The auxiliary
camera had no robot-occluded pixels. Maximum accepted visual pose XY error was
1.395703 mm for both. All recorded decisions reproduced exactly with unchanged
assigned model state. JEPA chose 23 right turns, eight right arcs, two left arcs
and two holds before its no-candidate selection. Direct chose 23 right turns,
eight right arcs and one hold before stopping.

Relative to each primary-only predecessor, the first executed command difference
is tick 17. The 18-frame causal prefix has exact native physics, policy and gyro
arrays; primary RGB and map evidence differ with the new renderer and sensor.
The new JEPA/direct pair first differs at command 20. Through the 21-frame causal
prefix, physics, policy, gyro, primary RGB, observer/map evidence and auxiliary
depth/valid arrays are exact. Subsequent outcomes are separate actual trajectories.

The two-process phase took 110.813528 seconds and bound 1,239 source paths. The
readout bound 1,242. Median complete command-iteration times were 830.905 ms for
JEPA and 836.441 ms for direct; all 48/45 iterations exceeded 100 ms. Simulation
was paused during computation, so these results do not establish real-time use.

| Artifact | SHA-256 |
| --- | --- |
| Probe launch.json | c85edf4b7a5c343d0d2eaae02f0e3d898c88952d125b8f025237e3efc4338496 |
| Probe result.json | f8bcad37fdf1623c30eb6782c4514b483473e8c6a4f3e404d77ab71cc0f80f21 |
| Readout launch.json | 8bc8f71a1474e62e17e045ed6667c2af613a7297e91ebda730d773d309562157 |
| Readout result.json | db1440338a68b8d8c2a04a7c11fa7b43e6bccd79ee428f607202936e6ddcfacb |

Roots are `go2_auxiliary_depth_goal_probe_v1_attempt_001` and
`go2_auxiliary_depth_goal_readout_v1_attempt_001` under the guarded development
artifact base. All 26 prelaunch focused checks and two readout tests passed.

Next, diagnose executed prediction and braking prefixes against the retained
native trajectory, including the real zero-command drain. Identify whether
small nominal margins, prediction errors or the stopping forecast caused the
loss of feasible actions before choosing a prospective planning change. Preserve
the observed wall and all original outcomes. Independent novel mazes, actual
backtracking, matched reactive/nonpredictive and planning/memory comparisons,
realistic timing and bounded hardware evidence remain outstanding.
