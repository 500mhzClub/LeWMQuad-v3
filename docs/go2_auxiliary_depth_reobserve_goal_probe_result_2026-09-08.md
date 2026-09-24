# Native bounded reobservation: two recoveries, zero arrivals

Both fresh native missions completed and passed all raw sensor, controller,
actuator and measurement audits. Verified arrivals remain 0/2. JEPA recovered
an originally admissible right arc at tick 39 after one wait, then an admissible
left turn at tick 64 after waits at ticks 62 and 63. These commands were executed
prospectively. Direct did not recover during its ten waits.

| Model | Terminal tick | Commands including terminal drain | Minimum goal distance | Terminal goal distance |
| --- | ---: | ---: | ---: | ---: |
| Full JEPA, seed 2026091001 | 99 | 109 | 1.110516819 m | 1.131129038 m |
| Full direct, seed 2026091001 | 45 | 55 | 1.098012382 m | 1.098012382 m |

JEPA then waited at ticks 89 through 98 and terminated at 99. Every candidate's
first and eight-step nominal checks passed during that final wait. All candidates
were surface-vetoed by auxiliary floor returns intersecting the front-left foot,
without a complete measured floor footprint witness. There was no auxiliary
other/unknown intersection for that foot. Primary surface checks passed. The
same floor-only pattern appeared during the temporary wait at 62 and 63.
This motivates a read-only check of combined retained camera coverage, while
preserving the full foot and every unknown return.

Direct retained its no-feasible-action stop after waits at 35 through 44. Both
missions completed the separate ten-command terminal drain, with no physical or
acquisition stop. All 110 JEPA and 56 direct paired-camera frames passed. Maximum
accepted visual XY errors were 1.395703 and 1.571150 mm.

Relative to its predecessor, JEPA changes terminal policy at 38 and its executed
command at 39; the 39-observation causal prefix is exact for native arrays,
primary RGB, observer/map evidence and forecasts. Direct changes terminal policy
at 35; the 36-observation prefix is likewise exact. Its first command-tape
difference is the predecessor tape ending at 45. The new JEPA/direct pair first
changes command at 20; its 21-observation prefix has exact native/public sensing,
observer/map and auxiliary depth/valid arrays, with different model forecasts.

The two-process run took 204.998918 seconds and bound 1,258 sources. The readout
bound 1,261. Median complete command iterations were 845.429 ms (JEPA) and
836.343 ms (direct); all 109/55 exceeded 100 ms. Physics remained paused during
computation. Six state-policy tests, four native-scope tests and one readout
boundary test passed before the corresponding launches.

| Artifact | SHA-256 |
| --- | --- |
| Native launch.json | 5164170b485e6a5fcda09788fe00bdef5a0dac3ec8c2a59470c6f27e6bff72fe |
| Native result.json | 3675f09b90770ed9345154b8fe926c66fd6029e9c7bc7a6f2d53cba0c6861193 |
| Readout launch.json | 09617973c10141f9677559ea44d864886cd533d817aeec6f1798b3a7cc9c9d8b |
| Readout result.json | 2fe50e68c57ebec738daebfc70f589752fda1dc7c43de2dfdc6f93a10efd9e8e |

Roots: `go2_auxiliary_depth_reobserve_goal_probe_v1_attempt_001` and
`go2_auxiliary_depth_reobserve_goal_readout_v1_attempt_001` under the guarded
development base. Actual temporary recovery does not establish goal-reaching,
maze exploration, physical backtracking, independent-layout generalization,
matched baseline advantages, realistic timing or hardware readiness.
