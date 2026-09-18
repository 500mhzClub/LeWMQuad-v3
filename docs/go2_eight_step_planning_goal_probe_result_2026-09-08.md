# Eight-step planning native result

Both fixed models failed to reach the goal. Full-path planning changed their
actions and trajectories, but both eventually stopped with no candidate that
satisfied the unchanged surface and nominal-clearance constraints.

| Outcome | Full JEPA | Full direct |
| --- | ---: | ---: |
| Verified arrivals | 0 | 0 |
| Terminal tick | 40 | 48 |
| Command ticks / RGB-D frames | 50 / 51 | 58 / 59 |
| Physics samples | 3,250 | 3,650 |
| Minimum actual goal distance, m | 1.1178301085794478 | 1.114311214284217 |
| Terminal actual goal distance, m | 1.1178301085794478 | 1.1219808479854698 |
| Decisions changed by eight-step planning | 16 | 20 |
| Candidate forecasts vetoed by later nominal segments | 63 | 103 |
| Complete-iteration median, ms | 526.332319 | 516.909454 |
| Complete-iteration maximum, ms | 564.652717 | 714.322824 |

Both terminals were
`NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`, followed by
ten zero commands. There was no physical or acquisition stop. JEPA selected
28 right turns, four right arcs and five holds; direct selected 24 right turns,
ten right arcs, nine holds and two left arcs. Each then had one no-action
selection. Final native XY displacements were
[0.08290681484413531, -0.040582845206630816] m and
[0.08407620363374303, -0.11642638854535788] m, respectively.

Every raw sensor/model/command replay passed with unchanged models. Both strict
physical-visibility checks passed with no hard measurement failures. Maximum
observed XY pose error remained 0.0021068214643736064 m for each case.

The JEPA/predecessor pair shared 24 exact observation frames and identical model
forecasts through the first differing command at tick 23. The direct pair
shared 20 frames and identical forecasts through its first command difference
at tick 19. Raw physics, policy histories, fast gyro, RGB and observer/memory
prefixes were exact. The two new cases shared 20 frames before their command
difference at tick 19; different models were not required to predict equally.
These checks support attribution of the same-model command changes to planning,
without claiming that the changed trajectories were successful.

The full-path guard caused earlier turns instead of continuing several arcs,
as intended. It did not provide a terminal-viability or model-error bound, and
both runs still exhausted their feasible candidates. All original first-step
surface checks and the 0.45-m nominal threshold were preserved. The separate
mapping optimization was not used.

The two-process phase took 96.744745 seconds, bound 1,153 source paths and 506
artifacts, and reached maximum worker RSS 2,300,370,944 bytes. Preflight measured
82,524,274,688 bytes available RAM and 69,451,743,232 bytes free artifact storage.
The readout bound 1,156 source paths. All 50/58 complete command iterations
exceeded 100 ms while physics was paused during compute. Acquisition medians
were 124.348037/122.584006 ms. There is no real-time or hardware qualification.

Artifact roots:
`go2_eight_step_planning_goal_probe_v1_attempt_001` and
`go2_eight_step_planning_goal_readout_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| Predecessor probe result | `f484a0c3b95cb2f3d0868dca1f1f40d11ba0d29c42e12cc9604061870a0d5608` |
| Probe launch | `140c80b95df74bd61e41c61bacec37efd38a4bbecf27582e97d2c2bde132f548` |
| Probe result | `a42774f8a7ebd98a46dcf8c6a4c90c49e8551a6bcb0ccd3031810abd039b53c5` |
| Readout launch | `8efd634d2b87bc3a3f2e006fc108e52f6b12c438d57389bc5cb017f44fd0f6c2` |
| Readout result | `9b80085ac93a16e7793f75cdae09971b7e4f6b4023167b25817cfecd8ee28660` |

There remain zero verified arrivals and zero independent novel maze missions.
The full navigation goal, matched baselines, physical backtracking, independent
layout evaluation, realistic sensing/timing and bounded hardware evidence are
not complete.
