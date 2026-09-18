# Active-view learned waypoint native result — 2026-09-08

The new controller acquired a side view and executed waypoint-directed commands
on one layout, but reached zero of two mission goals. The mirrored case stayed
in view acquisition and mostly held still. Neither case contacted an obstacle;
both passed complete raw replay and all recorded strict/hard depth checks. These
are failed known-layout integration runs, not independent-maze navigation evidence.

The [frozen protocol](go2_active_view_goal_probe_v1_2026-09-08.md) integrated the
existing floor/obstacle map with learned view acquisition and waypoint selection.
The original final full-JEPA model, six actions, 1.2-m mission goal, five-tick
commitment, 240-tick navigation limit and terminal/arrival/native audit rules
were retained. Scanning restricted choices to hold and in-place turns, using
predicted half-second heading improvement, drift and contact cost. Available floor
routes supplied intermediate targets for the original learned distance/contact
utility. Persistent surface conflicts still vetoed candidates. No intermediate
view or waypoint counted as mission arrival. No parameters changed during the run.

| Case | Terminal outcome | Terminal goal distance | Closest goal distance | Model forecasts |
|---|---|---:|---:|---:|
| `active_view_family_episode_052` | Visual continuity failure | 0.973902 m | 0.973902 m | 46 |
| `active_view_family_episode_039` | Navigation tick budget exhausted | 1.185256 m | 1.184504 m | 48 |

Case 052 began scanning at tick 3 and entered waypoint mode at tick 23, initially
targeting map point (0.275, 0.875) m. It returned to scanning at tick 108, briefly
re-entered waypoint mode at tick 113, then scanned again from tick 133. It completed
105 nonzero waypoint-mode command intervals and 124 nonzero scan intervals. The
waypoint commands include turns; this is not a claim of 105 forward intervals.
Across all phases it selected seven left arcs, 13 left turns and 26 right turns.
Native terminal displacement in the initial body frame was (0.296776, 0.364241) m.

At tick 232 the visual observer exhausted its bounded measured bridge without
an accepted retained-anchor observation. The eight attempted retained references
(220, 217, 215, 208, 200, 192, 184, 176) rejected with insufficient rigid-pose
matches. The floor map latched that unavailable pose, and the controller completed
ten zero-command drain intervals. The maximum accepted-pose XY error was 4.402 mm;
this does not supply a pose after failure or a calibrated physical error bound.

Case 039 never entered waypoint mode. It selected three right turns, one left
turn and 44 holds. Of its 240 completed scan command intervals, only 20 were
nonzero. Its measured map heading ranged from -0.691404 to 0.000603 rad, with the
first scan target at -0.785398 rad. It completed the original budget and ten zero
drain intervals. Native terminal displacement was (0.014746, 0.001969) m; maximum
accepted-pose XY error was 2.532 mm.

The recorded scan utilities expose a concrete forecast failure. At tick 238,
hold had utility 0.031709 m, versus 0.009983 for left turn and -0.029916 for right
turn; none of those three candidates had a surface intersection. The hold forecast
predicted half-second yaw change -0.219449 rad and XY displacement
(0.022349, 0.051292) m. The five fully completed zero-command intervals actually
changed yaw by -0.001831 rad and XY by (0.00007771, 0.00002325) m. This post-outcome
check used the hash-verified physics trace from samples 12649 through 12899,
with translation and rotation expressed relative to sample 12649. It evaluates
the executed hold only, not the unexecuted alternatives. Rewarding predicted
heading improvement consequently rewarded an almost stationary actual action.

Inspection of the frozen training-target derivation confirms that its relative
rotation is formed against each current window's native starting rotation, not
the episode's initial rotation. This source check does not establish why the
model generalized poorly to the switched-action scan/hold state. Such states
are not equivalent to the original fixed-action remaining-plan training windows.

The next model diagnostic should compare all six already fitted models on these
same executed half-second transitions, with identical past observations and no
counterfactual outcome claims. That can distinguish a JEPA-specific error from
insufficient switched-action training support before changing costs or choosing
another checkpoint. View-target tracking should also be assessed against a
declared measured-heading baseline. Long-horizon visual continuity and route
recovery remain separate unresolved requirements; preserve the failed attempt.

Five focused tests passed in 1.69 s, covering learned left/right scan ranking,
exclusion of forward arcs during scanning, conflict precedence, sensor-failure
latching, unchanged mission goals and exact inherited native-goal/actuator audit
functions. Both native cases then passed full sensor-to-observer-to-map-to-model-
to-command replay, with unchanged model state. All 497 camera frames and 26,250
physics samples were retained. No contact, strict visibility failure or hard
depth-measurement failure was recorded. This does not repair older failures on
different trajectories or establish general sensor qualification.

| Case | Full iteration median / maximum (ms) | Over 100 ms |
|---|---:|---:|
| 052 | 292.976 / 566.219 | 242 / 242 |
| 039 | 310.725 / 541.426 | 253 / 253 |

Physics remained paused during computation. Native execution and auditing took
462.141 s. Before launch the machine had 16 physical / 32 logical CPUs, 0.2% CPU
use, 82,576,023,552 available RAM bytes, idle GPUs and 96,028,516,352 artifact bytes
free, with no substantial competing Python job. Fresh workers ran serially and
resources were monitored. The added map/planning work still exceeds the control
period and grants no real-time or hardware claim.

Artifacts reside under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/`.
Native root: `go2_active_view_goal_probe_v1_attempt_001` (910 frozen sources;
2,054 bound artifacts totaling 865,207,788 bytes). Readout root:
`go2_active_view_goal_readout_v1_attempt_001` (912 frozen sources).

| Identity | SHA-256 |
|---|---|
| Native launch | `a28c0cad6ddfd7062c6bc56d60d35a1525ef3a80d659d8c8cc838df937ff4652` |
| Native result | `bdd1b21cec0d02413f5833f5ff2a27bfde9a95af72c5aa8c03056f593c5db3bb` |
| Readout launch | `e4c7c5ba347448599d2d068cb12fa3db4ca82e563a9717c34d80a92677e947b2` |
| Readout result | `83bdeaa21939bb9cc2c2a3cec351bf5445e2f4eb0c1b8225d195c039f01cf840` |
| Fixed full-JEPA snapshot | `bcb8874e2adf89053463206267a4ccb90380909c324e303734a59b038f5b1821` |

All source, native/input, URDF and artifact bindings were verified before and
after execution and [readout](go2_active_view_goal_readout_v1_2026-09-08.md).
No case was resumed, retried or promoted. The full objective—independent-maze
goal-reaching, useful exploration/memory, physical backtracking, matched model/
planning baselines, realistic sensing/timing and bounded hardware evidence—remains
active and unfulfilled.
