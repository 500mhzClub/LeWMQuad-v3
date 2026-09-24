# Joint tracking enables two fresh native room returns; paired audit remains failed

The complete, explicitly **unpaired** analysis verifies two of three native
room returns. Both nominal directions complete all seven mission stages, all
sixteen completed local position/yaw and signed-winding holds, and both final
home holds. Lower friction completes no stage and stops on the unchanged
observed excursion limit despite accurate, available tracking.

This establishes continuous sensor-feedback execution and physical return to
stored observed coordinates in this exposed room. It does not establish an
isolated observer improvement, autonomous maze exploration, learned world-model
action selection, independent-layout reliability, real-time control or hardware
deployment. The long-term goal remains active.

| Condition | Native full return | Stages | Last decision tick | Maximum native home-hold position / yaw error | Native path |
| --- | --- | --- | --- | --- | --- |
| Nominal left | Pass, unpaired | 7/7 | 2,806 | 31.434 mm / 0.006258 rad | 4.594 m |
| Nominal right | Pass, unpaired | 7/7 | 2,180 | 18.990 mm / 0.020197 rad | 3.597 m |
| Lower-friction left | Fail: observed excursion | 0/7 | 521 | No home hold | 1.113 m |

The two nominal missions each require eight local legs because a metric return
is clipped into bounded subgoals. Every one of their eight completed local
holds passes the unchanged external position/yaw and signed-winding checks.
Pulse counts are 106, 79 and 21; no mission or per-leg budget was extended.
Native home holds contain 501 physics poses over the final one-second interval.
The final native XY positions are (0.02004, -0.02397) m and
(0.00255, -0.01831) m, respectively, relative to the initial body frame.

All 5,510 decisions replay exactly from recorded current/past RGB-D and gyro
packets. All 5,520 raw depth checks meet the existing 1 mm tolerance. Contact,
actuator, physical scene, command application, clocks and terminal drain checks
pass. There are 278,100 physics samples, 5,517 commanded intervals and 556.2 s
of simulated time including settling and the low-friction drain. No native
physical stop or storage stop occurs. Evaluator-native state does not supply
high-level commands or home coordinates.

| Condition | Available poses | Maximum position error | Maximum orientation error | Accepted 20 mm / 2 degree allocation violations |
| --- | --- | --- | --- | --- |
| Nominal left | 2,807/2,807 | 13.889 mm | 0.010617 rad | 0 |
| Nominal right | 2,181/2,181 | 18.135 mm | 0.007352 rad | 0 |
| Lower-friction left | 522/522 | 7.081 mm | 0.004338 rad | 0 |

Every accepted noninitial pose is an anchor measurement; these trajectories use
zero measured-increment bridge frames. Their success therefore does not isolate
a benefit from the bridge mechanism. The empirical error allocation is not a
calibrated uncertainty bound. Sensor-model and pose availability conclusions
apply to these complete fresh trajectories, not arbitrary motions or scenes.

The low-friction final decision estimates position (0.48143, 0.15100) m and yaw
-1.76319 rad against the original (0.4, 0) m / zero-yaw goal. Its forward
projection crosses the unchanged 0.48 m excursion boundary. After ten terminal
zero-command intervals, native XY is (0.49314, 0.15396) m and yaw -1.76242 rad.
This retains the support-transfer failure of the fixed nominal pulse model;
available tracking does not make its dynamics predictions support-independent.

The original [fresh-collection protocol](go2_joint_room_return_v1_2026-09-07.md)
required exact initial RGB pairing with the historical inner-arrival baseline.
Its auditor stops at that assertion for nominal-left, after raw reconstruction
returns and all six 750-sample native setup comparisons pass. That audit is
terminal failed and remains unchanged. It produces no completed condition score.
The prepared paired scientific reader was never executed because its required
completed paired audit does not exist.

Subsequent bound inspection confirms initial RGB inequality in all three trials:
25,955 / 24,012 / 25,846 pixels differ, with maximum channel changes
149 / 148 / 149. This is not negligible RGB rounding. The scene specifications,
native setup traces, environment identities and all fifteen saved visual mesh
files match. Initial native depth differences are below 28 micrometres and
validity masks match. The render-difference cause is unresolved. No tolerant
pixel comparison is substituted for the failed equality criterion.

The separately defined
[unpaired analysis](go2_joint_room_return_unpaired_readout_v1_2026-09-07.md)
keeps every raw, replay, native hold, winding, home, depth and timing check from
the frozen auditor. It records native setup equality and RGB inequality
separately and labels native success as unpaired. All three complete conditions
are analyzed; no native collection, controller or old audit is restarted.
The prior failed audit has no durable raw-pass receipt, so its partial in-memory
work cannot replace this complete authenticated readout. Original pairing
failure is an explicit terminal fact in the new result.

The historical inner-arrival batch was 0/3, but these results do not establish a
matched causal increase from 0/3 to 2/3. Besides initial RGB inequality, the
historical observer was multi-reference gyro tracking without continuity;
the new observer family jointly fits RGB-D rotation and checks measured
continuity. Subsequent commands and observations differ. These are direction
and support conditions in one exposed room geometry, not independent maze
replications or a reliability estimate.

| Condition | Median acquisition | Median controller | Median decision elapsed | Median normal full iteration |
| --- | --- | --- | --- | --- |
| Nominal left | 98.01 ms | 64.48 ms | 163.32 ms | 193.87 ms |
| Nominal right | 95.45 ms | 61.35 ms | 154.24 ms | 184.79 ms |
| Lower-friction left | 98.08 ms | 55.73 ms | 154.22 ms | 184.06 ms |

Of 5,510 decisions, 5,507 exceed 100 ms before command execution. Of 5,507 normal
iterations, 5,504 exceed 100 ms. The maximum normal iteration is 693.13 ms.
Physics remains paused while acquiring and planning; these measurements do not
show behavior under actual computation delay. Available-time labels explicitly
leave processing latency unaccounted. Startup and final artifact persistence
are excluded from the 1,036.962 s of recorded iteration wall time.

Acquisition includes rendering, packet formation and acquisition-time image/
depth file work; it is not a camera exposure latency measurement. The existing
session also calls `capture_current()` at command dispatch when that frame has
not yet been captured. Thus terminal-drain `command_tick` wall time can include
acquisition; the saved `physics_command_interval` timing key does not isolate
pure physics compute. Separate normal/terminal iteration summaries preserve
this distinction. Complete iteration accounting includes all those operations.
The next timing work needs component profiling and causal delayed-observation
tests, without merely changing timestamps or removing failed deadlines.

The new execution files are
`lewm/joint_sensor_anchored_goal_development.py`,
`lewm/joint_inner_goal_pulse_feedback_development.py`,
`lewm/joint_continuous_pulse_execution_development.py` and
`lewm/joint_inner_goal_room_return_development.py`. They preserve the existing
inner-goal planner, stop rules, signed intent and persistent observed targets.
Joint evidence is explicitly typed and checked; no gyro label is substituted
and no frozen module is monkeypatched. The empirical pulse table remains the
same fixed nominal fit. No learned RGB world model chooses commands here.

The integration/collection/adjacent controller suite passed 43 tests in 6.76 s.
Six prepared paired-summary tests passed in 1.82 s, but that paired reader is
unexecuted. Four unpaired-reader tests passed in 1.77 s; AST comparisons confirm
that native acceptance and replay checks are unchanged. An initial unpaired
test expected the existing-output error before the root guard; the actual root
guard correctly rejected the temporary test path first. The corrected test
checks both guards in order. No runtime acceptance behavior changed.

Native collection ran serially after testing, with one OpenCV/BLAS thread and
no concurrent training/test suite. Hardware preflight found 16 physical / 32
logical CPUs, about 77 GiB available RAM and 106 GiB artifact storage. The
collection wrote 6,899,467,123 bytes before its terminal result, inside the
15 GiB planning allowance. Minimum recorded free space was 106,586,664,960
bytes, above the 40 GiB reserve. A live sample during collection showed about
2.81 GiB RSS and one CPU core. The unpaired readout writes 1,962,371 bytes.
Source/input checks passed before and after analysis; 863 readout source
bindings, all original handoff identities and custody files remain intact.

All roots below are inside
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/`.

| Artifact | SHA-256 |
| --- | --- |
| `go2_joint_room_return_v1_attempt_001/launch.json` | `83b1e0ad2913cc4693bfdaf3cd3f939d7452b97566dff4c4e18c332e72c16260` |
| Same root: `result.json` (16,659 bound collection artifacts) | `a62a1de6837a758709d386eaa3aacfa8d49b043a20e477c867058cc49174a2ca` |
| Same root: `raw_return_audit_launch.json` | `aee6daaf65f75f78352c33a7398e05dd1f080823ab9b98e8e869ae01eb8818cc` |
| Same root: `raw_return_audit_failure.json` | `09ce73a3d590004db18ff53c95e6e52ce6f7b8d436fb1f04401e2a8276bb8042` |
| `go2_joint_room_return_unpaired_readout_v1_attempt_001/launch.json` | `f0dfba0f89f5a59a67997c1a736fdba6ea7789eec21ab79ce429e26b7cc4549c` |
| Same root: `result.json` (complete three-condition readout) | `e962b9c44215fd75add587ff948e68118d4bfcf402b3cfcf2659f138cc758369` |

Collector 43312 is terminal exit 0; paired auditor 81251 is terminal exit 1;
unpaired reader 61282 is terminal exit 0. Do not restart any of them.

The nominal execution evidence supports moving beyond fixed command tapes to
balanced branch/obstacle situations where geometry changes the useful action
and actual observed goal progress prevents permanent stopping/turning from
counting as success. Keep the low-friction failure visible and retain the fixed
empirical controller as a baseline. Profile acquisition/recording overhead and
test actual computation delay before real-time claims. Future matched fresh
experiments also need a prospectively justified observation/randomization
contract; exact scene IDs and source hashes did not prove identical images.
Learned predictive action selection, independent mazes, useful place memory,
realistic sensors and bounded hardware evidence remain unfinished.
