# Independent current-depth reactive perception

The current obstacle observer no longer waits for a visual pose. Quiet initial
public specific force supplies the initial up direction; the uninterrupted
500 Hz gyro stream transports that direction. Each current paired-depth frame
must still fit the original strict measured common floor plane. Candidate
selection, the 3 mm coherence bound, extent/up-alignment gates, above-plane
height band and 0.45 m nominal disk remain unchanged. Gyro bias is uncalibrated;
this is ideal-sensor development evidence, not a hardware clearance guarantee.

The first implementation used a separate thread. All 61 planes were available,
but mean depth work was 73.48 ms and only three plans met their deadlines.
Three commands lasted 20 ms before the age veto. Result preserved in
`go2_independent_depth_native_prefix_layout00_v1_attempt_001` (session 62674).

The next implementation uses a separate depth process and advances the
latency-release clock at each actual 2 ms physics step. Previously it advanced
only every 20 ms, introducing avoidable rounding delay into worker pipelines.
One focused native-adapter test verifies all 50 exact physics-clock callbacks
per 100 ms and the unchanged command-slew behavior (1.78 s).

The new 61-frame native run completed 14 of 15 plans on time and executed
14 complete 100 ms forward-command windows, with no stale-observation veto or
disallowed contact. Every window began at its exact scheduled simulation
boundary and had all five expected 20 ms services. All main-loop requested
and clipped commands match the recorded physics samples. All 214 latency
releases occurred after their measured service budgets. Mean depth-process
service was 47.30 ms; maximum obstacle publication age was 108 ms.

Root: `go2_independent_depth_process_native_prefix_layout00_v1_attempt_001`,
session 78363, exit zero. Result SHA-256:
`004a730b3c333d3d11e357f34a057fa3852a270b005f4f5f1b229ae1c923d942`.

The robot displaced only 2.52 cm in XY from its settled origin. The controller
still commands 100 ms bursts separated by a declared 300 ms zero prefix, so
this result establishes complete command windows rather than useful sustained
navigation. No host real-time qualification, full sensor replay audit or
maze-goal success is claimed.

The new 601-frame prospective run completed in
`go2_independent_depth_60s_native_layout00_v1_attempt_001` (session 5599,
exit zero), including persistence of all camera frames. Result SHA-256:
`883764284d320c550f47841c01ab9590f3a72a55bf0a8b018dfa2829ff501b04`.
It was a fresh episode, not continuation of a saved command tape.

This longer result is negative for sustained timing: only 16 of 150 plans
were on time; 134 were late. There were 76 nonzero 20 ms services, one fresh
observation age veto and three subsequent latched-zero requests. Native XY
displacement was 12.71 cm. All 601 observations completed without a tracker
failure or disallowed contact, but this is not useful maze navigation.

Mean measured service times were 92.98 ms tracking, 65.64 ms registration,
132.29 ms mapping, 51.54 ms planning and 47.56 ms independent obstacles.
Tracking completion age reached 980 ms; planning reached 1,060 ms.
Obstacle publication age reached 356 ms and ended at 100 ms. Most on-time
plans occurred near the beginning or end as backlog formed and recovered.
The 60.1-second main run plus 0.6-second zero drain had 31,100 physics samples,
including initial settling. Timed host work was 60.43 seconds; final artifact
persistence occurred afterward.

The independent obstacle path resolves the short-window dependency, but the
longer experiment contradicts a claim of reliable sustained timing. Next
work needs additional visual-tracking headroom; a smaller feature population
is a concrete speed/accuracy experiment. Longer command windows will also
need an explicit future-command ledger, because the current 300 ms forced
zero prefix still limits useful gait motion. Do not treat the earlier
14/15 short-prefix timing result as representative of a whole journey.
