# Direct camera packets and latency-charged simulation

Direct in-memory paired acquisition reduced unprofiled mean acquisition time
from 222.07 to 35.87 ms. Captured arrays are saved after timed execution;
static renderer/environment checks occur at setup and termination. The camera
resources are warmed before the experiment clock. Four RGB/depth renders per
pair remain; evaluator-only auxiliary segmentation is omitted. No synthetic
legacy renderer witnesses are generated. This is a distinct development
acquisition path, not a claim that the old full audit has passed.

The first primary/auxiliary RGB arrays and native depth arrays exactly match
the earlier native prefix. This only checks the first paired acquisition.
The new setup and terminal static identities match. All 61 frames completed.
The wall-clock run took 6.304 seconds for 6.1 seconds of simulation, had one
on-time and 14 late plans, and issued no nonzero commands. Maximum simulator
lag was 55.08 ms; synchronous rendering still prevents host-real-time command
service at some policy boundaries.

Root: `go2_in_memory_camera_native_prefix_layout00_v1_attempt_001`, session
10749, exit zero. Result SHA-256:
`bbdea48e9188de9bc8094adcc0b51e6e7d0ec96e4eb55e9e6ceb2a56c13f6fa0`.

## Measured-latency simulation

The next experiment uses simulation time for action windows and charges
measured acquisition and worker service durations before releasing their
outputs. Physics keeps stepping during high-level processing. Rendering still
pauses the scene-owning host thread; this experiment does not establish host
real-time execution. It permits study of physical closed-loop behavior without
granting zero-time high-level computation in a slower-than-real-time simulator.

Planning uses a declared 300 ms zero-command prefix and one 100 ms candidate
interval. The same eight forecast horizons are used; progress and contact
scoring now end at 400 ms. Late initial command service is rejected, including
a plan first seen on the next 20 ms policy tick. The existing 200 ms
observation-age limit, nominal obstacle radius and model weights are unchanged.
Seven focused scheduling and latency-release tests passed in 2.35 seconds.

Root: `go2_measured_latency_native_prefix_layout00_v1_attempt_001`, session
20546, exit zero. Result SHA-256:
`1dde070495a84e754478d8dc615d0236d3f7a0876d2b7881eb19d2314365835b`.

All 61 frames completed. Six plans were on time, nine late. Six forward
commands began exactly at their simulation dispatch boundaries; each was
vetoed on the next 20 ms policy tick as obstacle evidence aged past 200 ms.
All 60 associated 2 ms physics samples match both the requested 0.2 m/s
command and its actual clipped command. These are actual native commands,
not shadow outputs. All 214 recorded worker releases occurred no earlier
than their measured service-time budgets. No disallowed contact was recorded.

The run contains 3,050 main physics samples plus 750 settling and 300 zero
drain samples. Simulated time after settling was 6.7 seconds, including the
0.6-second drain. Native final XY displacement from the settled origin was
only about 4.3 mm; this is not meaningful maze progress or a mission success.
The complete raw sensor replay audit remains outstanding.

The immediate scientific/implementation target is to obtain current obstacle
evidence without waiting for visual pose tracking. The tracker can remain in
the routing/planning path. A separate current-depth plane estimator must use
public sensor evidence, retain the original floor/height and unknown-space
semantics, and be charged for its actual computation. Do not solve these
20 ms truncations by weakening the observation-age limit or relabeling an old
plane as current. The active long baseline sources remain unchanged.
