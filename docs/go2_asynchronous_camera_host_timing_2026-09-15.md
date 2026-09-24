# Asynchronous camera acquisition and host command deadlines

The completed synchronous 60-second wall probe produced zero nonzero commands:
rendering and packet preparation blocked initial command service by about 43 ms.
The unchanged initial-dispatch tolerance is 1 ms and simulator-lag limit is 20 ms.

## Snapshot renderer evidence

`scripts/snapshot_camera_renderer_development.py` owns a separate identical scene.
It receives a configuration captured from the physical robot at a camera boundary,
updates robot visual geometry, and renders both calibrated cameras without a
physics step. Native configurations remain inside this sensor adapter; the
controller receives only its existing sensor packets. Static world geometry is
used only to construct the renderer, as in the original native sensor.

The first three-configuration check preserved exact depth but differed in RGB:
the non-stepping scene retained stale visual geometry/shadow bounds. The original
result and source files remain in `go2_snapshot_camera_renderer_check_v1_attempt_001`.
Explicitly refreshing the renderer once per captured configuration corrected this.
Attempt 002 produced bit-identical primary and auxiliary RGB and depth at all three
configurations, with exactly equal camera transforms. Snapshot reads took
0.021–0.090 ms; renderer round trips took 22.6–23.1 ms. This is a component check,
not navigation or real-hardware sensing evidence.

## Prospective 60-second host-clock probe

Launcher: `scripts/run_go2_async_camera_wall_probe_development.py`.
Root: `go2_async_camera_wall_deadlines_layout01_600_v1_attempt_001`.
It keeps the original wall probe's learned controller, local-reference tracker,
model, scene, 2-mm depth noise, 100-ms camera cadence, 20-ms physical command
service, 300-ms planning delay, and all command/pose thresholds. The batched
tracker is deliberately not added to this timing comparison.

The physics owner captures qpos and frozen co-timed body/gyro histories, then
submits rendering and RGB-D packet preparation to one worker process. At most
two camera acquisitions can be in flight; overflow is a recorded failure. The
owner polls completed packets without waiting for them and continues servicing
commands. Actual acquisition completion and controller submission times remain
recorded separately from original ideal simulated sensor timestamps. No host
clock, sensor timestamp or dispatch deadline is rounded or retimed.

The renderer never advances physics. The physical service ledger retains the
exact post-veto requests and unrounded host request times. Only terminal camera
and pipeline draining waits after the timed command loop. A new native loop
contains this acquisition change; the original synchronous loop and completed
comparisons remain unchanged. A short probe has now launched. Its result is
pending and must not be called real-time navigation based on image equivalence.

## Attempt 001 complete: movement enabled, camera-boundary jitter remains

The owner exited 0 after 1:56.96 including startup and recording, without swap.
All 601 camera frames registered; all 150 learned XY/yaw treatments were verified,
147 plans were on time, and pose error stayed below 3.993 mm. There were 1,272
nonzero requested and 1,423 nonzero applied intervals of 3,005 total, no arrivals
in this 60-second prefix, and no disallowed contacts. The applied/requested counts
differ because the unchanged command-rate limits ramp between commands.
All physical service requests match the exact recorded history and host timestamps.

Median host/simulation lag was 0.381 ms (p95 5.518 ms), but the maximum was
273.774 ms. Seventy initial dispatches were late, all on camera boundaries.
Their median lateness was 1.084 ms; same-tick snapshot/packet assembly took
1.004 ms median. Whole-run snapshot median/p95 was 0.900/1.114 ms. Render worker
median was 23.107 ms; acquisition-to-owner receipt median was 60.017 ms. Thus
remaining initial misses came from local snapshot work near the unchanged 1-ms
limit, not waiting for the rendering worker. Analysis is saved in
`async_host_deadline_diagnostic_v1.json` in this root.

## Attempt 002: cheaper snapshots, but a separate long stall

Move packet construction into the renderer using owned frozen body/gyro histories.
Four focused tests verify identical packet contents, independence from subsequent
sensor edits and episode resets, stale-history handling, and future-availability
rejection. The owner now copies only the chronological current sensor histories;
the same packet constructors validate them in the worker.

Attempt 002 exited 1 after 40.82 s with `renderer exceeded two in-flight camera
acquisitions`, after 65 submitted camera frames and 335 physical requests.
Its early snapshot costs were around 0.35 ms. A 230-ms request-to-physical-completion
interval preceded the overflow; the host was already 221 ms behind at its lag
check. This distinct long stall must not be explained as rendering cost.
The first probe also had occasional 225–294 ms intervals. The complete available
recording is retained; two pending render results were not recorded at shutdown.

Attempt 003 keeps the two-acquisition bound and all control limits. It polls
completed camera results before requesting a new slot, records separate controller
request and physical-service timings plus Python garbage-collection intervals,
and preserves pending completed render results during failure shutdown without
submitting them to a stopped controller. Its short timing probe is now running.
No full wall-clock mission or hardware qualification is established.

## Attempt 003 complete: camera-boundary misses removed; GC stalls measured

The owner exited 0 after 1:56.02, no swap. All 601 poses registered and all 150
learned XY/yaw treatments were verified; 148 plans were on time. Maximum pose
error was 4.989 mm, with no arrivals or disallowed contacts in the short prefix.
There were 2,001 nonzero requested and 2,095 nonzero applied intervals. No initial
dispatch missed its 1-ms tolerance. Snapshot submission median/p95 was
0.369/0.465 ms. Draining completed camera results before new submission avoided
counting completed results against the unchanged two-acquisition bound.

Two generation-2 garbage collections lasted 216.748 and 285.229 ms. The following
command services had host lag 212.043 and 297.010 ms; surrounding physical-service
gaps were 224.103 and 321.635 ms. Actual separate request/physics timestamps and
GC callbacks are saved, including `generation2_command_gap_diagnostic_v1.json`.
These measurements associate the large host pauses with cyclic GC. Other shorter
physics/service overruns remain; this is not a hard real-time guarantee.

Attempt 004 defers owner cyclic garbage collection during the bounded 60-second
probe, collecting before its clock starts and restoring the prior GC setting
afterward. Python reference counting stays active. The same controller, snapshot
code, queue bound, physical limits and host deadlines remain. Its run is pending.

## Attempt 004 complete: large GC pauses removed

The owner exited 0 after 1:56.71, without swap, maximum RSS 5,403,600 KiB.
All 601 frames registered, pose error stayed below 4.547 mm, and all 150 learned
plans finished on time with verified learned XY/yaw treatment. There were
2,143 nonzero requested and 2,225 nonzero applied intervals, no arrivals in the
60-second prefix, and zero disallowed contacts. All physical service requests
and ordered sensor acquisition timestamps match their recorded receipts.

No cyclic-GC event occurred during the probe. Maximum host/simulation lag fell
from attempt 003's 297.010 ms to 38.111 ms; median/p95 was 0.464/8.662 ms.
One initial dispatch was 1.355 ms late; 37 intervals triggered the unchanged
20-ms host-lag veto. Snapshot median/p95 was 0.392/0.483 ms, and physical-service
plus request duration still reached 54.832 ms. This removes the observed large
GC pauses but does not establish hard real-time execution or hardware readiness.
The actual actuator-target application timestamp remains unmeasured.

## First full wall-clock mission launched

`scripts/run_go2_async_camera_wall_mission_development.py` runs the same learned
controller and asynchronous sensors on exposed layout 1 with a 4,800-tick
navigation budget. Root:
`go2_async_camera_wall_mission_learned_layout01_4800_v1_attempt_001`.
One physics owner and one non-stepping rendering worker are used. Cyclic GC is
deferred only for this bounded mission; all host/physical guards, ideal sensor
stamps, noise, models, reference policy and command accounting remain unchanged.
It is not a fresh-maze or held-out evaluation, and does not alter the fixed
four-layout comparison. Verify physical goal and home arrivals after owner exit.

Before launch, routine retention retired only depth from fully analyzed current
reactive layout 2 and fitted-motion layout 3 successes, reclaiming 1.644 GiB.
All failures, first full successes per controller, current timing recordings and
pending diagnosis inputs remain retained. Free artifact storage was 4.83 GB.

## Full wall-clock mission complete: independently verified goal and home

The full mission owner exited 0 after 6:21.35 including initialization and
recording, without swap; maximum RSS was 14,140,164 KiB. Both physical arrivals
passed: goal frame 1567 and home frame 2480, with maximum one-second quiet-dwell
distances 13.217 and 13.745 mm. All 2,482 poses registered, maximum pose error was
4.072 mm, and there were zero disallowed contacts. Duration was 248.18 simulated
seconds and 249.569 timed-loop wall seconds. All 613 actual learned XY/yaw
assignments were verified, with 608 plans on time. Applicable saved forecast,
physical-arrival and navigation-summary evaluations are complete.

All 12,409 physical service requests exactly match the recorded post-veto
command histories and unrounded host request timestamps. All camera acquisition
receipts preserve their ordered actual completion times. There were 6,721 nonzero
requested and 7,008 nonzero applied intervals. The controller's unchanged limits
vetoed late/stale/inconsistent windows: 466 host-lag vetoes, five late initial
dispatches, and 3,298 prefix-execution mismatches, among other zero requests.
These vetoes remain part of the outcome rather than being retrospectively removed.

Maximum host lag was 130.837 ms over this longer run (median/p95 0.522/17.584 ms),
so the 60-second probe's 38-ms maximum was not a bound on a full mission. Snapshot
submission median/p95 was 0.368/0.457 ms and acquisition-to-owner receipt
median/p95 was 60.045/84.878 ms. Cyclic GC stayed deferred; other execution and
scheduling overruns remain. No hard real-time, real-sensor calibration or hardware
claim follows. Actual actuator-target application timestamps remain unmeasured.

This is the first independently verified full mission under actual host-clock
command deadlines, on one previously exposed development maze with the frozen
supervised learned controller. The original fresh-maze 4/4 result is unchanged.
There is still no repeatability, isolated predictive-planning advantage, JEPA
advantage or hardware evidence from this one run.

Comparison with its original measured-simulation learned layout-1 run:
`go2_async_wall_and_measured_mission_comparison_v1_attempt_001/result.json`, with
PNG/SVG trajectories alongside it. Shared scene/noise/controller settings match;
clock mode, acquisition architecture, physical-service history adapter, GC
scheduling and host workload differ, so this is not an isolated speed ablation.
Keep this first full wall-clock success and all failed attempts in full.

Next scientific work can use this wall-clock path for prospective repetition and
matched controller comparisons. Keep sensing/hardware calibration, actual actuator
timing, broader maze/training-seed coverage and isolated model/memory contributions
open. No native or replay process remains active after this completed mission.
