# Recorded host timing on the first eight transfer assignments

Subsequent work moved rendering and packet preparation into a separate snapshot
renderer and deferred cyclic GC during bounded execution. The short moving probe
completed; the first full wall-clock mission has now independently passed goal and return
verification with zero disallowed contacts. See
`docs/go2_asynchronous_camera_host_timing_2026-09-15.md` for current results.
The original synchronous failures below remain unchanged.


## Completed current-controller host-deadline probe: movement prevented

Attempt 002 exited 0 after 1:41.59 including initialization and recording,
without swap. It ran 60.10 simulated seconds (60.523 s in the timed loop),
captured and registered all 601 frames, and produced 150 selected plans, 148
on time. All 150 actual forecasts match the frozen learned XY/yaw treatment.
All 3,005 requested and applied command intervals were zero; no goal or return
arrival occurred, and there were zero disallowed contacts. This is a completed
negative timing result, not a successful wall-clock navigation trial.

Every physical service receipt exactly matches its recorded native request,
physical timestamp and unrounded host-request timestamp. Pose error remained
below 0.205 mm on this almost stationary trajectory. Integrated native XY path
length was 0.153 m from idle motion; zero requests are not physical stationarity.
Independent physical evaluation, navigation summary, treatment checks and
`host_deadline_diagnostic_v1.json` are saved in the attempt-002 root.

The 20-ms simulator-lag gate acted on 1,765 intervals; another 1,224 retained a
window veto. There were 145 underlying `FIRST_DISPATCH_TOO_LATE` events, all on
camera-acquisition ticks. Their median initial-dispatch lateness was 43.012 ms,
while median acquisition time on those same ticks was 42.925 ms. The initial
dispatch tolerance remains 1 ms. Across all cameras, acquisition median/p95
was 43.150/55.993 ms; host-minus-simulator lag median/max was 22.662/278.317 ms.
The native loop acquires cameras before servicing commands on those ticks,
with rendering and physics on the same owning thread. These recorded timings
identify acquisition blocking the scheduled initial command service.

Actual worker wall-duration medians were tracking 68.966 ms, registration
14.463 ms, mapping 68.943 ms, obstacles 15.955 ms and planning 35.674 ms.
Planner median/p95 was 35.674/48.532 ms. These near-stationary measurements do
not bound moving-scene tracking cost: the fitted-transfer return had a separate
queue overflow. No deadlines or sensor stamps were relaxed or retimed.

The simulation architecture therefore remains unqualified for host-real-time
navigation. Keep both wall attempts and the complete fixed twelve-run result.
Do not repeat a longer wall mission with this unchanged acquisition schedule.
Next, the retained fitted-layout-2 tracker is being replayed with per-call wall
timers around the unchanged estimator and registration; packet I/O is excluded.
This will test whether its expensive return-phase processing reproduces in
isolation. Asynchronous sensing/physical execution and bounded real-platform
timing remain necessary for deployment evidence.

## First host-deadline attempt exposed a command-history clock mismatch

The first probe exited 1 after 21.64 s during startup, with one captured camera
pair and one completed policy request. Its failure is
`ValueError('complete ordered 20-ms request history required')`: the inherited
commitment ledger received irregular host request timestamps instead of physical
20-ms service timestamps. This is an interface failure, not a timing or
navigation result. Attempt 001, all its outputs and its original launcher at
`.generated/current_controller_wall_deadline_attempt001_launcher.py` remain.

Fix the interface in a separate attempt 002 using the same 60-second probe.
`PhysicalServiceLedger` keeps each unrounded host request separately, then
records its exact physical service timestamp and the request after the external
host-lag veto. Arrival history and prefix checks therefore see the actual
physical request stream. The existing native loop still timestamps, gates and
reports deadlines on the host clock. No physical interval is invented and no
sensor timestamp is changed. An external command change invalidates the active
commitment. The original physical service still applies its existing limits.

The focused check reproduces the old jitter failure and verifies post-veto
physical history, preserved wall timestamps, commitment invalidation and
rejection of a missing physical interval. Record:
`.generated/current_controller_wall_physical_service_check_2026-09-15.json`.
Runtime decision/perception methods remain inherited; the worker and command-
history adapters are the deliberate wall-clock integration changes. The twelve
completed measured-simulation assignments remain unchanged. Attempt 002 output:
`go2_current_controller_wall_deadlines_layout01_600_v1_attempt_002`.

## Prepared prospective current-controller probe

After the fixed twelve-run stopping-projection transfer is complete, run one
60-second timing probe on exposed transfer layout 1, alone on its original odd
CPU group. Launcher:
`scripts/run_go2_current_controller_wall_deadlines_development.py`.
Root: `go2_current_controller_wall_deadlines_layout01_600_v1_attempt_001`.
Keep the frozen supervised model, learned XY/yaw, current local-reference
perception, committed views, stopping projections, 2-mm depth noise, ideal gyro,
300-ms planning delay and original physical/dispatch limits. This is a bounded
timing experiment, not a full navigation assignment or new-maze evaluation.

The base launcher selects `OverlappedObstacleRuntime` in wall mode; setting only
`MEASURED_RUNTIME_CLASS` would silently restore an older controller. The new
launcher explicitly binds the wall-mode runtime to the current controller.
Its only class override uses the original worker loop, bypassing measured-
simulation begin/end release waits. Tracking, registration, mapping, planning,
prediction selection and requests remain the current inherited methods. Their
completion timestamps use the actual host clock. Sensor stamps remain physical
simulation time, and the existing 20-ms simulator-lag and 1-ms initial-dispatch
limits remain unchanged. No clock retiming or relaxed timeout is introduced.

A small wiring check exercised the full nested launch writer and the preserved
method identities; final result `.generated/current_controller_wall_deadline_wiring_2026-09-15_v2.json`.
The final writer explicitly updates the existing `navigation_tick_budget` field
to 600; the initial wiring record is preserved separately.
An initial check named a nonexistent selection method; the corrected check uses
the actual `_select_clear_prediction` method. No native timing outcome exists
yet. Record actual controller treatments, command/service lateness, dispatch
vetoes, sensor age, motion and any failures after the single owner exits. A
future full wall-clock mission requires evidence from this probe; do not infer
navigation success from saved measured-simulation trajectories.

`scripts/summarize_saved_native_host_timing_development.py` reads completed
recordings without rerunning physics, sensors or inference. Each of the first
eight assignments now has `host_timing_diagnostic_v1.json`. All original
successes and failures remain in the population. The registration-failed
reactive layout 0 has no saved worker-release log: that source file is written
only after normal pipeline completion. Its acquisition, request and planning
timestamps remain available; absent worker timings are not inferred.

The six predictive trials have median planning service durations of 68.6–74.7
ms and 95th percentiles of 86.9–94.0 ms. Median observation-to-plan-completion
ages in simulation time are 218–226 ms. All six use the existing 300-ms dispatch
delay; 98–99% of selected plans arrived on time except JEPA layout 1, at 97.5%.
The reactive layout 1 planning stage has median 15.6 ms and p95 34.5 ms, with
median simulated observation-to-plan age 154 ms. These are different executed
trajectories on shared host resources, not a causal timing benchmark.

Paired acquisition medians are 45.1–51.5 ms across all eight recordings.
However, the measured host-minus-simulation lag at command service often
exceeds the separate wall-clock mode's existing 20-ms limit. The three
predictive layout-0 medians are about 31 ms; their layout-1 medians are
451–482 ms. Maximum lag across individual recordings ranges from 308 ms to
1.570 s. Near-unity total wall/simulation runtime does not prove bounded
command-service jitter or real-time operation.

The current runs charge measured acquisition and worker computation to their
simulation clock. Every available worker-release record respects its recorded
earliest release time. This supports the measured-simulation interpretation;
it does not establish host real-time qualification. Running the wall-clock
mode would change command vetoes, trajectories and future observations, so
its outcome cannot be obtained by relabeling or retrospectively censoring
these runs. A separate wall-clock execution remains necessary. Robot camera,
IMU, transport and actuator timing remain unmeasured.

No running controller, clock or frozen comparison assignment was changed by
this analysis. Worker service values are recorded per release and may be
cumulative within an item; the analysis does not sum them as disjoint work.
