# First prospective wall-clock prefix

The 61-frame native layout-0 prefix completed, but it did not issue nonzero
commands: the simulator could not supply the required 10 Hz paired camera
stream in wall time. This is an integration/timing result, not a navigation
success or a failure attributable solely to the learned controller.

The new session services requests every 20 ms and executes the original gait
policy and ten 2 ms physics samples per service. It retains the original
100 ms command-delta allowance rather than replenishing that allowance five
times faster. One focused test verifies the allowance and mid-window zero
request (1.87 s). The initial test had its synthetic min/max speed arguments
reversed; correcting the fixture required no change to the native adapter.

Perception, registration, mapping and planning run independently. The host
wall clock owns command deadlines; the existing acquisition adapter retains
its explicit ideal simulation timestamps. A simulator lag greater than one
20 ms policy interval forces zero and latches any active command window.
Rendering still pauses the scene-owning thread. These are limitations of
this prototype, not evidence of deployment-valid acquisition latency.

Result root: `go2_paced_native_prefix_layout00_v1_attempt_001` (session 72122,
exit zero). Result SHA-256:
`f7537d7bd2c713863c27e77034c23fa36f0bbe7c30874dcd9a790012cb70699f`.

- 61 actual paired acquisitions, 305 policy services, 3,800 recorded physics
  samples including the original 750-sample settling period.
- 6.1 seconds of simulation took 15.95 seconds of host time; maximum lag
  reached 9.684 seconds. The first acquisition alone took 382.46 ms.
- Paired acquisition mean/median was 222.07/220.05 ms, total 13.546 seconds.
  Gait/physics service mean/median was 7.12/6.43 ms per 20 ms, total 2.171 seconds.
- All 305 requested commands were zero due to simulator lag. All 15 plans
  missed their host deadlines. No native disallowed contact or physical stop
  was recorded. This does not test moving perception or goal-reaching.

The native artifacts and requests were persisted. A complete sensor replay
audit has not been performed, and no real-time or mission qualification is
claimed. The concurrent long direct-baseline collection continued unchanged;
these are shared-host timings, not an isolated hardware benchmark.

The 13-frame acquisition profile completed in
`go2_native_acquisition_profile_layout00_v1_attempt_001` (session 38603,
exit zero). Its native run again requested zero throughout. The profile
records five camera render calls per paired acquisition, synchronous PNG/NPZ
persistence and loading, repeated raster-order verification, and repeated
environment/triangle identity construction. It also includes concurrent
worker activity, so cumulative entries overlap and must not be summed or
treated as isolated exclusive acquisition costs. The profile is diagnostic;
the earlier unprofiled 222 ms/frame is the timing evidence.

The next implementation should remove repeated static identity reconstruction
and avoid the synchronous save/reload path for live packets, preserving
measured RGB-D arrays and actual timestamps. Static renderer/setup witnesses
can be captured at setup and termination for this development timing run;
the existing long baseline and its source bindings must remain unchanged.
This is acquisition work prompted by an actual native result, not another
tracker micro-optimization. Sensor equivalence and measured acquisition time
need testing before claiming that the acquisition change solves the deadline.
