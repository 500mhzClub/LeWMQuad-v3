# Completed native run: packet acquisition also exceeds the deadline

A read-only census of every completed chained-native decision separates the
collector's already recorded timing boundaries. This does not rerun the
controller, introduce component instrumentation, or replace the prospective
chained/single-pass replay. The native root result SHA-256 is
`163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849`.

| Recorded interval | Count | Median ms | p95 ms | Observations over 100 ms |
| --- | ---: | ---: | ---: | ---: |
| `session.sensor_packets()` | 4,014 | 219.556 | 248.391 | 4,014 |
| Controller `observe()` | 4,014 | 1,360.297 | 3,013.319 | 4,004 |
| Acquisition plus controller | 4,014 | 1,578.895 | 3,228.405 | 4,014 |
| Iteration through command execution | 4,013 | 1,610.668 | 3,260.337 | 4,013 |
| Decision receipt writing | 4,014 | 20.555 | 22.616 | 0 |
| Iteration through receipt writing | 4,014 | 1,631.407 | 3,281.907 | 4,014 |

The additional command-execution and surrounding overhead interval, computed
per row by subtracting acquisition-plus-control from iteration-through-command,
has median 31.709 ms and p95 32.447 ms. Its population is 4,013 because the last
observation has no following command. Medians of component populations should
not be added to reconstruct a total median.

Packet acquisition's minimum was 104.217 ms. Thus even eliminating controller
computation would not make the recorded sequential acquisition-plus-control
population meet 100 ms. Controller work remains the larger measured cost, but
the packet path also needs investigation before a real-time claim. These are
the existing simulator packet-acquisition boundaries, not measurements of
physical camera latency or proof that a particular optimization will work.
Physics was paused during computation; no continuous-execution result follows.

The census checked all 4,014 ordered decision/timing row pairs, finite
nonnegative component times, the per-row acquisition-plus-controller identity,
and the 4,013-command population against the completed collection. Input hashes
were checked before and after scanning, as was the completed root result hash.
The source collector records these intervals in
`scripts/residual_anchored_continuation_maze_episode_development.py`.

Exact statistics, linear-interpolation quantiles, input hashes and scope flags
are in
`docs/go2_measured_plane_chained_logged_timing_decomposition_2026-09-12.json`,
SHA-256 `a4e021a48171f3da2c63347ccbc17fc0976fe20ebebee173455f99223ef9924d`.
The original diagnostic copy remains at
`/tmp/go2_chained_native_logged_timing_decomposition_20260912.json`.
The scan took 45.4143592689652 seconds and ended successfully in tool session
79881 while the comparison launcher was still authenticating native inputs.

## Existing acquisition subintervals and source follow-up

The already saved auxiliary-camera audit and renderer witnesses contain
additional timing fields. Their complete 4,014-frame populations were checked
against ordered frame identities, timestamps and physical sample indices, with
both bound artifact hashes checked before and after inspection:

| Existing recorded subinterval | Median ms | p95 ms | Maximum ms |
| --- | ---: | ---: | ---: |
| Auxiliary capture | 73.498 | 89.510 | 100.538 |
| Primary renderer witness query | 0.191 | 0.212 | 0.536 |
| Paired renderer witness query | 0.298 | 0.318 | 0.454 |

Auxiliary capture exceeds 100 ms in two frames. Its source timer in
`scripts/auxiliary_downward45_depth_capture_development.py` includes separate
RGB, depth and segmentation renders, compressed depth/segmentation persistence,
PNG persistence, pixel hashes and acquisition checks. It stops before the
`finally` block restores the primary camera pose. It is not a pure renderer or
physical-sensor-latency measurement.

The witness timers cover context queries and intervening state checks, not
every operation in the wrapper. Their recorded costs are small; the evidence
does not support treating these measured query intervals as the acquisition
bottleneck. No witness or measurement check has been removed.

Source inspection also shows that the public packet path reads the saved
primary PNG, constructs auxiliary depth from the saved archive, and reads
auxiliary depth again when building the RGB packet. It then reads the saved
auxiliary PNG and validates its pixel identity. These are concrete candidates
for future measurement; the current logs do not isolate their individual
costs or establish the speedup from avoiding any duplicate work. Existing
pixel identities, packet semantics and evaluator-only boundaries must remain
testable in any successor.

Exact subinterval statistics and input hashes are in
`docs/go2_measured_plane_chained_recorded_acquisition_subintervals_2026-09-12.json`,
SHA-256 `718cc18205eb893262abbc7613838b773e1d20c031542bcbbd9a9af7243b3a30`.
The original diagnostic copy remains at
`/tmp/go2_chained_native_recorded_acquisition_subintervals_20260912.json`.
No new simulation, controller execution or component profiler was used.
