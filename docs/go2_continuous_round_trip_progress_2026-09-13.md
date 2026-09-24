# Continuous round-trip development

The mission uses the independent layout 0 public target, the existing JEPA
model, 150-feature tracking, current-depth vetoes, 400 ms action windows and
300 ms forecast prefixes. Outbound and return retain the same tracker and map.
Arrival requires the existing observed distance and measured stopping dwell.
The navigation budget is 1,800 camera intervals. Measured computation costs
are charged to simulation time; host real-time execution is not claimed.

All roots below are under the recovery-storage navigation development artifacts.
All three attempts failed before any observed arrival; none is a round trip.

- `go2_continuous_round_trip_native_layout00_v1_attempt_001`: stopped after
  143 acquisitions and 711 policy services because registration tried to read
  preceding commands after their two-second dispatch history had expired.
  Saved mission publication age grew from roughly 212 ms in the first 30
  frames to 2,016 ms. This exposed both a history-lifetime bug and backlog.
- `go2_continuous_round_trip_native_layout00_v1_attempt_002`: preceding requests
  were captured at observation submission. Four focused mission/history and
  commitment-ledger tests passed in 1.97 s. Registration then filled its queue:
  213 acquisitions, 1,061 services, mean registration service 110.8 ms and
  maximum age 3,520 ms. Native displacement was 0.512 m, without contact.
  Tracking averaged 91.3 ms and independent obstacles 60.0 ms. Stage evidence
  is now also persisted on failure. The process exited with failure preserved.
- `go2_continuous_round_trip_native_layout00_v1_attempt_003`: removed a second
  pose-witness reconstruction immediately after registration had already
  validated the same returned pose. The existing registration validation is
  unchanged. This was insufficient: 221 acquisitions and 1,101 services before
  the registration queue filled again. Registration averaged 81.8 ms over
  frames 0–39, then about 113–124 ms in later groups. Maximum age was 3,520 ms.
  The process exited with failure preserved.

A replay profiler reconstructs public camera/body packets from attempt 003,
re-runs tracking and registration, and compares both poses exactly with the
recorded outputs. It does not read native pose or scene state. Its purpose is
to separate intrinsic registration cost from contention during live execution.
No additional full mission should be inferred from these diagnostic runs.

The local replay completed over all 185 saved poses with exact raw and registered
pose equality. Registration averaged 40.24 ms including profiler overhead;
370 sampled-plane extractions accounted for 5.55 of 7.42 profiled seconds.
There was no dense-extractor fallback in the earlier 18-camera-sample probe.
The much larger live service time points to contention during concurrent work.
The next candidate moves the unchanged sequential registration instance to
one separate process, including transfer time in its measured service cost.

The process replay also completed all 185 frames with exact raw and registered
pose equality. Mean registration time including transfer was 36.56 ms. The
local profiler instruments registration internals while the process profiler
times the round trip; their means are diagnostic, not a clean speedup estimate.
The prospective native candidate ran in
`go2_continuous_round_trip_native_layout00_v1_attempt_004`, session 24605,
and exited with failure. Registration averaged 62.49 ms, with no registration
queue overflow. The tracking input queue instead filled after 531 acquisitions
and 2,652 completed policy services, around 53 simulated seconds. Tracking
averaged 104.47 ms and rose to 117.2 ms in its last group of frames. Planning
averaged 177.54 ms; only 12 of 123 plans were on time. No arrival occurred.
Native XY displacement was 0.515 m, with no disallowed contact. Separating
registration improved that stage but did not make the complete pipeline viable.
The next diagnostic profiles routing on the exact reconstructed attempt 003
camera sequence, preserving all original route decisions for comparison.

Routing replay completed 47 map queries across the 185 exact reconstructed
poses. The profiler attributed 7.42 of 7.89 seconds to 14,778 connector
supercover calls. It was repeatedly testing candidate connections across
grid cells. The new private routing function performs exactly the same
closed segment/box tests with array arithmetic, including the original
parallel-axis, corner and floating-point tolerances.

One focused boundary/random-segment comparison passed in 0.21 s. A paired
alternating-order comparison then reproduced all 47 full route dictionaries
exactly. Unprofiled mean routing time was 104.37 ms for the original and
15.97 ms for the vectorized candidate. Geometry, cell ordering, frontier
selection and unknown-connector reporting remain the original algorithm.
This is a computation improvement; it does not prove navigation reliability.

`go2_continuous_round_trip_native_layout00_v1_attempt_005`, session 88471,
combines separate registration with the vectorized routing kernel. Collection
completed its navigation budget: 1,808 camera frames, 9,038 policy services,
91,560 physics samples, no pipeline failure, no contact, no arrivals, terminal
`MISSION_TICK_BUDGET_EXHAUSTED`. Main work took 181.45 host seconds; the full
post-settle physics interval including zero draining was 181.62 seconds.
The process subsequently exited zero after saving all 1,808 image pairs;
camera setup/terminal identity was unchanged. Result SHA-256:
`57c6f81894ebdf535e4238ad1ddc68128aff383b2404c0b98531e05510bb8d5b`.
Successful process completion does not change the negative navigation outcome.

Only 65 of 450 scored plans were on time; 385 were late. There were 291
nonzero services. Mean stage service was 99.17 ms tracking, 56.78 ms
registration, 140.46 ms mapping, 71.61 ms planning and 61.18 ms current-depth
obstacles. Tracking's 95th-percentile age was 880 ms despite near-100 ms mean
service. This is insufficient headroom for the 300 ms planning deadline.

Post-estimation native comparison found XY displacement 0.5340 m. All 1,808
registered poses had median/max position error 5.00/6.03 mm. Thus accurate
pose estimation on this trajectory coexists with poor freshness and failed
navigation. It is not a full sensor/visibility audit or a general accuracy bound.

There is also a geometric stopping trap: 438 plans requested an additional
view, and 442 selected a left turn. The first current obstacle veto occurred
at 6.6 simulated seconds (including initial settling), with nominal connector
minimum distance 0.4206 m versus the fixed 0.45 m circular footprint. A late
turn veto still reported distance 0.4472 m. These are observed-cell distances,
not native clearance measurements. Future work must distinguish insufficient
tracking/deadline headroom from this inability to select an executable turn
near the conservative footprint boundary. Do not simply waive the veto.
