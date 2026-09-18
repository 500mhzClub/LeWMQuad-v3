# Matched layout-4 result: learned round trip, reactive stall

The learned controller with stable references, compiled floor extraction and
arrival-entry terminal priority completed a physically verified round trip.
The matched reactive controller did not reach the goal. Both owners exited 0,
and both full recordings are retained on RecoveryStorage.

| Measure | Learned | Reactive |
| --- | ---: | ---: |
| Verified outbound arrival | Yes, frame 3,475 | No |
| Verified home arrival | Yes, frame 4,412 | No |
| Disallowed contacts | 0 | 0 |
| Maximum position error | 8.29 mm | 9.38 mm |
| Closest outbound physical goal distance | 14.98 mm | 1.286 m |
| Physical path length | 22.555 m | 5.412 m |
| Plans on time | 1,008/1,096 | 1,190/1,199 |
| Timed wall duration | 442.59 s | 481.53 s |

The learned arrival dwells stayed within 17.28 mm outbound and 17.78 mm home,
below the unchanged 40-mm physical requirement. Both one-second dwells had zero
requested commands and passed the motion limit. The controller's observed
arrival radius remained 20 mm. Return took 93.7 simulated seconds after 347.5
seconds of outbound exploration; the trajectory figure shows a dead-end visit
outbound and a shorter return path. This does not by itself isolate memory's
causal contribution.

Reactive selected hold for all 801 plans from frame 1,600 onward. Its final
current stored clearance was 0.4150 m, below the nominal 0.45-m footprint limit,
so movement and further survey turns were rejected. It continued receiving
fresh poses and completed almost every plan on time. This is a current-clearance
deadlock rather than a tracking or deadline failure.

The comparison helper confirmed matching mission, sensing, stable-reference,
routing, timing and pulse settings, plus equal common recorded source hashes.
It compares the complete predictive selector—including its forecast clearance
and terminal rules—with instantaneous reactive feedback. It does not isolate
JEPA training, individual prediction components or memory, and one development
pair does not establish repeatability or novel-maze generalization. Sensors and
timing remain ideal/shared-host simulation; real-platform evidence is outstanding.

Comparison root:
`go2_arrival_entry_stable_reference_pulse_matched_comparison_layout04_v1_attempt_001`.
Its `result.json` contains the complete paired summary. Learned run:
`go2_arrival_entry_priority_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_learned_round_trip_native_layout04_4800_v1_attempt_001`.
Reactive run:
`go2_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_reactive_round_trip_native_layout04_4800_v1_attempt_001`.
The learned root includes `verified_round_trip.png` and `.svg`; both roots include
independent physical arrival evaluations and native-navigation summaries.

The unchanged learned controller was subsequently launched on development
layout 6 with `--layout-index 6 --arrival-entry-priority`, after both comparison
owners exited. No model refit, controller edit, tolerance change or budget
extension was made. The transfer owner exited 0 and physical evaluation passed
both arrivals: outbound frame 1,472 and return frame 2,415. Maximum physical
distance during the one-second dwells was 15.53 mm outbound and 20.88 mm home;
both dwells passed the zero-command and motion checks. There were no disallowed
contact samples. Median/maximum position error was 3.86/6.33 mm, and 572/579
plans arrived on time. The timed run took 242.43 wall seconds. Its full recording
and `continuous_native_arrival_evaluation.json` are retained under
`go2_arrival_entry_priority_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_learned_round_trip_native_layout06_4800_v1_attempt_001`.

The unchanged controller therefore has verified round trips on two development
layouts, one trial each. This is transfer evidence within development, not a
repeatability or unseen-maze reliability result.

The matched layout-6 reactive owner also exited 0. It exhausted the 4,800-tick
budget with no arrivals and no disallowed contact samples. It briefly came
within 11.84 mm of the goal but did not complete the required arrival dwell;
final physical goal distance was 65.66 mm. Median/maximum position error was
5.99/12.44 mm and 1,189/1,200 plans were on time. Physical path length was
17.835 m, compared with the learned round trip's 17.902 m. This failure is
terminal convergence rather than the distant clearance deadlock seen on layout 4.
The common settings and source hashes passed the comparison helper again.
The paired result is preserved under
`go2_arrival_entry_stable_reference_pulse_matched_comparison_layout06_v1_attempt_001/result.json`.
Both complete recordings are retained.

There are now two matched development pairs: learned round trips 2/2, reactive
round trips 0/2, one trial per layout. This small, development-exposed sample is
not a reliability estimate. The next fixed-controller cohort uses four fresh
topologies; see `go2_fresh_fixed_controller_transfer_2026-09-14.md`.
