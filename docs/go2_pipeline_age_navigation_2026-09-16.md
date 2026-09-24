# Observation age experiment

Run one complete supervised-rollout mission on exposed short-pulse maze 1
with a 250-ms maximum obstacle observation age. This intentionally changes
the original 200-ms bound. It is a development test, not calibrated sensing
or a physical safety qualification. Restore the reference parent Python
switch interval of 5 ms and use the original obstacle observer.

The 100-ms camera cadence and roughly 114-ms median observation-to-completion
latency imply observation age can reach about 214 ms between updates; the
95th-percentile completion age was 124 ms. A 250-ms bound covers that typical
cadence/delivery gap plus one 20-ms command interval, while still rejecting
longer delays. On the exact 20-ms request grid, the last admissible age is
240 ms and age 260 ms remains rejected. This is a prospective parameter
choice motivated by observed delivery, not an empirical worst-case guarantee.

The existing translation stopping connector already uses remaining command
time plus actual observation age plus its unchanged 0.5-s stopping allowance.
The 45-cm nominal footprint, predicted-path reserves, missing-primary
translation veto, commitment-prefix checks, measured arrival, original goal,
learned model, sensor noise and 4800-tick budget remain. Four focused tests
passed, covering the newly admitted gap, longer-delay and absent-observation
stops, obstacle/missing-primary vetoes, and inherited commitment checks.

Planning's stopping projections sample observations at offsets 100 through
600 ms for a 300-ms-delayed, 400-ms command (100 through 300 ms for terminal
translation pulses). With the unchanged 100-ms camera and 20-ms request grids,
increasing age to 250 ms admits no earlier camera tick than those already
sampled: a camera observation at offset zero is still too old at the first
300-ms dispatch. Actual-age charging is retained, not replaced by a fixed
200-ms projection. None of these nominal projections is a calibrated
whole-body collision or stopping certificate.

Compare with the completed fresh reference: 0 arrivals, 0 contacts, 726 stale
triggers, 885/1200 on-time plans and minimum physical goal distance 1.459 m.
The grouping and 1-ms thread-switch attempts are separate retained negative
outcomes, not successful fixes. Keep this outcome regardless of success.
Primary measures: physical arrivals, contacts, progress and command continuity;
also inspect actual accepted observation ages and stopping allowances. One
exposed layout does not establish general reliability or JEPA benefit.

Launcher: `scripts.run_go2_pipeline_age_navigation_development`; evaluate with
`--evaluate` after owner exit and full persistence. Output:
`go2_pipeline_age_250ms_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.

Launched in session 97247, PID 3909785. Native launch and the live owner were
confirmed at 20 seconds elapsed; available space before launch was
4,382,846,976 bytes. The final four focused tests passed in 1.80 s, including
an explicit stopping-margin veto. Heavy analysis is paused during simulation.
Poll this session/owner and preserve this attempt; evaluate only after actual
exit and complete persistence. The evaluator additionally checks actual
accepted request ages and actual-age charging for every accepted translation.
The broader navigation goal remains active and incomplete.

## Terminal result and next diagnosis

The owner exited 1 after `queue.Full` in the tracking input queue at 1952
acquired camera frames and 9760 requested command steps (about 195 simulated
seconds). All acquired sensors and partial runtime records were persisted.
Physical evaluation found no arrivals and no contacts; this is a failed
mission, not a successful navigation or a completed 480-s budget. Preserve
the full depth recording as an active tracking-throughput diagnostic input.

Actual age treatment passed: 4480 accepted requests, 186 older than 200 ms,
maximum accepted age 240 ms, and every accepted translation charged actual
age to the stopping connector. There were no stale-observation or latched
veto requests. Other requests were 25 settling and 5255 without an on-time
plan. The opening survey finished in 26.4 s versus reference's 158.8 s.
Of 479 saved selections, 224 were on time. Minimum physical goal distance
was 1.412 m, final distance 3.332 m and path length 6.948 m. Matched registered
position error was 2.900 mm median and 5.246 mm maximum across 1918 poses.

Recorded tracking stage elapsed time (simulation clock) increased from a
66-ms median in frames 0–499 to 106 ms in frames 1700 onward, exceeding the
100-ms acquisition cadence. Median obstacle stage time stayed 30 ms. At
camera frame 1900 the published pose was frame 1874. The shutdown tracking
fault about a closed simulation clock followed the queue overflow; it is not
the original failure. The exception path did not persist measured-service
release records, so stage elapsed time must not be mislabeled worker CPU time.
A two-line source fix now retains those records on future failures as well;
it does not restore the missing records from this attempt.

`scripts.profile_go2_pipeline_age_tracker_development` is replaying all 1952
saved public sensor packets with the unchanged tracker, comparing every
recorded pose exactly and profiling early/late windows. Its output is
`go2_pipeline_age_tracker_replay_v1_attempt_001`. No native mission is active.
Do not retire this input until the throughput diagnosis and selected repair
have been completed. The 250-ms change improved command continuity in this
execution, but did not demonstrate mission completion or general reliability.

The unchanged-tracker replay completed all 1952 frames in 223.16 s, exactly
reproducing position, rotation, reference selection and mode for all 1918
recorded poses. Unprofiled median tracker compute rose from 51.785 ms in
frames 0–499 to 93.804 ms in frames 1700 onward (95th 102.339 ms). In that
last unprofiled group, 200 old-view probes were unselected and only two were
selected. The late 50-frame profile recorded 406 candidate fits and 3.796 s
inside reference choice; total profiled tracking was 5.892 s. Active references
remained bounded at eight: the increasing keyframe counter counts historical
promotions, not an expanding retained reference population.

A new isolated candidate in `lewm/cadenced_view_revisit_tracking_development.py`
attempts the optional old-view bank probe every four frames (400 ms), while
tracking every camera/gyro frame and keeping original recent/stable-reference
acceptance rules. Its complete public-input replay is running via
`scripts.replay_go2_cadenced_view_revisit_development`, session 35595, output
`go2_cadenced_view_revisit_replay_v1_attempt_001`. This changes reference
selection opportunities; numerical equivalence is not assumed. After it
finishes, independently evaluate saved poses against native trace truth and
compare late-frame compute before any prospective navigation test. Preserve
the original queue failure and its full sensors. No native mission is active.

The cadenced candidate replay completed all 1952 packets in 205.56 s. Its
separate truth evaluator found raw position error 1.599 mm median, 3.666 mm
95th percentile and 5.098 mm maximum; maximum difference from the 1918 recorded
raw positions was 3.587 mm. On the exact same late unprofiled frame set,
original/candidate mean compute was 94.678/61.318 ms, median 93.804/52.035 ms,
and 95th percentile 102.339/97.511 ms. This supports a prospective native test,
not a claim of guaranteed deadlines. The fixed next mission is documented in
`docs/go2_cadenced_view_navigation_2026-09-16.md`; this failed run remains full.
