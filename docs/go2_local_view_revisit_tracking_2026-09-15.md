# Local viewing-reference retention experiment

## First native result

The exposed learned maze-3 run exited successfully after 4,805 accepted poses
and exhausted the unchanged 480-second mission budget during its return.
It reached a physically verified goal at frame 3076: quiet-dwell maximum
distance 16.879 mm, maximum 100-ms speed 0.01014 m/s, with zero requested
commands throughout the dwell. There were no disallowed contacts or pipeline
failures. The immediate committed-view predecessor's goal was invalid, with
physical distance up to 62.578 mm.

Median/maximum position errors were 13.309/19.069 mm, versus predecessor
39.716/47.400 mm. Native path length was 30.246 m and final distance from home
2.029 m. There were 1,196 selected plans, 1,167 on time. Wall time including
recording was 11:44.91, maximum RSS 25,550,484 KiB, with no swap. All four
standard evaluations are saved. The 1,131 overlapping executed windows had
corrected XY RMSE 6.436 mm and maximum error 29.710 mm; learned yaw RMSE was
3.981 degrees versus the saved command alternative's 2.683 degrees.

This is live evidence of improved localization and verified goal-reaching on
one exposed development maze. The alternative trajectory avoided terminal floor
rejection, but the independent replay still shows that the estimator does not
eliminate that rejection on the original recording. Return completion,
repeatability, fresh-layout transfer and learned-model advantage remain unproven.
Keep this complete recording for the remaining viewing/return-time diagnosis.

The pair report is `go2_local_view_revisit_layout03_summary_v1_attempt_001`.
It checks 148 identical common source bindings, shared settings and the actual
final learned XY/yaw channels. All 21 completed frontier-view events ended in
actual patch observation. The longest event decreased from 177.2 to 76.8 seconds,
but total turn-only command time was 317.90 seconds over the longer run, versus
249.76 before. The longest new event (46.3–123.1 s, target [11,23], unknown
neighbour [10,23]) never entered the committed-view branch. Diagnose approach
and action-selection behavior there before assuming further turn-commitment
changes are needed. Comparisons have different asynchronous trajectories and
historical host loads.

## First native test specification

After both committed-view learned follow-ups terminated and were evaluated,
test the frozen retained-reference estimator in one native learned maze-3 run.
Use `scripts/run_go2_local_view_revisit_native_development.py`, writing
`go2_local_view_revisit_learned_noise_2mm_native_layout03_4800_v1_attempt_001`.
The immediate comparison is the completed committed-view maze-3 run, whose
observed goal was physically invalid and whose return hit a floor conflict.

Only the pose initializer changes to `LocalViewRevisitMotion`. Keep committed
views, learned corrected XY/yaw, model/fits, disabled contact scoring, sensors,
maze, timing, mission budget and physical criteria unchanged. Floor rejection
and all visual pair/continuity checks remain active; do not combine floor
reacquisition with this first live estimator test. One assignment is fixed
before collection. This exposed case can test live drift and arrival behavior;
it cannot establish fresh-maze reliability. Evaluate the recorded physical
arrivals and retain the outcome, including any failure.

The native run is launched in session 24722 on CPUs 8–15,24–31. Its actual
launch record confirms `LocalViewRevisitMotion`, learned XY/yaw, committed
camera views, one planned assignment and disabled floor reacquisition. The
pose initializer uses the exact estimator already tested in both offline
replays. The completed native outcome is reported above.

## Second completed replay

The original learned layout-3 replay has now finished: 3,353 of 3,355 captured
frames were accepted, with the same floor-reference conflict at frame 3353.
Registered position error was median 2.139 mm, maximum 7.562 mm and final
6.080 mm, versus original median 44.344 mm and maximum 53.347 mm. The corrected
recorder counted 991 retained-reference attempts and 651 selections. Replay
elapsed time was 432.94 seconds; native state was loaded only after estimation.

Reference retention reduced drift on this second recorded trajectory but did
not resolve its terminal floor-registration rejection. This is offline estimator
evidence, not a new navigation success or a revision of the original false goal
claim. Both native committed-view learned follow-ups continue with the original
tracker; no native run yet uses the retained-reference variant.

## First completed replay

The complete original pose/command layout-1 replay accepted all 4,805 frames,
with no terminal failure. Registered position error was median 1.791 mm,
maximum 9.240 mm and final 5.035 mm, versus original median 17.129 mm and maximum
23.832 mm. The first 248 raw poses match the original; later estimates differ.
Elapsed replay time was 570.72 seconds. This supports a drift reduction on one
fixed recorded trajectory; it does not establish a new navigation outcome,
runtime timing qualification or general reliability.

The original result's attempt/selection counters were invalid because the raw
pose wrapper omitted those fields. The original result is preserved as
`result_original_attempt_fields_unavailable.json`; corrected `result.json`
marks both counters null and explains the instrumentation gap. Acceptance and
position-error results are unaffected. No replay was repeated merely to repair
these counters. Future replay rows read model-level attempt metadata directly.

Next, test the same frozen estimator on the original learned layout-3 recording,
whose 53 mm drift produced a false goal claim and which later failed floor
registration. The full 3,355-frame estimator replay is now launched separately;
native learned-controller viewing follow-ups continue with the original tracker.

The second replay is session 17133, writing
`go2_combined_perception_motion_learned_noise_2mm_native_layout03_4800_v1_attempt_001/gyro_coherent_floor_local_view_revisit_replay_v1/`.
Its attempt metadata uses the corrected row-level recorder. The terminal result
and accepted prefix are retained; the completed outcome is reported above.

## Design and collection history

The completed four-maze comparison's `pose_drift_sources_v1.json` separates raw
visual odometry from subsequent floor registration. Across all eight recorded
trajectories, the maximum horizontal floor-registration change is below 0.066 mm.
The raw tracker already contains the observed 52–53 mm errors on learned
layouts 2/3. This points to the visual odometry stage for the next drift study.
It does not prove which correspondence or reference decision caused the drift.

The current tracker keeps a stable preferred reference and seven recent ones.
It changes the preferred reference when translation exceeds 0.4 m, rotation
exceeds 0.35 rad, or the preferred reference loses support. Repeated turns can
therefore discard earlier views of the same local place. The separate
`lewm/local_view_revisit_tracking_development.py` tests a bounded alternative:

- Retain up to eight additional accepted image/depth/plane references, one per
  45-degree viewing bin. Preserve the earlier reference in a bin until a new
  accepted position is more than 0.30 m from it.
- Consider an inactive retained view only after one second, within 0.20 m of
  the preceding estimated position and 0.20 rad of the current measured gyro
  orientation. Try the oldest eligible reference first.
- Temporarily include that reference in the existing at-most-eight candidate
  population with its exact floor evidence. All original image/depth fitting,
  gyro, increment and anchor/increment disagreement checks still apply. Restore
  the ordinary active reference and plane populations after measurement.
- Record every attempt and actual selection. Retaining or proposing a reference
  is not an accepted pose, an uncertainty bound, or a loop-closure guarantee.

Two focused reference-selection/binning tests passed in 1.65 s. No native run
uses this variant. The fixed offline replay is the complete 4,805-frame original
pose/command layout-1 trajectory, including its long viewing episode and return.
It uses the original delivered noisy packets, and loads native physics only
after estimation for evaluation. The original 23.832 mm maximum registered
error and goal-only outcome remain the reference.

Run via `scripts/replay_go2_gyro_coherent_floor_development.py` with
`--variant local_view_revisit` and root
`go2_combined_perception_motion_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`.
Session 97965 has completed; output is
`gyro_coherent_floor_local_view_revisit_replay_v1/` under that root. No final
accuracy or acceptance claim extends beyond the completed replay reported above.
Do not alter this variant during subsequent replays or combine it with the
pending learned-controller viewing follow-ups. This remains an estimator replay,
not a closed-loop navigation test.

Instrumentation note during the first replay: the motion wrapper's public raw
pose omits the added model-level attempt metadata. The original running script
therefore cannot count attempts/selections from that pose, although changed raw
reference/position records prove the variant is active (2,180 of the first
2,428 poses differ). Its zero attempt/selection counters must be marked
unavailable after completion, preserving the original result. The analysis
script is corrected for future runs to read `motion.model.last_revisit_attempt`
directly into each row. No estimator source or parameters changed during the
running replay; pose acceptance, recorded trajectories and final physical error
evaluation remain usable.
