# Stronger reactive comparison with current noisy perception

The current-plane learned controller has physically verified noisy round trips
on development layouts 2 and 0. Layout 1 is running and layout 3 remains queued.
Those successes do not establish a benefit from learned predictive action
selection or JEPA training. Adapt the existing heading-first terminal reactive
controller to the same current perception and map before making that claim.

Fix four reactive assignments on development layouts 0–3, without tuning or
replacement. Start layout 2 on the even CPU group freed by completed learned
layout 0; subsequently execute 0,1,3 as the corresponding groups become free.
The pending learned layout 3 retains priority on the odd group after learned
layout 1 finishes. At most two native owners run at once, with disjoint
physical CPU groups. This is a subsequent development comparison, not final
evaluation or a claim that the layouts remain unseen after these experiments.

`scripts/run_go2_current_plane_heading_reactive_noise_development.py` uses
`MatchedHeadingReactiveRuntime`, combining the existing
`HeadingFirstReactiveRuntime` and persistent captured-map scope. It uses no
prediction model or motion-residual correction for action selection. The
existing terminal rule aligns measured heading, then uses the existing
100-ms forward pulse; elsewhere it uses the existing instantaneous six-action
feedback and observed frontier strategy. Current depth dispatch checks,
physical stops and measured quiet arrival dwells remain required.

Match the learned assignments' cached visual tracker, raw paired height-floor
registration and independent observer, current-plane/raw-pixel map with
unavailable-plane fallback, accumulated map/obstacles, fixed layout/physics/
appearance and 2-mm noise assignments, image/gyro streams, action bank,
4,800-tick budget, 300-ms dispatch delay, footprint and arrival criteria.
Predictive action scoring, prediction-based recovery/selection and the
heading-first reactive policy differ by design. Report a comparison between
complete predictive and instantaneous controllers, not an isolated JEPA
training effect or a claim that every action-policy detail is identical.

Three focused checks passed using the actual mixed-in reactive selection:
forward pulse, heading correction and hold within arrival radius. A model
sentinel would fail on any prediction-model access; selection returned no
correction and explicitly marked future outcomes unevaluated in every case.
The annotation chain check preserves the learned sensor/map/noise identities
while recording a reactive assignment, no training/correction root, and no
predictive heading/reposition/recovery flags. Actual native launch metadata
and planning receipts must also be inspected; these checks are not navigation
results. No extra full sensor replay is needed because that implementation is
unchanged from the verified learned trials.

Roots are
`go2_current_plane_heading_reactive_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
Run each once, preserve its full result or failure, and independently evaluate
arrivals against physics after owner exit and archive completion. Report
goals, round trips, contacts, pose error, floor availability, selected/on-time
plans, path lengths and terminal reasons for every assignment. Compare against
the corresponding current-plane learned root, including unsuccessful outcomes.
Matched direct/supervised training controls, repeatability, further independent
layouts and realistic hardware evidence remain separate outstanding work.

First reactive outcome (layout 2): owner exited 0 after 674.46 s including
archive, max RSS 25,188,156 KiB, zero swaps. Independent physics evaluation
verifies outbound arrival at frame 1661, maximum dwell distance 5.714 mm,
quiet speed 0.03015 m/s and all dwell requests zero. Return failed at the
4,800-tick budget. All 4,805 poses were accepted, with median/max error
10.937/11.204 mm, zero contacts and no pipeline faults. Selected/on-time
plans 1,195/1,184; sampled path 14.146 m, final distance from home 2.714 m.
All 1,195 selected-plan receipts explicitly exclude model forecasts and
motion correction. The final 684 selected plans (frames 2068–4800) all
hold with current nominal disk unclear: observed clearance 0.42772–0.44856 m
against the 0.45 m nominal radius. The selector excludes every action in
this condition, preventing recovery. See `terminal_reactive_hold_diagnostic_v1.json`.

The paired comparison is saved in
`go2_current_plane_matched_reactive_noise_layout02_comparison_v1_attempt_001/result.json`,
with `sensor_match_and_policy_differences_v1.json`. Actual sensor/cache/map/noise
settings and common source hashes match. Prediction-based recovery and
selection reserves differ explicitly, so this result does not isolate JEPA
training or forecast scoring alone. Learned layout 2 completed both legs;
reactive layout 2 reached its outbound goal earlier but failed the return.

After learned layout 3 ended, reactive layout 1 started on the freed odd CPU
group (session 22345). After reactive layout 2 archived and exited, reactive
layout 0 started on the even group (session 89404). Reactive layout 3 remains
queued after layout 1. No tuning, replacements or extra native owners were
introduced. The learned cohort finished at 3/4 round trips: layout 3 hit
the unchanged full-3D-speed limit, with no contact.

Reactive layout 1 subsequently exited 0 after 343.57 s including archive
(max RSS 13,303,052 KiB, zero swaps). Independent evaluation verifies goal
frame 1551 and home frame 2260, maximum dwell distances 15.847/8.234 mm,
quiet speeds 0.01957/0.01007 m/s and all dwell commands zero. Zero contacts;
all 2,262 poses accepted, floor available 2,261/2,262, median/max position
error 3.535/9.068 mm. Selected/on-time plans 558/554; path 25.455 m.
Both controllers completed layout 1, but reactive reached goal/home earlier
than learned (2612/3543) and followed a shorter sampled path (27.589 m for
learned). This is counterevidence to a blanket predictive-planner superiority
claim. The paired settings/source comparison passed and is saved under
`go2_current_plane_matched_reactive_noise_layout01_comparison_v1_attempt_001`.

Reactive layout 3 started after layout 1 owner exit and evaluation, on the odd
CPU group, PID 3558874 / session 57815. Reactive layout 0 remains active
(PID 3557568 / session 89404). Completed reactive outcomes so far: 2 goals,
1 round trip on 2 finished layouts, zero contacts. Remaining results pending.
The subsequent frozen training-method comparison is prepared separately in
`docs/go2_current_plane_matched_training_noise_2026-09-15.md`; no runs from
that launcher have started, and the reactive roster retains priority.

Reactive layout 3 exited 1 after 79.64 s including archive, max RSS
4,506,748 KiB, zero swaps. A registration fault rejected the transported
floor reference; 409 camera pairs, 405 registered poses, zero arrivals and
zero contacts. Median/max pose error 2.745/3.898 mm; floor available
404/409. The pair comparison preserves both the learned speed-stop failure
and reactive registration failure. It is saved under
`go2_current_plane_matched_reactive_noise_layout03_comparison_v1_attempt_001`.

`scripts/diagnose_go2_current_plane_reactive_floor_conflict_development.py`
replayed actual noisy camera packets using the unchanged cached tracker and
registration on the free odd CPU group (CPU 8). It reproduced every saved
raw pose witness (405/405) and the exact registration rejection at frame 405,
after last accepted frame 404. Runtime 43.93 s; no navigation, physics or
threshold change. Output: reactive layout-3 `recorded_floor_conflict_replay_v1/`.
The current points have insufficient two-axis extent (second covariance
eigenvalue 0.00032238 m² versus 0.0004); all 2,300 selected points come from
the auxiliary camera. Partial height updates by 0.28334 mm from the frame-404
anchor. Against its transported normal, RMS residual is 1.50475 mm and maximum
3.017867 mm; exactly two points exceed the fixed 3.000 mm residual limit.
The original failure latch remains intact. The raw snapshot and full anchor
state are retained to diagnose temporary measurement disagreement. This is
not evidence for widening the limit or accepting uncertain floor as safe.

Reactive layout 0 completed its navigation budget with no observed arrivals
and is archiving in session 89404; it must still be independently evaluated
after owner exit. Its last non-hold selection is frame 1752, with subsequent
nominal-clearance holds and no pipeline fault. The prepared larger training
comparison remains unstarted while this newly exposed perception failure is
under diagnosis.

## Complete fixed comparison

All eight owners have exited and every outcome has been independently evaluated.

| Controller | Verified goals | Verified round trips | Disallowed contacts |
| --- | ---: | ---: | ---: |
| Learned predictive | 3/4 | 3/4 | 0 |
| Heading-first reactive | 2/4 | 1/4 | 0 |

Reactive layout 0 exhausted its budget without an arrival, then exited 0
after 673.37 s including archive, max RSS 25,305,040 KiB, zero swaps.
All 4,805 poses were accepted, median/max error 3.614/10.696 mm. Its path
was 9.468 m and closest/final goal distance 1.258/3.599 m. Selected/on-time
plans 1,200/1,192; zero contacts or pipeline faults. Final behavior was the
same nominal-clearance hold as layout 2. No run was replaced.

The complete paired result, with all four sensor-setting comparisons, floor
counts, faults and trajectories summarized, is
`go2_current_plane_matched_reactive_noise_four_layout_summary_v1_attempt_001/result.json`.
Learned wins round-trip completion on layouts 0/2, both succeed on 1 (reactive
is faster), and both fail on 3 for different reasons. This supports a benefit
for the complete predictive controller on this small development cohort,
including its different recovery rules. It does not isolate JEPA, prove general
reliability, or establish hardware readiness. All native jobs are now stopped.
The immediate next task is the reproduced temporary floor-conflict behavior;
the prepared training-method cohort remains unstarted.
