# Image-support sensitivity experiment

Two recent public-sensor failures isolate different support thresholds:
the fine-stored-obstacle run retained 21/26 matches but covered five current
image cells rather than six; the memory-forecast run covered six cells but
retained 13/23 matches, below the 60% fraction requirement.

This explicit development variant requires five image cells and a strict
majority of inliers. It retains the absolute minimum of 12 matches, robust
proposal/refit/pruning algorithm, point and reprojection residual limits,
3-D conditioning, gyro and motion envelopes, temporal consistency, floor
registration and plane/image conflict checks. Accepted-support reference
promotion still triggers at seven cells, preserving its previous threshold.
The old tracker is unchanged. This is a support sensitivity experiment, not
a claim that relaxed support provides a calibrated pose-error bound.

Replay of all 289 frames in the failed memory-forecast trajectory completed.
Every pose before the previous frame-284 failure matched exactly. Median/max
post-estimation position error was 6.95/9.68 mm. No native state was read until
estimation finished. Evidence:
`go2_memory_forecast_clearance_native_layout00_v1_attempt_001/development_support_replay.json`.

The five-cell variant passes the previous frame-341 failure but fails at frame
344 of the 346-frame clip. It therefore does not resolve that trajectory.
On the completed 1,805-frame frontier trajectory it reproduces every original
raw pose exactly and completes, with median/max position error 3.05/8.86 mm.
No prospective native run of this five-cell variant has been launched.

A second explicit ablation removes image-bin counting while retaining the
original measured 3-D conditioning: second scatter RMS at least 2 cm and at
least 5% of the first, plus the unchanged point/reprojection residual, gyro,
motion, plane and temporal checks. It also requires a strict inlier majority
and at least 12 matches. This tests whether image-bin occupancy contributes
useful rejection beyond directly measured spatial conditioning. It does not
claim a calibrated uncertainty bound.

The conditioned-support variant completes both previously failed clips:
346 frames with median/max position error 4.44/9.63 mm (first changed raw pose
at frame 340), and 289 frames with 6.95/9.68 mm (every previously admitted raw
pose unchanged). A focused test accepts an exactly registered, conditioned
patch inside one image bin and rejects its collinear counterpart; it passed
in 1.97 s. The full 1,805-frame conditioned-support replay completed and
reproduced every original raw pose exactly. Its median/max position error was
3.05/8.86 mm, identical to the original result on this trajectory.
Evidence filenames: `conditioned_support_replay.json` within each original
artifact directory.

A fresh prospective experiment was run in
`go2_conditioned_support_native_layout00_v1_attempt_001`, session 38206.
It combines the conditioned-support tracker with the existing stored-map
forecast filter, measured frontier visits, fine obstacle memory and original
1,800-tick mission. It failed visual tracking at frame 58, after acquiring
66 frames and requesting 326 policy steps. All 66 camera pairs were saved;
the process exited with failure and no physical contact stop. Thirteen of
14 plans were on time, final observed goal distance was 2.582 m, and no
frontier visit completed. This prospective result does not establish robust
navigation despite the positive earlier replays.

Replaying the new clip with both the original optional-plane tracker and the
conditioned-support tracker reproduced exactly the same 58 accepted poses
and the same failure: insufficient rigid consensus after pruning. Median/max
position error was 3.24/6.01 mm. Thus the support-threshold ablation did not
cause or resolve this rejection. The next comparison tests 100, 150 and 300
features on the same 66 frames, with original support criteria, optional
plane refinement and partial-height registration held fixed (session 48288,
now completed). The 100-feature arm failed at frame 58; both 150 and 300
completed all 66 frames. Median tracking times after the first four frames
were 64.00, 66.17 and 75.18 ms respectively. Maximum post-estimation position
errors were 6.01, 6.43 and 4.19 mm. The 150-feature arm adds only 2.17 ms in
this replay and is the next native candidate. These short-clip times do not
establish live deadline performance. Evidence:
`paired_tracker_registration_diagnostic.json` and
`paired_tracker_accuracy_summary.json` in the conditioned-support native
directory. All jobs are terminal; no next native experiment has launched.
