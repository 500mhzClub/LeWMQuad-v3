# Gyro-consistent paired floor fitting

The completed noisy training comparison's JEPA layout-3 recording failed at
frame 441 despite fully valid primary depth. Exact replay reproduced 441/441
published raw poses. The final gyro translation refit applied offsets from two
independently oriented measured floor planes while using the gyro rotation.
For the consecutive auxiliary pair, 14/14 original matches passed the raw gyro
fit; its 1.126-mm height adjustment left 11, below the unchanged minimum 12.
Other attempted pairs also lost support after the floor adjustment. Evidence
is under `terminal_consensus_trace_v1` in
`go2_current_plane_matched_training_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.

The development revision fits one normal to both original raw floor patches,
transporting it by the same measured gyro rotation used for image fitting.
Separate measured offsets then belong to mutually consistent normals. Raw
point selection begins with the existing height-selected clouds. Bounded
monotonic pruning retains the original 3-mm maximum point residual, minimum
100 points, minimum quarter of the original raw pool, original two-axis extent
and .97 up alignment for each patch. Excluded point indices are recorded.
The independent original plane receipts are preserved. Image-match minimum,
strict majority, reprojection, motion and continuity thresholds remain unchanged.
This is a different estimator, not a calibrated uncertainty bound.

Two focused tests cover rotated/translated plane height recovery and rejection
of collinear support; both pass. On the same complete 450-frame saved recording:

| Estimator | Accepted registered frames | Failure | Median/max position error |
| --- | ---: | --- | --- |
| Original baseline | 441 | Raw tracking at frame 441 | 1.813 / 3.892 mm |
| Gyro-consistent floor | 450 | None | 2.289 / 4.503 mm |

The baseline reproduces all 441 recorded raw poses exactly under the edited
shared functions' unchanged default path. These error populations differ in
length; the revised median/max are not an accuracy-improvement claim. Native
state was loaded only after estimation. The nine additional recorded frames
were acquired during the original terminal stop, so this is recovery of saved
sensor tracking, not continuation of the original navigation trial.

Next run one prospective JEPA layout-3 native diagnostic with the revision,
the same frozen model/correction, mapping, independent obstacle observer, 2-mm
noise recipe, six actions, timing, arrival tolerances and 4,800-tick budget.
Use `scripts/run_go2_current_plane_matched_training_noise_development.py
--layout-index 3 --condition jepa --gyro-coherent-floor` on the original odd
CPU group. Output is
`go2_gyro_coherent_floor_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
It is a new development treatment on an exposed layout. It never replaces the
original failure or changes the completed twelve-assignment comparison.
No floor-reacquisition mechanism or new layout is introduced in this probe.
Realistic gyro calibration, timing qualification and hardware remain open.

The live probe launched successfully in session 53086 on CPUs 8–15,24–31.
Its saved launch identifies `GyroCoherentFloorMotion`, the unchanged JEPA
assignment and residual-fit root, and the original failed trial as reference.
Native navigation has begun; outcome is pending. Concurrent session 54833
replays the revised estimator over the complete previously successful current
training JEPA layout-1 recording on CPU 0. That longer replay checks accumulated
drift on a different trajectory while the separate CPU group runs navigation;
it does not change the original successful trial or its model/control decisions.

## Live probe: physically verified round trip

The owner exited 0 after 321.61 s including archive, maximum RSS 12,794,424 KiB
and zero swaps. Independent physics evaluation verifies goal frame 1197 and
home frame 2140. Maximum one-second dwell distances were 18.887/17.225 mm,
maximum 100-ms dwell speeds 0.010233/0.022733 m/s, with every dwell request zero.
All 2,142 acquired frames published registered poses. Median/max position
error was 2.786/5.473 mm; zero disallowed contact samples. Selected/on-time plans
527/517. This new live treatment succeeds on an exposed layout; it neither
replaces the original failed trial nor establishes general reliability.

The next experiment is the fixed new-maze comparison specified in
`docs/go2_post_training_comparison_transfer_preparation_2026-09-15.md`.

The live probe travelled 19.696 m and finished 16.928 mm from home. All 527
selected-plan correction bindings match the frozen JEPA fit. Its 515 fully
executed 700-ms windows have corrected endpoint XY RMSE 8.024 mm, maximum
23.600 mm, and zero windows with any corrected-path error over 30 mm. These
are overlapping trajectory-conditional errors, not a reserve certificate.

The longer revised-estimator replay also completed: 3,574/3,574 frames on the
saved current-training JEPA layout-1 successful trajectory, no tracking or
registration failure, 417.29 s on one CPU core. Registered position error is
median 3.717 mm, maximum 7.182 mm, final 5.602 mm. The original live estimator's
same recorded-frame population had median 5.115 mm and maximum 9.246 mm.
This is a same-sensor-trajectory estimator comparison; the revised estimator
did not control that historical trajectory. Native truth was loaded only after
estimation. Full replay result and per-frame records are under
`gyro_coherent_floor_coherent_replay_v1` in the original layout-1 root.
