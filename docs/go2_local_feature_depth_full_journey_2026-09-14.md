# Full recorded journeys with local correspondence depth

After the fixed eight short-replay assignments finish, extend the same local
feature-depth estimator to the entire four persistent-routing recordings.
These are the recorded successful development journeys, not fresh layouts or
counterfactual noisy navigation. No controller decisions are executed in replay.

Fix all eight assignments: 2-mm independent Gaussian depth noise on layouts
0–3, then zero-noise controls on layouts 0–3. Frame counts are respectively
3,106, 3,525, 3,105 and 3,027, matching camera metadata and saved original pose
counts. Keep seed 2026091414, the original perturbation helper, gyro-initial
camera consensus, local floor and feature depth estimation, consecutive retained
acquisition correction, all measurement thresholds and bridge limits unchanged.
Do not alter the estimator in response to any failure in this population.

Launcher: `scripts/replay_go2_local_feature_depth_full_journey_development.py`.
Each recording owns an exclusive
`depth_noise_{0|2}mm_local_feature_depth_full_journey_v1/` directory. It saves
source hashes, estimates, floor availability, every failure and whole-recording
completion. After estimation, compare the first 601 rows exactly against the
completed short replay and compute position error against physics. A partial
matching prefix must not be described as full prefix reproduction.

Use four independent CPU replay owners with the established disjoint groups and
single-thread libraries. The observed short noisy replays took 71–76 seconds for
601 frames at roughly 1.2 GiB RSS each. A rough extrapolation is 6–8 minutes for
the longest recording per four-run batch; this is an estimate, not a timeout or
a restart condition. No sensor archives are duplicated. Recheck capacity at
dispatch and monitor actual progress; preserve each owner's handle until exit.

Full replay survival would justify a live noisy-sensor navigation experiment.
It would still not establish noisy mapping/planning performance, realistic gyro
or RGB degradation, deployment timing, calibrated hardware noise or real-robot
success.

All eight short assignments have now exited successfully: 4/4 noisy and 4/4
clean prefixes complete, each 601 frames. Before full-study dispatch, CPU and
both GPUs were idle, available RAM was 76 GiB and artifact-volume free space
21 GiB. The four full noisy owners launched on the declared disjoint groups.
The full zero-noise batch remains scheduled after all four noisy owners exit.

## Full noisy recordings complete

All four noisy owners exited 0 with every recorded frame accepted: respectively
3,106 / 3,525 / 3,105 / 3,027. Every accepted frame had a floor plane, and all
four 601-frame short prefixes were reproduced exactly. Maximum registered
position errors were 5.821 / 5.534 / 7.372 / 14.133 mm; median errors
2.688 / 2.189 / 2.571 / 4.331 mm. Elapsed times were 361–409 seconds.
The four full clean controls launched next, with the same source and settings.

This supports tracking over the entire recorded journeys under synthetic
2-mm independent depth noise. The next experiment must exercise actual online
commands with noisy depth reaching tracking, registration, mapping and the
independent obstacle observer. Mapping and independent obstacle estimation
have not yet adopted local depth estimation; preserve that distinction.

## All eight full recordings complete

The four clean controls also exited 0 after all 3,106 / 3,525 / 3,105 / 3,027
frames. Their maximum errors were 7.221 / 6.391 / 5.402 / 10.241 mm; median
errors 3.970 / 2.561 / 1.869 / 6.097 mm. Clean runs took 331–369 seconds.
All eight runs reproduced their entire 601-frame short prefixes exactly,
shared identical source-hash sets and had floor estimates on every frame.
Combined result: `go2_local_feature_depth_full_journey_four_layout_summary_v1_attempt_001/result.json`.

The full replay experiment is complete: 4/4 noisy and 4/4 clean journeys.
The live follow-up is fixed in `docs/go2_live_local_feature_depth_noise_2026-09-15.md`;
its first two noisy native owners have launched. Replay survival is not a
live-navigation result.

Full-duration error curves (`position_error.png` and `.svg`) were generated
and visually inspected in the combined result directory, with a common
0–15-mm vertical range covering every trajectory. Layout 3 illustrates the
limit of the short test: its noisy maximum rose from 3.978 mm in the first
601 frames to 14.133 mm at frame 2434, and the last 601-frame median was
11.686 mm. Other noisy full maxima remained 5.534–7.372 mm. This is one fixed
noise realization on each revisited recording; lower noisy errors on some
layouts do not establish that noise is beneficial. Per-layout/time-window
statistics are saved in `error_time_profile.json`.
