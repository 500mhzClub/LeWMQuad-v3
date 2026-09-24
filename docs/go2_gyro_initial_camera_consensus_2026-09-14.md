# Gyro-conditioned initial camera consensus

The consecutive-retained correction completed 2/4 noisy 601-frame recordings,
with 4/4 zero-noise controls. At layout 0's remaining frame-397 failure, a
successful unconstrained joint fit was rejected by subsequent floor refinement.
A paired public-sensor probe instead started with gyro-conditioned fitting and
passed the unchanged floor refinement and final gyro consensus (17/18 matches).

Test `lewm/gyro_initial_camera_consensus_development.py` in full sequences.
Initial proposals use gyro rotation in single and pooled cameras, followed by
the existing floor refinement, final gyro consensus and temporal admission.
Retain local inverse-depth floor estimation and the consecutive-retained
acquisition correction. Match populations, minimum count, strict majority,
spatial conditioning, calibrated camera reprojection, residual/motion limits,
reference promotion, bridge allowance and original raw feature depth remain
unchanged. This is an estimator change, not weaker acceptance thresholds or a
claim of calibrated gyro accuracy.

The intended method chain imports and a 12-frame noisy public-sensor tracker /
floor-registration startup check passed before this study. Fix eight assignments
before reading outcomes: all four layouts with 2-mm noise, then all four with
zero noise, frames 0–600 and original perturbation seed 2026091414. Preserve
all results and do not change source or numerical settings between assignments.
The completed consecutive-retained population is the matched control.

Launcher: `scripts/replay_go2_gyro_initial_noise_development.py`.
Each persistent-routing recording owns an exclusive
`depth_noise_{0|2}mm_gyro_initial_tracking_601_v1/` directory. Preserve full
terminal snapshots, estimates, floor availability, source identities and pose
errors scored against physics only after estimation ends. No sensor archive is
duplicated. This remains synthetic-noise fixed-input replay, not closed-loop
navigation or hardware qualification.

Before launch, CPU and both GPUs were idle; 76 GiB RAM and 21 GiB artifact-volume
space were available. Previous replays used about 1.5–1.7 GiB each and one busy
CPU core. Run four independent replays concurrently for this study, with one
numerical-library thread each and disjoint CPU groups 0–3/16–19, 4–7/20–23,
8–11/24–27 and 12–15/28–31. This uses substantially less than available RAM and
does not change recorded timestamps or numerical sample order. Record actual
affinities and monitor throughput; keep native simulation concurrency separate.

## Noisy assignments complete

All four owners exited 0. Layouts 0/1/3 completed 601 frames, maximum position
errors 7.472/6.427/5.554 mm; layout 2 stopped at frame 221 with neither an anchor
nor a previous-frame pose available (maximum accepted-prefix error 5.571 mm).
Floor planes were available on every accepted frame. Selected consecutive
retained pooled fits numbered 7/7/5/10. Full noisy survival improves from 2/4
to 3/4 against the completed consecutive-retained control.

Four-way replay ran successfully: observed owners used about 1.3–1.4 GiB RSS
and one core each, leaving 74 GiB available RAM after the short run exited.
Completed 601-frame replay times were 68–69 seconds, comparable to the previous
two-owner runs. The four zero-noise controls launched next unchanged.

## All eight assignments complete

All four zero-noise controls completed 601 frames, with floor measurements
throughout and maximum position errors 5.406/8.003/4.816/4.498 mm. All eight
owners exited 0 and all recorded source-hash sets agree. Combined results are
in `go2_gyro_initial_noise_four_layout_summary_v1_attempt_001/result.json`.

| Layout | Corrected-reference control noisy frames | Gyro-initial noisy frames |
| --- | ---: | ---: |
| 0 | 397 | 601 |
| 1 | 601 | 601 |
| 2 | 198 | 221 |
| 3 | 601 | 601 |

Full noisy survival improves from 2/4 to 3/4, with 4/4 clean controls retained.
This remains a 60-second fixed-recording sensitivity result. The estimator has
not been selected for native navigation or demonstrated on full noisy journeys.

## Remaining frame-221 rejection: feature-depth sensitivity

A public-sensor probe compared the same adjacent 220/221 image pair at zero and
2-mm noise, using the existing gyro-conditioned fit. Original-depth primary
direct flow passed with 12/15 inliers; auxiliary descriptors passed with 16/16,
and auxiliary direct flow with 28/35. Under noise these fits rejected. On the
last tested transformations all 15/16/35 correspondences respectively passed
the 3-D residual bound, but only 10/10/13 passed both image reprojection checks.
The final consensus can be a subset of that last mask because pruning is
monotonic; the diagnostic does not relabel the last mask as an admitted fit.
Saved evidence: `adjacent_fit_rejection_diagnostic.json` beneath layout 2's
`depth_noise_2mm_gyro_initial_tracking_601_v1/` output.

A separate pair probe applied the already defined 5x5 local inverse-depth
estimator to feature depth, preserving invalid-neighborhood/surface-jump
exclusions. With 2-mm input noise, all three fits recovered exactly the clean
inlier counts: primary direct flow 12/15, auxiliary descriptors 16/16, auxiliary
direct flow 28/35. Primary descriptors still lacked enough matches. Clean
filtered-depth counts matched the clean unfiltered counts on this pair as well.
All eight outcomes are retained in `local_feature_depth_pair_probe.json` in the
same output directory. No threshold changed and no physics was read.

The next experiment should test locally estimated feature depths across all
four sequences, keeping the current floor treatment and gyro-initial estimator.
Avoid applying the floor filter twice; retain original camera packets and
explicitly distinguish estimated feature geometry from raw depth. The pair
result is not evidence of sequence survival or closed-loop robustness.
