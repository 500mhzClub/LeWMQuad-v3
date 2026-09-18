# Consecutive retained-camera pair eligibility

The local-floor replay retained frame 197, then used ten consecutive measured
increments and stopped at frame 208. On its public 197/198 pair, pooled chained
flow passes the existing geometric fit with 32/48 inliers (9 primary, 23
auxiliary), whereas pooled descriptors fail. The pooled-camera code allows a
one-interval fit for the `previous` object but excludes a distinct retained
Reference for the same acquisition; retained chains otherwise require at least
two intervals. Test whether that distinction causes avoidable anchor loss.

`lewm/consecutive_retained_camera_pair_development.py` permits the existing
consecutive path for a retained reference only when its frame and timestamp,
feature-object ownership, position, rotation and gyro match `previous` exactly.
It fits using that existing consecutive path, preserves all plane/refit/temporal
checks, then restores actual retained ownership. It does not increase the bridge
allowance, promote a bridge pose, reset history, change residual thresholds or
enable the gyro-proposal variant. Selected pooled fits receive explicit metadata.

Use the completed local-inverse-depth tracker/registration study as the matched
control. A 12-frame noisy tracker/registration startup check passed. Fix all eight
601-frame assignments before reading outcomes: 2-mm noise on layouts 0/1, then
2/3; zero-noise controls on layouts 0/1, then 2/3. Noise seed 2026091414, camera
packets, feature depth, gyro and timestamps are unchanged. Keep the source fixed
through all assignments and preserve every failure.

Launcher: `scripts/replay_go2_consecutive_retained_noise_development.py`.
Exclusive output beneath each persistent-routing recording:
`depth_noise_{0|2}mm_consecutive_retained_tracking_601_v1/`. Save source hashes,
per-frame selected-adapter use, floor availability, poses and terminal evidence.
Compute physical pose error only after estimation ends. No navigation or
hardware-robustness claim follows from fixed-input replay.

Before dispatch, CPU and both GPUs were idle, available RAM was 76 GiB, and the
artifact volume had 21 GiB free. Use the established two disjoint CPU groups
and single-thread numerical libraries. Recent equivalent replays used about
1.5–1.7 GiB RSS and 25–75 seconds per recording. No sensor archive is duplicated.

The first noisy pair completed and exited 0. Layout 0 accepted 397 frames
(versus 208 for local-floor control), then failed in tracking; 15 selected
poses used the equivalent retained/consecutive pooled path. Layout 1 completed
601 frames, with 16 selected uses. A floor plane was available on every
accepted frame. Accepted-prefix maximum position errors were 8.049 and
9.031 mm. The unchanged noisy layout-2/3 assignments launched next.

## Noisy assignments complete

Layout 2 stopped at frame 198 (four selected pooled uses); layout 3 completed
all 601 frames (16 selected uses). Both exited 0. Every accepted frame had a
floor plane. Their accepted-prefix maximum position errors were 5.307 and
9.590 mm. Full noisy survival is now 2/4, compared with 1/4 for the matched
local-floor control. Zero-noise controls 0/1 launched next; 2/3 remain scheduled.

The saved layout-0 frame-198 pose explicitly selects the equivalent retained
pooled fit and promotes a keyframe, addressing the prior bridge onset. This
does not merely extend the bridge allowance. Its later frame-397 failure instead
rejects an auxiliary image fit as inconsistent with measured floor height.

## Initial fit versus later floor rejection

A focused public-sensor replay of auxiliary descriptor matches on frames 396/397
reconstructed that later rejection. It used the recorded public visual pose and
both floor receipts from the completed run; no physics was read. The existing
initial joint fit admits 15 of 18 matches, then fails floor refinement with
`retained image fit conflicts with measured floor height`. Starting with the
existing gyro-conditioned fit admits 17 of 18 matches and passes both the same
floor refinement and final gyro consensus, retaining all 17. Every original
absolute measurement threshold remains in force.

Results are saved in `initial_fit_before_plane_diagnostic.json` under layout 0's
`depth_noise_2mm_consecutive_retained_tracking_601_v1/` directory. This is a
fully checked pair fit, not a full alternative tracking sequence. Unlike the
earlier gyro-proposal fallback, which only helps when initial joint registration
fails, this case needs gyro conditioning before a successful joint fit is later
rejected by floor constraints. Test gyro-conditioned initial proposals as a
separate estimator hypothesis after finishing the current fixed control roster.

## All eight assignments complete

All owners exited 0; variant source hashes match across every assignment. All
four zero-noise controls completed 601 frames, with 2/0/0/2 selected retained
pooled fits and maximum position errors 4.893/6.901/5.173/4.402 mm. Both control
and variant therefore retain 4/4 unmodified-sequence survival.

| Layout | Local-floor control noisy frames | Corrected noisy frames | Corrected outcome |
| --- | ---: | ---: | --- |
| 0 | 208 | 397 | Image/floor tracking conflict |
| 1 | 601 | 601 | Complete |
| 2 | 198 | 198 | Tracking failure |
| 3 | 529 | 601 | Complete |

This improves full noisy-sequence survival from 1/4 to 2/4 and resolves the
specific observed retained-reference exclusion. It does not establish reliable
noisy navigation. Combined records are in
`go2_consecutive_retained_noise_four_layout_summary_v1_attempt_001/result.json`.
No native controller has adopted the change yet.

The next prototype, `lewm/gyro_initial_camera_consensus_development.py`, applies
gyro conditioning to all initial single-camera and pooled-camera proposals,
before the existing floor refinement and final gyro consensus. It preserves
the retained-acquisition correction, local floor estimator, match populations,
majority/conditioning checks, camera-specific reprojection and temporal/bridge
limits. This differs from the completed fallback-only variant: a successful
unconstrained joint fit is no longer a prerequisite or the default starting
point. The prototype imports with the intended method chain; full-sequence
experiments have not started.

The gyro-initial prototype also passed a 12-frame noisy public-sensor tracker,
floor-registration and current-pose-readout startup check. The next work is a
fixed four-layout noisy/zero-noise sequence comparison against this completed
consecutive-retained study. All current replay owners have exited.
