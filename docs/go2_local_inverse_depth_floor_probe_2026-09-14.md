# Local inverse-depth averaging for noisy floor estimation

The fixed noise study showed drastic floor-candidate depletion. At layout 0's
terminal frame, replacing noisy depth with original depth while holding the
noisy estimated pose fixed left nearly the same 6-mm floor-reference mismatch.
The next hypothesis is that restoring current floor measurements can prevent
that mismatch from accumulating.

`lewm/local_inverse_depth_floor_development.py` estimates center-pixel depth as
the reciprocal of the mean inverse depth in a fixed 5x5 neighborhood. Inverse
depth is affine across an image of a plane, so this symmetric local average
preserves an ideal planar surface at the center. Only entirely valid windows
whose optical-depth range passes the existing depth-lifting surface-jump limit
contribute (0.02 m + 0.01 times mean depth). Unsafe windows are excluded; invalid
depth is never filled. The original floor candidate predicates and final 3-mm
plane-coherence threshold then operate on this locally estimated geometry.

This is explicitly derived geometry, not original raw pixel depth, an uncertainty
bound or a support certificate. It is currently a floor-only component: RGB-D
feature lifting, obstacle sensing and original camera packets are untouched.
The fixed window is one hypothesis, not the winner of a parameter sweep.

Focused checks passed for planar inverse-depth preservation to below 1e-12 m,
exclusion of an invalid pixel's full 5x5 neighborhood, and exclusion across a
1-m depth step. The probe source is
`scripts/probe_go2_local_inverse_depth_floor_development.py`.

The fixed roster contains four startup frames plus the three noisy baseline
failure frames: (layout, frame) = (0,0), (1,0), (2,0), (3,0), (0,236), (2,200),
(3,519). Compare original and local candidates at zero and 2-mm noise, using
the same fixed perturbations as the sequence study. Both algorithms use the
same orientation from the original recorded public-sensor visual pose and the
initial public specific-force direction. No physics or new tracker execution
is involved; the orientation is held fixed for this component comparison.

All 28 assignments completed. With 2-mm noise:

| Frame | Original candidates | Local candidates | Original/local plane accepted |
| --- | ---: | ---: | --- |
| Layout 0 startup | 487 | 11,781 | yes / yes |
| Layout 1 startup | 513 | 12,262 | yes / yes |
| Layout 2 startup | 398 | 11,697 | yes / yes |
| Layout 3 startup | 902 | 11,822 | yes / yes |
| Layout 0 frame 236 | 1 | 4,317 | no / yes |
| Layout 2 frame 200 | 0 | 300 | no / yes |
| Layout 3 frame 519 | 0 | 188 | no / yes |

Both algorithms accepted all seven unmodified-depth frames. Every reported
local-plane acceptance retained the 3-mm coherence threshold over all resulting
candidate points. This establishes component support recovery at these frames,
not sequence survival or navigation robustness. The source identities and all
plane statistics are retained at
`go2_local_inverse_depth_floor_probe_v1_attempt_001/{launch,result}.json`.

Next, test this candidate extractor in both tracker plane estimation and floor
registration, retaining the original compiled stable-reference image estimator
to isolate the floor change. Replay all four fixed 601-frame recordings with
zero and 2-mm noise before drawing a tracking conclusion. Keep the completed
gyro-proposal variant separate; combining changes is a later hypothesis if
the isolated result warrants it.

## Full-sequence study prepared

`lewm/local_inverse_depth_floor_tracking_development.py` binds the fixed local
candidate extractor into both the original tracker plane estimator and partial
height registration. Image feature lifting, gyro estimation, reference policy,
plane fitting/coherence, temporal checks and transport correction bounds remain
unchanged. Receipts explicitly identify the locally estimated depth geometry.
The gyro-proposal fallback is not enabled.

A 12-frame noisy public-sensor startup check passed tracker, registration and
current-pose readout. Fix the eight assignments before reading outcomes:
2-mm noise on layouts 0/1, then 2/3; zero-noise controls on layouts 0/1, then
2/3. All use frames 0–600, seed 2026091414 and the original perturbation helper.
No source or numerical setting will change between assignments.

The launcher is `scripts/replay_go2_local_floor_noise_development.py`; each
recording owns an exclusive `depth_noise_{0|2}mm_local_floor_tracking_601_v1/`
directory. Save per-frame floor availability and registered poses, all failures,
source identities and physical-error scores computed after estimation ends.
Compare against the completed original and gyro-proposal replays. Do not claim
navigation success from replay survival or treat local average residuals as
raw-depth noise bounds.

Before launch, CPU and both GPUs were idle, available RAM was 76 GiB, and the
artifact volume had 21 GiB free. Use two independent CPU replay owners on the
existing disjoint core groups, with single-thread numerical libraries. No
camera archive is duplicated; all previous experiment artifacts remain intact.

The first noisy pair completed. Layout 0 accepted 208 frames with a floor plane
available on every accepted frame, then exhausted the unchanged measured-bridge
allowance without a retained-anchor observation. Its pending terminal plane
also passed (430 candidates; maximum residual 1.014 mm). This is an earlier
tracking stop than the baseline's frame-236 registration failure, so it is a
sequence regression despite improved floor support. Maximum accepted-prefix
position error was 4.631 mm. Layout 1 completed all 601 frames with a plane
available throughout and maximum error 7.291 mm. Both owners exited 0; the fixed
noisy layout-2/3 pair launched next with unchanged source/settings.

Noisy layouts 2/3 also completed: 198 and 529 accepted frames respectively,
with a current floor plane on every accepted frame, followed by tracking
failures. Their accepted-prefix maximum errors were 3.405 and 7.453 mm. Thus
all four noisy assignments are finished and full-sequence survival remains
1/4. Floor availability improves but does not ensure reference tracking.
Zero-noise controls 0/1 launched next, followed by the fixed 2/3 controls.

Zero-noise controls 0/1 completed all 601 frames with floor measurements on
every frame, maximum position errors 5.477 and 6.901 mm. Controls 2/3 launched
next to finish the roster.

## Adjacent retained-reference eligibility diagnostic

In the noisy layout-0 sequence, frame 197 was promoted, then frames 198–207
used unpromoted consecutive measurements before the frame-208 bridge stop.
A focused replay of the public 197/198 image/depth pair found that pooled
descriptor registration fails (7 primary plus 16 auxiliary matches), but the
existing pooled chained-flow fit passes with 32/48 inliers, including 9 primary
and 23 auxiliary inliers. Its geometry thresholds were unchanged. This is a
pair-fit result, not a fully admitted alternative tracker pose. The complete
result is saved in `adjacent_retained_pooled_fit_diagnostic.json` under layout
0's `depth_noise_2mm_local_floor_tracking_601_v1/` output.

The pooled-camera candidate path in `_JointCameras._candidate` identifies a
consecutive pair using object identity (`ref is self.previous`). A retained
reference for that same frame is a separate object; the retained-chain fallback
only admits gaps of at least two frames. Consequently, the one-interval pooled
fit can be eligible as an increment while being ineligible as a retained-anchor
fit of the same acquisition. This warrants a separate, explicit equivalence
test: establish identical frame, timestamp, feature ownership and stored pose
before allowing the consecutive pooled path for that retained reference.
Do not increase the bridge allowance or promote unsupported poses. Full plane,
image, gyro and continuity checks must still apply. No eligibility change has
been implemented or used in this fixed floor study.

## Eight-replay study complete

All owners exited 0 and all eight variant source-hash sets agree. Every
zero-noise control accepted all 601 frames; the final two maximum position
errors were 5.173 and 4.752 mm. Current tracker floor planes were available on
every accepted frame in all eight runs. Full noisy-sequence survival remains
1/4 for the original, gyro-proposal and local-floor estimators.

| Layout | Original noisy frames | Gyro-proposal noisy frames | Local-floor noisy frames |
| --- | ---: | ---: | ---: |
| 0 | 236 | 236 | 208 |
| 1 | 601 | 601 | 601 |
| 2 | 200 | 220 | 198 |
| 3 | 519 | 531 | 529 |

Local-floor layout 2 stopped because neither an anchor nor the previous frame
supported the pose. Layout 3, like layout 0, exhausted the existing bridge
allowance. No run stopped in floor registration, but the shortened tracking
prefixes prevent claiming that the original later registration conflict has
been resolved end to end. Accepted-prefix accuracy is not a substitute for
tracking survival. No native controller has adopted this variant.

Combined records are saved in
`go2_local_floor_noise_four_layout_summary_v1_attempt_001/result.json`.
The next concrete experiment is the adjacent retained-reference eligibility
check described above, with existing measurement and bridge limits preserved.
