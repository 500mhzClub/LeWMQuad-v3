# Initial floor depth-noise sensitivity

While the fixed native tracking-refresh pair ran unchanged, a separate
low-priority, single-threaded saved-frame probe tested the current floor
candidate extractor and joint-plane acceptance. It used frame 0 from each of
the four completed persistent-memory runs. PublicReplay reconstructed the
original packets; the up direction came from the saved public-sensor initial
gravity estimate. No native physics was read.

The fixed perturbation grid was independent zero-mean Gaussian optical-depth
noise with standard deviation 0, 0.25, 0.5, 1, 2 or 5 mm, using seeds
2026091400–2026091404. Primary and auxiliary noise were independent. Original
invalid pixels remained invalid; perturbed values outside 0.2–5 m became
invalid. These are experimental sensitivity values, not a calibrated sensor
noise model. RGB, gyro, mount geometry and the floor algorithms were unchanged.

| Depth noise standard deviation | Accepted floor fits / 20 perturbations |
| --- | --- |
| 0 mm | 20 |
| 0.25 mm | 20 |
| 0.5 mm | 20 |
| 1 mm | 20 |
| 2 mm | 20 |
| 5 mm | 5 |

There are only four independent recorded frames; zero-noise seeds repeat the
same input. Zero-noise candidate counts and both camera residual statistics
match all four original saved plane receipts exactly. Candidate counts fall
from 24,811–25,923 without added noise to 404–927 at 2 mm and 28–303 at 5 mm.
The 5-mm failures report insufficient combined candidates, rather than a
failure of the final 3-mm plane-coherence check. The adjacent-pixel and
neighborhood selection predicates therefore deserve attention when introducing
sensor noise; widening the final plane gate alone would not address this result.

This test does not measure tracking survival, pose accuracy, correlated stereo
errors, invalid-depth patterns, timing, RGB degradation or closed-loop outcomes.
The next sensing experiment needs a defined noise/dropout model and tracking
evidence across complete sequences before live navigation. Keep the current
four-run tracker-refresh study unchanged.

Source: `scripts/probe_go2_initial_floor_depth_noise_development.py`.
Artifacts: `go2_initial_floor_depth_noise_sensitivity_v1_attempt_001/launch.json`,
`result.json` and `zero_noise_reconstruction_check.json` in the development
artifact root. No large sensor archive was created.

## Fixed tracking-sequence follow-up prepared

Prepared `scripts/replay_go2_depth_noise_tracking_development.py`. After the
four native reference-refresh assignments finish, replay frames 0–600 from
each of the four completed persistent-memory recordings. Compare unchanged
depth with 2-mm independent Gaussian depth noise, fixed seed 2026091414 and
separate reproducible draws per layout, frame and camera. The 2-mm condition
was selected to test temporal tracking where all startup planes passed but
candidate support had already fallen sharply. This is an explicit follow-up
to the sensitivity result, not an independently selected hardware noise level.

Keep the original compiled stable-reference tracker, partial-height registration,
RGB, gyro, calibration geometry, validity exclusions and timestamps. Rebind
auxiliary RGB to the synthetic perturbed depth; preserve original archives.
No bridge extension, plane-gate change, reference-refresh variant or hidden
pose input is introduced. Read physical truth only after estimation ends, to
measure registered-position error. Count tracking/registration failures and
verify exact original raw-pose reproduction in every zero-noise control.

Fix all eight replay assignments: zero-noise layouts 0/1, 2-mm layouts 0/1,
2-mm layouts 2/3, then zero-noise layouts 2/3. Use the existing disjoint CPU
groups, two replays at a time after native runs exit; save every failure and
finish the roster without changing the noise or tracker. Perturbed packets
are synthetic sensitivity inputs, not sensor-calibrated capture. Recorded
trajectories are fixed: replay cannot establish counterfactual navigation.

A focused packet check passed: zero noise returns the original packet;
fixed seed/layout/frame reproduces exactly; previously invalid rays stay
invalid; both primary and auxiliary packet bindings validate. No sequence
replay has launched yet.

The native tracker-refresh study is now complete, with all owners exited and
all four comparisons saved. Before replay dispatch, CPU was 0.5% busy, about
76 GiB RAM and 21 GiB artifact-volume space were available, and both GPUs
were idle. Use two replay processes on the existing disjoint CPU groups;
outputs contain estimates and diagnostics, not duplicate camera archives.
The zero-noise layout-0/1 controls launched in sessions 82849/69333.

Both initial controls exited 0 and reproduced all 601 original raw poses
exactly, with no tracking/registration failure. Maximum registered-position
errors against physics were 5.477/6.901 mm, matching the recorded prefixes.
Replay elapsed times were 70.95/73.46 s; peak process RSS was about 1.4/1.3 GiB,
with zero swaps. The fixed 2-mm layout-0/1 replays then launched in sessions
39258/51035, unchanged except for the declared sensor perturbation.

The first noisy pair completed. Layout 0 accepted 236 frames, then registration
rejected `current measured candidate conflicts with transported floor reference`
at frame 236; maximum position error over the accepted prefix was 8.148 mm.
Layout 1 accepted all 601 frames, maximum error 12.016 mm (versus 6.901 mm
in its zero-noise control). These outcomes remain in the fixed population.
The 2-mm layout-2/3 pair launched next in sessions 10512/31395. Layout 2
then stopped in tracking at frame 200 after 200 accepted frames; its full
tracking failure record is saved separately. Layout 3 remains active.

Layout 3 also stopped in tracking, at frame 519 after 519 accepted frames.
Both layout-2/3 failure chains state that neither retained anchors nor the
previous frame support the current pose. Maximum errors over their accepted
prefixes were 5.106/5.615 mm; these shorter-prefix figures must not be compared
as though they describe complete 601-frame survival. No threshold or noise
value changed. The remaining zero-noise layout-2/3 controls launched next
in sessions 56141/41935 to finish the fixed roster.

## Sequence study complete

All eight replays finished and all four zero-noise controls reproduced every
original raw pose exactly: 2,404/2,404 poses, 601 per recording, no failure.
Zero-noise layout-2/3 maximum position errors were 5.173/4.752 mm.

| Layout | Zero-noise accepted frames | 2-mm accepted frames | Noisy outcome |
| --- | --- | --- | --- |
| 0 | 601 | 236 | Floor-transport conflict at 23.6 s |
| 1 | 601 | 601 | Completed; maximum position error 12.016 mm |
| 2 | 601 | 200 | Visual pose unavailable at 20.0 s |
| 3 | 601 | 519 | Visual pose unavailable at 51.9 s |

Combined results, including original pose errors over each noisy run's exact
accepted prefix, are saved in
`go2_depth_noise_tracking_four_layout_summary_v1_attempt_001/result.json`.
The controls validate reconstruction of these recorded streams. The perturbed
results identify a concrete sensitivity in the current tracker/registration
pipeline; they do not establish a particular physical camera's noise tolerance.
RGB, gyro and timing remain ideal here, and fixed-input replay does not execute
the navigation decisions that noisy estimates would induce.

The next useful work is to diagnose the rejected image correspondences and
floor-reference residuals in these recorded failures, then test a separately
defined noise-aware estimator. Do not infer robustness from small errors in
accepted prefixes or loosen the existing rejection gates just to pass this set.

## Fixed failure diagnosis complete

`scripts/diagnose_go2_depth_noise_failures_development.py` reproduced all three
failures with unchanged packets, estimator and thresholds. Every accepted row
(position, reference frame and promotion reason) matched the completed noisy
study exactly: 236/236, 200/200 and 519/519 for layouts 0, 2 and 3. All owners
exited 0. The full terminal tracker snapshot, previous registered evidence and
floor state are retained under each recording's
`depth_noise_2mm_failure_diagnostic_v1/`. No native physics was loaded.

Before dispatch, CPU was about 99% idle, available RAM 76 GiB, both GPUs idle,
and artifact-volume free space 21 GiB. Two independent CPU replays ran together
on disjoint core groups, followed by layout 3; the first pair used approximately
1.2 GiB RSS each. Replay times were 33.30, 27.39 and 70.88 seconds. No camera
archives were duplicated.

- Layout 0, frame 236: raw visual tracking accepted the primary-camera anchor
  at frame 235. Registration extracted only one current floor candidate, with
  no auxiliary candidates; that point disagreed with the transported floor by
  6.529 mm, exceeding the unchanged 3-mm limit. Its last fully admitted floor
  anchor was frame 49 (18.7 seconds earlier). Tracker plane extraction uses the
  preceding attitude and found two candidates; registration uses the current
  attitude and found one. Both populations were insufficient for a new plane.
- Layout 2, frame 200: neither camera obtained an accepted geometric fit.
  Direct flow still supplied 12 primary and 47 auxiliary depth-lifted pairs.
  Fits failed consensus pruning or the final consensus check; older anchors
  mostly lacked enough matches. Current floor candidate count was zero.
- Layout 3, frame 519: direct flow supplied 36 primary and 22 auxiliary pairs.
  The primary fit failed spatial conditioning, and the auxiliary fit failed
  consensus pruning. Current floor candidate count was zero.

Thus the terminal tracking failures are not simply an absence of RGB matches.
Floor support loss and geometric fitting under noisy depth are separate issues.

## Adjacent-pair gyro-initialization probe complete

`scripts/probe_go2_noisy_failure_pairs_development.py` compared the existing
joint and gyro-conditioned registration modes on frames 199/200 and 518/519,
using both unmodified and exactly reproduced noisy depth, both cameras, and
descriptor/direct-flow associations. It integrated gyro only from the current
public 100-ms interval. It preserved the feature budget, proposal seed and all
geometric thresholds; it did not run floor registration or admit a tracker pose.
All 32 fit outcomes are saved in `adjacent_pair_fit_probe.json` inside the two
diagnostic directories, including failures.

With 2-mm noise, auxiliary descriptor fits failed in joint mode on both pairs,
but gyro-conditioned fits passed with 17 and 12 inliers respectively. Layout 2's
auxiliary direct-flow fit also changed from rejection to acceptance with 30 of
47 inliers. Layout 3's noisy direct-flow fits remained rejected. Unmodified
auxiliary descriptor fits passed joint mode on both pairs.

The current tracker first requires an unconstrained image fit and then refits
its surviving consensus using gyro rotation. These pair results motivate testing
gyro-conditioned proposals before that initial rejection, with no gate widening.
They do not prove full tracking survival, correct floor registration, noisy
closed-loop navigation or hardware robustness. The next experiment should use
a separate estimator variant and replay all four fixed sequences, including
zero-noise controls. The layout-0 floor-candidate/transport conflict remains an
independent problem and must not be hidden by the image-fit change.

A further single-frame readout held the noisy layout-0 pose and transported
floor fixed while substituting the original depth at frame 236. Original depth
supplied 1,595 primary and 11,506 auxiliary floor candidates versus 1 and 0 with
noise. Their mean signed residuals against that same transported floor were
-6.158 and -5.891 mm. At the exact failing primary pixel, the original-depth
residual was -6.593 mm versus -6.529 mm with noise: the instantaneous depth
perturbation changed this residual by only 0.064 mm. Thus this rejection is
consistent with an accumulated pose/floor-reference mismatch after support
depletion, rather than an isolated large depth outlier at the terminal pixel.
This readout used no physics and did not rerun a clean tracker. Results are in
`clean_depth_counterfactual_floor_diagnostic.json` under layout 0's diagnostic
directory.
