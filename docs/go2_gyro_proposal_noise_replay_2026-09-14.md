# Gyro-conditioned proposal fallback under depth noise

The completed adjacent-pair probe recovered auxiliary-camera geometric fits at
both noisy tracking failures by conditioning the initial fit on public gyro.
Test this in the full tracker using a separate variant; floor registration and
all absolute geometric thresholds remain unchanged.

`lewm/gyro_proposal_fallback_development.py` preserves a successful initial joint
fit exactly. On a single-camera joint registration rejection, it tries the
existing gyro-conditioned registration mode on the same points and pixels.
The alternative must pass the same minimum count, strict majority, spatial
conditioning, residual and bidirectional reprojection checks. Existing plane
refinement, final gyro consensus, temporal continuity, reference promotion and
bridge limits then apply. Descriptor, direct-flow and chained single-camera
paths receive the change; pooled-camera initial fitting remains unchanged.
This is a sensor-estimator change, not a JEPA or navigation-controller change.

The actual class chain and a 12-frame public-sensor tracker/registration startup
check passed. An initial import used the closure-free floor binding helper on a
method containing `super()`; it failed before observations and was corrected to
the existing closure-preserving tracker helper before the study. No fixed-study
attempt was used for that implementation check.

Fix all eight assignments before inspecting sequence outcomes: 2-mm noise on
layouts 0/1, then 2/3; zero-noise controls on 0/1, then 2/3. Each replays frames
0–600 of the persistent-routing recording using the same seed 2026091414 and
packet perturbation helper as the completed baseline study. All outcomes,
including earlier failures, belong to the comparison. Do not change the variant
between assignments or infer navigation success from replay survival.

Launcher: `scripts/replay_go2_gyro_proposal_noise_development.py`.
Each original recording owns an exclusive
`depth_noise_{0|2}mm_gyro_proposal_tracking_601_v1/` output containing source
identities, estimates, result and any terminal evidence. Physical truth is read
only after estimation ends. No camera archive is duplicated. The previous
estimator's completed eight replays remain the comparison population.

Before dispatch, CPU was idle, 76 GiB RAM and 21 GiB artifact-volume space were
available, and both GPUs were idle. Use two independent replay processes at a
time on CPU groups 0–7/16–23 and 8–15/24–31, with one numerical-library thread per
process. Recent unchanged replays took roughly 30–80 seconds each at about
1.2–1.4 GiB RSS; CPU replay is sufficient for this bounded study.

Noisy depth remains synthetic independent Gaussian optical-depth noise, not a
hardware-calibrated model. RGB, gyro and timing remain unperturbed.

## First noisy pair complete

Layout 0 accepted 236 frames and stopped at the same registration-conflict
frame as the baseline; maximum position error over that prefix was 8.788 mm.
Layout 1 accepted all 601 frames, maximum position error 9.376 mm. Both exited
0, with all failures preserved. The fixed noisy layout-2/3 pair launched next.

The launcher's `selected_gyro_proposal_frames` diagnostic is unavailable: it
searches the compact current-pose record, which does not contain registration
receipts. Its zero values must not be interpreted as no fallback use. This is
a reporting limitation; the actual class method chain was inspected and the
estimated positions differ from the baseline. Keep the study source unchanged
while assignments run; use the saved estimates and terminal evidence for
outcomes, and instrument full registration receipts separately if activation
counts are needed. No algorithm or packet setting has changed between runs.

## Noisy assignments complete; controls running

Layout 2 advanced from the baseline's frame-200 failure to frame 220, where
the unchanged measured-bridge allowance expired without an anchor observation.
Retained anchors 202–209 lacked enough matches; current floor support remained
absent. Layout 3 advanced from frame 519 to frame 531, then also stopped in
tracking. Their accepted-prefix maximum position errors were 5.224 and 4.700 mm.
All four noisy assignments have exited. Full 601-frame noisy survival remains
1/4, so the estimator change is insufficient by itself. Zero-noise layouts 0/1
launched next; layouts 2/3 remain in the fixed roster.

## All eight assignments complete

All four zero-noise controls accepted 601 frames without a tracking or
registration failure. Their maximum position errors were 5.078, 7.081, 5.346
and 5.598 mm. All eight owners exited 0, and every recorded variant source hash
is identical across the assignments. Estimated poses can differ from the
baseline even at zero noise because fallback fits can change reference selection.

| Layout | Baseline noisy frames | Variant noisy frames | Variant outcome |
| --- | ---: | ---: | --- |
| 0 | 236 | 236 | Floor-registration conflict |
| 1 | 601 | 601 | Complete |
| 2 | 200 | 220 | Tracking failure |
| 3 | 519 | 531 | Tracking failure |

Combined results are in
`go2_gyro_proposal_noise_four_layout_summary_v1_attempt_001/result.json`.
Both estimators complete 4/4 unmodified recordings and only 1/4 noisy recordings.
The proposal fallback gets beyond the two original tracking-failure frames but
does not provide adequate sequence robustness. It has not been selected for a
native navigation run. Accepted-prefix errors describe different survival
lengths and must not be treated as full-sequence accuracy comparisons.

While controls ran, an independent floor-only component probe recovered accepted
planes at all three baseline failure frames using local inverse-depth averaging.
Its fixed seven-frame results and limits are documented in
`docs/go2_local_inverse_depth_floor_probe_2026-09-14.md`. The next full-sequence
experiment should isolate that floor change using the original image estimator.
