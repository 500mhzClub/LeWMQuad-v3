# Support-aware RGB-D: full fitting-tape tracking, uncertainty still unqualified

The new support-aware reference selector completed all **336 observations / 33.5 s**
of the sustained fitting recording in nominal and both gyro-bias conditions.
Nominal maximum measured position error was **5.143 mm**. This is a substantial
continuity improvement over fixed-reference V1's 40 observations, with the same
feature matching and registration acceptance criteria. It is engineered sensor
estimation, not learned navigation or JEPA, and does not establish safe clearance.

## Frozen comparison

| Input | Admitted / 336 | Accepted reference promotions | Max position error | Max robust global radius |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 336 | 30 | 5.143 mm | 0.4595 m |
| Blank RGB | 1, coordinate anchor only | 0 | No estimated motion | No motion bound |
| Gyro-Z +0.001 rad/s | 336 | 39 | 105.323 mm | 0.5766 m |
| Gyro-Z −0.001 rad/s | 336 | 36 | 106.536 mm | 0.5669 m |
| Independent depth noise +0.1 mm | 202 | 16 | 3.985 mm | 0.2828 m |

All 121 promotions used accepted support margin, not the old distance/angle
thresholds. Registration still requires six occupied grid cells; promotion
occurs on an accepted view with at most seven cells in either image. No failed
observation was promoted, no failed owner was invoked again, and global pose,
gyro history and anchor error were retained. Exact initial-reference pose/
orientation/legacy-radius comparisons match V1 for 20 nominal, 1 blank, 18
positive-bias, 17 negative-bias and 20 noisy observations.

On the 40 nominal observations common with V1, maximum error is 1.071 mm versus
0.984 mm. Thus the continuity gain is not a uniform accuracy improvement. The
same-input comparison uses the corresponding V1 perturbation in each case.
Older `common_maximum_original_error_m` fields still reference the nominal
ShadowObserver and are not matched perturbed comparisons.

Blank RGB fails at frame1 / 1.6 s with zero matches. The noisy member fails at
frame202 / 21.7 s **before registration**: one perturbed valid pixel is
5.00003528595 m, outside the 5-m packet contract. The perturbation intentionally
preserved the old validity mask without clipping. This is a retained malformed-
packet failure, not evidence that the tracker failed on a correctly filtered
noisy observation. A future physical-noise experiment must inject noise before
validity/range filtering in a separately declared protocol; do not repair this
artifact or silently resume its model.

## What the outlier calculation establishes

A separate interval calculation permits at most floor(0.2*N) arbitrary outliers
among the accepted point pairs, leaving the pose mean and feature selection
unchanged. Coordinate-wise order statistics enclose translations consistent
with the assumed good-point majority. The box is an outer bound, not proof of
a common three-dimensional consensus. Global radii retain prior keyframe errors;
unknown promoted-anchor uncertainty would remain unknown in descendants.

All 1,206 non-anchor accepted observations produced nonempty local boxes and
retained global radii. Native relative translations were inside every local box;
no global position-radius exceedance was recorded. No frame exceeded the
declared 20% point-outlier allowance. Individual point-assumption violations
remain: 691/44,486 nominal, 362/45,154 positive bias, 345/44,150 negative bias,
and 518/29,411 depth noise. These checks use evaluator poses, not model inputs.

**The hypotheses are still not validated.** Local gyro-angle bounds were exceeded
on 41 nominal, 246 positive-bias, 204 negative-bias and 18 noisy observations.
Global angular allowances failed on 159 positive-bias and 138 negative-bias
observations. Aggregate translation staying inside a generous box does not
repair these violated assumptions. The maximum local robust translation radius
was 27.645 mm nominal, 23.673 mm positive bias, 26.048 mm negative bias and
27.554 mm noise; these are conditional radii, not measured error or clearance.

The large global radii are a real remaining limitation, not something to hide
with a reset. The new estimator produces diagnostic poses without applying the
old controller's 80-mm scalar gate. Its continuity must therefore **not** be
reported as safety-qualified continuation of that controller. No floor/ray
consumer, prospective gait envelope or navigation command was changed.

## Verification and scope

Fourteen focused tests passed. Full regression: **2,123 tests in 170 explicit
files passed in 185.86 s**. A separate source-bound analysis reconstructed all
1,680 member status rows, 163,201 accepted point pairs and 121 promoted parent
links. Raw depth/pixels reproduce lifted points; accepted point means reproduce
composed global poses. Independent long-double corner bounds and partition-based
box endpoints agree within 5.03e-16 m. Exhaustive enumeration checked 6,720 small-
population subset intersections against the outer enclosure. Terminal blank
and malformed-depth failures were independently accounted for.

This audits arithmetic, data provenance and history, not physical feature identity
or calibrated sensor uncertainty. Feature matching and native scoring reuse
reviewed components; native scoring was not independently reimplemented. Only
the fitting recording was analyzed. No new validation frames, physics, maze
mission, policy learning, JEPA evidence or hardware actuation occurred. The
earlier failed maze, action-response validation and missing full-body floor
coverage remain unresolved.

## Next execution

1. Freeze a comparison of **joint RGB-D rotation/translation** against this
   gyro-conditioned estimator. Use robust correspondence selection, an explicit
   geometric conditioning check and gyro disagreement diagnostics. Preserve this
   baseline unchanged. Test whether image-derived rotation corrects the observed
   bias-driven translation drift; do not assume success or merely increase the
   angular allowance. Revisit local error hypotheses and outlier coverage on new
   validation after method selection is frozen.
2. Collect the required longer supervised physical motion using fitting-only
   measured response and checked scene/camera/resource limits. The proposed
   40-s forward segment at unchanged speed must actually demonstrate full-body
   observed-floor coverage. Existing short-travel V1 is not a substitute.
3. Fit and independently validate prospective gait/turn/brake response, integrate
   one floor/non-floor evidence interface with relative error history, meet
   end-to-end timing, then complete exploration/backtracking/marker discovery/
   return. Establish JEPA training and genuine multistep/memory contributions
   against matched baselines on independent layouts/seeds/robustness conditions,
   then obtain bounded real-platform evidence when available.

## Artifacts

[Protocol](go2_support_aware_rgbd_pose_development_v1_2026-09-06.md).
Model: `lewm/support_aware_rgbd_pose_development.py`.
Runner: `scripts/probe_go2_support_aware_rgbd_pose_development_v1.py`.
Audit: `scripts/analyze_go2_support_aware_rgbd_pose_development_v1.py`.
Output: `.generated/go2_support_aware_rgbd_pose_development_v1_attempt_001`.

The launch binds 512 sources and 7,705 inputs plus native/OpenCV dependencies.
All predictions were persisted before native scoring. Source, input and result
bindings were verified unchanged. Launched source and artifacts remain immutable.

- Launch: `bfaacdf54e1b813dd4767be7af3b932e3fec9ee68d6dc9e74cf4429474b1b21e`.
- Result: `8e0103ac893398fad559976aead6a2a41b1babd4d5dc1b00b4f2700ebc729b7e`.
- Predictions: `8567930e30879a8644e7a57b059bb315384d9c8db91f7ad60583510a0f00a9e4`.
- Evaluation: `186592b87bfc650afaf8d14be992fc99047c478840ea07bdd82563c8b4a14844`.
- Chain/box audit: `e53868496ae3df30be6b214307b6e5c11750a604417b8921293a3195bfab3725`.

The full scientific goal remains active and unachieved.
