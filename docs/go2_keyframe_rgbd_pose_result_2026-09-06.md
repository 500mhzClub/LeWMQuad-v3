# Keyframe RGB-D V1: small nominal improvement; assumptions and continuity fail

Implemented a new direct-to-keyframe RGB-D pose estimator and completed a frozen
five-member replay of the sustained **fitting trial only**. The original estimator
was not modified or resumed. All 336 original shadow outputs replay exactly,
including its 5.1-s terminal failure. No validation image, new physics, navigation,
JEPA training or hardware was used.

The new nominal estimator admitted 40 observations, versus the original 36.
On their 36 common observations its maximum position error was 0.984 mm versus
1.803 mm. However, it failed at 5.5 s, never promoted a keyframe, and its accepted
point pairs sometimes violate the proposed error assumptions. **Do not integrate
this model into navigation or call the uncertainty calibrated.**

## Fixed recorded outcomes

| New estimator input | Admitted / 336 | Failure time | Max admitted position error | Max conditional position radius | Inlier point-assumption violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 40 | 5.5 s | 0.984 mm | 32.775 mm | 263 / 5,468 |
| Blank RGB | 1 | 1.6 s | Anchor only | Anchor only | No matches |
| Gyro-Z +0.001 rad/s | 19 | 3.4 s | 7.696 mm | 23.111 mm | 9 / 2,840 |
| Gyro-Z −0.001 rad/s | 17 | 3.2 s | 6.585 mm | 20.743 mm | 1 / 2,583 |
| Independent depth noise +0.1 mm | 40 | 5.5 s | 0.980 mm | 32.741 mm | 264 / 5,470 |

Times are absolute recording clocks; the coordinate anchor is 1.5 s. The blank
member's first output merely defines that coordinate origin; it never estimates
a translation. Every failure is terminal and subsequent rows explicitly record
non-reinvocation. Lower maxima over shorter biased histories are not better
full-trajectory performance. Biased/noisy members measure sensitivity of the new
estimator; their saved `common_maximum_original_error_m` fields refer to the
**nominal** predecessor, not an input-matched perturbed baseline. Only the nominal
old/new comparison is a matched estimator comparison here.

There were zero position-radius exceedances, but the angular hypothesis failed
on six positive-bias and four negative-bias observations. Maximum angular-radius
excess was 22.28 and 70.87 microradians. The stated total gyro/integration allowance
cannot automatically equal an injected bias while ignoring native integration
error. No new margin was fitted to cover these outcomes.

## What failed and what it means

1. **Reference support was lost before the promotion rule fired.** Nominal and
   noisy terminal fits retained 87/90 points (96.7%), but occupied five reference
   image-grid cells rather than the required six. Positive gyro bias retained
   113/146 points (77.4%); negative bias retained 104/152 (68.4%). Both also had
   five reference cells. In each non-blank case this was the sole failed
   registration requirement: means/inlier sets stabilized, residual/reprojection
   checks passed for the selected pairs, and displacement was within limits.
   Terminal nominal reference displacement was about 0.171 m, below the fixed
   0.4-m promotion trigger. There was no actual keyframe promotion to validate.
2. **Passing an average pose bound does not validate each matched-point bound.**
   Under native relative pose, 537/16,361 accepted point pairs violate the sum
   of the hypothesized lifting/localization radii. This falsifies the joint
   assumption that every accepted pair is the same static point with <=0.5-pixel
   coordinate and <=0.25-mm lifted-depth error. It does not by itself identify
   whether correspondence, localization or depth interpolation is responsible.
   Averaging can hide individual violations; low RMS or low mean position error
   is not evidence that the bound's assumptions hold.
3. **Known-rotation registration is sensitive to accumulating gyro bias.** The
   bias cases lose support earlier even at smaller displacement. A spatial-bin
   count is a heuristic, not a direct measure of pose observability. Replacing
   it requires a separately tested geometric criterion, not relabeling five as
   six or retrospectively changing this result.

## Implemented computation and verification

The new source retains a reference RGB-D image, matches actual features directly,
and solves equal-weight translation under causal gyro rotation. Promoted references
would retain parent identities, global pose and prior error allowance. Correlated
per-point lifting and rotation errors are combined without a 1/sqrt(N) reduction;
global composition retains anchor translation and orientation effects. The model
does not consume the old scalar budget as a permission gate: its continued output
is a **different conditional estimator diagnostic**, not qualified continuation
of the failed old controller.

Fifteen focused tests cover transform/lever-arm conventions, >single-tick reference
displacement, unsupported registration, lifting-error box corners, correlated
repeated points, radius composition, constant-reference error accounting, synthetic
keyframe promotion and packet/image terminal failure. Full regression:
**2,109 tests in 169 explicit files passed in 181.78 s**.

A separately source-bound analysis checked all 1,680 member status rows, terminal
histories and 16,361 accepted point pairs. Stored points reconstruct from original
represented depth and pixel coordinates; translations reconstruct from the
actual accepted-point means. An independent long-double eight-corner calculation
matches the declared radii within 2.12e-16 m. Terminal feature matching and gate
accounting reproduce the five-cell failures. This validates arithmetic/accounting,
not the physical error hypotheses. Feature matching is shared with the model;
native scoring was not independently reimplemented. Runtime promotion remains
untested because all five members retained only the initial keyframe.

## Next work toward the scientific goal

- In a new declared model, promote an **accepted** reference before its support
  becomes unusable, using observed matching/conditioning margin as well as motion.
  Preserve global pose/error history and the unchanged V1 comparator's rejection
  criteria and failures; no new anchor after a terminal failure. Validate promotion on changing
  views rather than only synthetic translations.
- Diagnose correspondence/localization/depth errors on fitting data. Test a robust
  estimator with an explicit tolerated-outlier model, rather than assuming every
  residual-gated point is correct. Joint RGB-D rotation/translation can also be
  compared against gyro-conditioned translation, with geometric degeneracy and
  gyro-consistency checks. Freeze alternatives and criteria before new validation.
  Do not install this V1 radius or select a multiplier from these maxima.
- The larger acquisition requirement is unchanged: use fitting-only measured gait
  response to design sufficiently long actual travel into observed floor. V1's
  0.716-m forward travel did not provide full-body coverage. A separately declared
  40-s forward collection at unchanged speed is a candidate, subject to scene and
  resource checks; it is not a guaranteed displacement or clearance envelope.
- Fit/validate prospective action and stopping response, integrate consistent
  floor/non-floor evidence, meet loop timing and complete exploration/backtracking/
  marker discovery/home return. Then establish matched JEPA-versus-supervised/
  geometric, real multistep rollout and memory effects on independent layouts,
  training seeds and robustness shifts, followed by bounded real-platform evidence.

## Artifacts

[Protocol](go2_keyframe_rgbd_pose_development_v1_2026-09-06.md).
Source: `lewm/keyframe_rgbd_pose_development.py`.
Runner: `scripts/probe_go2_keyframe_rgbd_pose_development_v1.py`.
Analysis: `scripts/analyze_go2_keyframe_rgbd_pose_development_v1.py`.
Output: `.generated/go2_keyframe_rgbd_pose_development_v1_attempt_001`.

Launch binds 507 sources and 7,700 inputs plus native/OpenCV dependencies. All
predictions were persisted before native scoring. Sources and inputs were verified
unchanged; these launched artifacts and sources are immutable.

- Launch: `0383825888a0cb374f38156406a483392ec948e3d74776d515b14ecfc03f8709`.
- Result: `c6b85182b915f14fea86256926cdeacb6b2f69997023d64deadbf3ccce05f73c`.
- Predictions: `083057a91f5622e15dc5485a1c102b41a4b794797ff9db63e8f64a48fc135791`.
- Evaluation: `ab883d1881dd18b4836ee4bab2729999732b71d45ff75ad822f1ef8b0db7ad43`.

The full scientific goal remains active and unachieved.
