# Actual RGB beacon-marker acquisition: six physical cases completed

The declared red-left/blue-right marker is detected from actual body-mounted RGB
in the positive stationary case, and not detected in the absent, occluded,
single-color, reversed-order or separated-color cases. This supplies a working
beacon-observation interface for the whole-task prototype. It does **not** yet
demonstrate discovery during maze exploration or an actual remembered return.

## Fixed outcomes

The [predeclared protocol](go2_marker_beacon_development_v1_2026-09-05.md) completed
all six cases with no native contact or body-limit stop. Each has 950 physics
samples, 95 ordinary body-sensor samples and five actual RGB captures at
1.5–1.9 s. Only zero commands are executed. All four enclosing collision walls
remain, and added marker/occluder collision geometry and colors are verified.

| Case | Positive frames / 5 | Distinct marker discoveries | First discovery |
|---|---:|---:|---|
| Correct adjacent red-left / blue-right panels | 5 | 1 | Frame 2, at 1.7 s |
| Absent | 0 | 0 | None |
| Occluded by opaque physical wall | 0 | 0 | None |
| Red panel only | 0 | 0 | None |
| Reversed panel colors | 0 | 0 | None |
| Widely separated colors | 0 | 0 | None |

The detector receives the current policy-only RGB/body packet and clock, not the
case label, geometry, proximity or simulator visibility. Body data are validated
by the packet contract but the detector itself uses RGB pixels only. Three
consecutive positive frames register one observable pattern identity; later
detections cannot increase the distinct count. Identical visual copies are the
same observable identity, not magically separable semantic beacon instances.

Actual positive and occluded camera frames were visually inspected. On the first
positive image, red/blue boxes occupy approximately x=236–329 and x=332–427,
y=75–256, with 15,195 and 15,353 positive pixels respectively. These measurements
were inspected after the run; no threshold or expected result was fitted to them.

## Audit, failure preservation and tests

Collection session **88299** completed six cases, exit 0. The initial V1 audit
**43351** failed, exit 1, because it compared Genesis's seven-value padded native
box data directly with three dimensions. The recorded collision data were intact.
The mock static-object tests had missed this native representation detail.

The separate [reader correction](go2_marker_native_box_audit_correction_development_2026-09-05.md)
requires exactly seven finite values, exactly four trailing zeros and the same
unchanged dimensional tolerance for the first three. It does not change the
collector, detector, physical trajectories, original audit or scientific criteria.
Source-identity tests establish that the full trial audit body and all other
static checks remain unchanged. Original failure evidence is preserved.

Corrected audit session **48022** passed all six trials, exit 0: **5,700 physics
samples, 570 ordinary sensor samples and 30 RGB frames**. It checks exact source/
artifact identities, native contact accounting, zero command tape, physical box
sizes/poses/colors/collision flags, actuator gains, reconstructed sensors and causal
histories, actual body-camera transforms and exact detector/discovery replay.
All separately evaluated declared responses match. Audit PASS means faithful
evidence; the response match is reported separately rather than assumed from PASS.

Twenty-five initial tests passed (41868, exit 0). Ten additional correction tests
passed together with those 25 (86739, exit 0). The final explicit suite passed
**981 tests across 84 files**, session **91791**, exit 0, 36.26 s. The prelaunch
suite was 971 across 83 files (48893, exit 0). Original six source/test/protocol
paths and the three correction paths are now bound: do not edit them or restart
the collection/audits. No experiment or audit remains running.

Post-document guard **43630** passed, exit 0: 221 source, 179 input and two gait
bindings, the three correction sources and all five launch/result/audit identities.

## Limits and interpretation

This is a deliberately simple marker baseline, not learned semantic perception or
a JEPA contribution. It covers one stationary frontal observation configuration,
one marker scale/range, the simulator palette and ideal body-sensor timing. Five
frames per case are temporally correlated, not independent trials; there is no
general false-positive-rate, appearance robustness or deployment claim.

The marker remains hidden throughout the occluded case; no controller moves to
discover it. Back-side views, larger pose/lighting variation, motion blur and
partial visibility remain untested. The pattern definition itself requires
ordering in the image and may fail from other viewpoints. There is no distance,
place identity, clearance or completion inference. This successful interface does
not alter the preceding local navigation result of 2/4 or its known failures.

## Evidence identities

Root: `.generated/go2_marker_beacon_development_v1_attempt_001`.

- Launch: `9514555d90fc360d07e0c7efe29afe8aa9047ddbab57295019e4a8dae321faa7`.
- Physical result: `4300c237d9ff9ba3677ea58bc904faa56f42089e03a01e9ca3dd29e84520c019`.
- Preserved original audit FAIL: `1a99c1ce9ede3712849d26b554c1a5804954781dcab94cec53976ff020ccc2b9`.
- Correction binding: `2da9963929ae2bb7b8171afe12b14007f6afa975ec35ed6738cc0e7f02b7ea9e`.
- Corrected audit PASS: `bda9e24e12fda6e60e8305cc8b564c976343d6999642a15bb9d2687d6a1b1ebf`.

Launch binds 221 source/test/protocol paths, 179 inputs and two gait bindings.
The correction additionally binds its exact three source/test/document paths.
Recorded packages: Genesis 0.4.6, torch 2.12.0+rocm7.2, NumPy 2.4.6,
SciPy 1.17.1, Pillow 11.3.0. Physics and gait run on CPU with the existing native
camera pipeline; this is not real-platform sensing or execution evidence.

## Next: connect the whole task

1. Implement a continuous controller using the existing episodic route hypotheses
   and this actual RGB marker observer. Run the observer at every actual camera
   tick; maintain uninterrupted gyro reference and fresh local executor histories.
   Replace the current fixed two-leg stopping point with bounded observed
   exploration and, after actual detection, tentative route-based return.
2. Keep visits provisional. A return intent must acquire a fresh exit and pass
   the existing bounded local executor; emptying the hypothesis stack remains
   HOME_CANDIDATE. Evaluate actual physical home arrival separately. Preserve
   sensor/native failures and all incomplete/false return claims.
3. Fix a fresh connected-maze development population with branch choices and an
   initially occluded marker before launch. Compare persistent route memory with
   the same executor/sensors and a declared local-only baseline. Do not inject
   known cell identities, beacon coordinates or an oracle route. Retain the
   known narrow-maze arrival and clearance failures; a separate integration
   domain cannot erase them or qualify independent-maze generalization.
4. Then proceed to supported scan/alignment/traversal action coverage, matched
   predictive-training/online-rollout comparisons, independent novel layouts and
   robustness, and bounded real-Go2 evidence when hardware is available. Do not
   replace whole-task integration with more stationary marker repetitions.
