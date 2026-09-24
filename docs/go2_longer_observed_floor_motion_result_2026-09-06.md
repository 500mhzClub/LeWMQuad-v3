# Longer observed-floor motion: acquisition audited

Two fresh 60-s supervised physical simulation trials completed and passed the
raw acquisition audit: 60,000 physical samples, 1,172 RGB-D frames, no native
guard violations and no incomplete expected artifacts. This is one wide-wall
development geometry, not novel-maze navigation or hardware evidence.

| Measured quantity | Fit | Validation |
| --- | ---: | ---: |
| Forward-segment net travel (40 s) | 2.121440 m | 2.149992 m |
| Whole-tape path length | 2.724876 m | 2.836231 m |
| Forward-segment yaw change despite zero yaw command | -.302485 rad | -.297805 rad |
| Left-turn yaw change | +.868973 rad | +.799139 rad |
| Right-turn yaw change | -1.120056 rad | -1.087440 rad |
| Forward-brake net displacement (1 s) | 9.545 mm | 29.962 mm |
| Terminal extra-zero-tail displacement (.5 s) | 1.000 mm | 1.063 mm |
| Initial measured-floor maximum footprint coverage | 27/27 shapes | 27/27 shapes |
| First native-pose full-coverage frame | 262 (27.7 s) | 262 (27.7 s) |
| Native-pose full-coverage frames | 324/586 | 324/586 |

The full-body floor-coverage acquisition deficit is resolved for these tapes:
the actual robot reaches floor observed in the initial camera frame, then turns
over it. This uses actual joint configurations, not a virtual translated initial
posture. Coverage here is evaluator-only: native relative poses, conditional
measured-surface assumptions and ZERO additional shape/pose error allowance.
It neither calibrates uncertainty nor permits a gait command. The old causal
ShadowObserver still stops at 5.1 s after 36 accepted frames, before any coverage;
it was never restarted or allowed to choose commands.

All 585 depth-motion pairs per trial remain rank two, including the first.
Interior analytic depth discrepancies are at most .103 mm. Raw timestamps,
requested/applied commands, gain identity, physical/visual geometry, camera and
body/gyro sensing, initial admission, exact-foot attribution and native guards
were audited. Sources were verified before each trial and after collection and
audit. Original V1 failures and this protocol's fixed duration were preserved.

The forward yaw drift and different brake displacements are important remaining
action-model problems. Requested velocity is not executed displacement, and zero
command is not immediate rest. Maximum angular speed in the last 200 ms of the
extra tail is .016910/.018455 rad/s; do not label the endpoint perfectly static.

## Identities and checks

Root: `.generated/go2_longer_observed_floor_motion_development_v1_attempt_001`.
The launch binds 524 sources and 7,715 inherited inputs plus native/OpenCV.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 1f762003db74c10d13f04af2e5164d5a00cb596f586cdc3e23a14885fa11ee04 |
| result.json | 87d5ab645887ac86475de8a26850fda0fd85b0eaac5e999a858c77db1a8b57d3 |
| raw_acquisition_audit_launch.json | 5fc6afc498937f9a7345a3f8f2360b16b8bb15a22499e4f0bae2d64400ea51fe |
| raw_acquisition_audit.json | 848784ac97f8118ed3ee631de70eede463b9e4c81295f03ce5bab12ffcb8c156 |

The 172-file regression passed 2,156 tests in 188.31 s. Eight new acquisition-audit
tests passed separately. The fixed pose/coverage transfer scorer passed seven
additional tests, including nonidentity world frames, unequal survival durations,
coverage disagreements and missing/nonquiet force. Test counts are engineering
checks, not independent scientific trials.

## Next

The [reserved frozen comparison](go2_longer_motion_frozen_pose_result_2026-09-06.md)
is now complete, including its preserved coordinator failure and explicit
coverage-status successor. The original requirement was:
unchanged nominal joint and gyro RGB-D estimators on both tapes, predictions saved
before native scoring, complete terminal histories and common-frame comparisons.
Report causal initial-floor footprint coverage separately from native diagnostics.
No fitting, parameter selection, uncertainty calibration or command permission is
allowed to follow merely from small replay errors. Future fitted action/error
models need a frozen procedure and fresh reserved validation; this validation's
motion results are now known.

Then resolve relative body/surface error and prospective gait/braking before
integrating control and memory. The full exploration/return and matched
JEPA/multistep/memory/layout/seed/timing/hardware objectives remain incomplete.
