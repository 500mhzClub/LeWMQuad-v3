# Bounded/aligned floor integrated with Go2: interface check passes

The previous goal turn made progress with a paired rendering experiment. This
turn implements an explicit new Go2 scene/initialization/acquisition path and
executes the fixed robot-level interface assay. It does not change old source-
bound controllers, recordings, thresholds or results. Full maze success remains0/2.

## What actually ran

Acquisition90572 completed one fresh CPU Go2 run with the existing locomotion
checkpoint and checkpoint gains. After15zero-command settling ticks, it executed
five ticks each of zero, forward, left yaw, right yaw and zero:2000physics samples
at500Hz,26RGBD frames, and approximately0.1324m of post-settling body path. This
is4seconds of simulation, not a long route or real-time/hardware demonstration.
No native disallowed-contact or body-stability stop occurred.

The new builder retains all non-floor construction statements. It uses one
collision-only plane atzero and one32m visual-only native plane translated+5mm,
whose actual mesh therefore aligns atzero. Existing floor material/surface
arguments, walls, robot, camera, gait and sensor wrappers are preserved. An
explicit initializer imports this builder; no global native-class monkeypatch
or old-builder edit is used. The visual-only entity is excluded from physical
ground/contact identities; existing native calf/foot support-group semantics
are not expanded or claimed to be newly validated contact permissions.

The declared[-4,4]² body workspace plus mount and all5m optical image-corner rays
fits inside the[-16,16]² visual square with4.5359m minimum conservative margin.
Actual capture frusta are checked too. These are scene/acquisition/evaluator
checks, not undeclared world-pose inputs to the navigation policy. This domain
must be reconsidered for another maze; missing/out-of-domain depth stays unknown.

## Reader failure, narrow correction, and verified evidence

The original frozen audit85679 stopped on a representation error: native BOX
data has seven slots (three dimensions and four zeros), while the reader
compared it with a three-element dimension vector. No physics was rerun.

A separate reader checks exactly seven finite slots, exact zero padding and
the SAME2e-7 dimension tolerance. An AST comparison test verifies that this is
the only change to the original audit function. Tests reproduce the old shape
failure and reject incorrect dimensions, nonzero padding, nonfinite or malformed
records. The original reader and its failure remain preserved.

Corrected reader88787 completed and bound its own source/test identities. It
reconstructs every native contact classification and50Hz body/500Hz gyro sample,
rebuilds the captured sensor histories/packets, and reproduces depth acquisition
from unmodified native depth. It checks actual camera poses, clips/intrinsics,
single-sample framebuffer, physical-only ground identities, fixed walls, source/
artifact/native hashes, complete command/camera cadence and unchanged gains.

All26frames meet the fixed1-mm interior ray check:

- 74,228 assessed rays: 47,071 floor and 27,157 wall rays.
- Maximum floor error: 0.045278 mm.
- Maximum wall error: 0.019039 mm.

This check uses the preregistered stride8,2cm box-edge exclusion and0.22..4.98m
range. It is not an all-pixel, edge/discontinuity, broad-domain or noisy hardware
bound. The eight-view earlier extent comparison supplies complementary all-pixel
plane evidence, not a replacement for those missing validations.

## Actual Go2 contact diagnostics

At all2000physics samples, the auditor uses the recorded evaluator body pose
and measured joint positions with the verified URDF primitive supports. This
gives instantaneous nominal physical gaps, not an uncertain-state certificate
or a future gait sweep. World pose/native contacts remain outside sensor packets.

| Foot sphere | Minimum gap, entire run | Minimum gap after settling |
| --- | ---: | ---: |
| FL | -2.113mm | -1.394mm |
| FR | -3.679mm | -2.059mm |
| RL | -3.525mm | -1.075mm |
| RR | -2.094mm | -1.288mm |

Every non-foot primitive stays nominally above the floor in this trace; the
smallest gap is23.143mm at FL_calflower1. Maximum summed ground-contact force
magnitudes reach1045.4N; this sum is not a net body force. These measurements
show that geometric foot/plane intersection occurs during the existing simulated
gait, while non-foot geometry remains separate in this short run. They do not
establish a universal3.7-mm or2.1-mm allowed penetration, inferred contact from
proprioception, or a validated compliance/friction model.

The preceding dropped-sphere assay's3.69-mm impact failure remains unchanged.
This Go2 assay explicitly reports contact dynamics without adding a hidden
zero-/1mm-penetration gate or turning the previous failure into a pass.

## Source/test evidence and identities

Focused49743 passed20 new integration tests. Full83143 passed1,544 across130
explicit files in94.51s. Focused5625 passed8 reader-correction tests. Final
expanded regression18562 passed1,552 across131 explicit files in94.81s.
All listed execution, audit and test handles are terminal. Tested/launched paths
were not edited concurrently with their tests or after their source bindings.

The acquisition binds350 paths (343predecessor plus7new) and12 installed native
paths; the corrected reader additionally binds its exact source and test. The
locomotion checkpoint/config and platform/registry are inherited bound inputs
within the historical source inventory. No JEPA/navigation model was loaded or
trained, and no held-out/sealed data or legacy evaluation was accessed.

Output: `.generated/go2_bounded_floor_robot_interface_development_v1_attempt_001`.

- Launch SHA-256: `fad3666196fa4e3b6eb04f0307e5f6a8066282ffb6215841095cf3a32f4b35a2`.
- Acquisition result: `3e9c84ddd1760789b489ae5aff194d121337f1d8cc1594ff6c1e48968cfefbd6`.
- Corrected reader evidence: `a0b88df8d706e6e037536c519d8508d89c05c630f8a7bd7da589c1c8909594f0`.

## Next work toward complete maze navigation

1. Use this explicitly integrated scene path for the fresh development controller.
   Resolve per-observation measured plane hypotheses independently of the current
   foot query, retaining observed support coverage, missing-floor/wall/overhang
   negatives, provenance and all-view contradictions. Do not simply label all
   upward-facing returns as traversable floor.
2. Specify modelled foot-contact admissibility separately from geometric free
   space and from observed contact. Use the new Go2 diagnostics to identify
   assumptions to validate, not to choose a universal tolerance from maxima.
   Cover foot loads/approach velocities, timing/compliance/friction and non-foot
   negatives with independent development validation; hardware remains separate.
3. Integrate prospective commanded motion, fused speed, observation/repositioning
   actions and explicit computation latency. Demonstrate the actual complete
   sensing-to-command budget rather than frozen-physics or warm-repeat timings.
4. Run complete discovery/return missions, then matched supervised/JEPA predictive
   training, online memory and genuine multistep rollout comparisons on independent
   layouts/seeds/robustness conditions; obtain bounded platform evidence when
   available. Short command execution is not novel-maze navigation or JEPA benefit.

The full scientific goal remains active. No new complete maze, learned navigation,
JEPA contribution, online-memory benefit, independent-layout or hardware result
is claimed.
