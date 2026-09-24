# Frozen RGB-D replay on the friction challenge

Development diagnostic, fixed before executing this replay. Both existing
24-second recordings have already been exposed through their raw/support audit;
this is not a reserved final evaluation. The nominal 14.5-second physics prefix
exactly repeats fitting data. No physical execution, model fitting, threshold
change, keyframe change, reset, source export or protected-data access.

Run the unchanged `RigidRGBDKeyframePose` joint and gyro modes on all 226 RGB-D
frames in each condition, with their acquired fast-gyro histories. Joint here
means jointly estimated rigid rotation/translation, not learned joint encoders.
All predictions in both conditions are saved before loading native evaluation
poses or foot-support predictions. Retain terminal visual failure and mark all
later observations unavailable; never reinitialize it. Record valid depth pixel
counts, accepted correspondence/conditioning evidence, keyframes and runtime.

Score drift and relative orientation against native pose in the initial body
frame. Score each 100ms translation increment in that same frame. Join the exact
six endpoint-inclusive 50Hz support samples, reporting complete-contact and
any-dropout strata, all eight executed phases and every contiguous run of missing
contact samples. Missing estimates are not zeros. Report their counts separately
from conditional error. The shared interval endpoints intentionally allow one
missing sample to affect two adjacent camera intervals. Sample ranges do not
claim exact physical dropout onset or duration. Visual success at 10Hz does not
establish motion observability or a safe command throughout the intervening 100ms.

Primary question: do these unchanged visual estimators retain useful motion
estimates in the observed intervals where foot-contact odometry is unavailable?
No measured maximum is a physical error bound. Both models share images/depth
and ideal gyro; disagreement is not independent truth. Retain hidden-robot
rendering, unqualified aperture/clipping, ideal sensor and deterministic-prefix
limitations. This is neither a fusion policy nor evidence of JEPA advantage.

Exclusive output: `.generated/go2_friction_frozen_rgbd_dropout_v1_attempt_001`.
Bind predecessor acquisition/audit identities, exact inherited input/source and
native dependencies, plus new protocol, replay and focused tests before output.
Use the existing CPU environment, one OpenCV thread, no new simulation/GPU.
After interpreting these results, develop explicit dropout-aware multimodal
fusion and prospective action/brake response, then genuinely fresh validation
and short sensor-only execution. Full memory/novel-maze/JEPA matched comparisons
and hardware evidence remain the objective, not this isolated diagnostic.
