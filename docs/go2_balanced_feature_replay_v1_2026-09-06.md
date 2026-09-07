# Fixed spatially balanced feature replay V1

Compare the frozen primary-first multi-reference observer with a distinct
spatially balanced SIFT frontend on every recorded frame of nominal_left,
nominal_right and lower_friction_left from the completed intent-return assay.
That batch remains0/3 full returns with one passing final-home hold. Its right
failure at1809 has24/48 inliers and5 current grid cells; every retained reference
fails. No scientific recovery or generalization claim follows from this replay.

The new frontend detects SIFT candidates without a global response cap, ranks
by descending response with stable original-order ties, retains one orientation
per half-pixel location, and allocates at most600 descriptors round-robin over
occupied160x160 pixel cells in sorted cell order. Empty/short cells redistribute
their unused quota; image resolution and descriptor budget remain unchanged.
This is one fixed frontend hypothesis, not a coefficient/threshold sweep.
Detecting more candidates can cost more compute even at the same descriptor
budget. Do not claim matched compute or realtime qualification.

Keep mutual0.7 descriptor ratio, duplicate-location checks, forward/backward
LK, depth lifting, rigid pruning,0.6 inlier fraction, six-cell support, gyro,
consecutive increment and reference-conflict gates unchanged. Keep the same
eight-reference primary-first logic and pose-frame lifetime. No reset after
failure, native input, weaker acceptance, inertial position fallback or
retrospective reference choice using native error. The inner-goal controller
is separate source work and is not executed by this replay.

Run both observers on the same actual packets in all4705 frames, including
terminal drains; validate original poses against every original decision.
Retain failure-latched unavailable frames rather than omitting failed runs.
Persist both complete estimate streams before loading native pose for scoring.
Report first failure, coverage, maximum/final available native error, fallback
frames, descriptor/candidate spatial counts and measured observer wall time.
Timing is a development diagnostic, not an isolated hardware benchmark.

Exclusive output `.generated/go2_balanced_feature_replay_v1_attempt_001`.
Freeze this protocol/module/replay/test closure at launch. Pin the exact
external collection launch/result/audit and artifact hashes; preserve separate
ordinary source/input and external artifact verification before and after.
Do not modify or resume the previous collection, change output roots, delete
data, copy source trees or access sealed material. No new physics or model
training occurs. Fresh closed-loop evidence is still required for any eventual
controller change, and novel-maze/matched-JEPA/memory/hardware aims remain open.
