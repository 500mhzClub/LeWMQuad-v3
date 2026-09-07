# Unchanged command odometry on longer routes and turns

The completed144-stream replay measured short-horizon command-plus-gyro baseline
error. It does not measure accumulated error through multiple junctions or
nominally in-place turning. Apply the identical frozen estimator, without scale,
bias, threshold or covariance fitting, to all eight already raw-audited continuous
multi-junction routes and all18 already raw-audited timed/gyro-feedback turns.
These source panels are contact-free and remain unchanged; four route final
heading failures and three timed-turn failures are not discarded.

Initialize at each first actual post-settling RGB/body packet, not each route
edge. Never reset translation or relative orientation between edges or during
return segments. Replay every actual100-ms packet through the terminal release.
Record any offclock terminal image separately and never integrate an unobserved
interval. Use the same independent evaluation-only quaternion-vector true
displacement comparison. Report all26 durations, endpoint xy errors, maximum
observed xy errors and zero-translation baseline errors. Nominal zero translation
during a turn is not proof the physical robot stayed stationary. Do not estimate
a covariance or tune a useful-looking subset after seeing these errors.

Fixed fresh output:
`.generated/go2_route_turn_command_odometry_development_v1_attempt_001`.
Bind prior odometry source/result, original source-panel results/launches/audits
and this new replay source/protocol before execution; verify bindings afterward.
This is a no-fitting descriptive extension using already observed development
data, not a retry of the completed144 replay, new physics or independent-maze
evaluation. It does not qualify RGB place association or hardware localization.
