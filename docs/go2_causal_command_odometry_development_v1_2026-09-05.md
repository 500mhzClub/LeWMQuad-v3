# Command-plus-gyro translation baseline: fixed descriptive replay

Replay the existing144-trial successive-control panel, whose corrected full
physical audit passes, without new physics, model fitting, policy changes or
checkpoint selection. This asks how much translation error remains if the gait's
applied velocity command is treated as body velocity. It is explicitly a
comparison baseline, not a translation measurement or certified localization.

Start at each actual teacher-terminal conditioning packet. Consume each actual
subsequent100-ms RGB/body packet through the recorded release, with the existing
50-Hz relative-gyro integration and strict causal command-history contract.
The command sample at time t describes the interval ending at t: integrate that
sample, not the previous interval's command. Rotate its x/y commanded velocity
using the average of the interval's endpoint relative rotation matrices and
integrate for100 ms. Initial relative translation and rotation are zero/identity.
No true initial world heading, future commands, actual twist or world pose enter
this estimator. Its rotation approximation and command-tracking error remain.

Evaluation alone reads the SHA-bound actual pose trace and independently rotates
true displacement into the initial body frame using quaternion-vector rotation.
Report per100-ms relative xyz estimates, true evaluation displacement and xy/xyz
errors, alongside a zero-translation baseline. Exclude native-contact terminal
images and off-command-clock terminal images explicitly; no extrapolation to an
unobserved endpoint. Report variable-duration last-precontact-control endpoints
separately from fixed-four-second control endpoints, with missing counts and
duration. Release observations stay a separate stage. All144 trials and all six
methods remain in accounting. Per-method differences describe different visited
states, not causal effects of method choice. No thresholds, scale/bias correction,
uncertainty covariance or best subset are fit on this replay.

Fixed fresh output:
`.generated/go2_causal_command_odometry_development_v1_attempt_001`.
Bind source and already completed physical-result/audit identities before replay;
verify them again afterward. Preserve terminal failures without retries. This
replay grants no deployment, final-maze, training or hardware authority. Passing
synthetic clock/causality tests cannot qualify metric translation; actual errors
must inform whether visual/inertial/kinematic estimation is needed next.
