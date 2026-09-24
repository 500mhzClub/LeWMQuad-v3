# Measured settling successor and saved-observation mission prefix

The ninth pilot counted ten zero-request intervals inside the goal region as
quiet, although the beginning of that window still contained braking above the
unchanged 0.05m/s native limit. Add a separately named mission/controller pair
that also requires every counted interval's admitted visual 3D displacement
divided by 0.1s to be at most0.05m/s. Movement resets the full dwell counter.
Preserve the original proximity/zero-request tests, ten-interval duration,
phase transition, mandatory physical return and shared navigation budget.
No native pose or velocity is an input. The estimate is an interval average;
it does not bound inter-frame motion or calibrate pose uncertainty. Original
native continuous-speed and strict sensor/visibility checks remain unchanged.

The new controller's advance method is the frozen registered-floor controller
method with exactly one call-argument change: pass admitted position p instead
of p[:2] to the successor mission. An AST equivalence check binds this limited
derivation. Mapping, model, forecasts, contact checks, action selection,
residual memory and terminal sensor handling remain inherited. The failed
subpixel feature candidate is not integrated.

First run a bounded mission-only comparison of the ninth saved decision stream.
Bind its launch, collection result and complete compressed stream to the exact
identities recorded in the transition diagnosis. Verify every current registered
pose witness and reconstruct the complete old mission receipt exactly before
comparing the candidate. Feed actual previous requested commands. Record every
processed mission receipt. Stop on the first difference in phase, active goal,
hold requirement or terminal state, or at the original terminal/max1880 frames.
Counter and descriptive receipt differences alone are not motion interventions.
Do not consume any later decision once the stopping condition is met.

This compares mission state on saved observations. It does not rerun the image
observer, learned model or controller, nor claim full prospective command-prefix
equivalence. The ninth native raw audit remains independently required.
Candidate integration needs a complete controller replay before a fresh native
experiment. Preserve all sources/results once launched.

Exclusive output go2_measured_settling_mission_prefix_v1_attempt_001 in the
authorized development root. Record hardware and competing jobs, use one CPU
process and one numerical thread, admit2GiB available RAM and64MiB output above
the40GiB storage reserve. Recheck every source/input binding after the comparison.
No simulator, training, model load, native motion, overwrite or predecessor
restart. Admission allowances are not OS-enforced resource limits.
