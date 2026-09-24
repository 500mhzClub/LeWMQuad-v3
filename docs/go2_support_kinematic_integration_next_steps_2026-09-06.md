# Next: make local support usable without inventing unseen terrain

## Execution update

The causal consumer and fitting comparison are now implemented and audited:
[result](go2_causal_support_kinematics_result_2026-09-06.md). Both fixed centre
and rolling-foot hypotheses completed; nominal mean errors improve with the
rolling correction, but worst-case errors and unavailable intervals remain.
Follow the [fresh contact-motion challenge preparation](go2_support_motion_challenge_next_steps_2026-09-06.md)
next. The original steps below describe the completed implementation's intent
and the still-required physical/causal limitations, not an instruction to
rerun or retune its frozen fitting attempt.

The [ideal foot-load reconstruction](go2_ideal_foot_load_reconstruction_result_2026-09-06.md)
provides a numerically audited hypothetical local-load stream. It is not a
controller and not a calibrated stock-Go2 sensor. Prior camera results still
exclude claiming qualified sensing. Keep the full scientific goal intact.

## Immediate bounded implementation

1. Build a NEW causal load/joint/IMU consistency consumer, separate from frozen
   policy packet definitions. Consume the recorded 50-Hz joint q/dq and body
   gyro/accelerometer packets with co-timed load histories; reject missing,
   stale, mismatched-identity or incompatible-calibration data. Do not load
   native root pose, world velocity, contact position, ground labels or original
   guard results into it. Acquisition-side force-frame conversion stays outside
   the consumer, as it does for the existing virtual IMU.
2. Derive foot positions and joint Jacobians from the declared URDF chain.
   Differentiate with independent numerical/synthetic checks. For a candidate
   stance foot, the centre-stationarity hypothesis gives
   `v_body = -(omega_body × r_foot + J_foot(q) dq)`.
   This is a HYPOTHESIS, not measured velocity: a spherical foot can roll and
   slip, and its centre need not be the stationary contact point. Preserve this
   discrepancy for native-only scoring. Do not silently replace the foot centre
   with evaluator contact coordinates to make the hypothesis pass.
3. Compute disagreement among simultaneously loaded feet and cross-check against
   RGB-D motion where available. Common-mode slip can make feet agree while the
   body velocity is wrong; low residual cannot certify no slip. Keep explicit
   unknown physical bounds. Separately transform vector loads using sensed
   kinematics and an explicitly conditional IMU-up estimate; magnitude alone
   cannot distinguish upward support from lateral loading.
4. Save all causal predictions before native scoring. Diagnose startup and
   forward/left/right/braking separately on fitting data. Quantify rejected and
   weakly constrained intervals, rolling/slip residuals and support-height
   incompatibility; do not report only the quiet startup frame. The current
   ground-only tape cannot validate contact-type discrimination, so collect a
   separately frozen challenge protocol when the consumer is ready. No adaptive
   reuse of the already exposed validation trial as fresh evidence.

## Necessary evidence before control

Keep three distinct decisions rather than one permissive “supported” Boolean:

- Current support: conditional contact loading/kinematic consistency at identified
  feet, with explicit local uncertainty and slip limitations.
- New support: proposed landing regions must have observed traversable terrain
  under the same error model. Loaded current feet do not fill in holes between
  or beyond them; a fitted support plane is not a global floor prior.
- Body/leg/brake sweep: prospective gait response and observed non-floor geometry
  must cover the actually intended motion and stopping trajectory. A changed
  camera mount needs its own optical and physical mounting verification.

No current-result flag can release an existing controller gate. Implement a
separately named execution consumer only after these obligations have evidence,
then obtain fresh reserved short start/forward/turn/brake trials with a native
stop-only supervisor. Failure should stop, not reset odometry or bypass unknowns.

## Deployment and scientific return

Determine which channels are actually available/calibrated on the target Go2.
The raw scalar foot fields and motor torque estimates are not interchangeable
with ideal three-axis load cells. Preserve the ideal-sensor arm as a simulation
capability study; use a separately declared feasible-sensor arm for deployment.
No hardware record or torque measurement exists in the current motion tape.
Do not fabricate either from a privileged simulator force or policy command.

Return promptly to full local execution and persistent place/branch memory,
exploration/wrong-branch recovery/goal discovery/home return. Compare identical
sensors, gait, budgets and memory across geometric, supervised and JEPA models;
test predictive training separately from genuine online multistep rollout and
memory contribution on independent layouts and training seeds. Complete full
loop timing, robustness and bounded hardware testing when available. Sensor
interfaces and unit tests are prerequisites, never the scientific endpoint.
