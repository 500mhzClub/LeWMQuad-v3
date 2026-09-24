# Motion identification A: execution passed, posture predictor failed

The fresh Go2 simulation completed sustained forward motion, turning and braking
under the declared calibration-arena conditions, preserving one observer/memory
throughout. Its independent raw audit passed. This advances local execution
beyond the startup-only maneuver, but does not establish maze navigation,
independent validation, calibrated support/error bounds or JEPA benefit.
Full discovery/return success remains 0/2.

## Executed and independently reconstructed

Output: `.generated/go2_action_motion_identification_development_v1_attempt_001`.
The launch froze 411 source paths, 4,301 inherited input bindings and 16 native
implementation identities. All 171 expected acquisition artifacts are present.
The new [-1.25,1.25]^3 initial-body non-floor region, static through 8.0 s, was
independently checked at 1.5 s, together with initial velocity and native support/
foot identity. This is an explicit calibration-arena condition, not a sensed map
or general maze/deployment prior. The old region and recording remain unchanged.

After startup and its measured tail, the owner became ready at 2.9 s. The complete
28-target schedule ran through 5.7 s: 0.6 s forward at target 0.12 m/s, braking,
0.4 s yaw at target 0.35 rad/s, braking, 0.6 s forward at target 0.15 m/s, braking.
Three additional actual zero ticks recorded the final tail through 6.0 s.
Acquisition contains 3,000 physics samples, 46 RGB-D observations, 43 controller
decisions and 45 post-settle command ticks. It took 33.99 s wall time.

Raw replay reconstructed sensor/command histories, all controller decisions,
the new setup and native foot/support evidence, per-sample guards and stopping.
There were no physical or sensor stops, disallowed contacts, non-foot ground
contacts, body-stability failures or padded-body region violations. Maximum
active base speed was 0.14952 m/s, below the unchanged 0.3-m/s declared cap.
All 28 full recorded motion-command envelopes contained the sampled padded
articulated body. The separate final terminal-decision horizon is truncated
(29 envelope records total); it is not evidence about an unrecorded future.

The two forward segments displaced the base 50.9 and 55.6 mm respectively;
post-handoff net displacement was 101.0 mm. These are observed outcomes, not
the command targets interpreted as measured velocities. During the final
100 ms, maximum actual linear/angular speeds were 0.02977 m/s and 0.01561 rad/s.
The additional three-tick stopping tail still moved the base 10.20 mm. A zero
target and even a previously quiet window are not literal stationary evidence.

The owner consumed all 46 observations without reset and remained rank 3 at
6.0 s, with combined position scale 34.210 mm and no factored current conflicts.
That scale remains an uncalibrated inherited proxy. Depth checks covered
129,254 interior rays with maximum error 0.06602 mm. Nominal active non-foot
floor gap stayed >=25.516 mm; nominal foot penetration reached 1.911 mm,
reported as a diagnostic rather than a new permitted penetration limit.
Native terminal foot/material/gain identities matched. Inherited Genesis
COM/joint-limit warnings remain unresolved for hardware qualification.

## The nominal predictor is inadequate for articulated motion

All 112 forecasts had fully recorded, actually executed command sequences:
28 overlapping examples at each horizon. These are not 112 independent trials.
The original predictor combines ideal-command body motion with measured
joint-velocity persistence; it is non-learned and was not used to authorize
commands. Its errors were:

| Horizon | Max body-position error | Max joint error | Max primitive-point error upper bound |
| --- | ---: | ---: | ---: |
| 0.1 s | 12.14 mm | 0.181 rad | 70.67 mm |
| 0.2 s | 19.88 mm | 0.444 rad | 173.11 mm |
| 0.3 s | 25.05 mm | 0.690 rad | 265.78 mm |
| 0.4 s | 28.60 mm | 0.935 rad | 345.07 mm |

An upper bound alone would not prove a 50-mm model-error assumption false.
A separate posthoc calculation therefore checked actual URDF primitive-centre
errors under the recorded root/joint poses. The FR foot-centre error reached
316.48 mm at 0.4 s from motion index 24, the first braking target after the
second forward segment. Fifteen of 28 examples at that horizon exceeded 50 mm
in at least one primitive centre; one already did at 0.1 s. Thus the earlier
interface-only 50-mm hypothesis is directly contradicted for this baseline,
not merely unsupported by a loose upper bound. It was never an acceptance
limit for this run.

To isolate posture extrapolation, a posthoc control held the current joints
fixed while retaining the exact same ideal body forecasts and executed commands.
At 0.4 s its worst primitive-centre error was 71.72 mm and its point-error upper
bound 76.82 mm; four of 28 centres still exceeded 50 mm. Position persistence
is better on this trace but still insufficient to validate a 50-mm envelope.
This comparison was motivated by A, not independently evaluated or silently
substituted into the saved predictions. Both results remain explicit in
[the derived descriptive metrics](go2_action_motion_identification_development_v1_derived_metrics_2026-09-06.json).

The sampled maximum material-point displacement-rate upper bound was
1.3066 m/s. A 2-ms endpoint displacement calculation is not a bound on
instantaneous or between-sample point speeds. Neither that value nor the
observed error extrema should automatically become certified action limits.

## Timing limitation and next implementation

All 43 recorded acquisition/controller/execution component totals exceed
100 ms: minimum 180.21, median 235.36, maximum 265.09 ms. The audit field is
named `full_loop_wall_ms`, but source inspection shows that it sums those
components and omits packet/image assembly and bookkeeping between their
timers. Treat it as a component total/lower bound, not a complete end-to-end
measurement. The next collector must add an outer timer spanning the entire
decision loop. The current evidence already rules out claiming 10-Hz real-time
execution on this path; simulation-time cadence is not wall-clock readiness.

Next fit and freeze a compact action/state-conditioned response model using A
only, retaining ideal-body/joint-velocity and joint-position-persistence
baselines. The fit must address command-dependent gait/braking response rather
than extending a large instantaneous joint velocity for an entire horizon.
Record all feature/normalization/regularization choices and distinguish fitted
error from validation. Then implement a new B collector/source binding for the
already declared different command schedule; correct the timing measurement
there without editing or rerunning A. Do not launch B until the A-only model and
reporting rules are frozen. B remains unlaunched; no model has yet been fitted.

After local validation, integrate the complete discovery/marker/return mission
and finish matched supervised/JEPA/geometry, genuine multistep-online-rollout,
memory, independent-layout/seed/robustness and bounded hardware work. This
experiment is not a learned navigation policy or completion of those aims.

## Verification and custody

- Focused 16107: seven tests passed, 8.41 s. Expanded focused 32588: ten tests
  passed, 8.53 s, including actual endpoint scoring, truncated/unexecuted futures
  and raw-command mismatch rejection.
- Preflight 98609/59673: identities verified; fixed output absent before launch.
- Regression 1299: 1,796 tests across 145 explicit files passed in 147.54 s.
- Physics 38769: acquisition complete, exit 0.
- Independent raw audit 70696: replay complete and bounded execution passed,
  exit 0. Derived metrics 47061 and posthoc posture comparison 60770 completed.
- No launched source, input or protocol was edited during or after execution;
  original trial/diagnostic results remain intact. No sealed access occurred.
  All listed handles are terminal. No GPU training or hardware actuation ran.

Identity witnesses:

| File | SHA-256 |
| --- | --- |
| launch.json | `874ae72a44e99dee15fbaf0a6c5c826ec1f9d427b23bd852780453d645e3491f` |
| result.json | `c39ddd478788e3053f0484928a8ff06ca736cbb86a26f9552200711d0063171f` |
| raw_artifact_audit.json | `2f6a6d8c257b20990e5a587a57d97ff3321b44c1c8a370fdba2c27fa01f82b46` |
