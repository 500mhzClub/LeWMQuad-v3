# Moment-aware fusion: improved nominal estimates, explicit prolonged-weakness limits

The separately named successor completed 10 nominal trajectory replays and four
sensor-degradation cases. Exact replay and independent error/component/composition
verification passed all 4,233 produced estimates. This is implementation and
development-state evidence, **not successful navigation or calibrated uncertainty**.
The previous goal turn was progress; this turn completed the planned moment fix
and its bounded stress experiment. The full scientific objective remains active.

## Completed implementation and verification

The estimator retains time moments of the five causal 20-ms accelerometer bins
instead of substituting their mean in displacement and endpoint-velocity updates.
It preserves original depth rank and constrained components, labels weak motion
as predicted, rejects stale endpoint force, and never erases accumulated position
error when depth constraints recover. The old estimator and its outcomes remain
unchanged. No navigation consumer has yet been connected to either fusion module.

Session 60661 passed 1,234 tests across 114 files in 81.78 s; focused session
77387 passed 45 checks. Tests include independently integrated fine-time motion,
equal-mean early/late acceleration, twelve seconds of switching weak directions
and rotating gravity, inherited causal faults, and an explicit unobserved
intra-bin-jerk counterexample. These analytic tests do not establish a real IMU
error envelope.

Preflight 5433 verified 333 source and 3,843 consumed-artifact bindings. Replay
17907 completed, exit 0. Audit 29510 completed all 14 cases and 4,233 estimates,
exit 0, including exact live depth/gyro/fusion recomputation for the perturbed
cases and exact reproduction of the dropout fault. All launched source, test,
protocol and completed output files are frozen. No replay/audit remains live.

## Nominal complete-history result

All ten fixed nominal trajectories / 3,610 observations pass the same descriptive
replay checks. Worst step error is 1.09151 mm. Worst position error falls from
7.50116 mm in V1 to 2.51507 mm. The three weak-trajectory final errors are
0.30896, 2.51507 and 0.66603 mm. The first slightly worsens from 0.23727 mm;
retain that result rather than reporting only improvements.

Endpoint velocity also improves, but is not uniformly within the declared scale:
maximum full-depth endpoint velocity error falls from 47.7944 to 13.4106 mm/s.
The number of full-depth intervals above 5 mm/s falls from 1,527 to 375 out of
3,582. Interval-average acceleration error is unchanged: 94 of 3,600 intervals
exceed 0.02 m/s², with maximum 0.03976 m/s². Correct temporal integration therefore
does not validate the old uncertainty assumptions.

## Prolonged weak geometry and bias: retained negative evidence

These four cases modify copies of the first 181 frames of one existing north
trajectory, not four independent physical trials. RGB and executed motion remain
unchanged. A central 80-column depth window is retained at ticks 80–139; depth
registration genuinely reports 61 weak intervals, ticks 80–140. Restoring the
full image does not instantly restore full registration against the previous
narrow cloud. Full constraints return on the following interval.

| Explicit intervention | Produced frames | Final position error | First proxy-budget stop | Proxy exceedances |
| --- | ---: | ---: | ---: | ---: |
| Six-second narrow depth, no added bias | 181 | 50.796 mm | Tick 93 | 0 |
| Narrow depth + 0.02-m/s² body-y bias | 181 | 412.665 mm | Tick 93 | 0 |
| Narrow depth + 0.2-m/s² body-y bias | 181 | 3.66948 m | Tick 93 | 96 |
| Total-depth dropout at tick 80 | 80 | 0.176 mm before fault | Sensor fault at 80 | 0 |

All three narrow-depth cases fail nominal position/budget checks. The 0.02 and
0.2 cases also fail the 1-cm step-error check. The dropout case is an explicit
truncation: its 80 pre-fault estimates passing descriptive checks is **not** a
successful 181-frame run. `completed_fixed_frames` is false and the original
fault is retained. No reset, retry or later-frame recovery is attempted.

At tick 93 the declared position scale is 81.1005 mm after 1.4 seconds of weak
motion. Actual position errors there are 0.402, 17.036 and 173.591 mm for the
three bias levels. The large-bias case first exceeds the proxy at tick 85,
before the proxy-budget stop. Thus a budget flag cannot certify safe control
when its bias assumption is false. Predictions after tick 93 are diagnostic
only; their large endpoint errors are not an executed online continuation.

## Scientific interpretation and next implementation

The moment fix removes a concrete timing error; it does not create missing
geometric information. This population supports short conditional state
continuity, not prolonged inertial-only navigation. Bias, initial gravity,
within-bin dynamics, and weak-direction velocity remain material limitations.
Another observer-only replay or an inflated proxy constant is not the next
navigation milestone.

1. Implement a separately named fusion-aware ray-memory consumer. Preserve the
   original depth evidence and distinct predicted displacement; do not adapt
   predicted motion into the old rank-3-only interface. Check current observation
   identity, clock, composition and fault/budget state. Model uncertainty in
   transport of historical observations, rather than adding an uncertainty field
   that leaves clearance decisions unchanged.
2. For an uncertain transported point, inspect the depth-pixel region that its
   declared pose-error envelope can reach, including missing returns and nearby
   surfaces. Merely enlarging the along-ray margin or sampling the nominal point
   does not account for sideways projection error. Current-frame evidence should
   not acquire fictitious historical pose uncertainty. Test monotonic loss of
   supported space as uncertainty increases, thin obstacles, unknown pixels,
   old/new-view conflicts, ground-support roles and budget exhaustion. Any use of
   the present proxies remains explicitly conditional development behavior, not
   a calibrated probability, guaranteed clearance or hardware safety claim.
3. Connect this consumer to an observation-action controller that seeks additional
   geometric/RGB information before prolonged weakness. Account for the measured
   translation during yaw; use observed clearance to reposition or choose another
   view, and stop when no supported action exists. Preserve native 2-ms contact
   stops, the nominal-volume guard, original task metrics and negative regressions.
   Do not postpone exploratory simulation indefinitely while pretending the
   uncertainty model has become hardware-qualified.
4. Give learned prediction a concrete role in that action choice. Collect matched
   yaw, hold, braking and mixed-action trajectories across clearances and gait
   phases. Compare geometric/empirical dynamics, supervised and JEPA predictors
   using identical observations, data, candidate tapes and budgets. Evaluate
   turning displacement, future observation quality and actual selected actions;
   nominal-state replay is not a JEPA contribution. Preserve an RGB-only versus
   added-range control so sensor information is not credited to the objective.
5. Execute fresh continuous exploration/discovery/return development trials with
   the new state/action integration. Require actual return-memory and genuinely
   multistep online-rollout comparisons, independent layouts/model seeds, sensor
   robustness and bounded real Go2 evidence when available before final claims.

Latest actual whole-task simulation success remains 0/2. Neither this study nor
the preceding fusion replay changes that outcome, demonstrates a learned maze
policy, establishes JEPA benefit, or supplies independent-maze/hardware evidence.

## Exact identities

Root: `.generated/go2_depth_inertial_moment_replay_development_v1_attempt_001`.

- launch.json: `dccdc5253fcad33f821922fa482a1e65ab9f53b3331630096e35d47c823d2453`
- result.json: `35f2da577d6a304330a507ee3f06d122e5aac85aee449ceb2dec726abae42b0d`
- independent_verification.json: `9e2e2383552aa8b942b34f101bdbe9c585408651c28bbfa806ea62075735a433`
