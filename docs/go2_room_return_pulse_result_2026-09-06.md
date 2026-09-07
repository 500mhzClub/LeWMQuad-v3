# Continuous room-return pulse result: no complete returns

The three fixed physical simulation trials are terminal: **0/3 full room
returns**. This is a negative execution result, not novel-maze or JEPA success.
The earlier independently audited three nominal short-goal successes remain
valid within their narrower scope. The ultimate scientific goal remains active.

## Physical outcomes and audit

The independent raw audit completed with **RAW_RETURN_AUDIT_PASS**: all 4,448
runtime decisions replay exactly, all 4,478 depth checks are within 1 mm, and
raw clocks, sensors, commands/slew, materials/gains, fresh initialization,
observation memory and native guards pass. Audit pass means faithful evidence,
not task success. Four of five declared local holds pass native scoring.

| Trial | Visually completed stages | Native-passing completed holds | Failure stage | Simulated duration |
|---|---:|---:|---|---:|
| Nominal left | 1/7 | 1/1 | Quarter-turn, pulse limit | 102.0 s |
| Nominal right | 3/7 | 3/3 | Half-turn, pulse limit | 172.7 s |
| Low friction left | 1/7 | 0/1 | Quarter-turn, pulse limit | 177.3 s |

Collection contains 4,448 decisions, 4,475 command intervals, 4,478 RGB-D frames
and 226,000 physics samples, totaling 452 simulated seconds including setup and
failure drains. All artifacts expected by the collector exist. None of the
three runs triggered native physical-stop supervision. Collection completion is
not motion success, and no return/home stage was completed.

Every completed hold is scored at all 501 native poses across a full second,
with fixed 0.06 m planar, 0.05 rad heading, 0.02 m/s speed and 0.05 rad/s yaw-rate
limits. The nominal-left forward hold has maximum position error 0.0479993 m.
The nominal-right forward/quarter-turn/forward holds have maxima
0.0462575/0.0534166/0.0411339 m; all their remaining hold criteria pass.
The low-friction forward hold has maximum native position error
**0.0600306616 m**, which fails despite visual acceptance. Do not add a scoring
tolerance after seeing this near-boundary result. Pose uncertainty is still
uncalibrated. An independent normalized-Hamilton-quaternion calculation agrees
with the nominal audit metrics within 1e-10 and independently obtains the
low-friction failure, subsequently confirmed by the completed raw audit.

## What this establishes, and what failed

Actual raw-sensor ownership, continuous visual/gyro state, multi-leg command
dispatch and sensor-recorded metric waypoints now run in simulation without
pose resets. The right run physically completes forward, quarter-turn and
forward again. This advances beyond source-only interface tests. Four actual
room walls supply surfaces around turns; this is a new environment rather than
a matched causal comparison with the preceding one-front-wall fixture.

The separable controller does not handle translation induced by turning. The
left quarter-turn leg executes 35 turn pulses, accumulates **+4.74095 rad** net
native rotation and about 0.0705 m net planar displacement, without completing
the requested +pi/2 pose. Near the intended heading its position has moved
outside the turn-goal region; subsequent position-directed steering can keep
rotating instead of converging jointly in position and heading. Nominal visual
tracking remains available, with maximum position errors about 7.02 mm left and
9.99 mm right. These traces do not support blaming a visual dropout or relaxing
the perception gates. Low-friction tracking also stays available, with maximum
position error 3.84 mm. Native path lengths are 1.5165/2.4051/2.7678 m for
nominal-left/nominal-right/low-friction, respectively.

There is also a turn-semantics defect. The nominal-right half-turn requests
**-pi**, but its first pulse commands positive yaw and the leg ends at +4.43719
rad net native rotation. The frozen goal stores a wrapped final orientation,
not signed winding. An orientation error near pi may therefore choose the
opposite direction. No signed-half-turn success may be claimed, and the frozen
protocol and result must not be retroactively rewritten to conceal this gap.

The room program has a declared route order and stores observed metric
waypoints. Its separate raw episodic route-bridge owner is implemented/tested,
but live RGB branch/marker-driven exploration and a physical memory benefit
are not demonstrated. A local hold is not a verified place, edge or home.

## New planning baseline and transfer evidence

`lewm/coupled_pulse_rollout_development.py` implements an **offline empirical
action/duration baseline**, not JEPA or a deployed controller. It jointly
composes start-body XY displacement and yaw, searches the six measured
forward/left/right pulse-duration cells, and distinguishes final orientation
from net signed winding. Predicted endpoints, endpoint excursion bounds and
search results grant no motion or body-sweep permission. Search exhaustion is
not proof of infeasibility. The model has no calibrated uncertainty.

The six response means are fitted only from the older two nominal pulse
characterization episodes, two samples per cell. Completed room responses are
evaluated but never fitted. For nominal-left/right turn pulses, planar RMSE is
6.016/6.117 mm versus 8.314/7.865 mm for zero translation; yaw RMSE is
0.02080/0.01854 rad versus 0.16224/0.15776 rad for zero rotation. For forward
pulses, the old yaw means are worse than zero yaw in both nominal runs.
Low-friction turn planar error rises to 14.795 mm and forward planar error to
32.756 mm. These correlated development events are not independent validation
trials. State/history, friction and fixed-20-tick versus quiet-dependent braking
horizon differences remain unmodeled.

Read-only model planning finds candidate forward, both quarter-turn and both
half-turn trajectories. Half-turn candidates include forward compensation.
Those endpoints are generated by the fitted model, not observed counterfactual
robot trajectories; they do not demonstrate improvement over the old controller.
The next action is a distinct one-pulse-at-a-time feedback controller and a
prospectively frozen physical evaluation, not an open-loop rollout deployment.

Full explicit regression: **2,443 tests passed across 193 files in 201.41 s**.
This includes nine room scheduling/ownership cases and fourteen analytic or
synthetic coupled-planner cases. Synthetic responsive plants do not establish
physical return. The preceding room-collection regression was 2,429 tests.

Follow the [joint pose/action plan](go2_coupled_pose_action_planning_next_steps_2026-09-06.md),
then actual observed-branch maze missions, matched memory/predictive-training/
online-rollout studies on independent layouts and seeds, deployment-valid
sensing/timing, and bounded hardware. Hidden-robot ideal RGB-D/gyro, continuous
floor and paused compute remain explicit limitations. Energy is unavailable
without torque/power. No intermediate success completes the original goal.

## Evidence identity

Room output: `.generated/go2_room_return_pulse_v1_attempt_001`.
The launch binds 641 sources and 31,056 inputs; launched sources remain unchanged.

- Launch SHA-256: `a3cefd2c431248523e28be9f6ab38826f52220a5f33594246eba93c72f45b5cf`
- Collection result: `8d86bb2d67885dbc4a1702ce156fc4bcf2c06702f14eddc5329a41b761382a93`
- Raw audit launch: `d5a90be1111e87650b342f674f7357e52c304beb7b5e71ab1103bfc875d7506e`
- Nominal-left evaluation: `a3e2a70e9c964aa5d756dd90fd51c733c57d5a0cf500ca21b4a404adfb1c716c`
- Nominal-right evaluation: `1a6c1243940a58a811895c34eacbda925311d6daaf6c2d3035c197773014be28`
- Low-friction evaluation: `367b8d862c05db0ecefa5c6a4bb114d25ac268dc13bf22ef3f518bfb44fd59a8`
- Completed raw audit: `eda0929e0088af65c38c36e2388e982b6e25928fbd55c8fb76533cd1f43aaf9f`
- Fitting pulse audit: `af809e5cb6101bf67169e1b2a3f5e66be424b53c316d7f4e40854ee9fd5d1d46`

The exclusive offline diagnostic completed at
`.generated/go2_coupled_pulse_rollout_diagnostic_v1_attempt_001`:

- Launch: `34fc1e379bc95ac35814c3c18abb7bccf2a5f6465597aff8abd6759e830335b1`
- Result: `500e0d973ce4b6d2f119af888557af0780dab84475b323f45df87ed275a4af47`

Its five model-generated candidates use 6/7/8/16/17 pulses for forward,
left-quarter/right-quarter/left-half/right-half. Some predicted endpoints are
near the task boundary; they have no physical error guarantee. No model was
used online, no room response was fitted, and no new physics was launched by
this diagnostic. The room and offline diagnostic sources are now bound; use
distinct successor files for changes.
