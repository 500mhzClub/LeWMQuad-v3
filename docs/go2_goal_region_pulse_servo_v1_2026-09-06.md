# Goal-region pulse V1: fresh local execution validation

Scientific task remains initial-body XY (.4,0)m, yaw +.3rad, planar error
<=.06m, yaw error <=.05rad and ten new quiet zero-command intervals.
Success additionally requires independent native checks over all501 physics
poses of that final second: speed <=.02m/s and yaw rate <=.05rad/s.
This is engineered feedback on the unchanged learned gait, not a JEPA policy.

Predecessor pulse-feedback V1 failed all four trials. Three lost visual pose;
one exhausted35pulses. Its .025m internal approach criterion required steering
even within the actual .06m position region. This successor changes only that
stage criterion: inside .06m pursue final yaw/hold, outside pursue position.
It does not retrospectively reclassify any failure. Removal of unnecessary
steering is a hypothesis, not evidence that visual tracking failures are solved.

All other controller rules remain: .20m/s forward, +/-.45rad/s yaw, 2/5tick
pulses; forward5 ifdistance>.12m, turn5 ifangularerror>.30rad; approach turns
ifbearing/courseerror>.15rad. Only completed forward pulses with displacement
>=.005m update the body-course offset. Past response persistence is an
unvalidated assumption, especially after turns and under low friction.
Every pulse waits20zero ticks and10quiet intervals; brake fails at40ticks.
No estimator reset, native pose/friction controller input or learned predictor.
Keep35pulse/1000controltick,1m translation/.48m forward-excursion limits.
Keep native nonfoot contact, .3m/s speed, floor-domain/fall/tilt stops and
ten guarded zero-drain ticks on controller failure. Maximum51250physics samples.

Freeze four trials before any new outcomes:
nominal_a(-.62,-.28,.015), nominal_b(-.52,.16,.065),
nominal_c(-.72,.26,-.045), then lower_friction_a matching nominal_a.
World x/y/yaw, z=.375m; physics seeds2026090671..673, appearance2026090675..677.
Friction1/.15 on robot AND floor. Check actual changed physics prefixes.
New starts are development validation in the same floor scene, not novel
layouts or a matched causal comparison against predecessor outcomes.
Report nominal and low-friction results separately; retain every failure.
No retry, in-run retuning, sensor-gate relaxation or success-tolerance change.

Launch exclusively into .generated/go2_goal_region_pulse_servo_v1_attempt_001.
Bind this protocol, controller, scene/session/collector, replay audit and both
focused test files before launch. Require10GiB disk reserve for each episode.
Raw replay verifies sensor acquisition, commands/slew, gains/friction/clocks,
native guards, actual starts and full final hold. Preserve durations, pulses,
path length, joint travel; no torque/power means mechanical energy unavailable.

Controlled continuous level floor, hidden-robot ideal RGBD, ideal gyro and
paused physics during computation remain explicit limitations. No hardware,
real-time, general maze, deployment sensor or JEPA qualification is claimed.
Reliable local execution must lead to persistent online place/branch memory,
complete exploration/backtracking/goal/home missions and matched geometric,
supervised and JEPA predictive-training/online-rollout/memory comparisons on
independent layouts/seeds. Do not equate this local test with that full goal.
