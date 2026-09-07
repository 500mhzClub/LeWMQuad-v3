# Frozen continuous room-return pulse assay V1

Purpose: validate raw-sensor ownership, continuous multi-leg execution, larger
signed turns and actual return to sensor-recorded waypoints. This is a scripted
motion assay, not autonomous maze exploration, learned navigation or JEPA evidence.

Three fixed trials: nominal_left, nominal_right, lower_friction_left, in that
order. Left spawn(-.30,-.20,.375), yaw+.025, physics2026090681/appearance2026090685;
right spawn(-.25,.20,.375), yaw-.025, physics2026090682/appearance2026090686.
Low friction shares left start/appearance, with .15 versus1 on robot AND floor.
Four walls at x/y=+/-2.5m, thickness.08m,height.6m,length5m, same distinctive
procedural appearance algorithm. Continuous physical floor remains. This room
supplies actual surfaces around larger turns instead of the old one-front-wall
fixture. It is a new environment, not a matched causal controller comparison.

Stages: forward .4m; turn sign*pi/2; forward .4m; turn sign*pi;
return to the observed first-arrival corner; turn toward stored home;
return to stored home with final initial heading0. All coordinates retained
online come from current visual observations, never simulator world state.
The stage order is externally declared experimental instruction, not a detected
route. Home/corner are observation records, not qualified place identities.

RawPulseExecution owns one VisualLedMotion/gyro stream for the entire trial.
ContinuousPulseExecution owns unchanged mission-wide360s/140pulse/36leg bounds;
each local leg retains100s/35pulses. No estimator reset between stages. Local
body displacement<=.4m and yaw request<=pi. Larger stored-point distances use
bounded subgoals; a clipped partial leg cannot complete its return stage.
Planar targets are transformed using the current body's invertible horizontal
2x2 rotation block; poorly conditioned transforms fail instead of projecting
away body tilt. Final stored-point legs target that point exactly in the visual
frame. This remains uncalibrated pose evidence, not localization assurance.

Same .20m/s forward and +/-.45rad/s yaw pulse domain;2/5tick durations, at least
20zero braking ticks and10quiet intervals,40maximum brake ticks. Final local
goal .06m/.05rad tolerances,10new quiet zero intervals with observed speed<=.02
and yawrate<=.05. Per-leg anchor-relative1m excursion and directionalovershoot
<=.08m beyond goal. Native contact/.3m/s speed/domain/fall/tilt guards retained;
native stop ends physics immediately. Controller failure drains10guarded zero
ticks and retains partial data. Maximum181250physics samples/362.5s per trial
including15setup ticks. No retuning, retries, pose resets or omitted outcomes.

Freeze protocol, source, tests and audit before exclusive launch at
`.generated/go2_room_return_pulse_v1_attempt_001`; require10GiB free before each
trial. Full raw audit reconstructs sensors/contacts/clocks, exact runtime/goal
dispatch and observed-waypoint memory, commands/slew, material/gains and actual
new spawn/prefix. Score every completed local hold at all501physics poses.
Full assay success requires all seven stages, no native stop, all local holds
passing, and independent final home XY(0,0)/yaw0 hold with unchanged.06m/.05rad
and.02m/s/.05rad/s limits. Report all failures, durations, paths and intermediate
stage/leg scores. Torque/power absent: energy remains unavailable.

The raw route-bridge owner is implemented/tested as a separate integration path;
this scripted assay uses stored metric waypoints, not invented RGB branches or
provisional-stack completion. It cannot establish an episodic-memory advantage.
Next: live branch/marker-driven connected-maze missions and matched memory,
predictive-training and actual online multistep rollout studies on independent
layouts/seeds. Hidden-robot ideal RGBD/gyro, uncalibrated sensing, floor/clearance
assumptions and paused compute remain; realistic timing and hardware remain.
