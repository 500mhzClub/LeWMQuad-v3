# Fresh joint-tracking sensor-feedback room return V1

This is one new three-condition development collection using the separate joint
RGB-D continuity observer and explicit joint-pose execution contract. The prior
32-condition tracking comparison met its fixed continuation criteria, but only
one of eight biased tapes completed per sign and nominal error increased.
That permits this fresh execution test; it does not establish adopted tracking
or navigation. Preserve every prior result and the failed original tracking run.

Use the exact `intent_room_return_scene_development` nominal-left, nominal-right
and lower-friction-left specifications and the existing seven-stage persistent
intent mission. The independently audited inner-arrival predecessor is the
historical setup/control baseline. It used multi-reference gyro tracking without
the measured continuity bridge; therefore this comparison changes the observer
family, not just rotation fitting. No causal joint-versus-gyro-continuity effect
is claimed from these three executions. Both pose and commands are freshly
computed throughout each complete trial; no tape playback or pose reset.

Keep the empirical nominal pulse table, inner 40 mm arrival region, external
60 mm / 0.05 rad hold, signed turn winding, ten quiet final intervals, pulse/
brake scheduling, excursion gates and mission budgets unchanged. At most 3,600
control ticks, 140 pulses, 36 local legs and ten terminal zero-command drain
ticks per trial. The new typed joint goal and current-pose reader retain the
original evidence labels and validate current rotation witnesses. No native
position, velocity, contact or evaluator geometry enters the high-level planner.
The unchanged native contact/speed/domain guard may stop a trial externally.
The low-level gait remains the existing frozen PPO policy.

Launch once under the exact external development root
`go2_joint_room_return_v1_attempt_001`. Read predecessor inputs only through
their exact previously recorded bindings. Record native/dependency/source hashes
before execution and verify them afterward. Bind the prior joint scientific
readout and its source identities before admitting the candidate. This protocol
does not authorize sealed access, source export, old-attempt retry or hardware
movement. The user-authorized ongoing goal supplies development execution scope.

Run one native scene at a time, with one OpenCV/BLAS thread and no concurrent
training or test suite during collection. Current hardware availability must be
recorded before launch. Serial execution is chosen to measure the full loop
without competing native scenes, not asserted to maximize throughput. The prior
CPU tracking benchmark does not establish native-simulation parallel scaling.
Require at least 24 GiB available RAM, 40 GiB artifact reserve plus a 15 GiB
three-trial planning allowance, and 45 GiB free before starting each trial.
These are resource checks, not an enforced filesystem quota. Preserve partial
artifacts and terminal failure if the bounds or infrastructure fail.

Measure acquisition, observation/control, command-interval execution and total
iteration wall time explicitly. The simulator remains paused while observing
and planning: this experiment cannot demonstrate a real-time physical loop.
Report deadline exceedances and which operations each timing includes. Any
future real-time claim requires asynchronous sensing/command aging and physics
that continues during compute. Keep ideal depth/RGB, hidden robot and controlled
level-floor assumptions visible.

The independent reader must authenticate the complete artifact population,
reconstruct raw sensors and contact/stop logic, verify the exact predecessor
750-sample setup and initial RGB pairing, replay every sensor-to-command
decision, and independently score each actual completed native hold and home
hold. Count unavailable poses and accepted pose errors, including orientation.
Preserve every depth residual failure, incomplete stage, stop and timing miss.
Full room-return success requires all seven stages, every completed local native
position/yaw/winding hold, the home hold, no physical stop and all raw depth
checks within their existing tolerance. Report per-trial results without
equating a partial hold population to mission success.

This is a scripted physical return assay in an exposed room, not learned online
planning, obstacle-aware exploration, independent-maze validation or hardware
deployment. Its purpose is to reveal tracking/control interactions on newly
executed trajectories while preserving existing tolerances. A negative result
must guide a mechanism-specific successor, not a tolerance search or repeat.
