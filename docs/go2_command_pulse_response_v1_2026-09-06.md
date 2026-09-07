# Command-pulse response V1: fixed development characterization

Four fresh simulated episodes: nominal_a, lower_friction_a, nominal_b,
lower_friction_b. Friction is 1.0 or 0.15 on robot AND floor. Start A is
(-0.70,-0.25,0.375)m, yaw +0.02; B is (-0.45,+0.25,0.375)m, yaw -0.04.
Physics seeds 2026090657/658, appearance seeds 2026090659/660. A and B are
development repeats with order/start differences, NOT train/held-out labels.
No historical checkpoint is changed, retrained or treated as navigation JEPA.

Following the recorded 15-tick setup, issue 10 initial zero-command ticks.
The ordered command list is forward .12, forward .20m/s; yaw +.03,-.03,
+.25,-.25,+.45,-.45rad/s, with zero lateral command. For each command, issue
first a 2-tick then a 5-tick pulse (100ms/tick), each followed by 20 requested
zero-command ticks. B reverses the complete 16-event list. This counterbalances
order, not exact starting body/gait state. Do not pool those as exchangeable
independent transitions. Fixed braking is not a claim of stationary reset.

Max 386 post-setup command ticks, 40.1s including setup. If visual acquisition
fails or visual excursion exceeds 1m, stop the schedule and drain ten requested
zero ticks under native guards: at most 396 post-setup ticks / 41.1s total.
Native contact, speed >.3m/s, domain exit, fall and tilt stops terminate physics
immediately, retain partial data, and never resume/replace that episode.
Each planned episode is attempted once; an infrastructure failure is terminal.
Capture RGB-D and body/joint/500Hz gyro histories; run frozen visual motion
tracking at 10Hz. Sensor/native state does not choose pulse amplitude or duration.
Native state only supervises stops and scores physical response.

The bank magnitudes .20m/s / ±.45rad/s exceed old servo caps but are within the
existing platform command limits. They are NOT automatically tracking- or
safety-qualified. Keep actual requested and slew-applied command records;
yaw slew is .35rad/s per tick, so both onset and stopping can contain transients.
Do not replace those labels with requested commands when learning dynamics.

Before launch freeze recursive source/input identities, schedule and this
protocol in `.generated/go2_command_pulse_response_v1_attempt_001/launch.json`.
Require 10GiB free-space reserve. Retain exact raw acquisition and all partial
traces. Audit sensor reconstruction, frozen visual/schedule replay, commands,
gains, materials, contacts and actual new spawn. Score each event at pulse end
and after .1,.5,1,2s of requested-zero braking, using sensor-relative labels
and separately native scores. Report missing/truncated labels and zero-command
tail motion, not just successful events or overlapping-window sample counts.
Report terminal speeds and whether each nominally quiet tail actually settles.

All episodes are exposed development characterization, not independent model
validation or a causal matched-state experiment. No predictor is fitted and
deployed on these same episodes. Use response evidence to choose a bounded
execution baseline and future sensor/action-conditioned prediction data; freeze
any fitted model/controller before genuinely new validation. Preserve the full
maze-memory/JEPA/online-rollout/independent-layout/seed objective.

Controlled continuous floor, hidden-robot ideal camera and paused-physics
computation remain explicit limitations. No deployment, real-time, hardware,
novel-maze or final-goal success claim follows from collection completion.
