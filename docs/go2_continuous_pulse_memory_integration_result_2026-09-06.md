# Continuous pulse execution and episodic memory: implementation result

Implemented the next runtime integration layer. The previous physical result
remains three nominal local successes and one low-friction failure. No new
physics, maze mission, place-recognition result or JEPA advantage is claimed.

## What now exists

- `lewm/sensor_anchored_goal_development.py` binds each bounded local request
  to the current episode, frame, time, RGB/depth hashes and measured visual pose.
  The requested body-frame displacement and final heading are transformed once
  into the uninterrupted visual frame. Inputs are copied; subsequent motion
  does not drag the goal along with the robot. No world-map target is required.
- `lewm/anchored_pulse_servo_development.py` generalizes the successful local
  algorithm to those goals, including either signed turn direction. It keeps
  pulse/braking/quiet-hold rules and final6cm/.05rad criteria. The former global
  .48m-x excursion gate becomes a per-goal directional overshoot of .08m and
  a1m anchor-relative excursion; this geometric change needs new physical
  validation. Declared local displacement is bounded at .4m, yaw at pi.
- `lewm/continuous_pulse_execution_development.py` retains one visual frame,
  chronological frame index, failure latch and mission-wide time/pulse/leg
  accounting. Defaults are360s/140pulses/36legs; individual legs retain100s/
  35pulse limits. Idle time counts. Starting a new leg does not reset sensor
  history or global budgets. It records local-goal candidates, not places.
- `lewm/pulse_route_bridge_development.py` connects actual current RGB/body
  packets, uninterrupted attitude and visual evidence to the existing
  `EpisodicRouteHypotheses`. It checks common RGB/episode/orientation bindings,
  uses fresh observed branch bearings for dispatch, and records departure,
  completed local motion, turn views and faults. Return dispatch still requires
  a current candidate matching memory's return direction. Branch validation is
  transactional; malformed proposals do not create live movement attempts.
  Missing terminal images abort memory without fabricating an arrival.
- `scripts/pulse_mission_session_development.py` provides a distinct .20m/s/
  .45rad/s command boundary using the existing native acquisition/guard path.
  The old .35rad/s validator remains unchanged. This class still inherits the
  old fixed-scene initializer: it is not a fresh-scene launcher or permission
  to repeat the old experiment. A new explicit initializer/protocol is needed.

## Verification, and its scope

The focused tests exercise coordinate anchoring/copying, both signed turns,
multiple legs without resetting pose, stale/missing/invalid/reset inputs, fault
timestamps, total budgets, memory-selected outward/return sequences, unverified
home hypotheses, and the new command boundary while retaining the old guard.
The responsive actuator and supplied poses in these tests are synthetic. They
do not prove physical return, calibrated place recognition or traversability.

The read-only checker `scripts/check_go2_anchored_pulse_replay_v1.py` verifies
the previous physical launch/result/raw-audit and all result-artifact bindings.
The parameterized local controller reproduces all864recorded decisions exactly:
263 each for nominal_a/b/c and75 for low friction. Previous outcomes are
unchanged. The replay uses already raw-audited visual evidence; it does not
reacquire sensors, run physics, evaluate the continuous dispatcher or establish
a physical memory benefit. Final checker session74761 exited0.

An earlier check61069 rejected JSON's list representation of the episode tuple.
The checker now explicitly restores that tuple at deserialization; the live
sensor contract was not relaxed and no observation values were invented.
Initial29tests passed in91583;31tests after lifecycle/return additions passed
in11673. The final session-boundary test is included in the full regression.

The full regression passed **2,420 tests across191 explicit files**, session2012,
exit0,194.75s. This includes all32new focused cases. All known test/replay
processes are terminal. The seven new implementation/checker/test sources remain
unlaunched development code; no predecessor source or runtime artifact changed.

Source SHA-256 at verification:

- sensor_anchored_goal: `56c2cbf4c55dee9bae90a94ed57015c359503bc29be4e595ada8ac66793e0081`
- anchored_pulse_servo: `e849330af1504b2beac1d2a0df0915d19463161adb206f2c87ee303d2d642015`
- continuous_pulse_execution: `93cafda23daca47e6c7a6af6553a74842d4b3a3b2aa5319f1caac850dc2b11df`
- pulse_route_bridge: `a42413087b2031ca4ee146b255f95d35487d334ccc5ef0249d5299d1993c14e4`
- pulse_mission_session: `f0f5b934e7f1c66fe4162521a2bd3c95b50dd15ed2541beba88b623ba36c57ad`
- anchored_pulse_replay checker: `6b03b1011a5c8ad2b48c1e2fe2ac9e633c23de2391576802d30af87ed7a7f30f`
- focused tests: `da9763dc5f468a71f5e4262133e4d8ce50271630e598e20a6e127e3a7af8b2b5`

## What still prevents a whole-mission claim

The bridge is an integration component, not autonomous mission scheduling or
an RGB/metric endpoint detector. The caller must own uninterrupted raw RGB-D/
gyro acquisition and supply actual detected branches. A .4m lookahead along a
branch is not the location of a corridor exit. Each recorded visit remains a
provisional event; trusted edges and verified-home claims remain zero/false.
The present return logic matches a remembered direction and advances a bounded
local lookahead. It does not yet aim precisely at a stored metric predecessor
position, and cumulative local tolerances can prevent actual return. Do not
promote an emptied route stack to physical home success.

Next, wire this layer into a separately frozen raw-sensor runtime and a fresh
multi-leg/connected-maze collector, with actual both-direction maze-scale turns,
observed branch/marker selection and independent arrival/return scoring. Bind
the new initializer and command domain before launch. Address metric predecessor
targeting or evaluate the current provisional scheme honestly; do not replace
that test with more same-tape replay. Preserve unknown clearance, optical and
low-friction limitations, then perform matched memory and JEPA predictive/
online-rollout comparisons across independent layouts and training seeds.
Realistic sensing/timing and bounded hardware evidence remain required.
