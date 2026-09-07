# Frozen coupled-pulse feedback room-return assay V1

Purpose: test actual continuous multi-leg return using joint XY/yaw empirical
action prediction and one-pulse-at-a-time feedback. This is a development
execution baseline, not JEPA, autonomous maze navigation or a causal comparison
with the previous changed-start/appearance room assay. Preserve that audited
0/3 result and its low-friction native-failing arrival.

## Fixed trials and execution

Three trials, once each in order: nominal_left, nominal_right,
lower_friction_left. Left actual spawn(-.18,-.32,.375), yaw+.045,
physics2026090691 and appearance2026090695. Right(-.12,.28,.375), yaw-.055,
physics2026090692 and appearance2026090696. Low shares the left spawn/seeds
but uses friction .15 instead of1 on both robot and floor. The same four native
walls at x/y=+/-2.5 m remain, .08 m thick, .6 m high and5 m long. Continuous
physical floor and the distinctive procedural texture algorithm remain.
These are actual new starts and random seeds, not renamed previous rollouts.

Keep the seven declared stages: forward .4 m; signed quarter-turn; forward
.4 m; signed half-turn; return to the observed first-arrival corner; turn toward
stored home; return to stored home and its initial heading. Home and corner are
copied actual visual observations in one uninterrupted RGB-D/gyro frame. The
stage order is experimental instruction, not sensed route topology. Larger
stored-point distances require clipped .4 m subgoals; partial legs cannot
complete a return stage. No place/home qualification from the route stack.

The old empirical table is reconstructed only from nominal_a and nominal_b of
`.generated/go2_command_pulse_response_v1_attempt_001`, audited identity
`af809e5cb6101bf67169e1b2a3f5e66be424b53c316d7f4e40854ee9fd5d1d46`.
Each of six forward/left/right 2/5-tick cells contains two nominal samples,
using visual start-body displacement and yaw at brake_20. The controller
receives only an immutable table, never fitting episode traces, current-trial
future responses, native pose, friction labels or world topology. No online
parameter updates. The launch serializes and binds this exact table.

CoupledPulseServo validates current co-timed RGB-D/gyro evidence, tracks local
unwrapped yaw from the observed goal anchor and jointly plans position and
signed yaw. The target winding is the goal orientation lifted nearest the
anchor yaw plus the requested signed yaw change. Every increment must be
within .20 rad at10 Hz; missing/invalid evidence latches failure, never reset.
The bounded fixed beam planner uses horizon min(24, remaining35 pulses),
width256, the original six-cell table, and the fixed model cost/pose-bin rules.
Only the FIRST planned pulse is dispatched; actual quiet braking and a fresh
observation precede the next plan. A nonempty best partial SEARCH_EXHAUSTED path
may supply an exploratory first action, explicitly not a reachability proof.
An empty path fails PLANNER_NO_PROGRESS. Predicted first endpoints must satisfy
the existing anchor-relative excursion limits. No alternative action vocabulary,
coefficient search, retrospective retuning or model replacement in this assay.

Same .20 m/s forward and +/-.45 rad/s yaw commands; 2/5 control ticks each.
Every pulse is followed by at least20 zero-command intervals and10 quiet
intervals; maximum40 braking intervals. Record predicted and observed complete
pulse/brake XY/yaw response and residual, action index and full plan diagnostic.
Each leg retains1000 control ticks,35 pulses,1 m anchor-relative excursion and
.08 m target-direction overshoot. Mission limits remain3600 ticks,140 pulses
and36 legs, including all idle/handoff time. Failed missions drain10 guarded
zero commands; native physical stops end physics immediately. No estimator
reset, native online pose, fabricated observation or predicted arrival.

Actual local completion requires visual position error<=.06 m, signed yaw
error<=.05 rad and10 NEW quiet zero intervals (speed<=.02 m/s,
yaw-rate<=.05 rad/s). These are the previous numerical goal/hold criteria, with
the signed-turn semantics made explicit. No extra scoring tolerance for a
near-boundary failure. Sensor/model uncertainty is not calibrated, and a model
endpoint within the region does not guarantee native success.

## Resource, source and independent audit

Exclusive output `.generated/go2_coupled_room_return_v1_attempt_001`.
Freeze this protocol, all implementation/checker/test sources, fixed older
fitting inputs and predecessor evidence before launch. No edits, retries or
resume of launched sources/output. The existing gait/gains/native geometry and
raw sensor/command pipeline remain bound. Native collision, speed>.3 m/s,
domain and fall/tilt supervision remain external, not online body clearance.

Observed prior room storage is about1.14-1.19 MB per RGB-D/native-depth frame,
maximum1.372 MB, plus traces/metadata. Budget15 GiB for the three bounded trials,
and require a further10 GiB reserve before launch (25 GiB total). Require15 GiB
free before starting each trial. This is a development storage estimate, not a
hard compression guarantee: sample free space at every decision and latch
STORAGE_RESERVE_STOP below10 GiB, retaining the ordinary zero drain and partial
data. Log this external resource observation for exact fail-stop replay; it
never selects a nonzero action. No deletion of previous evidence.

The independent audit reconstructs actual raw sensors, exact controller and
stored-waypoint memory, command/slew/phase tape, initial prefix/new spawn,
materials/gains and full native guards. Score each completed hold at all501
physics poses over1 second with unchanged .06 m/.05 rad/.02 m/s/.05 rad/s
limits. Separately score UNWRAPPED native net yaw from each observed goal
anchor against the requested lifted target; opposite winding cannot pass.
Full assay success requires all seven stages, no native stop, all local native
pose and signed-winding holds passing, and native final home XY(0,0)/yaw0 hold.
Report all failures and intermediate stages, physical duration/path and
observation/control wall time. Energy remains unavailable without torque/power.

The model is state/history/friction-independent with only two samples per cell;
its 20-tick fitting endpoint may differ from quiet-dependent feedback braking.
Predicted endpoint bounds are not swept-body or collision checks. Hidden-robot
ideal RGB-D/gyro, controlled floor and paused compute remain; this is not
deployment-valid sensing or real-time operation. The ultimate goal still needs
live branch/marker navigation, useful persistent memory, independent-layout/
training-seed matched JEPA predictive-training and online-rollout comparisons,
realistic calibrated sensing/timing, and bounded hardware when available.
