# Pulse-feedback V1: frozen complete local execution trial

Task unchanged: move to initial-body XY (.4,0)m, then yaw +.3rad and hold;
final planar tolerance .06m, yaw .05rad. Engineered feedback, not JEPA.

Finite forward .20m/s and turn ±.45rad/s pulses use the existing gait. Forward
duration is five 100ms ticks when position error>.12m, otherwise two ticks.
Approach steering error>.15rad chooses a turn; turn duration is five ticks
if angular error>.30rad, otherwise two. Latest completed forward-pulse net
displacement, including braking, estimates its body-frame course offset when
planar displacement>=.005m. This offset is transported with current body yaw
as an explicit persistence assumption. Turn-induced translation never updates
the forward offset. No fitted predictor, native pose or friction label in
control. No command-integration arrival or pose estimator reset.

Every pulse is followed by at least20 requested-zero ticks AND ten consecutive
visual quiet intervals (speed<=.02m/s, yaw rate<=.05rad/s); fail after40 brake
ticks. Initial acquisition also waits20 zero ticks. Slew-applied stopping may
differ from requested zero. Approach arrival within.025m changes task stage to
final orientation. If position subsequently escapes.06m, return to approach
within the same bounded episode. If final yaw is within.05rad, attempt a new
hold verification rather than demand the old optional.015rad internal margin.
Final hold requires ten new quiet zero-command intervals with both endpoints
inside the final pose region. Escape triggers another bounded correction.

Max35 pulses and1000 control ticks (100s); the longer declared budget accounts
for braking after every pulse, not a reinterpretation of old35s failures.
Observed norm>1m or forward X>.48m fails. Retain native contact/speed>.3m/s/
domain/fall/tilt stops; native stops immediately end physics. Controller failures
drain ten requested-zero ticks under those guards. Max51250 physics samples /
102.5s including15 setup ticks and failure drain. Preserve partial data and all
outcomes, with no retry/replacement or in-run retuning.

Four fixed trials, in order: nominal_a(-.65,-.35,0), nominal_b(-.55,.10,.05),
nominal_c(-.75,.20,-.06), lower_friction_a matching nominal_a. Coordinates are
world x,y,yaw; spawn z=.375m. Seeds2026090661..663 and appearance2026090665..667
depend only on start a/b/c. Friction1.0 nominal/.15 lower on robot AND floor.
These are fresh exposed development repeats, not independent novel layouts or
a causal ablation against old controllers. Report nominal successes separately
from the retained low-friction challenge. No promotion from one successful run.

Freeze source, this protocol and raw audit before launching
`.generated/go2_pulse_feedback_servo_v1_attempt_001`. Require10GiB free reserve.
Audit exact RGB-D/body/gyro acquisition and controller replay, commands/slew,
material/gains, actual new spawn and native stops. A complete task requires
sensor completion AND independent native checks at all501 physics poses during
its final one-second hold, including speed/yaw-rate limits. Report duration,
pulse count and native path/joint travel. No recorded torque/power exists;
mechanical energy remains unavailable, not inferred from joint travel.

Controlled continuous floor, hidden-robot ideal camera and paused physics during
computation remain explicit. No real-time, hardware, maze-memory or JEPA claim.
If a declared condition supports repeatable local execution, progress to useful
online memory and full novel-maze missions with matched predictive/multistep/
memory comparisons; do not substitute this engineering task for the full goal.
