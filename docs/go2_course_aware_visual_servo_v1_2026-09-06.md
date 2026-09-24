# Fixed course-aware visual-servo engineering trial

This is a distinct successor to the failed heading-based V1, not a retry or
replacement result. The course estimator/controller were already frozen by
`go2_visual_course_response_v1` before its regression results. The small fitted
response model is NOT used to choose commands; its poor cross-condition transfer
is retained. No controller coefficient is retuned after that result.

Two matched conditions: robot+floor friction coefficients 1.0 and 0.15, with
actual solver values and unit multipliers checked before settling, at every
decision and at termination. New shared spawn (-0.55,-0.20,0.375)m and yaw
-0.08rad; physics seed 2026090649, appearance seed 2026090651. Verify actual
first position/yaw and prefix differences from old fitting data and the previous
servo. Same controlled continuous level floor and explicitly ideal hidden-robot
forward RGB-D channel; these do not qualify unseen terrain or actual optics.

Unchanged pretrained gait, RGB-D/gyro observer and visual-led interface. Controller
uses the six-pose 0.5s causal course estimate. Course below 0.012m/s or an incomplete
window is unavailable; fallback to body heading is an explicit engineering
assumption, not an assertion of known course. Transporting course/body-yaw offset
assumes slowly varying slip. Forward command = min(0.12,1.8*target_distance) times
max(0,cos(course error)), with no persistent minimum speed. Yaw gain 1.5, cap
0.25rad/s. No lateral/reverse command or friction label enters the controller.

Same complete target sequence as V1: initial-frame xy=(0.4,0)m, forward braking,
yaw=+0.3rad, final braking. Same transition tolerances (0.025m and 0.03rad), final
tolerances (0.06m,0.05rad), ten quiet intervals (<=(0.02m/s,0.05rad/s)), brake
timeout 40 ticks, observed norm stop >1m and forward x stop >0.48m. New fixed
control timeout 350 ticks (35s) accounts for the observed slow translation and
heading correction. At controller failure allow only ten zero-command drain
ticks under native guards; a physical stop permits no further simulation.
Including the initial fifteen zero-command settle ticks, maximum simulation
duration is 37.5s. Maintain a 10GiB free-space reserve. No retries or model switch.

Native state is only sensor acquisition, setup/stop supervision and subsequent
evaluation. Record all sensor/controller histories, real requested/applied
commands, stages, gains, materials and partial failures. Physics pauses during
capture and computation; no asynchronous delay or real-time qualification.
Raw audit must reconstruct sensors/commands and independently score the entire
target sequence, including braking drift and failure drain. Compare against
the two failed V1 tasks honestly: changed start, controller and timeout prevent
attributing a difference solely to course feedback. This is initial engineering
development; a later matched ablation is necessary for a causal benefit claim.

Exclusive output: `.generated/go2_course_aware_visual_servo_v1_attempt_001`.
Freeze exact source/input/native identities before output/physics. No source
export, protected benchmark access, learned-policy promotion or goal completion.
