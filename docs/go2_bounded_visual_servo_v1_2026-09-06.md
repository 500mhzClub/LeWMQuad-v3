# Fixed controlled-floor visual-servo experiment V1

Two engineering trials, nominal coefficient 1.0 and lower friction 0.15 on both
robot and ground. Actual solver coefficients/ratios are read before settling,
before each decision and at termination. The unchanged gait runs at 50Hz and
physics at 500Hz. One predeclared gyro-conditioned RGB-D estimator feeds the
new visual-led motion interface and fixed 10Hz feedback controller.

New shared spawn: x=-0.65m, y=0.15m, yaw=0.12rad, z=0.375m. Physics seed
2026090645, appearance seed 2026090647. The changed position/heading—not merely
the seed—must be verified in native traces against the old fitting prefix.
This is a paired controlled intervention, not independent layouts or held-out
generalization. Both outcomes, partial traces and failure reasons are retained.

## Controller fixed before execution

After 15 zero-command settling ticks, establish the initial visual coordinate
frame. No translating privileged prelude. Target initial-frame (x,y)=(0.4,0)m;
forward proportional gain 0.8, moving speed clipped to 0.04–0.1m/s and multiplied
by max(0,cos(heading error)). Heading gain 1.5, yaw rate limited to ±0.25rad/s.
No reverse/lateral command. At planar error <=0.025m, command zero and require
10 consecutive observed 100ms intervals below 0.02m/s and 0.05rad/s. Brake timeout
40 ticks; target drift above 0.06m fails. Then target initial-frame yaw 0.3rad,
gain 1.5 and the same angular limit. Within 0.03rad command zero; complete only
after 10 quiet intervals with planar error <=0.06m and yaw error <=0.05rad.

Control times out after 200 command ticks (20s), or stops on observed translation
norm >1m / forward x>0.48m. Current visual failure requests zero and terminates.
After controller failure, at most 10 zero-command drain ticks are executed under
the unchanged physical stop guards; there is no recovery or reinitialization.
Native physical stop means no further physics. No other model or gain is tried.
All numerical tolerances are engineering choices, not calibrated safety bounds.

## Scientific and physical scope

Continuous level supporting floor is an explicit environment condition, not an
inference from invisible pixels or loads. The forward hidden-robot camera is an
explicitly idealized control-development channel. Its unqualified aperture and
raster issues remain; no real sensor, general terrain or maze qualification.
Native state is acquired for sensors, startup verification, contact/tilt/height/
speed/domain stops and later evaluation only. The high-level controller receives
visual pose evidence, not world position, floor geometry, future target outcomes
or native control hints. Contact odometry is not required and selects no command.

Capture raw RGB-D/body/fast-gyro histories, material/gain/geometry identities,
every controller input/output, stages, requested/applied commands and native stop
rows. Physics pauses for capture and inference: this is step-synchronous, not a
real-time or compute-delay-qualified run. Maintain a 10GiB free-space reserve.
Maximum post-settling duration is 21s including failure drain, 22.5s total.

After collection, replay actual sensor inputs through the unchanged motion
interface and fixed controller; require exact decision identity. Audit raw
acquisition, command timing/application and physical stops. Independently score
final position/heading, drift during braking, final native speed, visual/native
errors and full recorded progress/failures. Verify new physical initial state.
Completion of this stage is not the JEPA/memory/novel-maze/hardware objective.

Exclusive output: `.generated/go2_bounded_visual_servo_v1_attempt_001`.
Freeze the explicit source/input/native manifest before output or physics.
No source export, protected-data access, model training, retry or old launch edit.
