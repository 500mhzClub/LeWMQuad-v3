# Bounded goal-hold visual servo V1: prospective development protocol

New controller and new paired physical start; old failures remain unchanged.
Purpose: complete forward 0.4m, brake, turn +0.3rad, and hold the final pose.
This is an engineered baseline, not JEPA or a learned navigation policy.

The course-aware forward controller, 35s total command budget, 0.12m/s forward
and 0.25rad/s yaw command caps, observed excursion stops and native contact,
speed and domain stops remain. Final tolerances remain 0.06m planar and 0.05rad
yaw. At final braking, yaw error beyond an inner 0.04rad triggers a corrective
turn, at most three times across the whole episode. Correction stops within
0.015rad and restarts the quiet counter. The first turn retains its original
0.03rad arrival threshold. Position error beyond 0.06m during turn/correction/
final brake is failure, not a new forward attempt. No pose estimator reset.

Success requires ten consecutive executed zero-command 100ms intervals with
both visual endpoint poses inside final tolerances, measured translation speed
at most 0.02m/s and yaw rate at most 0.05rad/s. These are sampled observations,
not continuous physical bounds. The evaluator independently checks every native
physics sample during the same final one-second interval for planar/yaw target
and speed/yaw-rate compliance; sensor completion alone is not full task success.
Maximum 40 brake ticks per braking visit; overall time cap applies throughout.
Failed controllers drain ten zero-command ticks under native stops. Physical
stops terminate immediately and preserve partial data.

Two fixed trials, nominal friction 1.0 and lower friction 0.15 on robot and floor.
Both start at (-0.60,-0.10,0.375)m, yaw +0.06rad, physics seed 2026090653 and
appearance seed 2026090655. Verify actual pose and changed startup trace, not
seed labels alone. Wide textured wall and continuous controlled floor are
unchanged; robot-hidden ideal camera and ideal synchronized sensors remain.
The lower-friction approach failure is not expected to be solved by final-pose
correction. Report both trials regardless of outcome, with no replacements.

Freeze exact recursive source/input identities before collection in
`.generated/go2_goal_hold_visual_servo_v1_attempt_001/launch.json`. Exact raw
sensor/controller replay, requested/applied command checks, contact and friction
audits and native scoring follow. Fit no response model from these trials for
use in these same trials. Physics pauses for computation; no real-time or
hardware claim. A changed start and controller do not isolate a causal benefit.
Even 2/2 would not prove repeatability, maze memory or JEPA benefit.
