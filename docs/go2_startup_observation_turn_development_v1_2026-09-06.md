# Proposed fresh setup-conditioned observation-turn check

Status: controller, native-geometry reader, fresh physical launcher, live
admission/contact wrapper and raw replay auditor are implemented. Prelaunch
verification is in progress; no physical trial has run yet. This document is
not an execution result or a claim of contact, real-time, maze or hardware
qualification. It is frozen with the explicit launch source inventory.

## Scientific question

Can a promptly started bounded yaw observation action obtain full-rank depth
motion and settle, before the unchanged inherited-plus-prior uncertainty budget
or the explicitly supplied starting region expires? The preceding zero/forward/
turn interface tape never acquired full rank. Its no-prior frame-1 and explicit-
prior frame-12 stops remain unchanged; it must not be rerun or rescored.

## New trial's explicit conditions

Use one fresh development copy by construction of the existing four-wall
bounded/aligned-floor arena, with a new scene ID/seed/output and the same Go2
gait checkpoint, gains, camera, sensor histories, native solver and materials.
Planned scene ID: `go2-startup-observation-turn-development-v1`; seed:
`2026090602`; output:
`.generated/go2_startup_observation_turn_development_v1_attempt_001`.
At most 20 controller command ticks follow admission, plus a three-tick recorded
zero-command tail when native execution remains available. A terminal failure
must not be retried in that output or silently replaced by another initial state.
Do not export a whole tree or read sealed material. Bind the new complete
explicit source/input inventory before launching, including the native geometry
reader's installed implementation dependencies.

Settle with the established 15 zero-command ticks. Before enabling the new
controller, independently check the actual initial velocity, complete static
non-floor inventory, plane identity, support-group forces and native foot-sphere
geometry identities. Save raw evidence and a narrow admission containing only
episode/epoch, setup-definition/report hashes and check outcomes. Do not pass
world pose, environment geometry, native contact values or scene labels into
sensor packets, learned models or navigation consumers.

The velocity prior remains zero mean with 0.02 m/s Euclidean radius. This NEW
trial proposes a [-1,1]^3 metre initial-body non-floor-clear cube, expiring two
seconds after admission. This is deliberately distinct from the prior recorded
checker prism [-1,-0.75,-0.5]..[1,0.75,0.6]; that earlier result is unchanged.
Reason: the general all-joint-angle URDF sphere has radius 0.7444115633 m.
Adding 4-cm padding and a 0.12-m prospective translation allowance already gives
0.9044115633 m before current pose uncertainty. The old prism cannot enclose
that model. Validate the new cube in the fresh scene before motion; reject the
trial if it is not clear rather than resizing it after a failed check.

The starting support model is the explicitly verified static native z=0 plane
with the unchanged solver/material settings. The trial permits native contact
only between the individually matched 22-mm foot spheres and that ground,
alongside the existing disallowed-contact checks. This is an experimental
contact-model assumption, not a newly validated hardware model or an arbitrary
penetration tolerance inferred from a previous maximum. Record all actual
foot/ground penetration, forces, friction and solver parameters for audit.

## Prospective envelope and sensor-driven controller

The geometric sphere follows from sums of fixed kinematic translation norms,
collision-origin offsets and primitive circumspheres. It encloses every joint
angle and every root orientation, not merely observed posture history. It does
not by itself model native numerical contact margins or future root translation.

For each proposed command, expand the sphere by 4-cm padding, the full current
combined position scale, and 0.3 m/s times a 0.4-s command-plus-stop horizon.
The 0.3-m/s base-speed cap is an explicit development assumption. The physical
wrapper must monitor it at 500 Hz after admission and retain any violation as a
failure; such after-step checking is not a hardware safety guarantee. Ensure the
entire expanded volume fits in the initial cube and that the cube remains valid
THROUGH the end of that 0.4-s horizon. Do not drop uncertainty, use the prior
outside its region, or ignore observed conflicts.

Immediately request positive yaw at 0.35 rad/s, with zero commanded translation.
Keep the existing sensor contracts, fusion budget and observed primitive
contradiction vetoes. On the first full-rank depth increment, command zero.
Require at least 0.3 s braking and three consecutive rank-3 observations with
measured gyro norm <=0.1 rad/s and inferred speed <=0.05 m/s before declaring
the observation turn complete. Terminate on heading magnitude >0.45 rad,
inconsistent inferred speed, unavailable sensor/fusion state, violated envelope,
expired prior or observed obstacle contradiction. Terminal output is zero and
the controller cannot restart.

The executor must record the subsequent zero-command stopping tail, retaining
any native contact, speed, support or envelope violation. Do not silently freeze
physics and call it a successful physical stop. A hard native/sensor terminal
that prevents the planned tail is a reported incomplete/failed trial, not a
successful stationary state.

## Required raw audit before interpreting the trial

Reconstruct controller decisions from saved RGB/depth/body/gyro packets and the
exact admitted priors; compare requested/applied command cadence and every
terminal reason. Verify setup hashes and native geometry identities without
assuming IDs from ordering. Reclassify raw ground contacts by geom ID so a
non-foot collision on a merged calf link is not exempted. Retain the original
non-ground/body contact guard. Check actual speed and prospective-envelope
containment, stopping tail, floor/wall depth errors and all artifacts/bindings.

Measure full acquisition/observer/controller/command wall times separately from
simulation clocks. CPU/frozen-physics execution cannot establish a 100-ms
real-time loop. Record the first full-rank frame and whether it survives braking;
no full-rank frame or any failed gate remains a negative result.

The raw auditor additionally requires the entire final 100 ms of a completed
three-tick zero tail to have actual base-speed norm <=0.05 m/s and actual angular
speed norm <=0.1 rad/s; a quiet endpoint alone is insufficient. It reports actual
27-primitive padded containment in the declared starting cube and expiry through
the final recorded sample, as well as sampled prospective translation-envelope
errors. Horizons truncated by the end of recording are explicitly labelled;
they are not claims about unrecorded future motion. The total collector wall time
includes initialization, settling, acquisition, policy/control, persistence and
cleanup. Component timers are partial partitions, not a summed deadline proof.

After a verified bounded observation turn, connect the same admission and
observation-action logic to continuous discovery/return. Preserve the full
requirements for matched geometry/supervised/JEPA arms, identical starting
conditions, invalid-start/no-prior controls, genuine multistep planning,
independent layouts/seeds/robustness, and bounded hardware evidence when available.
