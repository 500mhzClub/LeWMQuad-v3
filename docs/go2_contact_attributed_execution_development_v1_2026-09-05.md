# Fresh Go2 contact-attributed execution development V1

Specified before collection, 5 September 2026. Development only; not a restart
or integrity replacement of the interrupted stratified generator. No sealed
material, previous candidate payload, trained navigation model or JEPA is used.

Question: with correct per-contact measurement, can the existing Go2 gait and
oracle pure-pursuit teacher traverse fresh physical openings and stop in a
usable arrival state? This distinguishes execution feasibility from visual
decision-making. It does not establish RGB navigation or a JEPA contribution.

Panel: exactly eight cases, ordered straight, offset, left90, right90, each at
widths 0.75 and 1.0 m. Case identities and seeds 2026090500–2026090507 are new.
The source room is x=[-1,0.6], y=[-1.2,1.2] in local coordinates; a 1.2 m long
exit corridor starts at x=0.6. Offset shifts the opening centre to y=0.25.
Turn cases rotate the room/corridor by +/-90 degrees while spawn yaw stays zero.
Thus these are turning approaches from a roomy source, not a complete narrow
L-maze benchmark. Wall thickness/height are 0.08/0.6 m. No outcome-based candidate
rejection, extra seeds or best-case selection is allowed within this study.

Reuse the platform's existing CPU gait checkpoint and command limits with hash
validation, 2 ms physics, 20 ms policy and 100 ms command ticks. Record 1.5 s
zero-command settling, then the existing teacher (at most 8 s), then 0.5 s of
zero-command braking. No snapshot or branch replay is required. Stop a case at
the first measured disallowed contact, body height below 0.15 m, or absolute
roll/pitch above 0.70 rad; retain its terminating sample and count the failure.
Nonfinite or malformed measurement is an integrity failure, not a learned-model
negative. Stop the study on an infrastructure/integrity error and preserve its
partial output. Do not automatically rerun an incomplete case.

Persist every physics sample's requested/applied commands, base pose/twist,
joint position/velocity, phase and contact flag. Preserve native contact vectors,
validity masks and link IDs with per-frame offsets, plus explicit link/object
mapping and first-contact attribution. Per-contact force must be available and
finite; use the existing force floor and support/ground ontology. Settling
contacts are not silently excluded. Normal foot/calf support contact remains
allowed under that declared ontology, not a new real-robot safety claim.

Capture initial and final native RGB from the fixed platform camera mount;
do not move the camera using oracle obstacle geometry. RGB is diagnostic only
and never drives this teacher. Persist a right-handed optical camera transform.
Simulated world pose is explicitly privileged teacher/evaluation information,
not a proposed deployed JEPA input. Hardware calibration remains unverified.

Primary endpoint per case: a positive-normal selected-opening crossing sustained
for the existing 100-sample dwell rule, no disallowed contact throughout recorded
execution, completed braking, final base >=0.02 m beyond the port, lateral base
margin >=0.10 m, heading error <=0.35 rad, planar speed <=0.10 m/s, absolute world
z angular velocity <=0.25 rad/s, base height >=0.20 m, and absolute roll/pitch
<=0.50 rad. These are prospective development targets, not hardware-certified
limits. Arrival checks are proxies for usability; actual consecutive-edge
execution remains required. Report every failed check, not only binary success.
The endpoint requires the full 100 samples beyond the port even if the inherited
teacher's own stopping rule accepts early target-region entry. Initial/final
image times refer to the latest recorded physics sample, including when a stop
interrupts a policy period; the policy clock can lag that sample.

Fix source/config/gait bindings and all case specs in a fresh launch record
before stepping. Run each case once, preserve raw artifacts and failures, and
report the eight outcomes without a population-level confidence claim. If this
oracle fails, localize the limiting measurement/control/geometry before changing
JEPA. If it succeeds, test consecutive traversals and matched deployable local
selection. Neither outcome completes the final novel-maze research objective.
