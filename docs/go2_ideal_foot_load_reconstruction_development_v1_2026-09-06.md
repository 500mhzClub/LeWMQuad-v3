# Ideal foot-load reconstruction V1 — fitting-only, offline

Reconstruct all 30,000 fitting samples from the existing longer-motion tape,
including settling, in order. This is a NEW hypothetical sensor stream, not a
claim that the original controller had foot loads. No new physics or training.
Do not use the already exposed validation trial to select this implementation.

Verify the four 22-mm native foot spheres against their actual startup-pose
URDF positions. For every raw contact side incident on those geometries, sum
the recorded force ON that side. Include self-contact, wall and other-object
loads without filtering by the other geometry's identity. Rotate the resultant
into each URDF foot frame using acquisition-side native orientation and measured
joints. Output no native world pose, geometry ID, contact position/count, ground
label, contact Boolean or support permission. This is an ideal THREE-AXIS net
contact-force transducer hypothesis, NOT a vendor foot_force conversion.

Explicit empty valid acquisition means zero resultant in this ideal model;
missing acquisition means unavailable, never zero. Opposing contacts can cancel.
No moment measurement, sensor inertia, noise, bias, finite bandwidth, delay,
hardware calibration or mounting equivalence is established. Recorded samples
receive hypothetical zero-latency acquisition timestamps; no real latency claim.

Freeze the descriptive local-load model before reconstruction: magnitude >5 N,
20-ms dwell, maximum sample age and gap 10 ms, explicitly zero conditional force
error for the ideal diagnostic only. Unknown error, stale/missing data and
saturation cannot establish loading. These engineering thresholds are not
learned, fitted, vendor calibrated or safety-qualified. Load does not prove
upward support, contact with ground, absence of slip, compatible foot heights,
continuous floor, future footholds or clear body/leg/braking sweep.

Keep raw hardware signed16 counts in a different type, preserving reported
channel order, validity/saturation, device identity and timing. No automatic
canonical ordering, units conversion, calibration or ideal-model consumption.

Persist the complete sensor/prediction arrays BEFORE loading terrain role labels
for evaluator-only breakdown of incident ground versus other loads. Audit the
saved forces with an independent contact-side loop and compare squared norms
under sensor-frame rotation; sensor reconstruction should be invariant to
relabeling all non-foot geometries. Include synthetic wall/self-contact,
cancellation, missing/stale/saturated values, offset uncertainty, clock/order
faults, and mounting-axis errors. Slip/uneven supports must remain unresolved
when the force history alone cannot distinguish them, not receive fake tests.

No retry or overwrite of this named attempt. Retain failures. Next integrate
load observations with causal IMU/joint kinematics for explicitly conditional
current-support consistency and slip diagnostics, not a flat-floor prior.
Validate proposed foot landings, body/leg motion and braking separately before
sensor-only short execution, memory and full matched JEPA maze experiments.
