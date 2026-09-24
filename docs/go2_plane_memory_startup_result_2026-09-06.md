# Recorded plane-memory check: startup observability is the next blocker

The new plane selector works on the bounded-floor Go2 frames, but historical
memory cannot initialize, and the current forward view does not observe the
robot's own support footprint. This changes the next implementation priority:
further plane-mask tuning cannot supply the missing starting-state information.
Complete maze discovery/return remains 0/2. No new physical mission or training
was executed, and no JEPA or memory benefit is established.

## Recorded evidence and retained failure

`scripts/probe_go2_measured_plane_obstacle_memory_development.py` was run against
the previously completed bounded-floor interface tape. Execution 57179 exited
with a sensor-contract error on frame 1, at 1.6 seconds. The original root cause
is `weak prediction requires a previously observed velocity`. Frame 0 produced
an observed plane and exact reference/compiled outputs, but no floor coverage,
conditional clearance or contact candidate. Its old foot-query-dependent
consumer had no seed; the new observation-bound consumer had one.

The probe source and failure remain preserved. A separate diagnostic,
`scripts/diagnose_go2_plane_memory_startup_development.py`, reproduced the same
historical-memory fault, left it latched, and evaluated only same-frame
q/depth geometry thereafter. It neither restarts memory nor commands the robot.
Execution 67266 completed with:

- 26/26 measured frames containing a plane hypothesis.
- 0/26 frames providing any complete current robot-primitive floor footprint.
- All 104 instantaneous foot centres behind their corresponding optical origin;
  the greatest optical depth is -0.1046249574 m. Valid measured depth starts at
  +0.2 m. These are measured-q/URDF/extrinsic calculations, not evaluator pose.
- All 25 inter-frame depth registrations have rank 2. The first missing
  translation direction is approximately (-0.0055, -0.9994, -0.0331) in the
  previous body frame. No fully observed velocity ever becomes available.
- The initial-gravity hypothesis is taken from the same quiet specific-force
  history and propagated with the recorded causal gyro orientation. Its error
  allowances are still conditional, not calibrated hardware bounds.

Plane separation/contact-straddling without measured footprint coverage is
reported explicitly as incomplete evidence. A foot gap straddling zero is not
permission for contact. The reported local diagnostics after frame 1 are not a
successful historical reconstruction, post-stop controller execution, or proof
of safety. They do not evaluate old camera coverage at later robot poses because
that would require the historical translation that failed to initialize.

The diagnostic verified all 350 inherited launched-source bindings, inherited
inputs, bounded-run artifacts, three exact launch/result/reader identities,
reader sources, and 14 explicitly named development source/test identities
before and after execution. This is not a recursive source-closure claim.
No sealed material was accessed. Four new diagnostic tests passed (75220).
Expanded regression 42668 passed 1,570 tests across 133 explicitly named files
in 95.88 seconds, with no concurrent edits to tested source. All execution and
test handles from this turn are terminal. These tests do not override the
historical-memory initialization failure or qualify complete navigation.

## Scientific consequence

Depth point-to-plane registration constrains translation normal to observed
surfaces, not every tangential component. Acceleration cannot supply an unknown
initial velocity just because the commanded velocity is zero. Quiet IMU history
also does not by itself prove support contact or collision-free surrounding
volume. The existing fail-closed initialization is therefore exposing missing
information, not a reason to relabel the weak direction as observed.

Visual inspection of the first RGB frame shows broad nearly uniform surfaces
and predominantly horizontal boundaries. RGB tracking should be tested for
additional observability, but its availability cannot be assumed from the
presence of an RGB camera. A learned prediction is likewise not a measurement
of an unobservable starting condition.

The present consumer also demands a complete floor footprint for every physical
primitive. That is a conservative implementation prerequisite, not a necessary
condition for all physically valid locomotion (for example, non-foot parts can
be above an unsupported region). Contact-bearing support, non-foot collision
clearance and future gait envelopes need separate contracts. Do not erase an
observed obstacle merely to relax this conservative prerequisite.

## Next implementation toward complete missions

1. Define a development-only starting-state contract: explicitly supplied
   bounded initial velocity, supported posture and local collision-free starting
   region. Choose the region, tolerances and validity duration before a new run;
   validate them using setup/evaluator evidence rather than feeding ongoing
   world pose or contact labels into the policy. Keep prior/setup evidence
   distinct from sensor-observed free space. This is a proposed experimental
   condition, not a claim that the current packets establish it.
2. Give both geometry and learned baselines exactly the same starting condition.
   Add rejection tests for an invalid start, expired/out-of-region prior, missing
   support and sensor failures. A geometry-only/no-prior variant must retain the
   startup failure. If a suitable starting contract is unavailable, supply and
   validate deployment-valid contact/velocity sensing instead; do not silently
   initialize unknown velocity or unseen floor as certain.
3. Preserve same-frame local evidence when historical pose is unavailable, while
   disallowing unsupported historical transport. Integrate modelled foot contact,
   prospective commanded motion, stopping dynamics and sensing-to-command latency
   into one new controller. Avoid further isolated floor-mask optimization as a
   substitute for completing this integration.
4. Execute fresh complete discovery/return missions. Then run matched predictive
   training, online memory and genuine multistep planning comparisons with
   independent layouts, model seeds and robustness conditions. Test startup
   dependence explicitly. Hardware evidence remains a separate requirement when
   a platform is available.

The full scientific goal remains active and unachieved.
