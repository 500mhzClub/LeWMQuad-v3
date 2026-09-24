# Ideal foot-load stream reconstructed and audited; support remains conditional

The fitting-only reconstruction completed all 30,000 recorded 500-Hz samples.
It creates an explicitly NEW hypothetical three-axis net contact-force sensor
at each foot, with all incident contact sides included. It does not retrofit
the original controller or turn native ground labels into sensor observations.
No new physics, learned model, validation trial or hardware access occurred.

The separate raw-hardware transport type preserves signed16 counts, reported
channel ordering, device identity, timestamps, missingness and saturation.
It neither converts counts to Newtons nor allows them into the ideal-force
consumer. No hardware sample was available to exercise that transport.

## Recorded results

At startup sample749, all four feet exceed the preregistered descriptive 5-N
resultant threshold and have completed its 20-ms dwell:

| Canonical simulated foot | Startup resultant, simulated N | Peak across tape, simulated N |
| --- | ---: | ---: |
| FL | 39.257 | 343.884 |
| FR | 51.159 | 730.107 |
| RL | 44.157 | 605.400 |
| RR | 14.522 | 312.476 |

There are 12,196 samples with four feet above threshold, 14,812 with three,
2,947 with two, two with one, and 43 with none. Four feet meet the dwell rule
in 9,769 samples. These include settling and dynamic motion, not just quiet
stance; peak forces are simulator outputs, not calibrated physical loads.

Evaluator labels, loaded only after the sensor/prediction file was saved,
identify 104,125 nonzero foot-contact sides, all on ground. There are no recorded
non-ground or self-contact positive loads in this tape. Therefore it supplies
NO empirical specificity test for wall/self-contact, uneven supports or slip.
Synthetic tests verify that such loads are retained and that indistinguishable
force histories cannot establish ground or exclude slip; they are not physical
robustness demonstrations.

## Independent numerical audit

An independent contact-side loop sums forces in extended precision. A separate
root-quaternion implementation then rotates each saved foot-frame vector back
to world coordinates. All 120,000 vectors agree within the fixed 1e-8-N
numerical tolerance; maximum coordinate discrepancy is 3.262e-13 simulated N.
The audit includes 104,137 incident sides (12 have zero force). It accesses no
terrain-role labels. URDF joint kinematics remain shared with the producer.
All threshold statuses and 119,960 direct dwell windows match the saved
predictions exactly. This validates bookkeeping, not force calibration.

Focused tests: 25 passed, covering both contact sides, foot-foot loads,
non-foot identity relabeling, frame transformations, cancellation, missing versus
empty acquisition, malformed data, raw counts, stale/saturated/error-unknown
samples, timing, dwell gaps and offset ambiguity. Full regression passed all
2,232 tests across178 explicit files in181.32s. No tested source changed during
the run. Neither regression nor the numerical audit qualifies physical support.

## What changes next

The recorded startup now has a candidate local-load modality that does not
depend on seeing the floor beneath the robot. It is still NOT ground support:
resultant magnitude loses direction, location, moments and opposing-contact
information. Neither four loads nor their persistence proves a plane or prevents
slip. No existing clearance guard was removed or relaxed.

Follow the [support-kinematic integration plan](go2_support_kinematic_integration_next_steps_2026-09-06.md).
The deployment gap must remain explicit: the ideal three-axis transducer is
not equivalent to vendor scalar fields. The existing tape has q/dq, IMU and
contacts but NO recorded joint-torque measurement; do not infer a torque-sensor
validation from commanded actions or contact resultants. A real implementation
needs verified/calibrated available channels or explicitly added instrumentation.

The unresolved startup-camera aperture/clipping and two ray discrepancies also
remain. No new view is qualified. The full short-execution, online-memory,
JEPA versus matched baselines, genuine multistep, independent-maze/seed,
real-time and bounded-hardware scientific requirements are unchanged.

## Identities

Output: `.generated/go2_ideal_foot_load_reconstruction_development_v1_attempt_001`.
Launch binds 550 source paths and 11,344 inputs plus inherited native identities;
the force audit has a separate source/input manifest. No whole-tree export.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 8428c077fa4b4dec8c6192f65ff9e26e727c141a5b4be39473555141516d4535 |
| result.json | c0f2de76355e43a231492743feb26e182773ea57baa90e85fb55a340f43a1717 |
| sensor_predictions.npz | 85afbbe21241259a50ed532c6dffa9569620c4aeb9eb8eead9d626546b6bd908 |
| force_conservation_audit_launch.json | fb93465a8080311f63300a38df495ac07865df725e43b52f54a3047bccd0b49e |
| force_conservation_audit.json | 9e6056846a72aed538a1172428fde239abc4f7bd976391b3c34fee018403a87d |
