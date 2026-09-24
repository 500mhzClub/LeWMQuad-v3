# Startup observation controller implemented; physical execution pending

The new `StartupObservationTurn` consumes the checked setup handoff and actual
RGBD/body-derived depth/gyro state through the explicit-prior plane memory. It
commands a bounded positive yaw immediately, brakes on the first full-rank depth
increment, and requires three quiet rank-3 frames and 0.3 s braking before its
observation-turn completion status. That status is not maze completion.

Sensor/fusion faults, the unchanged uncertainty budget, observed contradictions,
inferred speed inconsistent with the declared cap, heading limits and setup
volume/expiration all retain zero-command terminal responses. There is no
restart. Admission carries only episode/epoch, definition/check-report hashes
and check outcomes; it is not a sensor packet or a cryptographic authentication
scheme. A trusted live evaluator must bind it to the actual saved setup report.

The new geometric bound does not infer future joint motion from a short history.
Summing rigid-tree translation norms, primitive-origin offsets and primitive
circumspheres gives an all-joint-angle, all-root-orientation radius of
0.7444115632646775 m. Padding plus the explicitly assumed 0.3-m/s base-speed cap
over the 0.4-s command-plus-stop horizon adds 0.16 m, giving 0.9044115633 m before
current pose uncertainty. The initial-body cube must contain that entire envelope
and remain valid through the horizon, not merely at the current query time.
This radius bounds nominal URDF geometry, not independently calibrated native
contact discretization or hardware dynamics.

The proposed new trial uses a separate [-1,1]^3 initial-body setup cube, subject
to live validation before motion. The earlier [-1,-0.75,-0.5]..[1,0.75,0.6]
recorded-check prism cannot enclose this all-posture motion model. Neither its
dimensions nor its recorded result was altered. The larger starting condition
must be reported and shared across all comparison arms; it is not an unobserved
maze map inferred from RGB.

## Native-foot/contact preparation

The new evaluator reader captures native geometry IDs, groups, types, seven-slot
data, world centres/quaternions, friction and solver parameters. The foot matcher
requires four distinct 22-mm native spheres in the appropriate calf groups and
compares their centres against actual-pose URDF geometry. It does not guess IDs
from ordering. An additional raw-contact check rejects loaded non-foot ground
contacts even on a merged calf link; existing non-ground/body contact checks
must remain in force. No new empirical penetration allowance is introduced.

Installed source inspection confirmed that sphere data is a radius padded to
seven slots and that native geoms expose the required readback fields. The
reader has only been mock-tested so far; no new native robot has been instantiated
or observed by it. The fresh source binding must additionally cover its installed
geometry/URDF-reader/constants dependencies. A direct uninitialized Genesis
class import failed during inspection; no scene or physical run was started.

## Verification and remaining execution work

Controller-focused 40823 passed 16 tests in 4.10 s. Native-foot-focused 86728
passed 10 tests. Review then added native-capture mocks and corrected the setup
expiry check to include the full future horizon. Combined focused 5937 passed
all 29 tests in 4.18 s. Tests include actual consumer integration with synthetic
sensor streams, full-rank braking, retained rank-2 budget failure at frame 12,
wrong/missing setup admission, uncertainty/expiry/obstacle vetoes, unrestricted
joint-posture support checks, geometry identity mutations and strict contact
classification for both contact-pair orderings.
Expanded regression 83649 passed 1,664 tests across 137 explicit files in
98.33 s. No tested source was edited concurrently. All listed test handles are
terminal; no simulation or training job is running.

The fresh physical launcher, setup admission/contact wrapper and raw replay
auditor are not yet implemented. The proposed protocol is
`go2_startup_observation_turn_development_v1_2026-09-06.md`. Implement those
remaining pieces next, freeze the explicit source/input identities, and execute
the one bounded observation-turn check. Verify actual base-speed assumptions,
native foot/non-foot contacts, stop-tail motion, first full-rank frame and its
persistence through braking, controller replay, and full wall-clock accounting.
Preserve failures rather than enlarging old budgets or restarting stopped memory.

No new physics, training, full-loop timing result, calibrated contact model,
learned-navigation policy or JEPA contribution is claimed here. Whole-maze
discovery/return remains 0/2. After the local observation action works in physical
execution, integrate continuous missions and complete the matched learning,
memory, genuine multistep, independent-layout/seed/robustness and hardware work.
