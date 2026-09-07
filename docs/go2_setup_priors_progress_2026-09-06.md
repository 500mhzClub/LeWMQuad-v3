# Explicit setup priors: implemented, propagated, and conditionally diagnosed

The new setup-velocity adapter gets past the recorded frame-1 initialization
failure without labelling an unobserved velocity as measured. Historical memory
then stops at frame 12 on its unchanged combined-scale budget. This is conditional
software/recorded-data progress, not a new successful physical mission.

## Implemented contracts

`lewm/setup_velocity_prior_development.py` defines an immutable episode/epoch-
bound initial velocity mean, positive Euclidean radius and setup-definition
reference hash. There is no default prior or command-derived zero velocity.
The hash is provenance, not independent verification of the setup condition.

An initial-velocity error e propagates as position error X e and velocity error
V e. For a measured weak-space projector P in the initial-body frame, each
100-ms interval uses V' = P V and X' = X + 0.1 P V. A full-rank observed
displacement sets the velocity factor to zero but preserves the accumulated
position factor. Spectral norms give the corresponding conditional ball radii,
with a separately declared numerical allowance. These factors condition on the
recorded depth/gyro subspaces; they do not bound errors in those subspaces or
acceleration, bias, calibration or contact.

The adapter retains all inherited sensor/history validation and depth rank/status.
It adds the prior position radius to the existing uncalibrated proxy, reports both
components and their composition, and uses the same 0.08-m stopping budget. The
new ray/plane-memory adapter actually stores and transports the combined scale.
No original frozen integrator, memory, experiment source or result was edited.

`lewm/setup_region_prior_development.py` defines a finite initial-body-frame
non-floor-clear prism with an explicit expiration time. Query boxes must include
pose/geometry/motion uncertainty and remain strictly within its bounds. Any
observed conflict overrides the setup claim. Wrong episodes, malformed inputs,
negative errors and nonfinite expansions are rejected. Expired/outside queries
cannot use the prior. Results remain labelled setup-only, never measured free
space; supporting ground, foot contact and future gait are not established.
This region contract is not yet injected into a navigation controller.

## Recorded conditional diagnostic

Protocol: `go2_setup_velocity_prior_recorded_diagnostic_2026-09-06.md`.
Execution 4801 completed the fixed read-only diagnostic using zero mean and
0.02 m/s initial velocity radius on the previous 26-frame bounded-floor Go2 tape.
The protocol explicitly acknowledges that this proposal follows that recording;
it is not an independent preregistered physical validation of the selected bound.
No prior or stopping threshold was adjusted after seeing these results.

The evaluator comparison, performed after the policy-side computations, gives
initial-body velocity (-0.00068555, 0.00155355, 0.00008827) m/s, norm
0.001700376 m/s, inside the proposed ball. These evaluator values were not inputs
to the integrator or memory. This one case does not validate the setup procedure
across runs or establish deployment sensor bounds.

Frames 0 through 11 are accepted by the new historical memory. At frame 12
(2.7 s; 1.2 s after its anchor), the combined scale reaches 85.420191 mm:
61.420192 mm inherited proxy plus 23.999999 mm conditional prior radius. The
80-mm budget stop latches. The corresponding evaluator position error is
0.259123 mm, but a small error in this case does not justify relaxing the budget.

A separate offline integrator reports the remainder, explicitly post-stop and
not used for control. At frame 25, its position error is 8.089732 mm and combined
scale 275.124927 mm. There are no proxy exceedances on these 26 nominal frames.
All 25 depth motion increments remain rank 2. Original no-prior frame-1 failure,
unknown own-body floor coverage, missing contact permission and 0/2 whole-maze
results remain unchanged. No starting-region prior was injected.

The diagnostic checked all inherited source/input/artifact/reader identities and
18 explicit development source/test/protocol snapshots before and after. It
changed no recorded artifacts, accessed no sealed material, and ran no physics
or training.

## Verification and next action

Initial focused run 5510 passed 25 tests and failed one strict scalar-type fixture
case. The fixture now constructs immutable Python numeric tuples explicitly;
the input contract was not weakened. Expanded focused run 97250 passed 32 tests
in 2.13 s. Tests compare distinct prior-mean replays under rotating/switched
subspaces, independent sampled ball transport, preservation of position error
after full-rank recovery, fault latching, actual memory scale consumption,
region expiration/bounds/conflict vetoes, and absence of invented floor evidence.
Full regression 76414 passed 1,602 tests across 134 explicit files in 95.92 s.
No tested source was edited concurrently with that run. Diagnostic 4801 and all
listed test handles are terminal; no simulation or training job remains active.

Next implement independent setup validation and an explicit ground/contact
model alongside prospective commanded-motion and stopping envelopes. A region
reference alone is not evidence of clear volume or support. Keep the starting
condition identical across geometry, supervised and JEPA arms, and retain
invalid-start/no-prior controls. The 1.2-second accepted interval also exposes
the need for timely information-gathering actions or additional motion evidence;
do not obtain longer operation by silently removing prior or sensor uncertainty.

Then integrate the complete sensing-to-command path and execute fresh whole
discovery/return missions, followed by matched predictive training, memory and
genuine multistep-rollout comparisons on independent layouts/seeds/robustness
conditions. The full scientific goal remains active and unachieved.
