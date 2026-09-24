# Live high-rate gyro: better orientation, unchanged local success count

The eight-trial paired study is **COMPLETE**, and its full raw audit is **PASS**.
A separate live500 Hz virtual-gyro channel substantially improves scan orientation
accuracy while preserving the existing learned RGB/body interface. It does not
solve articulated wall clearance: both rates succeed on3/4 fixtures and contact
the0.9 m dead-end wall. No physical criterion was relaxed or outcome replaced.

## Fixed paired physical results

All fixtures start at heading0, unlike the preceding ±0.15 rad assay. The
dead-end/cross ×0.9/1.2 m fixtures each receive both sensor-rate arms with
identical settling physics and ordinary sensor histories. Both arms acquire
the fast stream; only the500 Hz arm consumes it for orientation. The gait,
scan decision rule,50 Hz ground estimator, RGB observer and release criteria
remain identical. No learned model is trained, selected or changed.

| Fixture |50 Hz full success|500 Hz full success|50 Hz contact|500 Hz contact|
|---|---:|---:|---:|---:|
|Dead end,0.9 m|No|No|Yes|Yes|
|Dead end,1.2 m|Yes|Yes|No|No|
|Cross,0.9 m|Yes|Yes|No|No|
|Cross,1.2 m|Yes|Yes|No|No|

The three completed physical trajectories are identical within each rate arm
despite different rendered geometries. There are four exact raw-array trajectory
groups overall: two contact trajectories and two completed trajectories. These
are deterministic paired fixtures, not eight independent stochastic dynamics
samples. Do not derive generalization confidence intervals from repeated traces.

For each of the completed fixtures:

| Endpoint |50 Hz|500 Hz|
|---|---:|---:|
|Maximum estimated-heading discrepancy versus true relative pose|0.038510 rad|0.000452 rad|
|Final true heading error after release|0.115461 rad|0.079055 rad|
|Maximum base XY drift|0.083249 m|0.082812 m|
|Scan + release duration|23.0 s|22.5 s|

This is approximately85-fold lower maximum estimated-heading error on these
executed trajectories, not an85-fold navigation improvement. Both final errors
already satisfy the fixed0.12 rad criterion at this new initial heading, so the
paired full-success difference is zero. The previous failed scans remain failed;
the different heading changes the population and prevents a before/after success
claim. At500 Hz the final error is dominated by the deliberately finite control
tolerance, not large integrated heading drift.

Both narrow dead ends stop on native disallowed contact. Contact occurs6.212 s
after settling at50 Hz and5.998 s at500 Hz; the better estimate does not prevent
it. Maximum base drifts are0.065235/0.064531 m. Sensor-only turning accuracy is
therefore useful but insufficient for safe articulated motion beside a wall.

## Observation and sensor evidence

Each arm covers10/10 actual opening sides at selected views, versus4/10 in the
initial views. Each produces13 selected-view proposals with no closed-side
proposals. These are opening-directed bearings under the existing nominal
floor/palette assumptions, not certified clear paths or graph edges. No sensor
fault occurs in either arm. Hardware noise, calibration and latency remain open.

The new runtime channel contains51 co-timed body-gyro samples spanning100 ms
with2-ms cadence, validity and explicit availability times. It rejects missing,
stale, rewritten, reset-mismatched or extra privileged inputs; all six overlapping
50 Hz measurements must agree. Midpoint integration consumes each of50 new
intervals once. The old strict RGB/body packet has no new field, and its learned
tensor layout is unchanged.

The simulator-side sensor adapter acquires each measurement during its actual
physical step. It necessarily uses native state to model an ideal body gyro,
but native orientation/world angular velocity are not passed to the controller.
This differs from the earlier retrospective500 Hz diagnostic: the new measured
stream actually controlled fresh commands causally. It is still an ideal
virtual sensor, not proof of real-Go2500 Hz sensing or hardware performance.

## Verification and identity

The full audit reconstructs all80,355 live fast-gyro measurements independently
from their co-timed native states, rebuilds every fast history, and checks the
unchanged8,034 ordinary body-sensor samples and1,496 actual RGB packets. It
verifies camera mounts, gains, applied-command slew, paired settling and immediate
native termination, then exactly reproduces all1,464 live scan/ground/proposal
decisions and recomputes all physical/observation outcomes.

All166 bound source/test/protocol paths and147 predecessor input bindings plus
the gait remain unchanged. The focused development suite passes782 tests in67
explicit files, including19 new high-rate channel/session tests. Test success
does not establish navigation, safety certification or real-platform transfer.

Exact root: `.generated/go2_fast_gyro_scan_development_v1_attempt_001`.

- Launch: `f0efc9cef7a4d06f3207394fc4851f997ee379ddb18f1fa0a133b35f25c1f3ec`.
- Result: `d298f8ad1dc4c4098f7ae45a6e2025b361f356503b29dc0e45252c215eace3de`.
- Full audit: `9001382cfb134436b2563cdbffedecf82349ac6e462076e616e844808bcf8289`.

## Next action toward the actual maze task

Keep this separately scoped high-rate orientation interface for further
development; no hardware qualification or alteration of frozen model inputs.
Next implement current articulated collision-geometry support bounds from
ordered joint sensing and the bound URDF, then distinguish that instantaneous
outline from uncertain future gait sweep. Include fixed-child calf collision
shapes when resolving native contact links. Test the geometry against independent
forward kinematics and actual contact locations, without using oracle walls for
runtime decisions.

In parallel with that implementation—not another model sweep—prepare a minimal
current-RGB-proposal to physical-traversal to observation-supported-arrival
prototype. A declared wider development maze may test integration, while the
narrow-wall failure stays an explicit unsolved requirement. Follow the
[sensor-to-navigation plan](go2_scan_to_navigation_next_steps_2026-09-05.md).
