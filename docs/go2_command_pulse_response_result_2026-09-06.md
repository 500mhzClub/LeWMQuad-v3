# Command-pulse characterization: complete acquisition, usable but variable responses

Executed all four fixed development episodes and all 64 pulse events, including
their two-second requested-zero braking tails. No native safety stop or visual
acquisition failure occurred. Raw audit passed. This supplies action-response
data for a bounded feedback execution baseline; it is NOT a completed local
goal-reaching task, maze navigation, JEPA contribution or deployment result.

## Actual work completed

The [frozen protocol](go2_command_pulse_response_v1_2026-09-06.md) compares
forward .12/.20m/s and yaw ±.03/±.25/±.45rad/s, each for .2s and .5s, followed by
2s of requested-zero braking. There are 16 events per episode. Two actual new
starts use forward/reversed event order, each at nominal/lower friction 1/.15
on robot and floor. The same learned gait remains underneath fixed development
excitation. No sensor or native state chooses pulse amplitude/duration; visual
input failure/excursion and native safety stops can terminate collection.

Each episode: 386 command ticks, 387 RGB-D/visual decisions, 20,050 physics
samples, 40.1s including setup. Total: 1,544 command ticks, 1,548 RGB-D captures,
80,200 physics samples and 160.4 simulated seconds. All 64 events have complete
visual and native pulse-end, +.1/.5/1/2s labels. No failed event was discarded or
replaced. Requested and slew-applied commands were retained; .45rad/s yaw onset
and stopping include the platform's .35rad/s-per-tick slew transients.

## What the responses show

The table reports net response from immediately before a BANK pulse until two
seconds after its command ends. Forward is body-X displacement in the pulse's
initial body frame; turn is absolute relative yaw. Ranges are observed values,
not calibrated bounds or confidence intervals. Forward cells contain two
events; turn cells pool both signs over both starts and contain four.

| Bank pulse | Nominal net response | Lower-friction net response |
| --- | --- | --- |
| Forward .20m/s, .2s | +0.01393 to +0.02212m | -0.00777 to +0.02241m |
| Forward .20m/s, .5s | +0.05915 to +0.07736m | +0.03752 to +0.06402m |
| Turn ±.45rad/s, .2s | 0.03989 to 0.08865rad | 0.07260 to 0.10924rad |
| Turn ±.45rad/s, .5s | 0.19530 to 0.22504rad | 0.16998 to 0.23828rad |

All 16 bank-turn events rotate in the requested direction overall, but their
effect is not constant. A nominal .5s forward bank pulse produces useful
centimetre-scale motion; that supports testing a pulse-and-observe controller,
not assuming exact command tracking or assigning a deterministic graph edge.

Low friction matters: one .2s forward-bank event ends 7.77mm BACKWARD and 32.40mm
sideways in its starting body frame. A .5s forward-bank event in the other start
ends +37.52mm forward and +39.74mm sideways. Neither body heading nor the label
"forward" is sufficient to predict progress towards a world-frame goal.

Stopping matters independently of amplitude. In nominal A, a .5s +.25rad/s turn
produces +.09680rad at command end, but +.04779rad after the braking tail. A
.5s +.45rad/s turn produces +.18714rad at command end and +.21013rad after
braking. Stopping can partly undo OR extend rotation depending on the state and
action. Requested command integration is not an adequate settled-effect model.

The earlier command-support finding must also remain qualified. Small .03rad/s
commands can produce transient rotation here: for example nominal A's .5s left
pulse reaches +.01125rad, then only +.00273rad after braking. The earlier stalled
continuous correction does not establish a universal hard cutoff. The saved
training bank mismatch is real, but state/history, pulse timing, gait dynamics
and stopping still need consideration; bank membership alone does not explain
or certify every observed response.

Every event's final 100ms meets the audited native speed/yaw-rate quiet
criterion. This does NOT prove that the entire two-second tail was quiet, nor
does it substitute for the full one-second goal-hold verification in a future
controller. Both starts and command order changed, so these are development
repeat/ordering observations, not matched-state causal interventions or a
randomized independent generalization sample.

## Verification

Raw audit PASS: all 1,548 visual-interface/schedule decisions replay exactly;
all 1,548 raw-depth diagnostics are within the existing 1mm criterion. Sensor
timestamps/history, raw contacts, native stop supervision, actual new starts,
different 750-sample prefixes from previous fitting/goal-hold recordings,
material/gain identities and requested/applied commands were checked.

Maximum visual position error per trial: nominal A 1.964mm, lower A 2.633mm,
nominal B 2.064mm, lower B 2.738mm. Maximum visual/native event displacement
difference across all reported endpoints is 1.887mm. These are observed
ideal-sensor errors, not physical sensor uncertainty bounds.

A separate read-only Hamilton-quaternion calculation reproduced all 640 native
endpoint and braking-change transforms without the audit's rotation-matrix
implementation: maximum position difference 2.78e-17m and yaw difference
1.39e-16rad. Source/input/result/audit bindings were reverified after completion.

Focused tests: 45 passed in 1.82s. Full regression: 2,348 passed across 187
explicit files in 191.69s. A prelaunch truncation-test fixture initially included
its intended missing endpoint; corrected the fixture and strengthened coverage
for interrupted pulses followed by zero-command drains before launch. No
launched source, protocol or artifact was changed. Tests overlapped collection;
physics pauses during capture/computation, so this does not establish real time.

The source implementation, collector, acquisition session and response audit
are separately named new files. No protected benchmark access, whole-tree source
export, predecessor retry, checkpoint training or hardware execution occurred.
Camera aperture/raster, controlled-floor/hidden-robot ideal-camera assumptions,
sensor calibration, timing, maze memory and JEPA comparisons remain unresolved.
Follow the [pulse-feedback execution plan](go2_pulse_feedback_execution_next_steps_2026-09-06.md).

## Frozen identities

Output `.generated/go2_command_pulse_response_v1_attempt_001`; launch binds
609 sources / 17,778 inputs plus inherited native identities.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 5fcd856e45d7580b4b9a069ca63f71afc86df18cae7b328085b93673fcbdb0de |
| result.json | f60e46072e1a33767ed221dae26e39ca382cfb96c07aa9be89023521c591dc9d |
| raw_pulse_audit_launch.json | 8e96217c2306ab55d0e45c654ef6ea4156665b776178b24d246084038dccd741 |
| raw_pulse_audit.json | af809e5cb6101bf67169e1b2a3f5e66be424b53c316d7f4e40854ee9fd5d1d46 |
| nominal_a_pulse_evaluation.json | fb9a60e9598019cd011868952c3684882778af710bdfa8b38d57f364b3a009c8 |
| lower_friction_a_pulse_evaluation.json | 48c65ece497ee9a36b525c643c76069f08c1a9a00e6bcfd33ae0b0e5ca95734e |
| nominal_b_pulse_evaluation.json | 8ac212e6111dd5e1d304f79aeeb66ddf3e8174fa00ddcc42dff51cf272216498 |
| lower_friction_b_pulse_evaluation.json | 64859472fdc7e32f04096b96db2d84773501970c79dabacb2861a5e82052a359 |
