# From actual opening observations to an end-to-end development prototype

This supersedes the immediate next actions in earlier chronological plans,
not their frozen experiments. The ultimate aim remains RGB plus available
body-sensor JEPA exploration, hidden-beacon discovery and remembered return
on a quadruped in novel mazes. BEV remains outside the work.

We now have actual sensor-generated opening proposals:39/40 sides at selected
views and40/40 across all control frames in the fixed sixteen-scene assay.
We do not have a qualified scan or traversal: four calf-wall contacts and twelve
final-heading misses yield0/16 full scan successes. The numerical diagnostic
identifies50 Hz sampling as the principal measured integration limitation;
an analytical coning correction is insufficient. These results motivate the
following short dependency chain, rather than more model capacity.

Latest implementation: the [eight-trial live50/500 Hz comparison and full audit](go2_fast_gyro_scan_development_v1_result_2026-09-05.md)
are complete. The separate high-rate channel now controls fresh actual scans;
both arms succeed3/4 at heading0, with greatly improved high-rate orientation
but unchanged narrow-wall contacts. Stage1 below is implemented and tested for
the ideal simulated500 Hz channel, not hardware or preintegrated transport.
The [articulated geometry kernel and corrected24-scan analysis](go2_articulated_scan_geometry_development_v2_result_2026-09-05.md)
also complete all4,504 current postures. Native feet/head shapes are merged despite
URDF retention hints; the resolver uses actual robot topology. Current extent
is available, but future sweep/clearance remains unqualified. Proceed to the
observation-to-arrival prototype; preserve narrow-wall and ground-support identity
limits rather than presenting the geometry kernel as a safe-turn shield.

Stage3 now has a [completed twenty-trial panel and full audit](go2_observed_traversal_development_v1_result_2026-09-05.md).
Each learned arm succeeds4/4 with identical all-forward choices and physics;
the directional baseline has three premature candidates and one missed arrival.
All20 trials are contact/sensor-fault free. Full raw replay covers1,399 decisions;
828 tests passed before launch. This demonstrates the first local integration,
not trustworthy arrival/place association or a JEPA-specific benefit. No trusted
edge was created. Sources/results remain bound; the next intervention is actual
observed branch/second-traversal continuation with a speed/cadence-matched
non-learned control, rather than another isolated straight-line/model sweep.

That continuation panel is now [COMPLETE with corrected full raw audit](go2_observed_continuation_development_v1_result_2026-09-05.md):
all four methods, including matched fixed-forward, achieve2/4 two-leg tasks. The
remaining fixtures contact a south wall during scanning after successful first
crossings/releases. There are only three distinct raw trajectories; no JEPA or
learning advantage is demonstrated. The next integrated intervention should
address observed heading and arrival pose before scanning, with a tolerance
appropriate to the small initial bearing error—not a centered-only turn assay.
Keep uncertain memory usable for bounded development, evaluate its association
errors, and add actual beacon acquisition/return. Do not demand perfect identity
certification as a prerequisite for testing an explicitly uncertain policy.
Any learned scan/repositioning comparison needs support for those command families;
they are currently outside the frozen five-action learned bank. All sources and
physical outcomes of the completed panel remain bound.

The [initial-heading successor and full audit](go2_initially_aligned_continuation_development_v1_result_2026-09-05.md)
complete16 trials and4,652 decisions;876 focused tests pass across79 files.
Every method succeeds2/4 and times out before traversal2/4, with no contact.
Error enters the0.02 rad band but zero-command drift repeatedly breaks the
0.3 s dwell. All learned arms remain identical. Preserve this negative result:
next test bounded feedback through the dwell and post-release pose under the
unchanged endpoint, including synthetic persistent-drift and both-sign cases.
Keep subsequent centering/clearance and uncertain-memory/beacon integration
in scope; a more permissive timeout criterion is not a navigation solution.

The [persistent-feedback successor and full audit](go2_persistent_alignment_continuation_development_v1_result_2026-09-05.md)
now complete16 trials and4,600 decisions;893 tests pass across80 files. Initial
alignment passes everywhere, but two-leg success remains2/4 for every method.
Negative corner: a fresh eligible side proposal appears1.04 s before continued
scanning contacts the south wall. Negative tee: physical arrival succeeds but
the three moving novelty frames fail; all five actual stopped release frames
pass novelty, with quietness only at the last (not enough dwell to rescue it).
Next factor two acquisition-policy changes: bounded stop-to-observe at the
translation cap, and terminating scan once a fresh branch is observed. Keep
arrival evidence and physical endpoints, explicitly report partial scans, and
test individual/joint effects with matched fixed-forward before learned/task
claims. Then proceed to the uncertain-memory/beacon prototype, not another
isolated heading-tolerance or model-capacity sweep.

That [28-trial acquisition factorial and full audit](go2_task_acquisition_continuation_development_v1_result_2026-09-05.md)
now complete6,502 decisions, with916 tests passing across81 files. Every arm
still succeeds2/4: early scanning avoids the negative-corner contact but leaves
a false second arrival; stopped evidence recovers the tee arrival then reaches
an unsafe scan. Successful fixed-forward tasks shorten45.4→23.2 s. All learned
arms remain identical. Proceed to the
[whole-task hypothesis-memory/beacon-return stage](go2_whole_task_hypothesis_memory_next_steps_2026-09-05.md)
while retaining failed narrow-maze evidence and improving observed portal/clearance
state. Do not keep tuning distance margins or require perfect place certification
before testing an honestly uncertain whole-task prototype.

## 1. Make the orientation interface match the physical task

Implement a separate timestamped high-rate body-gyro/preintegrated-rotation
channel. Preserve the existing50 Hz learned RGB/body tensors and their frozen
models; an additional control-estimation channel must not silently change a
matched model comparison. The source must be actual current virtual-IMU samples,
not direct reads of base orientation, diagnostic world angular velocity, future
physics or interpolation presented as new measurements.

First support the explicitly simulated500 Hz stream, with a configurable future
hardware adapter whose actual rate and filtering are measured, not assumed.
Specify measurement/availability clocks, body frame, maximum age, startup,
overlap identity, lost samples, reset and bias assumptions. Causal preintegration
can reduce bandwidth but must retain full evidence and fail on missing intervals.
Keep a50 Hz midpoint baseline and do not claim that the simulator-matched
right-endpoint rule is optimal for hardware.

Completion gate: synthetic fault tests, independent reconstruction from logged
simulation measurements, exact replay at control timestamps, and a fresh bounded
physical comparison using the unchanged heading/release endpoints. All outcomes
count. No old scan is reclassified by substituting a better replay estimate.

## 2. Separate seeing from fitting through space

Native contacts involve articulated calf links, not merely base-center drift.
Add a robot-geometry-aware swept-volume development baseline using measured
joint history and explicit future gait/posture uncertainty. Check link geometry
against the installed URDF/collision shapes and native contact locations.
A current kinematic outline alone is not a guarantee about a future stepping leg.

The current floor rule remains a renderer-palette shortcut. Add controlled
appearance variation and an appearance-robust floor/obstacle observation arm
before generalization claims; keep geometry supervision training-only where used.
Maintain unseen near-field regions as unknown. Any decision that needs clearance
there must abstain or acquire further evidence. Treat repositioning to an
observed larger region, a modified turning gait, and wider/additional RGB camera
coverage as distinct interventions; do not use known simulator walls to choose
the runtime maneuver. Optional depth/range sensing is a legitimate separate
sensor arm if it is genuinely deployable, not privileged geometry disguised as
a sensor. Compare its extra sensing cost explicitly.

For the first integration prototype, a declared wider-maze development domain
is acceptable. It demonstrates integration only and does not solve the failed
0.9 m dead-end/corner task. Preserve that challenge for later evaluation rather
than quietly dropping its failures.

## 3. Execute one observation-to-arrival transition

Connect a fresh RGB opening candidate to a bounded local traversal. The existing
JEPA/controller bridge can consume its body-relative bearing, but a stale scan
bearing must be reobserved or transported with stated uncertainty. Execute only
known short command prefixes; future feedback-dependent realized commands may
not be supplied to a current prediction.

At the destination, use new RGB/body observations to propose a local place and
arrival—not the known cell index or a fixed elapsed-time assertion. Keep place
association, execution success and traversability as separate hypotheses. Record
when an association is ambiguous, when the robot has not settled, and when
crossing evidence is unavailable. Runtime memory may keep provisional hypotheses;
only supported directed traversals become trusted edges. Never create a reverse
edge automatically, or interpret candidate type conversion as certification.

Completion gate: a fixed fresh small physical panel with all attempts retained,
evaluation-only true crossings/arrival states, false merges, missed arrivals,
native contacts and progress/stalls. An honestly labeled development prototype
can precede final perception qualification; it must not report hypotheses as truth.

## 4. Run a minimal whole-task prototype before another large training study

Construct fresh small connected development mazes with at least a branch,
an initially unseen physical beacon and a return requirement. Start with
observable environments, then explicitly add repeated-looking junctions and
loop closures. Discover beacon identity from actual RGB or another declared
sensor, never from simulator coordinates, a hidden map or an oracle goal image.
Initial centering is allowed only as a declared initialization condition.

Build memory online. Evaluate discovery count, wrong place merges, return success,
path length, contacts, interventions, progress stalls and time budget together.
A policy that stops safely but discovers nothing fails the task. Compare the
same local execution with and without persistent memory before interpreting
memory benefits. Fix an exact modest scene/seed/budget population before launch;
every inspected scene remains development material.

## 5. Ask the JEPA question on actual task outcomes

Use matched direct, supervised-rollout and JEPA arms with identical available
sensors, data, optimization exposure, action candidates, execution safeguards
and memory. Keep training seeds and maze identities as their real experimental
units. The completed18-model comparison does not establish a JEPA advantage;
its short switched-action contact endpoint has no positive hazards. The earlier
half-second online benefit remains a bounded result, not a contradiction to hide.

Distinguish benefits from latent prediction, learned direct heads, sensor changes,
memory and increased training coverage. Keep kinematic/action-mean and stop
references; stop is a safety floor, not an exploration solution. Only develop
longer latent planning if it improves executed progress/risk under a valid future
control contract. Larger models, new losses and more data require measured
limitations and separate interventions.

Final independent layout/appearance/dynamics tests and a supervised real-Go2
pilot follow development freezing. Hardware requires actual access, calibrated
sensors, watchdogs and human safety supervision. Protected legacy benchmarks
remain inaccessible. A credible negative JEPA comparison is scientifically
complete evidence; the overall navigation/transfer goal is still unachieved.
