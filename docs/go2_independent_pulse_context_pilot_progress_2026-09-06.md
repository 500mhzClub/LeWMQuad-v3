# Independent pulse-context pilot: execution and evidence ledger

Status: all12collection episodes and the separate corrected raw audit completed,
with the original auditor's command-encoding error preserved. The full
scientific navigation goal remains active and unachieved. This document is a
progress ledger, not a replacement for the frozen pilot protocol.

## Implemented and checked

The new collector constructs one connected eight-cell/eight-passage layout with
a loop, branches and dead ends. Twelve training-only episodes cross all six
existing pulse-duration cells with nominal and low-friction support. A common
eight-tick recent-forward history precedes each candidate pulse and twenty brake
ticks. Only causal packet validity gates this fixed simulation schedule; visual
tracking is a shadow diagnostic. Native contact/stability and additional
nonfoot-ground, speed and domain stops remain active. This is not hardware
authorization or a learned navigation controller.

The auditor independently reconstructs available sensor/contact streams, setup
geometry, decisions, applied-command slew and the first native stop. It includes
partial settling/zero-frame cases, without fabricated setup or training windows.
It compares all1,150 native prefix samples, native contact fields and nine actual
RGB/body/control/depth/fast-gyro packets within each support condition. Missing or
unequal prefixes remain explicit. Candidate contact before an interrupted endpoint
is retained as a positive event; absent future motion is censored. Available
windows are materialized through the existing pulse dataset without fitting.

Pre-launch review found an inherited setup mismatch: the old±0.8m empty-space
prism necessarily intersects the pilot's east wall. The distinct evaluator-only
setup envelope now encloses all27 articulated collision primitives, with4cm
padding plus1micrometre strict-interior slack. It is derived from setup geometry
and joints, not wall proximity or desired success. Full native wall separation,
foot matching, ground support and velocity checks are retained. Tests demonstrate
that intersecting walls and overextended configurations still fail. No frozen
predecessor source was changed.

Focused tests:36passed in2.36s (terminal89112). An earlier213-file regression
passed2,689tests in207.31s (terminal68645), before the final setup-envelope addition;
the final36 focused tests cover that addition. The complete final-source regression
also completed:2,690passed in211.64s,213files, terminal68502.

## Frozen attempt

Output:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_independent_pulse_context_pilot_v1_attempt_001`.

Physics handle70727 is terminal exit0. All12episodes passed setup and reached
departure, then completed their fixed schedules with no physical/acquisition stop.
These collector reports are now independently audited. Result SHA256:
`132bcf46e4c62d892765b8c61a1a38563aa2aeeacc5fa2e9d1896f27e8f567ae`.
Launch SHA256:
`bb06bb68d6cc8702c224d1a5b78d5b4285b2c3dcf6636e9c029f425d0519319d`.
The launch binds694 source paths, native dependencies, gait inputs and the exact
twelve specifications. Protocol SHA256:
`d359683b32832cd533daa830068d56c683be461b67cbc5b1eb83bd327181b19d`.
The attempt is exclusive: no source changes, overwrite, automatic retry, resume
or alternative spawn after launch. All scheduled action siblings remain in the
attempt accounting even when one physically stops.

The simulator reports inherited robot-inertia/geometry center-of-mass warnings
and a neutral-position joint-limit warning. These do not establish a new failure,
but they are another reason not to call this real-platform validation. The robot
remains visually hidden, body sensing ideal, and physics paused during compute.

## Preserved audit failure and narrow read-only correction

Original raw-audit handle29004 terminated exit1 before completing an episode.
The auditor rounded its expected requested command to float32, while the physical
recorder stores requests as float64:0.12 differs by2.68220901e-9. Its synthetic
fixture used float32 requests and therefore missed the bug. This was not a native
command deviation. The original source and failure remain unchanged; failure SHA:
`95883bdb2bed71d157c100b3b7566a6030420ab31a977462b5fbefba937ccdbe`.

A [distinct read-only correction](go2_independent_pulse_context_command_encoding_correction_v1_2026-09-06.md)
requires the actual float64 request encoding and exact command equality, keeping
the separate applied-command slew comparison unchanged. Seven focused tests pass,
including reproducing the old false rejection and rejecting a single request
altered to its float32-rounded value. A read-only full raw replay of actual
nominal_action_0 then passed (handle99996, exit0), including setup/commands/stops,
31paired sensor frames and five actual target endpoints. Shadow visual pose was
available for only1/31decisions; collection nevertheless completed. No source,
trajectory, target definition or threshold was changed in the completed experiment.

## Completed audited result

The corrected audit93561 is terminal exit0. All12setups, departures and fixed
schedules complete;27,900native physics samples (55.8simulated seconds),390RGB-D
frames and390replayed decisions are retained. All twelve pulse windows materialize
through the actual dataset interface, with60valid native motion/contact endpoints
and60actual future-RGB targets. No known endpoint is missing. Each of the six
action-duration cells has one sample under each support condition.

All ten non-reference sibling comparisons match exactly: for each support,
all six actions share the full1,150-sample native/contact prefix and nine complete
RGB/body/control/depth/fast-gyro packets. The additional two self-comparisons are
not counted as independent matches. These are two verified action-context groups,
not twelve independent contexts or layouts. No cross-support matching is asserted.

Every raw-depth comparison is within1mm; the maximum interior-ray discrepancy is
8.2611micrometres.1,771,200 compared native rays fall below the public sensor's
20cm validity cutoff. Per-frame public valid-depth pixel counts range from0to
307,200. Invalid public depth stays invalid; the near native comparison supplies
no extra sensor data to the model or selector.

Shadow visual pose is available on only12/390decisions (one initial frame per
episode). The collector nonetheless records every assigned action. This confirms
the intended removal of tracker-dependent collection censoring, not repaired
visual tracking or safe blind navigation. All60contact endpoints are negative;
the pilot has no collision-positive coverage and cannot train or validate a
calibrated collision-risk predictor.

Actual final pulse-plus-brake displacements in the departure body frame,
centimetres, show why support-conditioned dynamics matter:

| Action / pulse duration | Nominal (dx,dy) | Lower friction (dx,dy) |
| --- | ---: | ---: |
| Forward /0.2s |(1.50,-0.21)|(3.88,3.41)|
| Forward /0.5s |(9.30,-0.99)|(4.42,6.32)|
| Left yaw /0.2s |(-1.48,-0.83)|(1.67,0.94)|
| Left yaw /0.5s |(-0.61,-0.95)|(2.81,0.34)|
| Right yaw /0.2s |(-1.58,0.18)|(1.58,2.94)|
| Right yaw /0.5s |(-0.38,0.23)|(2.01,4.45)|

These are single observations per cell, not estimates across independent seeds.
They do not prove that the input history identifies support or that a learned
model can predict the difference; those require the planned causal-input tests.

Post-completion verification30396 exits0, rechecking all697correction-source
bindings,1,689explicit input artifacts (477,349,731bytes), and every declared
correction-output binding. This is an integrity check, not another physical run.
Correction launch SHA256:
`56aee0c0345cba37473d6c79bc7812aac0a7b34254ad25be4ecbc1adfcc492b1`.
Corrected audit result SHA256:
`4b51738c6ae50051373b407685181b6ef18598ae528701a70b11270069416399`.
Windows SHA256:
`f416a6357456c516ce8831327b0c471ff425d0f1addbe36c8a266aa4d481543b`.
Targets SHA256:
`7aed9a7f1d9ad9602309887c0ba6a859700f702fc46868181f816dd31ea91dcf`.

No live process remains. No training, sealed access, source export, hardware
control, frozen-source modification or navigation promotion occurred.

## What this cannot establish

One training layout is not an independent train/evaluation split. Successful
fixed command execution is not goal-directed maze completion. Positive contacts
are not guaranteed by a near-wall context; if absent, report that coverage gap.
No new model is fitted or installed by this pilot. The preceding same-room
JEPA position result remains1.07–1.95cm versus0.859cm for the empirical action/time
baseline; the latest room-return result remains0/3.

## Next implementation work

1. The constructor/pairing pilot is complete. Preserve its sources and results;
   do not rerun it, switch its role, or fit another one-layout pilot. Its two
   matched support groups establish collection mechanics, not generalization.
2. Implement a distinct context constructor with quiet/recent histories and
   multiple obstacle/branch positions. The measured1.5–9.3cm nominal forward
   response and support-dependent lateral motion should inform prospective hazard
   placement. Verify full initial and common-warm-up clearance, then collect fresh
   physical outcomes; translating an old trace cannot supply new collision labels.
   Include both depth-supported and near-range-invalid views, retaining all outcomes.
3. Build the prospectively split connected-layout inventory, conservatively group
   topology-equivalent variants, and include quiet versus recent histories and
   multiple obstacle/branch contexts. Keep this pilot in training role only.
4. Establish sensor/history/action utility on independent development layouts,
   then compare direct, supervised-rollout and JEPA training under matched data,
   labels, seeds and budgets. Do not run another same-room optimization search.
5. Integrate useful prediction into reliable local execution and matched online
   candidate selection. Evaluate observed branch choice, physically executed
   backtracking and memory benefit on complete novel-maze tasks. Realistic sensing,
   deadlines and bounded hardware evidence remain separate required stages.
