# Persistent feedback fixes alignment timeout, not the whole navigation task

All16 fixed trials completed and the full raw audit reproduced4,600 decisions.
Initial alignment completes in every trial. Each method achieves3/4 first-leg
integration successes and2/4 two-leg successes; one fixture contacts a wall
during scanning and another physically arrives but misses the sensory arrival
criterion. No sensor-contract or body-stability failure occurs. The final
two-leg success count remains unchanged from both predecessors.

## What changed, and what did not

The [fixed protocol](go2_persistent_alignment_continuation_development_v1_2026-09-05.md)
keeps proportional yaw feedback active inside the initial alignment acceptance
band until its unchanged0.3 s dwell completes. It preserves0.02 rad heading,
0.1 rad/s rate,12 s deadline, post-alignment zero hold, fresh traversal frames,
all subsequent operators and physical task criteria. No predecessor source,
outcome, model or experiment was edited or rerun.

| Method | Initial alignment /4 | First leg /4 | Two legs /4 | Scan contacts /4 | Missed arrivals /4 |
|---|---:|---:|---:|---:|---:|
| Fixed forward |4|3|2|1|1|
| Direct prediction |4|3|2|1|1|
| Supervised recurrent prediction |4|3|2|1|1|
| JEPA recurrent prediction |4|3|2|1|1|

Fixed-forward makes50 forward choices. Each learned arm makes48 forward and two
forward-right choices; the learned arms have identical physical trajectories.
There are four exact raw trajectory groups, sized4/2/6/4, not16 independent
replications. Successful fixed-forward tasks take45.4 s after settling versus
45.2 s for each learned arm. Neither this small difference nor task success
separates JEPA from the other learned methods.

## Alignment and release are different conditions

The negative initial conditions now complete alignment after1.5 s with estimated
error0.014860 rad. By first-traversal start after the zero hold, true relative
heading error is0.031632 rad, outside the alignment-time tolerance. Initial-phase
translation is0.005494 m. The positive conditions complete alignment in0.3 s;
their first-start true error is0.003106 rad and translation0.002131 m.

This supports the diagnosed command-switching mechanism: persistent feedback
resolves the timeout under the original acceptance predicate. It does not show
that post-release heading remains within that predicate, or that floor-extension
bearing is a corridor centerline. Acceptance under feedback is not a certificate
of zero-command settling, arrival quality or swept-body clearance.

## Remaining failures identify task-level sensing decisions

### Negative corner: an opening is seen before the collision

The first leg passes all original physical crossing/release checks. Scan starts
at(1.232468,-0.137019) m, less laterally displaced than the original unaligned
corner's(1.230017,-0.207407) m. It now completes its first quarter-turn view,
but continuing toward the second leads to a native RL_calf-group contact with
the south wall. Stop is immediate at17.840 s global simulation time,16.340 s
after initial settling. Contact position is(1.165046,-0.600048,0.188373) m,
force magnitude102.571 N. The rigid group includes fixed child shapes; its name
does not identify the exact contacting primitive. Scan drift through the last
observed decision is0.059273 m.

At16.800 s, observation153 already contains a fresh side-opening proposal with
body bearing0.052360 rad and27 supported angle bins/873 points. Reapplying the
unchanged branch-selection function to only observations available at that time
accepts it: relative incoming bearing1.449436 rad, age0. The actual RGB frame
was also visually inspected. That is1.040 s before contact, while the prescribed
full-circle scan is still running. No future observation or true wall coordinate
was used in this selection diagnostic.

This motivates a task-directed scan that stops after obtaining the required
branch evidence. It does not prove a counterfactual early-stop trajectory would
avoid contact or successfully traverse. Full-circle sensing may still be useful
when searching for more exits/beacons; it need not be mandatory before every
local move. A new policy must report interrupted scan honestly, not relabel it
as a completed360-degree assay.

### Negative tee: stopping provides better arrival evidence

All methods stop with FAILED_FIRST_NO_VISUAL_CHANGE. Evaluation shows a genuine
physical arrival:1.549556 m progress, full body0.500950 m past the crossing plane,
base inside the destination and successful release/stability checks. The runtime
does not receive these geometric labels and creates no arrival candidate.

At the translation cap, the three most recent moving floor-mask change fractions
are0.097292,0.088125 and0.111875. They fail the unchanged requirement that all
three exceed0.10. The current controller declares terminal failure immediately
rather than first stopping to acquire stable evidence.

The five actual zero-release observations subsequently give fractions0.103542,
0.112500,0.105833,0.115833 and0.116458. Body quietness is false on the first four
and true only on the last. These are real captured frames after stopping, not
predicted images. They support testing bounded stop-and-observe acquisition,
but do not retrospectively satisfy the0.3 s quiet dwell: that evidence was not
recorded. The existing result remains a missed arrival, not a rescued success.

## Verification and retained artifacts

Collection97516: COMPLETE16, exit0. Audit65700: PASS16/4,600 decisions, exit0.
Full evidence covers244,280 physics/live-fast-gyro samples,24,428 ordinary sensor
samples and4,664 actual RGB packets. It verifies exact causal packets, commands,
models/controllers, native contacts, camera transforms, ledgers and unchanged
physical outcome reduction. Only the two declared learned timing fields are
excluded from decision equality. Seventeen new tests and893 focused tests
across80 files passed before launch; the latter took26.43 s (session90838).

Launch binds207 source/test/protocol paths,173 inputs and two gait bindings.
Read-only diagnostics41339 (early eligible branch),82114 (actual release-frame
arrival evidence) and57905 (raw grouping/choices/geometry) completed exit0.
These diagnostics wrote no artifacts and did not change collection or audit.

Root: `.generated/go2_persistent_alignment_continuation_development_v1_attempt_001`.

- Launch: `9aa05275e42bae0cac32f7c7dd111926af552e4eb5a7b80418358d8d2d77415e`.
- Result: `522326480ad94873b7232858bcd7a8cf3f6a3ca6d86ce4fc2fdc6b2848970344`.
- Full audit: `1d478522b74501b0ed21cce16feaf3e3a12fe0f82b8c06ee9a650e0c145bd112`.

## Next steps toward the scientific objective

1. Implement bounded stop-to-observe arrival acquisition at the existing
   translation cap. Keep progress, three-frame visual evidence, quiet dwell,
   settling deadline and physical crossing/release checks; change how evidence
   is acquired, not its threshold. No unconditional candidate from elapsed time
   or command distance, and no simulator destination labels in control.
2. Implement task-directed scan stopping after a fresh eligible side branch is
   observed at an acquired view. Zero the command before further turning, retain
   release/hold handling, align and demand fresh forward reobservation before
   translation. Store partial-view history as partial. Report full-scan completion
   separately from navigation-task success; do not fabricate a COMPLETE scan.
3. Test the two changes individually and together with the matched fixed-forward
   control, then matched learned arms in the integrated task. Specify the new
   population and task metric before execution. Keep contact, physical arrival,
   false/missed candidate and stall accounting, including failures introduced
   by the new stopping choices. Neither intervention is yet physically validated.
4. Use the resulting local executor in an uncertain episodic memory/branch
   prototype with actually observed initially hidden beacons and directed return.
   Observation hypotheses may guide development without being called trusted
   edges. Repeated-looking places, loop closure and false associations must be
   evaluated, not hidden by injected cell identities or an oracle route.

These are task-directed observation/control hypotheses, not further model
capacity tuning. Reliable navigation, JEPA/online-rollout contributions on
independent mazes, robust deployment-valid sensing and bounded real-Go2 evidence
remain open. Current ideal simulation stops and sensors are not hardware safety
or transfer evidence, and this completed local panel is not final-goal success.
