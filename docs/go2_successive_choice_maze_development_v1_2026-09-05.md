# V1 successive RGB/body directional-choice diagnostic: fixed protocol

This is a new development study under the user's autonomous scientific task.
It neither accesses protected material nor reopens any historical experiment.
No training, checkpoint selection, architecture change or source export occurs.
Freeze this document and all launch-bound source before physical execution.

## Question and fixed population

Do the completed temporal models choose useful, safe successive actions when
their own actions change the RGB/body histories? Earlier offline replay passed
but does not answer this question. The training suffixes largely continue the
past action: candidate switches and later closed-loop states are a deliberate
distribution test, not certified counterfactual coverage.

144 fresh simulated trials: eight 4-by-4 procedural layouts, seeds
2026091800 through 2026091807; three initial-body-frame direction cues
(.8,0), (0,.8), (0,-.8); six fixed methods in this order:
always_stop, direct_direct, supervised_direct, supervised_rollout, jepa_direct,
jepa_rollout. Trial order is layout, direction (forward,left,right), method.
There is one physical execution per cell, no best seed and no rerun on failure.
Three final checkpoints per learned arm (2026091700–02), equal ensemble weight.
The eight layout graphs must be unique under the existing dihedral identity and
disjoint from the 24 training/development corpus layouts and eight previous
online pilot layouts. No outcome-dependent resampling. Widths .95–1.25 m,
neutral walls .6 m high/.08 m thick, four junction motifs repeated twice.
Exact geometry, initial pose, commands, model identities and source/input hashes
are recorded in launch.json before the first scene is initialized.

## Physical initialization and live causal loop

Use the unchanged corrected-gait RouteSession stack (kp20, kd.5), CPU Genesis,
one numerical thread, native 2-ms physics, 20-ms body sensing, 100-ms commands.
Each independent trial starts with 1.5 s zero-command settling and the unchanged
baseline teacher crossing into the initial junction (at most 85 command ticks).
Teacher pose/route are privileged initialization only. Prefix eligibility is a
sustained correct crossing with no disallowed contact and no physical stop.
Keep prefix failures in the 144-trial denominator; never resample them.
Check the paired physical/body prefixes using the existing V2 numerical RGB
tolerance; every method still consumes its own actual pixels.

Warm the four-frame live buffer using actual packets at every prefix command
boundary. Do not rebuild a historical packet from a later live sensor buffer.
At the eligible terminal prefix packet, initialize causal relative orientation
and the supplied initial-frame direction. Thereafter the policy receives only
RGB, body sensor/applied-command histories, clock and that direction. No world
pose, maze graph, contact label, future realized control or canonical sibling RGB
is passed to it. Orientation transports the fixed initial cue as the body turns;
there is no relative translation estimate, so this is not a point-goal task.

Execute eight .5-s choices: each ranks the existing five-action bank using
10 times mean contact probability plus Euclidean distance between mean predicted
.5-s body displacement and the transported .8-m cue. Average probabilities,
not logits. Use the direct or rollout head named by the method; always_stop
requests zero but follows the same observation/decision timing. Supply only the
prospective five post-slew commands to the model, masking remaining slots as
unknown. Execute all five selected requested commands, receiving a fresh packet
every100 ms; then replan. Log model/input hashes, predictions, chosen commands,
pre-dispatch sample indices, elapsed inference/adapter time and actual execution.
Finish with five zero-command release ticks (.5 s), without learned replanning.

Sensor/model contract faults cancel learned motion and explicitly dispatch the
five zero release ticks, bypassing the failed adapter. Record the fault before
release so a subsequent native stop cannot erase it. Native disallowed contact,
low height or excessive attitude stops halt physics immediately and destroy the
isolated simulated trial; do not continue stepping after contact. Unexpected
infrastructure errors retain partial raw evidence and terminate the study.
The native safety monitor uses privileged simulator state. It is an experimental
guard, not a deployable RGB/IMU contact detector or real-robot watchdog.

## Fixed reporting and failure policy

Separate all-trial contact, prefix failure, sensor failure, completion of the
40 control plus five release ticks, and release-motion pass (last .3 s maximum
world-horizontal speed <= .1 m/s and absolute world-z angular rate <= .25 rad/s).
Report signed displacement along the initial direction and lateral displacement
to the actually observed control endpoint, including early stops and their
observed durations. Fixed-four-second progress is null for incomplete trials;
do not impute stopped motion or hide missing cases in an unqualified mean.
Report the resulting conditional progress with observation counts and all-trial
failure incidence. Do not manufacture a favorable composite success score.

Report action changes, moving-to-stop and moving-to-reverse transitions, chosen
stop counts, individual .5-s executed contact Brier/motion errors and mask counts.
Only the actually executed candidate is labeled. Truncate at the final executed
control tick so a fault-release zero command cannot fill an unexecuted candidate
horizon. Motion targets are strictly pre-contact; early contact remains observed
even when its future motion endpoint is unavailable. No calibration claim follows
from Brier alone. Latency excludes simulation/capture unless explicitly included.

Average the three directions within layout. Fixed contrasts: each learned method
minus stop; supervised-direct minus direct-only; JEPA-direct minus supervised-
direct; JEPA-rollout minus supervised-rollout; and rollout minus direct within
both paired-head conditions. For contact, completion and release pass, report
paired layout deltas and descriptive 10,000-resample percentile intervals using
seed2026091899. No multiplicity-adjusted confirmatory claim. Fixed-horizon progress
contrasts use explicitly matched completed intent pairs with omission counts,
and are labeled survivor-conditional. Publish all individual outcomes.

One exact output root:
`.generated/go2_successive_choice_maze_development_v1_attempt_001`.
No CLI overrides, retries, resume, replacement seed, early superiority stopping,
or threshold tuning. Infrastructure failure ends this attempt and preserves
everything; any successor requires a separately documented integrity diagnosis.
Passing unit tests/source bindings is not a physical scientific result. After
collection independently replay packets/choices, commands/slew, native contacts,
sensor histories, prefix eligibility and metrics before interpreting efficacy.

This panel is conditional directional continuation in simulated fresh topologies,
not autonomous maze discovery/return, independent final evaluation, calibrated
hardware sensing or demonstrated JEPA advantage. Preserve negative evidence.
