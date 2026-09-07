# Fresh online conditional-choice Go2 pilot, development V1

Specified before physical collection. Prerequisites: the completed nine-model
learning/prediction audit and the 72-call strict adapter replay. No fitting,
seed/checkpoint selection, parameter search, corpus recollection or protected
benchmark access belongs to this experiment.

## Question and fixed population

Can the deployed-shaped **actual current observation → ensemble prediction →
chosen command tape → physical execution** path improve useful local progress
relative to stopping on fresh layouts? Does adding the latent-prediction loss
help compared with the matched supervised-rollout ensemble?

Eight new procedural seeds 2026091400–2026091407; all topology hashes distinct
from each other and the preceding 24 layouts under grid rotations/reflections.
No rejection sampling or replacement seed. Each 4×4 graph has a leaf approach
and one designated junction, four balanced exit motifs, width uniform in
[0.95,1.25] m, spawn lateral offset in [−0.07,0.07] m and yaw in [−0.12,0.12] rad.
The fixed generator uses the same visual/material/physics family. These are
fresh local-layout draws, not a broad visual-domain or full-maze test.

Three supplied body-frame displacement intents: forward `(0.8,0)`, left
`(0,0.8)` and right `(0,−0.8)` m. They do not reveal which exit is open. Three
methods: all stop, supervised-rollout ensemble, JEPA-rollout ensemble. Both
learned methods use all three previously fitted seeds, fixed hashes and the
same adapter/candidate bank. Exactly 8×3×3 = 72 planned physical trials, in
layout/intent/method order, without replacing failures or adding attempts.

## Execution and policy boundary

Recreate each scene with its identical procedural seed, corrected actual gains
20/0.5, fixed gait/checkpoint, 0.002-s physics and the existing ideal sensor/RGB
recorder. Settle 1.5 s and use the unchanged baseline teacher prefix, at most
85 command ticks, to establish a sustained contact-free junction crossing.
The teacher uses privileged pose; that initialization is explicitly conditional,
not autonomous navigation. Prefix failure counts in the final denominator.

After an available prefix, capture actual fixed-mount RGB and construct the
packet from the live body/control buffer. The selector receives only that packet,
the intent and current clock. It does not load a sibling canonical observation,
dataset row, future target, maze graph or simulator pose. Every physical trial
owns a fresh one-shot selector episode; recorder identity `(0,0,0)` is local to
that fresh recorder and is never reused across a persistent policy object.

The adapter reconstructs the five fixed stop/forward/left-curve/right-curve/reverse
plans from known past command and original slew limits. Mean member contact
probability and mean displacement determine cost `10*p + distance(mean_xy,intent)`
at 4 s. Execute the selected 40×100-ms requested-command tape and five zero-command
release ticks. Requested and post-slew commands are distinct, fully recorded and
audited. No replanning, fallback action, retry or model update is allowed.

Stop physics at the existing first disallowed native contact or body-stability
violation, retaining the stopping sample. These privileged simulator emergency
monitors are experimental safeguards, not demonstrated physical sensor channels.
The synchronous simulation pauses while rendering/inference run; report capture
and full adapter wall time, rather than claiming real-time sensor-delay handling.

Require exact physical/body/control prefix arrays and camera geometry across all
nine trials of a layout. Use the already fixed per-image RGB limits: RMS ≤1 raw
8-bit unit and changed-pixel fraction ≤0.001. Retain all actual images. Prefix
comparison is audit evidence only; it must not substitute a reference image for
the policy input. Unexpected mismatches/integrity errors terminate the attempt;
partial artifacts remain and no automatic retry/resume occurs.

## Endpoints and reduction

Primary all-trial realized utility: cost 10 for unavailable prefix, contact by
4 s, incomplete noncontact 4-s outcome, or any physical failure during release;
otherwise Euclidean error from the observed 4-s displacement to the supplied
intent. Post-stop displacement is never imputed. Release failure and prefix
failure are explicitly reported categories, not silently dropped examples.
This primary all-trial cost extends the earlier offline secondary diagnostic
to penalize release/init failure; do not pretend the two utilities are identical.

Report contact-stop fraction over the full trial, prefix availability, selected
stop fraction, observed 4-s progress, physical failure categories and capture/
adapter latency. Reduce the three intents within each layout first, then eight
layouts equally. Report paired method cost differences by layout and descriptive
95% percentile intervals from 10,000 layout-bootstrap resamples, fixed seed
2026091499. No independent-frame/intent/seed replication claim or multiplicity-
adjusted final-test claim. Only one action per method is executed in each state;
there is no fresh oracle action regret measurement.

This is an informative development terminal whether the methods succeed or fail.
Better paired cost with retained contact/progress accounting motivates successive
sensor-only decisions on broader training states. A null/negative result motivates
mechanistic observation/state-coverage diagnosis—not seed replacement or a loss
coefficient search. No numerical threshold here promotes a model to hardware.

## Required audit and custody

Bind the new generator/adapter/runner/spec, predecessor source/gait identities,
model checkpoint hashes and replay result at launch. Retain per-trial raw physics,
native contacts, gains, scene topology, causal observations, actual online
selection inputs/predictions/costs, requested tape and censored labels. Keep
the original compatible sensor/physical artifact hash population; bind the new
`online_selection.json` separately in each result.

After completion, reuse the unchanged raw branch audit with the independently
replayed selected action as its declared tape. Additionally reload each trial's
own starting policy packet, replay the fixed ensemble/choice, verify input hashes,
all action plans and executed slew, recompute utility including release failures,
and recheck paired reduction/prefix matching. Audit never turns simulator labels
into policy inputs. Do not claim a result until these checks pass.

Output is exactly `.generated/go2_online_choice_maze_pilot_development_v1_attempt_001`.
No clean-tree export or frozen-source mutation is needed. Full unseen-maze
exploration, beacon discovery, online place memory, return and hardware transfer
remain separate requirements in EXECUTION_PLAN.md.
