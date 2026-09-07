# Next data stage: independent connected layouts and identifiable pulse contexts

Implementation design, not a launched collection or evidence of navigation.
The fixed position-scale/budget comparison must finish without adaptive changes.
Regardless of its result, do not start another same-room loss/architecture search.

## What the current source permits and what it would bias

The pulse dataset already joins actual past RGB/body/control histories with exact
known-command prefixes and separately audited native motion/contact outcomes.
`PulseNativeTargets.labels` preserves contact evidence before interruption and
censors noncontact stops or post-divergence outcomes. Reuse that distinction.

However, `PulseSequence.step` in `lewm/command_pulse_response_development.py`
requires `CURRENT_VISUAL_POSE` before every fixed excitation command. The newer
room-return controllers also terminate on unavailable visual registration.
Collecting only such trajectories would censor the hard visual/body states that
the predictor needs to handle. The next simulation-only excitation selector must
depend on a prospectively fixed command schedule and causal packet validity,
not successful pose registration or favorable future labels. Log the original
tracker in shadow for diagnosis; do not relax its deployment acceptance gates.
Retain external native contact, stability, speed and workspace stops. This is
a bounded simulation data-collection change, not permission for blind hardware
motion or for bypassing navigation safety checks.

`CommandPulsePhysicalInit` and `IntentRoomReturnPhysicalInit` currently require
their exact old scene specifications. Create a distinct context-collection
specification and initializer; do not patch frozen constructors or monkeypatch
them at runtime. `fresh_fused_maze_scene_development.specification` demonstrates
connected cells → passage edges → unique physical wall boxes, but contains one
fixed layout. It is a construction reference, not a source of independent splits.

## Collection unit and controls

The scientific unit is a complete metric layout, not a frame, pulse, appearance,
friction setting or trajectory. Generate an explicit new development inventory
of connected layouts containing corridors, turns, branches and dead ends; include
loops for later memory tasks. Assign train/selection/development-evaluation roles
before collecting or fitting. Group identical wall geometry and rigidly transformed
copies together; group topology-equivalent variants conservatively rather than
count textures/spawn changes as independent layouts. Publish canonical geometry/
topology identities and cross-role duplicate checks. No final sealed manifest
belongs in this checkout, and these development roles are not final evaluation.

Within each layout, define several initial observation contexts: open passage,
turn/junction, dead-end approach and obstacle-proximal geometry. For each context,
cross the six existing command-duration cells (forward, positive yaw, negative
yaw × 2/5 ticks), at least two support conditions and both quiet and recent-motion
histories. Reset separate physical episodes to a shared construction and seed
for action siblings. Execute and record a common setup/warm-up independently in
each sibling; verify identical physical and observation prefixes before treating
them as same-context alternatives. Do not assert counterfactual matching from
matching seeds alone or borrow another sibling's later history.

Use a bounded initial constructor/pairing pilot on new **training-role** contexts
before the full inventory. Freeze its exact scene list, warm-ups, seeds, durations,
command tapes, stop conditions and resource budget. Keep all attempted setup,
warm-up and action outcomes in accounting. A pre-departure failure is a recorded
failed context, not an invented training window. A stopped action must not remove
the remaining planned sibling attempts from coverage accounting.

## Support must be identifiable, not supplied as a hidden oracle

The predictor receives four RGB frames spanning0.3s. Each packet contains up to
0.4s of body sensing (20 samples at50Hz) and up to1.5s of command history. These
limits are explicit in `simulated_body_observation_development.SCHEMAS` and the
four-frame tensor interface. No friction coefficient enters model inputs.

At rest, different friction coefficients may produce indistinguishable RGB and
proprioception. A deterministic point predictor cannot be required to infer an
unobserved parameter merely because the evaluator knows it. Include recent,
bounded excitation whose response remains in the actual input-history window at
departure; contrast it with quiet/ambiguous histories. Verify that the excitation
is present in captured timestamps, rather than assuming a probe several seconds
earlier is still visible. Keep warm-up commands equal across action siblings.
Longer persistent belief/history or uncertainty-aware prediction may later be
needed; do not conflate inability under unobservable state with optimization failure.

## Hazards, sensors and target integrity

Include obstacle proximity where short pulses can yield different actual contact
outcomes, while checking complete articulated setup clearance. Do not place the
robot in initial penetration to manufacture positives. Simulator geometry and
friction are construction/evaluator fields only. Foot support contact, wall/body
collision, missing RGB and tracking rejection are different events. Report their
separate frequencies and censoring denominators, all six action cells and all
layout/context/support/history strata before training. If intended hazard cases
yield no positive contacts, report absent coverage; do not call all-negative
accuracy successful collision prediction.

Keep RGB, body gyro/specific force/joints and command timestamps causal; audit
native targets separately. Depth may remain a declared sensor/control baseline
or audit channel, but do not quietly add it to one learning arm. Current body
sensing is ideal50Hz simulation, RGB-D is rasterized and the robot is hidden from
the camera. Those are still simulator assumptions, not deployment validation.
Plan separate robot-self-occlusion, realistic calibration/noise/latency and wall-
clock deadline studies. Native supervision during training is not a deployment
oracle, provided the inference boundary excludes it and baselines get matched
training-label access.

## Exit criteria and downstream scientific experiments

Before fitting: audit explicit artifact bindings, identical action prefixes,
source/target clocks, complete attempt accounting, no cross-role geometry leakage,
actual class/action/state coverage and adequate free storage. Limit cache/material-
ization to explicit bound paths. Use the RecoveryStorage artifact helper with a
fresh owned attempt root; do not clean, relocate or overwrite prior experiments.

Then compare action/time, body/history-conditioned and RGB-plus-body predictors
on independent development layouts, using paired seeds/schedules. Shuffle/remove
RGB, history and action under prospectively defined interventions to test their
actual contribution. Match direct, supervised-rollout and JEPA observation/label
exposure; report errors and decision consequences by support and obstacle context.
Do not replace a failed primary comparison with a favorable auxiliary head.

Prediction quality alone is insufficient. After reliable local execution,
integrate matched candidate selection, observed branch choices and physically
executed backtracking. Test predictive-training, online rollout and memory on/off
as separate factors with equal local-control/sensor budgets. Require completed
novel-maze outcomes, realistic safety/deadline evaluation and bounded real-platform
evidence when hardware is available. The current0/3 room-return result remains
negative; this design does not revise it.
