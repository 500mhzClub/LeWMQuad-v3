# Isolated recent-qualified-reference replay on direct-flow maze3

This prospective development replay tests whether the existing one-view
reference-retention rule changes the direct-flow maze3 controller before its
tracking failure. It grants no native execution, training, deployment, sealed
evaluation, navigation qualification or retry authority. Existing queued native
attempts remain unchanged.

The fixed input is go2_direct_flow_maze03_pilot_v1_attempt_001, case
full_jepa_direct_flow_maze_03, layout3, full/jepa,
seed_2026091001_full_jepa. Launch SHA256:
48493379a21f2691a82195873b62ed101e927d3667d36493e4e6aa90d5aa3f28.
Its closed collection identities are:

- result.json:659f3aa6a75cb29ce72847308e89abf691e703e81739ce07fd6448d9b2e4cd23.
- context_decisions.jsonl.gz:fd55017ee0e2463aa4d443508047c586e27d253c9561a8d5811b9bab1042ea36.
- command_tape.json:23f07bd793d166e1ff8e6ad1863b3c7cfe9939196865b1466d9aa364ce2f4afd.

The collection has1217 observations,1216 completed requests and61550 physics
samples. Its first failure is observation1206; the failed receipt retains
controller tick/mission frame1205. Frames1196..1205 exhausted ten measured
bridges after qualified-but-unretained1195. These saved-receipt observations
motivate this replay, but runtime admission also requires the completed native
pilot result and original raw audit. The final result SHA must be independently
observed after completion and supplied explicitly through --native-result-sha256;
it is then bound into the replay launch and checked throughout. Collection-only
or provisional inputs cannot launch this replay. Require the original265-frame,
13950-sample physical/public prefix,261 equal forecast banks, and completed
changed command at264. All fixed input bindings and original transitive input
verifiers must pass; use the existing scoped digest helper with its admitted
benchmark, preserving initial/final hashing and original verifier behavior.

RecentQualifiedDirectFlowController inherits DirectFlowFloorTransportController
and replaces only its visual-motion owner with the existing
RecentQualifiedAnchorVisualMotion. Registration, measured floor transport,
mapping, residual adaptation, model, selector, mission, memory and original
direct-flow fallback remain those of this native input. No partial-floor-height
change is included. The added controller identity and retention flag explicitly
identify the intervention. The comparator rejects the partial-floor-height
field on either side.

Reuse the existing RecentQualifiedAnchorPose rule without modification: one
immediately previous, already anchor-qualified view may be tried only after all
original references fail with NO_QUALIFIED_REFERENCE. Preserve the original
camera priority, descriptor/rigid/temporal/conflict gates and bridge budget.
Never retain a bridge or floor-transport pose as an extra reference. This rule
cannot insert1195 directly at1206. Within the inspected late window,1193 and1196
are candidate earlier opportunities; neither is assumed to be the first actual
intervention or to qualify. Frequent qualified chains still have uncalibrated
accumulated pose error.

Replay from fresh initial controller/model state, using the unchanged assigned
model state4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.
Use only the recorded deployment-facing primary/auxiliary RGBD and sensed
packets. Verify public arrays unchanged; no native pose input or gradients.
Compare every complete original/new decision before the first extra reference
attempt, allowing only the declared retention receipt and controller identity.
After a declared reference attempt, retain the actual new perceptual/mission
state and compare every available shared-input raw6x8x5 forecast bank exactly.
Never reset new state to the original receipt. Require bounded causal reference
receipts, actual original command completion and exact observation endpoints.

Stop at the first changed requested command or either controller terminal,
before consuming any following recorded observation. At most1207 observations
0..1206 are available to this comparison. An earlier changed command ends the
prefix regardless of whether the expected late failure is reached. A terminal
or unchanged negative result is evidence, not authority to tune or retry. This
replay establishes no counterfactual physical continuation or navigation result.

Use one CPU replay process, one OpenCV/BLAS/PyTorch thread, deterministic model
execution,8GiB replay memory admission alongside the original32GiB native
allowance,40GiB free reserve plus2GiB output allowance, and a1GiB compressed
decision-stream ceiling. These are admissions, not OS-enforced limits. Record
fresh hardware before execution and after completion. All source/input bindings
are checked before output admission and after replay. Preserve any exclusive
attempt failure and output; no overwrite, skip, resume or automatic retry.

The exclusive output is
go2_recent_qualified_direct_flow_maze03_prefix_v1_attempt_001 under the fixed
external navigation development artifact root. This preparation does not
modify the original queue, launch another native scene, or qualify the goal.
