# Moving-prefix counterfactual V1: fixed physical data protocol

This new development-data intervention addresses measured missing action-history
support, not a request to tune a model until JEPA wins. The completed successive
panel's bounded JEPA latent-head advantage and all previous negatives remain
unchanged. No protected material, new model training, checkpoint selection,
source export, final-maze evaluation or hardware execution is part of this run.

## Population and exact physical sequence

Execute384 new trials: each of the original24 development layouts × four moving
prefix actions × four different future bank actions. Keep original16 train/eight
development-validation layout roles; every branch/window of a layout has the
same role. This does not add independent layouts. The prefix action is forward,
forward-left, forward-right or reverse. The future action is each of the other
four members of the unchanged five-action bank, including stop. Ordering is
layout index, prefix action index, future action index. No random selection of
suffixes, alternative seeds, outcome resampling or early superiority stopping.

Each fresh scene uses its original procedural seed and geometry, corrected gait
kp20/kd.5,1.5 s recorded zero settling and the unchanged baseline teacher
initialization (at most85 command ticks). Teacher eligibility requires sustained
correct crossing, no disallowed contact and no physical stop. Then execute10
100-ms ticks (one second) of the moving prefix action. If the entire prefix is
available and contact-free, execute30 ticks (three seconds) of the chosen
different future action, then five zero-command release ticks. The existing
command limits and per-tick slew remain unchanged. Never assume reverse or
braking has instantaneous kinematics.

CPU Genesis, one numerical thread,2-ms physics and20-ms ideal body sensing.
Capture actual fixed-camera RGB at every command boundary and the terminal
sample; keep policy histories separate from privileged raw pose/contact/camera
geometry. No learned policy makes a choice in this collection. Raw simulation
pose/geometry are used only by the declared teacher and evaluation/labeling.
Native contact or body-stability limits stop physics immediately, even on an
exact command boundary. Keep that sample and its terminal evaluation image;
do not execute a release after a native safety stop.

Record each requested command before dispatch and both its pre/post physical
sample indices, including interrupted ticks. Separate teacher, moving-prefix,
suffix and release stages so a zero release cannot be mislabeled as execution
of an unfinished candidate action. Partial and failed trials remain in the384
planned-trial denominator. There is no retry/resume/replacement authority.

## Match to existing actual conditioning states

All96 required one-second moving contexts already exist in the audited original
corpus:64 train and32 development-validation. Before launch, validate their
member results, physical traces, histories and camera audit bytes, then bind
the full physical prefix and the exact moving-state history/image index.
The new run starts fresh; it does not load a simulator snapshot or copy a state.

For every eligible new suffix, compare its entire settling/teacher/moving prefix
to the corresponding original branch through that same one-second boundary.
Require exact physical-array and sensor/control-history identity, exact camera
geometry and frame count. Use the already-declared small RGB tolerance from the
existing prefix matcher: maximum per-frame8-bit RMS<=1 and changed-pixel
fraction<=.001. Log the actual differences; do not loosen thresholds on failure.
Matching is an evaluation step after artifacts are persisted. An eligible
prefix identity mismatch is an infrastructure/integrity failure that ends the
attempt with its evidence preserved, not an excuse to retain an unmatched
counterfactual. A physically unavailable prefix is retained as unavailable and
has no fabricated suffix label. Do not confuse these two cases.

The reference's future trajectory must never enter a current policy input.
Later training may use the matched original moving context as a canonical shared
conditioning history for its five alternatives, while future RGB is always
the actual corresponding branch. This is training-data construction only;
online policies must still consume their own actual RGB.

## Targets and planned composite dataset

For a branchable suffix, form targets at.5,1,1.5,2,2.5 and3 seconds from its moving
conditioning state. Translation is full3-D relative rotation into the current
body frame, retaining its x/y components; yaw uses the existing wrapped heading
change convention. Valid motion/future RGB must be strictly before first contact.
Within the known three-second suffix, early contact is absorbing for contact
labels even when its future motion endpoint is unobserved. Noncontact early
termination censors unobserved targets. Tensor slots at3.5 and4 s are outside
the known plan and have no motion, RGB or contact target, even if a release
sample exists there. Truncate evidence at the final suffix sample before making
these labels. The release is recorded and separately accounted, not hidden.

The intended600 context/action cells combine120 existing initial branches,
96 existing one-second moving continuations, and384 new off-diagonal suffixes.
This gives400 planned train and200 planned development-validation cells. These
are provenance-linked observations, not600 new environments or600 guaranteed
available labels. Existing diagonal outcomes are referenced, never imputed from
a model. Any unavailable new conditioning state remains a missing cell with
explicit failure accounting. Do not train until the actual composite dataset
and independent raw audit are implemented and verified.

## One-shot artifacts and stopping policy

Exact output root:
`.generated/go2_moving_prefix_counterfactual_development_v1_attempt_001`.
Freeze this protocol and all launch-bound source before starting. Launch records
all384 exact trial specs,600 planned provenance cells,96 reference bindings,
source/input/gait hashes and software versions. Verify bindings before and after
execution. A minimum10 GiB free space is required before collection.

Retain all raw physics/native contacts, sensor samples, per-frame RGB, camera
audit, command tape, teacher decisions, moving-prefix identities, suffix targets,
individual results and terminal status. Unexpected infrastructure error ends
the attempt; it does not permit silent reexecution. An independent audit must
reconstruct physical contacts, clocks, stages, clipping/slew, sensor histories,
camera geometry, both prefix endpoints, source matching and target censoring.
Passing source/synthetic tests does not qualify newly collected data by itself.

After collection, audit the actual data, implement its policy-only composite
loader, and only then specify a matched training comparison. In parallel with
this physical data work, advance observed exits/place associations, translation
and persistent directed memory toward hidden-beacon discovery and return.
This dataset is a causal-coverage intervention, not completion of the maze goal.
