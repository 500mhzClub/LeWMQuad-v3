# Go2 active interfaces and synthetic semantic checks

Date: 4 September 2026. Inspected source baseline: `1f7dd8e`.

Follow-up: the later [physical-interface development assay](go2_physical_semantics_development_v1_2026-09-04.md) executed fresh Genesis primitive scenes and passed render/contact, color-only dynamics and known-angle projection checks. It narrows several outstanding questions below but does not qualify the active Go2 pipeline or hardware. The source/synthetic-only results in this document describe the earlier package.

This completes the source-specification and synthetic-fixture portion of the
[future plan](../FUTURE_PLAN.md). It does **not** qualify physical navigation,
validate a dataset, or authorize a frozen experiment. The
[scientific review](scientific_review_and_navigation_plan_2026-09-04.md) remains
the evidence-backed account of the scientific aims and reported results. The
obsolete BEV direction is excluded.

The new [interface checks](../lewm/interface_semantics.py) are standalone,
in-memory, prospective guards. They are not installed in frozen collectors or
evaluators. The [synthetic tests](../lewm/tests/test_active_interface_semantics.py)
exercise selected existing source functions without loading models or datasets.
Frozen implementations and reported results remain unchanged.

## 1. Which interface is active?

There is no single interchangeable RGB/action interface across this repository.
In particular, the historical predictor, the analytic current-visual ranker,
and the physical handoff collector must not be conflated.

| Path | Geometry and RGB | Model-facing role | Qualification boundary |
|---|---|---|---|
| Historical `textured_v03` wrapper | `walls`, `obstacles`, `landmarks`; each box uses `center_xyz_m`, `size_xyz_m`, optional Euler angles. Direct 224×224 render; nominal mount, no camera-safety retraction in the wrapper | Historical frozen-feature action-conditioned prediction | Synthetic scene-submission tests only; no historical data inspected here |
| August 31 local-subgoal runner | Planar `Rectangle(cx,cy,sx,sy)` walls; analytic RGB raycaster, 224×168, declared horizontal FOV 92°; analytic collision/integration | Current-visual action ranking; not physical Go2 dynamics | In-memory obstacle insertion/removal changes RGB and collision status |
| Physical graph-edge handoff runner, inherited by successors | `geometry["wall_boxes"]` becomes one `ScenePack` used to build the physical/rendering scene; Genesis native 640×480, then area resize to 224×168 | Frozen current-visual ranker under physical dynamics and new target distribution | Camera conversion tested with an injected RGB array, not a Genesis render or robot |

Sources: [historical wrapper](../lewm/oracle/go2_textured_v03_renderer.py),
[legacy builder](../scripts/render_replay_v03.py),
[analytic runner](../scripts/run_non_greedy_local_subgoal_jepa_planning_v1.py),
[physical runner](../scripts/run_physical_graph_edge_handoff_qualification_v1.py),
[physical contract](../lewm/safety/physical_graph_edge_handoff_qualification_v1_contract.py).

The confirmed floor-only caller in
[the August 26 waypoint renderer](../scripts/render_safe_local_waypoint_route_intent_v2.py)
supplies `genesis_scene.json` with an `objects` schema to a builder that only
iterates the three historical collections. The builder does not reject this
schema; a recording-scene fixture reproduces submission of the floor alone.
The earlier [counterfactual builder](../scripts/build_go2_counterfactual_fidelity_stage_a_v1_2.py)
instead binds and loads `manifest.json`. This source distinction is important:
the demonstrated defect does not establish that every historical predictive
dataset was floor-only. That provenance question remains unresolved here.

For future legacy-builder callers, require all three collections explicitly,
even if empty; reject `objects`, missing collections, nonfinite coordinates,
nonpositive sizes, and malformed rotations. `validate_v03_manifest` implements
this narrow guard. It is intentionally not a general `ScenePack` validator.

## 2. Camera and coordinate contracts

Array shapes below are height-first; resolution pairs in platform manifests are
width-first. Confusing these conventions silently changes aspect ratio.

| Quantity | Source-defined value or convention | Remaining issue |
|---|---|---|
| Platform RGB | RGB8, native `(480,640,3)`, nominal mount `[0.326,0,0.043]` m, mount RPY zero, near/far 0.05/200 m | Hardware intrinsics, distortion, exposure and rolling-shutter behavior are not verified |
| FOV | Platform declares horizontal 78.323°; analytic training renderer declares horizontal 92° | Declaration alone does not validate Genesis's effective projection; test known angular targets in the actual renderer |
| Physical preprocessing | First RGB channels, optional float-to-uint8 conversion; require native shape; CPU area resize to `(168,224,3)` and round to contiguous uint8 | Tested with synthetic pixels, not scene content; do not infer sensor calibration |
| Encoder preprocessing | Contract specifies `(3,384,512)` float32 and spatial grid 24×32, resulting in 768×1024 descriptors | Source contract inspected; encoder/checkpoint not loaded or independently reproduced |
| Historical preprocessing | Wrapper requires a 224×224 training resolution | Platform's `training_resolution` is historical; it is not the physical ranker's final image shape |
| Base quaternion | Robot API WXYZ converted explicitly to XYZW for camera math | Tests check a known +90° yaw, mount translation, forward and up |
| Physical camera placement | Uses effective mount including any configured jitter and camera-safety placement, including possible retraction | Effective per-frame pose must accompany future geometric labels; a nominal mount is insufficient after retraction |
| Body/target convention | Planar body x forward, y left, positive yaw rotates x toward y; angles wrap modulo 2π | Roundtrip test treats +π and −π as the same orientation |

Sources: [platform](../config/go2_platform_manifest.yaml),
[camera safety](../lewm_genesis/lewm_genesis/camera_safety.py),
[scene loader](../lewm_genesis/lewm_genesis/scene_loader.py), and the physical
runner/contract above.

### A reflected camera basis is not an SE(3) rotation

`render_rgb_and_transform` constructs columns `[forward,right,up]`, with
`right = forward × up`. At an identity body pose this is
`diag(1,-1,1)`: orthonormal, but determinant −1. A synthetic call to the real
method reproduces this. The returned matrix describes a left-handed FRU basis;
it cannot be treated as an ordinary proper rigid rotation or directly converted
to a rotation quaternion.

This does **not** prove the RGB itself is wrong, nor that existing stored hashes
or descriptive snapshots are invalid. It is a hazard for future geometric
consumers. `validate_rigid_transform` explicitly rejects reflections. In a new
versioned interface, choose and name either a proper FLU frame or an optical RDF
frame and bind the conversion, intrinsics, and transform direction together.
For example, negating the right column yields FLU; optical RDF uses columns
`[right,-up,forward]`. Neither conversion is applied to frozen outputs here.

## 3. Timing and actions

The [platform contract](../config/go2_platform_manifest.yaml) and physical
runner define these nested clocks:

| Level | Period | Relationship |
|---|---|---|
| Physics | 0.002 s | 10 physics steps per policy update |
| Low-level policy | 0.020 s | 5 policy updates per command tick |
| Velocity command | 0.100 s | 5 ticks per macro-action block |
| Macro-action | 0.500 s | Physical H1/H2/H3 end at 0.5/1.0/1.5 s |

A physical 1.5 s candidate contains 15 command ticks, 75 policy updates and 750
physics steps. Historical predictive H1–H4 represent 0.5–2.0 s blocks. The old
manifest's `FRAMES_PER_TIMESTEP=48` is a lineage-specific frame-index convention,
not the current physical collector's 50 physics steps per command tick. Do not
infer sample time from a frame index across lineages.

Commands are `[vx_body_mps,vy_body_mps,yaw_rate_radps]`. The active platform clips
vx to ±0.3 m/s, vy to zero and yaw rate to ±0.5 rad/s, then limits changes per
0.1 s tick to 0.25 m/s, zero and 0.35 rad/s respectively. These are per-tick
increments, not acceleration values per second.

The historical [slew reconstruction](../scripts/dev_action_slew_reconstruction_v1.py)
has nine supported primitives and retains only vx/yaw in its model-facing tape:
five ticks × two channels = **10**, despite stale comments saying 15. Its
internal comparison tape retains all three channels. It reproduces the active
[limiter](../lewm_genesis/lewm_genesis/lewm_contract.py) on all nine primitives
from three synthetic previous-command states. This is not a claim of parity
outside the supported library: the historical helper lacks the active absolute
clipping and has a different unused lateral limit.

The previous applied command is state. For example, reversing from yaw +0.45
to requested −0.45 yields first applied yaw +0.10, whereas starting from reset
yields −0.35. Carry state across ordinary blocks and reset only at an explicit
episode/reset boundary. A hypothetical candidate uses a deterministically
post-slew command plan; actual measured body motion is an outcome, not an action
input. Store requested and applied tapes separately.

`validate_command_match` requires complete, finite `[ticks,3]` tapes with equal
shape and absolute-only numerical tolerance. The historical manifest's `_verify`
uses nested `zip`, so synthetic empty, missing-tick and missing-channel logs can
be marked verified. The new guard rejects each. The ordinary historical log
loader itself indexes five ticks and may reject malformed raw logs before they
reach `_verify`; therefore the reproduced helper defect is **not** evidence that
such traces entered the reported dataset.

## 4. Sensors, validity, and information available at selection time

### Historical proprioceptive predictor

The [manifest assembler](../scripts/build_dev_v03_proprio_action_manifest_v1.py)
builds 15 trailing 10 Hz measurements ending at the image step. The tensor is
three chronological slots × five chronological samples × 30 sensed channels:

| Slice | Meaning and units |
|---|---|
| 0:3 | Dimensionless unit projected gravity, shifted by subtracting `[0,0,-1]`; not z-scored |
| 3:6 | Body angular velocity, rad/s |
| 6:18 | Joint positions, rad |
| 18:30 | Joint velocities, rad/s |

Joint order is FL, FR, RL, RR hips; then the same legs' thighs; then calves.
The declared Unitree adapter order is FR, FL, RR, RL, each leg hip/thigh/calf.
Preserve and explicitly test that permutation against the deployment adapter;
no hardware joint stream was checked here.

Control history is a **separate** 15×2 tensor of `applied[k-1]` vx/yaw, shared by
all experimental cells. It is efference copy, not measured motion. Excluded
channels include absolute pose/yaw, simulator body linear velocity, contacts,
effort and IMU linear acceleration. Exclusion is not proof those modalities are
intrinsically unhelpful; actual availability and measurement quality determine
whether they can enter a future experiment.

The actual assembler passes synthetic tests for chronological construction,
past control, reset-crossing rejection, missing-history rejection, and truncation
of candidate blocks at a future reset. These fixtures inject all three source
loaders with in-memory data; no corpus glob or log read occurs. A new
`validate_causal_history` additionally checks nonnegative integer timestamps,
strict ordering, no future samples, the same `(environment,episode,reset)`
identity, and optionally exact sampling period. The source assembler does not
itself enforce all these timestamp properties. An image-age/staleness limit and
real synchronization tolerance still need a deployment-specific specification.

The [predictor](../scripts/dev_proprio_predictor_v1.py) gates absent slots before
projection, correctly preventing invalid NaNs from leaking through a later
multiplication. But it computes `proprio_in(samples).mean(sample_axis)` without
within-slot time encoding. Reordering five samples cannot change this pooled
feature. A future ordered encoder is a scientifically motivated comparison,
not a demonstrated improvement. The proposed per-channel validity summaries
are more granular than this existing slot mask and are not silently substituted
into it.

`sensor_channel_summary` distinguishes unavailable samples from valid stationary
zeros, one-sample insufficiency, variation and nonfinite values marked valid.
Variation does not prove correct units or useful information; constant values
do not automatically establish a broken sensor. Future physical checks must
excite each relevant degree of freedom and compare against an independent
reference, while recording missingness explicitly.

### Physical handoff qualification

The active frozen ranker receives one current visual token grid, the known
candidate plan, previous applied command and control history. It does **not**
receive sensed proprioception merely because the low-level gait controller uses
it. The runner preserves a 45-element policy observation and command/control
history as snapshot state; this does not add those policy features to the ranker.

The runner's `foot_contact_source="zero"` is a rollout-telemetry placeholder.
Its physical contact labels separately use `robot.get_contacts` at 2 ms physics
steps. The frozen 45-element PPO observation has no contact input. Treat these
three facts separately: placeholder telemetry, privileged contact outcome, and
policy input are not interchangeable sensor channels.

Future model inputs may use only measurements available before selection,
with units, calibration, time and validity. Simulator state, candidate contacts,
stuck labels, successor viability, teacher traces and correct-edge labels remain
training/evaluation-side information. An online target must come from observed
memory or a declared exploration policy; a target from an oracle graph is a
conditional local-control test, not unknown-maze exploration.

## 5. Target semantics and execution endpoint

The frozen ranker's goal projection is
`[dx,dy,sin(atan2(dy,dx)),cos(atan2(dy,dx))]` in the body frame. Its apparent
heading is **bearing to the target**, not desired arrival heading. The real
wrapper passes identical inputs when arrival-heading metadata changes at fixed
dx/dy; the test uses a stub ranker, not a checkpoint. A route tangent recorded
for audit does not provide route intent to this model.

The physical contract already documents consequential distribution shifts:
original ranker fitting used a fixed final-goal vector `[0,-2.66]`, zero previous
command/history, and analytic RGB. Physical qualification uses varied local
targets, physical histories and Genesis RGB. Poor transfer would not identify
JEPA representation failure without separating these changes.

The candidate success helper requires oracle admissibility, correct edge entry,
successor viability and positive port progress, with neither contact nor stuck.
Each requirement is independently exercised by a synthetic negative case. This
tests logical conjunction only, not the correctness of the underlying contact,
crossing or successor labels. Physical fixtures still need directed crossings,
wrong-side approaches, arrival speed/heading and continuation onto a second edge.

For a future reusable controller, specify a relative target region/port, desired
arrival heading where necessary, speed bounds, timeout, stop behavior and
continuation conditions. Train and evaluate on the same semantics. Do not append
new target features to an old frozen checkpoint and claim equivalent inference.

## 6. Diagnostic results and remaining gates

The explicit CPU suite passes **94 tests**: 83 new semantic cases and 11 existing
historical-wrapper compatibility cases. Initial fixture errors were corrected
to unpack the limiter's `(executed,clipped_any)` return and compare wrapped
angles correctly; no frozen source was changed to make tests pass.

| Finding | Status established here | Required disposition |
|---|---|---|
| Wrong legacy geometry schema silently becomes floor-only | Reproduced with actual builder and recording scene; prospective guard rejects | Use guard in a separately versioned future caller; audit affected provenance only within its permitted scope |
| Reflected FRU camera basis | Reproduced from real camera-conversion method with synthetic RGB; SE(3) guard rejects | Label basis explicitly and introduce a versioned proper-frame adapter before geometric fusion |
| Truncated command comparison | Reproduced at helper boundary; strict complete-tape guard rejects | Adopt complete-shape validation in a future data interface; do not infer corrupted historical data |
| Legacy “occupied IoU” | Perfect, sign-inverted and constant predictions all score 1.0 on a constructed case | Exclude as evidence of occupancy/navigation; use independent geometry truth if occupancy is evaluated |
| Directional feature fidelity | Correct, constant and inverted inputs are ordered sensibly by the existing cosine estimator | Retain as a feature metric, not a proxy for executed navigation success |
| Temporal/body-state semantics | Synthetic history/reset checks pass; sample-order loss identified in source | Add ordered temporal encoder comparison and real sensor characterization in future work |
| Physics/render semantic consistency | Analytic intervention passes; historical box submission and physical resize tested separately | Actual Genesis obstacle/render/collision and calibrated angular-target checks remain unperformed |
| End-to-end navigation or JEPA benefit | Not evaluated | Requires the staged controlled experiments in the future plan |

Tests named `known_limitation` assert a reproduced limitation, not scientific
acceptance of it. A green test run must not be reported as all scientific gates
passing. The historical occupancy formula thresholds each prediction by its own
median and uses an all-true reference; its exact score depends on ties and token
count. It is not universally 0.5. The synthetic 1.0 counterexample demonstrates
why that metric is not an independent occupancy measurement.

Reproduce from the repository root using this already available environment:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 \
.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -m pytest \
  -q --tb=short -p no:cacheprovider \
  lewm/tests/test_active_interface_semantics.py \
  lewm/tests/test_go2_textured_v03_renderer.py
```

Use these explicit paths, not blanket suite discovery: some other repository
tests read cached experiment data or perform optimizer steps. This run uses no
GPU, Genesis initialization, checkpoint, training, simulation rollout, protected
benchmark input, experiment output directory, or clean-tree export. Normal
Python imports and reading the named public platform YAML are required.

## 7. Next steps, in order

1. Preserve this source-level package as the interface baseline. In a new,
   separately versioned development harness, adopt complete-tape and causal-time
   checks, choose a proper named camera frame, and replace unsupported metric
   claims. Do not retrofit frozen evidence.
2. Complete physical semantic qualification: known obstacle insertion/removal,
   material-only changes, known-angle projection, motion/IMU/joint alignment,
   command delay, reset and stop tests. Declare scene identities and authority
   before collecting or reading runtime material. Synthetic tests here are not
   substitutes for these checks.
3. Resolve the already frozen September 4 stratified handoff experiment within
   its own authorization and stopping rules. This turn did not launch it or
   inspect runtime outputs. If panel construction is inadequate, model
   performance is still unmeasured. If an adequate panel exists, separate action
   feasibility, target semantics and visual transfer before changing JEPA.
4. Only after reliable local handoffs, run the planned matched current-history
   policy versus JEPA-rollout comparison, add ordered deployment-valid body
   sensing, then test consecutive edges and online memory. Final maze claims
   require independent layouts and maze-level uncertainty, not more frames from
   the same situations.

This is the completed first implementation package, not completion of the
multi-stage scientific programme. Continuing into physical execution requires
the relevant explicit scope and existing repository authority; this document
does not create it.
