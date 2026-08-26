# JEPA Local Waypoint Planning Cost Qualification V1

Date: 2026-08-26

Experiment: `JEPA_LOCAL_WAYPOINT_PLANNING_COST_QUALIFICATION_V1`

Status: prospective executable-contract scaffold; no inference or scientific
materialisation has run

Mode: `EVALUATION_FIRST_SINGLE_SEED`, development-only

## Scientific question and claim boundary

This experiment asks whether an untuned distance from a frozen candidate JEPA
latent state to one frozen local-waypoint goal-view latent provides useful local
route ordering on the already-frozen Route-Intent V2 panel. It separately tests
true-future latent cost, an autoregressively unrolled one-step checkpoint, and an
identically unrolled rollout-trained checkpoint.

This is not a learned safety head, a deployment-safety qualification, a new
maze-solving result, or a closed-loop navigation experiment. Oracle contact and
two-ply viability define evaluation populations only. True-future tokens are an
upper bound and are not available to a planning-time controller. One development
seed and one previously observed held-out role do not support a generalisation
claim.

The exact historical renderer has a prospectively disclosed structural-schema
defect. Its caller supplies `genesis_scene.json`, where structural geometry is
under `objects`, while `scripts.render_replay_v03.build_scene` reads only
`walls`, `obstacles` and `landmarks`. Every current, true-future and goal frame
is therefore floor-plane-only: no wall, obstacle or landmark entity is added,
and there is no direct structural visual evidence. This defect is preserved
because the frozen current RGB/token authority and true-future targets used the
same path, and current RGB/token reproduction remains an exact byte-match gate.
Results qualify only historical-renderer latent route ranking. They cannot
establish explicit wall visual reasoning or interpret contact downranking as
visual wall avoidance.

The requirements boundary remains exactly:

> Deployment hard-contact requirements, consequences and recovery criteria
> remain unresolved. No further deployment-safety scope reduction, sensor
> qualification or learned hard-safety model is authorised.

`REQUIREMENTS_ACQUISITION_REQUIRED` remains a separate next decision.

The authoritative protected-scope result is
`docs/lewm_protected_contact_scope_requirements_review_v1_result.json`, file
SHA-256 `c348d5e2d265a118922ae138c549d8a6e48e4e900a45b70d919de4aa530e4027`,
content digest `148a1757f4b8d55291ab38010a2dd0701e4606d61cd4c656de78cff571dac948`,
at result/source commit `b29eae1929725a4cc26a35d95662b545daee4553`.
Its eleven preserved classifications are:
`PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED`,
`SIMULATED_CONTACT_PROXY_SCOPE_ONLY`,
`DEPLOYMENT_MATERIAL_HAZARD_SCOPE_UNRESOLVED`,
`PERSON_AND_FRAGILE_ASSET_HAZARDS_NOT_REPRESENTED`,
`RECOVERABILITY_REQUIREMENTS_UNRESOLVED`,
`MISSION_PROGRESS_REQUIREMENTS_PRESENT`,
`MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO`,
`SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO`,
`SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO`,
`GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`, and
`REPLANNING_INTERFACE_UNRESOLVED`.

## Pre-freeze custody disclosure

The outcome barrier is: no route-outcome table, row ledger, result metric,
checkpoint tensor, or predictor inference may derive or tune the contract.
Three accidental terminal exposures are disclosed and excluded:

1. The contract-audit command
   `sed -n 1,180p docs/lewm_safe_local_waypoint_planner_route_intent_v2_result_2026-08-20.md`
   exposed a detailed held-out-state table. No exposed value is retained or
   used.
2. A broad requirements-classification `rg` command matched the one-line
   `docs/lewm_protected_contact_scope_requirements_review_v1_result.json` and
   emitted a truncated portion containing diagnostic outcome values. No exposed
   value is retained or used in this experiment.
3. Root's static-source search with pattern
   `commit|prefix|H3|execute|candidate bank|replan` over
   `docs/lewm_safe_local_waypoint*` surfaced one aggregate line in
   `docs/lewm_safe_local_waypoint_planner_route_intent_v2_result_2026-08-20.md`
   containing unsafe/safe H3 branch/state counts. Token cost, goal view and
   paired materiality had already been specified independently; the values are
   nevertheless excluded from every contract and gate decision.

All final thresholds, rules, goal construction, materiality, classification
precedence and evidence schemas come from user instructions or static metadata,
not these exposures. The preexecution receipt must reproduce this disclosure and
assert zero outcome values used for contract derivation, zero checkpoint tensors
opened before freeze, zero inference calls, zero G2 reads and zero training
steps.

## Failed first attempt and prospective goal-view amendment

The original freeze commit
`184e192f35740a2b300a771097c2c5c8d68ce4f9` remains an immutable ancestor.
Its first hidden execution failed closed during CPU materialisation at
`purpose-18: frozen path[2] goal cell is not free and reachable`. The untouched
attempt is archived at
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.jepa_local_waypoint_planning_cost_qualification_v1.failed-1787772621443348948-337116`.
Its failure receipt is 43,448 bytes, file SHA-256
`63aad5c1d1a2e8902088cff647465be7c17ef56b4d12705b42ee0135cdd925a0`
and content digest
`fb1b810fff46781243c0d0d36e4d3520789dd045f4f27fa6f353f43da51dbe64`.
The 41-file, 1,365,155-byte archive inventory is bound by canonical-record
aggregate SHA-256
`6442c13416dc1badd811ac4b19a0fd802fad4708c3698c7dcfada44d552469bb`.

That attempt published no canonical output, state index, oracle fanout, GPU
inference receipt, evidence ledger, aggregate metric, gate, classification or
result. Its three partial context images and every other partial file remain
archived and are non-reusable. No automatic retry or phase/shard resume is
permitted. The next attempt must start from a fresh empty hidden namespace after
a new prospective correction commit and fresh preflight.

The correction is based only on frozen state/scene metadata and source
semantics, with zero route-outcome rows, checkpoint tensors or predictor calls
used. Across all 48 fixed states, path[2] is endpoint-reachable in all 48 but
nav-blocked in 14: 13 are deliberately reachable beacon endpoints and one is a
low-clearance transit-blocked cell; the other 34 are unblocked. A SceneGraph
blocked goal may be reached as an endpoint without being free or transit-safe.
The amendment is frozen in
`docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_goal_view_amendment_2026-08-26.json`.

## Failed second attempt and prospective current-token amendment

The goal-view correction source freeze
`e81ef67763d1daeff9be6cef32ad031465bc57e8` also remains an immutable
ancestor. Its fresh hidden execution completed the CPU materialisation and
rendered-current RGB reproduction, then failed closed in GPU materialisation
before true-future copying, predictor tensor deserialization or predictor
inference. The untouched attempt
is archived at
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.jepa_local_waypoint_planning_cost_qualification_v1.failed-1787774675034819483-355288`.
The failure receipt is 1,552 bytes, file SHA-256
`48d77d46b91924cd97c9c064619261905763d8e73a21af5f945a7e00f76e2ba2`
and content digest
`3cd263684ab000934f776ce1591582ffbd2d80179371d5982790c2f8b9037d9c`.
Its complete 537-file, 322,350,756-byte inventory is bound by canonical-record
aggregate SHA-256
`b82763e28831814a48120a84aa0d209aa837f2911551dd27fc1b5738d325e71d`.
It contains no true-future or predicted latent directory, tensor index, GPU
inference receipt, evidence ledger, aggregate metric, gate, classification,
report or result. No file or phase from it is reusable.

All 48 newly rendered current RGB files matched their bound authority exactly.
Both predictor checkpoint files were SHA-hashed as pre-inference custody, but
zero predictor checkpoint tensors were deserialized and zero predictor calls
ran. All 48 newly encoded current token grids failed byte equality before any
predictor tensor load or inference. The historical current tokens were encoded inside
the bound 7,154-frame historical cohort; the failed attempt introduced a new
192-frame mixed context/goal cohort. BF16 output bytes can depend on device,
runtime, kernels and the complete cohort. The historical cohort/order and exact
token bytes are persisted and bound; only bitwise re-execution equivalence under
the changed cohort was unproven. Re-encoding is therefore unnecessary and is
not a current-token authority.

For diagnostic custody only, archived and authoritative FP16 grids were cast to
float64. Flattened cosine is the float64 dot product divided by the product of
float64 norms; token-mean cosine applies that operation independently to every
width-1024 token then takes a float64 mean. Across 48 states the flattened range
was `[0.9999043258031383, 0.9999475581155232]` with mean
`0.9999380251025588`; token-mean range was
`[0.9999058101016667, 0.9999499438888875]` with NumPy float64 mean
`0.9999401111347598`. Float64 per-state `mean(abs(archived-authority))` had
range `[0.012470918548312207, 0.017233230190110287]` and mean
`0.013817019145041817`; float64
`sqrt(mean(square(archived-authority)))` had range
`[0.01791116887331469, 0.024418504702133508]` and mean
`0.019687519709368287`. These are input-compatibility diagnostics only and do
not alter any scientific gate, cost, metric or classification.

The prospective authority rule retains the exact 48/48 current-RGB gate, then
validates each bound historical raw `.f16` current array, then writes exactly one
canonical attempt-local NPY per state whose C-order FP16 array payload bytes are
identical to the authority. The NPY container has its own file SHA-256. That one
attempt-local payload backs both logical context slot 2 and logical `CURRENT`;
their tensor-index records must alias the same NPY path, SHA-256 and byte count.
Current-token re-encoding, tolerance and post-hoc numeric acceptance are
forbidden. Encode only the 96 missing context slots 0/1 and 48 goal frames: 144
new frames in exactly nine batches of 16. Logical tensor counts remain 144
`CONTEXT`, 48 `CURRENT`, 48 `GOAL`, and 1,728 for each of `TRUE_FUTURE`,
`ONE_STEP_PREDICTED` and `TWO_STEP_PREDICTED`, totaling 5,424. This amendment is
frozen in
`docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_current_token_amendment_2026-08-26.json`.

## Frozen panel

Use exactly the existing Route-Intent V2 identities:

- 48 states, 12 candidates per state and 576 candidate rows;
- 32 fit, 8 calibration and 8 held-out states;
- four families: `large_enclosed_maze`, `medium_enclosed_maze`,
  `small_enclosed_maze`, and `loop_alias_stress`;
- H1, H2 and H3 endpoints;
- state manifest SHA-256
  `da67309c073f60d74e4b85427237b19691552a542136e6ddb95939f14b4c5c37`;
- split SHA-256
  `ebef7db828a4c754432375818fd6b1eff0731cc3bc546ff2b69667b03abe56a8`;
- branch ledger SHA-256
  `9b25b227c3e4de11e68e4abee454c4251399fafb468458a4e0d65f89bc6cdf7c`;
- route-intent labels SHA-256
  `e8d33671502f717426836ec9a1039d445558b81e636ae105a81e2113151a8b69`;
- data-audit SHA-256
  `73381d7dc834813286b52f571a6b5d3370d04582fb5bf27beeee9447e5e4fd92`.

No state, role, candidate, endpoint, action, route outcome, contact definition or
label may change. No fresh panel is collected. The held-out role is the primary
gate role but is explicitly previously observed, development-only and
non-independent. Fit, calibration, held-out and all-48 summaries are
descriptive. There is no threshold fitting, model selection or calibration.

## Frozen contact and two-ply population materialisation

One decision block comprises five 100 ms commands and 250 two-millisecond
physics frames. Immediate contact is any frozen disallowed robot-environment
contact during that first block/H1. H2/H3 contact is descriptive and must never
define `ORACLE_CONTACT_FREE`.

The unchanged 12-candidate bank induces nine unique first-block primitives:
hold, forward slow/medium/fast, backward, yaw left/right and arc left/right.
For every state, materialise all nine current blocks and all 9×9 next blocks,
including successors of contacting current prefixes. Candidate viability is:

1. its mapped current H1 prefix is contact-free; and
2. its exact successor has at least one contact-free next primitive.

Candidates with the same first primitive share the exact successor identity.
Persist `[9,250]` current and `[9,9,250]` successor contact bitsets, first-event
and link attribution, exact snapshot/action mappings, safe-next counts and the
12-to-9 map. Expected materialisation is 48×(9+81)=4,320 blocks and 1,080,000
physics frames.

The 9×9 fanout is unconditional. After each current H1 block, capture a
`RAW_CONTINUATION_AFTER_CURRENT_H1` evaluation snapshot containing the complete
solver, controller, harness, RNG and counter state. This capture preserves the
frozen snapshot digest construction but does not invoke the canonical-boundary
guard: fall, tip and out-of-bounds flags are permitted and persisted, while a
NaN fails closed. Production reset checks remain suppressed. The identical raw
snapshot is restored for all nine next primitives. This defines the requested
counterfactual safe-next count even after terminal H1 flags; it is not a
deployable replanning boundary. Persist nine snapshot digests, flags, capture
identities and nine-restore assertions per state.

Reconstructing each frozen state necessarily replays 40 production blocks with
the existing collector scheduler/RouteTeacher and frozen PPO. This is
`FROZEN_STATE_RECONSTRUCTION_REPLAY`, not an evaluated navigation system: it is
used only to reproduce the already frozen post-block-40 identity, may not alter
candidate or state selection, and must pass exact branch snapshot, current RGB
and H1-contact cross-checks. Across 48 states this is 1,920 reconstruction
blocks and 480,000 physics frames. Together with the 4,320 fanout blocks and
1,080,000 fanout frames, the exact simulator total is 6,240 blocks and
1,560,000 physics frames, with 48 snapshot reproductions. The final report may
state that no navigation system was trained, evaluated or used as the
experimental planner, but must not claim unqualifiedly that no navigation code
executed.

The frozen PPO/controller is invoked for all 6,240 simulator blocks: 1,920
RouteTeacher reconstruction-prefix blocks (480,000 physics frames) and 4,320
fixed open-loop oracle-fanout blocks (1,080,000 physics frames). The latter use
the prospectively fixed primitive identities, not a JEPA/MPC/navigation
candidate selector. The exact custody classification is
`FROZEN_CONTROLLER_REPLAY_AND_FIXED_ORACLE_FANOUT_ONLY`; experimental
candidate-selecting JEPA/MPC/navigation-planner executions remain exactly zero.

Every candidate row distinguishes `immediate_contact_h1`, integer
`successor_safe_action_count` in `[0,9]`, `successor_viable` (count >0), and
`oracle_viability_admissible` (no H1 contact and successor viable). A selected
nonviable-successor event means `not successor_viable`; it never means merely
`not oracle_viability_admissible`, so immediate contact and successor viability
remain separate.

The new H1 rows must match the bound structured raw-contact index
`.generated/contact_hazard_ontology_and_instrumentation_v1/raw_contact_event_index.json`,
SHA-256 `1eac5be90b48e88cac7aa8db7f3ce3bd6655e1404f7ba6591ac90afa9e1f0d4d`,
content digest `b0897c9fbc1e739495a0b1184ede11639d6932e6a5fa37761efc246f8b45d610`.
It contains 48 states, 576 branches, 750 physics steps per branch at 2 ms and
reports 48 passing states with no mismatch. No raw point cloud is required.

The realised H1/H2/H3 route and descriptive stuck fields are bound to
the existing `dense_replay/<state_id>.json` files below
`.generated/dense_temporal_true_future_safety_observability_v1`. Before any row
reduction or outcome JSON parse, construct the 48 repo-relative
`{state_id,path,sha256,bytes}` records in frozen state-manifest order. Their
compact sorted-key canonical JSON without terminal LF must contain 10,397 bytes,
sum to 10,155,856 source bytes and have SHA-256
`729271c8d8535d2f02433de03306ea8d113c654ecfb9f0671e6bd4e67f11183f`.
Only after that prospective byte binding passes may the self-digesting input
index parse and validate the 48 exact state files and the evidence receipt (SHA-256
`a547ac544a869a6ef75a4798b22875291e55604f9e53ceaea24a790db09df7e1`,
1,484 bytes). For every state, recompute its canonical `content_digest` after
removing only that field and require schema
`dense_route_intent_true_future_state_v1`, status `PASS`, 12 branches, H3 tick
count 15 and boundaries `[5,10,15]`. Missing, duplicate, drifting or invalid
state evidence fails before reduction. Descriptive contact remains bound to the
exact contact-event authority above. This dense authority supplies frozen outcomes
only after contract freeze; it never selects a goal, population, threshold,
candidate or contract rule.

## Context and action contract

The predictor input is exactly:

- context tokens shaped `[3,768,1024]`;
- three raw post-slew action blocks shaped `[3,5,2]`, each flattened tick-major
  to ten active `vx/yaw_rate` dimensions;
- initial control history shaped `[3,5,2]`;
- `vy` is inert in the frozen corpus and forbidden in evaluation.

For every one of the 576 frozen candidate rows, persist the requested command
tape `[3,5,3]` and the applied/post-slew tape `[3,5,3]` separately and require
exact equality to the branch ledger. The active `[3,5,2]` and flattened
`[3,10]` predictor actions are projections of the applied tape only. Requested
and applied actions may never be conflated.

The historical offsets `[-480,-240,0]` are renderer/source-frame offsets, not
physics ticks. With 48 source frames per 10 Hz command timestep, they correspond
to command-tick offsets `[-10,-5,0]`, elapsed times `[-1.0,-0.5,0.0]` seconds,
and deterministic post-warmup endpoints after blocks 38, 39 and 40. Never
conflate them with the 250 physics frames in an H1 contact block.

Training used `applied[k-1]`. The exact 15-command control span is block 37 tick
5, all of blocks 38 and 39, and block 40 ticks 1–4, reshaped as:

- slot 0: b37t5, b38t1–4;
- slot 1: b38t5, b39t1–4;
- slot 2: b39t5, b40t1–4.

Validate exact indices and timestamps. A duplicated/fabricated context or using
blocks 38/39/40 directly fails the state before inference. The snapshot after
block 40 must equal the unique branch-ledger snapshot digest shared by all 12
rows and the qualified current replay. The manifest snapshot digest is a
descriptive audit only: it exists for 22/48 states and includes historical
mismatches, so absence or mismatch cannot drop or replace a state.

The bound dense token index contains exactly one current-view occurrence for
each of all 48 state IDs. Newly rendered post-block40/current RGB must match its
existing current RGB byte SHA exactly. Validate the bound raw `.f16` token
SHA/shape/dtype, load it as a C-order FP16 array, and write one canonical
attempt-local NPY whose array payload bytes are identical. The NPY container has
its own file SHA; container-format bytes are not compared with the raw `.f16`
file. There is no current-token re-encoding, numeric tolerance or alternate
view. That one copied payload is both predictor context slot 3 and `CURRENT`
cost through two logical records that alias the same path/SHA/bytes. Any required
state reconstruction failure aborts the whole qualification before inference;
no state is silently omitted or replaced.

Normalise the initial observed control history with the frozen
`control_mean/control_std` receipt. Preserve the checkpoint's surprising mixed
scale exactly: `P.unroll` appends the raw
`control_slot_from_action(action_blocks[step-1])` to the already-normalised
window. Do not silently repair it.

## Checkpoints and rollout

Use seed `2026080901` only:

- one-step checkpoint SHA-256
  `20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a`;
- rollout checkpoint SHA-256
  `75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4`;
- encoder checkpoint SHA-256
  `7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6`.

Both predicted sources use the identical frozen
`P.unroll(..., max_h=3)` autoregressive path. `ONE_STEP_PREDICTED` unrolls the
one-step-trained weights autoregressively; `TWO_STEP_PREDICTED` unrolls the
rollout-trained weights identically. Each feeds its own prediction back into the
sliding window. Neither gets independent true contexts, future observations or
future latents. Use frozen FP32 weights under BF16 autocast.

The GPU receipt must bind the exact model configuration, strict state-dict
loads, `eval` plus `inference_mode`, all `requires_grad=false`, no optimizer,
zero training steps, encoder/predictor call counts, no future-input fields, and
identical parameter-state digests before and after inference for each model.
Digest encoder, predictor parameters and buffers from the complete sorted
`state_dict` under namespace `JEPA_LOCAL_WAYPOINT_PARAMETER_STATE_V1\0`, with
length-prefixed key, dtype, shape and contiguous CPU C-order tensor bytes.
Batching is frozen because BF16 kernels can depend on batch shape and cohort.
Encode only 144 new records—96 context slots 0/1 and 48 goals—in ascending RGB
SHA-256 order, then kind, numeric state identity and context slot, at batch size
16 (nine batches). Copy each bound historical current payload exactly once;
context slot 2 and `CURRENT` logically alias it. Current and true-future tokens
are not re-encoded. For each checkpoint, process one numeric-ordered state at a time as
one exact 12-candidate batch and unroll H1→H2→H3. Run the one-step checkpoint
before the rollout checkpoint: 48 unroll calls and 144 internal model forwards
per checkpoint. A dynamic OOM batch fallback is forbidden and fails closed.

Preserve only the prior aggregate findings: rollout improved direct future
fidelity H1–H4; improved selected action-specific metrics most strongly H3–H4;
H1 action retrieval remained weak/imprecise; planning utility was not
previously evaluated; the eight-seed experiment is not rerun.

## Deterministic goal view

Construct exactly one candidate-independent goal view per state:

- world x/y is the exact frozen SceneGraph cell centre of
  `waypoint_path_cells[2]`;
- z is the replayed frozen start/base z;
- roll and pitch are zero;
- yaw is `atan2` from the centre of path cell 0 to path cell 1.

Require path length at least three, valid node identities, consecutive frozen
route edges, and endpoint reachability from path[0] to path[2] under the frozen
SceneGraph nav-blocked endpoint semantics. Do not require path[2] to be free.
Persist per state whether it is nav-blocked and classify it exactly as
`UNBLOCKED`, `BEACON_ENDPOINT`, or `LOW_CLEARANCE_TRANSIT_BLOCKED`, with beacon
precedence. The complete index must reproduce 48 endpoint-reachable, 14 blocked,
13 beacon-endpoint, one low-clearance and 34 unblocked states.

This goal image is a
`VIRTUAL_COUNTERFACTUAL_GOAL_VIEW_NOT_A_PHYSICALLY_EXECUTABLE_SENSOR_POSE`.
It does not claim that the robot, camera or physical range sensor can occupy or
reach the pose, and it does not establish stopping or transit safety. In
particular, some path[2] coordinates lie inside a landmark footprint, while the
preserved historical renderer is floor-plane-only. This limitation is reported,
not repaired.

If optional manifest `waypoint_xy` or `waypoint_body_xy` exists, audit exact
equality; absence is permitted. Persist world pose and body-frame delta x/y plus
sin/cos of wrapped goal-yaw minus start-yaw, matching the existing local
waypoint `g_t`. Render with the frozen no-robot textured-v03 renderer/camera and
the same crop, preprocessing and encoder as context/targets. A missing,
non-finite, endpoint-unreachable, unrenderable or unencodable goal fails the
state before model inference. Substituting path[1], searching a standoff or
alternate goal, or dropping/replacing a state is forbidden.

## Cost and sources

Sources are exactly `TRUE_FUTURE`, `ONE_STEP_PREDICTED` and
`TWO_STEP_PREDICTED`.

Persist every context, current, goal, true-future and predicted token grid as
float16 with path, SHA-256, bytes, shape and identity. The GPU process only
encodes, predicts and persists. The canonical NumPy CPU reducer reloads each
float16 grid as float32, reapplies `T.normalise` (`layer_norm` over width 1024,
default epsilon 1e-5), L2-normalises each token with epsilon 1e-12, forms 768
aligned-token float32 dot products, clips numerical excursions to the
mathematical cosine range `[-1,1]`, and computes:

`C_h = mean_t(1 - cosine(candidate_h[t], goal[t]))`

The final mean uses a float64 accumulator. The sole scientific scalar authority
is the source-closed NumPy CPU function
`jepa_local_waypoint_planning_cost_metrics_v1.tokenwise_normalized_cosine_mean_cost`;
all row writing and terminal replay call that exact executable. The stdlib
contract helper is fixture-only and cannot author scientific scalars. There is
no token matching, pooling,
search, learned weight or tuned weight. H3 is primary. Current, H1 and H2 costs
and current→H1→H2→H3 monotonic deltas are diagnostics only. Persist a separate
per-candidate current→H3 nonincrease Boolean and strict-decrease Boolean plus
their aggregate fractions; these endpoint diagnostics are distinct from the
fraction whose complete CURRENT→H1→H2→H3 path is monotonic. A synthetic
Torch-versus-NumPy fixture must agree within absolute tolerance `1e-6`; the
NumPy CPU result is authoritative.

The encoder implementation is frozen to constructor
`vjepa2_1_vit_large_384` from the clean local V-JEPA 2 repository at commit
`204698b45b3712590f06245fbfba32d3be539812`; preflight and terminal custody
bind the repository path/commit/clean state and `src/hub/backbones.py` SHA-256.

Each candidate row also persists realised p_d, p_theta, completion and
descriptive contact/stuck separately at H1, H2 and H3. Horizon diagnostics
compare each cost only with outcomes at the same horizon. H3 aliases remain the
sole primary ranking authority.

## Populations and comparators

Populations are exactly:

- `ALL_CANDIDATES`;
- `ORACLE_CONTACT_FREE` (H1 immediate contact-free only);
- `ORACLE_VIABILITY_ADMISSIBLE` (H1 contact-free plus at least one safe next
  primitive).

Require viability ⊆ H1-contact-free ⊆ all. An empty oracle population abstains;
it never borrows a candidate.

Comparators are exactly `KINEMATIC_ROUTE_BASELINE`, `RANDOM`,
`TRUE_FUTURE_LATENT_COST`, `ONE_STEP_PREDICTED_LATENT_COST`, and
`TWO_STEP_PREDICTED_LATENT_COST`.

The kinematic baseline integrates all 15 `[vx,vy,yaw_rate]` commands in the
first three post-slew blocks with 0.1 s Euler steps and the existing full vy
formula (frozen vy is zero). It derives nominal p_d and p_theta only. Its order
is max p_d, retain values within 0.03 m,
then max p_theta, then candidate index; it has no 5° heading deadband.
It does not invent a nominal completion threshold; realised completion remains
the frozen route-label field used by the realised route authority.

Random ranks ascending SHA-256 of the frozen `RANDOM_V1` namespace, unsigned
64-bit big-endian seed, UTF-8 state ID and unsigned 32-bit big-endian candidate
index separated exactly as the contract specifies, then candidate index.

## Route authority and metrics

Realised route preference is H3 route-only: completed first, p_d with a 0.03 m
indifference margin, p_theta with a 5° indifference margin, then candidate index
only for deterministic total order. Contact, successor viability and stuck do
not enter route preference or utility; populations filter them and selected
outcomes report them separately.

Exclude oracle-unordered pairs. A predicted cost tie within `1e-12` earns 0.5
pairwise credit; selection ties use the lowest candidate index. Spearman is
`scipy.stats.spearmanr(-cost, continuous realised p_d)` with midranks. Kendall
is tau-b on the same values. Constant/undefined values are null and fail an
applicable gate. Combined route utility is the weight-free within-state
margin-Borda fraction `(wins + 0.5*unordered_or_ties)/(N-1)`.

Report Spearman, Kendall, pairwise accuracy, top-1/top-3, MRR, mean best rank,
cost spread/ties, selected identity, immediate H1 contact, descriptive H2/H3
contact, nonviability, stuck, p_d, p_theta, Borda utility, normalised regret,
completion and abstention. Report pooled, per-state, per-family, per-role and
all-48 summaries.

Aggregate Spearman and Kendall tau-b as the arithmetic mean of finite per-state
coefficients. Aggregate pairwise accuracy as pooled ordered-pair correct credit
divided by the pooled valid ordered-pair denominator. Apply the same reductions
within each family. Persist every per-state coefficient, credit and denominator
so these values reproduce exactly.

Per-state normalised regret is `(best p_d - selected p_d)/(max p_d-min p_d)`;
if range ≤`1e-12`, use zero only if selected equals best, otherwise fail. Mean
regret is over nonabstaining states. Progress gates use the signed ratio
`sum(selected p_d)/max(abs(sum(oracle-best p_d)),1e-9)`.

Family collapse requires at least one evaluable nonabstaining state and one
ordered pair. A family is collapsed when none is positive: pairwise >0.5,
best-route top-3 >0, or selected p_d sum >0. All four families must avoid
collapse.

## Gates

The true-future gate is on held-out `ORACLE_VIABILITY_ADMISSIBLE` and requires:

- pairwise ≥0.70;
- Spearman ≥0.60;
- mean normalised regret ≤0.25;
- best-route top-3 ≥0.75;
- signed selected-progress ratio to oracle best ≥0.80;
- no family complete collapse.

Report its separate gate classification exactly as
`TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL` on pass or
`TRUE_FUTURE_LATENT_GOAL_COST_NO_GO` on failure. This is separate from the
exactly-one primary experiment classification.

For each predicted source, report a base-preservation screen requiring pairwise
≥0.65, regret ≤0.30, progress ≥0.75 of true-future selected progress and no
family collapse. The full two-step gate additionally requires the true-future
gate; no more H1 contact/nonviable selections than one-step under
`ALL_CANDIDATES`; strictly improved pairwise; improved regret or progress; and
no family collapse. Do not weaken any gate.
On full pass, also report `TWO_STEP_JEPA_PLANNING_COST_SIGNAL`; on failure report
the Boolean and no predicted-gate signal string.

## Paired materiality

Use 10,000 state-bootstrap replicates with seed `2026080901`. State order is
frozen manifest order. Each resample index comes from the frozen SHA-256
`STATE_BOOTSTRAP_V1` byte rule, with replacement. Use deterministic Type-7
linear 2.5%/97.5% quantiles. These intervals are descriptive, not hypothesis
tests or generalisation evidence.

`JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS` requires, on
`ORACLE_VIABILITY_ADMISSIBLE`, any one of:

- two-step minus kinematic selected progress ≥0.05 m and CI lower >0;
- kinematic minus two-step normalised regret ≥0.05 and CI lower >0;
- equal-family mean selected-progress ratio over `large_enclosed_maze` and
  `loop_alias_stress` improves by ≥0.10 and paired CI lower >0.

Every trigger also requires no H1-contact-selection increase, no
nonviable-successor-selection increase, and neither hard family collapsed.

Persist an explicit 32-row paired-effect ledger: eight held-out states times
the four prospectively frozen `ORACLE_VIABILITY_ADMISSIBLE` comparisons
`TWO_STEP_MINUS_ONE_STEP`, `TWO_STEP_MINUS_KINEMATIC`,
`TWO_STEP_MINUS_TRUE_FUTURE`, and `TRUE_FUTURE_MINUS_KINEMATIC`. Each row binds
state/family/role, both selected identities, both raw progress/regret/
pairwise-accuracy/best-route-rank values, progress and pairwise left-minus-right
effects, regret and rank right-minus-left improvements, and contact/nonviable
selection deltas. The aggregate paired comparisons and bootstrap must reproduce
from these rows.

## Classification and next decision

Use exactly one primary, with precedence:

1. true-future gate fails → `RAW_LATENT_GOAL_COST_NO_GO`;
2. otherwise, full two-step gate passes, kinematics is materially superior and
   the incremental secondary is false → `KINEMATIC_BASELINE_DOMINANT`;
3. otherwise, full two-step gate passes →
   `TWO_STEP_JEPA_PLANNING_COST_SIGNAL`;
4. otherwise → `TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO`.

For the last class, say the complete two-step qualification failed. Persist
both predicted base screens and flags for both screens failed,
`ONE_STEP_BASE_SCREEN_ONLY`, and two-step base passed/full gate failed. Do not
claim both absolute screens failed unless the corresponding Boolean is true.
These are diagnostic flags, not extra classifications.

Next mapping, specification only:

- two-step signal → `ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1`;
- raw-latent no-go or true-future-signal/predictor no-go →
  `PLAN_AWARE_MONOTONE_JEPA_COST_V1`;
- kinematic dominance → `NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1`.

The conditional specifications are also frozen. Oracle-admissible closed-loop
MPC retains the fixed bank, executes only a short committed prefix, reobserves
and replans under exact oracle admissibility, and reports contact, successor
viability, route progress and abstention; CEM/MPPI is deferred to a later
separate contract. Plan-aware monotone cost is route-only local-waypoint and
future-trajectory pairwise/listwise progress with rollout consistency, never
safety, completion, aggregate utility or material-contact prediction. The
non-greedy local-subgoal experiment allows a temporary move away from the
direct route under oracle admissibility and uses no topological memory. No next
experiment runs in this pass.

## Environment, storage and evidence

CPU replay/render/oracle uses
`.generated/venvs/genesis_render_vulkan/bin/python`: Python 3.12.3, Genesis
0.3.14, NumPy 2.4.6, SciPy 1.17.1, Torch 2.12.0+cu130, Pillow 11.3.0 and
PyYAML 6.0.3 with CUDA unavailable;
use exactly `os.cpu_count()` workers, expected 32. Encoder/predictor inference
uses `/home/andrewknowles/TinyQuadJEPA/bin/python`: Python 3.12.3, Torch
2.10.0.dev20250926+rocm6.3, NumPy 2.4.2, SciPy 1.18.0, Pillow 12.1.0 and
PyYAML 6.0.3 on `cuda:0`, AMD
Radeon AI PRO R9700. Genesis installed in the inference environment must not be
used for simulation. Persist exact interpreter hashes and import receipts.
Both configured virtual-environment launchers must resolve to the prospectively
bound `/usr/bin/python3.12` binary, SHA-256
`1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`,
8,020,928 bytes. CPU preflight, GPU preflight, GPU materialisation/check and the
terminal checker must validate this binding before scientific imports or any
checkpoint tensor access.

For Torch, NumPy, SciPy, Pillow and PyYAML in both interpreters, freeze the
distribution name, exact version and resolved package root. At preflight and
terminal check, `importlib.util.find_spec` origin, every submodule search
location and the live module `__file__` must resolve inside that frozen root;
any `PYTHONPATH` or other import shadow fails closed. This is deliberately an
interpreter/version/root/import-resolution boundary rather than recursive byte
closure: same-version mutation of these foundational packages remains a stated
residual limitation. Do not recursively close further transitive or system
libraries. Genesis, rsl-rl and tensordict remain separately byte-closed through
their exact distribution `RECORD` inventories as specified below.
The GPU materialisation child must recompute these five foundational package
bindings, require exact equality to the persisted GPU-environment receipt and
bind that receipt into its inference receipt before any checkpoint is opened.

Before CPU materialisation, write the self-digesting
`materialization/cpu_runtime_input_inventory.json`. It binds the tracked Go2
platform manifest (SHA-256
`5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189`),
the primitive registry (SHA-256
`cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8`),
the frozen locomotion model (SHA-256
`e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff`)
and configuration (SHA-256
`bc3e68c18252475199e57b30c8ac49d813e3c784a3983e0e8b1a762490dde24f`),
and the installed Genesis Go2 URDF (SHA-256
`4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4`).
It also hashes the seven referenced visual meshes: `base.dae`, `calf.dae`,
`calf_mirror.dae`, `foot.dae`, `hip.dae`, `thigh.dae`, and
`thigh_mirror.dae`.
It records exact package roots and versions for Genesis 0.3.14, rsl-rl-lib
5.4.1 and tensordict 0.13.0.
For each distribution it also binds the exact `.dist-info/RECORD` path,
SHA-256 and bytes, validates every RECORD-declared digest and size, and hashes
every presently listed actual file—including unhashed rows—into one sorted
canonical aggregate. The exact absent-unhashed path set is also digest-bound.
Run CPU processes with `PYTHONDONTWRITEBYTECODE=1` so optional pyc presence
cannot drift between preflight and terminal validation.

Before parsing any scene, build 96 repo-relative byte records in frozen
state-manifest order, with `manifest` then `genesis_scene` for each state. The
records contain only `{state_id,scene_id,kind,path,sha256,bytes}`; their compact
sorted-key canonical JSON without terminal LF must contain 30,690 bytes, sum to
5,361,073 source bytes and have SHA-256
`87f989d37b23f974c0c98924b65f676ebc23dcc24c6a3cfc70b087e339a09c4a`.
The same inventory then parses all 48 `manifest.json` and all 48
`genesis_scene.json` files only after that byte binding passes. It recomputes each
canonical `manifest_sha256`, and requires agreement across the state identity,
canonical manifest and Genesis export. `LEWM_TEXTURE_ROOT` must be absent, so
the exact tracked `assets/textures` root is used; all 12 JPG paths, hashes and
byte counts are persisted. Because the frozen structural-schema defect never
reaches `add_box`, `.generated/box_meshes` is not a scientific runtime input:
zero OBJ files may be opened or used, and no mesh is regenerated during
scientific execution. The 176 scene-derivable cache names may be recorded only
as a non-executed diagnostic inventory, never as rendered geometry. Missing or
mismatching required runtime inputs fail before materialisation. The inventory reads no route,
contact, viability or held-out outcome field and is revalidated immediately
before and after CPU materialisation and at terminal check.

Output root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/jepa_local_waypoint_planning_cost_qualification_v1`.
Require ≥20 GB workspace free and ≥50 GB output free. Temporary ceiling is 20
GB; final ceiling is 12 GB. Stop before materialisation if the conservative
estimate exceeds either ceiling.

Persist self-digesting indices for context reconstruction, the 48 frozen dense
route-replay inputs, goal views, all 5,424
context/current/goal/true/predicted float16 tensor records, and all 48 oracle
9×9 fanout shards. Persist complete gzip JSONL candidate, selection and paired-
effect rows. Terminal validation must hash every shard, validate bytes/shape/dtype,
reload every float16 grid, reproduce every cost with the canonical NumPy CPU
reducer, then
reproduce every aggregate/gate/classification from scalar rows without
inference or checkpoint access.

Freeze operational watchdogs prospectively: each CPU worker has a 3,600 s
no-progress timeout and the complete CPU materialisation phase has a 10,800 s
global timeout; GPU preflight and terminal check each use 600 s, while the GPU
scientific materialisation child uses 10,800 s. Persist the exact configuration
in preexecution and successful CPU/GPU/terminal status in the phase/result
receipts. Any timeout terminates and then kills experiment children, archives
the untouched hidden attempt, performs no automatic retry, and permits no phase
or shard resume.

Launch all 32 CPU state workers with `MALLOC_ARENA_MAX=1`. This is frozen only
as a glibc allocator-fragmentation mitigation after the failed attempt; it does
not change the simulator, state, action, render, oracle or scientific reduction
semantics. The worker count remains exactly 32, dynamic worker fallback is
forbidden, and preexecution plus terminal custody must validate the exact worker
environment.

A failed hidden attempt namespace is archived untouched with a self-digesting
failure receipt. A later execution starts from a new empty hidden namespace and
reuses no scientific phase or shard; partial evidence is never resumed or
copied into canonical output. The canonical root remains absent until the full
attempt passes, then the complete namespace is published by one same-filesystem
atomic rename.

The terminal persistence manifest hashes every noncircular artifact, including
`report.md`. It excludes exactly its own receipt, the independently
self-digesting `result.json`, and the removed running marker. Every phase receipt
binds the source-freeze commit, contract digest and output-schema digest.

GPU child stdout and stderr must never again be discarded. For `PREFLIGHT` and
`MATERIALIZE`, invoke with non-raising subprocess status capture and atomically
persist full stdout, full stderr and a self-digesting execution receipt on
success, nonzero exit or timeout before returning or raising. The preexecution
receipt binds the completed preflight child receipt. The child-produced GPU
inference receipt binds neither parent execution receipt; parent-produced result
and persistence receipts bind both preflight and materialise receipts after the
materialise child exits. A successful terminal `CHECK` may not
mutate the finalized manifest: capture it only in ephemeral non-output storage
and remove that capture on success. On check failure, bind its captured streams
and metadata only in the outer failure archive/receipt.

## Commit and stop rules

The original freeze and goal-view correction commits remain ancestors because
they bind the two archived failed attempts. Make a separate prospective
current-token contract-correction commit before a new fresh preflight; do not
amend or orphan either ancestor and do not reuse a prior phase or shard. The
final result commit message remains exactly
`Evaluate JEPA local waypoint planning cost qualification`.

Stop after row evidence, aggregates, gates, classifications, runtime/storage
and result custody. Do not train or select a model; collect a panel; read G2;
change contact scope, roles, states, actions or labels; run closed-loop MPC;
implement memory, navigation, routing or beacon capture; or make a deployment
safety claim.
