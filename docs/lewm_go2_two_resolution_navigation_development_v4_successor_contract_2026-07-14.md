# Go2 two-resolution navigation development V4 successor contract

Date: 2026-07-14

Status: **SOURCE-FREE SUCCESSOR CONTRACT; SOURCE IMPLEMENTATION AND
DIFFERENT-AGENT SOURCE REVIEW ONLY; NO DATA, `.generated`, CHECKPOINT, GPU,
TRAINING, G2, DEVELOPMENT-RUN, RUNTIME, HARDWARE, HELD-OUT, OR PROMOTION
AUTHORITY**

## Result and boundary

This document freezes the additive source closure required to replace the
synthetic-only two-resolution navigation composition with a real post-G2
development controller. It defines the simulator/controller boundary, exact
one-encode rule, tick state machine, learned G4 requirement, target scheduling,
follower, trace, observer, and review closure before implementation begins.

It changes no reviewed predecessor byte and approves no experiment. A source
author may implement only the new versioned files and their synthetic/mock test
sources listed here. After an author handoff, a different agent may inspect the
exact source closure and run bounded CPU-only synthetic/mock tests. No actor may
use this contract to open data or `.generated`, load a checkpoint, use a GPU,
train a head, enter G2, execute Genesis, run a development scene, access a
held-out role, or set a production/runtime identity.

The current V3 integration remains a valid synthetic transaction coordinator.
It is not the real controller: construction is explicitly synthetic-only, its
target evidence originates from a synthetic V5 issuer, and its selected G4
route is not followed by a real external-command episode. V4 is standalone and
must not import, instantiate, retain, wrap, or mutate a V1, V2, or V3
integration engine.

## Governing frozen contracts

The following exact documents govern this successor. Any byte change requires
a dated successor amendment and a fresh different-agent source review.

| Document | SHA-256 |
|---|---|
| `docs/lewm_go2_generalization_execution_contract_2026-07-09.md` | `2a7f3f8f4943c7b4f62dbb09080da9b3fa23dabca9c99023debca13d70da15a8` |
| `docs/lewm_go2_first_principles_plan_corrections_2026-07-11.md` | `b1c5e6087e4956a71cf048cccdd8408384305761a64d9405e08906fd84cc8042` |
| `docs/lewm_go2_scientific_execution_authority_threat_model_2026-07-13.md` | `3fa8954455f88756f975ffa9e91f51bfd76b8c6461d77a171e145b0f5e43dee3` |
| `docs/lewm_go2_g3_qualified_learned_projection_plan_2026-07-13.md` | `daa0b06885a2e9b16e9c79b17417028b617830a20eaaace31ea907bcb008e69f` |
| `docs/lewm_go2_g4_two_resolution_frontier_viewpoint_v2_design_2026-07-13.md` | `de6cb956d97b9187281da948abcf700904969c3f91486e0c5390024fdd4ddc7f` |
| `docs/lewm_go2_g4_two_resolution_frontier_viewpoint_v2_independent_review_2026-07-13.md` | `428602be7b5f878b9f381e4847a96f07dcdb745924ef1f78c6f887f7b5d94f1d` |
| `docs/lewm_go2_g5_reversible_target_belief_implementation_plan_2026-07-12.md` | `c19c0b7f1aa4743b7b0bc79c84f9cbe984fc0fcc971937121df5cd6c3a349173` |
| `docs/lewm_go2_two_resolution_navigation_integration_gap_audit_2026-07-13.md` | `9d521b96befe475a81ad523d2f91cc3804a0cd6e530a85b73b2591c73f2d3d4f` |
| `docs/lewm_go2_two_resolution_navigation_development_integration_v3_independent_review_2026-07-13.md` | `2129e9ca2f58dabb4ca6f569821d8aa3002cc0658564daaa9c2da124a8e379d9` |
| `docs/lewm_go2_heldout_maze_goal_2026-07-14.md` | `2396cb0bbca39488b5c84ec527b38d5b389505d2292b6e7998584042cd93b965` |

The frozen runtime semantics are RGB plus deployment-equivalent odometry,
IMU/proprioception, and executed-command history; post-G2 shared features are
detached; physical evidence lives on the `0.05 m` lattice; planning and target
posterior cells live on the `0.10 m` lattice; and the four semantic colors are
exactly, in canonical order, `red`, `yellow`, `blue`, `green`.

## Retained reviewed source anchors

These are immutable dependencies or semantic controls, not implementation
shortcuts and not runtime authority. V4 successors are additive. Their source
review must rehash every anchor before and after review.

| Exact source | Frozen SHA-256 | Retained role |
|---|---|---|
| `lewm/models/shared_observable_camera_ray_jepa_v5.py` | `b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9` | one shared spatial frame contract; no post-G2 mutation |
| `lewm/planning/revisioned_physical_configuration_memory.py` | `bb05f957e0443e0c1e8405042b97c61948746a66040e84690e12b0a10887d483` | revisioned physical evidence owner |
| `lewm/planning/two_resolution_configuration_projection_v2.py` | `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` | exact current configuration snapshot/component/path authority |
| `lewm/planning/native_learned_physical_projection_v5.py` | `5ccd22e83c83a4c41db11286d31d417fe7af5615ebd7e62e51d7719d5378eca1` | synthetic learned-projection semantic control |
| `lewm/planning/two_resolution_frontier_viewpoint_v2.py` | `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82` | deterministic two-grid candidate semantics |
| `lewm/planning/two_resolution_target_evidence_v1.py` | `f731b848f6b7ced3b07e11d4f9edca81daa8c66f083f9d503ed069809e38a9a2` | synthetic target-evidence semantic control |
| `lewm/planning/two_resolution_reversible_target_belief_v1.py` | `6d17d06718df355893fa7a6f2f1f735fcf835933178e53c554f4d60181ae96c3` | sparse reversible posterior control |
| `lewm/planning/two_resolution_target_router_v2.py` | `c8e071d239d1b9894028752fdc090cc2e1be9273f6f9de5a7c7b4d147741b6d2` | all-mode-safe target routing control |
| `lewm/planning/two_resolution_world_waypoint_adapter_v2.py` | `9b710c6f6044bfefd3fd52bcdbb55a52f890b1fdc6c00629029bbf5a670e8fc1` | exact configuration-cell/world-centre conversion |
| `lewm/planning/two_resolution_navigation_development_integration_v3.py` | `6d8b00aa8ffaa0117efc01baa218cadd299a871732e86d2751e51463520d6523` | atomic rollback and exact-owner control only |
| `lewm/benchmarks/go2_physical_claim_evaluator.py` | `7ea003160ea03da6e989cb76124501b1e7de8571bf8586870b9c8dd7b42f04df` | canonical physical claim decision |
| `lewm/benchmarks/go2_physical_claim_trace.py` | `a41f1fa22f5a90503c82db459ccc9520af334173d416bac0b090308d69cc8fb3` | canonical raw semantic claim trace |
| `lewm/benchmarks/go2_physical_claim_observer.py` | `1db940a49f01313b23c5d37699796b52da776a3a5c88bf3af1381d7d58103e30` | observer-only evaluation and credit |
| `lewm/benchmarks/strict_result_scorer.py` | `d4d4fb6ddff297faaf86e0e1ec9590a35deca2f0f2b0e92fe46dfc31fdd187c2` | strict result-shape control |

The geometry-v2 semantic identity remains
`e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca`.
The directional footprint policy content identity remains
`c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc`.
Neither identity licenses a real runner by itself.

## Additive V4 source closure

The exact implementation closure is frozen below. Every new source hash is
`null` and unresolved until an author handoff freezes final bytes. No alternate
path, generated module, predecessor rename, or dynamically selected plugin may
satisfy a row.

| Required new source | Versioned responsibility | Predecessor/control | Source SHA-256 now |
|---|---|---|---|
| `lewm/models/shared_v5_target_observation_head_v1.py` | detached four-color presence/bearing/range/uncertainty/quality head | Shared V5 output and G5 plan | `null` |
| `lewm/models/two_resolution_frontier_value_head_v1.py` | learned value over an immutable G4 V3 candidate set | G4 contract | `null` |
| `lewm/benchmarks/qualified_shared_v5_navigation_runtime_v1.py` | captured post-G2 frame runner, exact one-inference accounting, tick admission and leases | staged Shared V5 runner semantics | `null` |
| `lewm/planning/native_learned_physical_projection_v6.py` | real runner-outcome-to-physical transaction/retraction bridge | native projection V5 | `null` |
| `lewm/planning/two_resolution_frontier_viewpoint_v3.py` | real runner-derived view admission plus exact learned/baseline candidate freeze | G4 V2 | `null` |
| `lewm/planning/two_resolution_target_evidence_v2.py` | real four-color positive/qualified-negative/abstain issuance | target evidence V1 | `null` |
| `lewm/planning/two_resolution_reversible_target_belief_v2.py` | exact V2-evidence consumer and current `0.10 m` posterior owner | posterior V1 | `null` |
| `lewm/planning/two_resolution_target_router_v3.py` | current V2-posterior, all-mode-safe claim/reacquisition routes | router V2 | `null` |
| `lewm/planning/revision_bound_waypoint_follower_v1.py` | deterministic path/yaw follower and one command-block issuer | G1 follower semantics and waypoint V2 | `null` |
| `lewm/planning/two_resolution_navigation_development_integration_v4.py` | real owner coordinator, scheduler, atomic tick transaction | integration V3 rollback semantics | `null` |
| `lewm/benchmarks/genesis_external_command_episode_v1.py` | fixed external-command physics broker; no controller logic | Genesis only as private physics | `null` |
| `lewm/benchmarks/go2_navigation_development_trace_v1.py` | closed schemas, tick chain, controller trace, actual-open ledger | provenance contract | `null` |
| `lewm/benchmarks/go2_visibility_opportunity_observer_v1.py` | post-seal ground-truth visibility-opportunity observer | G3/G4 opportunity contract | `null` |
| `scripts/execute_go2_two_resolution_navigation_development_v4.py` | captured, fixed-graph, one-shot development launcher | scientific authority threat model | `null` |

The corresponding focused test paths are exactly the same basenames under
`lewm/tests/test_*.py`, plus
`lewm/tests/test_two_resolution_navigation_development_v4_end_to_end.py` for
the complete mock chain and
`lewm/tests/test_go2_navigation_development_v4_source_closure.py` for graph,
path, import, authority, and one-encode checks. Those test files are also new,
additive, unresolved, and production-ineligible.

The author handoff must bind every source/test/document hash, a closed import
graph, and explicit `null` values for every unresolved post-G2 or trained-head
artifact. At minimum these remain unresolved until their separately authorized
stages complete:

- selected post-G2 Shared V5 checkpoint file and model-state hashes;
- passed immutable G2 report and candidate-publication hashes;
- physical calibration and admission-threshold hashes;
- target-head architecture/config/checkpoint/calibration hashes;
- G4-head architecture/config/checkpoint/calibration hashes;
- frozen G4 candidate/baseline configuration hash;
- follower and command-block configuration hashes;
- qualified runtime, controller binding, and source-graph hashes; and
- any development panel, result, robustness, or held-out identity.

No implementation constant may replace `null` with a plausible hash. A later
source-free binding amendment and different-agent review must bind real
artifacts before any such identity is set.

## Simulator-to-controller boundary

The external-command episode is a private physics broker, not a planner. It may
load a scene and collision geometry to render and execute physics, but those
objects and file descriptors never enter the controller process. The fixed
launcher must use a closed IPC schema; passing an arbitrary mapping, object,
callback, module, file handle, shared-memory view, environment variable, or
path is forbidden.

The controller input packet contains exactly:

- one content-committed ego RGB frame under the frozen camera/preprocessing
  contract;
- synchronized timestamp and synchronization identity;
- deployment-equivalent odometry mean/increment and registered covariance;
- registered IMU and proprioceptive measurements;
- the previous requested and actually executed command block, or the frozen
  tick-zero null value; and
- opaque reset/session identities and, only when separately qualified, a
  reset-clearance certificate that reveals no full map.

The packet must not contain, directly or by derivation-friendly alias:

- scene, role, split, family, manifest, evaluator, or scorer identity;
- scene or manifest paths, contents, hashes exposed as a lookup key, or open
  handles;
- world occupancy, planning grid, raster, depth, segmentation, normals, or
  collision geometry;
- exact simulator pose, actual trajectory, privileged velocity, or spawn
  coordinates;
- beacon/object coordinates, vectors, identities beyond the four semantic
  task colors, visibility, occlusion, distance, bearing, claim acceptance, or
  claim credit;
- contact geometry, ground-truth collision/fall/stall labels, reachable-area
  coverage, frontier truth, labels, oracle routes, or evaluator feedback; or
- any held-out/sealed namespace, role selector, comparator result, or retry
  signal.

Scene physics may prevent penetration and produce the next deployment-
equivalent odometry/proprioception. It may clip only to the frozen actuator,
command-duration, and platform envelope; it must not perform manifest-aware
steering, obstacle veto, route repair, or privileged safety control. Requested
and executed blocks are both committed. Actual full-precision pose,
trajectory, contact, collision, fall, and scorer inputs are retained only by
the broker for the post-controller observer.

The controller process may open only frozen source bytes and its bound model,
head, calibration, threshold, geometry-profile, follower, and controller
configuration artifacts. It receives no callback from the observer or
evaluator. The physics broker, controller, observer, and finalizer are fixed
actors in the captured source graph; arbitrary controller plugins are
forbidden.

## Exactly one shared visual inference

For every admitted observation tick, the qualified runtime must:

1. validate the raw input packet and pre-update physical revision;
2. preprocess the RGB frame once;
3. call Shared V5 `forward_frame` exactly once;
4. cause exactly one underlying `VisionEncoder.forward_tokens` call;
5. retain one immutable `SharedFrameOutcomeV1` containing detached patch/BEV
   features and the physical-head output; and
6. issue consumers only exact-object, single-tick leases derived from that
   outcome.

The target head is called once per observation tick as one batch covering the
canonical four colors. It consumes only detached cached features from that
frame. The G4 value head may be called at most once on an exploration tick and
consumes the same already-counted cached features. Neither head may own,
import, accept, resolve, or call an image encoder, `forward_frame`, RGB
preprocessor, RGB tensor, or second visual cache.

The invariant panel is exact:

```text
observation_tick_count
  == shared_frame_outcome_count
  == shared_v5_forward_frame_call_count
  == vision_encoder_forward_tokens_call_count
  == target_four_color_batch_count

g4_value_head_call_count <= observation_tick_count
target_head_owned_encoder_count == 0
g4_head_owned_encoder_count == 0
extra_rgb_decode_or_preprocess_count == 0
```

Any mismatch is a terminal episode fault. Recomputing a frame after an
exception, stale lease, target update, router failure, or G4 decision is a
second inference and is forbidden, not a retry.

## Tick admission and lease boundary

`TickAdmissionReceiptV1` is issued only after one shared frame, one physical
commit, and one fresh two-resolution projection have been validated. It binds
the input packet, pre-update physical state, shared frame outcome, physical
transaction/retraction, post-update physical state, current configuration
snapshot/component, all source/model/calibration identities, and the exact
tick/session/reset.

Only the exact receipt may mint:

- one physical-view lease for G4 V3;
- four color-indexed target-evidence leases in canonical order; and
- at most one cached-feature lease for the learned G4 head.

Leases are non-copyable, non-serializable, exact-object, one-consumer, and
expire at tick commit or fault. A caller cannot provide visible cells, target
distributions, negative domains, candidate sets, revisions, hashes, or
producer identities. Public synthetic issuers remain test-only and must be
structurally impossible to bind to the real runtime.

## Exact tick state machine

An episode and every admitted tick execute in this order. Reordering is a
source-review failure.

1. Before tick zero, construct fresh physical memory, configuration projection
   and planner, view memory, four target memories, router, follower,
   integration owner, action/claim journals, reset capability, and session.
   Every revision is zero and every journal is empty.
2. Receive one allowed simulator packet containing RGB, synchronized
   deployment-equivalent odometry/IMU/proprioception, and prior executed
   command history. No forbidden field is in the controller address space.
3. Run the one Shared V5 `forward_frame` call and freeze the immutable cached
   frame outcome.
4. Run the target observation head once as a four-color batch from the cached
   frame. No target memory mutates yet.
5. Issue the immutable shared observation bound to the pre-update physical
   revision, frame, pose/covariance, calibration, post-G2 checkpoint, G2
   report, thresholds, and raw physical/target outputs.
6. Native projection V6 prepares and atomically commits exactly one physical
   transaction, including any exact retractions. FREE requires complete
   destination-square support under every admitted transform; OCCUPIED uses
   the uncertainty union supercover. Invalid or over-uncertain input fails
   closed.
7. Project the new physical state through the exact two-support morphology and
   issue a fresh current `0.10 m` configuration snapshot, connected confirmed-
   FREE component, frontier artifact, and planner binding.
8. Issue the exact `TickAdmissionReceiptV1` binding the pre-snapshot, shared
   observation, physical commit, and post-snapshot. View and target leases may
   be created only now and only from this receipt.
9. G4 V3 records the exact runner-derived physical visible/swept cells and the
   configuration `(cell, yaw)` view history. No caller-authored cell is
   accepted.
10. In canonical color order, issue exactly one outcome per color:
    `positive`, `qualified_negative`, or `abstain`. Each V2 target memory then
    performs exactly one update-or-age transition and publishes a fresh
    posterior commitment.
11. If the deterministic four-color scheduler selects a current confirmed,
    controller-unattempted target with a current safe router V3 plan, route to
    that claim/reacquisition pose. Otherwise freeze the exact G4 V3 candidate
    set, compute the deterministic information-gain baseline for audit, score
    the same set with the mandatory learned G4 head from cached features, and
    select its route.
12. The deterministic revision-bound follower consumes the exact selected
    path/waypoints and terminal yaw and issues at most one command block for
    the current revisions.
13. A conservative deterministic readiness policy may emit a semantic
    `ClaimIntentV1` for one color. It receives no evaluator result, physical
    acceptance, credit, target coordinate, or ground-truth visibility.
14. The physics broker applies platform-envelope clipping, executes the block,
    and privately records requested/executed commands plus actual trajectory,
    contact, collision, and fall evidence. Only the next allowed odometry,
    IMU/proprioception, RGB, and command-history packet may return to the
    controller.
15. Atomically commit the owner revisions, action/attempt journals, exact-open
    events, and hash-chained `NavigationTickRecordV1`. Expire every tick lease.

The target head's four outputs are predictions, not evidence authority. A
`positive` requires the frozen calibrated presence/quality rule and a finite
bearing/range distribution. A `qualified_negative` additionally requires the
G5 conservative four-physical-child visibility/FREE/detection proof for the
configuration cell. Missing support, occlusion, UNKNOWN, occupied support,
disconnection, stale state, nonfinite output, below-threshold quality, or a
beyond-range/unlocalized result yields `abstain`; it never yields a fabricated
negative. Abstention ages the posterior without erasing unlocalized mass or
setting any mode to zero.

## Revisions, scheduling, claims, and termination

Physical, configuration, view, each color posterior, follower, integration,
action-journal, and tick-chain revisions are separate typed namespaces. An
accepted transition advances its owner by exactly one; an explicit abstention
advances only the color's age/posterior transition; a no-op owner cannot jump a
revision. Every downstream artifact binds exact pre/post revisions. Stale,
copied, reconstructed, foreign-session, foreign-reset, skipped-revision, or
future-revision input rejects before mutation.

All four colors are observed and aged every tick. The controller scheduler is
deterministic:

1. retain the currently locked color while it remains controller-unattempted
   and its exact current plan revalidates;
2. otherwise enumerate controller-unattempted colors with a confirmed live
   hypothesis and valid router V3 plan;
3. sort by claim pose before reacquisition pose, ascending exact route cost,
   descending selected-component posterior mass, canonical color order, then
   plan content hash; and
4. lock the first row. A target plan always takes priority over exploration.

When no row exists, the mandatory learned G4 branch selects exploration. When
the follower reaches the bound terminal pose and the frozen internal readiness
checks pass, the controller may append one claim intent for the locked color,
mark only `controller_attempted`, and clear the lock. There is at most one
semantic claim intent per color per episode. `physical_claim_evaluation` and
`verified_claim_credit` do not exist in controller state. An invalid attempt
is a scored failure; acceptance cannot trigger a retry, route change, target
deletion, threshold change, or next-color choice.

Any exception or failed binding before commit causes exact all-owner rollback,
a stop command if one can safely be issued, and a terminal fault record. An
authoritative episode never retries a failed tick with corrected inputs. A
fall, physics failure, source/hash mismatch, count mismatch, nonfinite command,
lease replay, impossible rollback, or forbidden open seals the episode as a
failure. There is no mid-episode reset, owner replacement, resume, or hidden
tick-budget extension. Development may start a later separately registered
episode only under the preregistered panel protocol; held-out retry remains
forbidden.

A reset is legal only before tick zero. It creates a new reset/session identity
and new owners; no mutable state, evidence, posterior, view history, follower
state, action record, claim attempt, or capability crosses the reset. An
optional trusted reset-clearance certificate must bind only the measured local
clearance support, geometry, pose uncertainty, issuer, and reset. It must not
contain or authorize a scene map, beacon, route, or arbitrary FREE cells.

## Mandatory learned G4 and exact candidate set

Every exploration decision in the learned arm uses
`TwoResolutionFrontierValueHeadV1`. A heuristic, nearest, random, deterministic
information-gain, target, oracle, or last-good fallback cannot select the
action when the learned head fails. Failure or abstention is fail-closed.

For each exploration tick, G4 V3 deterministically generates once the complete
current set of safe reachable `(configuration_cell, yaw)` viewing poses and
frozen scan sequences under the retained V2 semantics:

- goals and routes are in the current confirmed-FREE component;
- the yaw set is the exact frozen 16-heading world-frame set, with any frozen
  scan sequences expanded canonically;
- visibility, sweep, entropy, and discovery use `0.05 m` physical cells;
- route/history/candidate ownership uses `0.10 m` configuration cells;
- rays stop at missing domain or OCCUPIED before admission and at the first
  UNKNOWN group after counting that group;
- every route is an exact current planner-issued path; and
- ordering and tie breaks are canonical and deterministic.

The immutable `CandidateSetV3` records all candidates in canonical order and
binds both frames/revisions, memory/profile/support hashes, view revision,
complete candidate content, baseline configuration, and
`candidate_set_sha256`. The deterministic baseline and learned head receive
the exact same object and row order. Neither may add, remove, merge, regenerate,
reorder, relabel, or alter a candidate. The baseline is computed and recorded
for paired audit only; it cannot steer the learned arm.

The learned head consumes detached cached Shared V5 patch/BEV features plus
the frozen map/candidate features. It emits one finite scalar per exact row.
Selection is maximum learned score, with first canonical candidate as the only
tie break. The record commits the complete baseline and learned score vectors,
selected row, path, and terminal yaw. Candidate-set configuration, value
normalization, objective, labels, optimizer, seeds, early stopping, and DAgger
schedule must be frozen in a later source-free amendment before any learned G4
output. No learned-head training is authorized here.

## Deterministic follower and no-oracle surface

`RevisionBoundWaypointFollowerV1` is a deterministic state machine outside all
learned heads. It accepts only the current exact path/waypoint receipt, terminal
yaw, allowed odometry/IMU/proprioception, previous requested/executed command,
frozen primitive/command registry, geometry profile, timing, and follower
configuration. It produces one bounded command block or a typed stop/abstain.

It revalidates snapshot, component, path, posterior/view state as applicable,
start pose/covariance envelope, route membership, terminal yaw, revisions,
session/reset, and command limits before every block. A new map or target
revision invalidates the old route and requires a new current decision on the
next tick. The follower cannot open or accept a scene, manifest, grid, target
coordinate, exact pose, collision geometry, oracle waypoint, privileged
obstacle query, visibility, evaluator result, coverage, claim acceptance, or
callback. UNKNOWN is never traversable. There is no manifest-grid safety
adapter or simulator-pose fallback.

## Closed evidence schemas

Every schema below is a closed, exact-key, exact-type record with schema name,
integer version, lowercase SHA-256 content commitment, finite canonical
binary64 numeric encoding, and strict canonical serialization. Unknown,
missing, duplicate, nonstring, mapping-subclass, boolean-as-integer, NaN,
infinity, path-escaped, or noncanonical fields reject before any artifact open
or owner mutation.

### `ControllerEpisodeBindingV1`

This immutable pre-episode record contains:

- Shared V5 checkpoint file/model-state, passed G2 report/publication, target
  head checkpoint/config/calibration, G4 head checkpoint/config, physical
  calibration/threshold, geometry/profile, runner/controller/follower config,
  and captured source-graph hashes;
- the exact four semantic colors in canonical order;
- integer tick budget, registered execution seed, reset ID, session ID, and
  authority mode (`synthetic_mock`, `development_candidate`, or a separately
  authorized later mode); and
- canonical content hash.

It must never contain a scene/manifest/object identifier or coordinate, role or
held-out selector, evaluator/scorer identity exposed to the controller, or an
open path/handle. Development and later held-out orchestration bind such
environment identities outside the controller record.

### `ResetReceiptV1`

This record contains the binding hash; fresh reset/session/capability IDs;
fresh physical-memory, configuration-projection/planner, view, four color
memory, router, follower, integration, action-journal, claim-journal, and trace
owner IDs; exact zero revisions; empty-journal hashes; and either `null` or a
qualified reset-clearance-certificate hash. It asserts no reused owner and no
prior state. The optional certificate contains no full map.

### `NavigationTickRecordV1`

The broker-owned union record contains:

- contiguous tick index, exact timestamps/synchronization ID, reset/session,
  and controller-input commitment;
- one-inference receipt and cumulative/per-tick shared-frame, encoder, target-
  batch, G4-head, RGB-decode, and RGB-preprocess counts;
- pre/post physical and configuration revisions/content hashes, physical
  transaction/retraction, current snapshot/component/frontier, and exact
  `TickAdmissionReceiptV1` hashes;
- view admission and pre/post view revisions;
- four ordered target outcome kinds, evidence/abstention receipts, pre/post
  posterior revisions, posterior/component hashes, and ages;
- locked-color/scheduler rows, target route or exact G4 candidate-set hash,
  complete baseline/learned score commitments, selected row/path, and terminal
  yaw;
- waypoint and follower receipts, requested/executed command blocks,
  platform-envelope clipping, and action source;
- optional semantic claim intent and controller-attempt journal revision;
- controller fault/stall state and separately inaccessible broker-side
  execution/fall commitments; and
- previous tick-chain hash, current content hash, and current chain hash.

The controller is never given the broker-side fields. Full actual poses,
collision truth, fall truth, visibility, coverage, and evaluator decisions are
not embedded; they remain observer payloads committed by hash only.

### `ActualOpenLedgerV1`

Each append-only row contains exact actor, phase, role, no-follow canonical
path, expected hash, actual hash, monotonically increasing sequence, previous
row hash, row hash, and access disposition. Actors are the fixed launcher,
physics broker, controller, observer, evaluator, and finalizer only.

The controller may open only its frozen source/model/head/calibration/config
closure. The physics broker may open environment/renderer assets but cannot
pass their paths, handles, metadata, or contents through controller IPC.
Ground-truth observer/evaluator opens occur only after the controller trace is
sealed. Evaluator access count visible to the controller is exactly zero. Every
unexpected, duplicate, mismatched, symlinked, escaped, wrong-phase, or wrong-
actor open is terminal.

### `ControllerTraceV1`

This immutable controller-stage artifact contains the episode binding, reset
receipt, complete contiguous tick chain, semantic claim intents, action-source
counts, final owner revisions, terminal controller status, one-inference
equalities, zero evaluator/callback access, actual-open-ledger controller
projection, and final content/chain hashes.

It excludes canonical coverage, true visibility, object/beacon coordinates,
claim acceptance/credit, ground-truth collision/fall, reachable area, exact
simulator trajectory, and strict score. It is sealed before an evaluator or
visibility observer opens ground truth and is byte-identical when referenced by
every later observer artifact.

### `ObserverResultV1`

This post-controller artifact contains the exact sealed controller-trace hash,
observer/finalizer/source/evaluator/scorer hashes, full-precision broker pose
and trajectory rows, semantic intent attachment, canonical physical claim
trace/evaluation/credit commitments, per-color visibility opportunities,
strict result, actual-open-ledger suffix, one-shot scoring flag, and content
hash. It is never returned to the controller.

Strict episode rows have contiguous unique tick indices, full-precision
`pre_xy` and `pose_after_xy`, `ticks_used == len(tick_log)`, exact booleans for
`collision`, `stalled`, and `hard_stalled`, exact `final_xy`, and a bit-identical
canonical claim trace. Observer finalization invokes the canonical physical
claim trace/evaluator/observer chain exactly once and strict scoring exactly
once. A held-out mode, if separately authorized later, additionally requires
the immutable one-shot flag to be true. Missing/crashed/faulted episodes remain
explicit failed rows; they are never omitted.

## Visibility-opportunity observer

`Go2VisibilityOpportunityObserverV1` is outside the controller and runs only
after `ControllerTraceV1` is sealed. It consumes the broker's full-precision
actual pose/camera trace, immutable physical manifest, exact camera/render
calibration, semantic task-object registry, and zero-inflation geometry.

For each tick and each canonical color, a valid opportunity requires the
correct unique task object to project inside the actual camera frustum, meet
the pre-frozen minimum visible support, and be first-surface physically visible
after walls, obstacles, distractors, and other beacons. It records distance and
bearing diagnostics but does not require the controller to report a sighting.
Suppressing a target prediction, observation, or claim therefore cannot reduce
the opportunity denominator. The per-episode gate asks whether each of the
four colors had at least one valid opportunity; G4 requires `96/96` across the
24 development scenes.

Observer records bind episode/tick/color, manifest/task/camera/geometry hashes,
actual pose commitment, frustum/support/occlusion decision, and content hash.
No observer object, row, count, boolean, callback, ledger handle, or exception
is visible to the controller during the episode. The observer may score an
already sealed trace; it cannot amend a tick, create a controller claim, choose
an action, update a target memory, or trigger a retry.

## Ordered implementation and synthetic review gates

All source work is additive. Before any checkpoint or learned-head training,
the order is:

1. implement closed trace schemas, canonical hashing, actual-open ledger,
   reset freshness, and exact source-graph checks;
2. implement the qualified runtime and lease lifecycle with a fake frozen
   Shared V5 frame source, proving the one-encode equalities and zero encoder
   ownership in both auxiliary heads;
3. implement target and G4 head architectures with deterministic mock weights
   only, then prove shape, detachment, four-color batching, finite-score, and
   no-RGB/no-encoder contracts;
4. implement native projection V6, target evidence V2, target posterior V2,
   G4 V3, router V3, and follower V1 against exact synthetic runner outcomes;
5. implement integration V4 and exhaustively inject faults before and after
   every owner mutation, lease, journal, record, and chain boundary, proving
   exact rollback and terminal authoritative-run behavior;
6. implement the external-command broker and test it only with a mock physics
   transport, proving the closed IPC surface and that privileged broker fields
   cannot reach controller objects or trace payloads;
7. compose the complete CPU-only mock episode for all four colors, target and
   exploration branches, abstentions, stale revisions, reset/fault/fall paths,
   one-inference accounting, and post-seal observer isolation; and
8. freeze an author handoff, then have a different agent rehash and
   adversarially review every exact byte and retained predecessor.

Synthetic/mock review must cover ordinary imports, preloaded module
substitution, source replacement, copy/deep-copy/serialization/reconstruction,
mapping subclasses, extra/missing/nonstring keys, hash-shaped caller identity,
lease replay, wrong color/order, stale/cross-reset/cross-session revisions,
second encoder/RGB calls, caller-authored cells/distributions/candidates,
oracle/manifest/scene/evaluator injection, observer feedback, route through
UNKNOWN or any target mode, scene-aware broker clipping, and partial rollback.

CPU review uses at most six worker processes, one thread per numerical runtime,
disabled external pytest plugins, and blank HIP/CUDA/ROCr visibility. It must
not open data, `.generated`, checkpoints, model outputs, G2, development or
held-out roles, nor execute Genesis. Passing synthetic tests proves source
closure only.

Only after a different-agent PASS and a separately reviewed binding amendment
may the project, in its governing order:

1. bind the passed post-G2 checkpoint/calibration and run a payload-free
   preflight;
2. build/train/calibrate the detached target head on authorized train-only,
   scene-disjoint labels;
3. freeze G4 candidates/baseline/labels and train the detached learned G4 head
   with scene-disjoint oracle future gain and later DAgger;
4. run separately authorized synthetic-checkpoint, exact-map/oracle, fast, and
   full development gates; and
5. proceed to robustness, freeze, fresh opaque custody, and one held-out run
   only after every earlier gate passes.

None of those five later steps is authorized by this document.

## Source-review decision rule

An author candidate is a `BLOCK` unless all new source/test bytes and this
contract are exact-hashed, every retained anchor is unchanged, the import graph
is closed, production/runtime identities are `None`, and all authority flags
are exact false. The different-agent reviewer must explicitly verify the one-
encode call graph, closed IPC schema, exact tick order, lease issuance, four-
color scheduler, learned G4 mandatory path, deterministic follower, rollback,
trace/ledger schemas, observer-only boundary, unresolved-artifact nulls, and
absence of predecessor integration or synthetic issuers from the real path.

A PASS may say only: **source-ready for a separately authorized post-G2
binding step**. It cannot say navigation-ready, development-ready, runtime-
ready, production-ready, benchmark-ready, or held-out-ready.
