# Go2 canonical physical-claim evaluator binding

Date: 2026-07-11

Status: frozen before implementation, caller migration, new claim output, or
the canonical-claim oracle regression.

## In-place amendment: 2026-07-11 adversarial trace closure

This dated amendment was frozen before implementation. It closes trace-global
ambiguities found in adversarial review and is authoritative over any earlier
wording in this document. The canonical API is now one trace-level, two-pass
evaluation. Trace identity, exact task membership, event order, and duplicate
event IDs are resolved before any final event decision or content hash exists.
A single-event geometry routine may exist only as a private helper over an
already prevalidated event; it is noncanonical and must not be imported by a
caller.

The pre-amendment document SHA-256 was
`3d8cce6d59da37bbdc52a8499f23296d0ad6e368b394468179bfb4c844e8be34`.

The amendment also freezes exact task-set commitment, unconditional legacy
unverifiability, strict typed-reference shapes, manifest-compatible JSON
serialization, all-reason precedence, exact slab arithmetic, one-to-one strict
scorer parity, evaluator-to-controller feedback rejection, and a 96-attempt
oracle policy. The body of this document has been updated consistently with
those resolutions.

## Authority and purpose

This binding is the G0 prerequisite for every later Go2 task-success claim. It
implements the active execution contract,
`docs/lewm_go2_generalization_execution_contract_2026-07-09.md`, SHA-256
`0ba3a0699868da4edf7fdd36e57ff5a7cbd151e694970c58fbcce425137ad678`,
and the reviewed G3-G5 gap audit,
`docs/lewm_go2_g3_g4_g5_first_principles_gap_audit_2026-07-11.md`, SHA-256
`a6fd3d6c4c51c57b60470b0b6ef15e2e8554654c7b3777e70b34088a10054329`.

The evaluator is ground-truth measurement, not a controller input and not a
learned head. One shared implementation must determine whether a declared
claim is physically valid. The strict scorer, oracle, physical-eligibility
audit, runtime verification, batch aggregation, and G5 evaluation may adapt
their inputs to it, but may not duplicate or weaken its geometry.

This document authorizes implementation and development-only regression. It
does not authorize held-out, G2, sealed, model-output, runtime-promotion, or
fresh-seed access.

## Existing incompatibilities

The pre-binding implementations disagree materially:

- `generalization_protocol.strict_ground_truth_claim` takes a caller-supplied
  LOS boolean and evaluates only inclusive distance plus LOS. It has no
  requested-versus-claimed identity or yaw.
- `strict_result_scorer` resolves one target reference, has no event yaw, and
  can use a prior rounded `post_xy`. Its LOS includes walls, obstacles, and
  other landmarks, and separately includes distractors, but target exclusion
  is a 1 mm center tolerance rather than exact object identity.
- `go2_oracle_positive_control` checks distance, LOS, and bearing, but its LOS
  is a sampled zero-inflation occupancy grid configured not to treat any
  landmark as an obstacle. Its persisted claim poses are rounded to four
  decimal places.
- `go2_physical_eligibility` checks distance and `SceneGraph` LOS for a
  reachable center but does not require a valid terminal heading and does not
  include distractors in that LOS path.
- `benchmark_go2_memory_closed_loop` records controller proxy claims and
  rounded pose fields; `check_go2_generalized_suite` trusts claimed colors
  rather than shared physical verification.

Historical distance-only, distance-plus-LOS, color-proxy, and rounded-pose
counts remain diagnostics. None becomes a canonical physical claim by
conversion, tolerance, or favorable distance from a threshold.

## Canonical module and schemas

The new pure implementation must live in
`lewm/benchmarks/go2_physical_claim_evaluator.py`. It must not import a
benchmark driver, learned model, renderer, checkpoint, or dataset module.

It exposes one authoritative trace evaluator:

```text
evaluate_physical_claim_trace(
    trace,
    physical_manifest,
    expected_task_object_ids,
    expected_task_object_set_sha256,
)
```

The implementation performs two mandatory passes. Pass 1 validates the trace
envelope, exact task-set commitment, raw event structure, canonical order, and
the complete multiset of event IDs. Pass 2 evaluates geometry, gathers every
applicable reason, assigns duplicate credit, and only then creates final
per-event and trace hashes. No public single-event evaluator or caller-built
accumulator is permitted. A private `_evaluate_prevalidated_event` helper may
be used only by pass 2 and returns no canonical hash by itself.

The event-result schema is
`lewm_go2_physical_claim_evaluation_v1`. The scene aggregate schema is
`lewm_go2_physical_claim_summary_v1`. The raw runtime trace schema is
`lewm_go2_claim_trace_v1`; the authoritative returned trace schema is
`lewm_go2_evaluated_claim_trace_v1`.

Every commitment and output uses bytes from exactly:

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("utf-8")
```

This is byte-compatible with the repository's existing `manifest_sha256`
serialization for every prevalidated finite manifest while explicitly
rejecting NaN and infinity. The physical-manifest SHA-256 must equal
`lewm_worlds.manifest.manifest_sha256(manifest)` and a recomputation with the
explicit serialization above. A content SHA-256 omits only its own
`content_sha256` field; no other field is omitted.

JSON binary64 numbers retain the serializer's exact `-0.0` spelling. Pose
hex strings and packed bytes preserve signed zero bit-for-bit. No hashing path
normalizes `-0.0` to `0.0`. Fixed-key JSON objects rely on `sort_keys=True`;
all object-ID lists are independently sorted by exact UTF-8 encoded bytes
before serialization, so derived identity and blocker ordering is
deterministic. The physical-manifest hash still binds the exact list order in
`manifest.to_dict()` as the existing function does; reordering a manifest list
therefore changes the manifest and every downstream hash that binds it.

## Exact trace and event input

The raw trace envelope contains exactly these nine keys:

1. `schema`, exactly `lewm_go2_claim_trace_v1`;
2. `trace_id`, a nonempty string unique to this run;
3. `episode_id`, a nonempty string unique within the run;
4. `scene_id`, exactly equal to the physical manifest scene ID;
5. `physical_manifest_sha256`;
6. `task_object_ids`, the complete task-object set in exact UTF-8 byte sort;
7. `task_object_set_sha256`;
8. `controller_claim_attempts`, the raw ordered event list;
9. `evaluator_feedback_to_controller`, exactly an empty list.

The finalized returned trace contains exactly the same identity, task, attempt,
and feedback fields, changes `schema` to
`lewm_go2_evaluated_claim_trace_v1`, and adds exactly
`physical_claim_evaluations`, `physical_claim_summary`, and
`trace_content_sha256`. The last field is omitted from its own hash.

No extra or missing envelope key is accepted. For V4 and every current maze,
the complete task set is exactly all manifest landmark object IDs. More
generally it must equal the task-object set preregistered by the benchmark
before episode execution; the trace cannot choose a subset. Every task ID must
be a nonempty exact manifest landmark object ID and the list must be unique.
The suite finalizer also requires every `trace_id` to be unique and every
`(trace_id,episode_id,scene_id)` tuple to occur exactly once; it never merges
episodes or accepts a later trace as a replacement.

Every field named `*_sha256` must be a string of exactly 64 lowercase ASCII
hexadecimal characters. Identifiers are exact nonempty strings: no whitespace
trim, Unicode normalization, case folding, or numeric coercion is applied
unless the `legacy_alias` rule explicitly says otherwise.

The task commitment is SHA-256 of the canonical serialization of exactly:

```json
{
  "schema":"lewm_go2_claim_task_set_v1",
  "scene_id":"<exact scene ID>",
  "physical_manifest_sha256":"<exact manifest SHA-256>",
  "task_object_ids":["<exact UTF-8-byte-sorted IDs>"]
}
```

The evaluator receives the independently bound expected task commitment and
requires exact equality among the benchmark binding, manifest objects, trace
list, and trace commitment. It never derives the expected task set solely from
the untrusted trace.

Each canonical event contains exactly the following semantic inputs:

1. `trace_id`, `episode_id`, and `scene_id`, each exactly matching the envelope.
2. `event_id`: a nonempty attempt ID unique within the episode.
3. `tick`: a non-boolean integer in `[0, 2^63-1]`.
4. `event_index`: a non-boolean integer in `[0, 2^63-1]`, strictly increasing
   in trace order and used to break same-tick ties.
5. `requested_target`: a typed target reference.
6. `claimed_target`: an independently supplied typed target reference.
7. `robot_pose_world_xy_yaw`: the three binary64 values `(x,y,yaw)` at the
   exact claim-evaluation instant.
8. `pose_binary64_le_sha256`: SHA-256 of `struct.pack("<3d", x, y, yaw)`.
9. `pose_hex`: the three exact Python `float.hex()` strings, used for offline
   bit-exact reconstruction.
10. `pose_provenance`: exactly `runtime_full_precision`,
    `oracle_full_precision`, `eligibility_candidate_full_precision`, or a
    `legacy_*` value that can never yield canonical acceptance.
11. `physical_manifest_sha256`: the canonical physical manifest content ID.

The exact event key set is:

```text
trace_id, episode_id, scene_id, event_id, tick, event_index,
requested_target, claimed_target, robot_pose_world_xy_yaw,
pose_binary64_le_sha256, pose_hex, pose_provenance,
physical_manifest_sha256
```

There are exactly 13 keys and no extras. Unknown keys, missing keys, a
non-string identifier, a boolean integer, or a value with the wrong JSON type
is malformed and is gathered as an unverifiable reason. Pass 1 requires the
raw list already to be in strictly increasing
`(tick,event_index,event_id)` order; it never silently sorts. It sees all IDs
before pass 2, so every occurrence of a duplicated event ID receives the same
duplicate-ID unverifiable reason and no occurrence can be hashed as accepted.

The evaluator derives target position and all physical geometry from the
manifest. Caller-supplied target position, distance, bearing, LOS, probability,
controller confidence, proxy acceptance, map state, or rounded display value
is ignored and must not appear as evaluator truth input.

The authoritative pose is the reconstructed binary64 triple from `pose_hex`.
The JSON numeric triple must round-trip to the same bits and its packed-byte
hash must match. A mismatch makes the event unverifiable. Booleans are not
numbers. Every coordinate, yaw, manifest center, size, and box yaw must be
finite. Every physical box XY size must be strictly positive.

The evaluation instant is after the tick's executed motion and authoritative
pose read, and before any controller-declared or evaluator-verified claimed
set is mutated. The same immutable pose object is passed to the evaluator and
trace writer.

## Target-reference and identity resolution

A typed target reference is a JSON object with exactly two keys:

```json
{"namespace":"object_id|task_color|legacy_alias","value":"nonempty string"}
```

Modern runtime, oracle, and eligibility callers may use only `object_id` or
`task_color`:

- `object_id` matches one manifest landmark `object_id` byte-for-byte. There
  is no trimming, case folding, prefix removal, or substring matching.
- `task_color` is one lowercase ASCII member of `red`, `green`, `blue`, or
  `yellow`. It resolves only when exactly one manifest landmark has the exact
  case-folded material ID `landmark_<color>`. Zero matches is unresolved and
  multiple matches is ambiguous. A repeated-color scene must use `object_id`.

`legacy_alias` exists only in the offline legacy adapter. After surrounding
ASCII whitespace is removed and the value is case-folded, its candidate set is
the union of exact case-folded landmark object IDs, exact case-folded material
IDs, material IDs with one leading `landmark_` removed, and an exact registered
color token derived by the `task_color` rule. It resolves only when that union
contains one landmark. Ambiguity is never broken by manifest order.

The keys must be the exact lowercase strings `namespace` and `value`; no extra
key is permitted. Both values must be strings and `value` must be nonempty.
The namespace value is case-sensitive and must be exactly `object_id`,
`task_color`, or `legacy_alias`. An unknown namespace, wrong key set, wrong
type, empty value, extra field, or `legacy_alias` under non-legacy provenance
is a malformed-reference unverifiable reason, not unresolved identity.
Conversely, modern namespaces under legacy provenance do not make that event
canonical: every `legacy_*` provenance is unconditionally unverifiable even
if all fields and factors could otherwise pass.

Requested and claimed references are resolved independently. The identity
factor passes only when both resolve uniquely to the same exact landmark
object ID and both resolved IDs belong to the trace's committed task set. Two
aliases that merely share a color string do not establish identity unless both
independently resolve uniquely.

Manifest landmark object IDs must be nonempty and unique. All occluder object
IDs across walls, obstacles, landmarks, and visual distractors must also be
globally unique so blocker provenance is unambiguous. Duplicate IDs,
non-finite geometry, non-positive XY size, a missing requested or claimed
reference, or an unresolved/ambiguous reference makes the event unverifiable;
the implementation must not choose the first record.

When both identities resolve but differ, `identity_passes=false` and the event
is physically rejected. A resolved requested or claimed identity outside the
committed task set also makes `identity_passes=false` and is rejected. Thus an
accepted non-task landmark is impossible and can never receive task credit.
Distance, LOS, and bearing are still measured to the resolved claimed
landmark for diagnostics. When the claimed identity does not resolve, those
three factors are null.

## Frozen physical factors

The thresholds are literals in the evaluator, not caller arguments:

```text
claim_distance_m = 1.20
claim_absolute_bearing_rad = 0.25
line_of_sight_inflation_m = 0.0
```

Any migrated geometry contract must retain `claim_radius_m=1.20` and require
physical LOS. A mismatch fails that caller before evaluation. Geometry v2
predates a bearing field and is not rewritten for the regression: this binding
is the authoritative `0.25 rad` source, and the evaluator-contract hash is
bound beside the unchanged geometry-v2 hash. A future geometry schema may add
the bearing field only at the same literal value. No epsilon, learned
calibration, family override, color override, or legacy tolerance may change
these factors.

### Distance

Let `(rx,ry)` be the full-precision robot position and `(tx,ty)` the claimed
landmark's manifest center. Compute:

```text
dx = tx-rx
dy = ty-ry
distance_m = hypot(dx,dy)
distance_passes = distance_m <= 1.20
```

The boundary is inclusive. `nextafter(1.20,+infinity)` is outside. Distance is
to the landmark center, not its surface, a grid cell, a rounded trace point, or
a predicted target point.

### Wrapped bearing

Compute:

```text
target_world_bearing = atan2(dy,dx)
signed_bearing_error = atan2(
    sin(target_world_bearing-yaw),
    cos(target_world_bearing-yaw))
absolute_bearing_error = abs(signed_bearing_error)
bearing_passes = absolute_bearing_error <= 0.25
```

The boundary is inclusive and there is no tolerance. Arbitrary finite input
yaw is accepted and wrapped only by this formula. The zero-distance bearing is
the platform `atan2(0,0)` result and remains subject to the same formula; no
special favorable heading is invented.

All distance and bearing intermediates, including `dx`, `dy`, distance,
target bearing, sine, cosine, signed error, and absolute error, must be finite.
Otherwise their factors are null and `physical_computation_nonfinite` applies.

### Zero-inflation physical line of sight

LOS is the 2-D closed segment from `(rx,ry)` to `(tx,ty)` against exact
manifest XY oriented rectangles. The occluder set, in canonical order, is:

1. walls, sorted by exact object ID;
2. obstacles, sorted by exact object ID;
3. visual-randomization distractor objects, sorted by exact object ID;
4. every landmark other than the exact resolved claimed object, sorted by
   exact object ID.

Only the exact claimed landmark record is excluded. No landmark is excluded by
color, material, center proximity, or a 1 mm endpoint tolerance. Thus other
beacons are physical occluders. The target's own rectangle is not tested.

Every occluder object ID must be a nonempty string and all IDs in the complete
occluder-plus-target inventory must be unique before LOS starts. For one box
with center `(cx,cy)`, yaw `q`, and positive sizes `(sx,sy)`, use Python
binary64 scalar `math` operations in exactly this order; vectorized, fused,
fast-math, reordered, or float32 implementations are forbidden:

```text
c = cos(-q)
s = sin(-q)
x0 = c*(rx-cx) - s*(ry-cy)
y0 = s*(rx-cx) + c*(ry-cy)
x1 = c*(tx-cx) - s*(ty-cy)
y1 = s*(tx-cx) + c*(ty-cy)
dx_local = x1-x0
dy_local = y1-y0
hx = sx/2.0
hy = sy/2.0
t_enter = 0.0
t_exit = 1.0
```

Process X and then Y. For axis values `(p,d,h)`, if `d == 0.0`, the axis is
disjoint only when `p < -h or p > h`; otherwise that axis leaves the interval
unchanged. If `d != 0.0`, compute in order
`a=(-h-p)/d`, `b=(h-p)/d`, swap only when `a>b`, then
`t_enter=max(t_enter,a)` followed by `t_exit=min(t_exit,b)`. The box is
disjoint as soon as `t_enter>t_exit`; otherwise it blocks after both axes.

Every listed intermediate must be finite. A nonfinite manifest field produces
`manifest_invalid_physical_geometry`; a nonfinite intermediate computed from
otherwise finite inputs produces `physical_computation_nonfinite`. Either
makes the event unverifiable and is never treated as a clear line. The
comparisons are closed, so any nonempty
intersection at `t` in `[0,1]`, including tangency or an endpoint inside a
non-target box, blocks LOS. There is no grid sampling, body inflation, margin,
erosion, clearance allowance, or parallel epsilon.

All intersecting blockers are recorded as ordered
`(collection,object_id)` identities. `line_of_sight_passes` is true only when
that list is empty. A zero-length segment is tested as one point against every
non-target box rather than being accepted early.

This is deliberately a 2-D physical claim contract because its only pose input
is `(x,y,yaw)` and the maze task has no traversable vertical separation. It is
not the camera first-surface visibility contract used to train observations.

## Decision and reason semantics

Each factor is exactly `true`, `false`, or `null`:

- `identity_passes`;
- `distance_passes`;
- `line_of_sight_passes`;
- `bearing_passes`.

The final `decision` is one of `accepted`, `rejected`, or `unverifiable`.
`accepted` is true only when `decision=accepted`; it is always false otherwise.

- `unverifiable`: any required input, precision commitment, manifest
  invariant, or identity resolution is missing or invalid. A null required
  factor also forces this state.
- `rejected`: inputs are verifiable and at least one factor is false.
- `accepted`: all inputs are verifiable and all four factors are true.

Pass 2 evaluates every reason predicate without short-circuiting and gathers
every applicable unique reason in the following immutable order. If any
unverifiable reason applies, `decision=unverifiable`, `accepted=false`, and
`rejection_reasons=[]` even when an available physical factor is false. Only
when the unverifiable list is empty are all rejection predicates gathered. An
empty rejection list then means `accepted`; a nonempty list means `rejected`.
No caller or finalizer may infer a different precedence.

Trace-global reasons 1 through 12 are propagated to every event; envelope,
scene, manifest, task-set, feedback, or order failure makes every event
unverifiable. Reasons 13 and 14 apply to the malformed or mismatched event.
Duplicate-ID reason 15 applies to every occurrence of each duplicated or
invalid ID. Pass 2 may compute diagnostic geometry where inputs permit, but it
cannot remove a pass-1 reason.

Unverifiable reasons:

1. `trace_schema_or_key_set_invalid`
2. `trace_id_missing_or_invalid`
3. `episode_id_missing_or_invalid`
4. `scene_manifest_identity_mismatch`
5. `physical_manifest_commitment_mismatch`
6. `task_object_ids_not_exact_sorted_unique`
7. `task_object_set_mismatch`
8. `task_object_commitment_mismatch`
9. `evaluator_feedback_to_controller_nonempty`
10. `trace_event_order_invalid`
11. `manifest_duplicate_object_id`
12. `manifest_invalid_physical_geometry`
13. `event_key_set_or_type_invalid`
14. `event_trace_identity_mismatch`
15. `event_id_missing_or_duplicate`
16. `claim_tick_or_index_invalid`
17. `requested_reference_malformed`
18. `requested_namespace_forbidden_for_provenance`
19. `requested_identity_unresolved`
20. `requested_identity_ambiguous`
21. `claimed_reference_malformed`
22. `claimed_namespace_forbidden_for_provenance`
23. `claimed_identity_unresolved`
24. `claimed_identity_ambiguous`
25. `pose_provenance_invalid`
26. `claim_pose_missing_or_nonfinite`
27. `claim_pose_precision_commitment_mismatch`
28. `physical_computation_nonfinite`
29. `legacy_provenance_noncanonical`
30. `legacy_pose_missing_yaw`
31. `legacy_pose_rounded_or_inferred`

Rejection reasons:

1. `requested_identity_not_in_task_set`
2. `claimed_identity_not_in_task_set`
3. `requested_claimed_identity_mismatch`
4. `outside_inclusive_claim_distance`
5. `zero_inflation_physical_los_blocked`
6. `outside_inclusive_claim_bearing`

Predicate dependencies are also frozen. Trace and event key-set reasons do
not suppress independently testable identifier, task, order, feedback, or
type reasons. Reference resolution runs only for a structurally valid and
provenance-allowed reference; exactly one of unresolved or ambiguous can then
apply. Pose precision mismatch runs only when both decimal and hex triples
parse as three finite binary64 values; otherwise pose-missing/nonfinite
applies. Physical-computation nonfinite runs only after a finite pose and
uniquely resolved claimed target exist. Task-membership and identity-mismatch
rejection predicates run only when the relevant references resolve uniquely.
Distance, LOS, and bearing rejection predicates run only when their factor is
non-null. Every `legacy_*` provenance independently applies the unconditional
legacy reason; missing yaw and rounded/inferred reasons are additionally
gathered when their own predicates hold.

Available physical factors may still be reported for an unverifiable legacy
event, but they cannot change its decision. Proxy distance, proxy bearing, and
proxy acceptance discrepancies are caller diagnostics outside the evaluator
decision.

## Event-result schema

Every `lewm_go2_physical_claim_evaluation_v1` result contains:

- schema and evaluator-contract SHA-256;
- trace ID, episode ID, scene ID, and physical-manifest SHA-256;
- exact sorted task-object IDs and task-object-set SHA-256;
- event ID, tick, event index, and pose provenance;
- requested and claimed references exactly as supplied;
- independent resolution status and resolved object ID for each reference;
- independent requested/claimed task-membership booleans when resolved;
- robot pose as decimal binary64 values, `float.hex()` values, and packed-byte
  SHA-256;
- claimed target object ID and exact manifest center, including hex values;
- literal threshold and LOS-geometry contract;
- distance, signed/absolute bearing, and their hex values when available;
- ordered physical blocker identities;
- the four tri-state factor decisions;
- `decision`, `accepted`, ordered unverifiable reasons, and ordered rejection
  reasons;
- `credited` and optional `duplicate_physical_claim_not_credited`, assigned in
  canonical event order after all decisions;
- canonical content SHA-256.

The canonical event object and hash do not exist until trace pass 1 has
finished and pass 2 has applied trace-global and duplicate predicates. A
private helper's provisional geometry record is not serializable evidence.

No output field named `success`, `claimed`, or `verified` may be populated from
controller state. `physically_verified` is a derived alias for
`decision == accepted` only.

## Two-pass trace aggregation and duplicates

The public trace evaluator receives raw events in strict
`(tick,event_index,event_id)` order. Pass 1 checks the entire sequence and
builds exact event-ID multiplicities before any decision. Event IDs must be
unique; a duplicate makes every occurrence with that ID unverifiable and
credits none. Input order differing from canonical order applies
`trace_event_order_invalid` to the trace and every event rather than silently
sorting. Final event hashes are created only after these predicates are known.

Every accepted event remains an accepted physical observation. Only the first
accepted event for a resolved object ID receives `credited=true`. Later
accepted events for that object receive `credited=false` and aggregate reason
`duplicate_physical_claim_not_credited`. Rejected and unverifiable attempts are
never deduplicated or hidden.

The claimed-object set is derived only from credited accepted evaluations
whose resolved object ID belongs to the committed task set, and is sorted by
exact UTF-8 object-ID bytes. `all_targets_claimed` compares that set for exact
equality with the full committed task-object set. Counts by color are
presentation only. Four repeated color strings can never substitute for four
unique physical object IDs, and no accepted non-task object can be credited.

The finalized trace returned by the authoritative API contains the validated
input envelope, `physical_claim_evaluations`, the scene summary, and the final
trace content hash. The summary records attempted, accepted, rejected, unverifiable, credited,
duplicate, and per-reason counts; credited object IDs; first credited tick and
event ID per object; all event content hashes; and its own canonical content
hash.

## Controller and verifier separation

The runtime trace has two append-only streams joined only by `event_id`:

```text
controller_claim_attempts[]
physical_claim_evaluations[]
```

A raw controller-attempt entry is the exact 13-key evaluator event bound
above. The controller emits intent identity and event identity; an observer
atomically attaches the authoritative pose and commitments without changing
intent. Internal readiness, policy reason, and proxy values remain in the
ordinary controller log under the same event ID, outside the canonical claim
trace, and are never evaluator inputs. The controller may update only
`controller_declared_claimed_object_ids` and its private target scheduler
state.

The evaluator runs in an observer/scoring boundary. Its manifest, target
position, blockers, factors, and decision cannot be read by the controller,
policy, memory, target scheduler, stopping rule, learned head, or controller
log writer. `evaluator_feedback_to_controller` is always exactly empty. Only
the observer may update `physically_verified_claimed_object_ids`. Promotion
completion, claim rate, conversion, false accepts, and all-four status use only
the physical set.

The runtime access ledger contains
`evaluator_output_reads_by_controller`,
`evaluator_callbacks_into_controller`, and
`evaluator_derived_termination_signals`; all three must equal zero. The
finalizer rejects any nonempty feedback field, nonzero counter, evaluator
result passed into a controller method, or evaluator-derived branch in the
controller trace. This checks the forbidden evaluator-to-controller direction;
controller attempts flowing outward to the observer are expected input, not
feedback.

Simulation may continue or stop according to controller-declared state for a
matched policy-cost experiment, but the result must record that stopping basis
and may not call it physical completion. A canonical learned benchmark uses a
fixed budget or controller-only termination; evaluator-derived success may not
terminate it because that is evaluator-to-controller feedback. Physical
completion is computed by the observer after the executed trace.

Rounded copies for UI or replay must use names ending `_display` and are never
read by the evaluator or strict scorer.

The strict scorer requires an exact one-to-one join between raw
`controller_claim_attempts` and stored `physical_claim_evaluations` by event
ID, with equal cardinality, no duplicate, orphan, or omitted ID, and identical
canonical order. It independently reruns `evaluate_physical_claim_trace` from
the raw attempts, physical manifest, and independently bound expected task set
and commitment, then requires bit-exact equality of every event field, factor,
reason, blocker, credit decision, event hash, summary, and trace hash. Stored
evaluator output is never trusted without this recomputation. Any mismatch
makes the score incomplete and no stored event is credited. Legacy rows are
reported separately and never enter this join.

## Legacy trace handling

Legacy claim rows commonly omit yaw and event pose, infer position from a
prior rounded `post_xy`, or store four-decimal oracle claim poses. These rows
are not canonical evidence.

- Every provenance string beginning `legacy_` unconditionally yields
  `legacy_provenance_noncanonical`, even if exact pose commitments and all
  physical factors happen to be present and true.
- Missing yaw yields `legacy_pose_missing_yaw`.
- A pose inferred from another row, rounded before persistence, lacking the
  exact hex triple, or lacking its packed-byte hash yields
  `legacy_pose_rounded_or_inferred`.
- No radius or bearing uncertainty band converts such an event to accepted,
  even when the rounded point is far from a threshold.
- The legacy adapter may report distance, LOS, or an interval diagnostic, but
  the canonical decision remains `unverifiable`.
- Stored proxy distance, controller bearing, target vector, image mask, or
  subsequent pose must not fill a missing component.

Historical 8/72, 13/72, and proxy-conversion values therefore retain their
existing diagnostic labels. The old authoritative V4 oracle artifact is a
regression comparator, not reclassified event telemetry; the migrated oracle
must be rerun from development geometry with full-precision evaluator calls.

## Caller migration order

Migration is fail-closed and occurs in this exact order:

1. Implement the pure two-pass trace evaluator, typed event/summary results,
   canonical hashes, and synthetic mutation tests. Keep its provisional
   single-event helper private. Freeze a reviewed implementation manifest.
2. Replace `generalization_protocol.StrictClaimObservation`,
   `strict_ground_truth_claim`, and its summary with trace construction and
   shared trace evaluation. The old caller-supplied LOS API may remain only
   under an explicitly named legacy diagnostic function and cannot return a
   canonical schema.
3. Migrate `strict_result_scorer`. Modern claim events must provide the exact
   full pose and both references, join attempts and stored evaluations exactly,
   and recompute the whole trace as bound above. Legacy events become
   unverifiable. Remove its private alias choice, target-by-center exclusion,
   and distractor intersection decision from canonical scoring.
4. Migrate `go2_oracle_positive_control`. It may plan a privileged terminal
   standoff pose and yaw, but no `_true_claim`, `update_claims`, sampled
   visibility grid, evaluator call, or equivalent acceptance predicate may
   decide whether to emit an attempt. It emits exactly one attempt per task
   object, unconditionally at the reached planned terminal pose, then the
   completed scene trace is evaluated once by the shared two-pass API.
5. Migrate `go2_physical_eligibility`. Every candidate claim state supplies
   its lattice yaw. The selected witness for every task object is emitted once
   into one scene trace; eligibility passes only when the shared trace accepts
   and credits every witness.
6. Split `benchmark_go2_memory_closed_loop` controller declarations from
   observer verification and emit full-precision trace v1. The evaluator is
   not imported into the policy/controller namespace.
7. Migrate `score_go2_result_batch` and every suite checker, including
   `check_go2_generalized_suite`, to physical summaries. Remove promotion
   decisions based on `claimed_colors`, proxy `success`, or controller
   `claimed`.
8. Only after parity and the oracle gate may G5 observation, reversible target
   belief, routing, and claim-readiness work consume the new trace contract.

No caller may keep a fallback canonical path. During migration, old and new
outputs can coexist only when old fields are explicitly prefixed `legacy_` or
`controller_` and excluded from gates.

## Required tests

Before any development result is produced, synthetic tests must cover:

1. exact trace/episode/scene identity, exact sorted full task set, task-set
   commitment, and every envelope key/type/extra mutation;
2. exact object-ID and unique task-color resolution plus every missing, extra,
   wrong-type, empty, unknown-namespace, and provenance-forbidden typed-
   reference mutation;
3. unresolved, ambiguous, repeated-color, wrong-identity, non-task requested or
   claimed identity, duplicate/empty manifest object ID, and accepted non-task
   non-credit;
4. two-pass duplicate detection proving every occurrence is unverifiable and
   proving no provisional decision or hash escapes pass 1;
5. exact `1.20` acceptance and `nextafter(1.20,+infinity)` rejection;
6. exact `+0.25` and `-0.25` bearing acceptance, one-ULP outside rejection,
   and wrap across `-pi/+pi`;
7. finite arbitrary yaw, zero distance, signed-zero preservation, and
   nonfinite input or slab-intermediate rejection;
8. blocking by a wall, obstacle, rotated obstacle, distractor, and other
   beacon; exact target exclusion; tangent blocking; source or destination
   inside a blocker; zero-length segment; and a narrow uninflated gap passing;
9. scalar inverse-yaw and X-then-Y slab operation-order golden values;
10. UTF-8 object-ID sorting, manifest-compatible `ensure_ascii=True` JSON,
    `allow_nan=False`, blocker/factor invariance to semantically reordered
    geometry, and the required manifest/downstream hash change on that reorder;
11. decimal/hex/packed-byte pose parity and every commitment mutation failing;
12. unconditional legacy-provenance, missing-yaw, inferred-pose, and rounded
    legacy unverifiability regardless of otherwise passing factors;
13. all-applicable reason gathering, immutable reason order, unverifiable
    precedence, and empty rejection reasons whenever unverifiable;
14. first accepted duplicate-object credit, later duplicate non-credit, unique-object
    all-target aggregation, and same-tick event-index ordering;
15. bit-identical evaluator results through protocol, strict scorer, oracle,
    eligibility, and runtime observer adapters for the same synthetic event;
16. strict scorer one-to-one attempt/evaluation joining, full independent
    recomputation, and rejection of every omission, orphan, reorder, field
    mutation, event hash, summary hash, or trace hash mismatch;
17. controller-declared acceptance never changing physical verified state when
    any shared factor fails;
18. rejection of any nonempty evaluator-to-controller feedback and absence of
    evaluator-derived controller termination;
19. rejection of every old proxy-only suite success path;
20. canonical JSON, source-map, and evaluator import-purity
    mutations.

Finalizer tests must independently recompute all booleans and hashes and
reject omitted events, reordered events, duplicated credit, unknown reason
codes, non-literal thresholds, nonempty evaluator-to-controller feedback, a
nonzero evaluator-feedback access-ledger counter, a non-task credit, or any
legacy event counted as physical acceptance.

## Authoritative 96/96 development regression

After all four geometry callers migrate and before G5 model output, rerun the
authoritative V4 development oracle from the same bound development inputs.
The immutable comparator is:

- prior artifact:
  `.generated/oracle_positive_control/go2_generalization_v4_development/report.json`;
- prior artifact SHA-256:
  `7c0a63bb0548fee81918df22b227adec43d4bdc824875ef447793ef4f99d97a5`;
- V4 development-manifest SHA-256:
  `563f240a023309af42a05a9a8f29008f02a0629dee9f77f03568f779d1166d41`;
- materialization SHA-256:
  `a52bd82cb501481707d518d1fffd86e5475b440332f7d226586ebda47e6b1415`;
- geometry-v2 file SHA-256:
  `e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52`;
- directional-policy content ID:
  `c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc`.

The new exclusive result path is
`.generated/oracle_positive_control/go2_generalization_v4_development/canonical_physical_claim_v1_report.json`.
It binds this document, the evaluator implementation/test hashes, every
migrated caller/test hash, the prior comparator, development manifest,
materialization, geometry, primitive registry, and directional policy.

The oracle attempt policy is frozen independently of evaluator outcome:

- each scene binds its exact four-object task set before motion;
- the privileged planner chooses one terminal claim pose and yaw per task
  object and routes to it under its existing deterministic policy;
- upon reaching that planned terminal pose, it appends exactly one raw attempt
  with requested and claimed namespace `object_id` resolving to that task
  object, regardless of whether any private heuristic predicts acceptance;
- it never calls the evaluator, `_true_claim`, `update_claims`, sampled-grid
  LOS, or an equivalent four-factor predicate before deciding to append;
- it makes no retry, replacement, opportunistic, per-tick, or automatically
  detected attempt, and evaluator results cannot affect later routing;
- a route or follower failure that prevents the planned terminal attempt is a
  regression failure, not permission to log from another pose;
- each scene's four raw attempts are evaluated together once, after execution,
  by the two-pass trace API.

Consequently the run must contain exactly four raw attempts per scene and
exactly 96 overall, one for each committed `(scene_id,task_object_id)` pair.
Attempt identity is the SHA-256 of canonical JSON of exactly:

```json
{
  "domain":"lewm-go2-oracle-claim-attempt-v1",
  "episode_id":"<exact episode ID>",
  "scene_id":"<exact scene ID>",
  "task_object_id":"<exact task object ID>",
  "trace_id":"<exact trace ID>"
}
```

This makes attempt identity independent of tick and evaluator outcome while
remaining unique.

The regression passes only with all of the following:

- exactly 24/24 scenes evaluated, exactly 96 raw attempts, exactly 96 final
  evaluations, and 96/96 unique task objects accepted and credited;
- 24/24 scenes with all four physical objects credited;
- zero rejected or unverifiable events used as claims;
- every credited event carrying valid full-precision pose commitments and all
  four true factor decisions;
- all 24 routes still using `OnlineBeliefMap.shortest_path`;
- zero stalls, zero center-grid collision attempts, and zero actual-yaw
  observed-max polygon collision segments;
- physical-eligibility witnesses for all 96 objects under the same evaluator;
- exact reconciliation among event evaluations, per-scene aggregates, and
  the top-level 96 count.

Coverage is reported unchanged but is not part of this claim-evaluator parity
gate. If the migrated oracle falls below 96/96 because the old grid LOS omitted
another beacon, a distractor, or exact tangency, the evaluator remains frozen;
the oracle must choose another valid approach. Thresholds, target identity,
occluders, or full-precision requirements may not be relaxed to recover the
old number.

No held-out, G2, sealed, checkpoint, image, label, model-output, or runtime
payload may be opened by this regression.

## Pre-implementation source map

The reviewed pre-binding sources are recorded so implementation review can
prove which incompatible paths were replaced:

- `lewm/benchmarks/generalization_protocol.py`:
  `cd7dc59202000ed423fba88bf3d94723f7e1b4f1dc6d2040eccd9290c9386c5e`;
- `lewm/benchmarks/strict_result_scorer.py`:
  `332cfe9526f71f1800c1eefe5e65bf8a2491685e5080fe73265c0db513358723`;
- `lewm/benchmarks/go2_oracle_positive_control.py`:
  `0824f1078ccd8a798b47ddfd0795a8b2933e3a396b795238cfdeef6915c1cbb4`;
- `lewm/benchmarks/go2_physical_eligibility.py`:
  `16141944b5d544cdd5c98654956a40641944fa87ce1cc289d8bb3f253063583e`;
- `lewm/planning/exact_occupancy_belief_adapter.py`:
  `a255364fdf4de5e5509fd528497b424df4eb6146cfae30b010c64ad38ce8dd1c`;
- `lewm_worlds/lewm_worlds/scene_graph.py`:
  `5b7ae6f7e5ac5c3aa83e0f3cc392c00ef3756d41b6a7fbe2acffa38ff7a1ed46`;
- `lewm_worlds/lewm_worlds/manifest.py`:
  `5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888`;
- `lewm_worlds/lewm_worlds/planning_grid.py`:
  `e6f7e26d584dfd7923493803fc95a75135122b37a1f95cb51f9267b284649510`;
- `lewm/tests/test_generalization_protocol.py`:
  `088367cf0d4c33baae116949b7ae7b696738d44ddd2e34a7cfadf5e8db4ca922`;
- `lewm/tests/test_strict_result_scorer.py`:
  `25d1b85bf9b5cfa64693b296ecdfd3dce88d83eb55e943505b663f3edbb93610`;
- `lewm/tests/test_go2_oracle_positive_control.py`:
  `d6dc0b3493c158b54be2d728f7e9b1d27f4af465cd300cd5a0964085fde543c1`;
- `lewm/tests/test_go2_physical_eligibility.py`:
  `b3a8ae1af3c027ec463f022885e804f0727da10a87335db88dbfdacfd1538e29`;
- `scripts/benchmark_go2_memory_closed_loop.py`:
  `15f9198cfee18fc997e9b9821f5302c1d58e79794ee1b490d91b400fa8d3c5fc`;
- `scripts/score_go2_result_batch.py`:
  `27200b9cf1beb5a3d7b5b0b82c9b6faa0a2888f4f6d88936c468e98f219423bc`;
- `scripts/check_go2_generalized_suite.py`:
  `0a54ee9fadfd3cb850653d52d00394d23d9c239eb43c5d2d3ff301a1024426c1`.

The implementation manifest must replace this list with a complete source map
covering the evaluator, every migrated caller, every test, finalizer, and
authoritative regression runner before the new result path can be created.

This binding was produced without opening a held-out, G2, sealed, checkpoint,
image, label, model-output, or runtime result payload.
