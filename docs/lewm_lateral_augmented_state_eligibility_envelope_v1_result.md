# LATERAL_AUGMENTED_STATE_ELIGIBILITY_ENVELOPE_V1

Status: completed

Source baseline: `7d6672e53e567a2b07e51df506be5db4d6b2d04c`

Primary classification: `LATERAL_AUGMENTED_STATE_ELIGIBILITY_SIGNAL`

## Claim boundary

This is an oracle simulated-viability result for the physics-rate simulated
disallowed-contact proxy. It establishes no learned safety or planner claim.
The qualified lateral controller remains simulation-qualified, not physically
qualified. Temporary retreat is acceptable only when route action and progress
resume; repeated recovery or abstention can still fail mission-performance
requirements. Memory, novelty, topology, global routing, beacon discovery,
and navigation remain later layers.

The experiment preserves `LATERAL_RECOVERY_CONTROLLER_QUALIFIED`,
`LATERAL_CONTROLLER_SIGNAL_VIABILITY_NO_GO`,
`CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`,
`STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION`,
`ONE_TICK_FULL_JEPA_COMPUTE_NO_GO`,
`TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED`,
`REPLANNING_INTERFACE_UNRESOLVED`, and
`GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING` under their original scopes.
The new result does not retrospectively change current-boundary availability;
it asks whether an eligibility guard could have prevented entry into those
states.

## Frozen bindings

The twelve historical route actions used the original route checkpoint,
SHA-256
`e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff`.
They retained the unchanged deterministic H3 route ranking.

The mirrored recovery actions used the qualified lateral checkpoint,
SHA-256
`04a85caec6720da2e9c1beabc93817b2a264da7e2efbb87cd3d2b33c614cbaed`:

- lateral left: `[vx, vy, yaw]=[0,+0.20,0]` for 100 ms;
- lateral right: `[vx, vy, yaw]=[0,-0.20,0]` for 100 ms.

The 14-action bank was used only for immediate contact and successor
viability. Lateral actions were not passed through JEPA or assigned an H3
score. Every predecessor captured the simulator state, applied-command
history, route/lateral last-action transition state, enhanced embodied state,
robot-link transforms, and all robot collision-shape transforms. Both
controllers are feedforward; no unrecorded recurrent controller state exists.

Genesis 0.3.14 exposes exact contact/manifold penetration but not positive
pair distance. No positive exact clearance was fabricated. When safe lateral
actions tied on successor-safe-action count, the frozen action index remained
the final deterministic tie-break.

The evaluation-first fixture regenerated the same 14-action tree byte-for-byte
and passed controller-transition, snapshot-restoration, and serialization
checks.

## Frozen residual set and lineage

Exactly five states were frozen before predecessor execution:

| State | One-tick current result | Prior multi-cycle result | Augmented current result |
|---|---|---|---|
| `wide-cal-0-02` | `NO_SAFE_PREFIX_ACTION` | two-to-three-cycle intervention required | non-viable |
| `wide-cal-0-05` | `CONTACT_BEFORE_CONTROL_AUTHORITY` | unresolved/intermittent | non-viable |
| `wide-held-0-05` | `CONTACT_BEFORE_CONTROL_AUTHORITY` | four-to-ten-cycle intervention required | non-viable |
| `wide-held-2-04` | `NO_SAFE_PREFIX_ACTION` | persistent candidate-bank failure | non-viable |
| `wide-held-3-03` | safe-prefix-only/no-viable-successor | two-to-three-cycle intervention required | non-viable |

All ten predecessor boundaries were evaluated for every state; favorable
early results did not truncate the chain.

## Predecessor result

| State | Nearest >=1 admissible | Nearest >=2 admissible | Closest stable boundary | Causal classification |
|---|---:|---:|---:|---|
| `wide-cal-0-02` | 3 ticks | 3 ticks | 3 ticks / 0.3 s | `LATERAL_AUGMENTED_EARLIER_INTERVENTION_SUFFICIENT` |
| `wide-cal-0-05` | 3 ticks | 3 ticks | 3 ticks / 0.3 s | `LATERAL_AUGMENTED_EARLIER_INTERVENTION_SUFFICIENT` |
| `wide-held-0-05` | 1 tick | 1 tick | 3 ticks / 0.3 s | `LATERAL_AUGMENTED_EARLIER_INTERVENTION_SUFFICIENT` |
| `wide-held-2-04` | 2 ticks | none | none | `PRE_EXISTING_CONTACT` |
| `wide-held-3-03` | 2 ticks | 2 ticks | 2 ticks / 0.2 s | `LATERAL_AUGMENTED_EARLIER_INTERVENTION_SUFFICIENT` |

`wide-held-2-04` was already in exact disallowed contact at the registered
boundary. It had one isolated admissible predecessor action at depth two but
no stable three-cycle envelope. It is therefore not treated as an ordinary
false abstention or as evidence that a new diagonal/turning action would solve
the registered state.

The 50 predecessor boundaries generated 700 current-prefix and 3,892 successor
branches. Stability probes generated another 1,120 current-prefix and 12,432
successor branches.

## Multi-cycle recovery

The closest stable boundary—not the earliest or most conservative boundary—
was used for each avoidable residual.

| State | Cycles | Route / left / right | Progress | Outcome |
|---|---:|---:|---:|---|
| `wide-cal-0-02` | 10 | 9 / 0 / 1 | +0.01678 m | right recovery, route resumed |
| `wide-cal-0-05` | 10 | 10 / 0 / 0 | +0.08879 m | repeated reverse/turn route action; viable |
| `wide-held-0-05` | 10 | 9 / 0 / 1 | +0.19781 m | right recovery first, route resumed |
| `wide-held-3-03` | 4 | 3 / 0 / 1 | -0.03373 m | route resumed, then abstained |

No selected action contacted and no selected action entered a non-viable
successor. The `wide-held-3-03` result is a task-performance limitation: it
meets the three-cycle stable-envelope definition and resumes route actions
after recovery, but ends in abstention with negative net progress after four
cycles. It is not evidence of successful mission progress.

## Matched-control non-regression

The same eight prospectively frozen controls were rerun rather than inferred
from the prior summaries. All completed ten cycles using route actions only:

- contact: 0;
- non-viable successor selections: 0;
- transition failures: 0;
- unnecessary lateral recoveries: 0;
- progress: `1.47462 m`, exactly 100% of the completed prior oracle result.

Across the four recovered residuals and eight controls, 114 cycles executed:

- route actions: 111;
- lateral left/right: 0/3;
- controller transitions: 5;
- first-tick contacts/non-viable successors: 0/0;
- cycles with at least one safe successor: 114/114;
- cycles with at least two safe successors: 110/114, or 96.49%;
- minimum selected successor margin: one safe action;
- distance/heading improvement: `1.74426 m` / `1.47189 rad`;
- negative-progress cycles: 11;
- stuck cycles: 2;
- abstentions: 1;
- falls or unsafe terminations: 0.

The unchanged H3 ranker selected the maximum H3 route score except for its
frozen 0.03 m tie handling. Mean selected/oracle H3 ratio was 0.94264 on the
31 recovered-state route cycles and 0.99841 on the 80 control route cycles.

Rollout generation contributed 1,610 current-prefix and 21,182 successor
branches. Total newly evaluated branches were 40,936: 24,542 for residual
predecessor/probe/recovery evidence and 16,394 for matched controls.

## Decision

All frozen gate clauses pass, yielding
`LATERAL_AUGMENTED_STATE_ELIGIBILITY_SIGNAL`:

- every avoidable residual has a stable earlier boundary;
- recovered rollouts select zero contact and zero non-viable successors;
- 96.49% of executed cycles retain at least two safe successors;
- matched controls retain 100% of prior progress;
- route action resumes after every lateral sequence;
- no fall, unsafe termination, transition failure, or family collapse occurs.

No non-pre-existing/contact-before-control residual remains without a stable
envelope. `RESIDUAL_AUGMENTED_ACTION_SET_NO_GO` is therefore not supported,
and no additional recovery mechanism is recommended. The present action bank
is adequate only inside the demonstrated eligibility envelope; the historical
current-state `CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO` remains valid outside
that envelope.

## Exact next implementation

Specify, but do not implement,
`LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_AND_INTERFACE_V1`.

The 100 ms micro layer must predict next-tick contact, next-state safe-action
availability, viability margin, and whether lateral recovery is required. It
may use current depth/LiDAR geometry under the explicit changed sensor
contract, enhanced embodied state, current/candidate action, and control
history. Its complete inference-and-command-replacement path must target P99
at or below 60 ms, leaving interface and control margin.

The approximately 200 ms macro loop remains H1--H3 JEPA rollout plus the
unchanged deterministic H3 route ranking and local-waypoint intent. Before
learned planner evaluation, the system still needs per-tick observations,
lightweight micro inference, command replacement/acknowledgement,
stale-macro-score handling, and deterministic controller transitions.

A viability guard is not an emergency brake. Physical/vendor Go2 stopping
parity remains pending and blocks full runtime-assurance claims.

## Runtime and persistence

The accepted parallel collection wall time was `601.347 s` (10 min 1.347 s).
Generated and cache evidence occupies 62,958,051 bytes (about 60.04 MiB).

The 70-record row ledger is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/lateral_augmented_state_eligibility_envelope_v1/row_level_evidence_v1.jsonl`,
10,468,509 bytes, SHA-256
`ba0d0697fe7dacdc071be4ebede7e4ebd28b3e09be56ee14bcd499d68f34cc22`.
All aggregate metrics reproduce from the persisted records without replay.

No model or controller training occurred. No JEPA predictor was opened. The
action set was not extended beyond the already qualified 14-action micro bank,
and no memory or navigation system was implemented or executed.
