# H1_CANDIDATE_BANK_VIABILITY_SUCCESSOR_V1

Status: stopped at the prospectively frozen training-only controller-authority
gate

Source baseline: `4b655f054ffa1e7322d81a78a7920e260a8283bd`

Primary classification: `LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO`

Candidate-bank classification: `CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`

## Claim boundary

This was a development-only qualification of a proposed simulated
micro-viability mechanism. It did not enter scientific state execution and
supports no learned safety, planning, material-hazard, human-safety, or
navigation claim.

The macro/micro boundary remains explicit:

- the macro route bank is the unchanged historical twelve actions, scored by
  unchanged deterministic H3 route intent;
- a micro bank could include mirrored lateral retreat only after the low-level
  controller demonstrates deployment-valid lateral authority;
- lateral retreat was not added to the JEPA action contract or passed through
  the JEPA predictor.

The completed predecessor classifications remain unchanged:

- `STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION`
- `CANDIDATE_BANK_MULTI_CYCLE_VIABILITY_NO_GO`
- `ONE_TICK_VIABILITY_KERNEL_NO_GO`
- `ONE_TICK_FULL_JEPA_COMPUTE_NO_GO`
- `TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED`
- `REPLANNING_INTERFACE_UNRESOLVED`
- `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`

Inside the already represented viability envelope, the prior oracle result
also remains unchanged: one-tick viability plus H3 deterministic ranking
selected no contact or non-viable successor and retained approximately 99% of
oracle route progress.

## Frozen controller audit

The command vector structurally contains `(vx, vy, yaw_rate)`, but the bound
controller contract contains no nonzero lateral authority:

| Evidence | Frozen value |
|---|---|
| platform accepted `vy` range | `[0.0, 0.0] m/s` |
| platform maximum per-tick `vy` delta | `0.0 m/s` |
| PPO training `lin_vel_y_range` | `[0.0, 0.0] m/s` |
| PPO training-bank `vy` values | `{0.0}` |
| registry lateral left | `+0.20 m/s`, `train: false`, enable after validation |
| registry lateral right | `-0.20 m/s`, `train: false`, enable after validation |

The locally bound controller therefore accepts the field syntactically but
does not apply a nonzero value. The ±0.20 m/s registry magnitude was used only
as a qualification probe. No scientific lateral magnitude was frozen.

Bindings:

- platform manifest SHA-256:
  `5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189`
- primitive registry SHA-256:
  `cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8`
- policy configuration SHA-256:
  `bc3e68c18252475199e57b30c8ac49d813e3c784a3983e0e8b1a762490dde24f`

The locomotion checkpoint was not opened or executed.

## Training-only fixture gate

Eight specified fixture contexts, both mirrored directions, and two repeats
produced 32 adapter-gate rows. Every requested ±0.20 m/s lateral command was
clipped to exactly `vy=0.0`; every repeated reduction was byte-identical and
all applied controller values were finite. Zero rows contained a nonzero
applied lateral command.

Because the common command adapter rejected the mechanism before environment
dynamics, obstacle response, lateral displacement, contact, fall, torque, and
stability were not evaluated. Reporting ordinary zero-command drift as
lateral tracking would have been invalid. The required “measurable displacement
in the requested direction” fixture condition therefore failed, triggering
the explicit early stop.

## Scientific execution and bank availability

No frozen calibration, held-out, failure, matched-control, or full-panel state
was restored. Generated scientific branch counts were:

| Branch category | Count |
|---|---:|
| current-state lateral branches | 0 |
| existing-successor lateral branches | 0 |
| lateral-prefix successor branches | 0 |
| multi-cycle rollout branches | 0 |

The prior full-panel availability remains 40/48 states with a
viability-admissible historical action. There is no valid “after” value:
the operational micro bank was not augmented and remains the historical
twelve actions. Consequently `CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO` is
preserved.

The persistent `wide-held-2-04` failure, two intermittent envelopes, five
previously stable envelopes, matched-control rollouts, action-selection
frequencies, route progress, and temporary-retreat outcomes were not
re-evaluated. Their historical results remain authoritative. No zero-applied
command is counted as lateral recovery.

## Decision and exact next implementation

Primary classification:
`LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO`.

This result concerns controller authority, not the causal usefulness of
lateral retreat. `LATERAL_RETREAT_POSITIVE_TENDENCY` and
`LATERAL_RETREAT_VIABILITY_NO_SIGNAL` are not supported because the mechanism
never reached dynamics.

The exact prerequisite is
`DEPLOYMENT_VALID_LATERAL_LOCOMOTION_CONTROLLER_V1`:

1. prospectively train or bind a low-level controller whose command
   distribution includes mirrored nonzero `vy`;
2. qualify lateral tracking, stability, joint/torque limits, contact, and
   deterministic command application on training-only fixtures;
3. update the platform safety envelope only after that qualification;
4. rerun `H1_CANDIDATE_BANK_VIABILITY_SUCCESSOR_V1` without changing the
   historical macro route bank.

This prerequisite is specified only; it was not implemented. Adding gain
tuning, yaw coupling, another micro mechanism, or a speculative Unitree
controller is not authorized by this result.

## Predictor and replanning consequences

The frozen JEPA remains unqualified for lateral `vy`. It was not opened, its
action contract was not modified, and the lateral mechanism cannot yet enter
either a 100 ms micro loop or macro JEPA scoring. The prior compute and
interface terminals remain unchanged. `TWO_RATE_VIABILITY_AND_ROUTE_MPC_V1`
remains conditional on first qualifying an executable micro-action set.

A lateral viability action would not be an emergency brake. The independent
`GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING` track remains open. Viability
preservation and mission progress remain separate requirements; repeated
retreat without eventual progress would violate the task requirement.

## Persistence, runtime, and prohibitions

The 32-row training-fixture adapter ledger is:

`/home/andrewknowles/.cache/lewm_go2_temporal_v03/h1_candidate_bank_viability_successor_v1/training_fixture_command_adapter_ledger_v1.jsonl`

- SHA-256: `5df36a9aa54ca81778131ebfe20470e0a5144e980769b1179a7fcfd9e799a150`
- bytes: 17,620

The result JSON is 4,906 bytes with SHA-256
`bb6fc76214e15fcc6fe220f338fb9dcf23bda81fd90b2ddb4144ea8e30814d9a`.
The deterministic reduction took 0.0067 s. Total generated and cache storage
is 22,526 bytes.

No model was trained or executed. No JEPA checkpoint, frozen predictor, or
scientific state was opened. No predictor contract, historical candidate,
memory, novelty, routing, beacon capture, or navigation system was modified or
executed.

