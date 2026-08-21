# H1 safe-action-set successor V1 result

## Terminal

`H1_SAFE_ACTION_SET_SUCCESSOR_NO_GO`

Independent findings:

- `EMERGENCY_BRAKE_INSUFFICIENT`
- `CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO` — unchanged
- `KINEMATIC_ROUTE_RANKING_LIMITATION`

`PREDECESSOR_VIABILITY_GUARD_REQUIRED` was not supported: none of the 25
contact-triggered predecessor tests produced a qualified stop.

The predecessor formal terminal remains
`GENESIS_EXACT_GEOMETRY_QUERY_UNRESOLVED`, with the precise scope that Genesis
narrowphase and branch-level physics-rate contact querying are resolved and
the formal combined gate missed only best-safe top-3. The 576 historical
branches retained exact branch agreement, physics-step sensitivity 1.0000,
specificity 0.9981, and zero first-contact-step error. The new brake branches
also had native/exact branch-level agreement of 48/48; predecessor brake
branches agreed 25/25. There remains no evidence here that an articulated
contact-dynamics model is required.

## Claim boundary

The target is `H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT`, a simulated
separation/contact-avoidance proxy over the committed H1 block. This result is
not evidence of material-impact safety, injury or property-damage prevention,
human safety, fragile-infrastructure safety, mission safety, or a closed-loop
guarantee.

## Frozen emergency-brake contract

`EMERGENCY_BRAKE_V1` immediately requests `[vx, vy, yaw_rate] = [0, 0, 0]` at
the existing planner-to-locomotion-policy interface. It bypasses only the
planner-level command slew. The low-level locomotion policy, previous-action
state, joint limits, actuator dynamics, body momentum, collisions, contact
dynamics, stability dynamics, and Genesis physics are unchanged. No velocity
or pose clamp, teleportation, joint freeze, artificial damping, or collision
change is used.

The repository contains no actual deployment-equivalent stand, damping, or
emergency-stop controller path; the existing `safety_stop` signal is telemetry
only. Immediate zero velocity/yaw is therefore the fastest executable command
available at this stack's deployed planner/controller seam. The interface is
deployment-realizable, but its realised stopping behaviour did not qualify.

The stopped criterion was frozen before execution: planar speed below 0.05
m/s and absolute yaw rate below 0.10 rad/s for three consecutive 0.1-second
command ticks, with no fall or unsafe termination. Execution ended at stop,
first disallowed contact, fall/unsafe termination, or 2.0 seconds.

## Fixtures

Nine training/development-only fixtures passed deterministic serialization,
finite-value, and no-clamp checks; each was executed twice from independently
rebuilt simulator contexts. The zero-command, maximum forward/reverse/yaw,
combined, and obstacle-free cases remained contact-free but timed out without
meeting the stopped criterion. The front-wall and side-wall fixtures contacted
before stopping. These outcomes were retained rather than used to change the
brake.

## Branch execution and physical outcomes

- Frozen states: 48 (24 calibration, 24 held-out); exact snapshot identity:
  48/48.
- New scientific brake branches: 48, exactly one per state.
- Contact-triggered predecessor branches: 25.
- New state or historical route-candidate identities: zero.
- Current-boundary contact under the operational label: 0/48.
- Brake contacts: 25/48.
- Stops satisfying the frozen criterion: 0/48.
- Two-second timeouts without a qualifying stop: 23/48.
- Falls / unsafe terminations: 0 / 0.
- Stability flags remained acceptable: 48/48, though this does not rescue the
  absent stop or contact failures.
- Qualified safe brakes: 0/48. The relaxed descriptive tolerance (0.07 m/s,
  0.12 rad/s, three ticks) also qualified 0/48.

Because no branch stopped, stopping-time and stopping-distance distributions
are empty. Distance travelled until contact or timeout was 0.00012–0.34211 m
(median 0.20262 m, mean 0.16471 m). Net planar displacement was
0.00007–0.21485 m (median 0.07189 m, mean 0.08020 m). First-contact time over
the 25 failures is retained row by row in the evidence ledger.

Brake contacts/timeouts by family were:

| Family | States | Contact | Timeout | Qualified stop |
|---|---:|---:|---:|---:|
| large_enclosed_maze | 12 | 6 | 6 | 0 |
| medium_enclosed_maze | 12 | 10 | 2 | 0 |
| small_enclosed_maze | 12 | 5 | 7 | 0 |
| loop_alias_stress | 12 | 4 | 8 | 0 |

The peak observed response ranges were 13.85–75.11 m/s² acceleration,
83.76–570.21 rad/s² angular acceleration, and 23.70–35.55 Nm actuator torque.
They are descriptive plant responses, not approved safety limits.

## Historical no-safe states and predecessor audit

All eleven states without a historical contact-negative route candidate also
contacted under the brake; none stopped. Eight were classified
`CONTACT_BEFORE_BRAKE_CAN_RESPOND` and three
`BRAKE_COMMAND_AUTHORITY_INSUFFICIENT`. The planar-speed proxy was lower than
its boundary value before the historical first-contact time in 8/11 states,
but speed reduction did not yield a contact-free stop.

| State | Prior diagnosis | Brake contact time (s) | Brake diagnosis | Predecessor |
|---|---|---:|---|---|
| wide-cal-0-02 | slew-limiter authority | 0.036 | CONTACT_BEFORE_BRAKE_CAN_RESPOND | insufficient |
| wide-cal-0-05 | commitment latency | 0.002 | CONTACT_BEFORE_BRAKE_CAN_RESPOND | insufficient |
| wide-held-0-05 | commitment latency | 0.002 | CONTACT_BEFORE_BRAKE_CAN_RESPOND | insufficient |
| wide-held-1-02 | slew-limiter authority | 0.114 | BRAKE_COMMAND_AUTHORITY_INSUFFICIENT | insufficient |
| wide-held-1-03 | candidate-bank coverage | 0.148 | CONTACT_BEFORE_BRAKE_CAN_RESPOND | insufficient |
| wide-cal-2-05 | candidate-bank coverage | 0.220 | BRAKE_COMMAND_AUTHORITY_INSUFFICIENT | insufficient |
| wide-held-2-00 | unresolved physics | 0.050 | CONTACT_BEFORE_BRAKE_CAN_RESPOND | insufficient |
| wide-held-2-04 | slew-limiter authority | 0.010 | CONTACT_BEFORE_BRAKE_CAN_RESPOND | insufficient |
| wide-cal-3-04 | slew-limiter authority | 0.344 | BRAKE_COMMAND_AUTHORITY_INSUFFICIENT | insufficient |
| wide-held-3-03 | slew-limiter authority | 0.070 | CONTACT_BEFORE_BRAKE_CAN_RESPOND | insufficient |
| wide-held-3-04 | commitment latency | 0.018 | CONTACT_BEFORE_BRAKE_CAN_RESPOND | insufficient |

The bounded predecessor audit ran for every current-state contact failure,
including 14 regression contacts in historically safe-route states. Twenty
predecessors contacted and five remained contact-free but still did not stop;
all 25 were therefore `EMERGENCY_BRAKE_INSUFFICIENT`.

## Successor action availability and route/fallback result

Across all 48 states, 37 retained a safe historical route candidate, zero of
the eleven no-safe-route states gained a safe brake, and eleven had neither.
All six prior candidate-bank coverage failures remained unresolved. The brake
also contacted in 14 states that already had a safe route candidate, so its
general regression qualification failed.

On held-out states, the exact-contact route interface selected safe routes in
17/24 states and correctly selected no contact-positive route. Because the
brake never qualified, the remaining 7/24 states abstained: brake selections
0, correct fallbacks 0, false brakes 0. Mean route progress was 0.17040 m over
states with a safe route alternative (0.12070 m over all held-out states), with
normalized regret 0.09010, best-safe top-1 0.7059, and best-safe top-3 0.8824.
The latter preserves `KINEMATIC_ROUTE_RANKING_LIMITATION`; it does not make the
resolved collision query uncertain.

| Family | Safe routes | Abstentions | Mean progress over all states (m) | Regret | Top-3 |
|---|---:|---:|---:|---:|---:|
| large_enclosed_maze | 5 | 1 | 0.23423 | 0.08610 | 1.00 |
| medium_enclosed_maze | 4 | 2 | 0.04981 | 0.00000 | 1.00 |
| small_enclosed_maze | 4 | 2 | 0.18044 | 0.03505 | 1.00 |
| loop_alias_stress | 4 | 2 | 0.01832 | 0.20295 | 0.50 |

## Decision and exact next experiment

The successor gate fails: the brake contacted in non-boundary-contact states,
restored none of the six candidate-bank/slew failures, preserved neither
historically safe-state regression safety nor bounded stopping, and supplied
no safe fallback to the seven held-out no-safe-route states.

The smallest supported next step is
`DEPLOYMENT_VALID_STRONG_BRAKING_MODE_V1`: integrate and qualify one genuine
Go2 deployment-equivalent stand/damping/emergency-stop controller mode at the
platform interface, preserving actuator and physics realism. The immediate
zero-command locomotion interface should not be retried. Perception and learned
contact prediction remain out of scope until a physical fallback can first
establish a safe response envelope.

A contact-free stop would still not prove mission safety. Repeated braking or
abstention cannot satisfy search/inspection progress; operational contact
avoidance and task progress remain separate requirements.

## Evidence, runtime, and custody

The immutable raw traces occupy 22,192,633 bytes. The 67,511-byte row-level
ledger (SHA-256
`9245f35dbb1c1b99b977e6f21aa8f89150e9970260149fdae2790b41ebb06e1d`,
content digest
`d58a3642ef33920c8f4ad3b46f0f15c019f082ac1a39e6156f0d3d6cff020482`)
is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/h1_safe_action_set_successor_v1/row_level_evidence_v1.json`.
The machine result under `.generated/h1_safe_action_set_successor_v1/result.json`
binds the ledger SHA-256 and content digest.

Fixture runtime was 42.63 s; 48-state plus predecessor replay compute was
1,528.60 s and four-worker wall time was 410.43 s. Evaluation took under 0.1 s.
No model was trained or executed. No JEPA was accessed. No fresh scientific
panel, memory, novelty, navigation, routing, or beacon work occurred.
