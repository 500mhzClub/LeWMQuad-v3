# MULTI_CYCLE_VIABILITY_ENVELOPE_AND_STATE_ELIGIBILITY_V1

Status: complete development-only oracle-control experiment

Source baseline: `8ab19f4816aec7461072f45f48fd9a6f7ceac81e`

Primary classification: `STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION`

Independent candidate-bank classification:
`CANDIDATE_BANK_MULTI_CYCLE_VIABILITY_NO_GO`

Platform track: `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`

## Claim boundary

This experiment evaluates an oracle simulated physics-rate disallowed-contact
proxy. All contact and successor-viability queries are privileged. It does not
establish learned planner safety, material-impact safety, human or property
safety, physical Go2 stopping performance, or closed-loop navigation. Temporary
retreat is a valid viability action, but repeated retreat or abstention can
still fail mission progress. Beacon discovery, topology, and novelty remain
later layers.

The following completed results remain unchanged:

- `ONE_TICK_VIABILITY_KERNEL_NO_GO`
- `CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`
- `REPLANNING_COMPUTE_LATENCY_NO_GO`
- `REPLANNING_INTERFACE_UNRESOLVED`
- `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`

The prior positive sub-result also remains unchanged: on the 40/48 states with
a viability-admissible action, one-tick viability plus deterministic H3 route
ranking selected no first-tick contact or non-viable successor, retained
89.87% of oracle progress, had normalized regret 0.0786, and achieved best-safe
top-3 of 0.90.

## Frozen selection and fixtures

The eight frozen failures were the three `NO_SAFE_PREFIX_ACTION`, three
`CONTACT_BEFORE_CONTROL_AUTHORITY`, and two
`SAFE_PREFIX_ONLY_NO_VIABLE_SUCCESSOR` states from the predecessor experiment.
The eight matched controls were frozen before multi-cycle outcome generation:
two per family, matched using current speed, yaw rate, waypoint distance, and
old safe-candidate count. The selection content digest is
`18a383620e29a9b53fb783d07eff702b024e7633523fb6bc062017963d9f6f68`.

Six pure reducer fixtures passed. The Genesis training-only fixture passed
deterministic snapshot restoration, safe-prefix/viable-successor construction,
and complete serialization. Its digest is
`9888eee8ebcc5f0debc349ac4c2cc59ed378a4095025dbaf1691eada0a22d5c3`.

## Generated evidence

The bounded materialization reconstructed 62 predecessor boundaries and
generated:

| Evidence | Count |
|---|---:|
| predecessor current prefixes | 744 |
| predecessor successor branches | 3,552 |
| rollout current prefixes | 1,548 |
| rollout successor branches | 17,076 |
| row-ledger records | 23,049 |

The final immutable materialization index content digest is
`30eeff53ead3bc05e707ddc27e394b62626cf3f8e2dfa160810f76acd56e7361`.
No historical candidate identity was changed.

## Predecessor envelope and lead times

“Stable-envelope start” is the earlier edge of three consecutive viable
100 ms boundaries. “Closest stable boundary” is its member nearest the
historical failure and expresses the minimum demonstrated intervention lead.

| Failure state | Family | First viable depth | Stable start | Closest stable | Classification |
|---|---|---:|---:|---:|---|
| `wide-cal-0-02` | large | 3 | 5 (0.5 s) | 3 | `TWO_TO_THREE_CYCLE_INTERVENTION_REQUIRED` |
| `wide-cal-0-05` | large | 3 | — | — | `UNRESOLVED` |
| `wide-held-0-05` | large | 1 | 9 (0.9 s) | 7 | `FOUR_TO_TEN_CYCLE_INTERVENTION_REQUIRED` |
| `wide-held-1-02` | medium | 1 | 8 (0.8 s) | 6 | `FOUR_TO_TEN_CYCLE_INTERVENTION_REQUIRED` |
| `wide-held-2-00` | small | 4 | — | — | `UNRESOLVED` |
| `wide-held-2-04` | small | — | — | — | `PERSISTENT_CANDIDATE_BANK_VIABILITY_FAILURE` |
| `wide-held-3-03` | loop | 2 | 4 (0.4 s) | 2 | `TWO_TO_THREE_CYCLE_INTERVENTION_REQUIRED` |
| `wide-held-3-04` | loop | 1 | 6 (0.6 s) | 4 | `FOUR_TO_TEN_CYCLE_INTERVENTION_REQUIRED` |

Five of eight failures therefore have a demonstrated stable predecessor
envelope. `wide-cal-0-05` and `wide-held-2-00` contain isolated viable pockets
but no three-boundary stable run. `wide-held-2-04` has three contact-free turn
or reverse prefixes at depth 2, but none leaves a successor with any safe next
action; no viability-admissible candidate occurs across all ten predecessors.

The stable boundaries include slowing/hold, reversing, and turning mechanisms.
Some rollouts use temporary negative progress: 15/48 executed failure cycles
reverse progress. This supports viability intervention that can sacrifice
short-term route progress.

## Oracle multi-cycle outcomes

Five failure states started from their recovered stable-envelope boundary.
Four completed ten cycles. `wide-cal-0-02` safely completed eight cycles, then
abstained because the next two-step tree had no viability-admissible action.
It did not contact or enter a successor that the preceding decision had marked
non-viable.

| Population | Executed cycles | First-tick contacts | Non-viable selected successors | Cycles with >=2 safe next actions | Progress (m) | H3 oracle fraction | Regret | Top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recovered failures | 48 | 0 | 0 | 48/48 | 0.3637 | 0.9871 | 0.0230 | 1.000 |
| matched controls | 80 | 0 | 0 | 78/80 | 1.4746 | 0.9989 | 0.0011 | 1.000 |
| all rollouts | 128 | 0 | 0 | 126/128 (98.44%) | 1.8384 | 0.9955 | 0.0092 | 1.000 |

No rollout caused a fall or unsafe termination. The failure rollouts contained
five stuck cycles and 15 reverse-progress cycles; the controls contained no
stuck cycles and three reverse-progress cycles. No control abstained.

Per-family outcomes were:

| Family | Rollouts | Ten-cycle completions | Abstentions | Contact/non-viable selections | Progress (m) | H3 oracle fraction |
|---|---:|---:|---:|---:|---:|---:|
| large enclosed | 4 | 3 | 1 | 0 / 0 | 0.4816 | 0.9875 |
| medium enclosed | 3 | 3 | 0 | 0 / 0 | 0.4681 | 1.0000 |
| small enclosed | 2 | 2 | 0 | 0 / 0 | 0.5541 | 1.0000 |
| loop alias stress | 4 | 4 | 0 | 0 / 0 | 0.3347 | 0.9967 |

The small-family table contains matched controls only because neither small
failure had a stable predecessor envelope. It must not be read as recovery of
those failures.

## Gate and decision

The rollout safety, safe-action margin, matched-control route retention,
family, and stability checks passed. The complete gate failed because only
5/8 historical failures had an earlier stable viability envelope. The result
therefore is not `MULTI_CYCLE_VIABILITY_ENVELOPE_SIGNAL`.

Most failures are recoverable earlier and the deterministic H3 ranker remains
effective inside the viable set. A persistent small-maze action gap remains,
so the primary classification is
`STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION`, accompanied by
`CANDIDATE_BANK_MULTI_CYCLE_VIABILITY_NO_GO`.

The single next experiment is `H1_CANDIDATE_BANK_VIABILITY_SUCCESSOR_V1` with
one added mechanism: `DEDICATED_LATERAL_RETREAT`. The evidence is narrow:
turning and reverse already provide contact-free prefixes in the persistent
state but not a viable successor, while all twelve frozen primitives have
zero lateral velocity. The successor must be frozen prospectively and the
oracle viability kernel requalified before learned planning resumes. No other
candidate mechanism is authorized by this result.

## State eligibility and two-rate specifications

`MULTI_CYCLE_STATE_ELIGIBILITY_GUARD_V1` is supported for the recoverable
subset, but it is not sufficient until the candidate-bank successor passes.
It requires a three-tick viability horizon and a conservative lower confidence
bound of at least two safe next actions. An unresolved estimate is
inadmissible. Eligibility hard-filters first; the unchanged deterministic H3
route intent then ranks actions. Its fallback can be the dedicated lateral
retreat only after that action is oracle-qualified; otherwise it abstains.

`TWO_RATE_VIABILITY_AND_ROUTE_MPC_V1` remains a specification, not an
implementation:

- a target-100 ms lightweight micro loop handles committed-prefix contact,
  successor safe-action availability, non-viable-entry prevention, and command
  replacement;
- an approximately 200 ms macro loop handles H1--H3 rollout, deterministic H3
  ranking, waypoint progress, and continuation risk;
- a macro score may be reused briefly only while the micro loop confirms that
  the current action remains admissible.

The preserved timing is P50 179.052 ms, P95 179.768 ms, P99 180.005 ms, and
maximum 180.354 ms; all 500 iterations miss 100 ms and none misses 200 ms.
Therefore the accurate interpretation is
`ONE_TICK_FULL_JEPA_COMPUTE_NO_GO` and
`TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED`. A 0.005 ms miss of the
180 ms scheduling-margin target is not a fundamental 200 ms compute failure.
The actual blockers remain block-final RGB, the 500 ms replanning interface,
and missing per-tick replacement acknowledgement.

## Persistence, runtime, and prohibitions

The 23,049-row evidence ledger is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/multi_cycle_viability_envelope_v1/row_level_evidence_v1.jsonl`:

- SHA-256: `5bd5186054d698ac15f829900f09c5800c729466a16d023f2fee7642f2599f95`
- bytes: 6,137,229

The accepted revision-3 materialization took 340.15 s wall time (24.11 s
fixtures; 966.34 s summed parallel state compute). An earlier 316.85 s run was
discarded after detecting that rollouts began at the closest member rather
than the earlier edge of a three-cycle stable envelope; the state selection
was unchanged and revision-3 regenerated the affected evidence. Total bounded
branch-processing wall time was therefore approximately 657.00 s. Reduction
took approximately 0.13 s. Generated evidence occupies 24,568,817 bytes and
the cache directory 8,959,429 bytes.

No model was trained. No learned safety or planning inference and no JEPA
prediction-quality evaluation occurred. The frozen low-level Genesis
locomotion policy ran only as the already-bound simulator controller required
to execute the explicitly authorized oracle branches; it was not evaluated as
a learned model. No candidate-bank change, new scientific panel, global
memory, novelty, routing, beacon capture, or navigation work occurred.
`GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING` remains unresolved: oracle
viability does not replace a qualified platform stopping mode.
