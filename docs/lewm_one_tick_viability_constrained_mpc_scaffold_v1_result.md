# ONE_TICK_VIABILITY_CONSTRAINED_MPC_SCAFFOLD_V1

Status: complete development scaffold

Source baseline: `481253b5a504b0cd9fd05b14f5ad662b496fa0a8`

Primary classification: `ONE_TICK_VIABILITY_KERNEL_NO_GO`

Candidate-bank classification: `CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`

Compute classification: `REPLANNING_COMPUTE_LATENCY_NO_GO`

Platform track: `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`

## Claim boundary

This result qualifies an oracle development scaffold for the simulated
physics-rate disallowed-contact proxy. It is not learned safety, a material
impact or human-safety result, a physical Go2 stopping qualification, or a
closed-loop navigation result. One-tick viability-constrained replanning is not
an emergency brake. Task progress, successor viability, continuation contact,
and stuck are reported separately.

The following terminals remain unchanged:

- `ONE_TICK_SAFE_ACTION_SET_NO_GO`
- `REPLANNING_INTERFACE_UNRESOLVED`
- `DEPLOYMENT_VALID_BRAKING_MODE_UNAVAILABLE`
- `H1_SAFE_ACTION_SET_SUCCESSOR_NO_GO`
- `CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO`

## Interface contract

Genesis runs at a 2 ms physics step, the locomotion policy at 20 ms, and the
high-level command interface at 100 ms. The current MPC invokes observation and
prediction once per five-tick block, buffers/executes the full 500 ms block,
and does not acknowledge replacement before the next 100 ms tick. The
low-level policy can accept a new target every 100 ms and a development capture
path can render each tick, but production RGB remains block-final and no full
observation--prediction--replacement seam is implemented. Thus the 100 ms
interface remains unresolved independently of compute.

The frozen planner decomposition used 100 ms for hard committed-prefix contact,
one additional 100 ms tick for successor viability, and H3 (1.5 s) nominal
route intent for ranking. Only the first 100 ms prefix was executed in each
branch.

## Fixtures and generated evidence

Four pure reducer tests passed. The Genesis training-only fixture repeated both
a current one-tick branch and a successor one-tick branch twice with identical
contact/link traces and endpoint poses. Snapshot restoration and row
serialization were deterministic.

All 576 first-level registered prefixes reproduced their frozen first command
and first 50 physics-step contact labels. The bounded tree materialized 5,628
current-state successor branches. The six authorized predecessor audits added
72 one-tick prefixes and 288 successor branches. No historical candidate branch
or identity was changed.

## Current-state viability

There were 469/576 contact-free first-tick candidates. Of these, 438 were
successor-viable and 31 led to a successor with no safe next-tick action. Only
40/48 states (83.33%) retained a viability-admissible candidate: 22/24
calibration and 18/24 held-out. This is below the frozen 95% gate.

| Family | States retaining viability / 12 |
|---|---:|
| large enclosed | 9 |
| medium enclosed | 11 |
| small enclosed | 10 |
| loop alias stress | 10 |

Across all states, hold, reverse, and a turn were viability-admissible in 39,
39, and 40 states respectively. State classifications were:

| Classification | Count |
|---|---:|
| `VIABILITY_ADMISSIBLE_PROGRESS_ACTION_AVAILABLE` | 27 |
| `VIABILITY_ADMISSIBLE_NONPROGRESS_ACTION_AVAILABLE` | 13 |
| `SAFE_PREFIX_ONLY_NO_VIABLE_SUCCESSOR` | 2 |
| `NO_SAFE_PREFIX_ACTION` | 3 |
| `CONTACT_BEFORE_CONTROL_AUTHORITY` | 3 |

The eight states without a viability-admissible action were:

| State | Family | Classification | Safe prefix candidates |
|---|---|---|---:|
| `wide-cal-0-02` | large | `NO_SAFE_PREFIX_ACTION` | 0 |
| `wide-cal-0-05` | large | `CONTACT_BEFORE_CONTROL_AUTHORITY` | 0 |
| `wide-held-0-05` | large | `CONTACT_BEFORE_CONTROL_AUTHORITY` | 0 |
| `wide-held-1-02` | medium | `SAFE_PREFIX_ONLY_NO_VIABLE_SUCCESSOR` | 6 |
| `wide-held-2-00` | small | `NO_SAFE_PREFIX_ACTION` | 0 |
| `wide-held-2-04` | small | `NO_SAFE_PREFIX_ACTION` | 0 |
| `wide-held-3-03` | loop | `SAFE_PREFIX_ONLY_NO_VIABLE_SUCCESSOR` | 2 |
| `wide-held-3-04` | loop | `CONTACT_BEFORE_CONTROL_AUTHORITY` | 0 |

## Short execution, long utility

| Condition | Retained states | Nonviable selections | Immediate progress rate (m/s) | H3 realised progress (m) | Oracle fraction | Regret | Best top-3 | Later H1 / H2-H3 contact | Stuck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: one-tick safe, one-tick rank | 42 | 8 | 0.0420 | 0.1133 | 0.4902 | 0.3306 | 0.425 | 12 / 12 | 12 |
| B: one-tick safe, H3 rank | 42 | 5 | 0.0546 | 0.1981 | 0.8571 | 0.0733 | 0.850 | 14 / 11 | 11 |
| C: viable, H3 rank | 40 | 0 | 0.0519 | 0.2077 | 0.8987 | 0.0786 | 0.900 | 9 / 9 | 10 |
| D: viable, oracle route upper bound | 40 | 0 | 0.0527 | 0.2311 | 1.0000 | 0.0000 | 1.000 | 11 / 7 | 9 |

Condition C selected zero first-tick contacts and every selected successor was
viable. Its immediate progress rate was 95.10% of condition B, so the viability
constraint did not materially regress immediate movement. All ranking and
route-utility checks passed. Availability was the sole failed scaffold gate.

Condition C retained 9/12, 11/12, 10/12, and 10/12 states in the large, medium,
small, and loop families. Its respective realised H3 progress was 0.2921,
0.1297, 0.2932, and 0.1320 m; oracle fractions were 0.9388, 0.8379, 0.8949,
and 0.9014. Loop immediate progress rate was negative (-0.0191 m/s), and later
contact/stuck remain continuation and recoverability concerns rather than
first-tick violations.

## Predecessor audit

Only two of the six original one-tick no-safe states had a viable intervention
one cycle earlier: `wide-held-0-05` (5 admissible predecessor actions) and
`wide-held-3-04` (9). `wide-cal-0-05` was already unavoidable at the recovered
predecessor. `wide-cal-0-02`, `wide-held-2-00`, and `wide-held-2-04` had no safe
predecessor prefix and remain candidate-bank viability coverage failures.
Therefore `PREDECESSOR_VIABILITY_ENVELOPE_SIGNAL` is not supported.

## Timing-only predictor benchmark

The authorized frozen RGB two-step checkpoint was used only to measure runtime;
no prediction-quality metric was computed or inspected. Four cached RGB frame
families exercised disk availability, crop/resize/normalization, the official
V-JEPA encoder, one 12-candidate H1--H3 batched rollout, deterministic scoring,
and command serialization. Live sensor delivery and controller acknowledgement
were not available and are not claimed.

| Stage | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| cached observation availability | 0.024 | 0.070 | 0.078 | 0.081 |
| preprocessing and transfer | 2.498 | 2.718 | 2.863 | 3.003 |
| frozen visual encoder | 32.440 | 32.607 | 32.839 | 33.056 |
| batched predictor | 143.870 | 144.472 | 144.698 | 144.936 |
| scoring | 0.177 | 0.209 | 0.277 | 0.306 |
| serialization seam | 0.002 | 0.003 | 0.004 | 0.004 |
| **total** | **179.052** | **179.768** | **180.005** | **180.354** |

All 500 timed iterations missed 100 ms; none exceeded 200 ms. P99 exceeded the
frozen two-tick 180 ms compute limit by 0.005 ms, so neither compute gate passes.
GPU busy time averaged 98.18% (P95 100%). Peak allocated VRAM was 2.202 GB,
steady allocation was 1.559 GB, peak RSS was 7.564 GB, and memory was stable.

## Decision

Primary classification: `ONE_TICK_VIABILITY_KERNEL_NO_GO`.

The long-horizon deterministic route score works well inside the viable action
set, but faster execution does not make the frozen candidate bank viable in
enough states. Oracle planner research therefore cannot proceed to
`ORACLE_VIABILITY_CONSTRAINED_LOCAL_WAYPOINT_MPC_V1` yet. Learned planning and
learned safety replacement also remain blocked.

The single next development experiment is
`MULTI_CYCLE_VIABILITY_ENVELOPE_AND_STATE_ELIGIBILITY_V1`: prospectively prevent
entry into states whose frozen action bank lacks a two-step viability-admissible
response. It must qualify the state-eligibility envelope before any learned
planner resumes. Independently, a 100 ms implementation would still require
per-tick production RGB, predictor optimization or a smaller latency-qualified
path, command replacement and acknowledgement. The physical/vendor stopping
parity track remains pending and is not replaced by this scaffold.

## Persistence, runtime, and prohibitions

The 6,564-row evidence ledger is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/one_tick_viability_constrained_mpc_v1/row_level_evidence_v1.jsonl`:

- SHA-256: `af3480af2ae2c20769140e2ced6b3a2d0b9d0994bf21fb5eda93cd3644d7bb13`
- bytes: 2,564,586

The 500-row latency trace SHA-256 is
`ba5b91aefd79714dd165a4cd5f426f49a217449ac2cfb1d0bd00e1d0068f8d5f`.
Branch generation took 511.57 s wall time; reduction took 1.39 s and the timing
benchmark 98.31 s. The principal generated and cache directories occupy
5,557,640 and 4,698,285 bytes respectively.

No model was trained. No learned safety inference, new scientific panel, JEPA
training, global memory, novelty, routing, beacon capture, or navigation work
occurred. The frozen RGB predictor and visual encoder were executed only for
the explicitly authorized timing benchmark. Nothing remained running at
completion.
