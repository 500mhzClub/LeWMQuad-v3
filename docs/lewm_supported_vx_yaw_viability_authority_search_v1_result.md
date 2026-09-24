# SUPPORTED_VX_YAW_VIABILITY_AUTHORITY_SEARCH_V1

Status: complete post-outcome development diagnostic

Source baseline: `11a0c258e479f79a640ab237841f52ec0e6b6ecc`

Primary classification: `SUPPORTED_VX_YAW_CONTROL_AUTHORITY_NO_GO`

Candidate-bank classification: `CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`

## Claim boundary

This is an oracle simulated viability experiment against
`H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT`. It does not establish learned
planner safety, material-impact safety, injury or property-damage prevention,
human safety, a qualified emergency brake, or navigation performance.

The earlier lateral terminal is preserved with its exact scope:
`LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO` means that the frozen low-level
controller authorises only `vy=0`; lateral retreat itself was never physically
executed or scientifically evaluated.

The following terminals remain unchanged:

- `CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`
- `STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION`
- `CANDIDATE_BANK_MULTI_CYCLE_VIABILITY_NO_GO`
- `ONE_TICK_FULL_JEPA_COMPUTE_NO_GO`
- `TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED`
- `REPLANNING_INTERFACE_UNRESOLVED`
- `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`

Inside the already represented viability envelope, the prior oracle result
also remains unchanged: one-tick viability filtering plus deterministic H3
route ranking avoids contact and non-viable successors while retaining about
99% of oracle route progress. This experiment did not redesign or rerun that
ranker.

## Frozen controller and grid

The controller contract was bound before scientific execution:

| Item | Frozen value |
|---|---:|
| Manifest `vx` range | `[-0.30, +0.30] m/s` |
| PPO-training `vx` range | `[-0.20, +0.30] m/s` |
| Search reverse limit | `-0.20 m/s` |
| Manifest yaw range | `[-0.50, +0.50] rad/s` |
| PPO-training yaw range | `[-0.45, +0.45] rad/s` |
| Search yaw limit | `±0.45 rad/s` |
| `vy` | exactly `0 m/s` |
| Command period | `0.10 s` |
| Low-level policy period | `0.02 s` |
| Per-tick slew limits | `Δvx=0.25 m/s`, `Δvy=0`, `Δwz=0.35 rad/s` |

The prospectively frozen grid had 21 requests, below the explicit maximum of
25:

- zero;
- pure reverse `vx ∈ {-0.05,-0.10,-0.15,-0.20} m/s`;
- mirrored in-place yaw `wz ∈ ±{0.1125,0.2250,0.3375,0.4500} rad/s`;
- matched-fraction mirrored reverse arcs combining the corresponding reverse
  and yaw magnitudes.

The grid digest is
`64c139fc6a792019f2e7d6d2b33c9df241fbff6a26b3ce559c03584bd5732d04`.
The requested cross-product wording would have produced 45 combinations, so
matched fractions were frozen prospectively to cover every required magnitude
and mechanism family while respecting the stated 25-command cap.

Slew-dependent deduplication across the 16 scientific states reduced 336
requests to 235 unique applied commands. There were 101 within-grid duplicate
applications, 54 unique applications duplicated a historical candidate, and
181 were genuinely new relative to the historical first-tick commands. Exact
per-state requested/applied mappings are retained in the row evidence.

## Training-only fixtures

Nine contexts covered obstacle-free rest, forward and yaw initial motion,
rear clearance, left/right walls, both front corners, and a narrow corridor.
Every grid command was run twice, producing 378 fixture branches.

| Check | Result |
|---|---:|
| Deterministic reductions | 189/189 pairs |
| Finite controller outputs | 378/378 |
| Obstacle-free contacts | 0 |
| Falls or unsafe terminations | 0 |
| Maximum one-tick reverse response | `0.000551 m` |
| Maximum one-tick yaw response | `0.078232 rad` |

Contact, termination, and command fields were compared exactly. Continuous
fixture-only pose reductions were canonicalised to `1e-4` to remove one
sub-0.1-mm/rad backend discrepancy; the complete physical rows remain
persisted. The frozen controller therefore demonstrated measurable reverse
and yaw authority and passed the fixture gate.

## Bounded reachability search

The scientific stage generated 235 unique current branches and 2,904 unique
successor branches across the eight residual failures and the eight frozen
matched controls. Together with fixtures, 3,517 accepted branches were
executed.

The residual-state result was:

| State | Family | Previous class | Supported-space result | Safe prefixes | Viable commands | First-contact step range |
|---|---|---|---|---:|---:|---:|
| `wide-cal-0-02` | large | no safe prefix | `NO_SUPPORTED_VX_YAW_VIABLE_ACTION` | 0 | 0 | 17–18 |
| `wide-cal-0-05` | large | before control authority | `PRE_EXISTING_CONTACT` | 0 | 0 | 0 |
| `wide-held-0-05` | large | before control authority | `PRE_EXISTING_CONTACT` | 0 | 0 | 0 |
| `wide-held-1-02` | medium | safe prefix only | `SUPPORTED_COMMAND_RECOVERS_PREFIX_ONLY` | 8 | 0 | 49 |
| `wide-held-2-00` | small | no safe prefix | `NO_SUPPORTED_VX_YAW_VIABLE_ACTION` | 0 | 0 | 23–24 |
| `wide-held-2-04` | small | no safe prefix | `CONTACT_BEFORE_SUPPORTED_CONTROL_AUTHORITY` | 0 | 0 | 4 |
| `wide-held-3-03` | loop | safe prefix only | `SUPPORTED_COMMAND_RECOVERS_PREFIX_ONLY` | 6 | 0 | 31–36 |
| `wide-held-3-04` | loop | before control authority | `CONTACT_BEFORE_SUPPORTED_CONTROL_AUTHORITY` | 0 | 0 | 8 |

One low-level control interval is ten physics steps. Thus the last two
control-authority cases contact before a newly requested command can complete
one controller-response update. The medium state had safe zero/turn prefixes;
the loop state had safe left-turn and left reverse-arc prefixes, including
temporary negative progress. None of those 14 safe prefixes left even one
safe next-tick action. In particular, no supported command resolved
`wide-held-2-04`.

Genesis exposed the exact binary contact/manifold verdict but not a continuous
positive separation for these restored configurations. The ledger therefore
marks exact positive clearance unavailable and retains a scene-graph wall
clearance diagnostic separately; that diagnostic was not used for viability
or mechanism selection.

The eight matched controls had zero contact across all 121 unique first-tick
commands. All eight retained viability; 119/121 command successors were
viable. This rules out a broad regression in the bounded control population,
but it cannot rescue the residual failures.

## Mechanism selection and multi-cycle boundary

No requested command was viability-admissible in any of the eight residual
states. Therefore none of `PURE_REVERSE_RETREAT`,
`MIRRORED_REVERSE_ARC_RETREAT`, or `MIRRORED_IN_PLACE_ESCAPE_TURN` qualified
under the frozen mechanism-selection rule. No mechanism was selected, the
micro bank was not augmented, and the conditional ten-cycle evaluation was
correctly not run. There are consequently no new route-progress or recovery
selection frequencies to report; the previous oracle H3 ranking result remains
the authoritative route result.

Full-panel availability is unchanged at 40/48 states. Excluding the two
independently identified pre-existing-contact states, this is 40/46
(`86.96%`), below the required 95%. The before and after values are identical
because no successor mechanism qualified. The correct bank result remains
`CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`.

## Decision

The supported `vx–wz` space did not restore viability in the persistent or
intermittent failures. Some commands create a contact-free prefix, proving
that not every error is immediate contact, but the resulting states still
have no safe response. The primary classification is therefore
`SUPPORTED_VX_YAW_CONTROL_AUTHORITY_NO_GO`, not
`SUPPORTED_CONTROL_SPACE_SIGNAL_DISCRETE_MECHANISM_NO_GO` and not the global
`CONTACT_BEFORE_CONTROL_AUTHORITY` terminal.

This supports specifying—but did not train—
`DEPLOYMENT_VALID_LATERAL_LOCOMOTION_CONTROLLER_V1`. Its first attempt must:

1. use one exploratory seed and a prospectively frozen mirrored nonzero `vy`
   training distribution;
2. preserve existing `vx` and yaw tracking;
3. qualify lateral tracking, stability, torque, and contact on training-only
   fixtures;
4. produce a new checkpoint and explicit control envelope;
5. rerun the oracle viability successor experiment before any learned safety
   or planning experiment.

The macro route bank remains the historical twelve actions. The JEPA action
contract was not changed. A later qualified lateral action may initially
remain a micro-loop recovery mechanism outside macro JEPA scoring; it must not
enter macro prediction until `vy` is prospectively added to the action
representation and compatible data are collected.

The two-rate architecture and blockers remain unchanged. A lightweight micro
loop would target 100 ms for contact, successor viability, and recovery; the
approximately 200 ms macro loop would perform H1–H3 rollout and deterministic
route ranking. This experiment does not solve the command-replacement
interface or the independent platform stopping-mode parity track. A supported
reverse/turn action is not an emergency brake.

## Persistence, runtime, and prohibitions

The accepted fixture run took 64.945 s and the parallel scientific search
took 152.739 s, for 217.684 s of accepted branch materialisation. Including a
discarded pre-science runtime-environment attempt and the fixture
canonicalisation rerun, total task execution was approximately 310 s.

Generated artifacts occupy 9,435,586 bytes and the cache occupies 2,856,249
bytes (12,291,835 bytes total). The flattened 3,139-row scientific ledger is:

`/home/andrewknowles/.cache/lewm_go2_temporal_v03/supported_vx_yaw_viability_authority_search_v1/row_level_evidence_v1.jsonl`

- SHA-256: `42d6b59491539e3d61e3832c76aed9f95db7a4da0d210200b137f4496a416cee`
- bytes: 2,059,422

No model was trained and no low-level controller was retrained. No JEPA,
learned safety model, or learned planner was opened or executed. The frozen
PPO locomotion policy necessarily executed only as the already-bound simulator
control plant for the authorised command branches; it was not modified or
scientifically re-evaluated. No macro candidate identity, predictor contract,
memory, novelty, routing, beacon capture, or navigation system was changed or
executed.
