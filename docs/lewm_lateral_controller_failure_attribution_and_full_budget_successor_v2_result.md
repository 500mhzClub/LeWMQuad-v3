# LATERAL_CONTROLLER_FAILURE_ATTRIBUTION_AND_FULL_BUDGET_SUCCESSOR_V2

Status: completed at the conditional oracle-viability gate

Source baseline: `004ef60c81d98f744e5dad0206d4c6a618707196`

Controller classification: `LATERAL_RECOVERY_CONTROLLER_QUALIFIED`

Primary experiment classification: `LATERAL_CONTROLLER_SIGNAL_VIABILITY_NO_GO`

## Claim boundary

This is a simulation-only controller-development and oracle-viability result
against `H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT`. It establishes neither
learned planner safety nor physical Go2 qualification. It makes no
material-impact, injury, property, human-safety, fragile-infrastructure,
closed-loop navigation, or emergency-stop claim.

The V1 terminals `LATERAL_CONTROLLER_QUALIFICATION_NO_GO` and
`LATERAL_TRACKING_AUTHORITY_NO_GO` remain valid for the 125-update V1
checkpoint. The successor does not revise that historical result. The
following terminals also remain preserved:
`SUPPORTED_VX_YAW_CONTROL_AUTHORITY_NO_GO`,
`CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`,
`STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION`,
`ONE_TICK_FULL_JEPA_COMPUTE_NO_GO`,
`TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED`,
`REPLANNING_INTERFACE_UNRESOLVED`, and
`GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`.

## Frozen controller bindings

The original route controller remained frozen at
`models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt`, SHA-256
`e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff`.
Historical route actions continued to execute through this controller.

The V1 lateral checkpoint was bound at SHA-256
`9199cfde3d26b421fc50bc5a7a94f69b23eb7befc7509175174eb3059f35d18b`.
Its continuation contract was seed `2026082014`, iterations 500--624, 125
updates, 4,096 environments, `vy=[-0.20,+0.20] m/s`, and the frozen
50%/25%/25% historical-route/pure-lateral/route-to-lateral mixture.

## Failure attribution

The apparent V1 transition nondeterminism localized to
`EVALUATOR_ROW_ALIGNMENT_DEFECT`: the V1 reducer treated separate vectorized
environment lanes as repeats. In the corrected same-lane reset/replay, the
complete first-step observation, frozen normalization output, command,
deterministic mean policy output, applied joint action, simulator state, and
RNG witnesses agreed within the pre-existing `1e-4` tolerance. Scientific
evaluation used evaluation mode, policy-mean actions, no action sampling, no
observation noise or domain randomization, frozen identity normalization, and
full controller/gait-state restoration.

Requalifying V1 under this corrected harness did not repair lateral authority:
the `|vy|=0.20 m/s` median achieved/requested ratio was `0.22980`, below the
frozen 0.50 gate. V1 therefore remained a genuine lateral-tracking failure.

The lateral command path itself was sound: nonzero `vy` reached the 45-value
policy observation with its sign and magnitude preserved, no adapter or
limiter reclamped it in training, the reward used body-frame `vx` and `vy`,
and the 50%/25%/25% mixture was present. On 128 fixed observations the V1
policy was `POLICY_LATERAL_COMMAND_SENSITIVE`: all observations exceeded the
material-sensitivity threshold, the mean `vy=-0.20` versus `+0.20` action
distance was `1.74978`, mirror residual was `0.17404`, and sign symmetry was
`0.93294`.

The concrete defect was reward resolution. V1 used

`exp(-(vx_error^2 + vy_error^2) / 0.25)`.

At a requested `|vy|=0.20 m/s`, zero lateral response still receives
`0.85214`. Reward values 0.90, 0.75, and 0.50 correspond to errors of
`0.16230`, `0.26818`, and `0.41628 m/s`. Thus the final mixed global linear
reward of `0.93009` was compatible with weak lateral response because of both
the broad tolerance and mixture weighting. The requested-magnitude audit
found mean V1 absolute errors of `0.03629`, `0.08016`, `0.11154`, and
`0.14062 m/s` at `|vy|=0.05`, 0.10, 0.15, and 0.20 respectively. Intermediate
physical qualification checkpoints were unavailable, so only the training
metric curve—not an intermediate physical curve—could be inspected.

The V1 failure classification was therefore `REWARD_OR_BINDING_DEFECT`, and
the prospectively chosen path was `PATH_C_CORRECTED_SUCCESSOR_TRAINING`.

## Single corrected successor

Only the proven y-axis reward-resolution defect changed:

`exp(-vx_error^2 / 0.25 - vy_error^2 / 0.04)`, where `0.04=(0.20 m/s)^2`.

All unrelated rewards, PPO settings, domain randomization, controller
settings, environment count, command mixture, and command range were
preserved. The successor restarted from the original route checkpoint and
used exactly one seed, `2026082015`, for 500 PPO updates (iterations
500--999), with final-update selection only.

The final checkpoint is
`.generated/lateral_controller_failure_attribution_and_full_budget_successor_v2/seed_2026082015/model_999.pt`,
4,547,691 bytes, SHA-256
`04a85caec6720da2e9c1beabc93817b2a264da7e2efbb87cd3d2b33c614cbaed`.

| Metric | First update | Final update |
|---|---:|---:|
| Mean reward | 0.2630 | 33.2386 |
| Linear tracking reward | 0.01084 | 0.98567 |
| Angular tracking reward | 0.00866 | 0.78723 |
| Value loss | 0.03320 | 0.0000611 |
| Surrogate loss | 0.01219 | -0.00294 |
| Entropy | 0.46616 | 1.28927 |

Twenty fixed-case monitors were recorded every 25 updates and did not select
a checkpoint. The lateral achieved/requested median rose from `0.12617` at
update 25 to `1.01365` at update 500; final sign accuracy was 1.0, route
`vx`/yaw errors were `0.02334 m/s`/`0.01173 rad/s`, and contact, fall, and
joint/torque-limit counts were zero.

## Final controller qualification

All frozen gates passed, yielding `LATERAL_RECOVERY_CONTROLLER_QUALIFIED`.

| Route non-regression metric | Frozen route | Successor | Allowed |
|---|---:|---:|---:|
| Mean `vx` absolute error | 0.00654 m/s | 0.01189 m/s | <=0.02654 |
| Mean yaw absolute error | 0.00683 rad/s | 0.00789 rad/s | <=0.05683 |
| Unintended `vy` | 0.00832 m/s | 0.00309 m/s | descriptive |
| Energy proxy | 42.936 | 44.128 | descriptive |
| Action smoothness | 0.26365 | 0.27312 | descriptive |
| Peak tilt proxy | 0.07708 | 0.08778 | no material instability |
| Contact/fall/limits | 0 | 0 | zero |

All 48 lateral rows had the correct sign and measurable response by 0.2 s.
At 0.5 s the median achieved/requested velocity ratio for
`|vy|=0.20 m/s` was `0.98465`. There were no obstacle-free contacts, falls,
unsafe terminations, or joint/torque-limit violations. Same-lane repeats were
deterministic at `1e-4`.

All 12 route-to-lateral-to-route transition cases passed with zero contact,
fall, or limit violation; deterministic repeats passed and route tracking
resumed. Maximum entry/return joint-action discontinuities were
`2.95660`/`2.61852`, maximum base/angular acceleration was
`13.13674 m/s^2`/`89.68356 rad/s^2`, and maximum return `vx`/yaw errors were
`0.13819 m/s`/`0.08426 rad/s`, inside the frozen transition limits. These are
simulation results, not physical Go2 transition qualifications.

## Oracle lateral-viability result

Qualification opened the conditional scientific stage. It generated 27,991
bounded branches: 2,609 for the 48 current-state augmentations and 25,382 for
the 16 ten-cycle failure/control rollouts. Historical route actions used the
frozen controller; only mirrored `vy=+/-0.20 m/s` recovery actions used the
successor controller. The JEPA was not opened and lateral actions received no
H3 predictor score.

Current-state viability increased from 40/48 to 43/48:

| Family | Viable after | States |
|---|---:|---:|
| large enclosed | 9 | 12 |
| medium enclosed | 12 | 12 |
| small enclosed | 11 | 12 |
| loop-alias stress | 11 | 12 |

Lateral actions newly recovered `wide-held-1-02`, `wide-held-2-00`, and
`wide-held-3-04`. Five current states remained non-viable:
`wide-cal-0-02`, `wide-cal-0-05`, `wide-held-0-05`, `wide-held-2-04`, and
`wide-held-3-03`. Thus availability was 89.58%, below the required 95%, and
`CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO` remains preserved.

Across the bounded rollouts, 130 cycles executed: 127 route actions and three
right-lateral recoveries, with five controller switches. There were zero
selected first-tick contacts, zero selected non-viable successors, zero
transition failures, and 126/130 cycles (96.92%) retained at least two safe
successor actions. Eight matched controls completed 80 cycles, selected no
lateral action, retained exactly 100% of their prior oracle progress, and
introduced no safety regression. Overall progress was `1.97613 m`, with 15
temporary negative-progress cycles.

`wide-held-2-00` was the positive recovery case: three right-lateral actions
produced a stable ten-cycle envelope and route actions resumed. The five
previously stable earlier envelopes all retained at least three safe cycles.
However, intermittent `wide-cal-0-05` and persistent `wide-held-2-04`
abstained immediately with no viability-admissible augmented action. The two
intermittent cases therefore did not both stabilize, the persistent case was
not resolved, and the full-panel 95% gate failed.

Genesis 0.3.14 exposes exact manifold contact/penetration but not positive
pair distance. Consequently, safe-successor count determined lateral ties;
where positive exact clearance was unavailable, the prospectively frozen
action index was the final deterministic tie-break. No clearance was
fabricated.

## Decision and boundary to further work

The primary classification is `LATERAL_CONTROLLER_SIGNAL_VIABILITY_NO_GO`:
the corrected controller demonstrates qualified lateral authority, but the
single mirrored lateral-recovery mechanism does not close the residual
viability gap. This is not evidence against lateral locomotion itself, and it
does not authorize another action mechanism, controller seed, or longer
training run automatically.

The exact blocker is residual current-state/action-set viability—especially
`wide-held-2-04` and `wide-cal-0-05`—rather than controller tracking,
controller transition, or H3 route ranking. Learned planning remains blocked
pending a prospectively authorized decision about stricter state eligibility
or a different micro action mechanism. The qualified lateral controller may
be retained as a micro-loop development component, but historical route
actions remain the only macro-JEPA actions. Integrating `vy` into macro
prediction would require new action-compatible data and one separately
authorized predictor seed.

The 100 ms micro interface remains unqualified, the full JEPA loop remains too
slow for one tick, the approximately 200 ms compute result remains
interface-unqualified, and physical/vendor stopping parity remains pending.

## Runtime and persistence

Accepted attribution, V1 requalification, successor training, final
qualification, and scientific collection runtimes sum to `1369.313 s`
(`22 min 49.313 s`). Generated and cache evidence occupies 45,133,494 bytes
(about 43.04 MiB).

The row-level ledger is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/lateral_controller_failure_attribution_and_full_budget_successor_v2/row_level_evidence_v2.jsonl`,
SHA-256 `e16cba31cb4354339dd7edca60f65e17085ce54d37482fe90d42017e4cc25612`.
Content digests are persisted beside it in `content_digests.json`.

Exactly one successor training path and one controller seed ran. No JEPA
predictor, utility model, memory, novelty, routing, beacon-capture, or
navigation system was opened, trained, or executed.
