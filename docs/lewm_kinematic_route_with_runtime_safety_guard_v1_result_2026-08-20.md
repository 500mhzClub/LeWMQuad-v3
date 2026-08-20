# Kinematic Route with Runtime Safety Guard V1

Date: 2026-08-20  
Source commit: `026cd389deb5fa106b8a023530ac32e77013a34b`  
Terminal classification: `RUNTIME_SAFETY_GUARD_NO_GO_FOR_CANDIDATE_PLANNING`

## Scope and preserved result

`KINEMATIC_RESIDUAL_PLANNER_NO_SIGNAL` remains unchanged. This experiment trained no model, opened no predictor, and executed no candidate branch. It recovered the planning-time inputs for the existing runtime guard at the 48 frozen pre-action states and compared its 576 verdicts with the frozen H3 path-safety labels.

The evaluated obstacle adapter is the navigation benchmark's default `privileged_manifest_grid`. It is future-blind but explicitly deployment-invalid; therefore even a positive result could only have qualified a development scaffold.

## Guard-input contract

| Field | Frozen value |
|---|---:|
| Grid cell size | 0.05 m |
| Manifest-grid inflation | 0.20 m |
| Capsule nose/tail radius | 0.25 m |
| Continuation horizon | 2 primitive blocks |
| Admission threshold | feasible fraction >= 0.70 |
| Hold handling | always admitted |
| Candidate input | first primitive of each registered candidate sequence |
| Query pose | frozen current simulator pose |
| Deployment valid | no |

Each state was deterministically redriven for 40 blocks to its registered pre-action boundary. The frozen start pose and all candidate identities were hard-checked. Snapshot digests matched for 36/48 states; the same 12 states had already produced the same digest nonmatches in the earlier V2 deterministic replay. This is recorded as provenance metadata rather than reinterpreted as a new discrepancy.

## Determinism fixture

`purpose-0` and `purpose-1` were independently restored and evaluated twice. Verdicts, feasible fractions, and candidate order were identical. Simulator, controller, runner episode-step, and simulated-time counters were unchanged across every guard call. The fixture passed.

## Held-out guard confusion matrix

The primary development evaluation contains 8 held-out states and 96 candidates: 58 frozen unsafe and 38 frozen safe.

| Quantity | Value |
|---|---:|
| Unsafe rejected (TP) | 39 |
| Unsafe admitted (FN) | 19 |
| Safe admitted (TN) | 22 |
| Safe rejected (FP) | 16 |
| Unsafe recall | 0.6724 |
| Unsafe false-negative rate | 0.3276 |
| Safe specificity / retention | 0.5789 |
| Precision among rejected candidates | 0.7091 |
| Balanced accuracy | 0.6257 |
| Candidates admitted / rejected | 41 / 55 |

The guard retained at least one safe candidate in 7/8 states, but `purpose-10` admitted five candidates and every admitted candidate was unsafe. This violates the frozen qualification rule.

## Held-out family results

| Family | Unsafe recall | False-negative rate | Safe retention | Balanced accuracy |
|---|---:|---:|---:|---:|
| large_enclosed_maze | 0.7500 | 0.2500 | 0.2500 | 0.5000 |
| medium_enclosed_maze | 0.5556 | 0.4444 | 0.3333 | 0.4444 |
| small_enclosed_maze | 0.5000 | 0.5000 | 0.7000 | 0.6000 |
| loop_alias_stress | 0.8000 | 0.2000 | 1.0000 | 0.9000 |

## Held-out component sensitivity

Component labels come from the already existing deterministic Route-Intent V2 replay. The frozen aggregate label remains authoritative; 18/576 replay aggregates differ from the original frozen aggregate.

| Component | Positive rows | Recall | False-negative rate | Specificity |
|---|---:|---:|---:|---:|
| Collision/disallowed contact | 24 | 0.9167 | 0.0833 | 0.5417 |
| Clearance violation | 0 | NA | NA | 0.4271 |
| Stuck | 44 | 0.6591 | 0.3409 | 0.5000 |
| Fall | 0 | NA | NA | 0.4271 |
| Unsafe termination | 0 | NA | NA | 0.4271 |

The dominant failure is missed stuck/path-unsafe behaviour. Even collision/contact recall (0.9167) remains below the required 0.95 aggregate guard recall.

## Per-state held-out audit

| State | Family | Safe | Admitted | Admitted safe | Only unsafe admitted |
|---|---|---:|---:|---:|---:|
| purpose-10 | large_enclosed_maze | 1 | 5 | 0 | yes |
| purpose-11 | large_enclosed_maze | 3 | 1 | 1 | no |
| purpose-22 | medium_enclosed_maze | 8 | 5 | 3 | no |
| purpose-23 | medium_enclosed_maze | 7 | 4 | 2 | no |
| purpose-34 | small_enclosed_maze | 5 | 12 | 5 | no |
| purpose-35 | small_enclosed_maze | 5 | 2 | 2 | no |
| purpose-46 | loop_alias_stress | 8 | 11 | 8 | no |
| purpose-47 | loop_alias_stress | 1 | 1 | 1 | no |

## Frozen qualification gate

| Requirement | Result |
|---|---:|
| Unsafe recall >= 0.95 | fail (0.6724) |
| Unsafe FNR <= 0.05 | fail (0.3276) |
| Safe retention >= 0.50 | pass (0.5789) |
| At least 6/8 states retain a safe candidate | pass (7/8) |
| No state admits only unsafe candidates | fail |
| No complete family collapse | pass |

Because the guard failed, the runtime-guarded kinematic planner was not evaluated. Frozen-label diagnostic baselines were still reducible without executing branches: unguarded kinematics selected unsafe candidates in 3/8 states (0.375), with mean selected distance progress 0.2343 m and best-safe top-3 recovery 0.75. Oracle-safety plus kinematics selected no unsafe candidate, had mean selected progress 0.2184 m, and recovered the best-safe candidate in the top three for all states. These are diagnostic bounds, not a guarded-planner result.

## Artefacts and runtime

| Artefact | Location | SHA-256 |
|---|---|---|
| Evaluator/recovery source | `scripts/run_kinematic_route_with_runtime_safety_guard_v1.py` | `b8361d5ef9debdfc1b50b327940984affa46720eafaf771d2c51d878cc0cf027` |
| Complete machine result | `.generated/kinematic_route_with_runtime_safety_guard_v1/result.json` | `80c484e6919a8d9ffd4758f2b425222b2a2379f2f988cf4d61e4f00c79e162f0` |

The 48 state records and result occupy approximately 236 KiB. Diagnostic logs occupy approximately 2.2 MiB on the cache filesystem. Total observed development/recovery wall time was approximately 17.5 minutes; the final parallel recovery batch took 8 minutes 28 seconds and the reducer took under 0.2 seconds.

## Interpretation

The unchanged planning-time guard is insufficient for candidate planning on this frozen panel. It misses 19/58 unsafe held-out candidates and permits a state in which every admitted candidate is unsafe. The result does not weaken the established action-only route-ranking result; it localises the remaining problem to candidate-conditioned future safety. No model was trained, no candidate branch was rerun, no predictor checkpoint was opened, and no global memory, novelty, or beacon-capture layer was implemented.
