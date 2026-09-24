# Candidate-Conditioned Future Safety V1

Date: 2026-08-20  
Source commit: `9c7cef6a86fde16cd173ea48977458febfec64ea`  
Final classification: `TRUE_FUTURE_SAFETY_HEAD_NO_GO`

## Preserved results and scope

The experiment preserves `KINEMATIC_RESIDUAL_PLANNER_NO_SIGNAL` and `RUNTIME_SAFETY_GUARD_NO_GO_FOR_CANDIDATE_PLANNING`. It reused the frozen 48-state, 576-branch route-intent panel without simulation, rendering, target encoding, state replacement, or label changes.

One safety-head seed (`2026082003`) was trained. The true-future gate failed, so no predictor checkpoint was opened and one-step/two-step evaluation was not reached.

The exact pre-action ViT-L grid was not separately cached in this panel. The candidate-invariant H1 hold target was therefore bound as a state-shared current-context surrogate. It cannot distinguish candidates within a state, but it is a development limitation: this result is not a qualification of an exact-current-context interface.

## Evaluator-first fixture

The deterministic fixture covered perfect and reversed discrimination, one/all unsafe, one safe, false-negative admission, false-positive rejection, no admitted candidate, calibrated and miscalibrated probabilities, and JSON round-trip determinism. All checks passed.

Fixture SHA-256: `4f3c81bb68d5cdb6c3d33e16826d825d0266418845502f5bae9cd31e240c1c5e`.

## Frozen H3 safety prevalence

| Split | Rows | Aggregate unsafe | Collision/contact | Clearance | Stuck |
|---|---:|---:|---:|---:|---:|
| Fit | 384 | 277 (0.7214) | 189 (0.4922) | 14 (0.0365) | 204 (0.5312) |
| Calibration | 96 | 72 (0.7500) | 50 (0.5208) | 12 (0.1250) | 53 (0.5521) |
| Held-out | 96 | 58 (0.6042) | 24 (0.2500) | 0 | 44 (0.4583) |

Fall and unsafe-termination prevalence is zero. Complete H1–H3 split/family prevalence is stored in the machine result.

H3 component overlap across all 576 rows was: none 181; contact only 86; clearance only 6; contact+clearance 2; stuck only 119; contact+stuck 164; clearance+stuck 7; all three 11.

## Model and training

The main `CANDIDATE_FUTURE_SAFETY_HEAD_V1` follows the frozen contract: shared affine-free token normalisation and `1024→32` projection, two 3×3 spatial convolutions, spatial mean/max pooling at H1–H3, a 64-D action/control encoder, and 12 cumulative component logits.

| Model | Parameters | Epoch-1 loss | Epoch-60 loss | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| Action-only | 4,300 | 0.8916 | 0.6541 | `341bb9059fd9aacf50d6ffe2e68cb6e675c9660fe8b9ddfdb36661e281971920` |
| Current-context | 135,820 | 0.9055 | 0.4332 | `69fcfe6c6a53b387acdee8d3de70d2a15fd0bf2a110c5f6a74e49514288594b9` |
| True-future | 191,404 | 0.8605 | 0.000267 | `85d8965ab553ab4182a15f87bdb87ef65ed9f71566af0d68950ee8253201d07d` |

The near-perfect future-head fit did not transfer scene-disjointly.

## Calibration

Each learned condition used one scalar temperature and the frozen calibration rule.

| Model | Temperature | Admission threshold | Calibration unsafe recall |
|---|---:|---:|---:|
| Action-only | 1.06425 | 0.30747 | 0.9583 |
| Current-context | 2.95275 | 0.0 | 1.0000 |
| True-future | 15.96147 | 0.33281 | 1.0000 |

The current-context and true-future calibration distributions did not permit useful safe retention at the required high-recall operating point.

## Held-out branch-level results

| Condition | AUC | AP | Unsafe recall | FNR | Safe retention | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action-only | 0.7471 | 0.8350 | 1.0000 | 0 | 0.0526 | 0.1165 | 0.2088 |
| Current-context | 0.6606 | 0.7590 | 1.0000 | 0 | 0 | 0.1377 | 0.2287 |
| True-future | 0.6565 | 0.7697 | 1.0000 | 0 | 0 | 0.1720 | 0.2249 |
| Privileged static-grid guard | NA | NA | 0.6724 | 0.3276 | 0.5789 | NA | NA |
| Oracle safety | 1.0000 | 1.0000 | 1.0000 | 0 | 1.0000 | 0 | 0 |

The true-future head failed the primary discrimination and calibration gates. Its recall of 1.0 was achieved by rejecting all 96 candidates, not by useful candidate filtering.

## True-future component diagnostics

| Component | Positive rows | AUC | Recall | FNR | Precision |
|---|---:|---:|---:|---:|---:|
| Collision/contact | 24 | 0.4549 | 1.0000 | 0 | 0.2553 |
| Clearance | 0 | NA | NA | NA | NA |
| Stuck | 44 | 0.7863 | 0.9773 | 0.0227 | 0.4624 |
| Fall/unsafe termination | 0 | NA | NA | NA | NA |

The component outputs were evaluated under the one frozen aggregate temperature and admission threshold; no component-specific recalibration was performed.

## Per-family true-future result

| Family | AUC | Unsafe recall | FNR | Safe retention |
|---|---:|---:|---:|---:|
| large_enclosed_maze | 0.8125 | 1.0000 | 0 | 0 |
| medium_enclosed_maze | 0.7926 | 1.0000 | 0 | 0 |
| small_enclosed_maze | 0.8071 | 1.0000 | 0 | 0 |
| loop_alias_stress | 0.5481 | 1.0000 | 0 | 0 |

## Filtered kinematic-planning result

| Condition | States retaining safe candidate | Unsafe selections | Mean selected progress | Normalized regret | Best-safe top-3 | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| Action-only safety | 2/8 | 0 | 0.4238 m | 0.0000 (defined subset) | 0.25 | 0.75 |
| Current-context safety | 0/8 | 0 | 0 | NA | 0 | 1.00 |
| True-future safety | 0/8 | 0 | 0 | NA | 0 | 1.00 |
| Oracle safety | 8/8 | 0 | 0.2184 m | 0 | 1.00 | 0 |

The true-future model produced eight false abstentions. It therefore failed safe retention, state coverage, progress, regret, best-safe recovery, and false-abstention requirements. Predictor evaluation was prohibited by this terminal.

## Gate and interpretation

Passed: aggregate recall, false-negative rate, no only-unsafe admitted state, and no unsafe selected candidate.  
Failed: AUC, safe retention, ECE, six-state safe retention, selected progress, normalized regret, best-safe top-3, and false abstention.

The result is a valid development no-go for this post-hoc final-layer safety head. It does not weaken the deterministic kinematic route result. Under the frozen protocol, the next representation change would require explicit safety supervision during world-model training rather than another post-hoc safety architecture.

## Artefacts, runtime, and custody

| Artefact | SHA-256 |
|---|---|
| Training/evaluation source | `1c25a83750ebf41f23d86a47e110c9193547ea48824db863a16f9522877aa6b2` |
| Machine result | `c76e9d346dd1b56cdf442847bc4febe0615127728d99b441da14043f3f2d7ee0` |

Training consumed 300.48 seconds; final calibration/reduction consumed 2.52 seconds, for approximately 303.0 seconds of compute. Generated checkpoints and results occupy 1,433,776 bytes.

Exactly one registered seed was used for the safety heads. No predictor seed was opened. No simulation, rendering, target encoding, route-score modification, global memory, novelty, beacon discovery, or closed-loop navigation occurred.
