# TRUE_FUTURE_CONTACT_FILTER_DECISION_DECOMPOSITION_V1

Date: 2026-08-21

Source commit: `a2c2abfcf3d75a97371ddfbc73eaa6c79ed6f079`

Development status: `POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC`

Preserved result: `WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY`

Final classification: **`CONTACT_PROXY_FILTER_SCORE_NO_GO`**

## Claim boundary

The target remains `SIMULATED_DISALLOWED_CONTACT_PROXY`: the frozen binary robot–environment contact definition, excluding ordinary foot–ground support and robot self-contact. This is not a material-hazard, injury, property-damage, human-safety, or fragile-infrastructure claim.

No model or checkpoint was executed. The analysis used the immutable 3,456-row Stage-1 ledger; the checkpoint SHA-256 was checked as identity evidence only.

## Exact reproduction

The calibration threshold remained `0.06920835375785828`. All persisted admission and selected-candidate decisions reproduced exactly.

On calibration, that frozen threshold produced recall `0.956044`, retention `0.538071`, 20/24 states retaining a negative candidate, two selected contacts, and four false abstentions. It was selected under the historical recall/retention rule, which did not impose the later decision-level zero-selected-contact constraint.

| Metric | Reproduced value |
|---|---:|
| Contact AUC / AP | 0.931459 / 0.934783 |
| Recall / FNR | 0.942149 / 0.057851 |
| Contact-negative retention | 0.592814 (99/167) |
| States retaining an action | 21/24 |
| Selected contacts | 1 |
| False abstentions | 1 |
| Selected progress | 0.099315 m |
| Oracle-contact kinematic progress | 0.144730 m |
| Normalized regret | 0.347353 |
| Best-contact-negative top-3 | 0.500000 |

## Held-out threshold frontier

All 290 thresholds induced by the held-out calibrated probabilities were enumerated. This is a post-hoc diagnostic; no held-out threshold was adopted.

- Complete-gate operating points: **0**.
- Contact-filter-only operating points: **0**.
- Maximum negative retention at recall at least 0.95 was `0.580838` at threshold `0.0667026`; 19 states retained an action, but one contact was still selected, regret was `0.274837`, and top-3 was `0.500000`.
- With zero selected contacts and recall at least 0.95, maximum retention fell to `0.485030` at threshold `0.0571636`; only 16 states retained an action and six false abstentions occurred.
- The minimum regret at recall at least 0.95 was `0.115694`, but retention was `0.233533`, only eight states retained an action, and 14 false abstentions occurred.

The frontier therefore rules out calibration as the sole explanation. No threshold can simultaneously satisfy recall/FNR, retention, state availability, and zero selected contacts.

Frontier SHA-256: `52ed45d42323ee093fd0e6fc85cfd9cae21370f188e64721462be1bb6a2b7111`.

## State-level filter decomposition

Exclusive counts:

| Classification | States |
|---|---:|
| `BEST_SAFE_RETAINED_AND_SELECTED` | 5 |
| `BEST_SAFE_RETAINED_BUT_MISRANKED` | 5 |
| `BEST_SAFE_REJECTED` | 10 |
| `ABSTAINED_WITH_SAFE_CANDIDATE` | 1 |
| `SELECTED_CONTACT_POSITIVE` | 1 |
| `NO_CONTACT_NEGATIVE_CANDIDATE` | 2 |

| Family | Retained+selected | Retained+misranked | Best rejected | False abstention | Selected contact | No negative |
|---|---:|---:|---:|---:|---:|---:|
| Large | 3 | 2 | 1 | 0 | 0 | 0 |
| Medium | 0 | 1 | 4 | 0 | 1 | 0 |
| Small | 0 | 1 | 3 | 0 | 0 | 2 |
| Loop alias | 2 | 1 | 2 | 1 | 0 | 0 |

| State | Family | Classification | Best negative / admitted / nominal rank | Selected | Contact | Stuck | Progress (m) |
|---|---|---|---|---:|---:|---:|---:|
| wide-held-0-00 | large | retained and selected | 0 / yes / 1 | 0 | no | no | 0.4671 |
| wide-held-0-01 | large | retained and selected | 0 / yes / 1 | 0 | no | no | 0.4191 |
| wide-held-0-02 | large | retained but misranked | 3 / yes / 3 | 1 | no | yes | 0.1968 |
| wide-held-0-03 | large | retained and selected | 0 / yes / 1 | 0 | no | no | 0.1697 |
| wide-held-0-04 | large | retained but misranked | 3 / yes / 3 | 1 | no | no | 0.3545 |
| wide-held-0-05 | large | best rejected | 2 / no / 4 | 10 | no | yes | -0.0109 |
| wide-held-1-00 | medium | best rejected | 10 / no / 1 | 11 | no | no | -0.1168 |
| wide-held-1-01 | medium | best rejected | 11 / no / 4 | 6 | no | no | -0.1000 |
| wide-held-1-02 | medium | best rejected | 7 / no / 8 | 10 | no | no | -0.0440 |
| wide-held-1-03 | medium | retained but misranked | 0 / yes / 9 | 2 | no | no | -0.0500 |
| wide-held-1-04 | medium | selected contact | 9 / yes / 6 | 0 | yes | no | 0.2097 |
| wide-held-1-05 | medium | best rejected | 11 / no / 4 | 4 | no | no | -0.0880 |
| wide-held-2-00 | small | no negative candidate | — | hold | — | — | — |
| wide-held-2-01 | small | best rejected | 1 / no / 8 | 6 | no | yes | -0.0172 |
| wide-held-2-02 | small | retained but misranked | 1 / yes / 2 | 0 | no | no | 0.3481 |
| wide-held-2-03 | small | best rejected | 0 / no / 1 | 3 | no | no | 0.2328 |
| wide-held-2-04 | small | no negative candidate | — | hold | — | — | — |
| wide-held-2-05 | small | best rejected | 9 / no / 6 | 1 | no | yes | 0.2084 |
| wide-held-3-00 | loop | abstained with negative | 11 / no / 8 | hold | — | — | — |
| wide-held-3-01 | loop | retained but misranked | 5 / yes / 4 | 7 | no | yes | -0.0109 |
| wide-held-3-02 | loop | best rejected | 10 / no / 1 | 5 | no | yes | -0.1605 |
| wide-held-3-03 | loop | retained and selected | 10 / yes / 1 | 10 | no | yes | 0.0291 |
| wide-held-3-04 | loop | best rejected | 8 / no / 10 | 3 | no | yes | -0.0566 |
| wide-held-3-05 | loop | retained and selected | 4 / yes / 1 | 4 | no | no | 0.1052 |

By family, best-safe rejection affected one large, four medium, three small, and two loop states. The additional loop false abstention also rejected its best negative candidate.

## Candidate-set upper bounds

Oracle-progress ranking uses the frozen realised route-intent partial order: distance differences over 0.03 m, then heading differences over 5 degrees.

| Condition | Contacts | Stuck | Progress (m) | Heading (rad) | Regret | Top-1 / top-3 | Abstentions | Oracle fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Model filter + kinematic | 1 | 8 | 0.099315 | 0.118423 | 0.347353 | 0.2273 / 0.5000 | 3 | 0.6862 |
| B. Model filter + oracle progress | 2 | 7 | 0.122385 | 0.173472 | 0.245751 | 0.4545 / 0.5000 | 3 | 0.8456 |
| C. Oracle contact + kinematic | 0 | 12 | 0.144730 | 0.089219 | 0.191065 | 0.4545 / 0.8182 | 2 | 1.0000 |
| D. Oracle contact + oracle progress | 0 | 7 | 0.179305 | 0.035034 | 0.000000 | 1.0000 / 1.0000 | 2 | 1.2389 |
| E. Model filter + kinematic + oracle stuck tie-break | 1 | 7 | 0.098848 | 0.089013 | 0.348263 | 0.2273 / 0.4545 | 3 | 0.6830 |

The admitted set bounds progress even with oracle route ranking, while oracle contact filtering makes the unchanged kinematic ranker meet the aggregate regret and top-3 targets. The stuck tie-break removes only one stuck selection and does not improve progress or regret.

Current-versus-oracle-contact kinematic family results:

| Family | Current progress / regret / top-3 | Oracle-contact progress / regret / top-3 |
|---|---|
| Large | 0.2661 / 0.2314 / 0.8333 | 0.3055 / 0.0977 / 1.0000 |
| Medium | -0.0315 / 0.7118 / 0.3333 | 0.0563 / 0.1974 / 1.0000 |
| Small | 0.1930 / 0.2629 / 0.2500 | 0.2477 / 0.1955 / 0.5000 |
| Loop alias | -0.0187 / 0.1896 / 0.5000 | 0.0038 / 0.2752 / 0.6667 |

The complete per-family metric tree for all five conditions—including contacts, stuck selections, progress, heading improvement, regret, top-1/top-3, and abstention—is persisted in the machine-readable result. Notably, oracle-progress ranking under the model filter selected two contacts, while the oracle-contact/oracle-progress bound selected none and reached `0.179305 m` progress with zero regret.

## Progress-loss attribution

The positive per-state gap from model-filter-plus-kinematic to oracle-contact-plus-kinematic affected eight states and totalled `1.257545 m` when holds are assigned zero progress. Categories overlap and their loss totals are therefore not additive.

| Category | Affected states | Positive-gap sum | Family distribution |
|---|---:|---:|---|
| Filter loss | 11 | 1.204588 m | large 1, medium 4, small 3, loop 3 |
| Ranking loss | 6 | 0.322078 m | large 2, medium 1, small 2, loop 1 |
| Recoverability/no-useful-progress loss | 10 | 0.818748 m | large 2, medium 2, small 2, loop 4 |
| Candidate-bank limitation | 2 | 0 m | small 2 |

Overlaps were filter+recoverability in seven states, ranking+recoverability in two, and filter+ranking in one. Best-safe rejection is the dominant decision defect, while ranking and recoverability remain material secondary limitations.

## Score-margin analysis

The seven contact false negatives had probabilities `0.03557–0.06670`, median `0.04312`, and mean signed margin `-0.02182` below the threshold. One was within 0.01 of the threshold; six were 0.01–0.10 below it; none was more than 0.10 below it. Errors are therefore mixed moderate-margin errors, not a single isolated threshold tie and not extremely deep errors.

The selected contact was `wide-held-1-04:00` (`straight_fast`, medium maze), probability `0.057164`, or `0.012045` below threshold. False negatives occurred in all families—large 1, medium 1, small 1, loop 4—and across seven different primitives.

The model rejected 68 contact-negative candidates. Their median probability was `0.13943`; 14 were within 0.01 above threshold, 23 were 0.01–0.10 above it, and 31 were more than 0.10 above it. By contrast, 99 negative candidates were admitted. This broad high-confidence rejection of negative candidates explains why threshold adjustment cannot retain mobility while eliminating the selected contact.

## Decision

Primary classification: **`CONTACT_PROXY_FILTER_SCORE_NO_GO`**.

Secondary findings:

- `BEST_SAFE_REJECTION_DOMINANT`
- `MEDIUM_MAZE_ROUTE_FAILURE`
- `LOOP_ALIAS_ROUTE_FAILURE`
- `CANDIDATE_BANK_LIMITATION`

Calibration is not the sole bottleneck because the held-out oracle frontier contains neither a complete-gate point nor even a contact-filter-only point. The score rejects many route-useful contact-negative candidates yet still admits a contact that the kinematic selector chooses.

Close `WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1`. Do not train its candidate-conditioned predictor. No automatic successor is authorised; a future model must prospectively change its architecture, temporal target, contact-proxy definition, or sensor coverage.

A contact filter is not useful merely because it attains high contact recall. It must retain enough contact-negative actions to let the safety-related task proceed. Contact avoidance and task/recoverability performance remain separate requirements.

## Evidence, runtime, and custody

- Result SHA-256: `f3103eb1418294516cab9abdda70d2b4617cb0d5707d854f8eb1d0601b5889ec`
- Result content digest: `7e95adbc95619f7a688ae48a4ac7d547752a1c4ce28ab5f02f5b97aa40941df7`
- Result size: 170,918 bytes
- Frontier size: 36,687 bytes
- Result plus frontier storage: 207,605 bytes
- Diagnostic runtime: approximately 0.50 seconds

No training, inference, simulation, rendering, encoding, threshold adoption, JEPA access, navigation, memory, novelty, or beacon work occurred.
