# Balanced-feature replay and inner-arrival evidence

The complete paired replay rejects the balanced frontend as a general replacement.
The distinct 4 cm internal planner/controller remains a source-tested engineering
candidate, not a physically validated fix. The latest executed room assay remains
0/3 full returns. No new physics, training, learned-policy deployment or hardware
execution occurred in this work; the full scientific goal remains unachieved.

## Full recorded-stream comparison

Both observers processed the same 4,705 recorded RGB-D/body packets. Original
poses exactly reproduce all 4,685 original decisions, including failed decisions.
Failure-latched frames remain in coverage denominators. Both complete estimate
streams were written before native pose was loaded for evaluator-only scoring.

| Recorded stream | Frames | Original available | Balanced available | First failure |
| --- | ---: | ---: | ---: | --- |
| Nominal left | 2,413 | 2,413 | 1,888 | Balanced at 1,888 |
| Nominal right | 1,820 | 1,809 | 1,820 | Original at 1,809 |
| Lower-friction left | 472 | 472 | 472 | Neither |
| Total | 4,705 | 4,694 | 4,180 | Balanced loses 514 available frames overall |

Balanced features pass the right-hand interval that defeats the original, but
that is only eleven extra recorded frames, including the terminal zero-command
drain. It does not establish continued navigation or reaching home. Balanced
left fails against every retained reference while original left remains valid.
There is one balanced fallback at low-friction frame 367; original needs none.
Neither observer repairs the low-friction motion-model failure.

Median available-observer times are original/balanced 40.09/51.68 ms (left),
40.79/51.45 ms (right), and 38.55/49.57 ms (low friction). These are sequential
recorded-replay diagnostics, not isolated hardware or end-to-end deadline tests.
The descriptor budget is held at 600, but candidate detection costs more.
This variant also selects one orientation per half-pixel location before
matching; the result does not isolate spatial quotas from that selection change.
Lower maximum error on a shortened surviving prefix is not proof of improvement.
Do not select a frontend retrospectively per trial or weaken registration gates.

Replay `.generated/go2_balanced_feature_replay_v1_attempt_001` completed with
exit 0 (handle 79182), including final source/input/artifact verification.
Launch SHA256: `c33cea0bcd34476680c686b13553964b4659357bf16dda9dfd74e7ac3b756021`.
Result SHA256: `4710a28bab24babe71b6f48a288c9e2f9795937e11638996bbd38e50a3791ed3`.
The frozen [protocol](go2_balanced_feature_replay_v1_2026-09-06.md) specifies
unchanged matching, gyro, consensus, support, increment and conflict gates.

## Consistent 4 cm internal arrival region

Distinct inner-goal modules use 4 cm consistently in beam-search arrival,
candidate ranking, returned status and controller settling. External native
acceptance remains 6 cm. Signed yaw, ten new quiet intervals, original excursion
limits and local/mission budgets remain unchanged. This 2 cm engineering reserve
is not a calibrated uncertainty bound.

Thirteen focused tests pass, including both signed seven-stage missions in
synthetic and older-empirical model-matched plants. The full explicit 204-file
regression suite passes: **2,584 tests in 210.17 seconds**, handle 86942 exit 0.
That suite includes nine balanced-frontend tests; passing implementation tests
does not override the negative recorded-stream result.

The [recorded arrival diagnostic](go2_recorded_inner_arrival_diagnostic_2026-09-06.json)
checks all fifteen actual nominal completion states and the low-friction stop,
using sensor pose and the unchanged older nominal table, never native pose.
It binds the source closure and recorded inputs before and after calculation.
All fifteen nominal states yield predicted 4 cm candidates under remaining
local/mission pulse budgets and existing predicted-endpoint excursion checks.
Seven already meet the inner region; eight need 1–9 additional model pulses.
The failed left intermediate hold needs one predicted pulse; final home needs
nine, with a minimum 220 additional ticks including the final hold. Lower bounds
fit remaining local/mission time, but do not include all settling delays.
The low-friction stop only yields SEARCH_EXHAUSTED, not a complete candidate.

These are separate local, offline questions, not a counterfactual full-mission
replay. Changed earlier actions would alter later states, anchors, images and
budgets. More corrective pulses may increase tracking exposure. The initial
read-only diagnostic command (handle 4123) failed because JSON identity lists
were passed to a tuple-only contract; no result was produced. The explicit
diagnostic script converts that serialization boundary and completes with
exit 0 (handle 13699), without changing recorded data or controller contracts.

## Next experiment and scientific path

1. Retain the original frontend as the current baseline. Diagnose the balanced
   left failure through descriptor matching, flow, depth lifting and rigid
   pruning before proposing another estimator change. Do not launch a combined
   inner-controller/balanced-frontend variant as if both were improvements.
2. Test the inner controller in a distinctly frozen, prospectively paired
   physical simulation using the original frontend and unchanged external
   criteria. Keep both turn signs and the low-friction challenge. Same-condition
   pairing tests the arrival change; later independent starts/appearances are
   necessary for generalization. Retain every failed hold and full sensor audit.
3. In parallel scientific development, quantify actual action-response residuals
   against the fixed table and collect independent state/action/layout coverage.
   Backward/arc commands require measured controllability before planner use.
   Fit matched direct, supervised-rollout and JEPA arms only after the dataset
   and seed/split schedule are fixed; current room targets have no positive
   contacts and cannot substantiate hazard learning or maze generalization.
4. Demonstrate observed branch choices, useful persistent memory, actual
   backtracking and home verification, then independent-layout and matched
   training/rollout/memory comparisons. Restore realistic sensing, deadlines,
   self-occlusion and body-sweep evidence before bounded hardware claims.

The full [execution-to-JEPA plan](go2_intent_return_to_jepa_next_steps_2026-09-06.md)
remains in force. These results support the next experiment, not goal completion.
