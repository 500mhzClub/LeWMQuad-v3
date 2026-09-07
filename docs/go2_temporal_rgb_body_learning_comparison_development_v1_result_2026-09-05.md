# Temporal RGB/body comparison V1: completed scientific result

All nine models completed their fixed 1,200 updates and passed the full
checkpoint/prediction audit. **This comparison does not establish JEPA utility.**
Its direct head is more conservative and has worse contact prediction than the
matched supervised model; its rollout head has worse motion, contact prediction
and initial-context action choices. No model was replaced or extended.

The [design](go2_temporal_rgb_body_learning_comparison_development_v1_2026-09-05.md)
uses four-frame context, 610 training windows from 16 layouts and 304 validation
windows from eight layouts. The validation layouts have been used in earlier
development studies: these are paired development results, not a final held-out
maze benchmark. The four-times-larger training budget and expanded supervision
also prevent a clean temporal-history-only comparison against the old study.

## Prediction and decision results

Prediction entries below average within layout, then across eight layouts and
three seeds. Later-state scores cover 982 pre-contact motion targets and 1,090
observed contact targets per model. Lower errors and Brier scores are better.

| Training / inference | Later position error (m) | Later yaw error (rad) | Later contact Brier |
|---|---:|---:|---:|
| Training action/remaining-plan mean | .00284 | .00258 | .07853 |
| Command kinematics / no contact | .01850 | .01287 | .10013 |
| Direct / direct | .03468 | .03071 | .02076 |
| Supervised rollout / direct | .03196 | .02665 | .01979 |
| Supervised rollout / rollout | .04611 | .03514 | .01782 |
| JEPA / direct | .03094 | .02451 | .02404 |
| JEPA / rollout | .09963 | .06369 | .02453 |

The action-only mean dominates learned motion prediction on this deterministic
development corpus. Neural models contain useful scene-dependent contact
information; their motion accuracy must not be praised using only weak controls.
The empirical baseline needed no fallback for any of 1,410 known validation cells.

Initial-context decisions use five actually executed candidate branches and the
fixed three intents/cost. Counts below are **selections**, not new physical
collisions: 3 seeds × 8 layouts × 3 intents = 72 correlated decisions per arm/head.
Several selections can refer to the same recorded physical branch.

| Training / inference | Contacting selections / 72 | Stop selections / 72 | Mean executed-branch regret |
|---|---:|---:|---:|
| All stop (also empirical action-mean choice) | 0 | 72 | .29663 |
| Direct / direct | 0 | 56 | .20937 |
| Supervised rollout / direct | 0 | 52 | .19436 |
| Supervised rollout / rollout | 2 | 22 | .34772 |
| JEPA / direct | 0 | 70 | .28558 |
| JEPA / rollout | 10 | 47 | 1.49252 |

Supervised rollout's lowest contact Brier does not translate automatically into
better decisions: its rollout regret varies from .0492 to .8872 across seeds,
with both contacting selections in seed2026091700. JEPA-direct selects stop in
every case for two seeds. This is limited progress, not effective maze navigation.

Paired differences average seeds within layout before a 10,000-resample layout
bootstrap. Intervals are descriptive, uncorrected for multiple comparisons and
based on only eight reused development layouts:

- JEPA-direct minus supervised-direct: position −.00102 m [−.00270, +.00027],
  contact Brier +.00425 [+.00070, +.00880], regret +.09123 [+.04936, +.14020].
- JEPA-rollout minus supervised-rollout: position +.05352 m [+.05025, +.05642],
  contact Brier +.00671 [+.00087, +.01693], regret +1.14480 [+.29799, +2.44366].
- Within JEPA, rollout minus direct regret is +1.20694 [+.14752, +2.52833].
- Supervised-direct minus direct-only regret is −.01501 [−.04823, +.01755]; the
  difference is uncertain. Do not select a best seed or claim a decisive winner.

## What the diagnostics mean

RGB-history shuffling strongly degrades contact prediction in every new arm.
On the same 248 eligible windows (56 incomplete cross-layout cells omitted),
JEPA-direct Brier changes from .02644 to .18040; supervised-direct changes from
.02177 to .17983. Body-history shuffling changes these scores little. The
temporal JEPA demonstrably depends on RGB in this diagnostic; its failure cannot
simply be described as ignoring images. This is not proof of causal sensor
necessity on hardware or a complete representation-collapse diagnosis.

Same-action/offset scene feature standard deviations average about .33–.35 for
JEPA versus .45–.53 for the supervised controls. This is compatible with weaker
scene separation but does not identify the cause of poor decisions. There was
no loss-weight search or retrospective change to model costs.

Single-thread inference medians are roughly 1.49–1.52 ms for a direct head and
1.87–1.91 ms for both heads plus rollout, for one context and one plan. These
exclude acquisition, tensor conversion and multi-model/multi-candidate costs;
they are not demonstrated robot control-loop latency.

## Important remaining coverage gap

Later windows use the same fixed branch action that generated their past
history. Thus temporal coverage is broader, but candidate-action coverage after
movement is not. A model may exploit correlation between past and future action;
this result does **not** prove that it ignores the prospective command. More
importantly, it cannot verify prediction accuracy for braking, reversing or
changing turn direction at those later states. Replanning requires precisely
such counterfactual action changes.

Do not respond with another latent coefficient sweep. Preserve the result,
then test successive sensor-only decisions and explicitly record action changes,
stops, contacts and continuation. If switched-action failures occur, collect
actual alternative-action suffixes from matched moving states before fitting
another predictor. The [next execution plan](go2_temporal_post_comparison_execution_next_steps_2026-09-05.md)
keeps the full exploration/return objective, rather than declaring a conservative
local predictor to be a navigation system.

## Verification and identity

Training session94822 and full audit81706 both terminated exit0. Training took
1,275.28 seconds including loading/evaluation. All checkpoint predictions replay
exactly (maximum difference 0.0), all schedules and initial-state bindings match,
and independent scalar primary metrics/action costs agree. Both interim audits
are retained and superseded by the full audit. The source suite has 470 combined
passes plus nine separately passing live-history tests; no frozen tracked source
was changed or protected benchmark material accessed.

Root: `.generated/go2_temporal_rgb_body_learning_comparison_development_v1_attempt_001`.

- Launch: `a4e63ced7df5b8fef20b449037577ea9e473a0d40e5eccb44670e28f7dfc4055`.
- Schedules: `76dce2fb58d6a28dfc0ba38404a6c135b11b337f9fd55e23bb3f5bca75a99db8`.
- Result: `64c878d4a7524712eaba1648a5f0b251bf8015ddeace1ba991b358b19abf30e5`.
- Full audit: `9f65ca24c3c9ba4197ed7e3a5e6ef633226381c2030202c098fc368cee1f873d`.

Final-goal status remains active and unachieved: no new learned closed-loop maze
execution, online place/frontier integration, beacon return or hardware result.
