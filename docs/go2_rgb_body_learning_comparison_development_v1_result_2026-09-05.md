# RGB/body learning comparison: completed development result

**This experiment does not support a JEPA navigation advantage.** JEPA improved
motion prediction relative to the two small neural controls, but a training-only
action average predicted motion much better than every learned model. The
supervised rollout **without** latent-prediction loss supplied the most useful
conditional action choices. Preserve it as a serious non-JEPA baseline.

All nine fits completed at their fixed 300-update budget, with three seeds per
condition and no checkpoint selection or retry. Checkpoint reload, prediction
replay, independent scalar metric reduction, source/data bindings and recorded
optimization budgets all passed audit. This is not an independent training rerun.

## Primary offline prediction results

Means over eight development layouts and three training seeds. Motion is scored
only where observed; contact remains known after a contact stop. Lower is better.

| Training / inference | Position error (m) | Yaw error (rad) | Contact Brier |
|---|---:|---:|---:|
| Training action/horizon mean | 0.0093 | 0.0072 | 0.0581 |
| Command kinematics / no contact | 0.0348 | 0.0167 | 0.0750 |
| Direct / direct | 0.0701 | 0.0558 | 0.0568 |
| Supervised rollout / direct | 0.0676 | 0.0608 | 0.0557 |
| Supervised rollout / rollout | 0.0965 | 0.0890 | **0.0438** |
| JEPA / direct | 0.0496 | 0.0405 | 0.0594 |
| JEPA / rollout | 0.1023 | 0.0712 | 0.0592 |

JEPA-minus-supervised-rollout direct-head position error is −0.0180 m;
the descriptive layout-bootstrap interval is [−0.0228, −0.0140] m. Contact Brier
instead changes by +0.00374, interval [−0.00502, +0.01150]. This is evidence of
a neural-reference motion difference, not superiority over the action-only
baseline or a contact-prediction benefit. Eight locally similar development
layouts do not support a broad maze-generalization claim.

The all-negative contact heuristic scores 92.5% accuracy because only 7.5% of
available validation horizons contain cumulative contact. Accuracy alone is
therefore misleading. At 4 s, JEPA/direct Brier is 0.1622, versus 0.1550 for
direct and 0.1538 for supervised-rollout/direct. Full per-horizon results and
cumulative-probability monotonicity diagnostics are retained in the result.

## Secondary conditional decisions on executed branches

This [separately specified secondary analysis](go2_counterfactual_decision_diagnostic_development_v1_2026-09-05.md)
was planned after seven models' aggregate prediction scores were visible but
before any decision/regret calculation. It is not retrospectively preregistered
as part of the original learning study. Every candidate outcome had already been
physically executed from its matched prefix; no learned policy was run online.

For each layout, the same three supplied body-frame intents were scored using
the fixed cost `10 * contact probability + displacement error` at 4 s. Actual
contact costs 10; otherwise actual displacement error is used. These are declared
development utility units, not a physical safety calibration.

| Inference condition | Contact choices | Stop choices | Mean regret |
|---|---:|---:|---:|
| Always stop / action-only mean | 0% | 100% | 0.297 |
| Direct/direct | 4.2% | 95.8% | 0.680 |
| Supervised-rollout/direct | 4.2% | 95.8% | 0.680 |
| **Supervised-rollout/rollout** | **0%** | **26.4%** | **0.147** |
| JEPA/direct | 0% | 100% | 0.297 |
| JEPA/rollout | 16.7% | 51.4% | 1.926 |
| Command kinematics / no contact | 25% | 0% | 2.309 |

Percentages average intents within layouts and then seeds, not independent
robot trials. Supervised rollout has zero selected contacts in all three seeds,
with regrets 0.122, 0.173 and 0.147. Its RGB shuffle increases contact choices
to 20.8% and regret to 2.059; body-history shuffle leaves its choices unchanged.
This is useful visual-dependence evidence on this panel. It does not establish
that real body sensing is unhelpful: the initial body regime varies little.

JEPA/direct avoids contact by always stopping. JEPA/rollout does not improve
those decisions. Thus neither its better latent loss nor its motion improvement
establishes useful navigation. An independent scalar check rederived all 1,536
model/control/baseline decision rows from audited full-precision horizon labels;
maximum float32-label cost difference was 4.61e−8. Choice ordering and regret
matched. The result is conditional on the particular five actions and cost.

## Likely mechanism and limitations

JEPA's validation future-embedding MSE (0.035–0.044) beats latent persistence
(0.112–0.119), but its across-layout current-feature standard deviation contracts
to 0.050–0.070, compared with approximately 0.40–0.44 in the other conditions.
This is **not complete numerical collapse**: effective rank is nonzero and
actions are distinguished. Nevertheless, RGB shuffle changes JEPA/direct Brier
by only 0.00073, versus 0.00608 for direct and 0.00823 for supervised-rollout/direct.

A plausible interpretation—not yet a causal diagnosis—is that the joint
predictive objective prioritizes easy action/body-history variation and loses
some scene discrimination. The common regularizer pools current and future
observations, so variation across actions/times can satisfy it without preserving
variation between scenes under the same action. Future control/body information
is a legitimate target but can dominate an embedding. Increasing model size or
predictive horizon is not the first response to this result.

The corpus contains only sixteen independent training contexts at one junction
approach, with four repeated exit motifs and a single visual style. Future images
enlarge training exposure but do not create more independent starting mazes.
Fixed action execution is very predictable before contact; action-conditioned
average dynamics are consequently a strong baseline. Learning scene-dependent
collision risk and preserving useful memory are different problems from fitting
that nearly deterministic motion.

## Cost and provenance

Direct has 475,717 active trainable parameters; both recurrent conditions have
612,810. Mean fit time per seed: 32.5 s direct, 33.1 s supervised rollout, 43.0 s
JEPA. Mean median CPU inference is approximately 0.78 ms for one direct candidate
or 0.91 ms for both heads plus its eight-step rollout, excluding capture and
tensor conversion. A five-candidate planner is not benchmarked by that number.

- Learning output: `.generated/go2_rgb_body_learning_comparison_development_v1_attempt_001`.
- Learning result SHA-256: `1a3c5d91910d08825487991cc999d2f0976e7bb92db620b11acc86f457b4ccb9`.
- Prediction audit SHA-256: `ae29a92c2e9a5abc721a738a42b7f19443f294a8750b89bd959dd73e4a1081e1`.
- Decision output: `.generated/go2_counterfactual_decision_diagnostic_development_v1_attempt_001`.
- Decision result SHA-256: `3b784afed79896c0b49e7a893faad17b09a8524410a7322a73806aaaca695795`.

All collection, fitting, audit and secondary-analysis processes are terminal.
The final explicit 24-file development test suite passes 366 tests; all 41 source
bindings across the three new completed experiment launches remain unchanged.
No frozen tracked source or protected material was touched. Next steps are in
the [post-comparison plan](go2_rgb_body_post_comparison_next_steps_2026-09-05.md).
