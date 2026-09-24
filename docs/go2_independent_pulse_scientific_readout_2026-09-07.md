# Scientific readout for the fixed independent-layout study

Prepared before the original 36-fit study starts. This adds analysis code, not
another experiment, training gate, resource review, or change to the frozen
study. The collector and original learning experiment retain priority. No
partial collection outcomes or model results were used to choose these
contrasts. The original experiment definition remains
`3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`
(786 sources).

## What the result should answer

1. **Does latent prediction help?** Compare JEPA's primary recursive outcome
   head with the supervised-rollout primary head, within each of the four
   retrained input treatments. This is the closest matched intervention on the
   additional latent-prediction objective. It does not isolate every aspect of
   representation learning or establish a navigation-policy benefit.
2. **Does the recursive prediction package help?** Compare supervised rollout
   with direct prediction. The direct arm does not train the recursive branch;
   a difference includes architecture/active-parameter and loss differences.
   JEPA versus direct is a combined-package contrast, not isolation of JEPA's
   latent objective.
3. **Do learned predictions beat simple controls?** Compare each full-input
   primary model with action/time empirical prediction and zero-motion with
   empirical contact prediction. Beating zero motion while losing action/time
   means learned prediction has not beaten the stronger simple control.
4. **Which information is useful?** Within each objective compare full input
   with no RGB, latest packet only, and no candidate command. These remove
   information during both fitting and inference. Latest packet retains its
   internal histories; no candidate command retains duration/time and past
   applied controls. Neither is an ablation of *all* memory or action.

All 27 contrasts are reported, using fixed primary heads. Auxiliary heads remain
in the original experiment output but cannot replace a losing primary head.
Training, selection and development-evaluation roles remain separate. No best
seed, intermediate checkpoint, favorable horizon, or favorable stratum is
selected. The analysis includes every saved actual-time horizon and action,
context, history and support stratum, as well as the all-horizon summaries.

## Measurement and uncertainty

The readout reconstructs left-minus-right differences from saved per-layout
scores, not from the saved pairwise or macro summaries. For position error,
yaw error, Brier score, minimum-risk selected contact fraction and avoidable
contact regret, lower is better. Hazard concordance is oriented the other way;
raw signs are also preserved. The common contact ranking horizon remains two
seconds. Exact logit ties retain the original scorer's uniform-over-ties
interpretation; this is an offline diagnostic, not an executed action.

Three optimization seeds repeat on the same development-evaluation layouts.
There are **three layout units, not nine independent trials of generalization**.
The report retains every seed-by-layout cell, the per-seed means over layouts,
and the per-layout means over seeds. No frame/horizon-level confidence interval,
p-value, significance claim or probability-calibration claim is produced.

A complete-population macro is `null` if any required seed/layout value is
missing. An explicitly named observed-subset mean is shown separately; it cannot
be called the full-population result. Missing heads, baseline cells, target
counts, excluded episodes and contact-group availability remain visible.
Unavailable contact contrast is not perfect discrimination. Position, yaw and
contact findings remain separate, including possible tradeoffs.

An all-negative difference matrix is a consistent descriptive direction on this
small development cohort, **not an automatic practical-benefit pass**. The
6 cm training loss scale is not an accuracy acceptance threshold. Practical
utility must be demonstrated in physical command selection, with collision,
progress, tracking, intervention and full-loop timing outcomes.

## How the readout is obtained

`scripts/read_go2_independent_pulse_science_v1.py` accepts
`--result-sha256` with the actual terminal study result digest. It rejects an
absent/failed/incomplete study, altered output hashes, a different original
definition, missing score roles, changed primary-head mappings and promoted
claims. It authenticates the fixed original 36-fit/43,200-update terminal roster
and all bound outputs, then aggregates the nine seed/role score reports. Source
definition and output hashes are rechecked after aggregation.

The reader prints JSON and does not create an output root, train a model,
deserialize a checkpoint, rerun inference, access partial raw collections, or
launch physics. Its numerical core is
`lewm/independent_pulse_scientific_readout_development.py`. Both are outside the
frozen 786-source study closure; the original runner is unchanged.

Authentication is not independent reconstruction of optimizer updates or raw
prediction scores. The report states that limitation. Before substantive
scientific claims, inspect original losses/accounting, final predictions,
coverage and physical error strata; independently check the important raw
comparisons. A successful readout does not release any navigation or hardware
qualification gate.

## Next experiments after the result

- If JEPA improves over supervised rollout and simple controls, take that fixed
  candidate to a matched **online rollout versus no online rollout** intervention.
  Keep sensing, action candidates, safety supervision and physical budget
  matched. Compare goal progress and failures, not only predictive loss.
- If JEPA beats direct but not supervised rollout, attribute the result to the
  combined recursive prediction package; do not claim a latent-objective gain.
- If the empirical control wins, preserve the negative result. Inspect training
  loss, train-to-development error and sensor/action strata to distinguish
  optimization, information and transfer problems. A new optimization study
  needs a prospectively specified budget and fresh development evidence; do not
  tune the current evaluation layouts or indefinitely delay physical progress.
- If RGB removal does not hurt, RGB utility is unestablished at this budget and
  task. That is not proof RGB is intrinsically useless. Later scene-dependent
  action tests must require information the action/time prior cannot supply.
- If motion and hazard conclusions differ, retain both. Good hazard ranking
  without goal progress is not navigation, and low displacement error does not
  establish collision avoidance.

Regardless of the sign of JEPA's result, the next fixed physical observation
challenge remains the eight-trial tracking experiment after the original study
completes and its existing launch gates pass. Its result must establish actual
turn/translation coverage, retained failure behavior and sensor-only tracking
before a fresh closed-loop adoption test. Reliable low-friction execution and
the full-loop deadline remain unresolved. Online memory/backtracking, novel-maze
mission completion and bounded real-platform evidence remain later requirements;
none is replaced by this prediction analysis.

## Implementation verification at this handoff

The focused readout/scoring run passed 72 tests. The final six-file regression
passed **178 tests** in 38.60 seconds, with zero failures, errors or skips in the
independently parsed JUnit report. It covers the new reader and numerical
analysis, existing prediction/contact scorers, original study orchestration,
matched training mechanics and input ablations. These are synthetic/component
tests, not 178 scientific trials or new learned-model results.

The actual read-only definition check returned the unchanged 786-source study
identity above; none of the new analysis paths belongs to that closure. The
study output root was absent. The original supervisor PID 2063013 and `l08`
worker PID 2120652 were observed live; the latter's terminal audit was absent.
Eight completed layouts / 960 eligible trials remain the last verified count.
The 36-fit study and physical tracking challenge have not started.
