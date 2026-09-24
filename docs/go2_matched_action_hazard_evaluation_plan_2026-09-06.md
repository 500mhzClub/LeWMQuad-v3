# Matched action-hazard evidence for the independent-layout JEPA study

## Why this diagnostic is needed

Prediction error alone does not establish that a learned representation helps
choose actions. The new data provides six counterfactual action-duration cells
per context/history/support combination. The existing older decision diagnostic
uses five branches, treats action0 as stop and scores the last output slot;
those assumptions do not apply here. In this inventory action0 is a short
forward pulse, and the short/long pulses have different2.2s/2.5s terminal
endpoints. It must not be applied unchanged to this dataset.

`lewm/matched_action_hazard_evaluation_development.py` adds a secondary offline
diagnostic at exactly2 seconds after departure:the largest common regular
horizon before both pulse/brake schedules end. It neither runs a controller nor
claims that single-action hazard ranking solves navigation. The horizon is fixed
before any independent-layout model fits, not selected from their results.

## Population and exact pairing

Group by frozen layout, context, recent history and support; require all six
actions. All six raw1150-sample/nine-packet prefix witnesses must be complete
and match exactly. Require observed cumulative-contact labels for all six at
the common horizon. A collision observed earlier establishes a positive label;
a noncontact stop before the horizon leaves its outcome censored, not safe.

Retain every planned group with all missing-action, unavailable-prefix, unequal-
prefix and censored-contact issues. Training has120 planned groups across6
layouts; selection and development evaluation each have60 across3 layouts.
Uniform no-contact and all-contact groups remain observed groups but have no
hazard-ranking contrast. They must not become evidence of perfect discrimination.

For each named model or baseline, require the identical ordered evaluation-row
identities. A missing common-horizon hazard prediction in any otherwise scorable
group makes that head unavailable for the whole matched diagnostic; do not score
it on a favorable subset. The scope here is hazard outputs only:missing motion
predictions are handled by the separate motion evaluator, not disguised as
contact-prediction failure or silently repaired.

## Fixed descriptive metrics

- Contact Brier score over all six observed outcomes in each scorable group.
- Hazard concordance:each observed contact/no-contact action pair contributes1
  if the contact branch has the higher logit,0 if reversed and0.5 if tied.
  Rank logits directly so sigmoid saturation cannot manufacture ties.
- Contact fraction among exactly tied minimum-logit actions, averaged uniformly
  over the tied set. This is a tie-aware offline score, not an executed choice.
- Avoidable-contact regret:that fraction minus the smallest observed binary
  contact outcome among the six branches. All-contact groups have zero avoidable
  regret but retain contact fraction1 and no discrimination evidence.

Average groups within each layout, then layouts equally. Report exact paired
layout differences and contributing-layout counts for every metric. No frame-
level confidence interval or p-value, and no multiplication of independent N
by actions, target horizons or model seeds. Model-seed effects must later be
reported as paired seeds within the same evaluated layout population.

## Verification and remaining integration

Initial focused59283 passes61 tests in7.13s, covering20 new synthetic hazard
tests plus the existing independent-evaluation and cumulative-event suites.
Tests exercise ties, saturated extreme logits, reversed rankings, unequal layout
sample counts, missing/censored/unequal action sets, wrong row identities and a
role with no eligible samples. Synthetic predictions deliberately generated
from labels are test fixtures only, not fitted heads or scientific results.

The new function reuses the existing view's structural inventory/role/row joins.
It does not establish source provenance, physical validity or eligibility from
a dictionary of witnesses. The future study loader must bind completed fresh
batch audit receipts, verify their exact artifacts and modality-eligible windows,
and supply the audited prefixes. It must reject live/incomplete batches and
legacy/pilot data. That full-population loader and the matched study runner are
still integration work; no study fit has started.

Use this diagnostic alongside native motion/yaw errors, actual-horizon contact
metrics/calibration, empirical action/time and zero-motion controls, and
RGB/history/action ablations. Disallowed native contact is not every unsafe
event:falls, body stability and some execution-limit stops remain separate
outcomes. Even positive offline ranking results must be followed by matched
online-rollout tests, reliable local execution, useful memory/backtracking,
whole novel-maze missions, real-time/deployment-valid sensing and hardware
evidence. The ultimate goal remains unchanged and unachieved.
