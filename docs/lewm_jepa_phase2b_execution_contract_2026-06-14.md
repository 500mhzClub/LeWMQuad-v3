# JEPA Phase 2B Bounded Execution Contract

Date: 2026-06-14

## Question

Does training the encoder end to end to create action-predictable spatial
tokens improve held-out counterfactual prediction and action ranking over a
matched pooled-CLS LeWM?

This is a bounded learnability gate. It is not a final navigation or scaling
claim.

## Data

Train and evaluation are scene disjoint.

Each split contains:

- eight scenes, one from each registered scene family;
- eight source states per scene;
- nine candidate sequences per state;
- three deterministic candidate buckets: safe positive progress, kinematic
  unsafe, and safe other;
- two action blocks and two future-observation slots per sequence.

Every candidate remains in the dataset contract. Token prediction loss uses
only complete valid future observations. Missing and renderer-invalid
observations remain explicitly reported and are not treated as collision
labels.

## Factorial

All cells use the same data, seed, optimizer, epochs, and reduced architecture:

| Cell | State | Anti-collapse |
| --- | --- | --- |
| `pooled` | CLS | CLS SIGReg |
| `spatial_var` | ordered patch tokens | appearance CLS SIGReg plus spatial variance floor |
| `spatial_no_var` | ordered patch tokens | appearance CLS SIGReg only |

The reduced architecture is used to test whether the objective is learnable on
CPU before spending on the full default model. A later promotion experiment
must use the default models, which are capacity matched within `+0.27%`.

## Required Diagnostics

Prediction error alone is insufficient because an end-to-end encoder can make
prediction easy by making every state similar. Every cell must report:

- free-running versus persistence at each horizon;
- real-action versus zero-action prediction;
- real-action versus shuffled-action prediction;
- target step-change magnitude;
- mean feature standard deviation and collapse warning;
- goal-conditioned sequence selection consequences;
- valid-observation coverage and scene overlap.

## Gate

A spatial cell passes the bounded gate only if, on disjoint scenes:

- it does not trigger the collapse warning;
- real actions beat both zero and shuffled actions;
- free-running prediction beats persistence at one block;
- it improves over pooled CLS on two-block free-running prediction;
- safe positive progress improves without increasing newly unsafe selection.

Failure at the one-block persistence or action-sensitivity gate stops scaling.
If `spatial_var` passes while `spatial_no_var` collapses, the variance floor is
retained for the full-capacity experiment. If both fail, redesign the spatial
target or anti-collapse objective before adding recurrence.

## Execution Result

The bounded factorial is complete:

| Cell | Feature std | Step 1 / persistence | Step 2 / persistence |
| --- | ---: | ---: | ---: |
| pooled | 0.777 | 2.07x | 4.60x |
| spatial + variance | 0.947 | 2.69x | 3.68x |
| spatial without variance | 0.027 | 21.53x | 116.49x |

The unregularized spatial representation collapsed in every epoch. The
regularized spatial representation remained non-collapsed and improved the
two-step ratio over pooled, but persistence won at the first block in every
epoch. Real actions did not meaningfully outperform zero or shuffled actions.

The result fails the registered gate and the decision is:

`stop_and_redesign_before_scaling`

Action-sensitivity gates now require an advantage of at least 10% of the
target's actual latent change. This prevents microscopic numerical differences
from being treated as evidence that the model uses its action input.

Full interpretation and the Phase 2C EMA-teacher control are recorded in
`docs/lewm_jepa_phase2b_phase2c_findings_2026-06-14.md`.
