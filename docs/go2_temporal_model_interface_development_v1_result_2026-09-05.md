# Temporal model interface: completed readiness checks

The new temporal RGB/body model and matched loss/schedule implementation are
ready for training-runner integration. The **no-fitting** check passed48 actual
corpus windows: initial stop contexts and the last observed forward-branch context
from every one of24 layouts. This is an interface check, not learned performance.

The model consumes four chronological RGB/body/control packets and explicit
known-action masks. It rejects privileged extra modalities, noncausal plan gaps,
partial unsupported blocks, nonzero unknown commands and invalid command values.
Unknown future blocks do not advance the recurrent predictor. Both heads mark
their unsupported horizons invalid. Tests verify that earlier predictions are
unchanged by later commands or shortening the plan after those predictions.

JEPA targets are single future packet embeddings from an EMA observation encoder.
The past-history GRU supplies the predictor's starting context; no future history
is fabricated and no target enters inference. Direct, supervised-rollout and
JEPA objectives expose their online encoders to the same past/future images via
the common regularizer. Outcome losses average equally over sampled windows;
one window per layout therefore supplies equal outcome weight despite varying
remaining duration. Latent/regularization terms have their separately documented
embedding weighting.

The actual-data check exercised the full1,200-update seed0 index schedule without
executing updates, verifying16 unique layouts per batch. A training-only actual
minibatch produced finite gradients in every active parameter for all three
conditions. Validation labels were not used for that gradient check. No optimizer,
EMA update, fitting or checkpoint write occurred; every model state tensor was
verified unchanged afterward. There are575,045 direct-arm active parameters and
712,138 in each matched rollout arm.

The22 new focused tests cover these semantics, matched observation exposure,
target gradient isolation, EMA arithmetic and rejection of role leakage or missing
layout/action populations. The combined explicit30-file suite passes452 tests in
4.44seconds (session13101, exit0). They do not prove model accuracy, learned scene use,
closed-loop safety, persistent memory or hardware validity.

## Next executable package

Implement the runner and mask-aware metrics for the
[fixed temporal comparison](go2_temporal_rgb_body_learning_comparison_development_v1_2026-09-05.md).
Materialize and bind all three seed schedules; verify source dependencies and
audited input identities; preserve all conditions, budgets and failure accounting.
Metric tests must include empty offset/horizon cells, censored motion, masked
monotonicity and layout-first reductions. Add training-only empirical controls
with reported fallback coverage and the specified initial-context action-choice
analysis. Then launch the single bounded nine-model training package.

Keep this readiness evidence and previously completed datasets/models unchanged.
The fixed comparison is not yet fitted or evaluated. Earlier JEPA negatives
remain the scientific evidence until a new completed result says otherwise.
Next system milestones remain successive sensor-only replanning, observed
place/frontier memory, exploration/beacon return and bounded real-platform work.

## Evidence identity

Root: `.generated/go2_temporal_model_interface_development_v1_attempt_001`.
Interface session34495 terminated exit0 with PASS48.

Launch SHA-256: `01bb1c11200522e6e64982f2a574d766bbfa38db0664404240dca81fdb59be2b`.
Result SHA-256: `d7782c9adc75714c4179039fa179d2caec2e37c5ad2c5c5a96bcc229835a958f`.
The launch records the exact five new design/model/loss/test/checker sources and
the audited input result/tensor-check identities. It is not a recursive clean
export certification or a training-source closure. The later training launch
must bind the imported source dependencies as well.

Final-goal status remains active and unachieved. This turn produced tested source
and actual-input readiness evidence, not a wait or blocker recurrence.
