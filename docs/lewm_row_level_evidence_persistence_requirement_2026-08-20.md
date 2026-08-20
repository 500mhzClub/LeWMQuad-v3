# ROW_LEVEL_EVIDENCE_PERSISTENCE

Status: prospective project-wide evaluation requirement

Effective: 2026-08-20

## Requirement

Every future learned-model evaluation must atomically persist a row-level evidence ledger before reducing predictions to aggregate metrics. The ledger must be sufficient to reproduce operating points and downstream decisions without rerunning model inference.

Every row must retain:

- stable row identity;
- state and candidate identity;
- split and family;
- raw model logits at the evaluator's full temporal/component resolution, stored at no less than FP32 precision;
- calibrated probabilities;
- authoritative labels;
- the candidate inputs needed to reproduce selection, or immutable digests and indices that bind those inputs;
- calibration temperature and threshold identities;
- component and aggregate threshold decisions;
- selected-candidate inputs and deterministic tie-break fields.

The ledger or its companion index must bind:

- model/checkpoint digest;
- input-contract and preprocessing identity;
- label and split identity;
- row count, tensor shape, dtype, and field semantics;
- a canonical content digest over the decoded arrays;
- the serialized ledger SHA-256 and byte count.

## Required reproducibility

The persisted evidence must be sufficient to reproduce, without checkpoint execution:

1. aggregate and component metrics;
2. calibration and operating-point analysis;
3. component-wise or mechanism-specific composition;
4. candidate admission and rejection;
5. state-level candidate selection and abstention;
6. per-state, per-family, and pooled planning metrics.

Evaluation code must verify row identity, label, split, and input alignment before reduction. Equality at a threshold must follow the evaluator's frozen conservative tie rule and be stored explicitly.

## Custody boundary

This requirement is prospective. Historical evaluations that legitimately persisted only aggregate metrics are not rewritten or represented as possessing row-level evidence. If historical logits must be recovered, checkpoint inference requires explicit authority and the recovered ledger must be labelled post-outcome.

The first conforming recovery ledger is:

- experiment: `MECHANISM_SPECIFIC_SAFETY_COMPOSITION_INFERENCE_RECOVERY_V1`;
- schema: `row_level_component_predictions_v1`;
- conditions: frozen `ACTION_CONTROL_ONLY` and `ENHANCED_EMBODIED`;
- purpose: reusable component composition and operating-point reduction.
