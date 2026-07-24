# Go2 V4 sealed-manifest re-access record

Date: 2026-07-24

Status: recorded; V4 was already permanently invalid for G8

## Incident

During a delegated read-only source-closure audit, a broad `git grep` bypassed
the repository's `.ignore` rules and byte-read
`config/go2_generalization_v4/sealed_test.json`. The search exposed only one
commitment-SHA line to the audit agent. It did not expose a scene identity,
image, label, model output, navigation result, or aggregate performance metric.

The exact full command must be recovered from retained tool logs rather than
reconstructed from memory. The exposed line reached the delegated audit agent
and the primary review process through its custody disclosure. It was not
provided to a model, trainer, evaluator, navigator, dataset builder, or
checkpoint-selection process.

## Consequence and containment

- V4 was already permanently ineligible for G8 because of the recorded
  2026-07-10 incident. This re-access does not alter that status.
- The exposed commitment line is forbidden as an architecture, data,
  calibration, threshold, checkpoint, runtime, or successor-selection input.
- Source-closure conclusions from the delegated audit require independent
  reproduction from source paths that exclude all custody roots.
- No G2, training, dataset, navigation, benchmark, hardware, or held-out
  execution occurred.
- Future active G8 plaintext must remain outside the model-facing checkout
  under operating-system custody. Repository ignore files remain advisory
  defense in depth only.
