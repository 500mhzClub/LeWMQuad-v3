# Go2 V4 sealed-test invalidation

Date: 2026-07-10

Status: V4 sealed commitment invalidated before model evaluation

## Incident

During a read-only runtime-integration review, a broad repository text search
accidentally byte-read `config/go2_generalization_v4/sealed_test.json`. The
search exposed limited manifest metadata to a review agent. No image, label,
checkpoint output, navigation result, or aggregate performance metric from the
sealed role was opened, and the running N32 diagnostic has no path to this
manifest. Nevertheless, the final-test contract requires the manifest itself
to remain opaque, so V4 no longer qualifies as sealed.

The review was stopped immediately. Its post-access findings are discarded and
must not guide model, threshold, data, or runtime choices.

## Consequence

- G0/G1 development evidence remains valid.
- Train, checkpoint-selection, calibration, and train-role diagnostic evidence
  remains valid.
- The V4 development set remains development-only.
- The V4 sealed role is permanently forbidden for final evaluation.
- G8 is blocked until a fresh opaque sealed role is generated and committed.

## Replacement requirements

Before G8, create a new sealed role from a new preregistered seed and generation
namespace. It must exclude every train, development, physical-authoring, and
previous sealed scene hash. A mechanical builder may write and hash the opaque
manifest, but model-facing processes must receive only its commitment and
aggregate integrity counts. Add a repository guard test that fails any command
or helper attempting to read the sealed manifest outside the one-shot G8
launcher. Freeze that launcher, all model/code hashes, thresholds, and access
ledger before the replacement manifest is materialized for evaluation.
