# Go2 DINOv2 physical-readout calibration integrity replacement V1

**Frozen:** 2026-08-03, after the original attempt ended before any
calibration-role RGB access and before any DINO or readout computation, and
before this replacement opens an evaluation-role RGB leaf.

## Purpose and scope

This is one science-identical integrity replacement for the frozen Go2 DINOv2
physical-readout calibration V1. It is not a retry or resume of the consumed
original attempt.

The original scientific contract remains exactly the preregistration at:

- 10,285 bytes, SHA-256
  `ff6e42042792ffc66c51ac9e6fd31d9da194cb22c5526edfd1ce3cfe22db55ee`.

The original source review remains:

- 6,927 bytes, SHA-256
  `2e0305154674da1f39a621d4ac90e58652721403dde7e0cb1d782b8f79944174`.

The original authority remains consumed and grants no replacement authority:

- 6,407 bytes, SHA-256
  `3a403377b071a9f916882cc5315b6b5be4d097a6c215562a565245962e6e2cc2`.

Its exact terminal-failure review is:

- 6,201 bytes, SHA-256
  `7f99e3136857a5149acfa74daed5f2ba54be5110942544c2df5aa230e0dd7ea9`.

That review establishes `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`, no
scientific result, no calibration train/evaluation RGB access, no DINO encoder
load, no token extraction, no readout fit, and no evaluated scientific gate.
The original root remains immutable and terminally consumed.

## Exact failure and sole permitted change

The failure occurred while the frozen posthoc bundle loader recomputed an
upstream, independently reviewed, development-only task-relevance adequacy
result. In the exact authorized ROCm environment the recomputation was
canonical-exact in every field except:

`measurements.pixels.minimum_reference_candidate_rgb_ssim`

- stored: `0.999873849744854`;
- recomputed: `0.9998738497448542`.

This one-ULP measurement difference is far from the unchanged minimum SSIM gate
of `0.99`. The same upstream evaluator already permits absolute tolerance
`1e-12` for this field when it recomputes the parity result, but the later
task-relevance wrapper compared the complete adequacy document byte-exactly.

The replacement may add only a local compatibility admission around that one
task-relevance recomputation:

1. Load the stored adequacy result at exactly 94,165 bytes and SHA-256
   `5094104ac29b4652cd577015c5fbf23b42f0768c78a205cbf07a77d992339ca7`.
2. Load its independent review at exactly 2,080 bytes and SHA-256
   `29eb00a486604824effb56502194855553f87c81a9691d4075a5810273c92ca9`.
3. Call the unchanged live task-relevance evaluator, whose frozen replacement
   source review must bind its exact source.
4. Require stored and recomputed documents to have identical schemas, PASS
   statuses, thresholds, decisions, bindings, inventory, counts, keys, list
   lengths, and all other values.
5. Permit only the named SSIM scalar to differ. Both values must be finite and
   `math.isclose(stored,recomputed,rel_tol=0.0,abs_tol=1e-12)` must be true.
6. Require both values to remain at or above the unchanged `0.99` gate and the
   recomputed result to retain
   `PASS_TASK_RELEVANT_INPUT_ADEQUACY_DEVELOPMENT_ONLY`.
7. For the exact comparison expected by the frozen outer loader, return the
   already reviewed stored document only after checks 1--6 pass.

An exact recomputation with no differing field is also admissible. Any other
changed path, nonfinite value, threshold/status change, binding drift, source
drift, or difference larger than `1e-12` is an infrastructure failure.
Thus the complete allowed differing-path set is a subset of the singleton
containing only the named SSIM field; the three CPU-runtime latent drifts are
not admissible in this bound ROCm replacement.

The compatibility admission must be scoped to a context manager, restore the
original evaluator in `finally`, and leave every shared frozen validator and
artifact unchanged. The original posthoc loader then runs unchanged and must
reproduce and byte-check the same derived bundle leaves.

## Science-identical contract

The replacement must not change any of the following from the original
preregistration and reviewed source:

- the 128 train and 128 evaluation states, 32 scene identities, role split,
  artifact order, labels, dense-rank recomputation, or one-centimetre physical
  equivalence tolerance;
- the exact 302,107,682-byte frozen DINO train cache or its 1,770-byte receipt;
- the official DINOv2 source commit, `dinov2_vits14` architecture, 88,283,115-
  byte checkpoint, preprocessing, token normalization, shape, or float16
  storage;
- the 3,072-dimensional quadrant descriptor, relational feature, task
  conditioning, three separately fitted nine-head ridge sets, lambda, target,
  tie-break, or action IDs;
- any evaluation arm, metric, equal-family scene-cluster bootstrap, resample
  count, bootstrap seed, or gates 1 through 7;
- the zero-event safety interpretation;
- the one-open evaluation RGB rule or cache-only deterministic replay;
- any terminal status or claim scope.

The exact authorized environment remains Python `/usr/bin/python3.12`, Torch
`2.14.0.dev20260726+rocm7.1`, HIP `7.1.52802`, NumPy `1.26.4`, and Pillow
`10.2.0`.

## Replacement source and review

The replacement may add only:

- one pure compatibility-admission module;
- focused compatibility tests;
- one narrow replacement runner that validates replacement authority and
  invokes the frozen original `execute_v1` under the scoped admission;
- focused replacement-runner tests;
- one independent replacement source review;
- one replacement execution authority;
- terminal review documents after execution.

The replacement runner must bind and rehash the original preregistration,
original source review, original authority, original consumed reservation and
terminal, original terminal-failure review, stored task-relevance result and
review, complete original source closure, replacement source closure, fixed
posthoc inputs, DINO cache, DINO repository and checkpoint, environment, and
output root. All bindings and DINO repository state must be checked again after
execution.

Focused tests must prove at least:

- exact documents pass;
- the one SSIM field passes within `1e-12`;
- the SSIM field fails above `1e-12` or below the `0.99` gate;
- any second changed field fails, including a latent diagnostic;
- any status, threshold, binding, inventory, count, schema, or finiteness drift
  fails;
- the context manager restores the original evaluator on success and failure;
- the frozen posthoc loader is still invoked;
- the original attempt root is rejected;
- PASS, scientific STOP, and infrastructure FAIL propagate unchanged from the
  original executor;
- no replacement path can reopen train RGB or evaluation RGB during replay.

## One-shot authority and output

One separately reviewed authority may create only:

`.generated/dev/go2_dinov2_physical_readout_calibration_v1/attempt_v2_integrity_replacement_v1`

The output may contain one exclusive reservation, one compatibility receipt,
the evaluation feature cache and receipt, `result.json`, and `terminal.json`.
The compatibility receipt must record the stored and recomputed SSIM values,
absolute difference, tolerance, exactness of every other field, both statuses,
and exact input/source bindings. It is infrastructure evidence, not a new
scientific measurement.

The receipt must be written exclusively after compatibility admission passes
but before the strict loader returns and before any evaluation RGB access or
DINO work. A receipt-publication failure is an infrastructure failure and must
terminalize the replacement without scientific evaluation; the receipt may
not be appended only after the frozen executor has already emitted a
scientific terminal.

The replacement may open each of the same 1,536 evaluation-role RGB artifacts
exactly once only after strict bundle and compatibility admission passes. It
must not open train RGB, collect or generate data, access held-out or sealed
material, train or fine-tune a model, change a scientific threshold, retry,
resume, overwrite, promote, deploy, or create any other output root.

The scientific terminal statuses remain exactly:

- `PASS_DINO_TRUE_FUTURE_PHYSICAL_READOUT_HEADROOM_ESTABLISHED`;
- `STOP_DINO_TRUE_FUTURE_PHYSICAL_READOUT_HEADROOM_NOT_ESTABLISHED`; or
- `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`.

The replacement result, if PASS or STOP, is the sole calibration result. The
original consumed attempt remains a cited infrastructure failure and is never
relabelled.
