# RGB Swept-Progress Survival Joint-JEPA V4 Matched No-Persistence Control — Attempt V1 Failure

- Status: `FAILED_NO_RETRY_OR_RESUME`; the write-once `attempt_v1` is consumed.
- This is an execution-integrity failure receipt, not a scientific result.
- Independent failure audit: PASS; exactly one fresh science-identical integrity replacement is defensible under the user's standing permission for obvious implementation retries.
- No control checkpoint, training trace, evaluation, treatment predicate, or
  promotion decision was produced.

## Frozen bindings

- Preregistration commit:
  `3dd4ca0680347f0a7f35d42d387781ecf53b1685`.
- Pre-runtime clarification commit:
  `8cd4486ff8fc5e82dbfb745da1ed8d4b3a4101b1`.
- Frozen source commit:
  `4d55f6b68ac4edfa8aef93fdb3b2e4c7666f09e2`.
- Execution-binding commit:
  `49d281480db196187b20c34f4cb5a61beede264a`.
- Failure artifact:
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence/attempt_v1/failure.json`.
- Failure artifact file SHA-256:
  `b2a99cf0b88c918c80690620f5f9f7ee5c891fb60cde581eabe7118d3f89c6d8`.
- Failure artifact canonical content SHA-256:
  `86ce444bba577a3744606480fb08803b67ced42e02b86cae5c22c88802d685b9`.
- Failure schema:
  `lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence_failure_v1`.

## What completed successfully

- The fresh V4 control initialization was reconstructed twice from the
  authorized N320 encoder-only input. Both complete state payloads and digests
  matched exactly.
- The canonical reconstructed-state digest was
  `181b7cd4eef301a4986a9182940d0819b236ccf28876e471f5c30a62838112fd`.
  Both reconstructions recorded target hard-sync count `1` and EMA-update count
  `0`.
- The initial AdamW optimizer receipt was empty, excluded the target parameters,
  and placed every online parameter in exactly one inherited group. Its
  canonical receipt digest was
  `f45a9c253820a4bdab542e34ef07b8975bb799b7cdce2751ba781d905a386d2d`.
- The update-1 functional witness passed exactly after four backward calls and
  before the first optimizer step:
  - `S = 1.313827022910118`;
  - `P_diagnostic = 1.0`;
  - `U = 0.9792981296777725`;
  - `R = 1.0`;
  - `O = 1.026371382176876`.
- The fixed training loop reached its terminal cap with exactly:
  - `1,000` updates and `16,000` presentations;
  - `4,000` microbatch graphs and `4,000` backward calls;
  - `1,000` optimizer steps and `1,000` target-EMA steps; and
  - `4,000` predictor forwards and `4,000` predictor objectives.
- All three required gradient groups were present throughout: `encoder`,
  `lift_semantic`, and `predictor`.

## Terminal failure and root cause

- The executor failed immediately after terminal training, during its duplicate
  validation of the in-memory trace, with `RuntimeError: matched control loss
  identity changed` at failure stage `fixed_training_1000_updates`.
- The frozen training core had already validated each loss identity with
  `math.isclose(..., rel_tol=2e-6, abs_tol=2e-6)`.
- The executor then repeated the same integrity check using the stricter rule
  `absolute_error <= 1e-6`. This executor-side duplicate threshold was not
  identical to the core contract and rejected a trace that the frozen core had
  validly accepted.
- This is an integrity/schema-adapter mismatch after completed training. It is
  not evidence that the model, data, objective, optimizer, schedule, first-row
  witness, or terminal accounting failed.
- The failure artifact does not identify the offending trace row or preserve
  its two absolute-error magnitudes. This receipt does not infer those absent
  values.

## Outputs and scientific status

- Terminal control checkpoint: absent and recorded as `null`.
- Published training trace: absent and recorded as `null`.
- Calibration, checkpoint-selection, family, semantic, swept-progress, and
  inference-control evaluation: not run.
- Full-V4-minus-control deltas and the preregistered treatment predicate: not
  computed.
- There is therefore no positive, negative, or inconclusive scientific result
  about `P` from attempt V1. The treatment predicate is invalid because the
  required receipts and evaluation are incomplete.
- No checkpoint identity exists for this attempt. No checkpoint may be named,
  hashed, opened, loaded, resumed, substituted, or used for any downstream
  purpose.

## Access and custody

- The receipt records `forbidden_input_count = 0` and
  `g2_navigation_final_evaluation_open_count = 0`.
- Runtime raw-consumption roles were limited to `authority`, `index`, and
  `train`. Every forbidden semantic-input counter was zero.
- No endpoint-observation or fixed-negative evaluation RGB was requested.
- No predecessor experiment checkpoint was read, and no V4 runtime artifact was
  reopened. Held-out, sealed, G2/navigation, production, and final-evaluation
  material remained unopened.
- Drafting this document read only the exact `failure.json` named above and the
  committed preregistration, execution binding, and frozen source. It did not
  list the attempt directory or access a checkpoint, trace, dataset, evaluation
  output, or protected material.

## Lifecycle and recommendation

- Attempt V1 remains terminally consumed. It has no retry, resume, repair,
  extension, alternate loss, checkpoint recovery, or result-conditioned
  intervention authority.
- Recommendation: **authorize exactly one fresh science-identical integrity
  replacement only if independently approved.**
- Such approval should permit only the executor integrity/schema adapter to use
  the frozen core's exact loss-identity semantics and to emit complete offending
  row/error fields on any future failure. It must preserve the model, data,
  seeds, initialization, schedule, losses, direct no-`P` backward membership,
  optimizer, clipping, masks, target EMA, evaluation, thresholds, controls, and
  `1,000`-update / `16,000`-presentation cap.
- This receipt does not itself grant replacement execution, GPU, checkpoint,
  dataset, evaluation, navigation, held-out, sealed, promotion, deployment, or
  production authority.
