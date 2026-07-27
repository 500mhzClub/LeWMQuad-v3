# Geometry-Anchored Two-Mode Event-Delta Joint-JEPA V2 runtime-delegation integrity replacement

Date: 2026-07-27

Status: preregistered for source implementation, source-only CPU testing,
independent review, and a later separately authorized one-shot execution. This
document does not itself authorize runtime execution.

## Decision basis

The only V1 attempt is consumed and permanently closed. Its independent
terminal audit is
`docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1_terminal_audit_2026-07-27.json`
at commit `2f88edb653a93c5b9a98cfa8792a73fe4900fc9f`, raw SHA-256
`38417a41b0483cbba318fffb9460a14d021c07141525a4f19f2a2748e9398495`,
content SHA-256
`594db612aa4561688343f76c1c6f8579ac307f5f5289d72c58cef6ac20a41111`,
and 15,588 bytes.

That audit establishes a valid normal terminal receipt chain at update zero,
with zero training updates, zero scheduled presentations, zero objective or
backward calls, zero optimizer or EMA updates, and no predictor or joint
training. Fourteen update-zero structural conjuncts failed only because
`reviewed_model_source_synthetic_witness_sha256` was `null`. The exact expected
model-test SHA-256,
`09170a2cceb297df65bfd6c3bf6f4f3aedda077777c8f837095cbde3a53198d6`,
is present in the frozen source manifest, independent review, and current
source bindings. V1 therefore falsified its final runtime delegation and source
witness propagation, not the learned two-mode event-delta mechanism.

The root cause is one final-control-transfer defect. The event-delta V1 runner
first installed its reviewed event `_execute`, `_terminal_failure`, observation,
and training hooks into the frozen base runner. Its public `main`, however,
then called the predecessor rigid runner's `main`. That predecessor performed a
second rebind and replaced the base `_execute` with the rigid wrapper. The event
observation hook still ran, but the event `_execute` wrapper that populates
`_ACTIVE_SOURCE_BINDINGS` and owns the registered terminal lifecycle was no
longer the final executor. The source-test witness therefore had no runtime
binding.

Standing user authority permits one obvious science-identical correction when
the scientific mechanism has not been trained or tested. V2 is exactly one
separately preregistered integrity replacement. It is not a retry, resume,
repair, reuse, or continuation of the V1 root.

## Sole permitted implementation delta

V2 may change only the final runner and launcher delegation after the complete
V1 event-delta rebind:

1. load the frozen V1 event-delta runner and the V2 governance contract;
2. rebind the complete frozen V1 event model, objective, training, observation,
   accounting, access, warning, publication, and terminal hooks to V2 identity;
3. assert immediately before execution that the frozen base runner is bound to
   the V2 contract and to the V1 event `_execute`, `_terminal_failure`,
   `_load_post_reservation_stack`, `_parameter_receipt`,
   `_evaluate_observation`, and `_train_probe` functions; and
4. transfer control directly to the frozen base runner `main`, without calling
   any predecessor `main` that performs another rebind.

The launcher must analogously transfer control directly to the already rebound
frozen base launcher. The V1 event `_execute` must receive the exact reviewed
source map, populate `_ACTIVE_SOURCE_BINDINGS` before update zero, and make the
model-test witness SHA equal the manifest-bound value above. The complete V1
normal and operational terminal lifecycles must remain the final bound
lifecycles.

No fallback hard-coded witness SHA is permitted. No V1 source may be edited.
`MODEL_TEST_RELATIVE_PATH` must continue to name the frozen V1 model test; only
the generic V2 contract/runner/launcher/checker test aliases may name the new
combined V2 test module.
No model, data, seed, schedule, loss, threshold, coefficient, initialization,
optimizer, EMA, gate, warning rule, work count, receipt field structure,
inventory, semantics, lifecycle, or cap may change. The only permitted receipt
identity changes are the V2 schema prefix, experiment ID, governing bindings,
and new output root required to keep V2 distinct from consumed V1. No V1
runtime receipt, trace, checkpoint, tensor, state, optimizer, RNG, or model
output may be a V2 input.

## Required source-only falsification

Before source freeze, focused CPU/source-only tests must prove all of:

- importing the V2 runner, launcher, and checker in isolated `-I -B` mode
  imports neither Torch nor NumPy;
- the final V2 runner boundary has the V2 contract and V1 event `_execute`,
  `_terminal_failure`, `_load_post_reservation_stack`, `_parameter_receipt`,
  `_evaluate_observation`, and `_train_probe` identities;
- calling V2 `main` under a stubbed frozen-base `main` reaches that base directly
  and no predecessor `main` or later rebind runs;
- an event `_execute` boundary supplied a synthetic reviewed source map before
  its stubbed inherited body runs, and the update-zero reviewed model-test
  witness path and SHA are non-null and exact;
- the event normal and operational receipt lifecycle functions remain the
  frozen V1 functions at the final boundary;
- the embedded frozen V1 scientific contract and every V1 scientific
  component hash listed below are unchanged; V2 may add only versioned
  governance, output-root, schema-identity, and integrity-delta metadata;
- the exact 98 V1 source paths are unchanged and the V2 closure adds only one
  lean contract, runner wrapper, launcher wrapper, recursive closure checker,
  and focused test module; and
- no generated input, dataset row, RGB, raster, checkpoint, tensor, runtime
  output, trace, accelerator, navigation, G2, held-out, sealed, rejected, or
  production material is opened.

Constructed CPU values and temporary test directories are permitted. They are
not scientific presentations and may not create the prospective output root.

## Frozen scientific identity

V2 reuses the complete frozen V1 science and model. It preserves:

- RGB-only 112-pixel inputs, patch size 7, 16-by-16 tokens, width 192;
- the unchanged geometry-anchored deformable BEV lift, semantic head, latent
  width 64, and N320-compatible initialization;
- the unchanged fixed-identity zero-event and learned-event predictor with
  exactly 231,505 parameters in 15 tensors;
- the exact per-cell latent normalization, stop-gradient EMA current/next
  delta target, fixed zero mean, learned event mean, learned spatial event
  prior, stable mixture energy, T400, B400, and all ablations;
- the exact development data roles, rows, endpoints, mappings, negatives,
  actions, family populations, input bindings, and loader behavior;
- seed, deterministic controls, presentation schedule and hashes, effective
  batch size 16, microbatch size 4, and four backwards per update;
- updates 1-400 perception warmup and updates 401-1000 genuinely joint online
  encoder/lift/semantic-head/predictor training against the stop-gradient EMA
  target;
- every semantic, persistence, action, target, context, and combined objective,
  weight, reduction, comparator, and threshold;
- every update-zero, update-100, update-400, update-401, shared-gradient,
  update-1000, perception-retention, anti-collapse, mechanism, custody,
  accounting, warning, numerical, and terminal gate;
- AdamW groups, learning rates, weight decay, clipping, precision, target hard
  sync, EMA momentum 0.996, and update order; and
- one attempt, no retry or resume, at most 1,000 updates, 16,000 scheduled
  presentations, and 30 active GPU minutes.

The frozen V1 science-contract SHA-256 is
`26c095f0b330e6e43952814e6a3b910f15b72a906d1c2f3d931a70c959ae6974`.
The frozen V1 model, objective, optimizer, schedule, and gate-threshold hashes
are respectively
`4c84691d76eaf2c3b5eee345bb3b1c9cf8dd747e9512fc91c9d6f74b37337b03`,
`85017d1618e75970a2e70e1ace6f6930650aa5b351c60855753bcdceaa3515d4`,
`2bb70f943838b656540b3dac3b6e0f30bb384547180270274abfc5077e264b34`,
`bc0ad45c06171cff7533fbfcb054e5afecf6086de0a58060c35cb5ca0256c2e3`,
and `97fa8bb4b2740e68cadf90974ab80ff33419a854b07a16a258e2f49c3f177036`.
The frozen work-accounting, warning-policy, and runtime-input-template hashes
are respectively
`013837055e693ae754324d7c9b8b098d47efed5f569505cf0f58fca8b432e359`,
`01a958d0de33a399453c7262d07f6328aabb3bbeaa83cfa045f52cdd03b6a67b`,
and `393563699929bbfd7ca4d9c97c2c63b8a2583bfcc093f61ca0926cb63d24924b`.

The frozen V1 source closure is commit
`c414231d6d0e0d0cbf9282aec16944d4d4b7cfca`, raw manifest SHA-256
`f87aa717fd118f3fb6e0a0e169dd0f4aec812f5a305cf95eb5b809e0c6c13e50`,
content SHA-256
`db5c7fdab152f75a3bafd7c94ba555bac5c5441e44fbb1ddb7ddb439ae74aa70`,
source-bindings SHA-256
`d7f6d4302c6e5ab6ff1ce24089ba8c7b20df80dda92dcaea3897ccb200315f8b`,
98 sources, and 33,275 bytes. The independent V1 source review is commit
`60dea0ae159db279643e5dafbd5c5aa4701f436b`, raw SHA-256
`c22857709ff8eb6128e7957a45eb2ab6e1dae697dc9b7be1afa8b67ab3811177`,
content SHA-256
`c4bc70ccea0bb90c1d79f942c9a627c41c3de333635467b60614ff7028de1a4e`,
and 50,356 bytes. The consumed V1 authorization is commit
`9b5a5594c7bb2f7fd79f56dab83649b0eaca16b6`, raw SHA-256
`1eb762cbac646553bb3bda481478032a08b2a3dbec03eb3598a9991b4a800eba`,
content SHA-256
`520411f30753f9b4781d0bdfa09cdcd170b1cb5528a7b0d4718305832bcfb4b1`,
and 38,836 bytes. It is custody evidence only and grants no V2 execution
authority.

## V2 identity, receipts, and lifecycle

The V2 experiment ID is
`geometry_anchored_two_mode_event_delta_joint_jepa_v2_runtime_delegation_integrity_replacement`.
Its sole output root is
`.generated/go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v2_runtime_delegation_integrity_replacement/attempt_v1`.
The root must be absent before authorization and reservation. Any reservation
or partial attempt consumes it permanently. The V1 root and every V1 runtime
file remain closed and may not be reopened, copied, renamed, or reused.

Source implementation may add only a lean V2 governance contract, runner
wrapper, launcher wrapper, recursive closure checker, and focused test module.
It must not add or modify a model implementation. The recursive closure is the
exact frozen 98-source V1 closure plus those five additive V2 files.

Execution requires, in order: this committed preregistration; a committed V2
source freeze and recursive manifest; independent zero-findings source and
science-identity review; a distinct committed one-shot authorization; a final
absent-root check; and one exact launcher invocation bound to the frozen review
and authorization hashes.

Normal scientific terminal receipts remain exactly `reservation.json`,
`metrics.json`, `artifact.json`, `access.json`, `result.json`, and
`completed.json`. An operational or integrity exception retains the complete
V1 compact `failure.json` plus `completed.json` path after the reservation.
Every terminal path remains self-contained, write-once, and sealed. Checkpoints
and the training trace remain write-only and unqualified until a later separate
terminal audit explicitly authorizes only public receipt inspection.

There is no V2 retry, resume, second seed, extension, same-root reuse,
threshold change, coefficient change, mode change, model change, or second
integrity replacement authorized here. A scientific failure after scheduled
presentations closes this mechanism family. An operational failure consumes V2
and authorizes no automatic V3.

A V2 pass authorizes only an independent public-receipt terminal audit and a
later decision. It does not authorize checkpoint or trace reads, G2,
navigation, held-out, sealed, production, promotion, or deployment access.

No generated input, dataset, RGB, raster, checkpoint, tensor, runtime output,
trace, accelerator, navigation, held-out, sealed, or rejected material was
opened to write this preregistration.
