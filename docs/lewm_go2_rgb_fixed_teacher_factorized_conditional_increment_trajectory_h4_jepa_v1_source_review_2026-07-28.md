# Go2 RGB fixed-teacher factorized conditional-increment trajectory H4 JEPA V1 source review — 2026-07-28

## Review decision

- Status: `CLEAR_FOR_ONE_BOUND_PREFLIGHT_AND_ONE_ATTEMPT_IF_PREFLIGHT_PASSES`.
- Preregistration commit: `ce6a461`.
- Model and runner implementation commit: `89439e3`.
- Final operational-receipt and source-proof freeze: `40da82b`.
- This review authorizes one exact preflight. If and only if that preflight is
  clean, it authorizes one fresh capped probe at the registered output root.
- It authorizes no retry, resume, repair attempt, second seed, scale extension,
  checkpoint read, navigation run, held-out/sealed access, promotion, or
  deployment.

## Frozen primary bindings

| Source | SHA-256 | Bytes |
|---|---|---:|
| Preregistration | `ea1feaed1195e0b3bb6c289053249f6d59051fe9226cf5685132331044b5daa7` | 11,576 |
| Factorized model | `ff4b0b00b9f6bc165c603ea729b0082d5fc543e02f379427cf288ad8337d2f8c` | 15,009 |
| Bound runner | `459b1a2837704e4c9534ef3229070461f507630e07cb77ad5bacb96aac3f0c56` | 19,239 |
| Model tests | `4ddc648f97d6c87ee52b182f3ca92a444f84aee17509d50a3e72cd53da1f5af8` | 19,073 |
| Runner tests | `3201ef89cf4b762c0aab2327226f6aa8a2063a6eb12fcdc3a40619b9233de8a2` | 22,419 |

- The runner's external self-binding must be exactly:
  - `LEWM_FACTORIZED_CONDITIONAL_INCREMENT_TRAJECTORY_H4_V1_WRAPPER_SHA256=459b1a2837704e4c9534ef3229070461f507630e07cb77ad5bacb96aac3f0c56`
  - `LEWM_FACTORIZED_CONDITIONAL_INCREMENT_TRAJECTORY_H4_V1_WRAPPER_BYTES=19239`
- A source-only closure call resolved all 14 frozen dependencies exactly,
  including the V2 wrapper `b8d4f861...798b14e` / 10,129 bytes, V1 factual
  runner `693cbea4...cd977` / 34,730 bytes, V2 adapter
  `3d49e710...c51ea` / 21,001 bytes, V2 builder
  `6d4dc0ad...31ce4` / 7,995 bytes, and every transitive model/encoder/shared-
  runner dependency.

## Scientific and algebra review

- The exact step is implemented as preregistered:
  `v_hat = W0(d + B(z,h,d) * (1+tanh(D(d))) * c_a)`, followed by the inherited
  normalized local step.
- `D` is one non-affine LayerNorm plus bias-free Linear and maps exact zero to
  exact zero. The already computed `D(d)` enters the action-free spatial
  belief block directly; there is no duplicate learned increment adapter.
- The complete categorical action tower is evaluated on the fixed nine-row
  learned table and centered after the entire tower. The selected action code
  is the only current-action input to the learned correction.
- `W0` is one shared bias-free Linear initialized to exact zero and is the only
  post-sum map. There is no direct `z -> increment` or `B -> increment` path.
- Consequently the uniform-action mean pre-normalization increment is the
  learned inertial baseline `W0(d)`. A generic current-state successor cannot
  bypass requested action. If the action route collapses, the unchanged
  action/HOLD gates force STOP.
- The fixed `1` permits action-conditioned motion when incoming `d` is zero;
  the model is not structurally locked at rest. HOLD is not special-cased.
- `p0` uses exact zero incoming increment; `p1` uses factual normalized
  `e1-e0`; the packed future belief uses factual `e2-e1`; later increments are
  exact post-renormalization `next_z-z`. No absent predecessor is synthesized.
- Both observed priors are formed before destination RGB insertion. Future RGB
  remains confined to the fixed no-grad target encoder. The same transition
  and head are called on all six edges, and all four particles remain coherent
  and mode-permutation invariant.
- The only target is the unchanged accepted N320 RGB encoder. The online
  encoder, history/increment path, action table, spatial belief, particles, and
  predictor remain one jointly trained JEPA graph with one summed backward and
  optimizer step. No separately trained predictor or decoder exists.

## Prior-mechanism boundary

- This is not patch-residual V1-V6: there is no whitened pair target, scalar
  action gain, trained corruption hinge, action-indexed CE/NLL, flow/warp, or
  inverse head.
- This is not geometry Action-Query V1: there is no all-nine successor rollout
  or action-classification objective, geometry/BEV/raster/semantic target, or
  current-plus-action direct successor bank. Computing all nine small action
  codes is solely the algebraic centering operation.
- It introduces no numeric command semantics or linear speed/arc/yaw
  superposition assumption. It changes the V2 transition information flow and
  nothing in the data, loss, optimizer, selection rule, scientific threshold,
  or cap.

## Inherited science verified unchanged

- Causal V2 train index: SHA-256
  `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77`,
  10,328,000 bytes, 16,000 rows.
- Causal V2 validation index: SHA-256
  `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6`,
  1,317,888 bytes, 2,048 rows.
- V2 manifest: SHA-256
  `d19fd672d9878e064b20e40a12ce84849f0a13af05a73d2281505ea8d331a36e`,
  26,926 bytes.
- Exact objective remains weight-one online/fixed-teacher history alignment
  plus half all-six factual local energy score plus half cumulative H4 energy
  score. All training controls remain disabled.
- The runner literally reuses the V2 factual evaluator and wraps the V2
  decision only to relabel terminal identity. Tests compare and preserve all
  32 gates, diagnostics, thresholds, and the selection result.
- Seed `20260727`, batch `16`, observations `0/250/500/750/1000`, AdamW
  settings, group clipping, 1,000 updates, 16,000 train presentations, 10,240
  validation presentations, 183,680 expected RGB opens, and 5,400 GPU seconds
  are unchanged.
- The inherited receipt phrase claiming a science-identical V1 model is
  explicitly rejected and replaced with the truthful statement that only the
  exact causal V2 schedule is reused with a new factorized model.

## Test and inventory evidence

- Torch runtime:
  `/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python`.
- Full focused plus inherited compatibility suite: 90 passed, zero failed and
  zero warnings in 5.63 seconds. This includes factorized model/runner, factual
  V1 model/runner, V2 schedule runner, local-innovation model/runner, and
  trajectory-distribution model/runner tests.
- Full-size source-only parameter inventory:
  - encoder: 2,747,520 scalars / 78 tensors;
  - history: 124,224 scalars / 7 tensors;
  - predictor: 594,624 scalars / 20 tensors;
  - total trainable: 3,466,368 scalars;
  - fixed target: 2,747,520 non-trainable scalars.
- Groups are disjoint, cover every and only trainable online parameter, and
  exclude the fixed target. First-step gradient staging and opened-head
  gradients including a finite nonzero online-encoder path, exact update-zero
  persistence, causal timing, recursive realized increments, centered codes,
  no-bypass behavior, objective arithmetic, target detachment, K4 shapes, and
  mode permutation are all directly tested.
- Runner tests reject every bound-input substitution and all resume, prior-
  checkpoint, seed, batch, update, presentation, and GPU-cap surfaces. Literal
  seed/cap/accounting values and the complete inherited 32-gate decision are
  frozen.
- A mechanism-local operational adapter produces exactly the cross-bound
  `failure.json`, `failure_access.json`, and `completed.json` set for caught
  in-process failures. It records a complete registered counter snapshot
  through handler entry, preserves extra counters, derives all eight forbidden
  counters instead of hard-coding zeros, and cannot turn a forbidden access
  into a normal completion. Hard process death, pre-catch failures, and receipt-
  write failures are explicitly outside that completeness claim.
- Two independent source-only reviews returned **CLEAR** after commit
  `40da82b`. They independently checked the algebra and causal boundaries,
  opened-head encoder/factor gradients, exact V2 inheritance, all CLI locks,
  normal and failure counter truthfulness, handler identity, receipt cross-
  bindings, and the 14-path source closure. Neither review opened generated or
  runtime inputs.

## Custody and one-shot authority

- Exclusive output root:
  `.generated/go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_jepa_v1/probe_v1`.
- The root must be absent before reservation and be created mode `0700` by the
  inherited runner. Reservation precedes schedule, accepted N320, RGB, or torch
  runtime work.
- Preflight may open only the bound schedules and accepted N320 initialization
  needed by the inherited preflight contract. It performs zero RGB opens,
  updates, validation presentations, and output checkpoint writes.
- If preflight passes, execute exactly once with no retry/resume. Runtime
  checkpoints and traces remain write-only. Terminal audit may read only the
  canonical JSON receipts and must not list, stat, hash, or open `.pt` files.
- Test, held-out, sealed, navigation, labels, arbitrary checkpoints,
  predecessor predictors, retry, and resume remain forbidden with exact-zero
  counters.
- PASS would establish bounded development perception/world-model evidence
  only. STOP closes this exact mechanism. Neither result alone authorizes G2,
  navigation, held-out/sealed evaluation, scale promotion, or deployment.
