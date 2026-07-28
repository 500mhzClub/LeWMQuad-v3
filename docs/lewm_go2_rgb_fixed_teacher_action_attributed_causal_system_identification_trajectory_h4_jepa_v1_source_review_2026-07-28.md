# Go2 RGB fixed-teacher action-attributed causal system-identification trajectory H4 JEPA V1 source review — 2026-07-28

## Review decision

- Status: `CLEAR_FOR_ONE_BOUND_PREFLIGHT_AND_ONE_ATTEMPT_IF_PREFLIGHT_PASSES`.
- Preregistration commit:
  `8f4e5c9673efa7ed1bc697aaad973e187555db93`.
- Model, runner, and proof-test commit:
  `8b68ac24d2a320c53f7070056f77c4887fd95a3c`.
- This review authorizes one non-training bound preflight and, only if that
  passes, one fresh 1,000-update / 16,000-presentation attempt. It authorizes
  no retry, resume, extension, predecessor-checkpoint read, navigation,
  held-out/sealed access, promotion, or deployment.

## Frozen primary bindings

| Source | SHA-256 | Bytes |
|---|---|---:|
| Preregistration | `c83151988fb6a97b1efedc9c9dfac2c0e38dabd62698c2583302139a166a9972` | 18,220 |
| Action-attributed system-ID model | `8edff571ca262fcf3b1e505017beb3f73eee027fc2ae195caaa127bbee6b6f02` | 21,854 |
| Bound runner | `5ab759e0353366b6c5172a0a854ff150a10321b80dee3d645d56f5f286401759` | 21,651 |
| Model proof tests | `9ac0a28a8cfda5bd97ae1d7bf59c87a95902d4d82a0690efed1128ab41440262` | 23,167 |
| Runner proof tests | `749ba7926957f22235f26e020ccad91abe1385c56dbb00717f76095898968a5b` | 15,513 |

- The runner's external self-binding is exactly:
  - `LEWM_ACTION_ATTRIBUTED_CAUSAL_SYSTEM_IDENTIFICATION_TRAJECTORY_H4_V1_WRAPPER_SHA256=5ab759e0353366b6c5172a0a854ff150a10321b80dee3d645d56f5f286401759`
  - `LEWM_ACTION_ATTRIBUTED_CAUSAL_SYSTEM_IDENTIFICATION_TRAJECTORY_H4_V1_WRAPPER_BYTES=21651`
- Source-only closure verification resolved all 18 frozen sources, including
  the exact latent-momentum and factorized predecessor wrappers/models and
  every inherited V2 schedule, factual evaluator, trajectory model, shared
  runner, and encoder dependency.

## Scientific review

- The predictive state is exactly four equal-mass `(q,M)` atoms. `q` is the
  normalized feature lattice. `M` is an exactly locked `16 x 16` nonspatial
  response matrix with no patch axis and no nine-action slots. It starts at
  zero and has rank at most two after the only two permitted rank-one writes.
- One shared prior is called on all six edges. It combines the exact inherited
  action-free spatial context, `1+tanh(P_M(vec(M)))`, and the current complete-
  table-centered categorical action code before one shared bias-free,
  zero-initialized increment head. `M` has no generic state-only route.
- Each of the two observed priors is emitted before its destination is seen.
  The new prior error is pooled by non-affine token normalization, a fixed
  token mean, and bias-free `P_r`; its outer product with the centered key of
  the requested action that caused that error is added to `M`. The factual
  destination then replaces `q`.
- The exact causal order is initializer, prior/write/assimilate for `p0`,
  prior/write/assimilate for `p1`, then four future priors for `p2:p5`. After
  `e2`, the belief contains only packed `(q2,M2)`, with exact-zero carrier
  padding. `M2` remains bitwise fixed throughout open-loop rollout.
- The complete-table mean pre-renormalization increment is exactly zero.
  `M=0` leaves the ordinary centered-action path intact, and the zero output
  head makes update zero exact persistence. HOLD uses the same ordinary route.
- Observed local predictions are scored as `q^-_(t+1)-z_t`; future local
  predictions are recursively realized `q^-_(t+1)-q_t`. The cumulative future
  trajectory is the same recursively produced `q` sequence.
- The exact three-term jointly trained JEPA objective remains weight-one
  online/fixed-teacher history alignment, half all-six local proper energy
  score, and half cumulative H4 proper energy score. One summed backward and
  one optimizer step jointly train the online encoder and predictor/system-ID
  routes; the accepted N320 target remains fixed, no-grad, and at zero EMA.
- There is no separately trained predictor or system identifier, learned
  writer gain/decay/gate, recurrent updater, momentum, raw incoming-increment
  bypass, dense history, future target path, inverse/action-ranking loss,
  reconstruction, navigation loss, correspondence, geometry, or controller.

## Test and inventory evidence

- Reviewed Torch runtime:
  `/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python`.
- Focused plus inherited compatibility suite: 144 passed, zero failed and zero
  warnings in 5.92 seconds. It covers the new model/runner and the exact
  latent-momentum, factorized, factual V1, V2 schedule-integrity,
  local-innovation, and trajectory-distribution ancestors.
- Full-size source-only parameter inventory:
  - encoder: 2,747,520 scalars / 78 tensors;
  - history/system-ID: 105,216 scalars / 5 tensors;
  - predictor: 594,624 scalars / 20 tensors;
  - total trainable: 3,447,360 scalars / 103 tensors;
  - fixed target: 2,747,520 non-trainable scalars / 78 tensors.
- Parameter groups are disjoint, cover every trainable tensor, and exclude the
  fixed target. Proofs cover exact-16 locking, q/M packing, rank/write
  arithmetic, mode/action/key centering, update-zero persistence, exact
  initializer/prior/write/assimilate ordering, meaningful fixed-error/key
  swapping, modulation-only memory use, uniform-action mean zero, q/M-only
  history dependence, future-memory freeze, action/HOLD sensitivity, loss
  arithmetic, target isolation, gradient staging and opened gradients through
  every learned route, one joint optimizer step, and K4 permutation invariance.

## Runner, receipts, and custody review

- The exact causal V2 indexes, evaluator, selection rule, bootstrap procedure,
  thresholds, and all 32 gates are inherited without relaxation or addition.
  Seed `20260727`, batch `16`, observations `0/250/500/750/1000`, 1,000
  updates, 16,000 train presentations, 10,240 validation presentations,
  183,680 expected RGB opens, 1,000 bootstrap replicates, and 5,400 active-GPU
  seconds are frozen.
- Runner tests reject every train/validation/model binding substitution and
  every retry, resume, seed, update, presentation, batch, GPU-cap, and
  arbitrary-checkpoint override surface.
- The adapter verifies the inherited objective and mechanism receipts, removes
  every latent-momentum claim, and adds only the truthful `(q,M)` system-ID
  claims. Gate contents remain exact; only terminal PASS/STOP identity changes.
- The exact inherited complete terminal handler remains installed. It derives
  forbidden counters from registered access state and preserves cross-bound
  normal or caught-failure JSON receipt chains.
- Exclusive root:
  `.generated/go2_rgb_fixed_teacher_action_attributed_causal_system_identification_trajectory_h4_jepa_v1/probe_v1`.
  Preflight is zero-RGB and zero-reservation. Runtime checkpoint/trace files
  are write-only; terminal review is restricted to the six exact canonical
  JSON receipts and may never list, stat, hash, or open a runtime checkpoint.

## Independent review history

- The first independent model/science pass found three concrete pre-freeze
  proof/contract gaps: the matrix width was configurable, a key-swap proof
  relied on floating reduction noise, and the event proof instrumented the
  observer rather than the writer. The source now rejects every width except
  16, uses independently distinct errors with substantive swap bounds, and
  directly proves exactly two writes and zero future writes.
- Final independent model/science re-review: CLEAR on the corrected frozen
  model and proof tests.
- Independent runner/custody review and post-correction binding re-review:
  CLEAR. It independently resolved all 18 sources and passed all 21 focused
  runner tests.
- No review accessed runtime outputs, checkpoints, RGB, indexes, metadata,
  navigation, held-out, or sealed material.
