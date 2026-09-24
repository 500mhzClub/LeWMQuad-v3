# Go2 RGB fixed-teacher causal posterior-reweighted transition-expert trajectory H4 JEPA V1 source review — 2026-07-28

## Review decision

- Status: `CLEAR_FOR_ONE_BOUND_PREFLIGHT_AND_ONE_ATTEMPT_IF_PREFLIGHT_PASSES`.
- Preregistration commit:
  `b5971eba1f201efd537327b8f7c46f7023eaae9f`.
- Model, evaluator, runner, and proof-test commit:
  `e434c4c5dd8e751f9c027de3aece6a1d55c31020`.
- This review authorizes one zero-training bound preflight and, only if it
  passes, one fresh 1,000-update / 16,000-presentation attempt. It authorizes
  no retry, resume, repair, replacement attempt, predecessor-checkpoint read,
  navigation, G2, held-out/sealed access, promotion, or deployment.

## Frozen primary bindings

| Source | SHA-256 | Bytes |
|---|---|---:|
| Preregistration | `f7b6981573e41f4c27f944c6abfab1e267502ad22bb7327ddd64b5d21b4a390a` | 17,457 |
| Posterior-reweighted expert model | `cbbc7c7f27021dc77a38405136de473552809fd5141fe60ae773e2fb4772bb99` | 31,330 |
| Bound runner/evaluator | `49d9829124ad72ddf929711b01d548bd071bd707f1d7ab3d2c0d7a2ef23c8a44` | 44,121 |
| Model proof tests | `1c0401bafcd782082f39e6e937228406f9d53d1b3862898882ea269f7f5dfccb` | 28,020 |
| Runner/evaluator proof tests | `44afafc9be96336ff4e84883cc6062cd7bd96c6826b46d0d7fa4845995cadca5` | 34,406 |

- The runner's external self-binding is exactly:
  - `LEWM_CAUSAL_POSTERIOR_REWEIGHTED_TRANSITION_EXPERT_TRAJECTORY_H4_V1_WRAPPER_SHA256=49d9829124ad72ddf929711b01d548bd071bd707f1d7ab3d2c0d7a2ef23c8a44`
  - `LEWM_CAUSAL_POSTERIOR_REWEIGHTED_TRANSITION_EXPERT_TRAJECTORY_H4_V1_WRAPPER_BYTES=44121`
- Source-only closure verification resolved all 20 frozen sources, including
  the exact action-attributed system-ID, latent-momentum, factorized, V2
  schedule-integrity, factual-transition, trajectory, shared-runner, and
  encoder dependencies.

## Scientific review

- The complete causal state is exactly four normalized `q` feature lattices
  and four strictly positive probabilities `w` summing to one. `q0` is four
  copies of online `z0`; `w0` is exactly uniform. The serialized carrier holds
  only the first four probability scalars and exact-zero padding.
- Four centered learned mode embeddings and one learned spatial table enter
  the inherited action-free belief context. One complete-table-centered
  categorical action tower and one shared bias-free, zero-initialized head are
  used on all six transitions. There is no per-action operator bank or HOLD
  special case.
- Each observed edge follows the registered causal order: emit the prior,
  compare the full prior lattice with the newly available online destination,
  apply `exp(-d_k/(mean(d)+1e-6))`, normalize prior mass times likelihood, and
  then assimilate the factual `q`. Exactly two evidence updates occur, on
  `p0/z1` and `p1/z2`.
- The four future priors use `p2:p5`; `w2` is returned bitwise unchanged. The
  transition step receives only expert content and the current action, so
  posterior mass cannot move an expert location or increment.
- Future marginal and coherent joint energy scores use exact posterior fit
  mass and ordered-pair mass. The combined score remains half joint plus half
  mean marginal. Weighted centroid and spread use the same posterior. At
  uniform mass every formula reduces to the inherited K4 score.
- The one jointly optimized JEPA objective is weight-one online/fixed-teacher
  history alignment, half equal-mass all-six realized-local energy score, and
  half `w2`-weighted cumulative future energy score. One summed backward and
  one AdamW step train the online encoder, modes/context, action path, and
  shared head. The accepted N320 target remains fixed and no-grad.
- `final_hidden_particles` is retained only as an output-schema compatibility
  alias of `posterior_probabilities`; it is never consumed as an additional
  state. Receipts explicitly declare that there is no other hidden state,
  separately trained predictor, inference model, optimizer, or checkpoint.

## Test and inventory evidence

- Reviewed Torch runtime:
  `/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python`.
- Focused model proofs: 8 passed. Focused runner/evaluator proofs: 29 passed.
  Combined focused suite: 37 passed.
- Focused plus all 13 inherited trajectory-line compatibility files: 183
  passed, zero failed, in 5.99 seconds. The suite includes trajectory
  distribution, local innovation, factual shared transition, exact V2
  schedule integrity, factorized increments, latent momentum, and
  action-attributed system-ID ancestors.
- Full-size source-only parameter inventory:
  - encoder: 2,747,520 scalars / 78 tensors;
  - modes/spatial posterior context: 49,920 scalars / 2 tensors;
  - predictor: 594,624 scalars / 20 tensors;
  - total trainable: 3,392,064 scalars / 100 tensors;
  - fixed target: 2,747,520 non-trainable scalars / 78 tensors.
- Parameter groups are disjoint, cover every trainable tensor, and exclude the
  fixed target. Proofs cover exact state packing, strict posterior simplex,
  zero padding, centered modes/actions, update-zero persistence, six-prior and
  two-evidence event order, exact likelihood arithmetic, future probability
  freeze, probability/content separation, action and actual-HOLD routing,
  history causality, future-target isolation, weighted-score identities,
  loss arithmetic, one joint step, staged/opened gradients through every
  learned route, absent target gradients, and joint expert/mass permutation
  invariance.

## Runner, receipts, and custody review

- The train/validation indexes, manifest, seed `20260727`, float32, batch 16,
  observations `0/250/500/750/1000`, 1,000 updates, 16,000 train
  presentations, 10,240 validation presentations, 183,680 expected RGB
  opens, 1,000 bootstrap replicates, and 5,400 active-GPU seconds remain the
  exact causal V2 bindings.
- Selection remains minimum posterior-weighted combined normalized energy
  among eligible noncollapsed trained observations. The exact inherited 32
  gate names, thresholds, family rules, bootstrap seeds, and decision logic
  are unchanged; only terminal experiment identity is relabelled.
- The full evaluator route proves factual `w2` for real, wrong-action,
  all-HOLD, persistence, and future-local scoring; reverse and reset branches
  recompute their own probabilities; the `p0/p1` diagnostic remains
  equal-mass. Best-atom error remains support-location minimum.
- Metrics rename both cumulative-loss buckets and their semantics to the
  truthful posterior-weighted field. Mechanism receipts remove inherited
  factorized claims and state only the registered q-plus-probability mechanism.
- Runner tests reject every bound input substitution and every retry, resume,
  seed, cap, update, presentation, batch, likelihood, epsilon, expert-count,
  or arbitrary-checkpoint override surface.
- Direct synthetic `core.main` proofs cover both the canonical normal six-JSON
  chain and the caught-failure chain without filesystem, dataset, RGB, Torch
  runtime, or checkpoint access. The complete inherited terminal handler is
  retained.
- Exclusive output root:
  `.generated/go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_expert_trajectory_h4_jepa_v1/probe_v1`.
  Preflight is zero-RGB, zero-training, and zero-reservation. Runtime
  checkpoints/traces are write-only; terminal audit is restricted to the six
  exact canonical JSON receipts and may never discover, list, stat, hash, or
  open a runtime checkpoint.

## Independent review history

- Independent model/science review: CLEAR at model SHA-256
  `cbbc7c7f27021dc77a38405136de473552809fd5141fe60ae773e2fb4772bb99`.
  It found one proof-only HOLD-index error before freeze; both test uses now
  derive the actual frozen HOLD index `6`, and the corrected model suite passes.
- Independent runner/evaluator re-review: CLEAR at runner SHA-256
  `49d9829124ad72ddf929711b01d548bd071bd707f1d7ab3d2c0d7a2ef23c8a44`
  and runner-test SHA-256
  `44afafc9be96336ff4e84883cc6062cd7bd96c6826b46d0d7fa4845995cadca5`.
  It independently reran all 37 focused tests after receipt, evaluator-route,
  terminal-chain, model-binding, and compatibility-alias corrections.
- A separate final source/custody audit was CLEAR against implementation
  commit `e434c4c5dd8e751f9c027de3aece6a1d55c31020`. It independently resolved
  all 20 source bindings, verified the frozen V2 schedule and parser closure,
  traced evaluator branch weights, confirmed the inherited 32-gate decision
  and canonical normal/failure receipt chains, and reran all 37 focused tests.
- No review accessed runtime outputs, checkpoints, RGB, indexes, metadata,
  navigation, G2, held-out, or sealed material.
