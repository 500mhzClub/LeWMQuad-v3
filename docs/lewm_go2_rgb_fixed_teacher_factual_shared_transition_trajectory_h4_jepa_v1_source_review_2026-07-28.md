# Go2 RGB fixed-teacher factual shared-transition trajectory-H4 JEPA V1 source review — 2026-07-28

## Status

- Review decision: **CLEAR FOR ONE EXACT CAPPED PROBE**.
- Preregistration commit:
  `5c038f054f17d7d8928518723b12e1166db2d17a`.
- Frozen implementation commit:
  `065bae4069d53a4d2c87f781df5ae9e29d5027a2`.
- This review grants no checkpoint promotion, scale-up, navigation, held-out,
  sealed, production, or deployment authority.

## Frozen source bindings

| Role | Path | SHA-256 | Bytes |
|---|---|---|---:|
| Preregistration | `docs/lewm_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1_preregistration_2026-07-28.md` | `6a4fbe6cd8832e5dd3f961d0a6ffb8c7485042e4456e8eeeb49a6adda3564f89` | 11,985 |
| Model | `lewm/models/go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py` | `38e264f8e18ffa3c3da4775fdd7d4a38549e8544f99cd863bfd2534999cd5b36` | 21,734 |
| Model tests | `lewm/tests/test_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py` | `c1c68e80a731a05cb0cc5f455863c0c272b5d1a46409179d61ad871fff6a0703` | 12,382 |
| Runner | `scripts/run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py` | `693cbea45b2a49f0f3edfb7cabce347b852a67af78df1ecf5462c65be48cd977` | 34,730 |
| Runner tests | `lewm/tests/test_run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py` | `9154052cebd7ea67ed5a3ce6ca8aba7704eb2ee25697b8c479ceee7e55e80329` | 15,538 |

The runner also binds the unchanged trajectory, local-innovation, dense-H4,
base recurrent-H4, encoder, and shared-runner dependencies by their existing
reviewed SHA-256 and byte counts. Real source-closure verification passed all
nine paths.

## Independent model review

- One `_SharedSpatialTransition` instance and one zero-initialized residual
  head serve all six `p0:p5` edges. No GRU/RNN/LSTM, horizon embedding,
  action-prefix decoder, step-specific transition, or separate history/future
  cell exists.
- `p0/p1` priors are formed before inserting `e1/e2`; the factual online
  carrier is inserted only after its prior is scored while the particle hidden
  state is retained. `p2:p5` are carrier-recursive and fully open-loop.
- Future RGB is target-only. It cannot affect priors, belief, or trajectory
  atoms, and fixed-teacher targets are detached and optimizer-excluded.
- The objective is exactly `0.5 * all-six local ES + 0.5 * future cumulative
  ES + 1.0 * three-frame alignment`. Inherited prediction, variance, and
  action-ranking terms have weight zero. No synthetic control is trained.
- The zero head produces exact persistence. On the first backward its own
  gradient is finite and nonzero while upstream transition paths remain zero;
  a controlled nonzero head opens finite upstream gradients.
- K is configuration-locked to four equal-mass coherent particles. Mode
  permutation only permutes atoms and leaves the proper score unchanged.
- Every and only trainable online parameter appears once in the inherited
  encoder/history/predictor optimizer groups; the fixed target appears in none.

## Independent runner review

- The evaluator makes one seven-frame load per validation row and computes
  future and factual-prior metrics in that same pass. It does not add a second
  data traversal or a wrong-past predictor branch.
- `p0/p1` proper score normalization uses exact zero innovations, a `1e-6`
  denominator clamp, and no row filtering. Scene means are macro-averaged
  through families and receive a deterministic scene-bootstrap lower bound.
- The decision has exactly 32 gates: the predecessor-equivalent 28 future,
  distribution, action, history, hold, fixed-target, and noncollapse gates plus
  four preregistered factual-prior gates. Selection remains the minimum future
  cumulative combined score among trained noncollapsed observations; update
  zero is never selectable.
- The loss receipt admits exactly the three objective terms, requires both
  inherited disabled terms to be zero, and recomputes total arithmetic.
- Training-control counters must remain zero. The fixed-target identity, EMA
  count, source closure, access accounting, cap, seed, schedule, and failure
  receipts remain inherited and controlling.

## Verification

- Focused new-source suite: 17 passed.
- Broader recurrent/dense/trajectory/local/dual/factual suite: 85 passed in
  5.80 seconds.
- `py_compile` and `git diff --check` passed.
- Bound preflight decision:
  `PREFLIGHT_PASS_NO_OUTPUT_RESERVED_NO_RGB_OPENED`.
- Preflight loaded 16,000 train and 2,048 validation index rows, copied only 78
  accepted N320 `encoder.*` tensors, opened no RGB, and reserved no output.
- Exact trainable parameter counts: encoder 2,747,520; history 87,360;
  predictor 558,528.
- Synthetic production-batch-16 ROCm forward, target, objective, and backward
  smoke passed in 0.634 seconds. Peak allocated memory was 4,815,493,120 bytes,
  peak reserved memory 5,597,298,688 bytes, head gradient norm 6.577, and target
  gradient count zero.

## Custody

- Reviewers opened source, tests, public metadata, and the accepted preflight
  inputs only. No stopped/rejected checkpoint, generated predictor tensor,
  held-out, sealed, test-role, label, navigation, or benchmark content was
  listed, statted, hashed, opened, or reused.
- The output root remained absent through review and preflight. Only the exact
  execution command in the separate authorization may reserve it.
