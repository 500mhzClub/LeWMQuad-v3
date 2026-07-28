# Go2 RGB full-whitened predictive-state H4 JEPA V1 result — 2026-07-28

## Terminal status

- Decision:
  `STOP_MAIN_POOL_RGB_FULL_WHITENED_PREDICTIVE_STATE_H4_JEPA_V1`.
- The attempt completed normally at the exact cap. This was a scientific STOP,
  not an execution failure: 1,000 optimizer updates, 16,000 ordered training
  presentations, and 10,240 validation presentations completed in
  `629.570871` active GPU seconds.
- No trained observation was eligible. The full-whitening losses substantially
  improved marginal state geometry, but no observation met the complete
  predicted/target rank, scale, within-covariance, and cross-covariance gate.
  The runner therefore correctly left the selected checkpoint null and made
  every selection-dependent PASS gate false.
- The preregistered full-whitened D8 learned-target category is closed. There
  is no retry, resume, second seed, extension, coefficient sweep, or checkpoint
  access.

## What the structural repair achieved

The repair prevented the predecessor's immediate one-factor duplication:

| Update 1,000 H4 state audit | Predicted state | Target state |
|---|---:|---:|
| Participation-rank ratio | 0.681220 | 0.782536 |
| Approximate participation dimension (`ratio * 8`) | 5.450 | 6.260 |
| Minimum dimension std | 0.598060 | 0.552957 |
| Within covariance-to-identity error | 0.384775 | 0.361188 |
| RMS | 0.696251 | 0.668802 |
| Maximum absolute mean | 0.214803 | 0.033004 |
| Mean-energy fraction | 0.032605 | 0.000765 |

- For comparison, WDPS-D8's best diagnostic predicted H4 participation ratio
  was `0.126746`, approximately one active dimension. Full whitening raised it
  to `0.681220`, while the target reached and exceeded its `0.75` rank floor.
- By update 1,000, all four predicted within-covariance errors were below the
  registered `0.50` ceiling. Target H2--H4 also passed that ceiling; target H1
  was close but still failed at `0.546578`.
- All four predicted minimum standard deviations exceeded `0.50`. Target H3
  and H4 exceeded `0.50`; H1 and H2 remained below it at `0.439377` and
  `0.491533`.
- State RMS, mean-energy, maximum-mean, scene-denominator, fixed/online encoder
  noncollapse, and finiteness guards remained healthy. This run did not fail
  because of zero scale, a DC offset, numerical instability, or encoder
  collapse.

## What failed

The two full-rank-looking marginal states did not become the same predictive
state on corresponding samples:

| Update 1,000 cross-state audit | H1 | H2 | H3 | H4 |
|---:|---:|---:|---:|---:|
| Cross covariance-to-identity error | 0.847257 | 0.796947 | 0.765224 | 0.744375 |
| Maximum cross-diagonal error | 0.967084 | 0.943812 | 0.930899 | 0.919818 |
| Maximum off-diagonal cross covariance | 0.080478 | 0.087780 | 0.109461 | 0.107781 |

- Eligibility required every predicted-target cross-covariance identity error
  to be at most `0.50`. Every horizon failed. The small off-diagonal values
  alongside very large diagonal errors show that the branches were mostly
  decorrelated, not matched coordinate-by-coordinate.
- Predicted participation rank still failed the `0.75` floor at every horizon
  (`0.677947`--`0.683462`). Target rank passed at update 1,000, so the learned
  target's predecessor collapse was repaired more completely than the
  predictor's.
- Prediction was far worse than exact-zero persistence. At update 1,000, mean
  H1--H4 normalized error was `3.193653` and H4 error was `2.474828`. No trained
  observation beat persistence at any horizon in aggregate.
- Correct actions did not explain the target correspondence. Update-1,000 H4
  wrong-action gap was `-0.017207`, positive in only one of eight families;
  its bootstrap lower bound was `-0.027174`.
- Ordered visual history never helped. H4 history gap remained negative at
  every trained observation and was negative in all eight families at update
  1,000. Its aggregate value was `-0.004010`, with bootstrap lower bound
  `-0.004557`.
- The all-hold diagnostic became positive in all eight families at update
  1,000, but this isolated result is not useful when real prediction loses
  badly to persistence and both action and ordered-history tests fail.

## Learning trajectory

| Update | Mean error | H4 error | H4 persistence | H4 action | H4 history | H4 hold | Predicted H4 rank | Target H4 rank | H4 cross error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | -0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.338820 | 1.000000 |
| 250 | 2.292831 | 1.785897 | -0.785897 | +0.008264 | -0.021797 | -0.000024 | 0.251685 | 0.539600 | 0.870987 |
| 500 | 3.363705 | 2.594811 | -1.594811 | -0.041144 | -0.016855 | -0.060511 | 0.482805 | 0.654317 | 0.810406 |
| 750 | 3.217712 | 2.482040 | -1.482040 | +0.001937 | -0.005037 | +0.008141 | 0.487852 | 0.721293 | 0.778222 |
| 1,000 | 3.193653 | 2.474828 | -1.474828 | -0.017207 | -0.004010 | +0.015546 | 0.681220 | 0.782536 | 0.744375 |

- Geometry and cross-covariance metrics were still moving in the intended
  direction at the cap. That is evidence that the optimizer learned the
  whitening objective; it is not evidence of useful successor prediction.
- From update 750 to 1,000, H4 cross error improved by only `0.033847`, while
  H4 prediction stayed about 2.5 times persistence, action evidence became
  negative, and history evidence remained negative. Extending this exact
  category would violate its preregistration and is not supported by the
  task-relevant metrics.

## Scientific interpretation

- Full covariance constraints addressed the diagnosed coordinate-redundancy
  defect. They did not address the more important correspondence defect: the
  predictor and learned target could spread variance across dimensions without
  agreeing strongly on which sample should occupy which state.
- Removing pointwise prediction error avoided the old target-shrinkage route,
  but left covariance matching as the only direct predicted-target learning
  signal. In this run that signal improved marginal/cross geometry without
  producing a state that beat persistence or depended correctly on action and
  history.
- This is not evidence that RGB lacks controllable dynamics. Earlier branches
  found broad action sensitivity (WDPS-D8) and broad persistence improvement
  under a distributional score. Together with this run, the evidence points to
  a target/learning-signal mismatch: we have separately obtained action signal,
  uncertainty value, and multi-dimensional geometry, but not yet combined
  them into one aligned, history-conditioned predictive state.
- The next category should make the target a stable temporal innovation and
  train explicit per-sample correspondence while keeping a noncollapse
  mechanism. It should not be another whitening-weight tweak or longer run of
  this objective. It must remain one joint RGB/action-only JEPA with no
  navigation labels and fresh initialization.

## Execution and custody audit

- Frozen source commit:
  `756d4d1b4769421859bbee6e21dfe65d56673156`. Independent source review and
  one-shot authority commit: `5395ff7`.
- Train schedule SHA-256:
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`,
  16,000 rows / 1,000 scenes. Validation schedule SHA-256:
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`,
  2,048 rows / 150 scenes.
- RGB access was exactly 183,680 successful opens from 183,680 attempts and
  6,900,398,764 physical bytes. Test/held-out, sealed, label, arbitrary
  checkpoint, retry/resume checkpoint, and retry/resume counts were all zero.
- The fixed target-and-history teacher initial and final state SHA-256 were
  identical:
  `dd3c8f053808848f1caa63b5870b0948382c9c875b7d6848ab8a1cf05a8f3e4b`.
  It recorded zero EMA updates. The accepted N320 initialization was opened
  exactly once; no predecessor predictor checkpoint was opened.
- All eight runtime source bindings match the frozen reviewed source.
- Terminal JSON receipt file bindings:
  - reservation: `08509dd65c4115cf6a458bc99d3e69899d40ea1acfe2826e313e936d2a80d89d`,
    4,999 bytes;
  - metrics: `4cbd3c541a5997b0cd8bbccbedc647ab09e5978fb50a8103ea211ff60f5639bc`,
    41,565 bytes;
  - artifact: `629aebad0782cc00f63379593a73f85c8058a707fdc4ad1d7929e3db494855f5`,
    5,116 bytes;
  - access: `2066a9cf265ec9a5da75e03d1d9ce8e93175bc1a67b342d173b728854dce41d0`,
    1,265 bytes;
  - result: `d42f26257f59ebb24ce97a18778b5b7396139b3e7e7e325b46afa5f162b26a8f`,
    2,012 bytes;
  - completed: `65600b66662f6369e6afef8797fab604e45b6af6ead0f832a9a9d7f389d37aff`,
    1,834 bytes.
- All six canonical content hashes, canonical file encodings, byte counts,
  completion file bindings, and completion cross-bindings were independently
  recomputed and matched.
- The result receipt's inherited free-text `authority` leaf incorrectly says
  that STOP closes the old WDPS-D8 formulation. This is a wording defect only:
  the receipt schema, terminal decision, output root, source closure, and
  controlling preregistration all unambiguously identify this full-whitened
  attempt. The immutable terminal receipts were not rewritten post-run.
- Two other schema limitations are recorded for future runners. The
  reservation's `joint_online_components` list omits the jointly trained target
  compressor even though the science block and artifact record it, and the cap
  field `rgb_frame_views=112000` counts training views only; total train plus
  validation opens were `183680`.
- With no eligible selection, the result mechanically records every
  selection-dependent science gate as false. Raw update 250 nevertheless had
  nonnegative H1--H3 action gaps, a positive H4 action bootstrap lower bound,
  and positive H4 action gaps in six families. Those diagnostics cannot yield
  PASS because the observation was ineligible, but future receipt schemas
  should distinguish `not_evaluated_without_selection` from observed false.
- Four registered checkpoint metadata entries were counted from the artifact
  JSON only. No generated `.pt` file was opened, hashed, loaded, copied,
  statted, listed, or reused. They remain inaccessible under STOP.

STOP grants no checkpoint, navigation, held-out, promotion, or deployment
authority. The existing sealed benchmark remains unopened.
