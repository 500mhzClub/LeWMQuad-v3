# Go2 RGB whitened-delta predictive-state H4 JEPA V1 result — 2026-07-28

## Terminal status

- Decision: `STOP_MAIN_POOL_RGB_WHITENED_DELTA_PREDICTIVE_STATE_H4_JEPA_V1`.
- The attempt completed normally at the exact cap. This was a scientific STOP,
  not an execution failure: 1,000 optimizer updates, 16,000 ordered training
  presentations, and 10,240 validation presentations completed in
  `629.820302` active GPU seconds.
- No trained observation was eligible for selection. Every trained observation
  failed the preregistered compact-state participation-rank and minimum-scale
  requirements, so the runner correctly left `selected_update=null` and made
  all selection-dependent gates false.
- The exact WDPS-D8 formulation is closed. There is no retry, resume, second
  seed, extension, or checkpoint access.

## What worked

The run found a real action-conditioned signal before the learned compact state
collapsed:

| Best diagnostic trained observation (ineligible update 750) | Value |
|---|---:|
| Mean H1--H4 normalized error | 1.508646 |
| H4 normalized error | 1.276373 |
| H4 wrong-action gap | +0.232192 |
| H4 wrong-action bootstrap lower 95% | +0.159182 |
| H4 all-hold gap | +0.919643 |
| H4 ordered-history gap | -0.102973 |
| H4 persistence gap | -0.276373 |

- At update 750, wrong-action gaps were positive at every horizon:
  `+0.169575,+0.214214,+0.185486,+0.232192`. H4 was positive in all
  eight maze families, ranging from `+0.004678` to `+0.866249`.
- The H4 all-hold gap was positive in all eight families at update 750. This
  means the predictor was not merely returning one action-independent value.
- Four of eight families beat H4 persistence at update 750: large enclosed,
  local composite motifs, small enclosed, and visual sensor stress. Visual
  sensor stress reached H4 error `0.853793`.
- The target-scale calibration did its job. H4 target RMS reached `0.478182`
  at update 750, every scene denominator stayed above the near-zero floor, and
  mean-energy fractions remained far below `0.25`. The target was not
  microscopic and the DC-offset loophole did not activate.
- Fixed and online N320 feature audits remained above their inherited floors;
  all registered values were finite at every observation.

## What failed

The eight-dimensional state became approximately one-dimensional:

| Update 750 state audit | Predicted state | Target state |
|---|---:|---:|
| H4 participation-rank ratio | 0.126746 | 0.173593 |
| Approximate participation dimension (`ratio * 8`) | 1.014 | 1.389 |
| H4 minimum dimension std | 0.323166 | 0.458565 |
| H4 maximum absolute dimension mean | 0.050771 | 0.019793 |
| H4 mean-energy fraction | 0.010129 | 0.000582 |

- Eligibility required participation-rank ratio at least `0.75` and every
  dimension std at least `0.50` at all four horizons for both branches. Every
  trained observation failed both state-rank checks and at least one scale
  check. Mean, RMS, finiteness, encoder noncollapse, and scene-denominator
  checks passed.
- The predicted state's eight coordinates had similar marginal standard
  deviations but participation ratio stayed near the mathematical
  one-direction floor `1/8=0.125`. The coordinates therefore became highly
  correlated copies, not eight independent predictive factors.
- The learned target also lost diversity. Its H4 participation ratio fell from
  `0.338820` at update zero to `0.226170,0.200736,0.173593,0.164774` at
  updates 250--1000. Joint training selected an increasingly easy shared
  direction despite the raw covariance penalty.
- Prediction never beat persistence in aggregate. The best trained H4 error
  was `1.276373`, and the best trained mean H1--H4 error was `1.508646`, both
  at update 750.
- Ordered visual history never helped. H4 history gaps were
  `-0.033738,-0.178582,-0.102973,-0.059571` at updates 250--1000. At update
  750 all eight family history gaps were negative.
- Action sensitivity peaked at update 750 and regressed by update 1000, while
  normalized prediction error remained worse than persistence. A longer run
  of this objective is not justified.

## Learning trajectory

| Update | Mean error | H4 error | H4 persistence | H4 action | H4 history | H4 hold | Predicted H4 rank | Target H4 rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.338820 |
| 250 | 1.824834 | 1.472649 | -0.472649 | +0.005772 | -0.033738 | -0.008946 | 0.126541 | 0.226170 |
| 500 | 2.810426 | 2.223398 | -1.223398 | +0.141571 | -0.178582 | +0.350394 | 0.126324 | 0.200736 |
| 750 | 1.508646 | 1.276373 | -0.276373 | +0.232192 | -0.102973 | +0.919643 | 0.126746 | 0.173593 |
| 1000 | 1.577629 | 1.287719 | -0.287719 | +0.086702 | -0.059571 | +0.244914 | 0.125711 | 0.164774 |

## Scientific interpretation

- This is not evidence that the main pool lacks controllable visual dynamics.
  The broad, statistically strong wrong-action effect at update 750 shows that
  the joint JEPA learned an action-dependent compact signal from the reviewed
  16,000-presentation schedule.
- It is evidence against this soft-whitened learned-target formulation. The
  per-coordinate variance floor kept each coordinate moving, but the
  scale-dependent covariance penalty was too weak to stop all eight
  coordinates from carrying nearly the same factor. Similarity then rewarded
  the target compressor and predictor for agreeing on that easy scalar.
- The remaining signal was not a useful predictive state: aggregate errors
  lost to zero-state persistence and correct temporal ordering was worse than
  reset/reordered controls. The action result alone is therefore insufficient
  for navigation promotion.
- An obvious nearby repair is justified once, because the failure mode is
  specific and the action metric improved strongly: replace soft raw
  covariance decorrelation with a scale-invariant or hard full-rank redundancy
  constraint that cannot be satisfied by eight correlated copies. That next
  hypothesis must remain one joint, perception-only JEPA, use fresh
  initialization, preserve the held-out boundary, and close if history and
  persistence do not improve. Merely increasing duration, changing seed, or
  relaxing eligibility is not justified.

## Execution and custody audit

- Frozen source commit: `bc889a3fb6f7cf29ea6b2b2c6b45ddd4b4d83a12`.
  Source review and one-shot authority commit: `7abc844`.
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
- The six runtime source bindings match the frozen reviewed source.
- Terminal JSON receipt file bindings:
  - reservation: `b5db8083ee2808fff24df2686e64ee2876fffe6f78bd71f4efbcef918ee4bd0a`,
    4,387 bytes;
  - metrics: `6709f949df694e0be409a94d8bb6d6e1526e419e1c84d8e807cdf4826c856b12`,
    32,213 bytes;
  - artifact: `eef9b926fdc0c3ab1d2843c6c37f98d9bcc73a4ac331807782ba1096fdc57013`,
    4,692 bytes;
  - access: `b50fdb5c41547287ad308efad800f614204389e597288ec20afa465c4e7365e7`,
    1,266 bytes;
  - result: `f16f39def6fd09ab14c8f6ff43adf7b7b528bb8d6f0236e05dd64c87050c6b8d`,
    2,014 bytes;
  - completed: `26b30fe5ce27e14ec94268a6ff77e1c93b38aeb4f29592b3db5cf97acef6efeb`,
    1,836 bytes.
- All six canonical content hashes and every completion cross-binding were
  independently recomputed and matched.
- Four registered checkpoint metadata entries were counted from the artifact
  receipt only. No generated `.pt` file was opened, hashed, loaded, copied,
  statted, listed, or reused. They remain inaccessible under STOP.

STOP grants no checkpoint, navigation, held-out, promotion, or deployment
authority. The V4 30-scene sealed benchmark remains unopened.
