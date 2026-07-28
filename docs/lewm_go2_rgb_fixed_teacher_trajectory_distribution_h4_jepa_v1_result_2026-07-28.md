# Go2 RGB fixed-teacher trajectory-distribution H4 JEPA V1 result — 2026-07-28

## Terminal status

- Decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_TRAJECTORY_DISTRIBUTION_H4_JEPA_V1`.
- The attempt completed normally at the exact cap. This was a scientific STOP,
  not an execution failure: 1,000 optimizer updates, 16,000 ordered training
  presentations, and 10,240 validation presentations completed in
  `1173.951383` active GPU seconds.
- Update 750 / presentation 12,000 was selected by the preregistered minimum
  combined joint-plus-marginal normalized energy score.
- The probe passed 21 of 26 gates. Its five failures were:
  - H4 action gap at least 0.05;
  - H4 ordered-history gap at least 0.03;
  - positive H4 ordered-history bootstrap lower bound;
  - ordered-history benefit in at least six families;
  - positive H4 all-hold gap.
- This exact K=4 finite-support/full-fixed-teacher-latent formulation is closed.
  There is no retry, resume, second seed, K change, longer run, or nearby V2.

## What worked

The distributional prediction itself worked strongly and consistently:

| Selected metric | Value |
|---|---:|
| Combined normalized energy score | 0.725748 |
| Joint-trajectory normalized energy score | 0.722143 |
| H1 marginal normalized energy score | 0.780452 |
| H2 marginal normalized energy score | 0.769585 |
| H3 marginal normalized energy score | 0.752509 |
| H4 marginal normalized energy score | 0.742969 |
| H4 persistence gap | +0.257031 |
| H4 persistence bootstrap lower 95% | +0.231784 |
| Combined distribution-value gap | +0.247704 |
| Combined distribution-value bootstrap lower 95% | +0.244390 |
| H4 normalized pairwise spread | 1.119353 |

- The real four-atom distribution beat fixed-teacher `e2` persistence at every
  horizon and in all eight maze families at H4. Family H4 persistence gaps
  ranged from `+0.177403` to `+0.302790`.
- The ensemble beat its own spherical centroid under the exact combined score
  in all eight families. Family combined distribution-value gaps ranged from
  `+0.219490` to `+0.262652`.
- The selected checkpoint had nontrivial support spread, healthy fixed-teacher
  and online encoder rank, and no registered collapse:
  - target effective-rank ratio `0.174981`, constant across all observations;
  - online effective-rank ratio `0.202470`;
  - target and online near-zero variance fractions `0.0`.
- This is the first bounded main-pool H4 probe in this sequence to beat the
  persistence baseline broadly. Its energy-score values are not numerically
  interchangeable with the predecessor deterministic squared-error values,
  but the within-probe persistence comparison is registered and direct.

## What still failed

The distribution was not sufficiently conditioned on action or ordered visual
history:

| Selected H1--H4 metric | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|
| Wrong-action gap | +0.004792 | +0.010064 | +0.011742 | +0.010815 |
| Ordered-history gap | -0.013536 | -0.017192 | -0.013830 | -0.013974 |
| All-hold gap | -0.002003 | -0.009162 | -0.007035 | -0.006572 |

- H4 wrong-action sensitivity was statistically positive: bootstrap lower
  `+0.006373`, positive in seven of eight families, and no family below the
  registered -0.02 floor. It was nevertheless only about one fifth of the
  required 0.05 effect. Small enclosed mazes were negative at `-0.005524`.
- Ordered history lost to at least one of the reset/reordered controls in all
  eight families. Its H4 bootstrap lower bound was `-0.021095`.
- The all-hold future was easier than the real future in aggregate. Only large
  enclosed and open-obstacle families had positive H4 hold gaps.
- Point forecasts remained worse than persistence even though the proper
  distribution score won: H4 best-atom normalized squared error was `1.480741`
  and spherical-centroid normalized squared error was `1.377818`.

## Learning trajectory

| Update | Combined score | H4 score | H4 persistence gap | H4 action gap | H4 history gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 250 | 0.784081 | 0.822756 | +0.177244 | +0.011309 | -0.007143 |
| 500 | 0.761371 | 0.800773 | +0.199227 | +0.011826 | -0.006989 |
| 750 | 0.725748 | 0.742969 | +0.257031 | +0.010815 | -0.013974 |
| 1000 | 0.742350 | 0.776965 | +0.223035 | +0.009891 | -0.010700 |

- Distributional prediction improved materially through update 750 and then
  regressed at update 1000, validating the registered checkpoint selection.
- Action sensitivity stayed small throughout. Ordered-history evidence never
  became positive, so more updates to this same objective are not justified.

## Scientific interpretation

- The finite-support hypothesis passed its registered within-probe value test:
  four coherent support atoms beat both persistence and their own collapsed
  centroid across every maze family. This shows that distributed support
  helped this proper-score objective; it does not establish deterministic
  prediction as the predecessor's causal blocker.
- The result does not yet establish useful learned dynamics. A plausible
  inference is that the model learned a broad distribution of visually
  plausible future views, largely from the current frame, rather than a
  sufficiently action- and history-conditioned predictive state. The weak
  wrong-action gap, negative history gap, negative hold gap, and poor point
  errors support that inference; they do not prove its internal mechanism.
- Encoder collapse is not the blocker registered here. Nor is a single
  deterministic future the whole blocker. The remaining bottleneck is making
  the learned state encode controllable, history-dependent aspects of future
  geometry instead of nuisance/view uncertainty.
- The next scientific category should therefore use a compact learned
  predictive target/state whose tests are explicitly future- and
  action-dependent, while retaining one joint JEPA backward and no navigation
  labels. It must not reuse any checkpoint from this STOP branch.

## Execution and custody audit

- Train schedule: SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`,
  16,000 rows / 1,000 scenes. Validation schedule: SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`,
  2,048 rows / 150 scenes.
- RGB access was exactly 183,680 successful opens from 183,680 attempts and
  6,900,398,764 physical bytes. Test/held-out, sealed, label, arbitrary
  checkpoint, retry/resume checkpoint, and retry/resume counts were all zero.
- The fixed teacher initial and final state SHA-256 were identical:
  `dd3c8f053808848f1caa63b5870b0948382c9c875b7d6848ab8a1cf05a8f3e4b`.
  It recorded zero EMA updates. The accepted N320 initialization was opened
  exactly once; no predecessor predictor checkpoint was opened.
- Fresh source/model/action/history/mode/head initialization was recorded.
  The six runtime source bindings match the frozen reviewed source.
- Terminal JSON receipt file bindings:
  - reservation: `1ebaafed579a05f64fa06a0b777ed872ab8256a8eb7e88396f6408c9d28d5669`,
    4,460 bytes;
  - metrics: `8ce4a5fe09408591e385940241a3c298ffda2e779c6ccf1ea6ed78a925e17566`,
    52,811 bytes;
  - artifact: `4d2dcf0d12ccddd27dcde66de9b82e21c1cc2cba203e43f50b14ea8171c3976b`,
    4,641 bytes;
  - access: `5d8849df3e1132877306063f2e3fac92fb7779cbf7b363e72353d5a359e11e41`,
    1,272 bytes;
  - result: `a597df8e45e9c398c1d3ebb19b3f8d4571563189c2fe092530495931994e3390`,
    2,301 bytes;
  - completed: `c8ab1b498a68265449e678a0d0dd4c09b582f1f2583bb3532291046a4a30da27`,
    1,848 bytes.
- All six canonical content hashes and all completion cross-bindings were
  independently recomputed and matched.
- Four checkpoint filenames were inventoried only. No `.pt` file was opened,
  hashed, loaded, copied, or reused. They remain inaccessible under STOP.

STOP grants no checkpoint, navigation, held-out, promotion, or deployment
authority. The existing sealed benchmark remains unopened.
