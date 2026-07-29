# V10 Projective Cell-Volume Lift — Physical-Evidence Calibration Result

- Terminal status: `FAIL_DEVELOPMENT_PHYSICAL_EVIDENCE`.
- The sole authorized command completed once with the registered scientific-
  failure exit `2`. There was no crash, retry, resume, refit, threshold
  relaxation, or alternate calibration.
- Preregistration / source / execution-binding commits:
  `6bc4dca93daf0e220bbaa4fc524470addb880e21` /
  `861f8377539742ce28591e64b4bdae6f430cd939` /
  `9abcc1ac0dc475d502776c6126cbf1c7ef70a9e2`.

## Artifact and procedure integrity

- Result: 364,006 bytes; file SHA-256
  `795ceafae6d24c4f7766f4eccd7d9e1e81a088e896bb5d170d3a477d4b53e5f6`;
  canonical content SHA-256
  `0c352921d2587541debad2e1dd1448daf72828955f6aa1c869af87d212c46f77`.
- Calibration: 4,158 bytes; file SHA-256
  `3f17a58b0dc2549d209fc1391ebe2b7ffb99477b3599ba9847c5a035d95a7b16`;
  semantic content SHA-256
  `3920b6991423b5fa1e82c73ab3ad16234142a834fcc88e3312755633ea8a11f3`;
  ID `go2-hier-cal-3920b6991423b5fa`.
- The exact V10 result was validated before one checkpoint read/load. The
  checkpoint was reconstructed once on CPU with `weights_only=True`, frozen,
  and remained unmutated.
- Calibration used 415 ordered rows, 26 batches, and 1,699,840 cells;
  selection used 495 rows, 31 batches, and 2,027,520 cells. One calibrator fit
  and one search of all 2,016 tuples occurred.
- Predictor, backward, optimizer, EMA, training, navigation, G2, held-out,
  sealed, and train-role payload counts were zero. All 12 frozen dependency
  hashes matched.

## Physical result

- No threshold tuple passed on the calibration role. The reported tuple
  (FREE `0.50`, OCCUPIED maximum `0.35`, UNKNOWN maximum `0.35`, detection
  `0.50`) is the registered diagnostic fallback, not a qualified setting.

| Metric | Calibration | Selection | Required | Result |
|---|---:|---:|---:|---|
| Admitted FREE precision | `0.917733` | `0.926213` | `>=0.99` | FAIL |
| OCCUPIED detection within 2 m | `0.653867` | `0.617912` | `>=0.95` | FAIL |
| Useful FREE recall | `0.882843` | `0.835924` | `>=0.90` | FAIL |
| Obstacle exclusion within 2 m | `0.850266` | `0.809313` | `>=0.95` | FAIL |

- Selection shortfalls were `0.063787` FREE precision, `0.332088` obstacle
  detection, `0.064076` useful FREE recall, and `0.140687` obstacle
  exclusion. All five conjunctive checks failed, including existence of a
  feasible calibration tuple.
- Calibration itself was numerically effective but cannot create missing
  rank separation: on selection, joint accuracy improved
  `0.942790 -> 0.965231`, NLL `0.164311 -> 0.096376`, and multiclass Brier
  `0.084742 -> 0.051559`.

## What V10 learned relative to V9

- V10 materially improved selection OCCUPIED detection from V9's `0.322811`
  to `0.617912` (`+0.295101`) and obstacle exclusion from `0.744179` to
  `0.809313` (`+0.065134`). This supports the causal diagnosis that V9's
  ground-centre gate hid usable vertical obstacle evidence.
- It did not improve the FREE side: precision moved `0.928906 -> 0.926213`
  (`-0.002693`) and useful recall `0.842727 -> 0.835924` (`-0.006803`). The
  remaining failure is therefore not just missing projection support; FREE,
  OCCUPIED, and abstain/UNKNOWN evidence still overlap too heavily.
- V10 is closed, unqualified, non-resumable, and unauthorized for G2. Do not
  retry calibration, tune the support volume, change thresholds, extend the
  run, or reopen its checkpoint.
- Because obstacle metrics improved strongly, a successor is justified, but
  it must directly change learned physical-evidence separability while
  retaining joint JEPA training—not repeat V10 geometry or another monotone
  calibration.
