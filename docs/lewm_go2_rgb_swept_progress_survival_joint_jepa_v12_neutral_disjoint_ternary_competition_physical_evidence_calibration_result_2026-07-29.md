# V12 Neutral Disjoint Ternary Competition Physical Calibration — Result

- Date: 2026-07-29.
- Outcome: `FAIL_DEVELOPMENT_PHYSICAL_EVIDENCE`; no calibration tuple was
  feasible and all four fixed selection metrics missed their thresholds.
- Authority: physical preregistration
  `c63e98162a1b03a33225e6e0a04b67a357c7ed89`, frozen source
  `3a7f1ab0d002b159b9e57fa143faba324b81f278`, and execution binding
  `bea1b17afb1b8ac9426eba1079ef3df7f1a258bc`.
- The exact bound command completed once with exit `2`, a registered
  scientific failure. There was no crash, retry, refit, alternate tuple grid,
  threshold relaxation, or intervention.

## Artifact and custody integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `result.json` | 364,010 | `407d22437143affd403a1669a0320832b041e6b4dc3b94c04a676a054a8f7992` |
| `calibration.json` | 4,161 | `f39f25b8a0ee4eba8da20adbc7f2011f9ac366ca082f723dc460a5dfedaed258` |

- Result canonical content SHA-256 is
  `8607c5263c158376067629149f480417c60c1572e10902d8ae9cd93ee2f2a211`;
  calibration canonical content SHA-256 is
  `5ce224509ea51f52b020a358b023b06d572bf3010722bc25bf9ce2424e885c0e`.
- The candidate result was validated before one checkpoint read and one CPU
  `weights_only=True` load. There was exactly one calibrator fit and one
  threshold-selection call.
- Calibration used 415 rows / 1,699,840 cells; selection used 495 rows /
  2,027,520 cells. All 2,016 frozen threshold tuples were evaluated once.
- Predictor, backward, optimizer, EMA, training-role, accelerator, G2,
  navigation, held-out, and sealed operation counts were all zero.

## Fixed physical gate

| Selection metric | Observed | Required | Shortfall |
|---|---:|---:|---:|
| Admitted FREE precision | `0.961184` | `>=0.99` | `0.028816` |
| Obstacle detection within 2 m | `0.573632` | `>=0.95` | `0.376368` |
| Useful FREE recall | `0.769442` | `>=0.90` | `0.130558` |
| Obstacle exclusion within 2 m | `0.849022` | `>=0.95` | `0.100978` |

- Passing calibration tuples: `0/2,016`. The deterministic fallback was the
  loosest registered tuple: FREE minimum `0.50`, OCCUPIED maximum `0.35`,
  UNKNOWN maximum `0.35`, and OCCUPIED detection minimum `0.50`.
- Even on the calibration role, that tuple reached only FREE precision
  `0.964681`, near-obstacle detection `0.643579`, useful FREE recall
  `0.858954`, and near-obstacle exclusion `0.893257`. This was not a selection-
  only unlucky split and no hidden feasible operating point existed in the
  frozen grid.
- Of 250,024 true-FREE selection cells, 192,379 were admitted and 57,645 were
  rejected. The 200,148 total admitted cells also contained 5,593 UNKNOWN and
  2,176 OCCUPIED false admits. Simultaneously reaching 90% FREE recall and 99%
  precision would require at least 32,643 additional true-FREE admits and
  5,497 fewer false admits.
- Of 7,001 nearby OCCUPIED cells, 4,016 were directly detected, 1,057 were
  dangerously admitted as FREE, and 1,928 were excluded but not detected.
  Passing requires 2,635 additional detections and 707 fewer near-obstacle
  FREE admissions. This is a substantive score-separation problem, not a
  finer-threshold problem.
- The probability fit itself behaved normally. Calibration-role joint NLL
  improved `0.179756 -> 0.098358`, joint Brier `0.094665 -> 0.052393`, and
  confidence ECE reached `0.007038`. On selection, calibrated joint accuracy
  was `0.959725` and joint NLL `0.107543`. The failure is therefore confidence
  separation at the conservative physical operating point, not optimizer
  failure or global class collapse.

## Comparison with V10 physical evidence

| Selection metric | V10 | V12 | Delta |
|---|---:|---:|---:|
| Admitted FREE precision | `0.926213` | `0.961184` | `+0.034970` |
| Obstacle detection within 2 m | `0.617912` | `0.573632` | `-0.044279` |
| Useful FREE recall | `0.835924` | `0.769442` | `-0.066482` |
| Obstacle exclusion within 2 m | `0.809313` | `0.849022` | `+0.039709` |
| FREE-probability Brier | `0.092096` | `0.113245` | `+0.021149` |
| FREE-probability ECE | `0.185649` | `0.220320` | `+0.034671` |

- V12 is more conservative: it improves FREE precision and obstacle
  exclusion while admitting less useful FREE and detecting fewer nearby
  obstacles. This is a mixed trade, not monotonic progress toward all four
  physical requirements.
- V12 also generalizes less cleanly from calibration to selection than V10.
  Its near-obstacle detection falls `0.643579 -> 0.573632` and useful FREE
  recall falls `0.858954 -> 0.769442`; its selection probability Brier/ECE are
  worse despite a well-behaved fitted calibrator.

## Adjudication

- V12's development pass remains valid: neutral competition repaired V11's
  FREE overcall while preserving its development obstacle recall. The
  physical result shows that those class-recall gains do not provide the
  high-confidence spatial separation needed for safe navigation.
- The exact V12 checkpoint, seed, schedule, neutral algebra, and physical
  calibration attempt are closed. Do not retry, relax thresholds, search a
  second grid, refit, reopen the checkpoint, or promote the nearest tuple.
- V12 is not G2-qualified. G2, learned navigation, held-out, sealed,
  deployment, and promotion remain closed.
- Any successor must change the learned perception mechanism or its spatial /
  cross-scene generalization—not another semantic operating point or
  calibration adapter.
