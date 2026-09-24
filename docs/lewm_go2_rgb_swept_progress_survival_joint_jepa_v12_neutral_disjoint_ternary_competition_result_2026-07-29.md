# RGB Swept-Progress Survival Joint-JEPA V12 Neutral Disjoint Ternary Competition — Development Result

- Date: 2026-07-29.
- Outcome: `PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION`; all 24 frozen
  development checks passed at the sole update-1000 decision state.
- Authority: preregistration
  `ae1568e8f434d715d379eefc3eaf644369154f76`, frozen source
  `1c18fae5325b0ab1dd6b7c4e20fa51fb411f26aa`, and execution binding
  `aae34576576e1ba2daadc9fe5dbe023030813e09`.
- The exact bound command completed once with exit `0`. The two expected ROCm
  nondeterminism warnings were permitted by frozen `warn_only=True`; there was
  no crash, retry, resume, second seed, or intervention.

## Artifact and training integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `result.json` | 74,226 | `8268cabd23b57c66597c8ffd0f0b18b3eb296e9887acbc81363a666b70ff6ab6` |
| `training_trace.json` | 1,614,355 | `88aab1c17728e5f1272e8e313a2b1513de2f0b016ea2111588276151dfc80829` |
| `checkpoint_update_1000.pt` | 29,676,571 | `8212925759c0f496b0b6b1690168391d497c13688ba3cbb47b57640d173fe33f` |

- Result canonical content SHA-256 is
  `6a6a4ef0d8545b1510f9830cb35ebf67ea3e8cdff25006b889b2ef6d0511feff`;
  trace canonical content SHA-256 is
  `1de84b2c10a7b63f27408c219998b4952e8cc2b5ef15ec09a378bfbd1a9731cd`.
- Exact accounting: 1,000 optimizer updates, 1,000 EMA updates, 4,000
  microbatch graphs/backwards/predictor objectives, and 16,000 ordered
  presentations.
- All 14 online floor/elevated attention tensors had finite nonzero gradients
  from update 1 through all 1,000 updates; aggregate gradient L2 was
  `0.061312` to `0.264043`. Every one of the 12 semantic-axis tensors was
  active by update 2; aggregate gradient L2 was `0.005417` to `0.321268`.
  Target gradient-tensor count remained zero.
- Fresh V12 and V11 witnesses had identical parameter and buffer inventories,
  bit-identical initial values, 233 parameter tensors, and 6,122,053
  parameters. V12 added no parameter or optimizer member.
- Forbidden-input and G2-navigation open counts were zero. Held-out and sealed
  material remained unopened. No predecessor experiment checkpoint was read.

## Development result

- Every semantic check passed: balanced accuracy `0.904974` (required
  `>=0.80`), FREE recall `0.881789` (`>=0.85`), OCCUPIED recall `0.905441`
  (`>=0.70`), rough OCCUPIED recall `0.828240` (`>=0.65`), and UNKNOWN recall
  `0.927692` (`>=0.90`).
- The joint navigation surrogate passed: selection utility was `0.891992`,
  unequal-pair concordance `0.863743`, and zero-prefix rate `0.020050`; all
  registered family checks passed.
- Full-model advantage passed every registered wrong-RGB, shuffled-action,
  train-action-mean, and coordinate-matched-persistence conjunct. Bootstrap
  lower-95 deltas were respectively `0.066521`, `0.230382`, `0.003886`, and
  `0.035549`. Train-action-mean was the narrowest control and passed at the
  exact required 6 of 8 positive families.
- The probability-calibration role also remained viable: utility `0.895296`,
  pair concordance `0.866870`, and zero-prefix rate `0.014837`.

## What changed relative to V11 and V10

| Selection metric | V10 | V11 | V12 | V12 - V11 | V12 - V10 |
|---|---:|---:|---:|---:|---:|
| Balanced accuracy | `0.902738` | `0.885763` | `0.904974` | `+0.019211` | `+0.002236` |
| FREE recall | `0.882500` | `0.821735` | `0.881789` | `+0.060053` | `-0.000711` |
| OCCUPIED recall | `0.874255` | `0.903167` | `0.905441` | `+0.002274` | `+0.031186` |
| Rough OCCUPIED recall | `0.734971` | `0.794288` | `0.828240` | `+0.033952` | `+0.093269` |
| UNKNOWN recall | `0.951460` | `0.932386` | `0.927692` | `-0.004694` | `-0.023768` |
| Navigation-surrogate utility | `0.906305` | `0.893548` | `0.891992` | `-0.001556` | `-0.014313` |
| Pair concordance | `0.867378` | `0.864447` | `0.863743` | `-0.000704` | `-0.003635` |

- Relative to V11, V12 recovered 30,063 of the 30,419 lost correctly predicted
  FREE cells (`98.83%`). It removed 27,077 FREE-to-OCCUPIED errors and 2,986
  FREE-to-UNKNOWN errors. OCCUPIED true positives also rose by 66; the
  obstacle gain was not traded away to restore FREE.
- Relative to V10, V12's FREE confusion is nearly unchanged: 356 fewer FREE
  true positives, while OCCUPIED true positives rose by 905. This is the
  intended combination of V10-like floor recognition and stronger V11-style
  obstacle recognition.
- UNKNOWN recall paid a modest cost, but remained `0.027692` above its gate.
  Navigation utility and concordance also remained comfortably above their
  conjunctive thresholds, and all causal controls passed.

## Adjudication and next authority

- V12 passes the registered falsification. The result supports the diagnosis
  that V11's role-separated learned evidence was useful while its hard
  occupied-priority composition prevented strong floor evidence from winning.
- This is still a joint JEPA result: the shared 64-channel state, action
  predictor, survival head, semantic axes, encoder, and EMA target trained
  together from update 1. No separately trained predictor or semantic-only
  bypass was introduced.
- V12 earns exactly one separately preregistered application of the unchanged
  V10/V4 physical calibrator over the exact 2,016 threshold tuples. It does
  not yet pass the physical-evidence gate and is not yet qualified for G2.
- The checkpoint remains closed until that physical attempt is separately
  frozen and bound. G2, navigation, held-out, sealed, deployment, and
  promotion access remain closed.
