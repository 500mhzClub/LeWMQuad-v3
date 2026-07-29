# RGB Swept-Progress Survival Joint-JEPA V8 — Native-Aspect High-Resolution ViT Result

- Terminal status: **VALID COMPLETE FAIL — DEVELOPMENT FULL ARM**.
- V8 passed 23 of 24 unchanged development checks. Its sole failure was free
  recall `0.8491042820` against the fixed `0.85` floor, a shortfall of
  `0.0008957180` or 0.0896 percentage point.
- Preregistration / source / execution-binding commits:
  `b17599fa1bb49017178f45d0e1a4c83ac8bb9314` /
  `0cd50cee09e7e4fbdfc696001e13c1b41a6e6772` /
  `fb6c13a5f09e37f0bfd9642016adc911ebfe9dff`.
- The sole authorized process exited `2` after writing a complete result. This
  is the registered scientific gate-failure exit, not an infrastructure
  failure. No retry, resume, alternate seed, extension, square variant,
  calibration, or G2 run occurred.

## Integrity and execution

- Result: `64,161` bytes; file SHA-256
  `6e711c1c1cf1c13eb9deda793eee2a56f46b3f15d82b66d1c2fad71b1ae065ef`;
  self-verifying content SHA-256
  `602d26a47e6688eeef571859d6de77807f5dbb61e838193b350f7b099bf94b82`.
- Training trace: `935,190` bytes; file SHA-256
  `8df7dc53e70ba3313453ee179d11f2b443d60640d8483ba751215cfae38fe459`;
  self-verifying content SHA-256
  `fc4656a37c4f1d126398bb28618e2a49e72e0d913042eaccfaf43dedffef557b`.
  The result's embedded trace binding matches all three values.
- The result embeds a development-checkpoint receipt of `26,463,423` bytes
  and SHA-256
  `73ff4bf332edbc7552629df747c9ea96a4b2961bba241f0fe6467ecfeb4e1b92`.
  That rejected checkpoint was not listed, statted, opened, or independently
  hashed; this line records only the already-authorized result receipt.
- Accounting is exact: 1,000 updates, optimizer steps, and EMA steps; 4,000
  microbatch graphs, backward calls, predictor forwards, and predictor
  objectives; 16,000 presentations; and 1,000 ordered trace rows with
  `presentations = 16 * update`.
- All 78 online encoder tensors, totaling `2,845,824` parameters, received
  finite nonzero gradients on all 1,000 updates. Encoder gradient L2 ranged
  `0.544177–0.978206`; target-gradient tensor count was zero.
- Initial migration receipts confirm the native `[3,168,224]` input, `24x32`
  patch lattice, `[B,769,192]` tokens, exact V4 non-positional state, exact
  target copy/freeze, one hard sync, CPU-float32 positional interpolation,
  V4-identical normalized sampling grids/masks, and native offset receipt
  `tanh(raw)*[4,3]`.
- The native loader decoded 9,460 exact `224x168` PNGs, with zero size
  mismatch and zero resize/crop/pad calls. Forbidden input count, fixed-negative
  RGB count, every forbidden semantic counter, and G2/final-evaluation opens
  were zero. Held-out and sealed material remained unopened.
- ROCm emitted the preregistered inherited `warn_only=True` nondeterminism
  warnings for grid-sample backward and memory-efficient attention. No
  nonfinite value, missing gradient, accounting break, or execution error
  occurred.

## Training behavior

| Loss mean | Updates 1–100 | Updates 801–900 | Updates 901–1000 |
|---|---:|---:|---:|
| Total `L` | `8.018095` | `5.531970` | `5.740078` |
| Semantic `S` | `2.136582` | `1.893177` | `1.968095` |
| JEPA persistence `P` | `2.546417` | `1.168131` | `1.191168` |
| Survival `U` | `0.701035` | `0.326879` | `0.344914` |
| Ranking `R` | `0.843236` | `0.510442` | `0.527599` |
| Half-weight occupied auxiliary `O` | `1.790825` | `1.633341` | `1.708302` |

- Joint learning was real and substantial, but every loss rebounded in the
  last 100 updates. The trace therefore gives no evidence that a longer
  schedule would rescue the terminal gate, independently of the explicit
  no-extension authority.
- `L` equals `S+P+U+R+O` throughout within maximum floating error
  `5.3e-7`. Ranking was active in all 4,000 microbatches, with 284,795 eligible
  pairs and 1,318,068 supervised survival decisions.

## Unchanged development gate

| Selection metric | V4 | V6 | V7 | V8 | Gate | V8 |
|---|---:|---:|---:|---:|---:|---|
| Balanced accuracy | `0.850286` | `0.849965` | `0.832915` | `0.849307` | `>=0.80` | PASS |
| Free recall | `0.857970` | `0.848419` | `0.878615` | `0.849104` | `>=0.85` | **FAIL** |
| Occupied recall | `0.744512` | `0.753093` | `0.685758` | `0.749337` | `>=0.70` | PASS |
| Rough occupied recall | `0.703615` | `0.729179` | `0.443779` | `0.754943` | `>=0.65` | PASS |
| Unknown recall | `0.948376` | `0.948383` | `0.934373` | `0.949480` | `>=0.90` | PASS |
| Informative action utility | `0.906910` | `0.902561` | `0.837848` | `0.906094` | `>=0.85` | PASS |
| Selected zero-prefix rate | `0.035088` | `0.042607` | `0.027569` | `0.032581` | `<=0.05` | PASS |
| Unequal-pair concordance | `0.868433` | `0.867261` | `0.808865` | `0.862805` | `>=0.75` | PASS |

- Relative to V4, V8 improved occupied recall by `0.004825` and rough occupied
  recall by `0.051328`, while free recall fell `0.008866`. Utility was nearly
  tied (`-0.000816`), zero-prefix improved `0.002507`, and concordance fell
  `0.005628`.
- Relative to V6, V8 improved free recall `0.000685`, rough occupied recall
  `0.025764`, utility `0.003533`, and zero-prefix rate `0.010026`; occupied
  recall fell `0.003756` and concordance fell `0.004456`.
- V8 decisively recovered the global-transformer/action signal lost by V7:
  utility `+0.068246`, concordance `+0.053940`, occupied recall `+0.063579`,
  and rough occupied recall `+0.311164`.
- All family utility, zero-prefix, and concordance checks passed. All twelve
  causal-control checks passed. Equal-scene delta / bootstrap lower 95% /
  positive families were: persistence `+0.143425 / +0.088514 / 8`, shuffled
  action `+0.323173 / +0.256775 / 8`, wrong RGB
  `+0.092280 / +0.044007 / 7`, and train-action prior
  `+0.070797 / +0.023909 / 7`.

## Decision and next scientific direction

- V8 demonstrates that native high-resolution global attention preserves the
  learned action-JEPA behavior and materially improves rough-obstacle
  recognition. It does not clear the complete frozen gate, and its slight
  closeness does not convert failure into a pass.
- Native-resolution/aspect variants, square follow-up, alternate interpolation,
  another seed, longer schedule, threshold or class-bias tuning, late fine-RGB
  fusion, and CNN or CNN/ViT variants are closed. The checkpoint is rejected,
  unqualified, non-resumable, and unauthorized for calibration or inspection.
- The strongest remaining common bottleneck is the inherited sparse
  four-sample image-token-to-BEV lift retained by V4, V6, V7, and V8. A next
  probe, if independently shown not to repeat committed history, should replace
  that sparse sampler with one learned geometry-aware dense or ray-column
  cross-attention evidence lift while preserving the proven V4 global ViT,
  joint JEPA predictor, data, losses, cap, and gates.
- This result opens no physical calibration, G2, navigation, held-out, sealed,
  production, deployment, or promotion authority.
