# RGB Swept-Progress Survival Joint-JEPA V6 Fine RGB BEV Fusion — Result

- Terminal status: `FAIL_DEVELOPMENT_FULL_ARM`.
- Scientific disposition: valid complete capped run; V6 failed one of the unchanged 24 development checks and is closed without calibration, retry, resume, tuning, checkpoint use, or G2 access.
- Preregistration / model / executor / execution-binding commits: `cc9ec66d796b37724e0a9e15d737813817e95265` / `407da6be6fa7e52c08844aedc883520d636b15b8` / `b8624910642e13fb12bb970e70e3cd96027f6699` / `b02ce3fdb8d4e1d32a14d05f2626f4f65b7fead8`.
- Independent frozen-source review: PASS with zero blocking or unresolved material findings. The relevant regression suite passed `82/82` tests; the focused V6 suite passed `14/14`.

## Execution and integrity

- The sole attempt completed exactly 1,000 optimizer updates, 1,000 EMA updates, 4,000 microbatch graphs/backward calls, 4,000 predictor forwards/objectives, and 16,000 presentations.
- The initialized 12,256-parameter fine-RGB branch had bitwise V4 latent and semantic parity through its exact-zero output projection. Its six online parameter tensors entered the lift/semantic optimizer and clipping group exactly once; its target copy remained frozen.
- The output projection and complete branch received gradient at update 1. Both earlier convolutions unlocked at update 2 and were active for the remaining 999 updates. Target gradient tensor count stayed zero.
- Result: 62,499 bytes; file/content SHA-256 `af153ededb66774763c9103e20f04b7c5930c8ad3a659666a5394ee439df63a7` / `2c2a645498c824e00f8113d8dc416411806d80fd1d87afc7ce8758854763f1c7`.
- Training trace: 743,854 bytes; file/content SHA-256 `dcd11bbcf7f702ca00b47dda6c90c81ed3385cf598e1e191be1c4aba0f83b78b` / `750c97b2945cd2677eca8669918488f0ae0e98554d60b408a60c92865aa9b605`. The binding in `result.json` matches.
- An independent result/trace-only audit returned PASS for schema, hashes, accounting, gate recomputation, access receipts, and scientific classification.
- The result declares the terminal checkpoint as 25,777,767 bytes with file SHA-256 `6a8d3126c495361a90b1049acc2d39deee7bf6f7bd89f9642efeb22cedee8a5f`. This value is recorded from the executor receipt; the rejected checkpoint was not opened, loaded, independently hashed, or otherwise inspected during audit, and no further access is authorized.
- All recorded result and trace numeric values are finite. Per-update total loss equals `S+P+U+R+O` to maximum absolute error `5.29e-7`.
- Hardware and access receipts are valid: one visible `AMD Radeon AI PRO R9700`, forbidden input count zero, G2/navigation-final opens zero, fixed-negative RGB requests zero, and every forbidden semantic-loader counter zero. Only the frozen authority/index/train/probability-calibration/`checkpoint_selection` roles were consumed. Held-out and sealed access remained false.

## Training behavior

| First/last 100-update mean | First 100 | Last 100 | Change |
|---|---:|---:|---:|
| Total `L` | `7.904026` | `5.780856` | `-26.86%` |
| Semantic `S` | `2.132469` | `1.974280` | `-7.42%` |
| JEPA persistence `P` | `2.458838` | `1.215706` | `-50.56%` |
| Survival `U` | `0.684192` | `0.345751` | `-49.47%` |
| Ranking `R` | `0.842336` | `0.534192` | `-36.58%` |
| Half-weight occupied auxiliary `O` | `1.786190` | `1.710927` | `-4.21%` |

- The fine branch mean gradient L2 rose from `0.025552` to `0.046152`; its full observed range was `0.009525–0.093721`. It was trained, not bypassed.
- Inherited encoder, lift/semantic, and predictor gradients remained finite and nonzero. Their first/last-100 means were `3.825366 -> 9.617260`, `2.277441 -> 4.539457`, and `6.772925 -> 4.460929`.
- V6's first/last loss windows were almost identical to clean V4's `7.903935 -> 5.779305`. The new path was alive, but it did not materially reshape joint learning within the cap.
- Total loss rebounded mildly from `5.565483` over updates 801–900 to `5.780856` over updates 901–1000, but there was no nonfinite value, missing gradient, accounting break, or execution instability. The terminal outcome is a scientific failure, not a runtime failure.

## Unchanged development gate

| Semantic metric | V4 | V5 | V6 | Gate | V6 result |
|---|---:|---:|---:|---:|---|
| Balanced accuracy | `0.850286` | `0.814381` | `0.849965` | `>=0.80` | PASS |
| Free recall | `0.857970` | `0.783955` | `0.848419` | `>=0.85` | **FAIL** |
| Occupied recall | `0.744512` | `0.709053` | `0.753093` | `>=0.70` | PASS |
| Rough occupied recall | `0.703615` | `0.671260` | `0.729179` | `>=0.65` | PASS |
| Unknown recall | `0.948376` | `0.950136` | `0.948383` | `>=0.90` | PASS |

- V6 passed 23/24 checks. Free recall missed by `0.0015808903`, or about 0.158 percentage point.
- V6 improved occupied recall by `0.008581` and rough occupied recall by `0.025564` over V4, but reduced free recall by `0.009551`; balanced accuracy was effectively unchanged (`-0.000321`). This is a small conservative class-operating-point shift, not demonstrated improvement in obstacle/free separability.
- V6 full-arm selection utility was `0.902561`, selected zero-prefix rate `0.042607`, and unequal-pair concordance `0.867261`; all passed, but all were slightly worse than V4's `0.906910`, `0.035088`, and `0.868433` respectively.
- All three family-level checks passed across all eight families, and all twelve causal-control checks passed. Wrong RGB, shuffled action, train-action mean prior, and coordinate-matched persistence retained positive aggregate and bootstrap effects.
- The complete selection confusion matrix, true rows and predicted columns ordered `UNKNOWN, FREE, OCCUPIED`, was `[3343445,22682,159290]`, `[3991,424722,71891]`, `[4917,2248,21854]`.

## Decision

- The 112-square fine-RGB residual route learned and recovered nearly all of V4's semantic behavior while improving occupied/rough-occupied recall, but it did not preserve enough free-space recall and did not improve the complete navigation-facing result. More updates, a second seed, threshold/calibration access, loss-weight adjustment, branch-width/depth change, alternate tap, or another fine-RGB-fusion variant are not authorized.
- Because the unchanged full-arm gate failed, the separately staged V6 physical-calibration step is `CLOSED_FULL_ARM_GATE_FAILED` and did not run. The checkpoint is unqualified and rejected. G2, navigation, held-out, sealed, production, deployment, and promotion remain closed.
- The next probe, if pursued, must be a materially different learned RGB perception mechanism justified by the combined V4 physical-calibration failure and the V5/V6 development evidence—not a refinement of V6 or another data/boilerplate iteration.
