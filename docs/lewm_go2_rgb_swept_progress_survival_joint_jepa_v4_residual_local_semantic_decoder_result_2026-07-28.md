# RGB Swept-Progress Survival Joint-JEPA V4 Residual Local Semantic Decoder — Terminal Result

- Terminal status: `PASS_FULL_ARM` — a valid, complete full-arm scientific pass.
- Independent terminal receipt audit: PASS; all hashes, accounting, architecture receipts, and gates recomputed cleanly.
- The single write-once V4 attempt completed on 2026-07-28 with all 24 registered gate checks passing and no failed checks.
- Result file SHA-256, observed by the parent executor: `bf93c96cf020553be74d51847c6876e345cd6cc391b05cec186e36b20ca15aa4`; result content SHA-256: `27ecf4895dfea01a1e5bb4f6f13f3add6a182a8dfa4b9f8651204bd1e6222ad8`.
- Training-trace file SHA-256, observed by the parent executor and bound in the result: `2ad16afd722ada26439c4ebfb2993330ec3abe1cbe4a75ced496a7c2a2b1580b`; trace content SHA-256: `bb027f8af94f352aac3ca1291a84285e25df431ca90682660afc7e1b476d4c12`.
- No produced model-artifact identity is included in this receipt, and no produced model artifact was opened, hashed, inspected, loaded, or otherwise accessed during result review.
- This is not yet a JEPA treatment-effect result or a navigation qualification. It authorizes only the preregistered matched no-JEPA arm using the identical V4 decoder.

## Execution validity and frozen identity

- Exact terminal accounting completed: 1,000 updates, 16,000 presentations, 4,000 microbatch graphs, 4,000 backward calls, 4,000 predictor forwards/objectives, and 1,000 optimizer and EMA steps.
- The trace is `COMPLETE` and contains exactly 1,000 rows: updates `1..1000` and presentations `16..16000` are contiguous and exact. Every row has frozen loss keys `S,P,U,R,O,L` and gradient groups `encoder`, `lift_semantic`, and `predictor`.
- The identity `L=S+P+U+R+O` holds to floating-point roundoff; maximum observed absolute row error is `7.152557373046875e-07`.
- Preregistration commit: `9f9ab784b4bfa827585ec095f2a7f7a30333480a`; source commit: `aaa47a138d0eeb78aa20d9524e67f813f7a74a41`.
- Source hashes copied from the frozen execution binding:
  - behavior-preserving V2 shared training internals: `6f76dd5b098ff360a3ada5bbb18f74a13342f3a5212e871da6db8f5f3a5bb1bf`;
  - frozen V3 coefficient-`0.5` training core: `7cab73752593b12b638b55710714ff956a2441e92df2fe775902472a7b69a8cb`;
  - frozen V3 executor helper source: `164e2baf53f2a882ef18eabeee99ae4b2c27a7d8d799543c798f24a49782b182`;
  - V4 model: `1c5a26f02a856d9a84903063c53bf23095142d86885787556b09388c508711ef`;
  - V4 model test: `05e2783eeeffbe231b9e1128aae4695d5a6f695ea566ca64f0336bbf730763b2`;
  - V4 executor: `243ef91ccec4e1fcdfa5a0c3f112bf4c645f46ba7de8692c1dddcb47f87c9f40`;
  - V4 executor test: `712a7666a7ff1fb610c0d9a6e5013125db9ac543d98580e652c71d61e89fb021`.
- Frozen runtime identities recorded by the result also match:
  - label-manifest file/content SHA-256: `edc0df8c796f97d3f91c8c3796e9795a4355dceac79770b91de382132fe8e1d3` / `6e0ea572612cdf94cb6dd91dffb90e50c828053617f69b42307161c958700c03`;
  - schedule-prefix SHA-256: `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`;
  - predicted-next/post-action sweep-mask SHA-256: `11ae5e26b182da85c8a7ca866ee4914c72b5b84b8b601dd807903097d754485c`;
  - coordinate-matched current-frame persistence-mask SHA-256: `c4b8c475032433e448cd7df9decfead2c0800426219098f45306a0540154d2ff`.
- Initialization used only the accepted N320 encoder input; the result records no predecessor experiment-state read.
- Hardware and determinism matched the binding: one visible `AMD Radeon AI PRO R9700`, `HIP_VISIBLE_DEVICES=0`, deterministic algorithms enabled, cuDNN benchmark disabled, cuDNN deterministic enabled, and TF32 disabled.

## Verified V4 architecture receipt

- Sole scientific change from V3: the semantic readout became the exact inherited base `Conv2d(64,3,1,bias=True)` plus a residual `Conv2d(64,64,3,padding=1,bias=True) -> GELU(approximate="none") -> Conv2d(64,3,1,bias=True)` branch.
- The residual branch added exactly 37,123 parameters; the complete wrapped semantic decoder contained 37,318 parameters. Every semantic parameter entered the inherited lift/semantic optimizer and clipping group exactly once.
- Decoder initialization seed was exactly `20260713`; all inherited component seeds remained unchanged. The residual output weight and bias were exact zero at initialization, preserving the inherited base logits initially.
- No normalization layer was added. The exact inherited post-logit visibility mask remained active: bool shape `[64,64]`, 1,964 visible cells, SHA-256 `cbcdb7d6fda08626522732ff092d90a87f5b5f2cd2534baf2bb4aa556d832753`.
- V3's occupied-vs-rest auxiliary remained exactly coefficient `0.5`, with unchanged occupied logit, row-present-class balancing, current/next averaging, and `log(2)` normalization. Data, optimizer rules, losses, schedule, controls, evaluation, caps, and accounting did not change.

## Registered semantic gates

| Metric | V4 | Gate | Result |
|---|---:|---:|---|
| Balanced accuracy | `0.850286` | `>= 0.80` | PASS |
| Free recall | `0.857970` | `>= 0.85` | PASS |
| Occupied recall | `0.744512` | `>= 0.70` | PASS |
| Unknown recall | `0.948376` | `>= 0.90` | PASS |
| Rough-family occupied recall | `0.703615` | `>= 0.65` | PASS |

- V4 cleared the V3 free-recall failure: `0.846040 -> 0.857970`, moving from `0.003960` below the floor to `0.007970` above it.
- Occupied and rough-family occupied recall remained above their fixed floors despite moving slightly downward from V3; balanced accuracy and unknown recall also passed.
- Full selection confusion matrix, with true rows and predicted columns ordered `UNKNOWN, FREE, OCCUPIED`:
  - unknown: `[3343420, 24805, 157192]`;
  - free: `[3274, 429503, 67827]`;
  - occupied: `[4875, 2539, 21605]`.
- Rough-family confusion matrix in the same order:
  - unknown: `[284086, 8049, 12474]`;
  - free: `[991, 192965, 20716]`;
  - occupied: `[376, 1108, 3523]`.

## Registered swept-progress gates and diagnostics

- Selection population: 495 states, including 399 informative states and 8,528 unequal non-HOLD action pairs.
- Overall normalized chosen/oracle prefix utility: `0.906910` versus floor `0.85` — PASS.
- Overall selected zero-prefix rate: `0.035088` versus ceiling `0.05` — PASS.
- Overall unequal-prefix pair concordance: `0.868433` versus floor `0.75` — PASS.
- Expected-progress MAE: `0.248097 m` over all selection actions and `0.213705 m` on informative states. Weighted progress-calibration gap: `0.044989 m`.
- All eight registered families passed utility `>=0.70`, zero-prefix rate `<=0.20`, and concordance `>=0.60`:

| Selection family | Informative states | Utility | Zero-prefix rate | Concordance | Unequal pairs |
|---|---:|---:|---:|---:|---:|
| large enclosed maze | 64 | `0.889619` | `0.031250` | `0.866147` | 1,412 |
| local composite motifs | 51 | `0.938405` | `0.019608` | `0.876712` | 1,095 |
| loop alias stress | 61 | `0.893863` | `0.049180` | `0.837798` | 1,344 |
| medium enclosed maze | 64 | `0.877259` | `0.062500` | `0.844571` | 1,409 |
| open obstacle field | 26 | `0.893483` | `0.038462` | `0.810127` | 474 |
| rough local dynamics | 22 | `0.943015` | `0.000000` | `0.854077` | 466 |
| small enclosed maze | 47 | `0.922340` | `0.021277` | `0.933124` | 957 |
| visual sensor stress | 64 | `0.922902` | `0.031250` | `0.898614` | 1,371 |

- The ungated probability-calibration role remained strong overall: utility `0.916780`, zero-prefix rate `0.023739`, concordance `0.873236`, all-action MAE `0.229328 m`, informative MAE `0.195976 m`, and weighted calibration gap `0.065980 m` across 415 states/337 informative states.

## Registered control gates

- Every control comparison passed all three requirements: positive equal-scene mean utility delta, strictly positive 10,000-replicate paired-scene bootstrap lower bound, and at least 6/8 positive families.

| Control | Equal-scene delta | Bootstrap lower 95% | Positive families | Result |
|---|---:|---:|---:|---|
| Coordinate-matched persistence | `+0.179119` | `+0.115771` | 8/8 | PASS |
| Shuffled predicted-action slots | `+0.337657` | `+0.280273` | 8/8 | PASS |
| Wrong RGB | `+0.096937` | `+0.059089` | 7/8 | PASS |
| Train action-mean prior | `+0.075950` | `+0.040777` | 7/8 | PASS |

- Bootstrap seed was `20260728` for every comparison. Wrong RGB had one negative family delta (`open_obstacle_field`, `-0.002244`), and the action prior had one zero family delta (`open_obstacle_field`, `0.0`); both retained positive aggregate effects, positive lower bounds, and 7/8 positive families.

## Training trend

| Loss | Updates 1–100 | Updates 801–900 | Updates 901–1000 |
|---|---:|---:|---:|
| Semantic `S` | `2.132545` | `1.895888` | `1.974049` |
| JEPA persistence `P` | `2.458515` | `1.180631` | `1.215299` |
| Survival `U` | `0.684298` | `0.333538` | `0.344653` |
| Ranking `R` | `0.842298` | `0.517157` | `0.534385` |
| Half-weight occupied auxiliary `O` | `1.786279` | `1.635081` | `1.710919` |
| Total `L` | `7.903935` | `5.562295` | `5.779305` |

- Training learned substantially rather than collapsing: total, JEPA persistence, survival, and ranking losses all improved strongly from the first to last 100 updates.
- Every loss worsened from updates 801–900 to 901–1000. The registered decision nevertheless uses the terminal update-1000 model, and the complete fixed gate passed; this does not authorize schedule extension or intermediate selection.
- Ranking was active in all 4,000 microbatches, with 284,795 eligible action pairs and 1,318,068 supervised survival decisions.
- Finite gradient-L2 ranges remained nonzero in all trained groups: encoder `1.930747–49.041371`, lift/semantic `1.143748–31.809569`, and predictor `1.993783–41.400567`.

## Scientific interpretation and narrow next authority

- V4 is the first full-arm pass in this line: all semantic-retention, swept-progress, family, and control gates passed together under the frozen terminal evaluation.
- The result supports the narrow hypothesis that a small nonlinear one-cell-local semantic decoder can use useful spatial information in the jointly learned BEV latent that the inherited per-cell linear readout did not expose reliably.
- This was a semantic-decoder test, not an encoder test. It does not by itself show that JEPA training caused the useful features, and it does not establish a navigation or held-out-maze result.
- The matched no-JEPA arm did not run in this attempt, and no JEPA treatment-effect claim is made yet.
- The only newly authorized experiment is one matched no-JEPA arm using the identical V4 decoder and the frozen matched-control contract. The decoder architecture, initialization, data, schedule, optimizer, cap, evaluation, and gates must remain identical; only the preregistered removal of the JEPA contribution may differ.
- No retry, resume, schedule extension, intermediate selection, alternate decoder, coefficient change, G2, navigation, sealed, held-out, production, deployment, or promotion action is authorized by this full-arm pass.

## Access and custody

- Forbidden input count: `0`; fixed-negative RGB requests: `0`; every forbidden semantic-loader counter: `0`; G2/navigation/final-evaluation open count: `0`.
- Authorized N320 gate and encoder-input opens completed successfully. Raw consumed roles were limited to authority, index, training, probability calibration, and development selection across 9,640 consumed records.
- The result is development-only. Qualification and promotion remain false, retry/resume remains unauthorized, and held-out/sealed access is recorded false.
