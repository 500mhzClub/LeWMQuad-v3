# RGB Swept-Progress Survival Joint-JEPA V3 Half Occupied-Safety Auxiliary — Terminal Result

- Terminal status: `FAIL_FULL_ARM` — a valid, complete scientific failure, not an execution failure.
- Independent terminal receipt audit: PASS; execution validity and scientific outcome were classified separately.
- The single write-once V3 attempt completed on 2026-07-28. The terminal `result.json` is complete; an exit code of `2` is the registered scientific-gate outcome.
- Exactly one of 24 registered gate checks failed: `semantic_free_recall`.
- Result file SHA-256, observed by the parent executor: `fcbfa78521276f9db4abd703f716ae4fa2b2b2bef7bacb2b89d3624fd6fb0ab3`; result content SHA-256: `acb546b13be5c7cdf699e2d57c6fdbf24e1b1363ebd380aeb94e1defd7d163a7`.
- Training-trace file SHA-256, observed by the parent executor and bound in the result: `0f5aba10e2e79f829a137ee30e2e39378fd80bc1f357d23cc0a8f46d19c9445b`; trace content SHA-256: `f7b22492e9c0c609fa9d162e912ef37c37f1bb0def4105195cc0a8a9b59e1b82`.
- No produced model-artifact identity is included in this receipt, and no produced model artifact was opened, hashed, inspected, loaded, or otherwise accessed during result review.

## Execution validity and frozen identity

- Exact terminal accounting completed: 1,000 updates, 16,000 presentations, 4,000 microbatch graphs, 4,000 backward calls, 4,000 predictor forwards/objectives, and 1,000 optimizer and EMA steps.
- The trace is `COMPLETE` and contains exactly 1,000 rows: updates `1..1000` and presentations `16..16000` are contiguous and exact. Every row has the frozen loss keys `S,P,U,R,O,L` and gradient groups `encoder`, `lift_semantic`, and `predictor`.
- The traced identity `L=S+P+U+R+O` holds to floating-point roundoff; maximum observed absolute row error is `6.854534149169922e-07`.
- The sole V3 scientific delta is exactly the occupied-vs-rest auxiliary coefficient `1.0 -> 0.5`. The occupied logit, per-raster-row present-class balancing, current/next equal averaging, `log(2)` normalization, update-1 joint route, model, data, optimizer, schedule, evaluator, controls, and gates remained unchanged.
- Frozen source commit: `5543a3a25bc9de0519165e8006aba3faff597ef1`; preregistration commit: `2b917fcd4d8e4115f15b57d4fc26691a39c37328`.
- Source hashes copied from the frozen execution binding:
  - refactored behavior-identical V2 training core: `6f76dd5b098ff360a3ada5bbb18f74a13342f3a5212e871da6db8f5f3a5bb1bf`;
  - V3 training core: `7cab73752593b12b638b55710714ff956a2441e92df2fe775902472a7b69a8cb`;
  - V3 training-core test: `48376097957911eadf1c40db3b2e28cb1ead0b7e93384a605841f17e5a273852`;
  - V3 executor: `164e2baf53f2a882ef18eabeee99ae4b2c27a7d8d799543c798f24a49782b182`;
  - V3 executor test: `a291adfcf9d42e16db40f17c477827bb9db280fe8f6054d7862a271fd67d4c7f`.
- Frozen runtime identities recorded by the result also match:
  - label-manifest file/content SHA-256: `edc0df8c796f97d3f91c8c3796e9795a4355dceac79770b91de382132fe8e1d3` / `6e0ea572612cdf94cb6dd91dffb90e50c828053617f69b42307161c958700c03`;
  - schedule-prefix SHA-256: `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`;
  - predicted-next/post-action mask SHA-256: `11ae5e26b182da85c8a7ca866ee4914c72b5b84b8b601dd807903097d754485c`;
  - coordinate-matched current-frame persistence mask SHA-256: `c4b8c475032433e448cd7df9decfead2c0800426219098f45306a0540154d2ff`.
- Initialization used only the accepted N320 encoder input. The result explicitly records that no predecessor experiment state was read.
- Hardware and determinism matched the binding: exactly one visible `AMD Radeon AI PRO R9700`, `HIP_VISIBLE_DEVICES=0`, deterministic algorithms enabled, cuDNN benchmark disabled, cuDNN deterministic enabled, and TF32 disabled.

## Registered semantic gates

| Metric | V3 | Gate | Result |
|---|---:|---:|---|
| Balanced accuracy | `0.845907` | `>= 0.80` | PASS |
| Free recall | `0.846040` | `>= 0.85` | **FAIL** |
| Occupied recall | `0.745270` | `>= 0.70` | PASS |
| Unknown recall | `0.946411` | `>= 0.90` | PASS |
| Rough-family occupied recall | `0.725384` | `>= 0.65` | PASS |

- The sole miss is free recall by `0.003960` absolute.
- Full selection confusion matrix, with true rows and predicted columns ordered `UNKNOWN, FREE, OCCUPIED`:
  - unknown: `[3336492, 24511, 164414]`;
  - free: `[4139, 423531, 72934]`;
  - occupied: `[5040, 2352, 21627]`.
- Rough-family confusion matrix in the same order:
  - unknown: `[283070, 7840, 13699]`;
  - free: `[1474, 189583, 23615]`;
  - occupied: `[384, 991, 3632]`.
- Most remaining free-space errors still lie on the directly targeted conservative boundary: `72,934` free cells were predicted occupied versus `4,139` predicted unknown.

## Registered swept-progress gates and diagnostics

- Selection population: 495 states, including 399 informative states and 8,528 unequal non-HOLD action pairs.
- Overall normalized chosen/oracle prefix utility: `0.908511` versus floor `0.85` — PASS.
- Overall selected zero-prefix rate: `0.037594` versus ceiling `0.05` — PASS.
- Overall unequal-prefix pair concordance: `0.863508` versus floor `0.75` — PASS.
- Expected-progress MAE: `0.247495 m` over all selection actions and `0.210166 m` on informative states. Weighted progress-calibration gap: `0.032757 m`.
- All eight registered families passed utility `>=0.70`, zero-prefix rate `<=0.20`, and concordance `>=0.60`:

| Selection family | Informative states | Utility | Zero-prefix rate | Concordance | Unequal pairs |
|---|---:|---:|---:|---:|---:|
| large enclosed maze | 64 | `0.899664` | `0.031250` | `0.852691` | 1,412 |
| local composite motifs | 51 | `0.922719` | `0.039216` | `0.881279` | 1,095 |
| loop alias stress | 61 | `0.910302` | `0.032787` | `0.843750` | 1,344 |
| medium enclosed maze | 64 | `0.867514` | `0.078125` | `0.837473` | 1,409 |
| open obstacle field | 26 | `0.893483` | `0.038462` | `0.791139` | 474 |
| rough local dynamics | 22 | `0.943015` | `0.000000` | `0.841202` | 466 |
| small enclosed maze | 47 | `0.932270` | `0.021277` | `0.929990` | 957 |
| visual sensor stress | 64 | `0.922121` | `0.031250` | `0.892779` | 1,371 |

- The ungated probability-calibration role also remained strong overall: utility `0.932423`, zero-prefix rate `0.026706`, concordance `0.880985`, all-action MAE `0.220902 m`, informative MAE `0.183200 m`, and weighted calibration gap `0.061484 m` across 415 states/337 informative states.

## Registered control gates

- Every control comparison passed all three registered requirements: positive equal-scene mean utility delta, strictly positive 10,000-replicate paired-scene bootstrap lower bound, and at least 6/8 positive families.

| Control | Equal-scene delta | Bootstrap lower 95% | Positive families | Result |
|---|---:|---:|---:|---|
| Coordinate-matched persistence | `+0.156161` | `+0.098134` | 8/8 | PASS |
| Shuffled predicted-action slots | `+0.335193` | `+0.276343` | 8/8 | PASS |
| Wrong RGB | `+0.098577` | `+0.055884` | 7/8 | PASS |
| Train action-mean prior | `+0.077225` | `+0.041892` | 7/8 | PASS |

- The bootstrap seed was `20260728` for every comparison. Wrong RGB had one negative family delta (`open_obstacle_field`, `-0.004808`), and the action prior had one zero family delta (`open_obstacle_field`, `0.0`); both retained positive aggregate effects, positive lower bounds, and 7/8 positive families.

## Training trend

| Loss | Updates 1–100 | Updates 801–900 | Updates 901–1000 |
|---|---:|---:|---:|
| Semantic `S` | `2.145104` | `1.901689` | `1.978013` |
| JEPA persistence `P` | `2.468646` | `1.188364` | `1.214977` |
| Survival `U` | `0.680868` | `0.331406` | `0.342635` |
| Ranking `R` | `0.843038` | `0.514199` | `0.526438` |
| Half-weight occupied auxiliary `O` | `1.790144` | `1.639204` | `1.713821` |
| Total `L` | `7.927801` | `5.574862` | `5.775885` |

- Training learned substantially rather than collapsing: total, JEPA persistence, survival, and ranking losses all improved strongly from the first to last 100 updates.
- Every loss worsened from updates 801–900 to 901–1000, including the separately traced half-weight auxiliary. This is a late plateau/regression, not evidence that more identical updates would resolve the free-recall miss.
- Ranking was active in all 4,000 microbatches, with 284,795 eligible action pairs and 1,318,068 supervised survival decisions.
- Finite gradient-L2 ranges remained nonzero in every trained group: encoder `1.932219–45.078938`, lift/semantic `1.057430–34.339401`, and predictor `1.993783–41.915590`.

## Coefficient-family conclusion

| Attempt | Auxiliary coefficient | Free recall | Occupied recall | Rough occupied recall | Terminal semantic outcome |
|---|---:|---:|---:|---:|---|
| V1 | `0.0` | `0.885680` PASS | `0.644302` FAIL | `0.580587` FAIL | rejected |
| V2 | `1.0` | `0.838621` FAIL | `0.777180` PASS | `0.768724` PASS | rejected |
| V3 | `0.5` | `0.846040` FAIL | `0.745270` PASS | `0.725384` PASS | rejected |

- V3 moved the semantic boundary between the two endpoint attempts and preserved every swept-progress and control gate, but it did not meet the fixed free-recall floor.
- The registered midpoint falsification therefore failed. The occupied-safety coefficient family is closed: do not extend V3, select an intermediate update, warm-start, retry, resume, or run another coefficient.
- This result does not falsify the learned swept-progress predictor itself; it shows that coefficient-only rebalancing did not make all required dense semantic recalls pass simultaneously under the fixed test.

## Access, authority, and next boundary

- Forbidden input count: `0`; fixed-negative RGB requests: `0`; every forbidden semantic-loader counter: `0`; G2/navigation/final-evaluation open count: `0`.
- The authorized N320 gate and encoder input opens both completed successfully. Raw consumed roles were limited to authority, index, training, probability calibration, and development selection across 9,640 consumed records.
- This was development-only. Qualification is false, promotion was not performed, retry/resume is unauthorized, and no G2 navigation final evaluation was opened.
- The matched no-JEPA arm did not run because the full arm failed; no JEPA treatment-effect claim is made.
- No G2, navigation, sealed, held-out, production, deployment, or promotion execution is authorized by this result. Any successor must be a materially different preregistered mechanism rather than another occupied-safety coefficient adjustment.
