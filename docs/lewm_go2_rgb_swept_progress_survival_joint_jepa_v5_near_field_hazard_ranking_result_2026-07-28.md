# RGB Swept-Progress Survival Joint-JEPA V5 Near-Field Hazard Ranking — Terminal Result

- Terminal status: `FAIL_DEVELOPMENT_FULL_ARM` — valid complete scientific failure, not an execution or integrity failure.
- Independent result/trace audit: PASS. The unchanged gate passed 23/24 checks; only `semantic_free_recall` failed.
- Preregistration / interpretation / source / execution binding commits: `7fe075d752b5d14c539eaed213c9f28510659c79` / `23f7388b68897a9db876909fd4ebd5b3f0bae52b` / `f01a66f12f6cac0da69b86c3668484a7b771d0bc` / `1f1097f0463e8b0856b911e9dc7f2db8fd55f8d8`.
- Result file/content SHA-256: `ed70fe7fd677b14b43017914d27f3d24fbc2abe2c0038f8a119421794bb77ae6` / `b2c3e96cf6f0b5b586de56104166dce8e4b38e28b580d7e750cefe228fb9b291`; 66,923 bytes.
- Training-trace file/content SHA-256: `3c3074697a833409b9da9a8300b01fea7cc684aaeee3bafa5f9a12d98c0b4fa4` / `3ff158befe85f269f0694159984446d0f35ca61c6fdd654e2a31131e5461fc77`; 1,370,786 bytes.
- Rejected terminal-checkpoint file SHA-256: `3199b8c19416df7f116c1bfee6322b7d055a2d961339ba109b1fba821fd8acb8`; 25,677,375 bytes. It was hashed for custody but not loaded or tensor-inspected after the failure, and no further access is authorized.

## Execution validity

- The one write-once attempt completed exactly 1,000 updates, 16,000 presentations, 4,000 microbatch graphs/backward calls and predictor objectives, and 1,000 optimizer and EMA steps.
- The canonical trace contains contiguous updates `1..1000` and presentations `16..16000`. Every row has exactly `S,P,U,R,O,H,L`; `L=S+P+U+R+O+H` holds with maximum absolute receipt error `7.674098e-7`.
- The model was freshly initialized from the accepted N320 encoder-only source with the unchanged V4 model, decoder, seeds, data, schedule, optimizer, controls, metrics, and 24-check evaluation.
- The only scientific delta was the coefficient-one, no-new-parameter near-field hazard-ranking loss. Training remained joint: `H` backpropagated through the semantic decoder, BEV lift, and RGB encoder while the action-conditioned predictor retained all inherited JEPA objectives.
- Finite nonzero gradient-L2 ranges were encoder `3.798419–137.191526`, lift/semantic `1.641304–23.642539`, and predictor `1.634242–41.527541`.

## Registered loss behavior

- `H` was active in 3,968/4,000 microbatches, with 10,359 eligible current samples, 10,005 eligible next samples, and 22,579,859 complete Cartesian hazard pairs.
- Mean microbatch `H` fell from `2.393182` in updates 1–100 to `0.553726` in updates 901–1000, a 76.9% reduction. The registered surrogate therefore trained strongly rather than remaining inactive.

| Loss | Updates 1–100 | Updates 801–900 | Updates 901–1000 |
|---|---:|---:|---:|
| Semantic `S` | `2.519685` | `2.015132` | `2.085713` |
| JEPA persistence `P` | `2.630470` | `1.101786` | `1.104798` |
| Survival `U` | `0.657401` | `0.327425` | `0.338871` |
| Progress ranking `R` | `0.849516` | `0.522121` | `0.531833` |
| Half-weight occupied auxiliary `O` | `2.296330` | `1.814352` | `1.882458` |
| Near-field hazard ranking `H` | `2.393182` | `0.582354` | `0.553726` |
| Total `L` | `11.346584` | `6.363170` | `6.497399` |

## Development gate

| Semantic metric | V4 | V5 | Gate | V5 result |
|---|---:|---:|---:|---|
| Balanced accuracy | `0.850286` | `0.814381` | `>=0.80` | PASS |
| Free recall | `0.857970` | `0.783955` | `>=0.85` | **FAIL** |
| Occupied recall | `0.744512` | `0.709053` | `>=0.70` | PASS |
| Rough occupied recall | `0.703615` | `0.671260` | `>=0.65` | PASS |
| Unknown recall | `0.948376` | `0.950136` | `>=0.90` | PASS |

- V5 predicted FREE less often: derived global free precision rose from about `0.9401` to `0.9504`, but recall fell by `0.0740`. Occupied precision also fell from about `0.0876` to `0.0735`. Thus the learned surrogate changed the class operating point and reduced useful-free coverage; it did not establish better obstacle/free separability on development selection.
- All swept-progress, family, and causal-control checks still passed. Full-arm selection utility was `0.888790`, selected zero-prefix rate `0.032581`, and unequal-pair concordance `0.863977`.
- Control equal-scene delta / bootstrap lower 95% / positive families were: coordinate-matched persistence `+0.108590 / +0.051065 / 7/8`; shuffled action `+0.312112 / +0.265725 / 8/8`; train action-mean prior `+0.059357 / +0.020317 / 6/8`; wrong RGB `+0.087044 / +0.051533 / 7/8`.

## Scientific conclusion and stopping decision

- V5 successfully optimized its training ranking objective but failed the prerequisite needed to trust the terminal representation. A lower `H` is therefore not evidence that the physical obstacle/free problem was solved.
- The failure does not authorize physical calibration, threshold selection, candidate admission, G2, navigation, held-out, sealed, production, deployment, or promotion access. The physical-calibration stage is recorded `CLOSED_FULL_ARM_GATE_FAILED` and did not run.
- V5 is closed with no retry, resume, coefficient/margin/range change, intermediate-checkpoint selection, schedule extension, or calibration attempt. Another loss-only variant is not the next step.
- The justified next experiment is one materially different architecture-level mechanism that adds higher-resolution spatial RGB evidence to the joint JEPA latent while returning to the clean V4 loss set and preserving the capped one-shot falsification.

## Access and custody

- Forbidden input count, fixed-negative RGB requests, every forbidden semantic-loader counter, and G2/navigation-final-evaluation opens were all zero.
- Consumed roles remained authority, index, train, probability calibration, and development checkpoint selection. Held-out/sealed access, physical calibration, qualification, and promotion are all recorded false.
