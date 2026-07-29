# RGB Swept-Progress Survival Joint-JEPA V10 Projective Cell-Volume Token Lift — Development Result

- Date: 2026-07-29.
- Outcome: `PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION`; all 24 frozen
  development checks passed at the sole update-1000 decision state.
- Authority: preregistration `b9eaae6560c42e588c86fb8bf949cc95bd9e29e9`,
  frozen source `8a239d2c9a7d602533cd76545b32a9672d187b48`, and
  execution binding `f47dcc364f8f2383a04ed1eba55027d1024b5e19`.
- The exact bound command exited zero. The two ROCm nondeterminism warnings
  were permitted by the frozen `warn_only=True` setting; no error, retry,
  resume, or second attempt occurred.

## Artifact and training integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `result.json` | 70,550 | `f62fa6c908fe8cfb4ae838878d40b615e14ad343d5f123c1dd24e16f274bbb70` |
| `training_trace.json` | 967,734 | `383e002bc06fd2f319ccfb963c56c7a0be7eb16e7d23fe48a95f0f140450ca45` |
| `checkpoint_update_1000.pt` | 29,741,203 | `f63a037868de1e4db465fb4f85af2b8e6eba9883880c19d908216db20d82faa0` |

- Result content SHA-256 is
  `01ce5f55d3b2cc264b21a9924d27e64568873dfaf2a2364e1448991adda0b6b6`;
  trace content SHA-256 is
  `6501c51b4d3317fd5525816258f73cf9c4ba8015b1a32e648dcff75be82eb05b`.
- Exact accounting: 1,000 optimizer updates, 1,000 EMA updates, 4,000
  microbatch graphs/backwards/predictor objectives, and 16,000 ordered
  presentations. The trace contains exactly updates 1 through 1,000 and all
  12,000 audited loss/gradient values are finite.
- All seven online Q/K/V/O tensors were active from update 1 through all 1,000
  updates; attention gradient L2 stayed finite (`0.051426` to `0.232006`).
  Target attention gradient-tensor count remained zero.
- Total joint loss fell from `4.206309` at update 1 to `2.277561` at update
  1,000. Encoder, lift/semantic, and predictor gradients remained finite and
  nonzero across the trace.

## Development evidence

- Selection semantic balanced accuracy: `0.902738` (gate `>=0.80`).
- FREE / OCCUPIED / UNKNOWN recall: `0.882500` / `0.874255` / `0.951460`
  (gates `>=0.85` / `>=0.70` / `>=0.90`). Rough OCCUPIED recall was
  `0.734971` (gate `>=0.65`).
- Navigation-surrogate selection utility was `0.906305`, unequal-pair
  concordance `0.867378`, and zero-prefix rate `0.017544`; every registered
  family passed its frozen utility, concordance, and zero-prefix checks.
- Full-model advantage remained positive against every control. Bootstrap
  lower-95 deltas were `0.064947` versus wrong RGB, `0.259910` versus shuffled
  action, `0.026101` versus the train-action mean prior, and `0.052961` versus
  coordinate-matched current-frame persistence. The corresponding positive
  family counts were 8, 8, 6, and 7.

## Decision boundary

- V10 passes the development falsification: the new 3D cell-volume routing is
  trainable, the RGB/action predictor uses it, and it clears the unchanged
  semantic, utility, family, and causal-control tests.
- This is not physical qualification and does not open G2 or any held-out
  maze. It authorizes only one separately frozen use of the unchanged 2,016
  threshold-tuple physical-evidence calibration gate on this exact terminal
  checkpoint.
- No predecessor experiment checkpoint, G2, navigation, held-out, or sealed
  material was opened. The checkpoint remains development-only and
  unqualified pending the physical gate.
