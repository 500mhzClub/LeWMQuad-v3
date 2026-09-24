# RGB Swept-Progress Survival Joint-JEPA V7 Hierarchical CNN Encoder — Result

- Terminal status: `FAIL_DEVELOPMENT_FULL_ARM`.
- Scientific disposition: valid complete capped run; V7 failed five of the
  unchanged 24 development checks and is closed without calibration, retry,
  resume, tuning, checkpoint use, or G2 access.
- Preregistration / model / executor / execution-binding commits:
  `34c4a33e2fa25926b3127e0c893755757426cfd4` /
  `79bbc50f57bc6a6ca20b77d85c5f86dc740e77f5` /
  `23f1d97bd148a9554715ad4c670c41cee1bca0e7` /
  `824921afe09400ab879bd417001d85f6f075f2b1`.
- Independent frozen-source review passed with zero blockers. Focused V7 tests
  passed 14/14 and the complete relevant V1--V7 regression set passed 151/151.

## Execution and integrity

- The sole attempt completed exactly 1,000 optimizer updates, 1,000 EMA
  updates, 4,000 microbatch graphs/backward calls, 4,000 predictor
  forwards/objectives, and 16,000 presentations.
- The trace contains exactly 1,000 ordered rows for updates `1..1000` and
  presentations `16..16000`. All values are finite and
  `L=S+P+U+R+O` holds to maximum absolute floating-point error
  `1.25169754e-6`.
- Result: 63,366 bytes; file/content SHA-256
  `3027d6f16278cb3dd66ff0cf6e1fa920d9e03f24c0721375ad9e9c7f735d68bc` /
  `1bfeb2142e9dec6f4bedb1fb786aef840bd579cbc5cf0be4e726c0f5aa0d81d7`.
- Training trace: 901,536 bytes; file/content SHA-256
  `780efcd705343bc9ce474d8863ba3d5788e1bb0bcc20529860b2001e0c104deb` /
  `389adf9c5310e4c9ecf4d4f503cd75a123526a5ccb78775a915d692815c2228a`.
- The result embeds a terminal-checkpoint receipt of 19,645,823 bytes and
  SHA-256
  `a71ac3b6f54e8f364ce85e345f2c20670994b752a57d401cef6b191ab2557ea7`.
  The rejected checkpoint was not opened, loaded, independently hashed,
  listed, or otherwise inspected during audit, and no further access is
  authorized.
- Independent result/trace-only audit passed canonical JSON, hashes and
  binding, accounting, loss identity, gradient receipts, gate recomputation,
  access receipts, and terminal classification.
- Hardware and access receipts are valid: one visible
  `AMD Radeon AI PRO R9700`, forbidden input count zero, G2/navigation opens
  zero, fixed-negative RGB requests zero, and all forbidden semantic-loader
  counters zero. Held-out and sealed access remained false.

## Encoder and training behavior

- The fresh CNN had exactly 1,994,880 online parameters across 62 tensors.
  Every tensor received a finite nonzero gradient on every update; aggregate
  CNN gradient L2 stayed in `0.802872--0.991367`. Target-gradient tensor count
  remained zero.
- The runtime proved exact initial online/target CNN equality and exact initial
  equality of every inherited non-encoder V4 state tensor. No N320 parameter
  value was retained in the replacement CNN.

| First/last 100-update mean | First 100 | Last 100 | Change |
|---|---:|---:|---:|
| Total `L` | `11.821540` | `7.165786` | `-39.38%` |
| Semantic `S` | `2.296898` | `2.002447` | `-12.82%` |
| JEPA persistence `P` | `6.001055` | `2.418441` | `-59.70%` |
| Survival `U` | `0.774868` | `0.402212` | `-48.09%` |
| Ranking `R` | `0.888615` | `0.617479` | `-30.51%` |
| Half-weight occupied auxiliary `O` | `1.860105` | `1.725207` | `-7.25%` |

- V7 learned rather than collapsing, but it remained materially behind clean
  V4's last-100 total / JEPA means `5.779305 / 1.215299`. V7's corresponding
  values were `7.165786 / 2.418441`.
- Total loss reached `7.096716` over updates 801--900 and rebounded slightly to
  `7.165786` over updates 901--1000. The fixed terminal checkpoint remains the
  only evaluated checkpoint; the trend does not authorize an extension or
  intermediate selection.

## Unchanged development gate

| Metric | V4 | V7 | Gate | V7 result |
|---|---:|---:|---:|---|
| Semantic balanced accuracy | `0.850286` | `0.832915` | `>=0.80` | PASS |
| Free recall | `0.857970` | `0.878615` | `>=0.85` | PASS |
| Occupied recall | `0.744512` | `0.685758` | `>=0.70` | **FAIL** |
| Rough occupied recall | `0.703615` | `0.443779` | `>=0.65` | **FAIL** |
| Unknown recall | `0.948376` | `0.934373` | `>=0.90` | PASS |
| Informative action utility | `0.906910` | `0.837848` | `>=0.85` | **FAIL** |
| Selected zero-prefix rate | `0.035088` | `0.027569` | `<=0.05` | PASS |
| Unequal-pair concordance | `0.868433` | `0.808865` | `>=0.75` | PASS |

- V7 passed 19/24 checks. In addition to the three numeric failures above,
  the full arm did not reliably beat the train action-mean prior: its
  equal-scene utility delta was only `+0.003785`, bootstrap lower 95% was
  `-0.049211`, and only 4/8 families were positive. The lower-bound and family
  checks failed.
- All per-family absolute utility, zero-prefix, and concordance floors passed.
  Coordinate-matched persistence, shuffled-action, and wrong-RGB controls also
  retained their complete aggregate gates. Thus V7 was neither constant nor
  wholly action-blind, but its action-specific advantage was too weak and
  inconsistent.
- Relative to V4, V7 gained `0.020645` free recall but lost `0.058754`
  occupied recall and `0.259836` rough occupied recall. It moved toward
  predicting free space at the expense of conservative obstacle evidence—the
  opposite of the physical-separability improvement required after V4.
- Selection utility fell `0.069062` and concordance fell `0.059568` from V4.
  The hierarchical local bias therefore did not preserve enough of the
  transformer route's scene context and action-relevant representation under
  the frozen cap.

## Decision

- The wholesale hierarchical-CNN encoder hypothesis is falsified for this
  campaign. V7 and its checkpoint are closed. Do not try another CNN width,
  depth, normalization, seed, warm start, longer schedule, hybrid-CNN variant,
  threshold, or calibration pass.
- Physical calibration is `CLOSED_FULL_ARM_GATE_FAILED` and did not run. G2,
  navigation, held-out, sealed, production, deployment, and promotion remain
  closed.
- A successor, if pursued, must retain the evidence that global/contextual
  encoding and action-specific prediction matter while changing a mechanism
  outside the closed CNN, fine-RGB-fusion, loss-ranking, radial/ray,
  projective-height, and simple temporal families.
