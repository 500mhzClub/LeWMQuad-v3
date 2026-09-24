# RGB Swept-Progress Survival Joint-JEPA V11 Height-Role Factorized Evidence Lift — Development Result

- Date: 2026-07-29.
- Outcome: `FAIL_DEVELOPMENT_FULL_ARM`; 23 of 24 frozen development checks
  passed at the sole update-1000 decision state. The failed check was FREE
  semantic recall.
- Authority: preregistration
  `b8ca8bd267e233a11f29da82842dcf5429743c18`, frozen source
  `8906f3922703785c38d52d00cc54d26bd81f8543`, and execution binding
  `4254109b120c8cbcae4cf3ca0bb9b0508c8e41c1`.
- The exact bound command completed once with exit `2`, a registered
  scientific failure. The two expected ROCm nondeterminism warnings were
  permitted by frozen `warn_only=True`; there was no crash, retry, resume,
  second seed, or intervention.

## Artifact and training integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `result.json` | 76,319 | `487ef2a99be04adbd262330cbcc3643d76d258d12f879fec9c631b5a5af2eeed` |
| `training_trace.json` | 1,615,152 | `31373bf21c3536ca85303517b149a678cc5962fdf84a3d9383812a489ff758b6` |
| `checkpoint_update_1000.pt` | 29,677,467 | `2feb8cd4e9096d726e50028e07cf7f5b3890d8cd3fdf57a052e171e28619ba3a` |

- Result canonical content SHA-256 is
  `470977ef5e285170bd6df7dbb691eb0d2335e2eed350107a3d0e2064d906ace8`;
  trace canonical content SHA-256 is
  `2f2996e9d87d842d537cee8f13debd0c52064efe23bec6d27d98f95eeed12a88`.
- Exact accounting: 1,000 optimizer updates, 1,000 EMA updates, 4,000
  microbatch graphs/backwards/predictor objectives, and 16,000 ordered
  presentations.
- All 14 online floor/elevated attention tensors had finite nonzero gradients
  from update 1 through all 1,000 updates; aggregate gradient L2 was
  `0.066709` to `0.260743`. Every one of the 12 semantic-axis tensors was
  active by update 2; aggregate gradient L2 was `0.002364` to `0.298679`.
  Target gradient-tensor count remained zero.
- Forbidden-input and G2-navigation open counts were zero. Held-out and sealed
  material remained unopened. The rejected checkpoint is closed and may not
  be loaded, resumed, calibrated, or otherwise reopened.

## Development result

- The full gate failed only `semantic_free_recall`: observed `0.821735`,
  required `>=0.85`, a shortfall of `0.028265`.
- The other semantic checks passed: balanced accuracy `0.885763`, OCCUPIED
  recall `0.903167`, rough OCCUPIED recall `0.794288`, and UNKNOWN recall
  `0.932386`.
- The joint navigation surrogate remained viable. Selection utility was
  `0.893548`, unequal-pair concordance `0.864447`, and zero-prefix rate
  `0.020050`; all registered family checks passed.
- Full-model advantage passed every registered wrong-RGB, shuffled-action,
  train-action-mean, and coordinate-matched-persistence conjunct. Bootstrap
  lower-95 deltas were respectively `0.080606`, `0.250026`, `0.009536`, and
  `0.044808`.

## What changed relative to V10

| Selection metric | V10 | V11 | Delta |
|---|---:|---:|---:|
| Balanced accuracy | `0.902738` | `0.885763` | `-0.016976` |
| FREE recall | `0.882500` | `0.821735` | `-0.060765` |
| OCCUPIED recall | `0.874255` | `0.903167` | `+0.028912` |
| Rough OCCUPIED recall | `0.734971` | `0.794288` | `+0.059317` |
| UNKNOWN recall | `0.951460` | `0.932386` | `-0.019074` |
| Navigation-surrogate utility | `0.906305` | `0.893548` | `-0.012757` |
| Pair concordance | `0.867378` | `0.864447` | `-0.002932` |

- The FREE loss is specifically an OCCUPIED overcall, not general collapse.
  Of the 30,419 V10-correct FREE cells lost by V11, 27,841 (`91.5%`) moved to
  OCCUPIED and only 2,578 moved to UNKNOWN. V11 predicted 83,942 true-FREE
  cells as OCCUPIED versus 56,101 in V10.
- The same trade appears on rough cells: V11 lost 13,266 correct FREE cells,
  with 12,279 of those additional errors going to OCCUPIED.
- Real obstacle recognition improved: V11 recovered 839 additional OCCUPIED
  cells, reducing OCCUPIED-to-FREE errors by 655 and OCCUPIED-to-UNKNOWN errors
  by 184. UNKNOWN also shifted toward OCCUPIED: 51,352 additional true-UNKNOWN
  cells were called OCCUPIED.

## Adjudication

- The fixed height-role split is trainable and preserves a genuine jointly
  used JEPA state: the predictor and every causal control passed, and both
  attention branches were active throughout training.
- The result directionally supports retaining the role-separated lift for one
  narrow successor: both aggregate and rough OCCUPIED recall improved while
  the only gate failure was a localized, interpretable FREE-to-OCCUPIED shift.
- It rejects this exact occupied-priority adapter under the inherited
  `S+P+U+R+O` objective. The adapter makes OCCUPIED the first decision and the
  inherited `O=0.5` auxiliary supplies an additional occupied-vs-rest term;
  the observed confusion shows that composition over-prioritized elevated
  evidence on floor-labelled cells.
- V11 did not earn physical calibration, so no physical metric or V10
  directional physical baseline may be claimed for it. The 2,016-tuple search,
  G2, navigation, held-out, and sealed stages remain closed.
- V11 is terminally closed. A successor must alter the evidence decision or
  its objective—not rerun this seed, tune supports/heights, extend training,
  or reopen the rejected checkpoint.
