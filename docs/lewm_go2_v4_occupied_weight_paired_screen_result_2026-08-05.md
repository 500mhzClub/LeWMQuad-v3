# Paired A/B development screen: O-field occupied weighting

Date: 2026-08-05
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** Neither arm passed the reviewed
execution path. No manifest or authorization status is inherited. Evaluation-role
scenes were never loaded.

Artifacts: `.generated/dev/DEVELOPMENT_ONLY_v4_occupied_weight_screen/arm_{A,B}/`
Diagnostics: `.generated/dev/go2_representation_qualification_probe_v1/attempt_v3_arm{A,B}/`

---

## Design

Paired A/B under the current committed runtime, both arms from the same N320
initial state, identical seed, 16,000-presentation ordered schedule
(`9f17253778affffb…`), loader, corpus, split, optimiser, K loss, JEPA objective
and checkpoint-selection rule.

| arm | O-field reduction |
|---|---|
| A | inherited class-macro mean, implying ~28:1 per-cell occupied:free |
| B | committed v4 fixed 5:1 over known cells, `w_free 0.8778`, `w_occ 4.3890` |

Initial parameter and buffer state hashed identically for both arms
(`50faee942bb76c34…`); v4 adds no parameter or buffer.

**Causal credibility.** Arm A's final weights reproduce the official v3
`update_400` checkpoint **bit-for-bit**: `5387c155ae0f5f19…`, with the same
registered terminal. The development launcher therefore reproduces the official
training run exactly, and arm B's divergence (`84b9e4a4b1d07394…`) is
attributable to the O-field weighting alone.

## Denominators

2,048 frames (1,585 fit / 463 validation), 4 validation scenes, all 4
contributing to the occupied aggregate. Over all 64×64 cells:
`unknown 0.8590`, `free 0.1334`, `occupied 0.0076`. Occupied is `0.0540` of
observable cells.

## Result

Stage 2, BEV state, held out:

| probe | arm | occ IoU | occ precision | free IoU | unknown IoU |
|---|---|---:|---:|---:|---:|
| linear | A | **0.2292** | **0.2486** | **0.8558** | 0.8553 |
| linear | B | 0.2142 | 0.2329 | 0.8455 | 0.8546 |
| shallow | A | **0.2394** | **0.2560** | **0.8556** | 0.8663 |
| shallow | B | 0.2227 | 0.2373 | 0.8409 | 0.8732 |

Token → BEV gap:

| arm | tokens occ IoU | BEV occ IoU | IoU gap | precision gap |
|---|---:|---:|---:|---:|
| A | 0.3155 | 0.2394 | +0.0760 | +0.3726 |
| B | 0.2683 | 0.2227 | +0.0456 | **+0.3984** |

Stage 3, head:

| arm | occ IoU | occ precision | occ recall | occ predicted fraction |
|---|---:|---:|---:|---:|
| A | 0.2122 | **0.4406** | 0.2904 | 0.0360 |
| B | 0.2225 | 0.3112 | 0.4384 | 0.0667 |

## Reading

- **BEV occupied precision and IoU worsened** under the 5:1 weighting, under
  both probes. Free-space IoU also worsened at both stages.
- **The token→BEV IoU gap narrowed only because the ViT tokens degraded**
  (`0.3155 → 0.2683`), not because the BEV state improved. The precision gap
  widened.
- **Stage 3's small IoU gain is a calibration shift**: precision `0.441 → 0.311`,
  recall `0.290 → 0.438`, occupied predictions nearly doubled. Tolerant metrics
  agree — at ±2 cells recall rises `0.551 → 0.729` while precision falls
  `0.302 → 0.262`. It does not rescue the representation.
- Per-scene, B is better on 3 of 4 validation scenes and worse on
  `local_composite_motifs` (`0.868 → 0.831`); a mixed small-sample picture that
  does not offset the aggregate regression.
- **Open-obstacle-field scenes are absent from this validation split**, so that
  criterion is unevaluated here.
- Both arms terminate on the same pre-existing anti-collapse gate, so neither
  checkpoint is qualified regardless.

The prediction motivating the intervention was **wrong**. Reducing occupied
weight from ~28:1 to 5:1 was expected to reduce over-prediction and sharpen
precision; instead the model predicted *more* occupied, less precisely. No
supported explanation is offered.

## Conclusion, stated narrowly

**This particular learned-query BEV decoder and training line are rejected as the
primary JEPA latent.** ViT patch tokens become the primary representation; the
BEV branch is retained as an auxiliary geometric output or planning readout.

This is **not** a claim that BEV representations are impossible in general, nor
that additive row/column queries mathematically cannot represent isolated
obstacles. That remains an untested hypothesis and was never measured.

## Next

One token-primary successor: preserve the 16×16×192 patch-token grid and adapt
the action-conditioned JEPA predictor to operate directly on that sequence,
without routing the predictive state through the 64×64×64 BEV bottleneck. The
BEV branch is retained with its baseline objective unchanged. No further BEV
weighting run.
