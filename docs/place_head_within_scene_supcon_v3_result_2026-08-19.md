# PLACE_HEAD_WITHIN_SCENE_SUPCON_V3

One seed (`2026081902`) was trained using the frozen ViT-L token cache and
unchanged 52-scene/18-scene split. Sampling used 104 fixed batches per epoch,
with four scenes, four nodes per scene, and two views per node. No predictor or
simulator was opened.

V2 attribution on registered triplets showed 86.86% fit and 85.76% held-out
margin satisfaction, with mean positive cosine 0.7384 and registered-negative
cosine 0.1761. Yet V2 fit retrieval was only top-1 0.0325 (MRR 0.2908),
supporting `EASY_OR_MISALIGNED_TRIPLET_OBJECTIVE`.

V3 supervised-contrastive loss decreased from 1.2532 to 0.2321. Final
checkpoint SHA-256: `982f9e2939077ec629c4acd89f4253e1f29d765d0d387a557a4d9be5fe93ce92`.

| condition | top-1 | top-3 | MRR | median rank |
|---|---:|---:|---:|---:|
| V3 SupCon | 0.02199 | 0.32870 | 0.24135 | 5 |
| V2 triplet | 0.01157 | 0.26389 | 0.20264 | 7 |
| mean-pooled ViT-L | 0.01042 | 0.30671 | 0.21479 | 6 |

V3 improved modestly over V2 but remained far below the frozen development
gate (top-1 ≥0.50 and ≥0.15 improvement over mean pooling), with zero top-1 in
several held-out families. Final classification:
`PLACE_HEAD_WITHIN_SCENE_SUPCON_NO_SIGNAL`.
