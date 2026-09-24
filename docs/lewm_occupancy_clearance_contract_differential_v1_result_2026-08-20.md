# Occupancy–Clearance Contract Differential V1

Date: 20 August 2026  
Source commit: `94711820f212b583196c5abe13820c9852cfe46c`  
Experiment: `OCCUPANCY_CLEARANCE_CONTRACT_DIFFERENTIAL_V1`  
Preserved terminal: `TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO`

## Outcome

The bounded differential classification is:

`STRUCTURED_SAFETY_LABEL_OR_ALIGNMENT_DEFECT`

The defect is specifically a clearance-consumer semantic and temporal mismatch, not an occupancy raster alignment defect:

- the Stage-A prediction was an **instantaneous H3 configuration-clearance proxy**, computed as nearest predicted occupied-cell range minus a `0.47 m` footprint radius;
- the frozen target was a **cumulative tick-level minimum centre-point clearance** through H3, with no footprint subtraction.

This finding was determined from the producer and consumer contracts before using held-out safety performance. The diagnostic correction therefore removed the footprint subtraction and used the cumulative minimum of the H1–H3 point-range estimates. It did not change a route label, branch, latent, model, threshold, or checkpoint.

The one authorised corrected Stage-A evaluation still failed:

`TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO`

Stage B remains prohibited and was not trained.

## Frozen panels

The identities were written before the probe was loaded.

| Panel | Rows | Selection |
|---|---:|---|
| Counterfactual reference | 16 | Two registered, defined H3 rows from each of eight families, ordered by structural SHA-256 |
| Route intent | 16 | Four rows from each maze family; two safe and two unsafe where available, ordered by structural SHA-256 |

Panel identity digest: `854398536aecbdeefa575eb03e4fff67564508dde88d52b6313d015841556001`.

The reference panel binds registered occupancy result `09dc413d9ce30c2cb19c99e93eeaad410983a7f53575387bc6694f3844a070d6`. The route panel binds target index `df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874`.

## Exact reference reproduction

All 16 reference rows reproduced exactly under the current Stage-A consumer:

- observable/ignored counts matched;
- occupied support, prediction, intersection, and union counts matched;
- row-level occupied IoUs matched;
- defined/undefined status matched;
- selected-panel pooled occupied IoU was `0.43609022556390975`.

The frozen records do not persist logit or probability arrays. Consequently, direct numeric comparison of those arrays is unavailable; the registered class-map sufficient statistics and every persisted row-level IoU reproduce exactly. There was no divergent reference field.

## Contract comparison

| Field | Counterfactual reference | Route intent | Verdict |
|---|---|---|---|
| Target latent | FP16 `[768,1024]` | FP16 `[768,1024]` | Same |
| Encoder checkpoint | `7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6` | Same | Same |
| Crop and resize | RGB rows `28:196`; bicubic `512×384` | Same encoder/preprocess implementation | Same |
| Normalisation | ImageNet input; FP16 reload as FP32; affine-free token LayerNorm over 1024 features | Same | Same |
| Token order | Row-major `24×32` patch grid | Same encoder class | Same |
| Probe rule | Unchanged three-class logits; absolute `argmax` | Same | Same |
| Label grid | uint8 `[64,64]` | uint8 `[64,64]` | Same |
| Coordinate frame | Per-H3 endpoint yaw-body frame | Same | Same |
| Spatial axes | Row = forward; column = left | Same | Same |
| Camera | 78.323° horizontal FOV, 62.8370386364° vertical FOV, `0.05 m` near plane, body mount `[0.326,0,0.043] m` | Same composition | Same |
| Camera attitude | Endpoint quaternion retains roll/pitch in the yaw-aligned body frame | Same | Same |
| Pose | H3 endpoint base pose, not branch-start pose | H3 endpoint base pose, not branch-start pose | Same |
| Raster source | Frozen scene walls, obstacles, landmarks; distractors excluded | Same | Same |
| Classes | unknown `0`, free `1`, occupied `2` | Same | Same |
| Mask/NA | `truth != unknown` after rasterisation; zero union yields NA | Same | Same |
| Timestamp binding | Paired H3 frame receipt and H3 pose receipt | H3 RGB SHA and pose from the same replay H3 record | Same |

The route target index is less self-describing than the reference index: it omits preprocessing digest, token order, normalisation, and render-contract fields. Narrow source inspection nevertheless confirms it invokes the same `VJepa21CroppedV03Arm`. This is a provenance weakness, not a numerical divergence in this diagnostic.

## Route label comparison

The existing route occupancy arrays and an in-memory rerun of the exact qualified reference producer were byte-identical for all 16 rows at H1, H2, and H3.

- current/reference-producer agreement: `16/16` rows, all horizons;
- selected H3 pooled occupied IoU: `0.0`;
- H3 occupied intersections: zero in every selected route row;
- no transpose or flip diagnostic was run, because the precondition—structural disagreement between A and B—was absent.

Thus there is no evidence for a pose-frame, timestamp, axis, transpose, flip, crop, scale, class, mask, or raster-producer defect. The route rows genuinely remain near zero under the correctly reproduced frozen occupancy interface.

## Clearance audit

All compared quantities are in metres. There is no normalisation, denormalisation, or clipping. The old consumer used `10 m` as a sentinel when no occupied class was predicted. Across the 16 route rows:

- frozen cumulative clearance labels ranged from `0.12218` to `0.33095 m`;
- old Stage-A predictions ranged from `1.48576` to `10.0 m`;
- the approximately `4.08 m` held-out MAE was dominated by large predicted ranges and the `10 m` sentinel, not a metres-versus-centimetres conversion.

Four geometry hand-checks showed the distinction clearly:

| Branch | Endpoint scene clearance | Stored path-min clearance | True-raster nearest occupied range | Predicted occupied range | Old footprint-subtracted value |
|---|---:|---:|---:|---:|---:|
| `purpose-6:00` | 0.25675 | 0.25675 | 0.69642 | 2.73222 | 2.26222 |
| `purpose-7:07` | 0.24397 | 0.18960 | 0.69642 | 1.95576 | 1.48576 |
| `purpose-10:09` | 0.37604 | 0.24054 | 0.55227 | 10.0 | 10.0 |
| `purpose-2:11` | 0.19374 | 0.17706 | 0.69642 | 10.0 | 10.0 |

The comparison also shows why an endpoint quantity cannot be treated as a path-minimum target: for three of four rows, the stored minimum occurred before H3.

## Corrected Stage-A evaluation

Only the proven clearance consumer defect was corrected. The occupancy arrays, route labels, stuck checkpoint, stuck temperature `7.112425804138184`, and stuck threshold `0.11016567796468736` were unchanged. No label regeneration was necessary because the raster labels already matched the qualified producer exactly.

| Metric | Corrected value | Gate |
|---|---:|---:|
| H3 occupied IoU | 0.0011524 | ≥ 0.35 |
| H3 clearance Spearman | 0.21805 | ≥ 0.60 |
| H3 clearance MAE | 2.14991 m | diagnostic |
| H3 low-clearance recall | NA (zero positive held-out rows) | ≥ 0.90 |
| Stuck AUC | 0.79502 | ≥ 0.85 |
| Stuck recall / FNR | 1.0 / 0.0 | ≥ 0.90 / ≤ 0.10 |
| Aggregate unsafe recall / FNR | 0.96552 / 0.03448 | ≥ 0.95 / ≤ 0.05 |
| Safe-candidate retention | 0.07895 | ≥ 0.40 |
| States retaining a safe candidate | 3/8 | ≥ 6/8 |
| States with only unsafe admissions | 2 | 0 |
| Selected unsafe rate | 0.40 | 0 |
| Selected distance progress | 0.02638 m | ≥ 80% oracle |
| Normalised safe-progress regret | 0.66667 | ≤ 0.20 |
| Best-safe top-3 | 0.125 | ≥ 0.75 |

The correction improved H3 clearance MAE from approximately `4.08144 m` to `2.14991 m` and Spearman from `0.11192` to `0.21805`, but it did not approach qualification. More importantly, the unchanged and correctly aligned H3 occupancy result remains `0.0011524` over the full held-out Stage-A set.

## Interpretation and stop

The differential found a real clearance-consumer defect, so the prescribed diagnostic classification is `STRUCTURED_SAFETY_LABEL_OR_ALIGNMENT_DEFECT`. The corrected evaluation does not rescue the scientific interface: `TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO` remains the corrected Stage-A outcome.

Accordingly:

- a corrected Stage-A evaluation was justified and has been completed once;
- Stage B is not justified and was not trained;
- no further occupancy/clearance threshold or transform search is justified;
- no model was trained;
- no simulator was run;
- no RGB was rendered;
- no latent was encoded;
- no predictor checkpoint was opened;
- no branch, latent, or label was overwritten;
- nothing remains running.

Diagnostic runtime was approximately 11.5 seconds on CUDA. New generated JSON storage was approximately 239 kB; large frozen arrays were read in place.

Generated result SHA-256: `3934640b22efd60f1b3ade07841323175d867a466ffe1d2cc8e06242cf5d7465`  
Generated identity-manifest file SHA-256: `31636ecff0467e90cad0ddc905b9b3eed7a5f7ed641d1c7993e0f6943c8430a1`
