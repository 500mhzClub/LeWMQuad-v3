# Frozen dense-representation screen: pretrained tokens vs our task-trained ViT

Date: 2026-08-06
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** Every encoder here is frozen, so
**no arm in this document is a JEPA result.** No manifest or authorization status
is inherited. `probability_calibration`, `evaluation`, `untouched` and sealed
data were never opened.

Artifacts: `.generated/dev/DEVELOPMENT_ONLY_frozen_dense_representation_screen_v1/`
Code: `scripts/run_dev_frozen_dense_representation_screen_v1.py`,
`scripts/dev_frozen_dense_representation_encoders_v1.py`

---

## 1. Question and answer

> Do strong frozen pretrained dense representations already retain more
> transferable navigation geometry than our task-trained ViT tokens?

**Yes, decisively, and both external arms beat the incumbent on the primary
metric.** On scene-disjoint `checkpoint_selection`, observable occupied IoU:

`V-JEPA 2.1 ViT-L 0.5103` > `DINOv2 ViT-L 0.4709` > `project ViT 0.3724`

The ordering is the same on occupied precision, on the macro average over the
eight selection scenes, on all eight families individually, and on
`open_obstacle_field`. Every arm clears its own shuffled-token control by a wide
margin, and the ordering of the margins matches the ordering of the results.

This lands on **interpretation path 3** (V-JEPA 2.1 succeeds best; DINOv2
retained as the required comparator) with a **path-4 qualifier that survives
intact**: pretraining raises `open_obstacle_field` from `0.117` to `0.219` but
does not rescue it. Isolated-obstacle geometry remains the weakest family for
every representation tested, at roughly two-fifths of each arm's own aggregate.

The correct statement of the finding is:

> **Strong pretrained dense representations contain substantially more
> transferable geometry than the current task-trained encoder.**

This is *not* a claim that pretraining caused the gain. Capacity (2.7M vs 304M),
input resolution, token density and the pretraining corpus and objective all
differ between arm A and arms B/C and are fully confounded; this screen was
designed to find the strongest achievable frozen baseline, not to attribute the
difference to any one of them.

What does follow is that the geometry is recoverable from RGB at all: a frozen
public encoder plus a 25M-parameter probe reaches `0.51` occupied IoU at `0.65`
precision on scenes it has never seen. So the limitation in our current line is
not that a single RGB frame cannot carry the required geometry.

---

## 2. Arms and checkpoint identities

| | A — project ViT | B — DINOv2 | C — V-JEPA 2.1 |
|---|---|---|---|
| model id | `project_direct_bev_vit_n320_update_400` | `dinov2_vitl14` (LVD-142M) | `vjepa2_1_vit_large_384` |
| source | in-repo `direct_egocentric_bev_state_jepa_v1` | `facebookresearch/dinov2` @ `7764ea0f912e…` | `facebookresearch/vjepa2` @ `204698b45b37…` |
| release | local `update_400.pt` | official torch.hub, no registers | official release, `ema_encoder` key |
| checkpoint file | `update_400.pt` | `dinov2_vitl14_pretrain.pth` | `vjepa2_1_vitl_dist_vitG_384.pt` |
| checkpoint sha256 | `81682a1a25fb7706…` | `d5383ea8f4877b24…` | `7ea9b7cb4a75d106…` |
| source URL | n/a (local run) | `dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth` | `dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` |
| parameters | 2,747,520 | 304,368,640 | 304,680,960 |
| output layer | `forward_tokens` final block, CLS discarded | `forward_features()["x_norm_patchtokens"]` | encoder final block, `norms_block[-1]`, `return_hierarchical=False` |
| inference dtype | float32 | float32 | float32 |
| peak VRAM | 0.126 GiB | 1.388 GiB | 1.994 GiB |
| extraction wall | 8.9 s | 231.5 s | 1013.2 s |

Arms B and C are within `0.1%` of each other on parameter count (304.37M vs
304.68M), so nothing below is a model-capacity effect.

Both external checkpoints load `strict=True`. Arm D (V-JEPA 2) was **not run**:
it was not needed to answer the question, and adding it would have delayed A–C.
The local `vjepa2_1_vitb_dist_vitG_384.pt` was also not used — the ViT-L was
fetched instead so that B and C are size-matched.

### Preprocessing and token shapes

The render is `224×168`. No arm pads, so **no arm has a single pure-padding
token** and the padding-mask path is a documented no-op.

| | input | resample | patch | token grid (H×W) | tokens | dim | cached tensor |
|---|---|---|---|---|---|---|---|
| A | `112×112` | PIL bilinear, **anisotropic** — its trained contract | 7 | 16×16 | 256 | 192 | `[4757, 256, 192]` |
| B | `168×224` | none: native render pixels (14 divides both) | 14 | 12×16 | 192 | 1024 | `[4757, 192, 1024]` |
| C | `384×512` | PIL bicubic, isotropic ×16/7, official short-side 384, **centre crop omitted** | 16 | 24×32 | 768 | 1024 | `[4757, 768, 1024]` |

All arms use ImageNet normalisation `(0.485,0.456,0.406)/(0.229,0.224,0.225)`.
Arm A keeps its anisotropic squash because its weights were trained under it; the
squash is a property of the incumbent, not a choice made by this screen.

Arm C uses V-JEPA 2.1's **official single-frame image tokenizer**
(`img_temporal_dim_size=1` routes a `(B,3,1,H,W)` input through `patch_embed_img`,
tubelet 1, with the image modality embedding). **No frame was duplicated to
manufacture a clip and there is no additional temporal context** — exactly the
labelled current frame, same as A and B.

### Parity check (per encoder, on two fixed corpus rows)

| | input tensor | grid | real-content tokens | repeat max|Δ| | distinct-observation max|Δ| |
|---|---|---|---|---|---|
| A | `[2,3,112,112]` f32 | 16×16 | 256 / 256 | `0.0` | `10.13` |
| B | `[2,3,168,224]` f32 | 12×16 | 192 / 192 | `0.0` | `20.39` |
| C | `[2,3,384,512]` f32 | 24×32 | 768 / 768 | `0.0` | `32.41` |

Extraction is bit-deterministic on repeat, and different observations give
plainly different features. Denormalised input images are written to
`parity/<arm>/input_row{0,_mid}.png`.

---

## 3. Data and custody

`development_raw_supervision_v1`, manifest `74ae5799919ff4d9…`.
Roles partitioned **before any cap**; empty roles are fatal; no fallback split.

| | pairs | scenes |
|---|---:|---:|
| `train` | 4,262 | 72 |
| `checkpoint_selection` | 495 | 8 |
| scene overlap | — | **0** |

Ordered-pair hash `cde10e28a1f3bd07…` (identical for all three arms; the cache
receipts refuse to reuse a cache built on any other ordering).

Targets are `raster_labels.u1` **used verbatim** — no occupancy target was
reconstructed and the V3 matched-branch corpus was not substituted. The native
support evidence `ground_support_in_frustum.u1` and
`ground_support_clear_to_target.u1` (8,701 rows each, `128×128×5`, across 80
shards) was checked present and shape-correct as the provenance of those labels,
not re-reduced. Class definitions are the corpus native `UNKNOWN=0, FREE=1,
OCCUPIED=2`.

**Denominators, `checkpoint_selection`:** 495 frames, 2,027,520 cells, of which
265,338 observable and 14,758 occupied. Occupied is `5.56%` of observable cells
and `0.73%` of all cells. All 8 selection scenes have nonzero occupied support.

This screen is **geometry-only**. The corpus does not support a defensible
task-semantic result, and none is claimed.

---

## 4. Probe

One probe for every arm, from the existing Stage-1 dense-token-to-BEV family:

```
tokens (N, T, D) -> reshape to native grid -> 1x1 conv D->16 -> GELU
   -> deterministic bilinear resample to the common 24x32 grid
   -> flatten 12288 -> linear 1024 -> GELU -> linear -> 64x64x3
```

The common grid is the **largest** arm grid, so the adapter never discards any
arm's spatial detail; smaller grids are deterministically upsampled, which adds
no information to anybody. The only per-arm component is the 1×1 input
projection (3,088 params for A, 16,400 for B and C) against an **identical
25,179,136-parameter shared head**. Same optimiser (AdamW, lr `1e-3`, wd `1e-4`,
clip `1.0`), same 30 epochs, batch 32, same class-weighted cross entropy with
weights from train-role counts, same seed `2026080611`, same checkpoint-selection
rule (best `checkpoint_selection` observable occupied IoU over epochs). **No
encoder-specific architecture or hyperparameter tuning, and no post-hoc layer,
preprocessing or schedule selection.**

Probes were fit on the `train` role only and selected on `checkpoint_selection`
only. Frozen features are cached once per arm with a receipt recording
encoder/checkpoint hash, ordered-pair hash, preprocessing identity and hash,
token shape, dtype and cache hash.

---

## 5. Controls

| control | occupied IoU (selection) | reading |
|---|---:|---|
| class-prior mean map, **no image input** | `0.0000` | fixed per-cell priors predict no occupied cell at all |
| shuffled tokens, arm A | `0.1262` | fixed positional/frustum/decoder priors |
| shuffled tokens, arm B | `0.1249` | same floor |
| shuffled tokens, arm C | `0.1218` | same floor |

The shuffled control deranges complete feature tensors between observations while
preserving token positions, using **separate fixed-point-free derangements within
`train` and within `checkpoint_selection`**, so no row ever receives another
role's features. It is trained and selected under the identical schedule
(`scripts/recompute_dev_frozen_dense_screen_shuffled_controls_v2.py`, seed
`2026080631`; encoder extraction and the main probes were not rerun).

All three arms land on essentially the same `≈0.12` prior floor, so the shuffled
margin is a clean read of image-conditioned content. The superseded cross-role
derangement gave `0.1212 / 0.1234 / 0.1251` — within `0.005` of the within-role
values, so no conclusion in this document depended on it.

**Free IoU must not be read without the all-free baseline.** Predicting FREE at
every cell gives observable free IoU `0.9444` on the selection role. Arm A scores
`0.9251` — *below* the trivial baseline; arm B `0.9451` and arm C `0.9571` are
barely above it. Observable free IoU carries almost no signal here and was not
used to rank anything.

---

## 6. Comparative result — `checkpoint_selection`, scene-disjoint

| metric | A project ViT | B DINOv2 ViT-L | C V-JEPA 2.1 ViT-L |
|---|---:|---:|---:|
| **observable occupied IoU** | 0.3724 | 0.4709 | **0.5103** |
| **observable occupied precision** | 0.4644 | 0.6187 | **0.6471** |
| **observable occupied recall** | 0.6527 | 0.6634 | **0.7071** |
| macro occupied IoU over 8 scenes | 0.4086 | 0.4877 | **0.5261** |
| macro occupied precision over 8 scenes | 0.4960 | 0.6235 | **0.6526** |
| tolerant ±1 cell (P / R) | 0.4656 / 0.8599 | **0.6477** / 0.8805 | 0.6450 / **0.9118** |
| tolerant ±2 cells (P / R) | 0.6467 / 0.9126 | **0.7904** / 0.9404 | 0.7874 / **0.9550** |
| UNKNOWN IoU | 0.9679 | **0.9786** | 0.9782 |
| observable free IoU | 0.9251 | 0.9451 | **0.9571** |
| *(all-free baseline)* | *0.9444* | *0.9444* | *0.9444* |
| observable balanced accuracy | 0.7941 | 0.8087 | **0.8364** |
| **shuffled-token margin (occ IoU)** | +0.2462 | +0.3459 | **+0.3885** |
| train occupied IoU | 0.7456 | 0.9108 | 0.9241 |
| **fit-to-selection gap** | **0.3732** | 0.4399 | 0.4138 |
| selected epoch | 6 | 9 | 6 |
| predicted occupied fraction (of all cells) | 0.0268 | 0.0166 | 0.0177 |
| *(target occupied fraction)* | *0.0073* | *0.0073* | *0.0073* |

Every arm shows a large train-to-selection gap — the 25M-parameter decoder does
memorise the fit scenes. That is why the ranking is taken **only** from the
scene-disjoint selection role, and why the shuffled and prior controls are
reported next to it.

Arm A over-predicts occupied by `3.7×` the target rate at `0.46` precision; arm C
over-predicts by `2.4×` at `0.65` precision while also finding more of the true
occupied cells. The pretrained arms are not simply trading recall for precision.

---

## 7. Per-family — `checkpoint_selection`, one scene per family

Occupied IoU / precision / recall:

| family | occ cells | A project ViT | B DINOv2 | C V-JEPA 2.1 |
|---|---:|---|---|---|
| `small_enclosed_maze` | 711 | 0.729 / 0.817 / 0.872 | 0.784 / 0.924 / 0.838 | **0.804** / 0.925 / 0.861 |
| `local_composite_motifs` | 2071 | 0.500 / 0.608 / 0.738 | 0.573 / 0.691 / 0.769 | **0.607** / 0.726 / 0.788 |
| `large_enclosed_maze` | 1891 | 0.477 / 0.573 / 0.740 | 0.547 / 0.680 / 0.736 | **0.606** / 0.729 / 0.782 |
| `medium_enclosed_maze` | 2290 | 0.453 / 0.535 / 0.749 | 0.472 / 0.622 / 0.661 | **0.537** / 0.707 / 0.692 |
| `rough_local_dynamics` | 2519 | 0.278 / 0.409 / 0.464 | 0.478 / 0.636 / 0.657 | **0.529** / 0.685 / 0.700 |
| `visual_sensor_stress` | 2101 | 0.453 / 0.563 / 0.698 | 0.511 / 0.681 / 0.671 | **0.522** / 0.664 / 0.709 |
| `loop_alias_stress` | 2653 | 0.262 / 0.329 / 0.561 | 0.346 / 0.507 / 0.522 | **0.384** / 0.515 / 0.602 |
| **`open_obstacle_field`** | **522** | **0.117** / 0.135 / 0.473 | **0.192** / 0.247 / 0.466 | **0.219** / 0.271 / 0.533 |

**Arm C wins all eight families; arm B beats arm A in all eight.** The ordering is
not carried by one scene.

`open_obstacle_field` is the weakest family for all three arms by a wide margin —
`0.219` against arm C's own `0.510` aggregate, at `0.27` precision. Note the
denominator: it has 522 occupied cells, `4–5×` fewer than the maze families, and
its observable space is `99.5%` free by the all-free baseline. Isolated obstacles
are both rare and small in this target, and pretraining narrows but does not
close the gap.

`loop_alias_stress` is the second-weakest for all three, consistent with its
known aliasing character rather than with anything about the representations.

---

## 8. Overlays

`overlays_<arm>.png`, best / median / worst selection frame by per-frame
observable occupied IoU, RGB alongside target and prediction.

| arm | worst | median | best |
|---|---|---|---|
| A | `open_obstacle_field` 0.000 | `visual_sensor_stress` 0.171 | `rough_local_dynamics` 0.543 |
| B | `open_obstacle_field` 0.000 | `local_composite_motifs` 0.286 | `visual_sensor_stress` 0.733 |
| C | `rough_local_dynamics` 0.000 | `loop_alias_stress` 0.327 | `small_enclosed_maze` 0.727 |

Arm C's worst frame is a single thin distant post over open ground: the free
wedge is recovered almost exactly and the few occupied cells are missed by a
couple of cells. Per-frame IoU is brittle where occupied support is a handful of
cells, so the worst panels understate the qualitative agreement; the aggregate
and tolerant metrics are the ones to read.

---

## 9. What this does and does not establish

**Selected frozen spatial baseline: V-JEPA 2.1 ViT-L/16-384 dense image tokens**
(`vjepa2_1_vitl_dist_vitG_384.pt`, `7ea9b7cb4a75d106…`, final-block
`norms_block[-1]`, 24×32×1024 at `384×512`). DINOv2 ViT-L/14 is retained as the
required comparator and is a close second at a quarter of the extraction cost.

**Confound that must not be dropped.** Arm C sees `2.6×` the source pixels and
`4×` the tokens of arm B, because each external encoder was run at its own
official nominal inference scale rather than reduced to a common one. That was
the instructed design and it answers "what is the strongest achievable frozen
baseline", but it means **the C-over-B gap cannot be attributed to the video
pretraining or to the V-JEPA 2.1 dense-loss changes rather than to resolution and
token count.** Parameter count is matched; input scale is not. A resolution- and
token-matched comparison would be needed to make a family-level claim, and this
screen does not make one.

**Not comparable to the earlier native-contract number.** The corrected
native-contract probe scored this same arm-A encoder at `0.2832` observable
occupied IoU, but on a random 39/13 scene holdout with the 16×16 `TokenToBev`
decoder. Here it scores `0.3724` on the designated 72/8 role split under the
shared 24×32 probe. Different split, different probe — the two numbers are not
interchangeable, and only the within-this-table comparisons are meaningful.

**No registered perception threshold was applied.** No existing gate matches this
target, mask, aggregation and metric definition exactly, so the result is
reported comparatively and no new threshold was invented.

**None of this is a JEPA result.** All three encoders are frozen. Arm C is a
strong frozen representation with a supervised probe on top; it is not a world
model, it predicts nothing, and it must not be described as one.

---

## 10. Implication for the next encoder-moving JEPA

1. **Initialise from V-JEPA 2.1 ViT-L dense tokens**, with DINOv2 ViT-L kept as
   the standing comparator. Both are `304M` parameters against the incumbent's
   `2.7M`; the next JEPA is a fine-tuning problem on a large pretrained encoder,
   not a from-scratch one, and the corpus (4,262 train pairs) is far too small to
   move `304M` parameters freely.
2. **Stop treating the spatial gate as possibly unreachable from RGB.** A frozen
   public encoder plus a small probe reaches `0.51` occupied IoU at `0.65`
   precision on unseen scenes. Whatever the remaining problem is, it is not that
   the geometry is absent from a single RGB frame.
3. **The `112×112` input is now a measured liability, not a suspicion.** Arm A is
   the only arm whose observable free IoU falls below the all-free baseline, and
   it trails on every occupied metric. Raise the input resolution before
   attributing any further failure to the objective.
4. **Carry the WP-E requirement forward unchanged.** WP-E's open question is
   whether an encoder can be moved so its representation becomes *more*
   action-discriminative rather than merely more predictable. This screen changes
   the starting representation; it says nothing about that question, and the
   correct-versus-shuffled action margin must still be measured on any successor.
   `0.5103` is the **frozen single-frame V-JEPA 2.1 development reference** — the
   matched comparator a successor's spatial readout is reported against. It is
   **not a floor, not a qualification threshold and not a gate**; it is one
   development measurement under this probe, split and preprocessing.
5. **`open_obstacle_field` needs a targeted intervention, not more pretraining.**
   `0.219` at `0.27` precision is the weakest result for the best available
   frozen representation. The next lever there is higher effective resolution on
   small distant obstacles or a training-only geometric teacher — not another
   unmasked all-token prediction objective, which was already rejected in WP-E.

## 11. Reproduction

```
~/TinyQuadJEPA/bin/python scripts/run_dev_frozen_dense_representation_screen_v1.py \
    --arms project_vit,dinov2,vjepa21
```

Arms may be run in separate invocations; the runner merges into an existing
`result.json` only when the ordered-corpus hash is identical, and refuses
otherwise. Feature caches (9.8 GB) and external checkpoints are deliberately not
tracked in Git; both are reproducible from the recorded URLs and hashes.
Total wall time: extraction 1,254 s, probes and controls ~1,100 s.
