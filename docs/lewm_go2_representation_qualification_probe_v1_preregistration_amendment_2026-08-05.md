# Amendment to the representation-qualification probe V1 preregistration

Date: 2026-08-05
Amends: `docs/lewm_go2_representation_qualification_probe_v1_preregistration_2026-08-05.md`
Attempt identity: `go2_representation_qualification_probe_v1_attempt_v1`

Status: **prospective amendment, made before the probe runner existed and before
any representation result was observed.** It supersedes the original document
for **architecture, extraction points, arm structure, and target grid only**.
The original remains in the repository unaltered; it is not rewritten, squashed,
or removed.

---

## 1. The error being corrected

The original preregistration §6 registered our internal arm as
`go2_jepa_geometric_encoder_v4_medium41_crossfam_lat192_img128` and described it
as "a 192-d global vector — it has no spatial tokens by construction."

**That was the wrong model.** It belongs to the older `go2_wallaware_learned`
lineage, not the current direct-BEV perception line. The two were conflated
because both use the number 192 — a coincidence between a CNN latent width and
a ViT embedding width.

The current direct-BEV line **is a Vision Transformer**, exactly as the SAINTS
progression document describes. From
`lewm/models/direct_egocentric_bev_state_jepa_v1.py` and the checkpoint tensors:

| component | value |
|---|---|
| input image size | `112 × 112` |
| patch size | `7` → `16 × 16 = 256` patch tokens |
| encoder | ViT, dim `192`, depth `6`, heads `6`, MLP ratio `4` |
| BEV decoder | learned `row_query` and `column_query`, `(64, 64)` each, cross-attention, `bev_dim 64` |
| BEV state | `64 × 64 × 64` |
| state head | `2 × 64 × 1 × 1` → two `64 × 64` signed-boundary fields |
| EMA targets | `target_encoder`, `target_bev_decoder`, `target_state_head` |

No probe had been fit and no representation score existed when this was found.

## 2. Corrected extraction points and their status

Three stages along the **actual deployed pathway**:

| stage | tensor | status |
|---|---|---|
| 1. backbone | final ViT patch tokens, `16 × 16 × 192` | **representation** |
| 2. BEV latent | learned-query BEV state, `64 × 64 × 64` | **representation — the primary preservation claim** |
| 3. operational output | signed-boundary fields, `64 × 64 × 2` | **model output, not a latent** |

Stage 3 is explicitly **not** a latent-preservation claim. It qualifies whether
preserved information is successfully converted into the operational spatial
output. The strongest "the representation preserves geometry" statement this
probe can make concerns stages 1 and 2, and stage 2 above all, because there the
latent and the target are already spatially aligned.

## 3. Native spatial contract — adopted wholesale

The provisional `4 m × 4 m` grid of the original document is **withdrawn**. The
target is rebuilt on the model's own registered contract:

| property | native value |
|---|---|
| BEV size | `64 × 64` |
| forward range | `[-0.95, +5.35]` m — **includes 0.95 m behind the robot** |
| left range | `[-3.15, +3.15]` m |
| metres per cell | `6.3 / 64 = 0.0984375` |
| classes | `UNKNOWN = 0`, `FREE = 1`, `OCCUPIED = 2` |

The original document's class indices (`free 0, occupied 1, unknown 2`) are
replaced by the model's own ordering. Note the model **already** treats
`UNKNOWN` as a first-class predicted category, which independently confirms the
correction made in the original §0.3.

Because the native grid extends behind the robot and to `±3.15` m laterally,
well outside the `±39.1615°` horizontal frustum, a substantially larger fraction
of cells will be `UNKNOWN` than under the withdrawn grid. That is a property of
the model's own contract, not a defect, and the observable-cell-conditioned
metrics of §6 exist precisely so it cannot mask the result.

Frames are resized to the native `112 × 112`. Frozen comparators keep their own
native input sizes and are evaluated against the identical target.

## 4. Registered deterministic field interpretation

From `lewm/models/direct_egocentric_bev_signed_boundary_distance_state_v1.py`,
channel order `(K, O)`:

- **K** — known/unknown signed boundary distance: **known positive, `UNKNOWN`
  negative**;
- **O** — free/occupied signed boundary distance: **`FREE` positive, `OCCUPIED`
  negative**, with `UNKNOWN` exactly zero and excluded from the O loss mask.

The registered deterministic conversion, used unchanged for stage 3:

```
if K < 0:            UNKNOWN
elif O >= 0:         FREE
else:                OCCUPIED
```

Stage 3's **primary** result is this deterministic conversion. A learned probe
on the two fields may be reported as a **secondary recoverability** figure only,
and may never replace the direct result: a failed operational output must not be
hidden behind a decoder.

## 5. Checkpoints — frozen, not swept

| role | checkpoint |
|---|---|
| **primary** | `signed_boundary_semantic_anchor_state_v3`, **update 400**, **online** branch |
| secondary, temporal | same family, **update 100** — tests whether geometry deteriorated during continued optimisation |
| secondary, labelled | EMA `target_*` branch of update 400 |

Update 400 is the checkpoint the SAINTS account describes and the one that
already failed the existing qualification gate (`gate.passed: False`,
`checkpoint_qualified: False`, every authority flag `False`). It is therefore the
scientifically relevant member, not a retrospectively flattering one.

**The other 13 direct-BEV checkpoints are not swept.** Registering the primary in
advance is what prevents post-hoc selection of whichever member tells the
cleanest story. The online branch is primary because it is the operational
inference path; the EMA result may not replace it after the numbers are seen.

Frozen comparators `dinov2` and `vjepa2_1` contribute spatial tokens on
identical frames and identical targets, unchanged from the original §6.

## 6. Probes and metrics per stage

Identical optimiser, schedule, and selection rule everywhere; selection only on
validation scenes; no per-encoder architecture search.

- **Stage 1, ViT tokens** — one fixed spatial decoder, preserving the `16 × 16`
  patch arrangement. Image-plane-to-BEV conversion is nontrivial, so a failure
  here licenses only the narrow statement *"the registered decoder could not
  recover transferable geometry from these tokens"*, **not** that the
  information is absent.
- **Stage 2, BEV latent** — a **per-cell linear** head plus one fixed shallow
  nonlinear alternative. Latent and target are already spatially aligned, so this
  is the cleanest preservation test.
- **Stage 3, fields** — the §4 deterministic conversion as primary; optional
  learned probe as clearly-labelled secondary.

**Metrics, all reported:** occupied precision, recall and IoU; free precision,
recall and IoU; unknown recall and IoU; macro/balanced accuracy; per-scene
values; whole-scene cluster intervals. Additionally **occupied and free metrics
conditioned on observable cells only**, so that performance on the large
`UNKNOWN` mask cannot obscure whether geometry inside the visible region is
retained.

Legacy direct-BEV thresholds remain **context only and gate nothing**, unchanged
from the original §7.

## 7. Target validation — beyond the internal oracle

The reported oracle result (balanced accuracy `1.0`, occupied IoU `1.0`) shows
only that the target and the oracle agree internally. It does not establish that
the coordinate transforms, frustum, or occlusion correspond to the rendered
observation. Before the result is reported, the following must pass:

- synthetic fixtures with known geometry: **walls only, obstacles only, both,
  and neither**;
- rotation and translation equivariance tests;
- visual overlays of the derived observable grid against sampled camera frames;
- comparison against simulator depth or ray-cast evidence where available.

**Already verified at amendment time:** the combined `walls`-plus-`obstacles`
occluder set was confirmed against the real source — the earlier `raise`/`return`
was fully replaced, not left ahead of the new code — and all four fixture cases
behave correctly, with a real `open_obstacle_field` scene (0 walls, 5 obstacles)
returning 5 footprints. The remaining checks in this list are outstanding.

## 8. Unchanged from the original

The semantic family remains **formally deferred** on the measured label
inadequacy of Amendment 1 — landmarks visible in at most 9 of 32 scenes,
saturating with range. The scope of this run is therefore:

> **This run qualifies transferable observable geometry only. Semantic
> preservation remains untested and requires a suitably constructed corpus.**

Also unchanged: the three-way scene-cluster split, per-scene aggregation, the
controls (shuffled latent, class frequency, analytical oracle, RGB baseline),
the prohibition on giving any probe the world pose, map, successor, or
privileged state, and the §10 authorization limits.

## 9. Stopping rule

Once the probe identifies the **earliest failing stage**, stop expanding the
assay and implement the corresponding representation-training intervention. This
exists to prevent the probe becoming another measurement programme detached from
improving the encoder. The 304-scene matched-branch programme is cancelled and
is not restarted.
