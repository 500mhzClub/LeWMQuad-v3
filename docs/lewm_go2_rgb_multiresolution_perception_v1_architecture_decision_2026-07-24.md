# Go2 RGB multiresolution perception V1 architecture decision

Date: 2026-07-24

Author: `/root`

Status: **SELECTED ARCHITECTURE; SOURCE-ONLY IMPLEMENTATION AUTHORITY AFTER
INDEPENDENT PREREGISTRATION REVIEW; NO GPU, DATA, CHECKPOINT, TRAINING,
CALIBRATION, G2, NAVIGATION, RUNTIME, PRODUCTION, OR HELD-OUT AUTHORITY**

Governing predecessor commits:

- `f3568880ecdda0d3f01ff8f661b19eb0753b58c9`
- `031568aac537f7c707a8e27b13a243ec1a02c646`
- `8cce85e016fb0160b65407dab17db7abad0397e3`

## Decision

Continue the RGB-only repository objective with exactly one materially
different perception mechanism: a progressive learned multiresolution spatial
decoder consuming the existing final `16x16` grid of 192-dimensional image
tokens and producing the unchanged `36x112x112` dense evidence feature map.

The successor is additive and separately versioned. It must not modify
`VisionEncoder`, Shared V5, the V4 evidence implementation, or the committed
72-path G2/runner closure.

The one allowed architecture delta is:

```text
(B,192,16,16)
  Conv3x3 192->112 + GroupNorm(8) + GELU
  bilinear resize to 28x28, align_corners=False
  Conv3x3 112->80 + GroupNorm(8) + GELU
  bilinear resize to 56x56, align_corners=False
  Conv3x3 80->56 + GroupNorm(8) + GELU
  bilinear resize to 112x112, align_corners=False
  Conv3x3 56->36 + GroupNorm(4) + GELU
  Conv3x3 36->36 + GroupNorm(4) + GELU
-> (B,36,112,112)
```

Every convolution has kernel size three, stride one, padding one, and a bias.
Every resize uses an explicit target size rather than a scale factor.

This is a progressive decoder over the final normalized ViT token grid. It is
not an intermediate-transformer feature pyramid; exposing intermediate encoder
blocks would be a second mechanism.

## Frozen capacity

| Component | Parameters | Parameter tensors |
|---|---:|---:|
| Progressive decoder | 345,264 | 20 |
| Unchanged pixel head | 4,736 | 2 |
| Unchanged ground head | 2,689 | 4 |
| Successor evidence head | 352,689 | 26 |
| Existing encoder | 2,747,520 | 78 |
| Total trainable | 3,100,209 | 104 |

The evidence-head ceiling is the predecessor's 357,993 parameters. The
successor is 5,304 parameters smaller, so this experiment does not test a
capacity increase.

## What remains unchanged

- RGB size, normalization, patch size, encoder topology, and one-online-encode
  semantics.
- Pixel first-hit hazards, within-bin offsets, ground-support logits, tensor
  axes, camera geometry, raster construction, and output schemas.
- Raw V13 train and checkpoint-selection roles, ordered samples, wrong-RGB
  mapping, physical evaluator, nine scopes, 189 margins, thresholds, and zero
  calibration access.
- The V6 tail-depth loss and all other Camera terms and coefficients.
- AdamW settings, float32 without autocast, four real B=4 microbatches per
  update, group order `evidence_head` then `encoder`, the uncompressed
  predecessor learning-rate function, encoder learning-rate scale `1.0`, and
  independent clip norm `1.0` for both groups.
- Frozen `bev_decoder`, `predictor`, `occupancy_head`, `target_encoder`, and
  `target_bev_decoder`; JEPA objective, JEPA backward, and EMA update counts
  remain zero in this perception-only probe.

No temporal frames, attitude conditioning, depth input, direct-BEV output,
raster-rule change, data refinement, loss tuning, threshold search, second
seed, or schedule extension is part of V1.

## Initialization and migration

Source work opens no tensor artifact. A later separately authorized execution
may open only the already-bound N320 fit checkpoint as its tensor
initialization input. No V1 through V6 adaptation checkpoint or terminal
runtime may be opened.

The runtime constructor must:

1. use base initialization seed `20260712`;
2. initialize only the new decoder under an isolated CPU generator with seed
   `20260724`, Xavier-uniform convolution weights, zero convolution biases,
   unit GroupNorm weights, and zero GroupNorm biases;
3. restore the caller RNG state exactly;
4. strict-copy only the N320 encoder, pixel head, and ground head;
5. require exact equality of the canonical ground-support buffer;
6. copy zero predecessor dense-decoder entries;
7. hard-sync the EMA target once; and
8. publish component hashes, copied-entry counts, runtime version, and a zero
   rejected-checkpoint-open count.

The inherited V5 migration method is prohibited because it assumes an
identical dense-decoder topology. Direct strict loading of a V5 state into the
successor must fail.

## Bounded falsification

The one attempt is capped at 1,000 optimizer updates and 16,000 ordered
pair-index presentations:

| Update | Presentations | Role |
|---:|---:|---|
| 100 | 1,600 | integrity and informational snapshot |
| 400 | 6,400 | integrity and informational snapshot |
| 1,000 | 16,000 | terminal scientific decision |

Updates 100 and 400 have no numeric continuation gate. Only an integrity
failure may stop before update 1,000. The operation-count compute ceiling is
one R9700, 1,000 optimizer steps, 4,000 real B=4 backward calls, 2,000
independent group clips, and exactly three inline nonmutating
checkpoint-selection evaluations. There is no observer evaluation rerun.

The terminal PASS is the strict conjunction:

- at least `1/9` complete physical scopes;
- at least `98/189` passed margins;
- total shortfall `< 41.01776266878769`;
- rough-motion pixel balanced accuracy `> 0.8198594673963917`;
- rough-motion ground balanced accuracy `> 0.647134926562893`; and
- rough-motion depth p95 `< 0.9777327477931971 m`.

Equality fails. A failed conjunct terminates this mechanism with no retry,
resume, second seed, extension, or automatic temporal successor. PASS licenses
only a separately preregistered bounded perception-qualification attempt; it
does not qualify a checkpoint.

## Source and custody order

1. Commit this decision, the machine-readable preregistration, and an
   independent source-free review.
2. Implement and test the separately versioned successor in place.
3. Commit a new recursive source manifest and independent source review.
4. Amend the clean-export allowlist only after that manifest exists and binds
   every exported path.
5. Perform an allowlist-only clean export. Whole-tree archive, worktree, or
   checkout export remains forbidden.
6. Create a separate execution authorization only after source, lifecycle,
   hardware-preflight, runtime-binding, and output-root checks pass.

Generated inputs are declared runtime bindings, not source-review inputs. They
may be validated only after reservation by the authorized runner.

## Relationship to the repository goal

This perception probe cannot establish JEPA navigation. If perception later
qualifies, the shortest valid next stage freezes the encoder and evidence head,
trains an action-conditioned predictor under a separate optimizer/clipping
boundary, and runs the mandatory matched no-JEPA arm. Deployed predictor
rollouts must causally affect learned target, frontier, route/subgoal, and
ordinary-motion scores.

The current Shared V5 development controller remains barred from promotion
because it uses current-frame-only memory, a fixed target, exact-pose
kinematics, A*, and hand-authored commands. Promotion remains fresh G2, then
G3 through G7, complete freeze, and a newly generated externally custodied G8.

