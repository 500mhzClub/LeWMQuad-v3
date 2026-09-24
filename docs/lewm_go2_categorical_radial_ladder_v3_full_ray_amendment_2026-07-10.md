# Go2 categorical radial ladder v3 full-ray amendment

Date preregistered: 2026-07-10 22:10 BST

Status: active; written before any v3 model output

## Scope

V3 changes exactly one architectural mechanism in the train-only N=1/4/16
ladder: the radial-context receptive field. The frozen frame identities,
images, labels, factorization, projective anchors, encoder, token projection,
channel widths, angular context, categorical head, loss, batch order,
initialization seed, V2 learning-rate schedule, stage budgets, batch sizes,
controls, evaluation cadence, and terminal gates remain unchanged.

This remains an implementation and capacity diagnostic. It cannot select or
promote a checkpoint, authorize a full-dataset run, open a non-train role,
pass G2, or support a perception, JEPA, memory, exploration, or navigation
claim.

## Immutable evidence

The amendment responds only to the completed V2 artifact:

- result:
  `.generated/go2_categorical_radial_micro_overfit/v2/seed_20260710_ladder_result.json`;
- result file SHA-256:
  `06517e2c6641495a6262aa9f8a5cb45648912c575f1c3663df899c50a2867daa`;
- result content SHA-256:
  `8528ae02d6faaf25eb666d591e15180e82f74c9cf4d798c8322f9d5c50c910bc`;
- frozen ladder manifest:
  `.generated/go2_categorical_radial_micro_overfit/v1/ladder_manifest.json`;
- ladder manifest file SHA-256:
  `967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12`;
- V2 optimizer amendment file SHA-256:
  `58f994a639c8e5a733d92c6da1fad63fa654e1f57aa7be0a8373e3eaa47b3f46`.

V2 passed its fixed N=1 and N=4 terminal gates. N=16 improved smoothly to
the fixed step-2,000 checkpoint but failed with balanced NLL `0.01151191`,
UNKNOWN/FREE/OCCUPIED recalls `0.98747 / 0.99828 / 0.97007`, and wrong-view
minus correct-RGB NLL `3.53510`. The terminal errors included 12 of 401 truly
OCCUPIED cells predicted UNKNOWN and 736 of 58,734 truly UNKNOWN cells
predicted known. Geometry scatter/gather is injective and exact, correct-view
dependence is strong, and there is no late excursion at the recorded 100-step
aggregate evaluations.

This is an operational decoder structure/capacity ceiling under the frozen
budget, not proof of mathematical parameter insufficiency or exclusion of
under-optimization. The V2 cosine has mean rate about `1.05e-4` and integrated
rate about `0.21` across 2,000 updates, versus `0.40` for the original constant
rate; V3 keeps that schedule frozen rather than using this observation to tune
it. The sparse OCCUPIED support makes its 0.99 recall threshold strict, but the
residual error is materially larger than a one-cell threshold accident.

## Sole architecture change

The V2 decoder has one residual radial convolution with kernel `(5, 1)`. Its
direct, location-specific radial feature path spans five bins, or 0.5 m on the
registered 0.1 m lattice. GroupNorm statistics and the ViT encoder already
create indirect global dependencies, but neither supplies a learned,
range-indexed convolutional path that can transport the location of surface
evidence along a ray. That direct path is structurally mismatched to the
registered mechanism: a bin's class may depend on the ordering of FREE space,
an OCCUPIED surface, and UNKNOWN space anywhere along the complete 6.4 m ray.

V3 replaces only that one radial block with six residual radial blocks. Each
uses:

1. a 64-to-64 convolution with kernel `(3, 1)`, dilation `(d, 1)`, and padding
   `(d, 0)`;
2. GroupNorm with eight groups;
3. GELU;
4. a 64-to-64 pointwise convolution;
5. an identity residual connection.

The dilation sequence is fixed to `(1, 2, 4, 8, 16, 32)`. Its nominal radial
receptive-field width is:

```text
1 + 2 * (1 + 2 + 4 + 8 + 16 + 32) = 127 bins
```

The power-of-two composition supplies every integer offset through 63, so each
output bin has a direct convolutional feature path to every valid bin in the
64-bin ray, including from either edge. Zero padding is used; there is no
circular range wrap. Angular context remains the single registered `(1, 5)`
block.

Each new block has 16,640 parameters. The six-block stack has 99,840 versus
24,832 in V2, for a net increase of 75,008. The exact registered model count is
therefore 2,887,067 parameters, 2.67% above V2; decoder parameters grow from
64,539 to 139,547, or 116.2%, at the hypothesized radial bottleneck without
widening or changing the encoder. The radial module uses about 4.02 times the
V2 multiply-accumulates, and dilation 32 is padding-heavy.

This is one bundled full-ray-stack intervention, not an identified receptive-
field-only effect. Replacing one block with six also adds five GroupNorm/GELU
stages, nonlinear depth, parameters, and compute. A pass cannot distinguish
the full-ray path from that added nonlinear capacity; it only supports the
registered bundle.

V3 construction must preserve the initialized values of every common V2
parameter outside `radial_context` under seed 20260710. The authority runner
must configure the production runtime, construct the complete V2 base, and
first reproduce its full initial-state SHA-256
`ad120b467aeabb60f20b9fd663d0438451298895ee87717a6795810bbb5b8f75`.
It then resets the same RNG seed, constructs V3 by completing that unchanged
base initialization before replacing only `radial_context`, and proves every
common tensor bitwise equal. New blocks are constructed in the fixed dilation
order. No V2 source or immutable artifact may be modified.

## Frozen optimization and gates

V3 reuses the exact V2 stage-local cosine schedule, with no warmup:

```text
lr(u) = 1e-5 + 0.5 * (2e-4 - 1e-5)
                  * (1 + cos(pi * (u - 1) / (U - 1)))
```

Update 1 uses `2e-4`; update `U` uses `1e-5`; the rate is assigned immediately
before each optimizer step. The stage budgets remain 1,000/1,500/2,000,
batches remain 1/4/4, AdamW defaults and weight decay `1e-4` remain unchanged,
gradient clipping remains 1.0, and evaluation remains every 100 updates.

The complete terminal gates, wrong-view controls, stage restarts, and
first-failure stopping rule remain unchanged. There is no longer run, learning
rate change, early stopping, EMA, checkpoint averaging, retry, best-step
selection, or second seed. This amendment does not change or license either
separately registered N32 optimizer branch.

## Pre-output implementation gates

Focused tests must prove before GPU output that:

- the dilation sequence and exact parameter count are frozen;
- a binary convolutional-reachability audit, independent of normalization
  statistics, computes the complete 64x64 output-bin/input-bin transitive
  matrix and proves every entry true after the six-block stack;
- each individual layer's adjacency equals only the in-bounds clipped offsets
  `{-d, 0, +d}`, proving there is no circular range wrap;
- polar and Cartesian shapes, finite logits, factorization, support mask, and
  deterministic gather remain unchanged;
- common V2 parameters outside `radial_context` initialize identically;
- every stage restarts from one identical V3 initial state;
- schedule endpoints and update timing remain exactly V2;
- smoke execution visits N=1, N=4, and N=16 and records zero non-train access;
- the runner refuses any drift in V1/V2 sources, the new V3 model and runner
  sources, V2 result, manifest, or this amendment.

## Decision rule

Authoritative V3 consumes the complete fixed budget and is judged only at each
stage's final checkpoint. It stops at the first failure. Only fixed-terminal
passes at N=1, N=4, and N=16 license construction of the already-registered
N32 diagnostic. A failure triggers a new diagnosis; it does not license a
width change, schedule sweep, longer run, second seed, holdout read, or G2
access.
