# RGB Swept-Progress Survival Joint-JEPA V6 — Fine RGB BEV Fusion Preregistration

- Status: frozen before implementation, source review, runtime/data access, or training.
- Clean scientific baseline: V4 source / full-arm result commits `aaa47a138d0eeb78aa20d9524e67f813f7a74a41` / `8b3a8063b087c81030189deadc6c5f6e1c7d44c3`.
- Closed loss-only predecessor: V5 result commit `ea6bc794fc9bc7dbc20e730cbb36e940ef3a3e96`.
- Purpose: test one architecture-level mechanism for spatial RGB detail that the existing `16x16` ViT token route may discard before the geometry-anchored BEV lift. This is not another ranking-loss, coefficient, margin, calibration, threshold, token-overlap, or detached-head variant.

## Falsified predecessor and rationale

- V5's near-field ranking loss was active in 3,968/4,000 microbatches over 22,579,859 pairs and fell `2.393182 -> 0.553726`, but selection free recall fell `0.857970 -> 0.783955`, occupied recall fell `0.744512 -> 0.709053`, and occupied precision also fell. The loss changed the operating point without demonstrating better obstacle/free separation. V5 and its checkpoint are closed with no further access or variant.
- The current encoder reduces `112x112` RGB to non-overlapping `7x7` patches and a `16x16` spatial token map. Transformer depth does not restore spatial resolution. The earlier overlapping-tokenization mechanism retained stride seven and the same `16x16` output and already failed; V6 does not repeat it.
- V6 instead preserves one learned pixel-scale `112x112` feature map until the existing projective BEV sampler. The new evidence enters the same latent consumed by the semantic decoder and action-conditioned predictor, so it remains one jointly trained RGB JEPA.

## Sole scientific change from V4

- Retain the exact V4 residual-local semantic decoder, accepted N320 encoder-only initialization, ViT encoder, deformable BEV lift, action-conditioned predictor, survival head, EMA target, data, labels, action vocabulary, masks, optimizer rules and hyperparameters, clipping, seeds, schedule, controls, metrics, gates, and V4 losses `S+P+U+R+O` with `O` coefficient `0.5`.
- Do not carry V5 loss `H` or open any V5 or V4 produced checkpoint. Construct V6 freshly through the same N320 and V4 constructor path.
- Add exactly one fine-RGB residual module as a child of both the online and EMA-target BEV lifts:
  - normalized RGB `[B,3,112,112]`;
  - biased `Conv2d(3,32,kernel_size=3,stride=1,padding=1)`;
  - exact `GELU(approximate="none")`;
  - biased `Conv2d(32,32,kernel_size=3,stride=1,padding=1)`;
  - exact `GELU(approximate="none")`;
  - biased `Conv2d(32,64,kernel_size=1,stride=1,padding=0)`, with weight and bias initialized to exact zero.
- There is no pooling, downsampling, normalization, dilation, coordinate channel, depth input, new sample location, new sample weight, semantic-only head, or predictor-only path. Added trainable parameter count is exactly 12,256.
- Initialize the first two convolutions with the standard PyTorch `Conv2d` initialization under isolated CPU seed `20260714`; zero the final projection exactly; restore caller RNG. Copy the complete branch exactly into the target lift, keep it frozen there, and include it in the inherited hard-sync/EMA module identity.

## Exact geometry fusion

- Run the unchanged V4 deformable lift first and retain its final `[B,64,64,64]` latent plus exact four `sample_grid_xy`, `sample_valid_mask`, and normalized `sample_weights` per BEV cell.
- Apply the fine-RGB module through its zero projection to produce `[B,64,112,112]`. Sample it with `grid_sample(mode="bilinear",padding_mode="zeros",align_corners=False)` at the exact existing four grids; do not detach or alter the grids or weights.
- Reshape samples per BEV cell, multiply by the exact existing normalized weights, and sum the four samples. Set the residual to exact zero wherever the inherited `cell_valid_mask` is false.
- Return `fused_latent = inherited_final_latent + fine_residual`. The fused latent replaces the inherited latent for the V4 semantic decoder, the full action-conditioned JEPA predictor, and the online/EMA persistence target. All other sampling receipts remain unchanged.
- Exact-zero final projection makes initial V6 latent, semantic logits, and predictor inputs bitwise equal to V4. The final projection receives gradient on the first backward pass; earlier fine convolutions unlock after the first optimizer step. This is not a staged training procedure.

## Training and falsification

- One fresh V6 run only: exactly 1,000 optimizer/EMA updates, 4,000 microbatch graphs/backward calls, and 16,000 presentations. No retry, resume, extension, intermediate selection, channel-width variant, alternate tap, normalization, or fusion variant.
- All 12,256 online fine-branch parameters must enter the inherited lift/semantic optimizer and clipping group exactly once. All target copies remain frozen and receive no gradient. Record initial parity, parameter/partition identities, branch gradient receipts, and unchanged accounting.
- The terminal checkpoint must first pass the complete unchanged V4 24-check full-arm development gate. Any failure closes V6 without calibration or G2.
- If and only if that gate passes, package the V6 terminal checkpoint as a development-only pre-calibration candidate and execute one separately source-frozen V6-specific refit of the already-reviewed calibration protocol: one four-parameter fit on the calibration role, one unchanged 2,016-tuple threshold search there, and one unchanged application to selection.
- Physical development success still requires at least one passing calibration tuple, selection free precision at least `0.99`, near-obstacle detection at least `0.95`, useful-free recall at least `0.90`, and near-obstacle exclusion at least `0.95`.

## Authority and stopping

- A V6 full-arm failure terminates this fine-RGB fusion mechanism without calibration, tuning, retry, or checkpoint access. A physical failure terminates it without G2.
- A physical pass authorizes only preparation and review of a one-shot G2 binding; it does not itself open G2 or qualify navigation.
- No G2, navigation, held-out, sealed, production, rejected-checkpoint, no-persistence-checkpoint, original-V4-runtime, V4-candidate, or V5-runtime access is authorized by this preregistration.
