# RGB Swept-Progress Survival Joint-JEPA V11 Height-Role Factorized Evidence Lift — Preregistration

- Date: 2026-07-29, after the terminal V10 physical result and before V11
  source, data access, GPU work, or model output.
- Parent authority: V10 physical-result commit
  `2710b01751ec1cf3e9e9d5a0eeda10aabff6c8f5`.
- Status: one fresh, capped falsification of one cohesive perception mechanism
  is selected. This record grants implementation, focused source-only tests,
  independent review, and—only after a separate frozen execution binding—one
  development run. It grants no retry, resume, rejected-checkpoint reuse, G2,
  navigation, held-out, sealed, production, deployment, or promotion access.

## What V10 established

- V10 passed all 24 unchanged development checks at exactly 1,000 updates and
  16,000 presentations. Its selection semantic balanced accuracy was
  `0.902738`; FREE/OCCUPIED/UNKNOWN recall was
  `0.882500/0.874255/0.951460`; and its jointly trained predictor beat every
  registered control.
- The sole unchanged physical-evidence calibration then failed. Selection
  FREE precision was `0.9262134889719079`, near-OCCUPIED detection
  `0.6179117268961577`, useful FREE recall `0.8359237513198733`, and
  near-obstacle exclusion `0.8093129552921011`. There were zero feasible
  calibration tuples.
- Relative to V9, V10 improved near-OCCUPIED detection by `+0.295101` and
  exclusion by `+0.065134`, while FREE precision and recall were essentially
  flat (`-0.002693` and `-0.006803`). The 3-D support routing therefore fixed
  a real obstacle-visibility problem, but the shared unordered aggregation
  still did not separate safe floor, vertical obstacle, and abstention tails.
- V10 is closed. Its checkpoint, calibration, support coordinates, heights,
  thresholds, seed, schedule, and run length may not be reopened or tuned.

## History-aware choice

- Do not repeat a generic KNOWN-versus-UNKNOWN then
  FREE-versus-OCCUPIED-given-KNOWN head. Categorical-radial N32 V4 already
  tested that exact output hierarchy and worsened every critical terminal
  recall relative to its width-24 control. Its adjudication explicitly
  rejected last-layer coupling as a sufficient cause.
- Do not repeat V2/V3 occupied-loss coefficient variants, V5 near-hazard
  ranking, V6 fine RGB fusion, V7 CNN, V8 higher-resolution ViT, V9 unordered
  local attention, Camera ray/depth variants, learned global queries, or a
  temporal mechanism. Those causal families have already received valid
  negative tests or are terminally closed.
- The remaining specific defect is in V10's lift: its Q/K/V attention has no
  support-position or height identity and is permutation-invariant over the
  valid 25-token bag. A floor sample and a vertical-surface sample can enter
  the same aggregate without a registered semantic role.
- V11 therefore makes one cohesive change: route the existing V10 support
  samples through fixed floor and elevated evidence branches, preserve those
  roles as two halves of the same 64-channel JEPA state, and decode them with
  an occupied-priority abstaining semantic adapter. This is not a separately
  trained encoder or head; semantic, predictor, survival, and EMA objectives
  remain joint from update one.

## Frozen V11 mechanism

### Inputs and geometry

- Construct fresh from the same clean V4 architecture and accepted N320
  encoder-only initialization. Read no V4–V10 experiment checkpoint, trace,
  intermediate tensor, calibration artifact, or model output.
- Preserve V10's exact 112x112 RGB input, 16x16 final token lattice,
  64x64 metric BEV, camera contract, bilinear sampling, 25 horizontal-major
  supports, support coordinates, heights, validity bits, and cell-valid OR.
- The full cell-valid mask remains exactly 2,062 cells with row-major uint8
  SHA-256
  `4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b`;
  222 of the 1,016 cells within 2 m are valid.

### Fixed support roles

- FREE/floor supports are exactly indices `[0,5,10,15,20]`: all five
  horizontal offsets at registered height `z=-0.333 m`.
- OCCUPIED/elevated supports are exactly the complementary 20 indices at
  heights `[-0.133,0.067,0.267,0.467] m`. The role masks are disjoint and
  exhaustive; there is no learned routing, tuned height, or third branch.
- OR over valid floor supports contains 2,024 cells and has row-major uint8
  SHA-256
  `8b6b4202d04cf08de9813a4fc12deff9ea35de8d8c7adc8eb40a117593694bbc`.
  OR over elevated supports contains all 2,062 V10-valid cells and therefore
  has the V10 cell-mask SHA above. They overlap on 2,024 cells; exactly 38
  cells are elevated-only. Within 2 m, floor/elevated valid counts are
  `184/222`, including the same 38 elevated-only cells.

### Learned branch aggregation and shared JEPA state

- Project patch tokens from 192 to 64 channels exactly as V10 and sample the
  exact same 25 image positions once. Each branch forms an arithmetic mean of
  only its valid registered supports; invalid supports contribute exact zero.
- Each branch has one local two-head attention with head width 16. Its query,
  key, and value projections are `64->32`; its output projection is `32->32`.
  Query and output projections have bias, key has no bias, and value has bias.
  Attention uses scaled dot product, no dropout, and a masked softmax over only
  that branch's fixed supports. Cross-role attention weights are exact zero.
- A branch output is its projected masked-mean query plus its projected
  attended value. FREE/floor occupies latent channels `0:32`; elevated/OCCUPIED
  occupies channels `32:64`. Concatenate them into one 64-channel latent.
- Apply the two inherited 64-channel residual-local refinement blocks without
  adding parameters, but evaluate each role in a separate padded call and
  retain only its registered output half. Thus each final half depends only on
  its own support role plus inherited learned null evidence; there is no
  cross-role sample mixing. Concatenate the retained halves before every JEPA
  predictor call.
- Reuse V10's token projection, null evidence, and both refinement-block
  tensors bit-for-bit at initialization. Replace V10's four-head unordered
  attention with the two registered role attentions. Initialize their 14
  parameter tensors (14,528 parameters per online or target lift) in the
  fixed order floor Q/K/V/O then elevated Q/K/V/O using an isolated CPU
  generator seeded `20260730`, Xavier-uniform weights, and exact-zero biases.
  Restore the caller CPU RNG byte-for-byte.
- Mirror the complete lift and role buffers in the EMA target. The action-
  conditioned predictor consumes and predicts the same role-ordered
  64-channel state; no semantic-only side path or predictor bypass is allowed.

### Occupied-priority abstaining semantic adapter

- Replace the three-output V4 decoder with two disjoint residual-local evidence
  axes. The FREE axis reads only latent channels `0:32`; the OCCUPIED axis reads
  only channels `32:64`. Each axis is one biased `32->1` 1x1 base projection
  plus a biased `32->32` 3x3 GELU branch and zero-initialized biased `32->1`
  residual output, matching V4's local-decoder pattern at half width.
- Initialize both axes in the fixed order FREE then OCCUPIED with the isolated
  seed `20260730`; base/local weights use Xavier uniform, their biases are
  zero, and residual-output weights/biases are exact zero. The two axes contain
  18,628 parameters in 12 tensors and remain in the existing
  `lift_semantic` optimizer group.
- Let learned evidence logits be `f` and `o`. On branch-valid cells, expose
  normalized three-class log probabilities exactly as:

  ```text
  log P(OCCUPIED) = log_sigmoid(o)
  log P(FREE)     = log_sigmoid(-o) + log_sigmoid(f)
  log P(UNKNOWN)  = log_sigmoid(-o) + log_sigmoid(-f)
  ```

  This makes OCCUPIED evidence safety-prioritized, makes FREE require both
  positive floor evidence and absence of elevated obstacle evidence, and sends
  lack of both to UNKNOWN. A floor-invalid cell cannot emit learned FREE
  evidence; an elevated-invalid cell cannot emit learned OCCUPIED evidence.
  Every all-invalid cell retains exact inherited logits `(0,-20,-20)`.
- This adapter is distinct from the rejected KNOWN/OCCUPIED-given-KNOWN head:
  it factorizes OCCUPIED-versus-rest first, then FREE-versus-UNKNOWN, and each
  axis receives a different height-routed latent. Its purpose is not added
  last-layer capacity but a registered physical-evidence route.

## Frozen learning, integrity, cap, and gates

- Preserve exact Raw-V13 data identities and roles, endpoint order, labels,
  negative construction, action vocabulary, RGB tensorization, schedule
  prefix, batching, optimizer learning rates/weight decay, gradient clipping,
  EMA momentum, masks, controls, bootstrap, evaluator, and thresholds.
- Preserve the exact joint objective `L=S+P+U+R+O`, including occupied
  auxiliary coefficient `0.5`. No new loss, reweighting, margin, mining,
  curriculum, threshold loss, calibration change, or separate head/predictor
  training is allowed.
- Source/integrity tests must prove exact role indices/counts/hashes, zero
  cross-role attention, finite normalized class probabilities, invalid-cell
  UNKNOWN behavior, exact inherited V10 tensors outside the replaced
  attentions/semantic decoder, complete optimizer partition, online gradients
  for every new branch tensor by update two, zero target gradients, predictor
  use of the shared 64-channel latent, and exact optimizer/EMA accounting.
- Execute at most once: exactly 1,000 optimizer/EMA updates, four B=4
  microbatches per update, and 16,000 ordered presentations. The terminal
  update-1000 state is the only decision point. No retry, resume, second seed,
  extension, or checkpoint selection is authorized.
- The unchanged conjunctive 24-check development gate remains authoritative:
  balanced accuracy `>=0.80`, FREE recall `>=0.85`, OCCUPIED recall `>=0.70`,
  rough OCCUPIED recall `>=0.65`, UNKNOWN recall `>=0.90`, utility `>=0.85`,
  zero-prefix rate `<=0.05`, concordance `>=0.75`, all family floors, and all
  registered causal-control checks.
- A 24/24 pass earns only one separately preregistered and frozen use of the
  numerically unchanged V10/V4 physical calibrator and exact 2,016-tuple grid.
  Physical qualification still requires a feasible calibration tuple and
  selection FREE precision `>=0.99`, near-OCCUPIED detection `>=0.95`, useful
  FREE recall `>=0.90`, and near-obstacle exclusion `>=0.95`.
- For interpretation only, not promotion, the height-role hypothesis counts
  as directionally improving after physical scoring only if all four V10
  selection baselines improve by at least `0.01`: FREE precision
  `>=0.9362134889719079`, near detection `>=0.6279117268961577`, useful FREE
  recall `>=0.8459237513198733`, and exclusion `>=0.8193129552921011`.
- G2 remains closed unless both unchanged development and physical gates pass.
  Held-out and sealed material remain unopened.

## Falsification and stopping rule

- Success would show that the learned RGB tokens already contain both floor
  and vertical-surface evidence, but V10 erased their height role before the
  shared JEPA predictor and semantic decoder could use it.
- The primary known risk is that the elevated-only OCCUPIED branch may miss a
  truly low obstacle. Test that risk rather than adding a third branch or
  changing the registered height split before the result.
- Any source-integrity failure is repaired only before the bound scientific
  command. Once the command begins, an operational or scientific failure
  closes this exact V11 mechanism. Do not retry, resume, tune heights, add a
  branch, change the loss/thresholds, reuse the checkpoint, or extend the run.
