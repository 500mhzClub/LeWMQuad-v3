# RGB Swept-Progress Survival Joint-JEPA V12 Neutral Disjoint Ternary Competition — Preregistration

- Date: 2026-07-29, after the terminal V11 development result and before V12
  source, data access, GPU work, or model output.
- Parent authority: V11 result commit
  `d0fcd87594fd7608d7d74e95dc1a2c83748c24c6`.
- Status: one fresh, capped falsification of a zero-parameter semantic-algebra
  correction is selected. This grants implementation, focused source-only
  tests, independent review, and—only after a separate execution binding—one
  development run. It grants no retry, resume, rejected-checkpoint use,
  physical calibration, G2, navigation, held-out, sealed, deployment, or
  promotion access.

## Terminal evidence and causal diagnosis

- V11 trained exactly once for 1,000 updates and 16,000 presentations. All 14
  floor/elevated attention tensors were active from update 1, all 12 semantic
  tensors were active by update 2, the shared JEPA predictor/control gates
  passed, and 23 of 24 development checks passed.
- The sole failure was FREE recall `0.821735 < 0.85`. Relative to V10, V11
  improved OCCUPIED recall `0.874255 -> 0.903167` and rough OCCUPIED recall
  `0.734971 -> 0.794288`, but reduced FREE recall by `0.060765` and UNKNOWN
  recall by `0.019074`.
- This was an OCCUPIED overcall, not abstention or collapse. Of 30,419 lost
  correct FREE cells, 27,841 (`91.5%`) moved to OCCUPIED. Total OCCUPIED
  predictions rose by 80,032, yielding only 839 extra true positives and
  79,193 extra false positives.
- V11's occupied-priority adapter is:

  ```text
  P(OCCUPIED) = sigmoid(o)
  P(FREE)     = (1-sigmoid(o))*sigmoid(f)
  P(UNKNOWN)  = (1-sigmoid(o))*(1-sigmoid(f))
  ```

  It gives elevated evidence a hard veto. `P(FREE)>P(OCCUPIED)` requires
  `sigmoid(f)>exp(o)`, which is impossible for every `o>=0` even as floor
  evidence `f` grows without bound. The terminal FREE-to-OCCUPIED shift is the
  predicted signature of this coupling.

## History-aware choice

- Do not remove or retune the inherited occupied auxiliary. The V1/V2/V3
  coefficient `0/1/0.5` sequence already closed that family: reducing the
  coefficient raised FREE while reducing aggregate and rough OCCUPIED recall,
  and the midpoint still failed. V5's added hazard loss again sacrificed FREE
  and closed further loss-only safety variants.
- V4 and V10 both passed development with neutral three-way semantic
  competition and the exact inherited `O=0.5`. The coefficient therefore does
  not by itself explain V11; the new hard priority rule is the narrow untested
  difference that matches the confusion.
- Do not use the rejected KNOWN then OCCUPIED-given-KNOWN hierarchy, reverse
  FREE-priority hierarchy, symmetric conflict-to-UNKNOWN rule, FREE auxiliary,
  margin, mining, threshold loss, new branch, support-height change, extra
  resolution, or schedule extension. Each either collides with prior history,
  risks erasing the demonstrated obstacle gain, or adds an unnecessary second
  intervention.
- V12 changes exactly one function: how the already learned disjoint `f` and
  `o` axes compete. It adds no parameter, representation, loss, optimizer
  group, input, target, or framework.

## Frozen V12 mechanism

- Construct fresh from accepted N320 encoder-only initialization and the exact
  V11 source architecture. Read no V4–V11 experiment checkpoint, trace,
  intermediate tensor, calibration artifact, or model output.
- Preserve every V11 parameter name, tensor shape, initialization seed, and
  initial value bit-for-bit: encoder; 25-point V10 geometry; five `z=-0.333 m`
  floor supports; twenty elevated supports; two 2-head 64-to-32 attentions;
  separate refinement calls; role-ordered 64-channel latent; disjoint
  half-width FREE/OCCUPIED residual-local axes; action predictor; survival
  head; EMA target; and learned null evidence.
- Preserve exact role masks and validity. FREE evidence uses only latent
  channels `0:32` and is fixed to `-20` where no floor support is valid.
  OCCUPIED evidence uses only channels `32:64` and is fixed to `-20` where no
  elevated support is valid. All-invalid cells retain exact inherited logits
  `(0,-20,-20)`.
- On every supported cell, replace only the V11 occupied-priority composition
  with neutral UNKNOWN/FREE/OCCUPIED competition:

  ```text
  z_UNKNOWN  = 0
  z_FREE     = f
  z_OCCUPIED = o
  log_probabilities = log_softmax([z_UNKNOWN,z_FREE,z_OCCUPIED])
  ```

- This makes UNKNOWN the zero-evidence reference, FREE win exactly when
  `f>max(0,o)`, and OCCUPIED win exactly when `o>max(0,f)`. Both low axes
  abstain as UNKNOWN. Strong registered floor evidence can now outvote weak
  spurious elevated evidence; strong obstacle evidence can still outvote
  floor evidence.
- The V12 semantic wrapper must reuse the exact V11 `free_axis` and
  `occupied_axis` module objects/names without cloning or reinitialization.
  V12 and a fresh V11 witness from the same N320 state must have identical
  parameter/buffer inventories and bit-identical initial state. Construction
  must restore caller RNG byte-for-byte.
- The predictor consumes and predicts the same single role-ordered 64-channel
  latent. No semantic-only bypass, separate head phase, frozen encoder, or
  separately trained predictor is allowed.

## Frozen learning, cap, and gates

- Preserve exact V11/V10 Raw-V13 data identities and roles, endpoint order,
  labels, negative construction, RGB tensorization, schedule, batching,
  optimizer, weight decay, clipping, EMA, masks, controls, evaluator,
  bootstrap, thresholds, seeds, and hardware binding.
- Preserve exact joint objective `L=S+P+U+R+O`, including coefficient
  `O=0.5`, per-row present-class balancing, current/next averaging, and
  `log(2)` normalization. No loss source or coefficient changes are allowed.
- Reuse the reviewed V11 wrapper around the unchanged V3 update and V2 fixed
  driver. Preserve the 14 attention and 12 semantic gradient receipts,
  update-two activity requirement, zero target gradients, parameter partition,
  optimizer-step/EMA order, and accounting.
- Source tests must prove exact V11 state identity, zero new parameters,
  neutral algebra and normalization, disjoint axis routing, branch-invalid and
  all-invalid masks, unchanged occupied auxiliary identity, shared predictor
  state, EMA isolation, and exact cap/gate delegation.
- Execute at most once: exactly 1,000 optimizer/EMA updates, 4,000 B=4
  microbatch graphs, and 16,000 ordered presentations. Update 1,000 is the only
  decision state. No retry, resume, second seed, extension, intermediate
  checkpoint selection, or predecessor checkpoint use is authorized.
- The unchanged 24-check development gate is conjunctive. A 24/24 pass earns
  only one separately preregistered use of the numerically unchanged V10/V4
  physical calibrator and exact 2,016-tuple grid. Physical qualification still
  requires a feasible calibration tuple and selection FREE precision
  `>=0.99`, near-OCCUPIED detection `>=0.95`, useful FREE recall `>=0.90`, and
  near-obstacle exclusion `>=0.95`.
- G2 remains closed unless both development and physical gates pass. Held-out
  and sealed material remain unopened.

## Falsification and stopping rule

- Success would show that the role-separated RGB evidence was useful but
  V11's fixed hard priority prevented floor evidence from resolving ambiguous
  elevated activations.
- Failure closes this exact neutral disjoint competition, seed, objective,
  schedule, and cap. Do not follow it with a priority reversal, coefficient
  sweep, threshold change, retry, or rejected-checkpoint reuse. Any successor
  must be justified by a new terminal diagnosis rather than another semantic
  operating-point variant.
