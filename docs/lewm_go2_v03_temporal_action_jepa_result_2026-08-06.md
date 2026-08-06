# Matched temporal action-conditioned JEPA: frozen vs top-block encoder movement

Date: 2026-08-06
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** No manifest or authorization
status is inherited. `probability_calibration`, `evaluation`, `untouched` and
sealed data were never opened.

Artifacts: `/home/andrewknowles/.cache/lewm_go2_temporal_v03/` (root filesystem —
the workspace pool is full at 658 MB free)

---

## Verdict

> **REJECT ENCODER-MOVING RECIPE.** Encoder movement made the future *more
> predictable* and *less action-discriminative*, in 8 of 8 selection scenes,
> while spatial information on true tokens was preserved.

Bounded to the fixed six-epoch recipe (see §Checkpoints).

---

## Verification performed before interpreting any difference

### Float32 vs float16-cache-and-batched parity, 24 fixed selection rows

| | frozen | moving |
|---|---:|---:|
| feature max abs diff | 0.015617 | 0.015621 |
| feature mean abs diff | 0.000204 | 0.000159 |
| feature relative mean | 1.76e-04 | 1.75e-04 |
| prediction max abs diff | 0.007917 | 0.008228 |
| prediction mean abs diff | 0.000091 | 0.000079 |
| changed cosine, fp32 → fp16 | 0.850782 → 0.850782 (**+0.0e+00**) | 0.868995 → 0.868994 (**−1.2e−07**) |
| normalised error, fp32 → fp16 | 0.547480 → 0.547479 (−6.6e−07) | 0.547385 → 0.547386 (+6.0e−07) |

The derived cosine and error metrics agree to **1e−6 or better** — six orders of
magnitude below the 0.0098 margin difference being interpreted. The repaired
cache and batched-prediction path are not a source of any reported effect.

### Mask and input invariants

- **One** changed-token threshold, derived from the **frozen** arm's train
  representation only: quantile 0.75, value `0.76190`, selecting **94,540 of
  377,088** selection tokens. The identical boolean mask is applied to both arms
  and to the correct, shuffled and persistence comparisons.
  *This corrects a defect in the first pass*, which derived threshold and mask
  per arm and so scored the two arms on different token subsets. Correcting it
  moved the moving arm's margin from +0.0479 to +0.0488 and left the conclusion
  unchanged.
- **Context is identical** across correct, shuffled and persistence for each
  sequence. `Predictor.forward` emits all 768 target tokens and *ignores* its
  mask argument, so the mask cannot differ between action arms by construction.
  Note the consequence honestly: **context frames are fully visible** — the
  masking selects which target positions enter the loss and the score, it does
  not hide context. This is a masked *target*, not a masked *context*, objective.
- **The action tensor is the only input that differs** in the sensitivity
  comparison; context, mask, encoder and predictor weights are held fixed.

### Checkpoints

All **six** epoch checkpoints exist for both arms (`epoch0`–`epoch5`). No epoch
selection was performed: both arms are evaluated at epoch 5. **The conclusion is
therefore bounded to the fixed six-epoch training recipe**, and says nothing
about whether a different schedule would behave differently.

---

## 1. Prediction on changed tokens (shared mask, 94,540 tokens)

| | frozen | moving |
|---|---:|---:|
| correct action, cosine | 0.7534 | **0.7865** |
| correct action, normalised error | 0.4202 | 0.4229 |
| shuffled action, cosine (mean of 3) | 0.6949 | **0.7377** |
| shuffled action, normalised error | 0.5172 | 0.5173 |
| persistence, cosine | 0.4800 | **0.5446** |
| persistence, normalised error | 1.0000 | 1.0000 |
| **correct − shuffled** | **+0.0586** | **+0.0488** |
| correct − persistence | +0.2735 | +0.2419 |

The moving arm's correct-action cosine is higher and its JEPA loss was 11.5%
lower (0.10231 vs 0.11562). Neither is an acceptance criterion, and here they are
actively misleading: its shuffled and persistence baselines rose further, so both
margins fell.

## 2. Margin per selection scene — 0 of 8 favour the moving arm

| scene | family | frozen | moving |
|---|---|---:|---:|
| `large_enclosed_maze_d78318b1e87b` | large_enclosed_maze | +0.0686 | +0.0588 |
| `local_composite_motifs_811b818f1914` | local_composite_motifs | +0.0679 | +0.0564 |
| `medium_enclosed_maze_f30352cb052e` | medium_enclosed_maze | +0.0640 | +0.0523 |
| `visual_sensor_stress_dc440a3fb679` | visual_sensor_stress | +0.0595 | +0.0493 |
| `loop_alias_stress_aeb36ab10bc1` | loop_alias_stress | +0.0577 | +0.0464 |
| `small_enclosed_maze_16b0fc2c449b` | small_enclosed_maze | +0.0474 | +0.0412 |
| **`open_obstacle_field_25cc6fe2de4f`** | open_obstacle_field | **+0.0469** | **+0.0397** |
| `rough_local_dynamics_0e631dbfbd46` | rough_local_dynamics | +0.0409 | +0.0356 |

**8 of 8 down.** The independent units here are the scenes, and none of them
dissents.

## 3. Spatial on true encoder tokens (current frame, current labels)

| | frozen | moving |
|---|---:|---:|
| fixed-probe precision | 0.6489 | 0.6543 |
| fixed-probe recall | 0.6829 | 0.6461 |
| **fixed-probe occupied IoU** | **0.4986** | **0.4817** |
| fresh-probe precision | 0.6551 | **0.6864** |
| fresh-probe recall | 0.6805 | 0.6620 |
| **fresh-probe occupied IoU** | **0.5010** | **0.5082** |

Under a probe retrained on its own features the moving encoder is **slightly
better** spatially. Under the fixed probe it is 0.0170 worse. The representation
moved relative to the fixed probe's input space without becoming less
informative — drift, not loss. Arm A's fixed probe reproduces the frozen
reference (0.4986 vs 0.4986), as it must.

## 4. Frozen probe on **predicted future** tokens vs persistence (future labels)

Probe trained once on normalised frozen true-future tokens, applied unchanged.

| | occupied IoU | precision | recall | predicted occ. fraction |
|---|---:|---:|---:|---:|
| true future (upper reference) | 0.4970 | 0.6398 | 0.6900 | — |
| frozen — **persistence** | **0.3133** | 0.5076 | 0.4501 | 0.01643 |
| frozen — predicted | 0.2654 | 0.4969 | 0.3629 | 0.01111 |
| moving — **persistence** | **0.3071** | 0.4998 | 0.4435 | 0.01642 |
| moving — predicted | 0.2304 | 0.4886 | 0.3036 | 0.00989 |
| *(target occupied fraction)* | | | | *0.00701* |

**This is the sharpest negative result in the run, and it indicts both arms.**
Decoded to occupancy, *neither* predictor's forecast beats simply reusing the
current tokens: 0.2654 vs 0.3133 for frozen, 0.2304 vs 0.3071 for moving. The
latent-space win over persistence (+0.27 cosine) does **not** survive decoding to
the geometry we actually need. And the moving arm is worse than the frozen arm
here too.

## 5. Raw token health

| | frozen | moving | delta |
|---|---:|---:|---:|
| raw token variance | 0.5685 | 0.5009 | **−11.9%** |
| raw effective rank | 85.81 | 97.71 | **+13.9%** |
| raw temporal delta | 0.4965 | 0.4430 | **−10.8%** |

No collapse — effective rank *rose*. But variance and temporal delta both fell
~11%, and persistence cosine rose 0.0646: consecutive frames became more similar
in latent space. The encoder made the sequence smoother, which helps every arm of
the prediction equally and therefore shrinks the margin.

## 6. `open_obstacle_field`

| | frozen | moving |
|---|---:|---:|
| action margin | **+0.0469** | +0.0397 |
| fresh-probe IoU / precision | 0.2198 / 0.3005 | **0.2491 / 0.4136** |
| fixed-probe IoU | 0.2127 | 0.2072 |
| predicted-future IoU / precision | **0.1385 / 0.2759** | 0.1310 / 0.2611 |
| persistence-future IoU / precision | 0.1329 / 0.1930 | 0.1276 / 0.1922 |

The moving arm is clearly better on this family's *static* geometry (+0.0293 IoU,
+0.1131 precision) and clearly worse on its *action margin* and its predicted
future. It remains the weakest family on every axis.

## 7. Predicted occupied fractions — over-prediction ruled out

Target occupied fraction is `0.00726` of all cells.

| probe | arm | predicted occ. fraction | precision |
|---|---|---:|---:|
| fixed | frozen | 0.01674 (2.31×) | 0.6489 |
| fixed | moving | 0.01526 (2.10×) | 0.6543 |
| fresh | frozen | 0.01646 (2.27×) | 0.6551 |
| fresh | moving | **0.01469 (2.02×)** | **0.6864** |

Both arms over-predict occupancy roughly 2×, but the moving arm over-predicts
*less* while scoring *higher* precision. Its small fresh-probe IoU gain is
therefore genuine sharpening, not diffuse over-prediction. That check clears the
moving arm — and it still fails the margin criterion.

---

## Accepted narrow conclusion

Accepted for the **fixed six-epoch run**:

1. **V-JEPA 2.1's inherited spatial geometry survived encoder movement** and
   remained recoverable under a fresh probe — fresh-probe occupied IoU
   `0.5010 → 0.5082`, precision `0.6551 → 0.6864`, `open_obstacle_field`
   `0.2198 → 0.2491`. Nothing was lost from the representation.
2. **Encoder movement nevertheless reduced correct-versus-shuffled action
   sensitivity in all eight selection scenes** — margin `+0.0586 → +0.0488`,
   0 of 8 scenes favouring the moving arm.
3. **Most importantly: predicted future tokens were spatially worse than
   persistence in both arms** — frozen `0.2654` vs persistence `0.3133`; moving
   `0.2304` vs persistence `0.3071`, against a true-future reference of `0.4970`.
   This is the finding that matters most, and it is not about encoder movement:
   it indicts the predictor in both arms.

### Implementation scope, stated explicitly

**The completed model used fully visible three-frame context with a masked
target/loss.** `Predictor.forward` emits all 768 target tokens and **ignores its
mask argument**; the mask selects only which target positions enter the training
loss and the evaluation score. **This was therefore not a full masked-context
V-JEPA implementation.** Context tokens were never hidden from the predictor, so
the V-JEPA 2.1 context—target asymmetry that motivates the recipe was not
reproduced. Any claim about "masked context—target objectives" from this run must
be read with that limitation attached.

### Gate on reintroducing encoder adaptation

**The next frozen-encoder predictor must beat persistence on future occupied
geometry before encoder adaptation is reintroduced.** No further encoder-moving
run is authorised until that holds.

## Localisation diagnostic: where does the predicted-token geometry go?

Read-only, on the completed checkpoints and cached features. **Confirmed first:**
the acceptance metrics in §4 use a probe trained on **true future encoder tokens**
(normalised, frozen arm, train role, t+240 raster labels) and applied
**unchanged** to true-future, persistence and predicted tokens; its checkpoint was
selected on true-future selection tokens.

Four identical-capacity probes, each trained on the train split of one token kind
and evaluated only on the matching `checkpoint_selection` tokens, all against
future labels:

| probe (fresh, diagnostic only) | occupied IoU | precision | recall | `open_obstacle_field` IoU / P |
|---|---:|---:|---:|---|
| *true future, fixed acceptance probe* | *0.4970* | *0.6398* | *0.6900* | — |
| frozen — **predicted** | **0.3774** | 0.4981 | 0.6091 | 0.1387 / 0.2030 |
| moving — **predicted** | **0.3818** | 0.4890 | 0.6353 | 0.1313 / 0.1720 |
| frozen — persistence | 0.3285 | 0.4009 | 0.6453 | 0.1085 / 0.1308 |
| moving — persistence | 0.3179 | 0.4578 | 0.5099 | 0.1169 / 0.1669 |

| | frozen | moving |
|---|---:|---:|
| predicted gap to true-future reference | +0.1195 | +0.1151 |
| predicted − persistence (fresh probes) | **+0.0489** | **+0.0639** |

### Reading: option 2, and option 3 is refuted

**Geometry is present but prediction leaves the encoder's canonical feature
basis.** Under the fixed true-future probe, predicted tokens scored 0.2654
(frozen) and 0.2304 (moving) and *lost* to persistence. Given a probe fitted to
their own output distribution they reach 0.3774 and 0.3818 and *beat* persistence
in both arms, including on `open_obstacle_field` for the frozen arm. Roughly
0.11–0.15 IoU of the apparent deficit was basis mismatch, not missing
information.

It is **not** wholly basis mismatch: both arms remain ~0.12 IoU below the
true-future reference, and their precision (0.4981 / 0.4890) is well below the
reference 0.6398. Prediction does genuinely degrade geometry — it just degrades
it far less than the fixed probe implied.

**Option 3 is refuted.** The two arms are indistinguishable on this axis:
predicted IoU 0.3774 vs 0.3818, gaps to reference +0.1195 vs +0.1151. **Encoder
movement does not destabilise predictor/target compatibility.** Whatever the
moving arm cost, it was not this.

Note the calibration caveat: all four fresh probes over-predict occupancy 3–4.7×
the target fraction (frozen persistence worst at 0.03315 against 0.00701), well
above the acceptance probe's ~2×. Part of their higher IoU is a looser operating
point, which is one more reason these probes are diagnostic only.

### Consequence for the next objective

An objective built solely on "the predictor discards geometry" would attack the
smaller half of the problem. The larger half is that the predictor's output does
not live in the encoder's canonical basis, so a fixed decoder trained on true
encoder tokens cannot read it. That points at output-space alignment — predicting
in the encoder's own basis, or constraining the predictor's output distribution
toward the target encoder's — **in addition to**, not instead of, an
action-difference term.

## What this establishes

**This closes the "maybe it was the objective" hypothesis.** WP-E's
encoder-moving failures were single-frame, unmasked, on a 2.7M task-trained
encoder. The available rebuttals were capacity, missing temporal context, and an
objective that rewarded contraction. This run removes all three — 304M pretrained
initialisation, genuine three-frame same-episode history, masked context—target
with an EMA target and a distinct future temporal position — **and the margin
still falls, in every scene.**

**The failure is isolated to action discrimination.** Not collapse (rank +14%),
not lost geometry (fresh-probe IoU and `open_obstacle_field` both rose), not
over-prediction (ruled out in §7), not capacity, not temporal context, not
masking.

**A second, independent problem surfaced.** §4 shows that neither arm's predicted
future decodes to better occupancy than persistence. The predictor is winning in
latent cosine while losing in the geometry that matters. Any future acceptance
test for this line should include the §4 comparison, because the latent metrics
alone would have passed a predictor that is spatially useless.

## Next

Unchanged from WP-E §6 and now better motivated: **an objective term that makes
action-conditioned futures differ**, not merely be predictable — latent-difference
action decoding (Delta-JEPA), or an action-contrastive term at the same current
state. Neither is authorised by this document.

Before attributing more to the objective, measure the corpus-side ceiling: the
nine primitives are coarse, `hold` is among them, and the 0.5 s horizon moves the
robot a median 0.08 m, so part of the residual margin may not be recoverable at
all.

**Do not add a geometry teacher on this evidence** — geometry did not regress.

## Corrections made during this work

- **Per-arm changed-token mask (this pass).** The first evaluation derived the
  threshold and mask separately per arm, scoring the arms on different token
  subsets. Corrected to a single frozen-derived mask; the moving margin moved
  +0.0479 → +0.0488 and the conclusion did not change.
- **HIP OOM in the battery** (8.63 GiB against 23.27 GiB held): the predictor was
  called on all 491 sequences at once, features were float32, and the threshold
  materialised the full train tensor. All three fixed; parity above confirms no
  measured quantity moved.
- **Wrong temporal blocker.** An earlier reading looked only at the v04 render
  and missed the dense v03 render on the 3.7 TB pool, which covers 100% of the
  corpus scenes.
- **Wrong analytic crop derivation.** The v03 crop was first derived as a 1.333×
  focal mismatch from the platform manifest's `native_resolution: [640,480]`. The
  pixels disproved it — crop-offset and scale sweeps both peak sharply where a
  shared focal length predicts. The empirical result governs.

---

# DECISION

> ## REJECT ENCODER-MOVING RECIPE
