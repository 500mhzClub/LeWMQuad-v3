# V22 scene-action contrastive innovation joint-JEPA preregistration

Date: 2026-07-30

Status: preregistered fresh scientific successor only. No V22 source root,
output root, reservation, generated-input access, GPU work, training,
checkpoint, calibration, G2, navigation, held-out, or sealed access has
occurred.

## Why V22 exists

- V21 is consumed and terminal. Its committed result is
  `docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21_scientific_result_2026-07-30.json`,
  commit `e5b5e56b30cee0c1eb818d52c4d886909f570f4d`, file SHA-256
  `c9544055b11d162b5b5fc9b02d0a04f3961a61b4547411964812a9ae4c5da1e7`,
  byte count `15724`, and canonical content SHA-256
  `2195025bf24e3de621e76a5a5e3ea272ced05bd9f6e4fb91302035137ab7b9ec`.
- V21 passed structural integrity and all three registered update-400 physical
  thresholds, but only four of twelve causal checks. It beat persistence while
  shuffled actions and the train-action mean prior beat the requested action.
  Wrong-scene RGB gained only one weak check.
- The V21 auxiliary route was connected and successfully optimized its local
  objective: its scene advantage was positive on 379 of 400 updates and rose
  from `0.0026668235659599304` at update 100 to
  `0.09704933129251003` at update 400. That objective compared scenes only
  while reusing the same action, so scene-specific but action-agnostic
  evolution could satisfy it.
- V22 is the evidence-led correction named by the V21 terminal result: require
  the positive to win along both the scene and requested-action axes. It is a
  new mechanism from fresh initialization, not a V21 retry, resume, extension,
  checkpoint continuation, or coefficient search.

## Sole scientific change

V22 replaces V21's predictor-only objective `I_scene = I_fit + I_scene_rank`
with one predictor-only two-axis objective
`I_two_axis = I_fit + 0.5 * (I_scene_rank + I_action_rank)`. It does not add a
second auxiliary objective and does not change the model architecture.

For each reviewed four-row train microbatch:

- `z[b]` is the online current latent, `p[b,a]` is the already-computed
  all-action prediction tensor, `a[b]` is the requested/executed action, and
  `zt0[b]` and `zt1[b]` are the stop-gradient EMA current and next latents.
- The positive residual remains
  `r_pos[b] = p[b,a[b]] - stopgrad(z[b])`.
- The scene-negative row is selected exactly as in V21: scan cyclic offsets
  one through three and take the first row whose exact `scene_id` differs from
  row `b`. Every index must be in range, non-self, and different-scene, and a
  microbatch lacking a valid choice fails closed before tensor use. The
  scene-negative residual remains
  `r_scene[b] = p[negative_row[b],a[b]] - stopgrad(z[negative_row[b]])`.
- For every action `a'` other than `a[b]`, the same-row action-negative
  residual is
  `r_action[b,a'] = p[b,a'] - stopgrad(z[b])`. Thus the action contrast keeps
  the current scene fixed and changes only the requested action. It uses all
  and only non-requested actions already present in `p`; it does not sample a
  new row or call the predictor again.
- The learned target remains the detached EMA innovation
  `d[b] = stopgrad(zt1[b] - zt0[b])`. Labels, semantic logits, geometry labels,
  and evaluator outputs do not select or define it.
- The frozen V18 `model.bev_lift.cell_valid_mask` remains the only eligible
  cell set. The detached target scale, stable salience ordering, and exactly
  128 highest- plus 128 lowest-salience valid cells are unchanged from V21;
  the two sets must be disjoint and fewer than 256 valid cells fails closed.
- Let `E(r,d)` be V21's normalized target-matching energy: the detached scale
  is `s[b] = max(sqrt(mean_valid(d[b]**2)), 1e-3)`; per-cell error is the
  channel mean of beta-one Smooth-L1 between `r/s` and `d/s`; and the row
  energy is one half the mean error on the high-salience cells plus one half
  the mean error on the low-salience cells. The same row target, scale, and
  high/low cells are used for its positive and both negative axes.
- Define `E_pos[b] = E(r_pos[b],d[b])` and
  `E_scene[b] = E(r_scene[b],d[b])`. The action-negative energy is the
  arithmetic mean of the normalized target energies over all non-requested
  actions,
  `E_action[b] = mean_{a' != a[b]} E(r_action[b,a'],d[b])`.
  The energies are averaged after evaluation; alternate residuals are not
  averaged before the loss, so opposite action errors cannot cancel.
- `I_fit = mean_b(E_pos[b])`,
  `I_scene_rank = mean_b(softplus(E_pos[b] - E_scene[b])) / log(2)`, and
  `I_action_rank = mean_b(softplus(E_pos[b] - E_action[b])) / log(2)`.
  The registered auxiliary is exactly
  `I_two_axis = I_fit + 0.5 * (I_scene_rank + I_action_rank)`.
  A tie on either axis has that rank term exactly one; `I_fit` prevents success
  by merely inflating negative errors.

The positive, scene negative, and every action negative are gathered from the
single inherited all-action prediction tensor. V22 adds no encoder pass,
target-encoder pass, predictor forward, RGB read, label read, presentation,
microbatch graph, or gradient call.

## Gradient and accounting boundary

- `I_two_axis` is differentiated only with respect to the same exact 13
  tensors and 259008 parameters of the latent-transition predictor used by
  V21. Its independent gradient route retains the exact unit-norm cap.
- The source-latent subtractions, EMA innovation target, salience masks,
  scale, negative-row indices, and action-index sets are detached.
  `I_two_axis` sends no gradient to the encoder, BEV lift, representation,
  semantic head, evidence heads, survival head, swept-progress head, or EMA
  target.
- The inherited joint `P` objective remains in `N` and continues to train the
  online encoder, representation, and predictor together against the same EMA
  target in every update. V22 remains joint JEPA training, not a separately
  trained predictor.
- The auxiliary route is renamed `two_axis_innovation_predictor`. Each
  completed update remains exactly four microbatch graphs, twelve autograd
  calls, four calls for each camera, joint, and auxiliary route, four
  predictor forwards, eight predictor objectives, 32 camera-frame objectives,
  16 ordered presentations, one optimizer step, and one EMA step. The two
  rank terms are components of one auxiliary objective per microbatch.
- The loss receipt is `L = N + C + I_two_axis`; all inherited `S/P/U/R/O/C`
  definitions and coefficients remain unchanged.

## Frozen identity

Except for `I_two_axis`, its action-axis calculations, corresponding receipt
names and diagnostics, and V22 lifecycle bindings, V22 preserves V21 exactly:

- the learned RGB encoder, eight-height object-space volume, semantic and
  survival heads, local action-conditioned predictor, parameter counts, and
  initialization;
- N320 initialization, constructor seed `20260712`, projection seed
  `20260729`, experiment seed `20260728`, schedule and bootstrap seeds,
  float32 AdamW settings, learning rate, EMA, route clipping, parameter
  groups, and inherited joint-JEPA losses;
- the 4262-pair schedule from presentation zero, four microbatches of four,
  train and checkpoint-selection roles, data and source files, labels, camera
  metadata, observations, eight-family registry, physical metrics, causal
  controls, and all gate thresholds; and
- maximum 1000 updates and 16000 ordered presentations.

V22 must start in a new process from exact initialization. No V21 model,
optimizer, EMA, RNG, schedule state, trace, metric, receipt, checkpoint, or
mutable runtime state may be opened or reused. V21 published no checkpoint.

## Focused source acceptance

- Pure tensor tests prove that the positive and scene negative preserve V21's
  same-action, different-scene construction and that the action negative uses
  the same row/current latent and all and only non-requested actions.
- Energy tests prove that `E_action` is the arithmetic mean of independently
  normalized target energies after evaluation, with the same detached target,
  scale, valid mask, and high/low cells as `E_pos`; the requested action is
  excluded exactly.
- Synthetic fixtures prove a scene-only solution cannot earn action advantage,
  an action-only solution cannot earn scene advantage, and a jointly
  scene-and-action-conditioned positive can earn both advantages.
- Gather tests prove the two axes reuse exactly one all-action prediction
  tensor and add no predictor forward, RGB presentation, or data access.
- One real CPU synthetic update proves a finite nonzero exact 13-tensor
  auxiliary route, zero target gradient, unchanged inherited joint routes,
  one auxiliary objective per microbatch, exact accounting, one optimizer
  step, and one EMA step.
- Existing inherited model, evaluator, comparison, causal-control, custody,
  and source-closure tests must remain passing. Recursive closure, independent
  review, narrow clean-export certification, and separate one-shot authority
  must be committed before reservation or execution.

## Falsification and scale gates

- Update 0 is informational and must pass structural, source, access, custody,
  accounting, and target-isolation integrity.
- Each update trace publishes, separately for the scene and action axes,
  `sum(E_negative - E_pos)`, exact contributing row count `16`, and the
  arithmetic mean for that update. These are diagnostics and are not silently
  accumulated across updates.
- Update 100 is informational. Both scene-innovation and action-innovation
  advantages, both rank terms, and all inherited causal controls are recorded.
  Mandatory structural, accounting, finite-gradient, target-isolation,
  access, and custody integrity still applies, but neither axis introduces a
  new update-100 terminal threshold.
- Update 400 retains the inherited gate exactly: structural integrity, all
  twelve causal-control checks, physical margin count strictly greater than
  `72`, total physical shortfall strictly below `68.96954700805838`, and rough
  depth p95 strictly below `1.8582415819168085` m. Both axis diagnostics must
  be finite, but they add no replacement threshold or branch. Any inherited
  gate failure is terminal.
- Passing update 400 continues in the same process; it does not restart the
  first 400 updates.
- Update 1000 retains the inherited gate exactly: V12 full arm `24/24`, at
  least `112/189` physical margins, total shortfall strictly below
  `33.05143763708337`, at least one complete physical scope, rough pixel
  balanced accuracy strictly above `0.8198594673963917`, rough ground balanced
  accuracy strictly above `0.647134926562893`, rough depth p95 strictly below
  `0.9777327477931971` m, and structural integrity. The V12 full arm already
  contains the registered causal checks.
- Only a complete update-1000 pass publishes the development checkpoint. No
  checkpoint is written or retained at update 100 or 400, and no incomplete
  or failed run may publish one.

## One-shot and protected-access boundary

- Schema/evidence prefix:
  `lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22`.
- Fresh output root:
  `.generated/go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22/attempt_v1`.
- Fresh clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v22-scene-action-innovation-source`.
- Both roots must initially be absent. There is one attempt and no retry,
  resume, recovery, extension, coefficient search, integrity replacement, or
  same-mechanism replacement.
- Authority covers only the exact independently reviewed source plus frozen
  train and checkpoint-selection inputs on the reviewed runtime and hardware.
- Until a complete update-1000 pass, probability calibration, G2, navigation,
  held-out, sealed, promotion, production, and deployment remain forbidden.
  This preregistration itself grants none of those accesses and grants no
  source export, data, GPU, training, reservation, or execution authority.
