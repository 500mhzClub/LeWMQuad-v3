# V21 same-action cross-scene contrastive innovation joint-JEPA preregistration

Date: 2026-07-30

Status: preregistered fresh scientific successor only. No V21 source root,
output root, reservation, generated-input access, GPU work, training,
checkpoint, calibration, G2, navigation, held-out, or sealed access has
occurred.

## Why V21 exists

- V20 is consumed and terminal. Its result is
  `docs/lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v20_scientific_result_2026-07-30.json`,
  commit `8321d76004aa1f3c87dfa04c3b18d701267a89ec`, file SHA-256
  `d76fd16732d15b7637bbe8f68df65ba23990046812f4ec3d85297f7f8ea64956`,
  byte count `17166`, and canonical content SHA-256
  `37f683c1b2a5086c92d9cb081e9ba55b4fef4ed61f8cefea99fb0e5760e5cab2`.
- V20 passed structural integrity and every matched update-400 physical
  threshold, but only six of twelve causal checks. Wrong-scene RGB and the
  train-action mean prior each had exactly zero advantage on all three
  registered statistics. V20 therefore learned useful RGB geometry and a live
  action path, but did not establish current-scene-by-action dynamics.
- V20's executed-successor semantic route was not disconnected: it had a
  nonzero predictor gradient in all 400 updates and beat persistence in 372.
  Its absolute reconstruction target was nevertheless dominated by static
  cells and could be reduced by an action-conditioned average transition.
- More V20 updates, coefficient tuning, retry, resume, or checkpoint reuse are
  rejected. V21 starts from the same fresh initialization and tests one
  materially different learning mechanism.

## Sole scientific change

V21 replaces V20's predictor-only absolute next-semantic objective `Q_succ`
with one predictor-only same-action, cross-scene contrastive latent-innovation
objective `I_scene`. It does not add a second auxiliary and does not change the
model architecture.

For each reviewed four-row train microbatch:

- `z[b]` is the online current latent, `p[b,a]` is the already-computed
  all-action prediction, `a[b]` is the executed action, and `zt0[b]` and
  `zt1[b]` are the stop-gradient EMA current and next latents. All latents have
  shape `(4,64,64,64)` after selecting an action.
- The launcher derives `negative_row[b]` only from the already-selected train
  pair metadata. It scans cyclic offsets one through three and selects the
  first row whose exact `scene_id` differs from row `b`. Every selected index
  must be in range, non-self, and different-scene. A microbatch containing no
  different-scene choice for any row fails closed before its tensors are used.
  The mapping need not be bijective. It is published in the batch as one
  `(4,)` integer tensor and is not a new data presentation.
  The localized V21 batch adapter inspects and validates these selected pair
  records before calling the inherited tensor builder, then adds the tensor as
  exact `torch.int64` on the batch device. V21 explicitly extends the otherwise
  frozen batch-key schema by this one field; all other keys and their order are
  unchanged.
- The positive and negative use the same requested action:
  `r_pos[b] = p[b,a[b]] - stopgrad(z[b])` and
  `r_neg[b] = p[negative_row[b],a[b]] - stopgrad(z[negative_row[b]])`.
  Subtracting each source latent removes the predictor's identity skip. An
  action-only residual therefore makes `r_pos` and `r_neg` equal in exact
  arithmetic and cannot earn material contrastive advantage. Float32
  subtraction residue is treated as zero within `1e-6` in focused tests.
- The learned JEPA target is
  `d[b] = stopgrad(zt1[b] - zt0[b])`. Labels, semantic logits, geometry labels,
  and evaluator outputs do not select or define this target.
- The frozen V18 `model.bev_lift.cell_valid_mask` defines the only eligible
  cells. It is detached and must contain at least 256 cells. Invalid cells do
  not enter the scale, salience ordering, or either energy.
- A detached scale
  `s[b] = max(sqrt(mean_valid(d[b]**2)), 1e-3)` is shared by the positive and
  negative. Per-cell positive and negative errors are the channel mean of
  Smooth-L1 with beta one between `r/s` and `d/s`.
- Target salience is the detached channel mean of `(d/s)**2` on valid cells.
  A stable ascending sort with flattened cell index as the tie-break selects
  exactly the 128 highest-salience and 128 lowest-salience valid cells per row;
  the sets must be disjoint. Each
  energy is one half the mean error on the high set plus one half the mean
  error on the low set. This gives the approximately three-percent changing
  region equal weight with an equally sized stable control without using an
  oracle mask.
- `I_fit = mean(E_pos)` and
  `I_rank = mean(softplus(E_pos - E_neg)) / log(2)`. The registered objective
  is `I_scene = I_fit + I_rank`. An action-only tie has rank value exactly one;
  the fit term prevents success by merely inflating the negative error.

The positive and negative are gathered from the one inherited all-action
prediction tensor. V21 adds no encoder pass, target-encoder pass, predictor
forward, RGB read, label read, or presentation.

## Gradient and accounting boundary

- `I_scene` is differentiated only with respect to the same exact 13 tensors
  and 259008 parameters of the latent-transition predictor used by V20. Its
  independent gradient route retains the exact unit-norm cap.
- The source latent subtraction, EMA innovation target, salience masks, scale,
  and negative-row indices are detached. `I_scene` sends no gradient to the
  encoder, BEV lift, representation, semantic head, evidence heads, survival
  head, swept-progress head, or EMA target.
- The inherited joint `P` objective remains in `N` and continues to train the
  online encoder, representation, and predictor together against the same EMA
  target in every update. V21 is joint JEPA training, not a separately trained
  predictor.
- The route is renamed `scene_innovation_predictor`. Each completed update
  remains exactly four microbatch graphs, twelve autograd calls, four calls for
  each of the camera, joint, and scene-innovation routes, four predictor
  forwards, eight predictor objectives, 32 camera-frame objectives, 16 ordered
  presentations, one optimizer step, and one EMA step.
- The loss receipt is `L = N + C + I_scene`; all inherited `S/P/U/R/O/C`
  definitions and coefficients remain unchanged.

## Frozen identity

Except for `I_scene`, its train-metadata negative-row field, V21 receipt names,
diagnostics, gates stated below, and lifecycle bindings, V21 preserves V20
exactly:

- the learned RGB encoder, eight-height object-space volume, semantic and
  survival heads, local action-conditioned predictor, parameter counts, and
  initialization;
- N320 initialization, constructor seed `20260712`, projection seed
  `20260729`, experiment seed `20260728`, schedule and bootstrap seeds, float32
  AdamW settings, learning rate, EMA, route clipping, and parameter groups;
- the 4262-pair schedule from presentation zero, four microbatches of four,
  train and checkpoint-selection roles, source files, labels, camera metadata,
  observations, eight-family registry, physical metrics, and causal-control
  definitions; and
- maximum 1000 updates and 16000 ordered presentations.

V21 must start in a new process from exact initialization. No V20 model,
optimizer, EMA, RNG, schedule, trace, metric, receipt, checkpoint, or mutable
runtime state may be opened or reused.

## Focused source acceptance

- A pure tensor test proves an action-only residual keeps positive and negative
  innovations and energies equal within `1e-6`, scene advantage within `1e-6`
  of zero, and rank value within `1e-6` of one. A scene-conditioned fixture
  must produce advantage greater than `1e-4`.
- Gather tests prove the positive and negative use exactly `a[b]`, remove their
  respective identity skips, and add no predictor forward.
- Negative-row tests prove exact cyclic selection, non-self different-scene
  provenance, pre-tensor metadata validation, tensor shape `(4,)`, exact int64
  dtype and device, the one-field batch-schema extension, and rejection of
  malformed or all-one-scene microbatches.
- Salience tests prove exactly 128 high and 128 low valid cells, disjoint
  detached masks, stable flattened-index tie-breaking, rejection of fewer than
  256 valid cells, and expected selection of a synthetic changed cell.
- One real CPU synthetic update proves a finite nonzero exact 13-tensor
  scene-innovation route, zero target gradient, unchanged joint routes, exact
  accounting, one optimizer step, and one EMA step.
- Existing inherited model, evaluator, comparison, control, custody, and
  source-closure tests must remain passing. Recursive closure, independent
  review, narrow clean-export certification, and separate one-shot authority
  must be committed before reservation.

## Falsification and scale gates

- Update 0 is informational and must pass structural, source, access, custody,
  and target-isolation integrity.
- Each update trace publishes `sum(E_neg - E_pos)`, exact contributing row
  count `16`, and their arithmetic mean for that update. Observation updates
  report that update's mean; the statistic is diagnostic and is not silently
  accumulated across updates.
- Update 100 remains informational with mandatory structural, accounting,
  finite-gradient, target-isolation, access, and custody integrity. Its
  scene-innovation advantage and all causal controls are recorded without a
  new terminal branch.
- Update 400 retains the already implemented V20 gate exactly: structural
  integrity, all twelve causal-control checks, physical margin count strictly
  greater than `72`, total physical shortfall strictly below
  `68.96954700805838`, and rough depth p95 strictly below
  `1.8582415819168085` m. The diagnostic scene-innovation advantage must be
  finite; the registered wrong-RGB and action-prior controls remain the
  decisive shortcut test. Any gate failure is terminal.
- Passing update 400 continues in the same process; it does not restart the
  first 400 updates.
- Update 1000 passes only with the inherited V12 full arm `24/24`, at least
  `112/189` physical margins, total shortfall strictly below
  `33.05143763708337`, at least one complete physical scope, rough pixel
  balanced accuracy strictly above `0.8198594673963917`, rough ground balanced
  accuracy strictly above `0.647134926562893`, rough depth p95 strictly below
  `0.9777327477931971` m, and structural integrity. The inherited V12 full arm
  already contains the registered causal checks.
- Only a complete update-1000 pass publishes the development model checkpoint.
  It does not authorize calibration, G2, navigation, held-out, sealed,
  promotion, production, or deployment access.

## One-shot boundary

- Schema/evidence prefix:
  `lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21`.
- Fresh output root:
  `.generated/go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21/attempt_v1`.
- Fresh clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v21-scene-innovation-source`.
- Both roots must initially be absent. There is one attempt and no retry,
  resume, recovery, coefficient search, or same-mechanism replacement.
- Authority covers only the exact reviewed source plus frozen train and
  checkpoint-selection inputs on the reviewed runtime and hardware.
- Until a complete update-1000 pass, probability calibration, G2, navigation,
  held-out, sealed, promotion, production, and deployment remain forbidden.
