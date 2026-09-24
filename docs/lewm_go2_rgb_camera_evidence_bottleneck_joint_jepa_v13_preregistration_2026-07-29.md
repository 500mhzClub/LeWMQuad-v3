# RGB Camera-Evidence-Bottleneck Joint JEPA V13 — Preregistration

- Date: 2026-07-29, after the terminal V12 physical result and before V13
  source, scientific-input access, accelerator work, or model output.
- Parent result: V12 physical failure receipt commit
  `2fde0fffbd3596600cdc057be54a3ab0e67fb90e`.
- Status: one fresh, capped falsification of one coherent scientific package
  is selected. This record grants only source implementation, focused
  source-only tests, and independent source review. A later committed
  execution binding is necessary but not sufficient for execution. Before
  that binding, V13 also requires a committed recursive source-closure
  manifest, independently reviewed exact path/SHA-256/byte-count bindings, and
  a repository-custody-approved narrow clean-export exception plus
  certification covering only those paths. Every exported byte must validate
  against the frozen source-and-review commit and certification before copy.
- Current authority explicitly denies opening the Raw-V13 generated-input root
  or any dataset tensor; opening the N320 gate, checkpoint, or tensor state;
  RGB decode; calibration or label reads; probability-calibration or
  checkpoint-selection evaluation; accelerator query; training; checkpoint or
  metric generation; physical calibration; clean export; retry; resume; G2;
  navigation; held-out or sealed access; deployment; and promotion.

## Decision and diagnosis

- V12 passed its complete `24/24` development gate but failed the frozen
  physical calibration: `0/2,016` feasible tuples, FREE precision `0.961184`,
  near-obstacle detection `0.573632`, useful FREE recall `0.769442`, and
  near-obstacle exclusion `0.849022`.
- Calibration improved NLL and Brier score but could not repair the ordering
  overlap between safe floor, UNKNOWN, and nearby obstacles. The next change
  must improve learned physical evidence, not thresholds or semantic algebra.
- Camera V6 previously showed that RGB contains strong direct evidence:
  aggregate pixel first-hit balanced accuracy was `0.977179` and ground-clear
  balanced accuracy was `0.974710`. Its parallel JEPA branch could ignore that
  evidence, its joint predecessor starved JEPA gradients by approximately
  `8.38e7`, and its final soft and hard raster compositions both failed.
- V13 therefore makes pre-composition Camera evidence the sole shared BEV
  state used by semantics, prediction, survival, and the EMA target. It does
  not restore the failed Camera raster and has no direct-token BEV bypass.

## History-aware novelty boundary

- Do not implement a two-frame correspondence or temporal mechanism. Local
  correspondence V7/V8, dense all-pairs cost volume V9, target retrieval,
  masked-pair tubelets V11-V13, temporal/alignment probes, rigid BEV transport,
  recurrent H4, and dense spatiotemporal H4 have already tested that family.
- Shared V5 is not this mechanism: it emitted Camera evidence and an unrelated
  learned BEV in parallel, while its predictor consumed only the unrelated
  BEV. Camera V6 then trained Camera components with JEPA backward and EMA
  counts fixed at zero.
- The old final evidence composition is closed. V13 must not multiply the 20
  ground supports into one FREE probability, union all 64 depth bins into one
  OCCUPIED probability, apply `FREE * (1-OCCUPIED)`, or feed the old soft or
  hard ternary raster to its predictor.
- V13 is one inseparable three-part scientific package, not three selectable
  ablations: (1) the retained Camera-evidence bottleneck, (2) fine-evidence-only
  Camera supervision `C`, and (3) objective-route gradient balancing so `C`
  cannot starve the joint objective `N` or vice versa. No other model, loss,
  data, schedule, seed, optimizer, threshold, or evaluation change is allowed.

## Frozen RGB-only model

- Input at inference is exactly one normalized current `112x112` RGB image.
  Executed action is accepted only by the predictor. No depth, pose, attitude,
  odometry, per-row camera calibration, ground-plane tensor, map, goal, label,
  future RGB, previous RGB, or recurrent state may enter the inference graph.
- Reuse the patch-seven `192/6/6` VisionEncoder and the accepted N320
  encoder/evidence-head initialization only. No Shared V5, Camera V6, V4-V12
  experiment, or other rejected checkpoint, trace, or state may be read or
  reused. N320 access remains denied until a later execution binding.
- Reuse `ObservableCameraRayEvidenceV4Head`: a shared dense image decoder
  predicts 64 ordered first-hit hazards and bounded within-bin offsets on each
  of the `84x112` rays plus five ground-clear logits for every cell on the
  `128x128` canonical source lattice.
- Register immutable nominal geometry buffers: camera origin
  `(0.326,0,0.043)` m; forward/right/up rows
  `((1,0,0),(0,-1,0),(0,0,1))`; ground `z=-0.333` m; horizontal/vertical FOV
  `78.323/62.8370386364` degrees; and near plane `0.05` m.
- The public online and target encode APIs accept RGB only. A separate
  development-supervision-only auxiliary-evidence method may accept the bound
  Raw-V13 per-frame camera origin, basis, and scalar ground height solely to
  align fine Camera supervision. Dense features are decoded once; nominal and
  auxiliary ground queries may share those features. Auxiliary output may
  feed only `C` and explicitly labelled diagnostics. It may not feed the
  bottleneck latent, semantic output, predictor, survival head, EMA target,
  causal controls, navigation, or deployment. Changing auxiliary geometry
  while holding RGB fixed must leave the nominal evidence planes and RGB-only
  bottleneck latent bit-identical.

## Frozen pre-composition evidence state

### FREE role

- Evaluate the ground branch under the fixed nominal geometry used by the
  RGB-only state.
- For each `64x64` output cell, retain the ordered four `2x2` source subcells
  and five support queries per subcell. Flatten in source-row delta,
  source-column delta, then support-index order.
- Form exactly 40 channels: 20 raw ground-clear logits with invalid entries
  replaced by exact zero, followed by the corresponding 20 fixed nominal
  in-frustum bits cast to the logit dtype.
- Project `40 -> 32` with one learned biased `1x1` convolution and GELU. There
  is no support product, threshold, union, attention, or extra residual block.

### OCCUPIED role

- Convert the 64 hazards with
  `ordered_obstacle_first_hit_log_probabilities_v4`; retain its depth order.
  Bin centres are `0.05 + (d+0.5)*0.10` m for `d=0..63`, and the inherited
  offset is exactly `0.05*tanh(raw_offset)`, hence lies in `(-0.05,0.05)` m.
- Form body-frame rays with `calibrated_pixel_ray_directions_torch_v4` and the
  fixed nominal basis. For every depth-ray point, use the V4 coordinates
  `row=(forward-(-1.0))/0.10-0.5` and
  `column=(left-(-3.2))/0.10-0.5`. The inclusive geometric extent is
  `[-0.5,63.5]` on both axes.
- Floor row and column, then visit candidate deltas in exact order `(0,0)`,
  `(0,1)`, `(1,0)`, `(1,1)`. A candidate is valid only when the point is in
  the inclusive extent and the candidate index is in `[0,63]`. Renormalize
  the four bilinear weights over valid candidates with
  `torch.finfo(float32).eps`; invalid contributions are exact zero.
- A per-ray, per-depth bilinear splat conserves that hit probability over its
  valid cells. Distinct rays within the same depth plane combine as
  `1-product_r(1-contribution_r)` using `log1p`/`expm1`; this union is not
  described as mass-conserving. Before `log1p`, clamp each assembled
  per-ray/per-depth/per-cell contribution to
  `[0,1-torch.finfo(float32).eps]`. Process flattened rays in ascending order
  in chunks of exactly 256 and candidate deltas in the order above. The log
  no-hit accumulator has shape `(B,64,4096)`, so no depth index is ever summed
  or unioned with another depth index.
- The resulting ordered 64 per-depth planes are the complete OCCUPIED input.
  Project `64 -> 32` with one learned biased `1x1` convolution and GELU. There
  is no across-depth union, threshold, old rasterizer, attention, or extra
  residual block.

### Validity, shared state, and JEPA

- `free_cell_valid_mask` is the batch-independent Boolean OR of the 20 fixed
  nominal ground-query validity bits for each output cell.
- `occupied_cell_valid_mask` is a frozen conservative nominal support mask.
  Re-run the exact candidate-validity calculation above without learned hit
  values for every nominal ray and depth at offsets `-0.05`, `0.0`, and
  `+0.05` m, and OR every valid candidate into a `64x64` Boolean mask. It is
  computed once from registered buffers, does not depend on RGB or a per-row
  calibration, and is not a V10/V11 volume mask.
- Concatenate FREE then OCCUPIED projections into the sole `(B,64,64,64)`
  state. The two projections add exactly `3,392` trainable parameters. They
  use one private CPU generator seeded `20260729`. In exact order FREE weight,
  FREE bias, OCCUPIED weight, OCCUPIED bias, initialize each weight with
  `torch.nn.init.xavier_uniform_(weight,gain=1.0,generator=generator)` and each
  bias with exact zero. Thus the FREE weight bound is
  `sqrt(6/(40+32))` and the OCCUPIED weight bound is `sqrt(6/(64+32))`.
  Construction restores the caller RNG state.
- Reuse V12's neutral UNKNOWN/FREE/OCCUPIED readout algebra, semantic head,
  action-conditioned local predictor, and swept-progress survival head, but
  use only the two V13 masks above. An invalid role logit is exact `-20`; the
  UNKNOWN logit is exact zero; a cell invalid for both roles therefore has
  logits `[0,-20,-20]` in class order UNKNOWN, FREE, OCCUPIED.
- The predictor consumes and predicts only the V13 state. No patch-token,
  V10/V12 lift, Camera-raster, semantic-label, or hidden-feature bypass exists.
- After accepted N320 state migration and projection construction, deep-copy
  the complete migrated online V13 encoder/evidence/projection bottleneck to
  its target, freeze it, put it in eval mode, and assert exact state identity.
  This is the one accounted hard sync. Any predecessor target built and then
  discarded during subclass construction is excluded from final accounting;
  the final receipt must read hard-sync count `1`, EMA count `0` before
  training. Momentum is `0.996`, one EMA update follows every optimizer step,
  the target remains in eval mode, and it receives no gradient.

## Frozen data and supervision

- A later binding may authorize only the immutable Raw-V13 roles and pair
  identities: 4,262 training pairs in 72 scenes, 415 probability-calibration
  pairs, and 495 checkpoint-selection pairs. Reuse the exact swept-progress
  label bundle, endpoint mapping, action vocabulary, negatives, role order,
  schedule prefix, and selection populations used by V12.
- Do not render, rebuild, relabel, filter, resample, or add data. The eventual
  narrow V13 loader may admit exactly RGB, `raster_labels.u1`,
  `pixel_hit_mask.u1`, `pixel_first_hit_distance_m.f4`,
  `ground_support_in_frustum.u1`, and
  `ground_support_clear_to_target.u1`. Per-row camera origin, basis, and scalar
  ground arrays are permitted only as target context for the auxiliary method
  and the metric adapter described below.
- Semantic, action, progress, survival, and physical-evidence labels remain
  unavailable to the inference graph. All supervision arrays are targets.

## Frozen objective, gradients, and optimizer

- Preserve V12's exact jointly trained objective
  `N = S + P + U + R + O`, including current/next semantic grounding,
  action-conditioned latent prediction against the stop-gradient EMA target,
  survival, progress ranking, and occupied auxiliary coefficient `0.5`.
- Add one fine Camera term `C`. For each individual frame, invoke exactly
  `hierarchical_first_hit_nll_breakdown_v9(...).total`,
  `_skew_balanced_pixel_offset_loss_v5`, and
  `balanced_ground_clear_bce_v4`, and take their arithmetic mean. Average the
  four B=1 frame values within a microbatch, then average current and next
  frames `0.5/0.5`. The ground loss retains distance edges
  `(0,1,2,3,4,5,+inf)` m. The failed derived soft/hard raster losses are
  absent.
- Those three inherited functions are bound at the parent commit by source
  SHA-256 values `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd`
  (`observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py`),
  `ee3fd612bb0a40d615fda7f7110091a330c849f8b1ec2b48cc0af3e406c928fc`
  (`shared_observable_camera_ray_jepa_v5.py`), and
  `c0f3f944883987950edb7579a9e108171486122a9a3ae9d84d2a1abb6ac015ed`
  (`observable_camera_ray_evidence_v4_training.py`), respectively.
- The shared route is exactly `encoder.* + bev_lift.evidence_head.*`. Across
  the four microbatches, separately accumulate gradients of `C/4` and `N/4`
  for this route; an absent tensor gradient is an exact zero tensor. For each
  complete route compute `norm=sqrt(sum_tensor(sum(g*g)))` in float32 and
  `scale=min(1.0,1.0/max(norm,torch.finfo(float32).tiny))`. The installed shared
  gradient is `scale_C*g_C + scale_N*g_N`, with no post-sum or global clip.
- Separately accumulate `N/4` and apply the same norm-one rule to exactly
  `bev_lift.free_projection.* + bev_lift.occupied_projection.* +
  semantic_head.*`, and separately to exactly `predictor.*`, including the
  survival head. The three named online groups must be disjoint and cover
  every trainable parameter; the target is excluded. There is no other clip.
- Every update must record finite, nonzero complete Camera and joint-JEPA
  shared-route norms, a finite nonzero predictor-group norm, every pre-clip
  norm and applied scale, and zero target gradients. A missing route is a
  structural failure, not a trainable result.
- Use one AdamW optimizer, float32, betas `(0.9,0.999)`, epsilon `1e-8`, weight
  decay `1e-4`, encoder learning rate `1e-4`, and every other online parameter
  learning rate `3e-4`. There is exactly one optimizer step and one EMA update
  per completed update. No coefficient sweep or alternate optimizer is
  allowed.

## Seeds, schedule, cap, and attempt custody

- Freeze inherited model/constructor seed `20260712`, schedule seed
  `20260713`, stochastic execution seed `20260728`, projection seed
  `20260729`, and bootstrap seed `20260728`.
- The ordered schedule-prefix SHA-256 values are exactly update 100
  `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`,
  update 400
  `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`,
  and update 1,000
  `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.
- There is one seed and at most one attempt. Its maximum is 1,000
  optimizer/EMA updates, four `B=4` microbatches per update, and 16,000 ordered
  presentations. Observe immutable metrics at updates `0`, `100`, `400`, and
  `1,000`; they are not selectable checkpoints and update 1,000 is the only
  promotable state.
- Continued attempts run exactly 1,000 updates. If the frozen update-400
  directional gate fails, stop after update 400, publish a terminal scientific
  failure receipt, and close the attempt. That early stop is not a retryable
  partial run.
- The exact output root is
  `.generated/go2_rgb_camera_evidence_bottleneck_joint_jepa_v13/attempt_v1`.
  Only the future custody-certified narrow export may host execution. A later
  launcher must require the output root to be absent, then atomically create
  it as mode `0700` and publish an immutable reservation before opening
  scientific payload. Reservation consumes the sole attempt. Every exception,
  access denial, integrity failure, or scientific failure after reservation
  is terminal and must leave an immutable success/failure receipt; it grants
  no replacement, repair, retry, or resume. A no-payload preflight completed
  before reservation does not consume the attempt.

## Frozen development evaluators and gates

- Structural integrity is conjunctive: fixed RGB-only latent identity, no
  bypass, exact parameter/state/optimizer/EMA accounting, finite state, active
  Camera and JEPA routes, noncollapsed target, wrong-RGB dependence, and zero
  forbidden input or target-gradient counts.
- The established physical metric adapter is bound to the tracked parent
  sources `scripts/run_go2_shared_jepa_v5_matched_training_v1.py` SHA-256
  `e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578`,
  `lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py` SHA-256
  `6a0e40f9dcb496831553dc5bbc6d1efcdf6d82676d6f18aa20e417f8de4fa6a0`,
  and `lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py` SHA-256
  `53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a`.
- Preserve the inherited physical threshold maps exactly. Ordered lower
  thresholds are pixel first-hit BA `0.95`, ground-clear BA `0.95`, raster BA
  `0.95`, wrong-RGB pixel BA drop `0.12`, wrong-RGB depth median-error increase
  `0.12` m, wrong-RGB depth p95-error increase `0.20` m, wrong-RGB ground BA
  drop `0.12`, wrong-RGB raster NLL increase `0.12`, and wrong-RGB raster BA
  drop `0.12`. Ordered upper thresholds are depth median error `0.10` m, depth
  p95 error `0.25` m, and raster NLL `0.15`. Every distance-band BA threshold
  is `0.92` and every present-class recall threshold is `0.95`.
- Evaluate all 924 unique checkpoint-selection endpoints. In each registered
  family, sort endpoint SHA-256 IDs; the wrong-RGB mapping is the exact cyclic
  one-step rotation `ids[1:]+ids[:1]`. Both the matched and wrong arms use the
  target endpoint's targets and per-row calibration only as evaluator context,
  while the model receives only the selected RGB through its nominal RGB-only
  path.
- Pixel hit/no-hit logits, depth-bin choice/offset, and ground-clear logits for
  both arms must be the nominal RGB-only evidence outputs. To satisfy the
  inherited accumulator schema without feeding calibration to the model, the
  adapter copies only target-derived ground-validity and target-distance
  metadata onto those nominal logits after model execution. Per-row auxiliary
  logits are diagnostic only and cannot supply any qualifying metric.
- Raster metrics use V13 neutral semantic probabilities directly in exact
  class order UNKNOWN, FREE, OCCUPIED. The old V4 soft/hard Camera raster is
  never constructed. Targets, target masks, hit threshold `0.5`, argmax depth,
  ground-logit threshold `0`, six ordered distance bands
  `[0,1),[1,2),[2,3),[3,4),[4,5),[5,+inf)`, and all accumulator arithmetic
  otherwise remain unchanged.
- Preserve exact scope order: `aggregate`, `large_enclosed_maze`,
  `local_composite_motifs`, `loop_alias_stress`, `medium_enclosed_maze`,
  `open_obstacle_field`, `rough_local_dynamics`, `small_enclosed_maze`, and
  `visual_sensor_stress`. Each scope yields exactly 21 ordered physical
  margins: the nine inherited lower-threshold metrics in dictionary order,
  the three inherited upper-threshold metrics in dictionary order, the six
  distance-band balanced accuracies above, then present-class recalls sorted
  by class name (`free`, `occupied`, `unknown`). Require exactly 189 finite
  margins; a missing class, group, scope, or nonfinite value is structural
  failure.
- Continue past update 400 only if update `100 -> 400` strictly increases the
  count of nonnegative physical margins, strictly reduces total physical
  shortfall, improves at least two of rough pixel balanced accuracy, rough
  ground balanced accuracy, and rough depth p95 in their correct directions,
  and all twelve causal-control checks below are true. Otherwise close at 400.
- For each of `coordinate_matched_persistence`, `shuffled_action`,
  `wrong_rgb`, and `train_action_mean_prior`, the required update-400 keys are
  exactly `positive_equal_scene_delta`, `positive_bootstrap_lower_95`, and
  `positive_family_count`. All twelve must be true.
- At update 1,000 require all unchanged V12 `24/24` development checks. Also
  require at least `112/189` nonnegative physical margins, total shortfall
  strictly below `33.05143763708337`, at least one complete physical scope,
  rough pixel balanced accuracy strictly above `0.8198594673963917`, rough
  ground balanced accuracy strictly above `0.647134926562893`, and rough depth
  p95 strictly below `0.9777327477931971` m. Equality fails.
- Development scoring may use only training-derived action priors and the
  checkpoint-selection role. Probability-calibration remains closed until a
  complete development pass. A pass earns only a separately preregistered and
  reviewed V13-specific physical-evidence calibration adapter. The V12 adapter
  is checkpoint/schema-bound and cannot load V13; only its role separation,
  2,016-tuple grid, and four fixed selection gates may be inherited: FREE
  precision `>=0.99`, near-obstacle detection `>=0.95`, useful FREE recall
  `>=0.90`, and near-obstacle exclusion `>=0.95`.
- G2 remains closed unless that later physical stage finds an actually
  feasible tuple and passes all four selection gates. Navigation, held-out,
  and sealed material remain unopened.

## Falsification and stopping rule

- Success would show that the accepted RGB encoder/Camera head contained
  useful fine physical evidence, but parallel/brittle composition and
  unbalanced joint gradients previously prevented the JEPA from using it as
  its navigational state.
- Failure closes this exact evidence-plane construction, role projections,
  fine objective, gradient routing, initialization, seed, schedule, and cap.
  Do not follow it with an old-raster restoration, depth-plane grouping sweep,
  projection-width change, coefficient tweak, threshold search, or
  rejected-state reuse.
- A later successor is justified only by a new causal diagnosis and coherent
  improving evidence. This preregistration does not authorize a Camera V7,
  two-view correspondence successor, or navigation/held-out attempt.
