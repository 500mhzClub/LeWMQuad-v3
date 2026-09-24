# RGB JEPA encoder pretraining V1 preregistration

Date: 2026-07-25

## Decision

Run one bounded, stage-separated test of an encoder-level JEPA mechanism.
This is not another Camera decoder, token-overlap, temporal-fusion, or
motion-alignment variant.

The scientific question is:

> Can an N320-initialized, action-conditioned JEPA-adapted frozen RGB encoder
> satisfy the absolute scene-disjoint physical perception gate?

The experiment has two phases in one fresh attempt:

1. train a genuine RGB JEPA without Camera labels or losses; then
2. freeze and transfer only its encoder into the unchanged multiresolution
   physical evidence model and train only that evidence head.

The cumulative cap is 2,000 optimizer updates and 32,000 pair presentations:
1,000/16,000 in each phase. This deliberately exceeds the handoff's
single-stage 16,000-presentation cheap-probe cap because the mechanism requires
one label-free representation stage and one frozen physical readout stage.
Phase B is conditional on the fixed Phase-A gate. GPU-active monotonic time is
capped at 60 minutes through the Phase-A terminal decision and 120 minutes
cumulatively through the Phase-B terminal decision. Exceeding either cap is
terminal and authorizes no retry.

Failure in either phase terminates this mechanism. No retry, resume, second
seed, schedule extension, threshold change, data rebuild, G2, navigation, or
held-out access is part of V1.

## Why this mechanism

The prior Camera plus JEPA run did not fairly test representation learning:
the Camera gradient norm was about `8.38e7` times the JEPA gradient norm
before one global clip. Later isolated Camera mechanisms changed multiscale
decoding, temporal fusion, motion alignment, and overlapping tokenization,
but all ended at zero complete physical scopes and essentially the same
development endpoint.

This leaves an upstream hypothesis open: the supervised Camera objective may
be learning a representation that fits direct labels but generalizes poorly
across scenes. A separately optimized JEPA objective can test that hypothesis
without Camera-gradient interference.

## Fixed data and role boundary

Reuse Raw V13 exactly; do not rebuild, refine, rebalance, filter, or resample
it.

- train: 4,262 pairs, 7,777 unique endpoints, 72 scenes;
- checkpoint selection: 495 pairs, 924 unique endpoints, 8 scene-disjoint
  scenes;
- probability calibration: closed;
- G2, navigation, held-out, and sealed roles: closed.

Each pair spans one reset-safe 0.5-second requested primitive block. The sole
action input is the requested categorical primitive from the fixed nine-token
vocabulary:

`arc_left`, `arc_right`, `backward`, `forward_fast`, `forward_medium`,
`forward_slow`, `hold`, `yaw_left`, and `yaw_right`.

The predictor must not receive realized SE(2), camera origin or basis, ground
plane, depth/raster labels, scene/family identity, row identity, paths, hashes,
or other future/privileged fields. The next RGB is visible only to the frozen
EMA target path. The online path receives only current RGB and the one-hot
requested primitive.

Phase A must use a dedicated RGB-only loader. It may parse the bound pair and
endpoint indexes and open the two bound RGB payloads, but it must not call the
general Raw V13 frame loader or open Camera supervision arrays. RGB
preprocessing is fixed to the existing Camera path: PIL decode, `convert("RGB")`,
bilinear resize to `112x112`, float32 division by 255, CHW conversion, then
channelwise ImageNet normalization with mean `(0.485, 0.456, 0.406)` and
standard deviation `(0.229, 0.224, 0.225)`.

## Initialization

Use seed `20260712`. Load only the already reviewed N320 fit checkpoint:

- copy its `VisionEncoder` state into the Phase-A online encoder;
- copy the same state into the Phase-A EMA target encoder;
- initialize the JEPA predictor and projection modules from the fixed seed;
- open no rejected Camera-adaptation or prior probe checkpoint.

This is explicitly a **JEPA adaptation of the supervised N320 RGB encoder**,
not a JEPA-from-scratch claim.

For Phase B, initialize
`SharedObservableCameraRayJepaV5MultiresV1` through its reviewed N320
initialization path, replace only `encoder.*` with the Phase-A terminal online
encoder, and leave the N320-derived pixel/ground heads plus fresh
multiresolution decoder at their prescribed initialization.

## Phase A: RGB action-conditioned patch JEPA

Reuse `Phase2DSpatialLeWorldModel` with this fixed configuration:

- image size `112`, patch size `7`;
- latent width `192`, encoder depth `6`, encoder heads `6`, MLP ratio `4`;
- encoder dropout `0`;
- command width `9`;
- patch-token target geometry;
- two predictor layers, six heads, head width `32`, MLP width `384`;
- predictor dropout `0`;
- state-plus-action prediction mode;
- EMA target momentum `0.996`;
- appearance SIGReg weight `0.09`;
- spatial variance weight `1.0`;
- wrong-action identifiability weight `1.0`;
- zero-action weight `1.0`;
- action margin fraction `0.10`, floor `1e-4`;
- detach the encoder state in wrong/zero-action control branches;
- no consequence, utility, Camera, depth, raster, occupancy, pose, attitude,
  or ego-motion objective.

The API literals are exactly `target_geometry="patch"` and
`prediction_input_mode="state_action"`. SIGReg uses exactly 64 projections and
9 knots.

Do not call `Phase2DSpatialLeWorldModel.forward()` in Phase A because that
method sends both frames through the online encoder for auxiliary
regularization. Use one small current-only adapter:

1. the online encoder, online target projector, appearance projector, and
   action-conditioned predictor see current RGB only;
2. the frozen EMA encoder and EMA target projector encode current RGB for the
   persistence/control target and next RGB for the stop-gradient prediction
   target;
3. prediction MSE compares the normalized prediction only with the normalized
   EMA next target;
4. SIGReg is applied to the online current CLS only;
5. the spatial variance floor is applied to online current projected patch
   tokens only; and
6. the EMA modules update exactly once after each completed optimizer step.

Use AdamW with betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`,
float32, no autocast, and global clip norm `1.0`. The encoder learning rate is
`1e-4`. The `3e-4` group contains exactly
`appearance_projector.*`, `online_target_projector.*`,
`prediction_projector.*`, and `predictor.*`. `online_geometry.*` has no
parameters under patch geometry. All `target_*` parameters are frozen and
excluded from the optimizer.

For every training row, construct one wrong action by cyclically rotating the
one-hot action index by `+1 mod 9`; the wrong-action mask is true for every
row. The zero-action mask is true exactly when the requested primitive is not
`hold`. Real, wrong, and zero losses are reduced over the same eligible rows
for each comparison.

Reuse schedule seed `20260713`, effective batch 16 as four real microbatches
of four, and the already frozen presentation prefix. Observe update 0 and
informational updates 100 and 400. Update 1,000 is terminal:

- maximum Phase-A updates: 1,000;
- maximum Phase-A pair presentations: 16,000;
- no checkpoint selection or early stopping before the fixed observations.

At selection time, compute true-pair loss plus deterministic cyclic controls
within each selection scene:

- shuffled next target;
- shuffled current RGB;
- cyclically wrong requested action;
- zero action; and
- mean target.

For shuffled-current and shuffled-next controls, sort rows within each scene
by pair `content_sha256` and select the first later row in cyclic order whose
relevant endpoint identity differs. Fail integrity if a derangement cannot be
constructed; never compare a row with the same endpoint identity. The wrong
action is the same `+1 mod 9` construction used in training. The mean-target
baseline is the per-patch mean of normalized EMA next targets across all 495
selection pairs. All MSE values are per-row means over patch and feature axes
before population reduction.

Let raw EMA next tokens be `R` with shape `(P,256,192)`, where `P=495`, and
let `Q = R - mean_P(R)` independently at every patch position. Define:

- raw cross-sample variance as `mean_{patch,feature}(var_P(R))`;
- content-residual within-image spatial diversity as
  `mean_{row,feature}(var_patch(Q))`;
- raw effective rank from the eigenvalues of
  `cov(reshape(Q, P*256, 192))`, using
  `exp(-sum(p_i log p_i))`; and
- projected effective rank by the same position-centered formula on normalized
  EMA projected targets.

These formulas remove the fixed per-position mean before rank/spatial-health
measurement, so positional embeddings alone cannot pass the health gate.

Phase A passes only if, at update 1,000:

1. all values are finite and the EMA target stayed gradient-free;
2. centered raw EMA encoder patch-token effective rank is at least `48` of
   192, and projected target effective rank is also at least `48`;
3. raw cross-sample patch-token variance and within-image spatial diversity
   each remain at least 25% of their update-zero values;
4. aggregate true-pair MSE is at most `0.90` times shuffled-next MSE and at
   most `0.90` times mean-target MSE, on all 495 rows;
5. aggregate true-pair MSE is at most `0.95` times wrong-action MSE on all 495
   rows, and non-hold true-pair MSE is at most `0.95` times zero-action MSE on
   the exact same non-hold rows;
6. both `wrong_action_mse - true_pair_mse` and, on non-hold rows,
   `zero_action_mse - true_pair_mse` are positive in at least six of eight
   scene families; and
7. aggregate true-pair MSE is at most `0.95` times shuffled-current MSE on all
   495 rows.

If Phase A fails, Phase B is not entered and no encoder is qualified.

## Phase B: frozen-encoder physical probe

Freeze the complete Phase-A JEPA state. In the physical model, train only
`evidence_head.*`; `encoder.*`, the JEPA predictor/projectors/EMA state, BEV
modules, and occupancy head remain byte-for-byte frozen.

Only the Phase-A **online raw encoder** is copied into the Phase-B model.
Phase-A predictor/projector/EMA state remains a separately frozen JEPA
artifact. After the encoder copy, the Phase-B target encoder is hard-synced
only to avoid an internal mismatch; the unrelated Shared-V5 BEV decoder and
predictor remain at initialization and receive zero JEPA objectives or EMA
updates. Consequently, the Phase-B artifact is a perception probe, not a
coherent or promotable Shared-V5 JEPA checkpoint.

Use the unchanged multiresolution evidence head, Camera geometry, tail-depth
loss, physical evaluator, wrong-RGB mapping, train/selection populations, and
fixed Phase-B schedule prefix. Use the existing head learning-rate schedule,
AdamW settings, four-by-four accumulation, and head clip norm `1.0`.

Observe updates 100 and 400 informationally. Update 1,000 is terminal:

- maximum Phase-B updates: 1,000;
- maximum Phase-B pair presentations: 16,000.

The exact terminal conjunction is:

- at least `1/9` complete physical scopes;
- at least `98/189` passed margins;
- total shortfall strictly below `41.01776266878769`;
- rough pixel balanced accuracy strictly above `0.8198594673963917`;
- rough ground balanced accuracy strictly above `0.647134926562893`; and
- rough depth p95 strictly below `0.9777327477931971 m`.

The tracked static multiresolution V3 endpoint remains an external development
reference. V1 does not add a matched no-JEPA training arm and therefore will
not make a causal claim that JEPA alone produced any improvement. A matched
control is deferred unless a passing candidate later needs that causal claim.

## Interpretation and stop rules

- Phase-A pass plus Phase-B fail: the JEPA learned a healthy predictive
  latent, but that encoder did not solve the physical generalization gate.
- Phase-A fail: action-conditioned latent prediction itself did not qualify.
- Phase-A and Phase-B pass: the encoder-level mechanism earns a separately
  bounded qualification step; it does not directly earn G2 or navigation.
- Similarity to static V3 is failure, even if the within-run loss decreases.
- No checkpoint may be selected by loss alone.

V1 may write only its fresh attempt root and source/result receipts. It may
not access probability calibration, prior attempt roots, rejected
checkpoints, G2, navigation, held-out, sealed, production, or deployment
artifacts.
