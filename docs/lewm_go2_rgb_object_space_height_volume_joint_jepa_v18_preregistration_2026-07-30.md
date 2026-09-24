# RGB Object-Space Height-Volume Joint JEPA V18 preregistration

Date: 2026-07-30

Status: preregistered source-only falsification. No V18 reservation, scientific
input access, GPU training, checkpoint, calibration, G2, navigation, held-out,
or sealed access has occurred.

## Question and committed evidence

V14/V15 learned useful RGB-conditioned ray depth and FREE-space evidence, and
their joint action-conditioned predictor remained active, but the representation
never became physically complete. The matched V15 state at update 400 had 72
passed physical margins, total shortfall `68.96954700805838`, and rough depth
p95 `1.8582415819168085` m. Longer training improved V15 to 94 margins and
shortfall `36.23231227320197` at update 1,400, but depth p95 remained
`1.4730476856231687` m and no physical scope completed. The exact V15 result is
commit `7a0dbc1f850bc8917bc45566425116fdef87ef42`, file SHA-256
`f2597d6d73d39c66352eda301a661650cbd52e7936143512aa76bac9f5a58a01`,
and byte count `10935`. The exact V14 result is commit
`d54dfea445dc9bc80cee6421c1b0aea2639463f1`, file SHA-256
`290cde5ef5dd2bf4fc93fd15b5fc1fd107fd857291abf29d4d57351d843f5263`,
and byte count `9806`.

V16/V17 tested temporal agreement between successive image-ray fields. Delayed
onset removed the early interference but still did not beat the V14/V15
update-400 state, so that mechanism family is closed. The V17 terminal result
is commit `a6c5e96627251eb578f46b0cb5d3883c30788d72`, file SHA-256
`063c4159423576afec5ef7926eb0ee9dc0cde8d57e7542f0ed79bdc7912b1126`,
and byte count `12464`.

V10 established a useful causal clue: projecting RGB evidence through 3-D cell
supports increased selection obstacle detection by `0.295101` over V9, but V10
immediately attention-pooled its 25 supports into one 2-D vector per cell and
failed physical calibration. V11's fixed height-role split then traded FREE
quality for OCCUPIED quality. V18 asks whether retaining metric height identity
inside the state predicted by the JEPA fixes the missing physical separation.

## Sole material mechanism

V18 starts fresh from the accepted N320 initialization and the V14 scientific
parent. It keeps V14's learned RGB encoder, dense decoder, unified 64-bin
first-hit hazard/offset field, and survival-derived ground-clear evidence as the
jointly trained supervised front end. It removes V14's 40-plane/64-plane 2-D
FREE/OCCUPIED projection bottleneck and replaces it with one explicit nominal
body-frame height volume:

- XY is the unchanged `64 x 64`, `0.10` m navigation grid with centres
  `[-0.95, 5.35]` m forward and `[-3.15, 3.15]` m left.
- Z has exactly eight registered centres, in metres:
  `(-0.333, -0.183, -0.033, 0.117, 0.267, 0.417, 0.567, 0.717)`.
- At each voxel, fixed nominal camera geometry projects its centre into the
  learned ray-depth field. One `torch.nn.functional.grid_sample` call with a
  `(B,3,D,Hray,Wray)` input, trilinear mode, zero padding, and
  `align_corners=False` returns ordered first-hit probability,
  survival-to-bin-centre probability, and normalized within-bin offset. Fixed
  visibility and normalized registered height are concatenated afterward,
  giving exactly five inputs. All five inputs and every learned output are
  forced to zero outside the registered frustum/depth support.
- Geometry is exact: for voxel delta `d = xyz - camera_origin`, let
  `f/r/u = dot(d, camera_forward/right/up)` and let range `rho = ||d||_2`.
  Image coordinates are `gx = r/(f*tan(hfov/2))` and
  `gy = -u/(f*tan(vfov/2))`; depth coordinate is
  `gz = 2*(rho - 0.05)/(64*0.10) - 1`. A voxel is visible iff
  `f >= 0.05`, `|gx| <= 1`, `|gy| <= 1`, and
  `0.05 <= rho <= 6.45`, all inclusive. At ray/depth bin `j`, the sampled
  first-hit channel is
  `exp(logsigmoid(h[j]) + sum(k<j, logsigmoid(-h[k])))`; the sampled clear
  channel is
  `exp(sum(k<j, logsigmoid(-h[k])) + 0.5*logsigmoid(-h[j]))`; and offset is
  normalized by division by `0.05` m. Registered height is normalized as
  `(z - 0.192)/0.525`.
- A shared learned pointwise `5 -> 8` projection followed by exactly one local
  residual block of two `3 x 3 x 3`, `8 -> 8` convolutions produces an
  `8-channel x 8-height x 64 x 64` object-space volume.
- The sole JEPA latent is that exact volume flattened in registered
  height-major/channel-minor order to the inherited `64 x 64 x 64` interface.
  The unchanged action-conditioned predictor predicts it against the EMA
  target from update 1. Height is never mean-pooled, attention-pooled, assigned
  a hand-coded floor/elevated role, or removed before prediction.
- Cell validity is exactly `any(voxel_visible over the eight heights)`. The
  volume mask is reapplied after the pointwise projection and after its
  residual block. Cells with no visible height are forced to UNKNOWN.
- The semantic decoder consumes all 64 flattened height/channel inputs—there
  is no inherited `32/32` role split. Its exact shared trunk is one residual
  block `x + Conv2d(64,64,3,padding=1,bias=True)(GELU(Conv2d(64,64,3,
  padding=1,bias=True)(x)))`, followed by one biased `1 x 1`, `64 -> 2`
  convolution whose ordered outputs are FREE then OCCUPIED evidence. The
  unchanged neutral disjoint ternary transform combines those axes.
- Every new convolution uses Xavier-uniform weights with gain `1.0` and zero
  bias. Each of the volume lift and semantic decoder uses its own CPU generator
  freshly seeded `20260729`, initializes layers in declaration order, and
  restores the caller RNG state.

The ray-depth tensor is an intermediate learned RGB field and Camera
supervision surface, not a parallel JEPA representation or inference output.
Both semantic and predictive losses traverse the object-space projection back
into that shared RGB field. The object-space volume is the only online/target
state presented to the JEPA predictor. The predictor is not trained separately.

This differs materially from V10 because V10 sampled image tokens at 25
supports and collapsed all height evidence before its JEPA state. V18 samples
the learned ordered depth field at metric voxels and retains all eight height
slices throughout EMA targeting and action-conditioned prediction. It differs
from V11 by learning all height interactions with shared local convolutions,
and from V16/V17 by having no temporal warp, KL, consistency loss, coefficient,
or onset schedule.

## Frozen inherited identity

Except for the object-space lift and its semantic decoder, preserve V14:

- RGB-only inference with registered fixed nominal camera geometry; no depth
  sensor, pose input, map, collision state, labels, future frame, or held-out
  information;
- fresh accepted N320 initialization and no V10-V17 checkpoint, optimizer,
  EMA, RNG, trace, metric, or calibration reuse;
- exact 4,262-pair train and 495-pair checkpoint-selection roles and the first
  16,000-presentation schedule order;
- constructor seed `20260712`, schedule seed `20260713`, stochastic execution
  and bootstrap seed `20260728`, and isolated projection seed `20260729` for
  every new V18 parameter;
- four `B=4` microbatches per update, float32 AdamW, inherited learning rates,
  betas, epsilon, weight decay, route-wise norm-one clipping, one optimizer
  step, and one EMA update per completed update;
- unchanged semantic, persistence, action-conditioned prediction, survival,
  progress-ranking, occupied-safety, hierarchical first-hit, within-bin offset,
  and balanced ground-clear objectives and weights;
- unchanged causal controls, physical evaluator, final thresholds, and
  observations at updates `0`, `100`, `400`, and `1000`; and
- exactly one fresh attempt capped at 1,000 updates and 16,000 presentations.

## Required preflight

Before scientific reservation, focused source-only tests must establish exact
voxel centres/order, projection centre alignment, visibility masking,
height-major flatten/unflatten identity, nonzero gradients from semantic and
JEPA losses through the volume into the shared ray field, absence of target
gradients, exact online/EMA initialization identity, and one synthetic joint
update with finite route/accounting receipts. Synthetic tensors confer no
scientific presentations.

## Gates and stopping rule

Update 100 is a health gate only: exact accounting, finite state/losses,
nonzero Camera, JEPA-shared, representation, and predictor routes, one EMA step
per optimizer step, no target gradients, RGB sensitivity, and all structural
invariants must pass.

At update 400, continue only if integrity and all twelve causal-control checks
pass and V18 strictly beats the matched V14/V15 state on every primary residual:

- at least 73 passed physical margins;
- total physical shortfall strictly below `68.96954700805838`; and
- rough depth p95 strictly below `1.8582415819168085` m.

Equality fails. Any failed conjunct terminates the attempt at update 400.

At update 1000, retain the exact V14 final gate: structural integrity, inherited
V12 checks `24/24`, at least `112/189` physical margins, total shortfall strictly
below `33.05143763708337`, at least one complete physical scope, rough pixel
balanced accuracy above `0.8198594673963917`, rough ground balanced accuracy
above `0.647134926562893`, and rough depth p95 below `0.9777327477931971` m.

Only a complete update-1000 pass may publish a development perception
checkpoint and request a separately governed next stage. Failure is a valid
scientific negative: no retry, resume, second seed, height/resolution/channel
sweep, renderer variant, loss coefficient search, threshold relaxation, or
automatic 1,400/2,000 extension. Probability calibration, G2, navigation,
held-out, sealed, production, promotion, and deployment remain unauthorized.
