# LeWM Go2 dual residual token-adapter JEPA V1 preregistration

Date: 2026-08-03

## Purpose and exact claim

This experiment must attempt both of the next trainable dense-token
representation mechanisms:

1. `residual_joint_vjepa2_1`, initialized from the exact bound V-JEPA 2.1
   train-token cache; and
2. `residual_joint_dinov2`, initialized from the exact bound DINOv2
   train-token cache.

Both arms use one identical mechanism and fixed optimization contract, in the
fixed execution order shown above.  Each
must execute through at least update 800 independently of the other arm's
scientific outcome.  This is a train-only capacity screen.  It opens no RGB,
uses no evaluation targets or labels, authorizes no data generation, and
cannot establish fresh-scene generalization, physical action ranking,
planning utility, or navigation.

The exact mechanism name is:

> Joint residual token-adapter JEPA over frozen pretrained feature caches.

The pretrained image backbones remain frozen and are not executed.  This is
not V-JEPA or DINOv2 backbone fine-tuning, end-to-end RGB JEPA training, or an
attempt to prove that either pretrained encoder can navigate.  It is the
smallest cache-compatible mechanism in which downstream context and target
representation geometry can change under an action-conditioned predictive
objective.

## Why this successor is allowed

The frozen-feature four-arm screen failed its capacity gates.  Dense V-JEPA
then showed strong action-successor retrieval growth but insufficient
successor fidelity, reaching retrieval `0.4305555555555556` and
error-to-persistence ratio `0.8723369759128065` at its preregistered
update-1,600 futility stop.  Its terminal forbids another predictor horizon,
width, seed, optimizer, or loss-weight retry over unchanged frozen geometry.

This successor changes the representation mechanism rather than extending
that stopped predictor.  A trainable bounded residual spatial adapter is
placed before the exact predecessor predictor, and an EMA copy supplies
stop-gradient target tokens.  A simultaneous frozen-target objective and
frozen-space evaluation prevent success through learned-metric contraction.

## Exact predecessor and cache bindings

The authority must bind these predecessor records as one unit:

- four-arm result: 21,377 bytes, SHA-256
  `a6caf2ed1950781815925ccc76b4dbbf40b0f331f4b14a5e60befc88f3aae605`;
- four-arm terminal: 510 bytes, SHA-256
  `bf3bf322c2f3db877be405ebf5ca1daf9dd1a5ffd667b769d44cccab22ede758`;
- four-arm terminal review: 4,991 bytes, SHA-256
  `c450baab14b50caed3469fa88f5812c92c02b04676059568e8dae3dc2e5bad83`;
- V-JEPA horizon result: 8,598 bytes, SHA-256
  `ade09fc81d950bb4bf4d26f9620da9c46bacea945e39cef261020e6eb2121cad`;
- V-JEPA horizon terminal: 631 bytes, SHA-256
  `39a5b3498be7b4fa84abd6ec566b01969b348c44f4403d834c585a0ef4e7c68a`;
  and
- V-JEPA horizon terminal review: 4,913 bytes, SHA-256
  `0751a9c2d6d2d7d7131ca32f3d3fdc5b4aa9740632fd9a84a51f5e87b82ee1cd`.

The train-only feature inputs are:

| Encoder | Receipt | Cache | Shape and dtype |
|---|---|---|---|
| V-JEPA 2.1 | 1,822 bytes; `5d4f8a82d10a33c21b41f1543d6f56b3a230a38f67b02d3f8e7330a8d30180f5` | 604,097,648 bytes; `3549855ea857906dfe3a4b55fc817633b5114b2457f8facaa4fa87f9eddd798b` | `[1536,256,768]`, float16 |
| DINOv2 | 1,770 bytes; `e94ec5d188811c44d4cc870e76d1888aa6f30ee6d423557ee9f3e2918a700994` | 302,107,682 bytes; `164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b` | `[1536,256,384]`, float16 |

Both receipts must bind artifact-order SHA-256
`68f19fc1121d4e5d6cd85c8ac50dab8538c8507ebb9a0e70258228147be2ec73`
and metadata-index SHA-256
`b740e3efead2f79fd17337a9fa10784c91989e52e837d023b2cc02a2c19d018d`.
Each must report 1,536 train opens and zero evaluation opens.

The runner reconstructs the exact train index through the frozen metadata-only
loader and validates cache binding, payload, artifact order, preprocessing,
shape, dtype, finiteness, and input token norms.  It must not call feature
extraction or the RGB reader.  Metadata evaluation artifact IDs may be
consulted only by the already reviewed train/evaluation disjointness check.

## Shared model mechanism

For native token width `D` (`768` V-JEPA, `384` DINOv2), construct the exact
existing `DenseActionConditionedPredictorV1` first, before any new
RNG-consuming module.  Predictor hidden width remains 128 and action
vocabulary remains nine.

Then construct an online adapter with two identical residual spatial blocks.
Each block contains:

- LayerNorm over native width `D`, with `eps=1e-5`, affine weight and bias;
- a biased linear `D -> 64` bottleneck;
- exact `GELU(approximate="none")`;
- a biased 3 by 3 convolution with 64 input and output channels, groups 64,
  stride one, padding one, and dilation one over the 16 by 16 token grid;
- a biased 1 by 1 convolution with 64 input and output channels, groups one,
  stride one, and no padding;
- a linear `64 -> D` projection initialized exactly to zero; and
- a bounded residual update followed by per-token L2 normalization.

For each block, let `r` be the raw projected update and use
`delta = r / (1 + ||r||_2)`, where the norm is computed separately for every
token over its last, native-width axis with the axis retained for broadcast.
Add `0.125 * delta`, then apply `torch.nn.functional.normalize` over the last
axis with `p=2` and `eps=1e-12`.  Two blocks therefore allow spatial
adaptation while sharply limiting coordinate-system rotation and making
collapse or learned-metric contraction structurally difficult.  All
non-output adapter parameters use the fixed seeded PyTorch defaults; both
output projections are biased and their weights and biases start at exact
zero.  Because the float16 cache tokens are only numerically unit length,
zero adapter outputs produce a deterministic near-identity normalization, not
an exact replay of the predecessor predictor.

Create a target adapter as an exact detached copy of the online adapter.
It receives successor observation tokens only, never action IDs.  After each
optimizer update, update every target parameter by EMA momentum `0.996` and
keep the target adapter in evaluation mode with gradients disabled.

The predictor consumes online-adapted three-frame context grids plus the same
two historical requested action IDs and candidate requested action as the
predecessor.  It predicts native-width dense tokens for the candidate
successor.

Cache tensors are promoted from float16 to float32 before model use.  The
predictor, both adapters, every loss, and every training forward/backward pass
use float32 with autocast disabled.  No alternate mixed-precision or
reduced-precision training path is allowed.

## Fixed objective

Define the predecessor common objective

`L_common(prediction, target) = matched mean token cosine distance`

`  + 0.25 * within-state nine-way cross-entropy`

using the full predicted-action by true-successor cosine-distance matrix and
temperature `0.1`.

For each minibatch, adapt the 12 unique observation grids per selected state
(three contexts plus nine successors) before candidate expansion.  Let
`u` be those frozen unique tokens, `A_online(u)` their online-adapted values,
and `A_ema(target)` the detached EMA successor targets.

Use exactly:

`L = 0.5 * L_common(prediction, stopgrad(A_ema(target)))`

`  + 0.5 * L_common(prediction, frozen_target)`

`  + 0.10 * L_identity`

`  + 0.10 * L_relative_variance`.

`L_identity` is mean `1 - cosine(A_online(u), u)`.

For `L_relative_variance`, compute per-channel population standard deviation
over the minibatch's unique-observation and spatial-token axes.  Penalize
`relu(0.90 * stopgrad(std(u)) - std(A_online(u)))`, averaged across channels.

The frozen-target half is mandatory: learned EMA geometry alone can never
make the primary fidelity or retrieval gates pass.  No semantic, depth, pose,
contact, reward, occupancy, or physical-label loss is allowed.

## Fixed optimization

Each arm is initialized and trained independently with:

- seed `2026080301` reset before the arm;
- one AdamW parameter group containing every and only trainable predictor and
  online-adapter parameter, with constant learning rate `3e-4`, weight decay
  `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, and `amsgrad=False`; the EMA
  target parameters are excluded; set `maximize=False`, `foreach=False`,
  `capturable=False`, `differentiable=False`, and `fused=False` explicitly;
- batch size eight states and all nine actions per selected state;
- seeded state permutations identical to the predecessor;
- global gradient-norm clipping at `1.0`;
- target EMA momentum `0.996` after each optimizer step;
- traces at updates 0, 400, 800 and, if continued, 1,600; and
- minimum 800 and maximum 1,600 updates.

No checkpoint selection, alternate seed, resume, retry, coefficient change,
adapter width/depth change, optimizer change, or 3,200-update continuation is
allowed.  Both arms must reach update 800 even if the first arm fails, except
that a detected nonfinite arm terminates immediately under the numerical-
failure rule below.

## Frozen controls and primary evaluation

Do not retrain the stopped frozen predictors.  Bind and report their exact
update-800 metrics:

| Control | Error / persistence | Branch retrieval | Intervention margin |
|---|---:|---:|---:|
| Frozen dense V-JEPA 2.1 | `0.9164363539053353` | `0.2803819444444444` | `0.028666746492187187` |
| Frozen dense DINOv2 | `0.9666892593579309` | `0.2048611111111111` | `0.04603223171499038` |

Every primary metric is computed only against unchanged frozen cache targets,
in the original pretrained token coordinates.  EMA-space metrics are
not required; if present they are diagnostic and cannot determine eligibility.
The primary evaluator must reuse the predecessor cosine-distance matrix and
normalization exactly.  In particular, persistence is always the frozen last
context grid compared with the nine frozen successor targets; neither an
online-adapted nor EMA-adapted context may be used in that denominator.
The distance matrix axes remain predicted action by true successor, retrieval
is the rowwise argmin against the matching action ID, and the cyclic
derangement uses predicted action `(a + 1) % 9` against true successor `a`.

Final train-set capacity requires the conjunction:

- frozen-space error-to-persistence ratio at most `0.80`;
- frozen-space branch retrieval at least `0.50`;
- frozen-space cyclic action-intervention margin strictly positive;
- no nonfinite loss, gradient, parameter, token, checkpoint, or metric; and
- exact deterministic repeated evaluation.

## Retention and anti-collapse qualification

At each capacity decision point—update 800 and update 1,600 if reached—adapt
all 1,536 train artifacts in deterministic artifact order and require:

- mean per-token cosine between online-adapted and frozen tokens at least
  `0.965`;
- entropy effective-rank ratio at least `0.90`, computed from float64
  covariance eigenvalues of the 1,536 spatially pooled artifact vectors for
  adapted versus frozen tokens;
- all online and EMA tokens finite and unit-normalized within absolute
  tolerance `1e-5`; and
- exact deterministic repeat of the retention calculation.

For effective rank, mean-pool the 256 tokens of each artifact, convert the
resulting `[1536,D]` matrix to CPU float64, subtract its column means, and form
`C = X.T @ X / 1536`.  Compute `torch.linalg.eigvalsh(C)`, clamp each
eigenvalue below zero to zero, fail if their sum is nonpositive or nonfinite,
let `p` be each strictly positive eigenvalue divided by the sum, and use
`exp(-sum(p * log(p)))`.  Apply this identical estimator to adapted and frozen
vectors and divide adapted rank by frozen rank.  Artifact order and adapter
batching are fixed and repeated exactly.

For this retention calculation only, stream all 1,536 artifacts through both
the online and EMA adapters.  Applying context artifacts to the EMA adapter
for this no-gradient integrity measurement does not make them EMA training
targets or inputs to any loss.

Report online-adapter parameter L2 movement from initialization as a mechanism
execution witness at updates zero, 400, 800 and 1,600 if reached.  The
update-zero value is expected to be zero.  At update 400 and every later point
it must be strictly positive for capacity eligibility, but its magnitude is
not a performance gate.  The `0.965` cosine threshold is a bounded-adapter
integrity check, not independent evidence of rich spatial retention.

The relative-variance term is exactly the stated unsquared linear hinge.  Do
not divide it by the frozen standard deviation or add a different epsilon.

## Update-800 decision rules

If an arm already passes final capacity and retention at update 800, stop that
arm successfully.  Otherwise it may continue to 1,600 only if retention
passes, intervention margin is positive, both primary metrics improve
strictly from update 400 to 800, and its fixed midpoint conjunction passes:

| Arm | Maximum error / persistence | Minimum retrieval |
|---|---:|---:|
| `residual_joint_vjepa2_1` | `0.8582181769526677` | `0.3901909722222222` |
| `residual_joint_dinov2` | `0.8833446296789655` | `0.3524305555555556` |

Failure produces an arm-local `COMPLETE_UPDATE_800_FUTILITY_STOP`.  It must not
prevent the other arm from executing its own update-800 attempt.

## Update-1600 decision rules

An arm continued to 1,600 passes only the unchanged final capacity and
retention conjunction.  Otherwise its terminal is
`COMPLETE_TRAIN_SET_CAPACITY_NOT_ESTABLISHED`.

An arm that passes at either decision point receives
`COMPLETE_TRAIN_SET_CAPACITY_ESTABLISHED`.  Passing establishes only bounded
matched-branch training-set capacity for this residual-adapter mechanism.

Write a round-trip-validated checkpoint for each arm at update 800 and another
at update 1,600 only when that update is reached.  Each checkpoint contains
the complete predictor, online-adapter, and EMA-target state; optimizer state;
exact config, arm, seed, feature width and update; and the initial online-
adapter state used for the movement witness.  Checkpoints are evidence only
and grant no resume or selection authority.

## Joint terminal and next route

The joint result must contain a terminal result for both arms, all traces,
checkpoint bindings, frozen controls, retention diagnostics, parameter and
memory counts, and exact access claims.  This execution must report zero RGB
leaf opens and zero evaluation-feature or evaluation-target opens.  The
frozen loader's metadata-only train/evaluation role-disjointness inspection is
allowed and must be reported separately rather than mislabeled as evaluation
data use.

A detected nonfinite loss, gradient, parameter, token, checkpoint, or metric
is an arm-local scientific failure with status
`COMPLETE_NONFINITE_CAPACITY_NOT_ESTABLISHED`; record the last completed
update without serializing nonfinite state, then still launch the other arm.
Failure of the post-update-400 execution witness or either exact deterministic
repeat is an arm-local qualification failure with status
`COMPLETE_QUALIFICATION_FAILURE_CAPACITY_NOT_ESTABLISHED` and does not block
the other arm.  A checkpoint round-trip failure or any other exception is an
infrastructure failure, must write joint status
`CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE`, makes no claim that both arms were
attempted, and authorizes no retry or replacement.  The two scientific joint
statuses below are available only after both arms have been launched and
resolved under these rules.

If neither arm passes, use
`COMPLETE_BOTH_ATTEMPTED_NO_CAPACITY_ESTABLISHED` and stop this cached-token
adapter family before fresh data generation.

If one or both pass, use
`COMPLETE_BOTH_ATTEMPTED_AT_LEAST_ONE_CAPACITY_ESTABLISHED`.  This screen still
authorizes no collection.  A later separately preregistered experiment may
advance only the eligible treatment arms, while retaining frozen V-JEPA,
DINOv2, conventional state-space, and RSSM comparisons regardless of the old
controls' candidate-gate failures.

No possible result from this screen establishes navigation usefulness.

## Source review and one-shot authority

Before execution, implement focused model, data, runner, and terminal tests;
obtain an independent no-findings source review; commit the reviewed source;
and issue a caller-bound one-shot authority that binds the exact source commit,
both caches and receipts, predecessor records and source closure, environment,
configuration, and fresh output root.  The authority grants no RGB, feature
extraction, evaluation, data generation, held-out, sealed, planning, rollout,
navigation, deployment, retry, resume, or replacement-attempt access.
