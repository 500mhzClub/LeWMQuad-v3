# Geometry-anchored global action-indexed rigid BEV transport joint JEPA V1

Date: 2026-07-27  
Status: preregistered for implementation, source-only closure, independent
review, and one separately authorized capped attempt. No checkpoint, G2,
navigation, held-out, sealed, production, promotion, or deployment access is
authorized by this document.

## Decision

Test one final cross-mechanism synthesis: keep the successful RGB-only,
geometry-grounded BEV representation from the completed V3 experiment and
replace only its spatially constant broadcast-action predictor with a learned,
action-indexed global rigid BEV transport followed by an action-agnostic local
corrector.

This is not claimed as a new transport hypothesis. Earlier patch-whitened
local-flow, local-correspondence, cost-volume, coordinate-FiLM, nominal-warp,
and causal-motion-alignment branches did not establish reliable executed-action
identity. The present test is justified once, and only once, because V3 is the
first run in this line to provide all of the following at the same time:

- a strong geometry-grounded `64 x 64` RGB-derived BEV;
- a complete, balanced, genuine joint-JEPA phase;
- strong next-state and correct-next/deranged-target prediction; and
- weak but nonzero average action separation close to the registered action
  thresholds.

If this experiment fails, no learned-warp, transform-bank, flow-bank, bound,
padding, residual-depth, or transport-loss variant follows. The next category
must change the predictive state or uncertainty model rather than spatial
transport parameterization.

## Frozen evidence

The scientific predecessor is the complete V3 attempt closed by:

`docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_scalar_tensor_state_hash_integrity_replacement_terminal_audit_2026-07-27.json`

- audit commit: `6b48b528c53766276f4912626728611910837a92`;
- raw SHA-256:
  `bbb1d82faefc62c0358df531941ab07f2b3253d274eca2156df378ffb17a52c4`;
- content SHA-256:
  `595ac5198edfcba196ced8213c3f83ff9a5fa2c8100231b028bb99690c8a5d2b`;
- completed work: 1,000 updates, 16,000 presentations, 4,000 objectives and
  backwards, 600 joint predictor/representation updates, and 600/600 passed
  shared-gradient gates.

V3 established useful perception and generic temporal prediction:

- raster balanced accuracy `0.8429101797371322` and raster NLL
  `0.15151258928157607` at update 1,000;
- paired correct-RGB wins in `8/8` scene families;
- latent prediction loss `0.027304887696348056` against frozen
  `B400 = 0.057197078741672965`;
- same-action correct-next/deranged-target NLL `0.21442622318745308`, strict
  win rate `0.9291497975708503`, and positive margins in `8/8` families.

It failed exactly five final conjuncts:

- action macro balanced accuracy `0.21137491807807904`, required strictly
  above `2/9`;
- action NLL `2.1093718971868958`, required strictly below
  `0.95 * log(9) = 2.0873633484694083`;
- executed action beat the hardest wrong action in `0/8` families, required
  at least `6/8`;
- occupied recall `0.6686247657488829`, required at least `0.70`; and
- rough occupied recall `0.6203554119547657`, required at least `0.65`.

The original geometry-anchored V1 preregistration remains the scientific
envelope and is bound by raw SHA-256
`4d59e71702a716fb3a669ddffd87fb124da76ee2957c3e4dd16a8ad6dadbf402`.
V1 and its V2/V3 integrity replacements remain consumed and closed. No V3
checkpoint, tensor, optimizer state, trace, observation payload, registry,
RNG state, or runtime output may be opened or reused.

## Scientific question

Can a learned global action-indexed rigid transformation make the successful
geometry-grounded latent distinguish the executed primitive from all eight
wrong primitives, while genuine joint JEPA training retains the registered
RGB perception quality?

The falsifying observation remains the complete conjunctive update-1,000 gate.
A lower training loss, a plausible transform, average wrong-action separation,
or improvement over V3 does not qualify the mechanism by itself.

## Exact frozen envelope

Except for the predictor below, preserve V3 exactly:

- RGB-only `112 x 112` input and the frozen `7 x 7` tokenization;
- N320 encoder-state initialization only;
- geometry-anchored four-sample deformable lift, `64` latent channels, and
  `64 x 64` BEV lattice;
- semantic head and unknown/free/occupied ordering;
- online encoder/lift/semantic representation and stop-gradient EMA
  encoder/lift target at momentum `0.996`;
- nine-action vocabulary and ordering;
- train and checkpoint-selection roles, rows, scene families, fixed deranged
  targets, endpoint mappings, loader, and all custody restrictions;
- model seed `20260712`, schedule seed `20260713`, exact schedule, microbatch
  `4`, four microbatches per update, and effective batch `16`;
- 400 perception-only updates followed by 600 genuine joint updates;
- AdamW class, betas, epsilon, weight decay, learning rates, parameter groups,
  float32 precision, clipping, and optimizer construction time;
- semantic, latent-prediction, nine-way action, and same-action contrast losses,
  their normalization, detached scales, and unit weights;
- observations at updates `0`, `100`, `400`, and `1000`;
- all scientific, representation-retention, integrity, accounting, access,
  source, and custody gates; and
- caps of 1,000 updates, 16,000 presentations, and 30 active GPU minutes.

Construction is fresh. The preserved N320 encoder-only initialization is the
sole pretrained state.

## Sole scientific change: global learned rigid transport

Replace `_LocalActionConditionedPredictorV1` only. The replacement receives
only:

- current online RGB-derived latent `z` with exact shape `(B,64,64,64)` and
  float32 dtype; and
- exact one-hot commanded action `a` with shape `(B,9)`.

It owns one trainable table `raw_twist` of shape `(9,3)`. Rows follow the frozen
action order, while columns mean `(forward_cells, left_cells, yaw_radians)`.
Selection is by the validated one-hot index only; source may not inspect action
names or install action-specific signs, magnitudes, axes, or identities.

For the selected row `(r_f, r_l, r_y)`, compute:

```text
forward_cells = 8 * tanh(r_f)
left_cells    = 8 * tanh(r_l)
theta         = (pi / 4) * tanh(r_y)
tx            = 2 * left_cells / 64
ty            = 2 * forward_cells / 64
A             = [[cos(theta), -sin(theta), tx],
                 [sin(theta),  cos(theta), ty]]
```

`A` is an output-to-input sampling transform. Apply exactly:

```text
grid = affine_grid(A, z.shape, align_corners=False)
u = grid_sample(z, grid, mode="bilinear", padding_mode="zeros",
                align_corners=False)
```

The registered bounds are symmetric architectural safety bounds, not nominal
motion supervision. There is no learned scale, shear, cell-wise flow,
correspondence field, cost volume, pose input, or command-displacement table.

Apply one shared, action-agnostic corrector:

```text
RB(x) = x + Conv2( GELU(Conv1(x)) )
prediction = u + residual_head(RB2(RB1(u)))
```

Every convolution maps `64 -> 64`, uses kernel `3`, stride `1`, padding `1`,
and bias. Both residual blocks and the head are shared across all nine actions.
There is no action embedding, broadcast action map, FiLM, action-specific
convolution, coordinate channel, pooling, attention, or alternate prediction
path.

The predictor inventory is exactly 184,667 parameters:

- `raw_twist`: 27 scalars;
- four residual-block convolutions: 147,712 parameters; and
- residual head: 36,928 parameters.

## Initialization and symmetry

Use the same single CPU default-generator scope as V3: save caller CPU RNG,
seed the generator exactly once with `20260712` immediately before constructing
the deformable lift, continue that one stream through the semantic head and the
replacement predictor without reseeding, and restore caller CPU RNG after
predictor construction. Shared encoder/lift/semantic initialization must be
byte-identical to V3 before predictor construction.

- initialize all 27 twist scalars to exact zero without an RNG draw;
- use PyTorch default Conv2d initialization for the four residual-block
  convolutions;
- initialize residual-head weight and bias to exact zero; and
- make no accelerator RNG call during construction.

Predictor construction and registration order is exact:

1. `raw_twist` as `Parameter(torch.zeros((9,3), dtype=float32))`;
2. `residual_blocks.0.conv1`, then its GELU, then
   `residual_blocks.0.conv2`;
3. `residual_blocks.1.conv1`, then its GELU, then
   `residual_blocks.1.conv2`; and
4. `residual_head`.

Every Conv2d constructor performs its ordinary registered CPU RNG draws in
that order. The residual-head constructor draws normally before its weight and
bias are overwritten with exact zeros. The ordered predictor parameter names
are exactly:

```text
predictor.raw_twist
predictor.residual_blocks.0.conv1.weight
predictor.residual_blocks.0.conv1.bias
predictor.residual_blocks.0.conv2.weight
predictor.residual_blocks.0.conv2.bias
predictor.residual_blocks.1.conv1.weight
predictor.residual_blocks.1.conv1.bias
predictor.residual_blocks.1.conv2.weight
predictor.residual_blocks.1.conv2.bias
predictor.residual_head.weight
predictor.residual_head.bias
```

All nine predictions therefore begin with the same numerical identity warp and
zero correction. HOLD receives no special identity treatment. Action identity
must be learned only from scheduled RGB transitions and labels.

## Genuine joint JEPA contract

Updates 1-400 remain perception-only: `L = S`. Predictor forward, objective,
backward, optimizer-state, and update counts must remain zero. Compute and
freeze `B400` exactly as V3.

Updates 401-1000 preserve:

```text
L = S + P_latent_prediction + R_action + C_same_action_contrast
```

- `P` compares the executed-action prediction with the stop-gradient EMA
  next-RGB target;
- `R` evaluates all nine candidate transports and applies the unchanged
  energy-based nine-way cross-entropy;
- `C` compares the executed-action prediction against the correct next target
  and the fixed same-role, same-scene deranged target; and
- `S` continues to train the online RGB encoder, lift, and semantic head.

The online representation and transport predictor optimize together on every
joint update. The predictor is never trained behind a frozen representation.
The EMA target receives no gradient and updates once after each successful
online optimizer update. Add no motion target, transform loss, HOLD loss,
identity regularizer, diversity loss, smoothness loss, inverse head, auxiliary
classifier, adaptive reweighting, or new coefficient.

## Unchanged final qualification gate

Copy the complete V3 update-0, update-100, update-400, phase-switch,
update-1000, accounting, access, and integrity gates byte-for-byte where their
identities do not need version rebinding. In particular, final qualification
remains conjunctive and requires:

- latent prediction loss `<= 0.90 * B400`;
- action NLL strictly below `0.95 * log(9)`;
- action macro balanced accuracy strictly above `2/9`;
- executed action beats the hardest wrong action in at least `6/8` scene
  families;
- mean wrong-action energy strictly above executed-action energy;
- non-HOLD HOLD/zero-action energy strictly above executed-action energy;
- same-action correct-next/deranged-target NLL strictly below `0.95 * log(2)`,
  strict win rate at least `0.65`, and positive margins in at least `6/8`
  families;
- target rank, channel variance, and spatial diversity each retain at least
  75% of update-400 values;
- aggregate occupied recall at least `0.70` and rough occupied recall at least
  `0.65`, plus every other original perception-retention conjunct;
- all 600 shared semantic/dynamics-gradient gates pass; and
- exact optimizer, EMA, predictor, objective, backward, presentation, and
  update accounting.

No V3 near-miss permits a threshold change.

## Deterministic warning policy

Deterministic algorithms remain enabled with warn-only scope for the known
ROCm grid-sampler backward. Record every warning. Permit only `UserWarning`
whose message is the exact registered grid-sampler warning, optionally followed
by one PyTorch provenance suffix of the exact form:

```text
 (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:<decimal-line>.)
```

Canonicalize only that optional suffix before exact comparison. Reject every
other warning category, base message, suffix, prefix, or additional text.
Scientific gate outcome and operational receipt outcome must remain distinct;
warning finalization may not erase a gate already computed by the callable.

## Required source-only and synthetic closure

Before execution, independently verify at least:

- exact predictor type, inventory, parameter names, construction order, and
  absence of the predecessor action embedding/input projection;
- exact-zero twist table and residual head;
- independent bounded-twist, affine-matrix, grid, and sampled-output reference,
  checked with `atol=2e-5` and `rtol=2e-5` in float32;
- `R.T @ R` is identity and `det(R) = 1`, checked with `atol=2e-6`
  and `rtol=2e-6` in float32; every registered bound is checked with absolute
  tolerance `1e-6`;
- changing one twist row changes only that action candidate;
- for any permutation vector `p`, defining `T_prime = T[p]` and
  `a_prime = a[:,p]` preserves the selected prediction within
  `atol=2e-5, rtol=2e-5`; the all-candidate relation is explicitly
  `Y_prime[:,i] = Y[:,p[i]]` under the same tolerance, proving no action-name
  mapping;
- a spatial impulse is globally transported and conditioning cannot reduce to
  a spatially constant additive channel;
- corrector sharing across actions;
- at exact initialization, finite nonzero executed-action gradients to the
  selected twist row, residual-head parameters, and online encoder/lift;
  upstream residual-block parameter gradients must be exact zero because the
  zero head blocks them, and EMA-target gradients remain absent;
- on a source-only clone, install the sole deterministic reachability witness
  `residual_head.weight[c,c,1,1] = 1/64` for every channel `c`, with every
  other head weight and bias zero; without an optimizer step, the same fixture
  must then produce finite nonzero gradients to all four shared residual-block
  convolutions. This witness is not an initialization, training state, loss,
  checkpoint, continuation gate, or runtime input;
- finite all-nine-action objective gradients and no predictor activity during
  updates 1-400;
- exact inherited objective arithmetic, optimizer membership, EMA accounting,
  scalar-safe state hashing, isolated imports, source closure, runtime-input
  hashes, and forbidden-access guards;
- warning canonicalization accepts only the optional registered provenance
  suffix and preserves an already-computed scientific result; and
- no checkpoint or training-trace read path.

Synthetic checks may use CPU Torch and constructed tensors only. They may not
open dataset rows, RGB, rasters, N320, generated inputs, V3 outputs, checkpoints,
traces, GPU state, held-out, or sealed material.

## Attempt, stopping, and authority

The experiment ID is
`geometry_anchored_global_action_indexed_rigid_bev_transport_joint_jepa_v1`.
The sole output root is:

`.generated/go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_transport_joint_jepa_v1/attempt_v1`

It must be absent before a write-once reservation. Exactly one fresh attempt is
permitted after source freeze, independent review, and a separately committed
execution authorization. There is no retry, resume, alternate seed, extension,
second attempt, or same-root reuse.

Stop at the first applicable scientific, numerical, integrity, accounting,
custody, or cap failure. A failure after any scientific presentation closes the
mechanism. A zero-presentation operational defect may justify only one
separately preregistered, science-identical integrity replacement in a new
root; it never reopens this attempt.

Every written checkpoint and trace is write-only until the complete gate passes
and an independent terminal audit separately qualifies a specific binding.
Failure disqualifies all written checkpoints. Passing this probe authorizes only
independent audit and a later decision; it does not authorize G2, navigation,
held-out, sealed, production, promotion, or deployment access.

## Registered risks

- A single rigid transform may be too simple for parallax, occlusion, newly
  revealed space, control variation, or locally nonrigid appearance.
- Similar forward speeds may converge to similar transforms and still fail the
  hardest-wrong gate.
- The shared corrector may learn generic temporal change and weaken action
  separation.
- Bilinear interpolation and zero padding may soften obstacle boundaries and
  worsen occupied recall.
- Twist parameters may saturate at their bounds.
- ROCm grid-sampler backward remains warning-level nondeterministic.
- Joint dynamics gradients may still erode occupied-space semantics.

These are falsification risks, not permissions for variants. Only a complete
conjunctive pass can earn the next step.
