# RGB geometry-anchored Action-Query Spatial Successor joint JEPA V1

Date: 2026-07-27

Status: exact scientific preregistration for document review only. This
document authorizes no implementation, generated-input access, runtime-output
access, checkpoint access, training, GPU use, execution, navigation, G2,
held-out, sealed, production, promotion, or deployment activity.

## Decision and prior evidence

The repository goal remains a fully learned RGB-only, perception-grounded JEPA
navigation stack, eventually validated on untouched externally custodied
held-out mazes. Preregister exactly one materially different capped probe:
jointly learn a geometry-grounded RGB representation and an action-query
predictor that asks, at every spatial token, what successor latent each of the
nine actions would produce.

The hypothesis is that prior predictors learned generic next-frame structure
while global aggregation washed out small, local, action-specific differences.
This mechanism therefore contrasts all nine actions locally before a smooth
spatial aggregation. It predicts one continuous successor residual; it does
not estimate motion, transport, or event modes and has no inverse classifier
or future-conditioned action head. During training only, the executed action
label ranks the nine forward successor energies described below.

This document binds and respects these public terminal audits:

| Closed evidence | Commit | File SHA-256 | Content SHA-256 | Bytes | Binding |
|---|---|---|---|---:|---|
| Deformable-BEV joint JEPA V3 | `6b48b528c53766276f4912626728611910837a92` | `bbb1d82faefc62c0358df531941ab07f2b3253d274eca2156df378ffb17a52c4` | `595ac5198edfcba196ced8213c3f83ff9a5fa2c8100231b028bb99690c8a5d2b` | 10,661 | Completed 1,000 updates/16,000 presentations; strong generic prediction and perception but failed action identity and occupied-recall conjuncts. Its exact residual-predictor attempt is closed. |
| Global action-indexed rigid BEV transport V1 | `293a02e1cc99c3b5aac876efc68104093468506c` | `38d65b46bd4ff83ab67924233badb96c37bd079f0b85b36c69512c905557f25e` | `5a87a441560c8ab397a0cdfab5ca08fc1c7ad7b8294d323194f712ad834821ca` | 19,570 | Completed budget and worsened action identity; the transport/warp/flow/transform-bank family and its tuning variants are closed. |
| Fixed two-mode event-delta V2 | `2a8561617ace1e47f769e8849238ad0b20bc32dd` | `4fd36445ebad3db5d568dab2444eeb4350ae698f0ec54f476e35242f175d2096` | `ba8a089e86dcc5ebad69d924b413f9476d0e3089d3321e7b22764946f66420a5` | 21,027 | Perception passed through update 400, then the first joint backward failed the frozen gradient-ratio gate; the fixed event-delta family is scientifically falsified and closed. |

This successor also differs materially from three earlier action-identification
branches. Patch-whitened V4's terminal audit
`docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v4_action_indexed_energy_nll_terminal_audit_2026-07-25.json`
(commit `20a5099f17a6da17bb2858d96724f9f8e88ae3f9`, file SHA-256
`ddb3c784382f92161b82d7321c8ad3c70901cb8d5a813c3ecc7153083480d809`)
closed its shared trunk plus nine zero-initialized per-action `1 x 1` operator
bank and globally reduced Energy-NLL. Local-correspondence V8's audit
`docs/lewm_go2_rgb_action_conditioned_local_correspondence_all_candidate_identification_jepa_v8_terminal_audit_2026-07-25.json`
(commit `9f3e2bc96a6e4ea419574f109c890299d0608659`, file SHA-256
`3ea4a8cc4405b0880d2e05217e4b4acefc5b9df5fad9bcdd9a682db42e273173`)
closed bounded `3 x 3` correspondence transport and its detached wrong-candidate
identification route. V10R's audit
`docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_v10r_integrity_replacement_terminal_audit_2026-07-26.json`
(commit `79d6de74b795065f7a5a47b32f1a56fc4fd4580a`, file SHA-256
`8cd27a7d21e9ce1875d322cad2ea5aae8a846a301247f774d4da86074ebd28a5`)
records the broader conclusion
`single_frame_current_plus_action_family_closed=true` and named a masked
current-to-next pair tubelet as its separately preregistered successor. That
successor was executed as V11 and its audit
`docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_terminal_audit_2026-07-26.json`
(commit `4d3e967f1d30bc3843626a9b5aaecd79e6f1dca0`, file SHA-256
`89ac1155e7108118133d6eb0648437e3a337f03e31c6c93e6ca63cc590f27044`,
content SHA-256
`9641274f58e84b4a3c3603f7cf19714e006ec27d062d57a0f24f0bb38677aec9`,
7,876 bytes) closed exact V11 at its registered update-0 scientific gate with
zero training presentations.

This V1 does not narrow or silently ignore those conclusions. It explicitly
reopens the broad current-plus-action interface under the later standing user
authority in this thread to change scientific scope while preserving a fully
learned perception-only JEPA and sealed held-out-maze goal. It is a separately
preregistered direct physical-perception mechanism informed by the subsequent
geometry-BEV evidence, not an automatic V10R/V11 successor or retry. The one
new hypothesis is that geometry-grounded future queries plus token-local
all-candidate contrast expose sparse action evidence that every globally
reduced predecessor erased. The mechanism has no operator bank,
correspondence/transport distribution, flow, projector-token retrieval list,
masked tubelet, or detached wrong-action route. One failure closes this exact
reopened hypothesis; it does not justify another current-plus-action variant.

The V3 representation result is public evidence, not reusable state. This V1
constructs a fresh model and changes the predictive mechanism. It is not a
deformable-predictor retry, transport parameterization, event model, loss-only
variant, or threshold retune. No predecessor checkpoint, tensor, optimizer,
EMA, RNG, trace, observation, receipt payload, registry, or runtime output may
be opened or reused.

## Frozen data, representation, and runtime envelope

Preserve the already reviewed development envelope exactly:

- fresh trainable RGB encoder: input `(B,3,112,112)`, patch `7`, width `192`,
  depth `6`, heads `6`, initialized only from the exact N320 encoder state;
- fresh fixed-projective-anchor, four-sample bounded deformable lift, local
  refinement, `64` channels, and `64 x 64` BEV lattice; no global-attention or
  non-RGB bypass;
- `Conv2d(64,3,1,bias=True)` semantic head in `UNKNOWN`, `FREE`, `OCCUPIED`
  order, and a requires-grad-false EMA copy of the encoder/lift/refinement;
- train role: 4,262 pairs, 72 scenes, 7,777 unique endpoints; checkpoint-
  selection role: 495 pairs, eight frozen scene families, 924 endpoints;
- exact current/next endpoints, fixed same-role same-scene deranged-next
  mapping, row order, loader, action vocabulary/order, labels, and role
  isolation, bound by train-row SHA-256
  `bb119abb33b7c56f3c1d96e7cb1b52fbe4d2db27d80df4f95a5b1cd9d0cf729e`,
  selection-row SHA-256
  `81f85cdf0ad00ec68918f5eeb7637bf20aa3f17f5615bedfb10e6a4859eb91f1`,
  train-mapping SHA-256
  `c9c914422927670ffce8e2a967bf264725b9ae3c55c353ee0a1a16e44044196b`,
  selection-mapping SHA-256
  `95d42273a8319316ad68781cb2158146e7672eda529984c3aeddc0937d87a9c1`,
  and selection-family SHA-256
  `c39efe48afd6d4c02a24af77f1f11e7f6cd5a69d571b0a9416924b07bbacbb11`;
- load only current RGB, next RGB, fixed-negative RGB, and
  `raster_labels.u1`; labels are supervision only. Next/fixed-negative RGB is
  target/semantic-training material only and is never a predictor or
  inference input;
- schedule file/content identities
  `08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270` /
  `274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15`,
  schedule seed `20260713`, and exact prefix hashes at updates 100/400/1000:
  `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`,
  `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`,
  and `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`;
- model seed `20260712`, float32, microbatch `4`, four microbatches per update,
  effective batch `16`, and observations at updates `0`, `100`, `400`, and
  `1000`;
- one AdamW instance created at update 0, betas `(0.9,0.999)`, epsilon `1e-8`,
  weight decay `1e-4`, encoder LR `1e-4`, and lift/semantic/predictor LR
  `3e-4`; separate representation/semantic and predictor L2 clips of `1.0`;
- EMA momentum `0.996`, exactly one hard sync before update 0, then exactly one
  EMA update after every successful online optimizer step; and
- one attempt, at most 1,000 updates, 16,000 presentations, and 30 active GPU
  minutes. No render, rebuild, filter, resample, remap, new row, scene, label,
  seed, schedule, role, or cap change is permitted.

The sole output root is
`.generated/go2_rgb_geometry_anchored_action_query_spatial_successor_joint_jepa_v1/attempt_v1`.
It must be absent before authorization and before mode-`0700` reservation.

## Exact Action-Query Spatial Successor

Use NCHW tensors and per-cell, non-affine channel LayerNorm:

```text
N(Z) = layer_norm(Z.movedim(1,-1), (64,), eps=1e-5).movedim(-1,1)
x     = N(Z_online(current_rgb))                         # B,64,64,64
y     = stopgrad(N(Z_EMA(next_rgb)))                    # B,64,64,64
y_neg = stopgrad(N(Z_EMA(fixed_negative_rgb)))          # B,64,64,64
```

The target is exactly the stop-gradient EMA next-RGB latent. The predictor
receives only `x` and an action index. It has these modules and no others:

1. learned `Conv2d(64,128,kernel_size=4,stride=4,padding=0,bias=True)` giving
   `16 x 16` current-state tokens. Convert NCHW to tokens exactly as
   `conv(x).flatten(2).transpose(1,2)`, so token `q=16*u+v` is row `u`, column
   `v` in row-major order;
2. a fixed float32 two-dimensional sine/cosine position buffer `P[256,128]`:
   for row/column coordinates `u,v in {0,...,15}` and `i in {0,...,31}` use
   `w_i=10000^(-i/32)`, concatenate
   `[sin(u*w_i),cos(u*w_i)]_i` and `[sin(v*w_i),cos(v*w_i)]_i`;
3. `Embedding(9,128)` providing nine distinct learned action tokens and one
   learned future-query table `Q[256,128]`;
4. two identical-in-shape but fully separately parameterized pre-norm
   future-query blocks. For action `a`, initialize `q=Q+P+action[a]` and set
   immutable memory `m=concat(action[a][None,:],current_tokens+P)`. Each block
   owns three distinct affine `LayerNorm(128,eps=1e-5)` modules named
   `query_norm`, `memory_norm`, and `ffn_norm`; none is shared within or
   between blocks. With that block's
   `MultiheadAttention(128,4,bias=True,batch_first=True,dropout=0)`, compute
   `q_attn=q+MHA(query_norm(q),memory_norm(m),memory_norm(m),need_weights=False)[0]`
   using the same normalized memory tensor as both key and value. Then compute
   `q_out=q_attn+linear2(GELU(linear1(ffn_norm(q_attn))))`, where the block's
   biased MLP is exactly `128 -> 256 -> 128`. Feed `q_out` to the next block;
   memory `m` is unchanged. There is no self-attention, dropout, parameter
   sharing, or global pooled bypass; and
5. reshape the 256 future tokens to `128 x 16 x 16`, bilinearly upsample by
   four with `align_corners=False`, and apply one
   `Conv2d(128,64,kernel_size=3,padding=1,bias=True)` output head. Its only
   output is a continuous residual `r_a`; the successor is `p_a=x+r_a`.

All nine actions are evaluated in one vectorized `(B,9,...)` predictor call.
No loop may change batch-normalization or RNG behavior; the model contains no
batch normalization. There is no warp, flow, rigid transport, deformable
sampling inside the predictor, event mixture, mode, codebook, pair posterior,
inverse head, action classifier, future-RGB predictor input, coordinate/pose/
odometry/attitude/map/goal input, or auxiliary branch.

## Local energies and exact joint objective

Let `huber(a,b)` be elementwise smooth-L1 with `beta=1`. Reduce channels, then
non-overlapping `4 x 4` spatial cells to the predictor-token grid:

```text
e_pos[b,a,q] = avgpool4(mean_c(huber(p_a,y)))       # B,9,256
e_neg[b,a,q] = avgpool4(mean_c(huber(p_a,y_neg)))   # B,9,256
e_per[b,q]   = avgpool4(mean_c(huber(x,y)))         # B,256
```

Define the smooth spatial soft-min
`SSM(v)=-0.25*(logsumexp(-v/0.25,dim=q)-log(256))`. It equals the common
value when all 256 tokens tie and gives localized low-loss action evidence a
bounded, differentiable route without averaging it away. Every scale below is
detached, per row and token, and clamped to `1e-3`:

```text
s_action = stopgrad(mean_a(e_pos))
local_action_ce = CE(-e_pos/s_action[:,None,:], executed_action)  # per q

e_exec = gather_action(e_pos,executed_action)
e_exec_neg = gather_action(e_neg,executed_action)
s_target = stopgrad(0.5*(e_exec+e_exec_neg))
local_target_ce = CE(
    stack(-e_exec/s_target,-e_exec_neg/s_target,dim=class), label=0) # per q

S = (0.5*A_current + 0.5*A_next)/log(3)
P_successor = mean(e_exec)
R_local_action = mean_b(SSM(local_action_ce/log(9)))
C_deranged = mean_b(SSM(local_target_ce/log(2)))
L = S + P_successor + R_local_action + C_deranged
```

`A_current` and `A_next` retain the inherited equal-row,
equal-present-final-class macro NLL. All four objective coefficients are
exactly `1.0`; the displayed channel, token, class-log, detached-scale, and
`SSM` reductions are the complete normalizations. Add no loss, margin,
temperature, learned scale, annealing, adaptive weighting, phase-dependent
coefficient, or gradient rescaling beyond the registered `/4` effective-batch
normalization.

The online encoder/lift/refinement/semantic head and every predictor parameter
train together on every update from update 1. There is no perception-only
phase, frozen-encoder predictor phase, predictor pretraining, or separately
trained predictor. The EMA target never receives gradients or optimizer
membership.

For reporting only, define row/action scores
`E_a=SSM(e_pos[:,a,:]/s_action)` and correct/deranged scores with the same
`SSM` and `s_target`. Lower is better. Raw action NLL is
`CE(-E_a,executed_action)`; macro balanced accuracy is the unweighted mean of
the nine nonempty per-action recalls. The hardest wrong is the lowest-energy
non-executed action. A family win requires a strictly positive equal-row mean
margin; ties fail.

## Initialization and update-0 proof

Save caller CPU RNG, seed the CPU default generator exactly once with
`20260712`, construct/load the fresh online encoder and construct the lift and
semantic head in the inherited order, then construct the predictor in the
module order above without reseeding. Explicitly overwrite the action-token
matrix, future-query table, and shared output-convolution weight with nonzero
`xavier_uniform_` draws from that continuing seeded stream; set only their
biases to zero. Action rows must be pairwise nonidentical, the future-query
table and shared output weight must have finite nonzero norms, and the nine
action-conditioned tensors entering the shared output head must be pairwise
nonidentical on the frozen synthetic witness. Other layers retain their
constructor initialization. Restore caller CPU RNG. Construct the EMA copy, hard-sync once,
set every target parameter `requires_grad=False`, and freeze optimizer ordered
membership before update 1.

Before runtime access, CPU-only source tests must prove exact tensor shapes,
position values, action permutation equivariance, nine-action vectorization,
distinct action/query/output initialization, residual identity arithmetic,
and no forbidden branch. A synthetic update-0 autograd witness must prove
finite nonzero dynamics gradients reach the online encoder/lift, downsampler,
action tokens, future queries, both attention modules, both MLPs, and output
head; semantic gradients reach the online representation/semantic head;
target gradients and optimizer membership are absent; and the target is
bitwise equal to the hard-synced online representation. These checks use only
constructed tensors and synthetic RGB, consume zero presentations, and take
no optimizer or EMA step.

At every real update call `optimizer.zero_grad(set_to_none=True)` exactly once
before the first microbatch. For each of the four ordered microbatches compute
the displayed combined objective `L`, call `(L/4).backward()` exactly once,
and retain no graph after that call. Thus the accumulated parameter gradient
is the mean effective-batch gradient, not the four-microbatch sum. At update 1
only, before each combined backward, use two non-mutating `torch.autograd.grad`
calls with `retain_graph=True` on `S/4` and
`(P_successor+R_local_action+C_deranged)/4` with respect to the shared online
encoder/lift/refinement parameters. Sum their detached results across the four
microbatches solely for the route witness; these eight gradient-probe calls
are separately counted and are not backward calls or optimizer gradients.

Before the update-1 step require exact objective arithmetic, finite component
values, finite nonzero accumulated semantic and dynamics route-witness
gradients into the shared online representation, finite nonzero accumulated
`.grad` tensors for every predictor component, absent target gradients, and
exact optimizer membership. At every later update require all accumulated
optimizer gradients finite, target gradients absent, and the complete
predictor parameter set's aggregate L2 gradient finite and nonzero. Individual
tensor gradients may be zero on a particular later batch; update-0 proves all
routes and update-100 requires every component to have moved. Record semantic
norm, dynamics norm, and both ratios informationally. No finite pre-step ratio,
however large or small, is by itself an abort condition. Clip the combined
encoder/lift/refinement/semantic group to global L2 norm `1.0`, then clip the
complete predictor group to global L2 norm `1.0`, call the sole AdamW
`step()` exactly once, and perform exactly one EMA update. No scaler,
per-microbatch clip, intermediate zeroing, second optimizer step, or gradient
rescaling beyond the registered `/4` effective-batch normalization is
permitted.

## Conjunctive gates

Stop at the first failed applicable conjunct. All inputs, intermediates,
metrics, margins, denominators, states, gradients, and counters must be finite;
strict ties fail. Observations use the frozen 495-row/eight-family selection
role, do not advance the schedule, and preserve model/optimizer/RNG hashes.

### Update 100 (1,600 presentations)

- Perception health: `A_100<A_0`, raster NLL decreases, semantic balanced
  accuracy is at least `0.60`, FREE recall at least `0.55`, OCCUPIED recall at
  least `0.30`, FREE/OCCUPIED gap at most `0.50`, rough balanced accuracy and
  rough OCCUPIED recall both strictly improve, and paired-RGB margin improves
  with wins in at least `6/8` families.
- Raw action NLL is strictly below `log(9)`; action macro balanced accuracy is
  strictly above `1/9`; and executed action beats its hardest wrong action in
  at least `1/8` families.
- Correct-next strict win rate over the fixed deranged next is strictly above
  `0.50` and its raw two-way NLL is strictly below `log(2)`.
- The RGB encoder and each predictor component listed in the update-0 witness
  has a finite, strictly positive L2 parameter displacement from initialization.

### Update 400 (6,400 presentations)

- Semantic balanced accuracy `>=0.80`, OCCUPIED recall `>=0.60`, rough
  OCCUPIED recall `>=0.55`, paired-RGB margin positive, and paired wins
  `>=6/8`.
- Raw action NLL `<0.98*log(9)`, action macro balanced accuracy `>=0.18`, and
  hardest-wrong positive margins in at least `3/8` families.
- Correct-next/deranged strict win rate `>=0.70`.
- EMA-target effective rank, channel variance, and spatial diversity are each
  finite, strictly positive, and at least `0.75` of their update-100 values.

### Update 1000 (16,000 presentations)

- Inherited strong perception: `A_1000<=A_400`, raster NLL
  `<=min(0.38,NLL_400+0.01)`, semantic balanced accuracy
  `>=max(0.80,BA_400-0.01)`, UNKNOWN recall `>=0.80`, FREE recall `>=0.75`,
  OCCUPIED recall `>=max(0.70,OCCUPIED_400-0.03)`, FREE/OCCUPIED gap `<=0.25`,
  rough balanced accuracy `>=max(0.772,rough_BA_400-0.01)`, rough OCCUPIED
  recall `>=max(0.65,rough_OCCUPIED_400-0.03)`, positive paired-RGB margin,
  and paired wins `8/8`.
- Anti-collapse: EMA-target effective rank, channel variance, and spatial
  diversity each retain at least `0.75` of their update-400 value.
- Raw action NLL `<0.95*log(9)`, action macro balanced accuracy `>2/9`, and
  executed action beats the hardest wrong in at least `6/8` families.
- Mean wrong-action energy strictly exceeds executed-action energy; on
  non-HOLD rows, HOLD energy strictly exceeds executed-action energy.
- Correct-next/deranged raw NLL `<0.95*log(2)`, strict win rate `>=0.70`, and
  positive correct-next margins in at least `6/8` families.
- The executed successor's mean unscaled local energy is at most `0.90` times
  persistence `mean(e_per)`, with strict successor-over-persistence wins in at
  least `6/8` families.
- An observation-only eight-step autoregressive rollout, started from every
  selection-row current latent and run separately for each of the nine
  constant actions, feeds each `p_a` directly as the next predictor state with
  no renormalization, and has every intermediate and final value finite. It
  uses no future RGB and performs no objective, backward, step, or EMA update.
- Every predictor component and the online encoder/lift remain displaced from
  initialization; target gradients/optimizer membership remain zero; all
  source, schedule, access, custody, optimizer, EMA, objective, backward,
  presentation, warning, and state-hash counters are exact.

Passing semantics, generic prediction, target contrast, or average wrong-
action separation cannot substitute for another conjunct.

## Exact work and receipts

One scheduled pair is one presentation. A complete attempt has exactly:

- 1,000 genuinely joint online optimizer updates and 1,000 EMA updates;
- 16,000 presentations, 4,000 microbatches, 4,000 combined objective
  evaluations, and 4,000 combined backward calls;
- 4,000 evaluations each of `S`, `P_successor`, `R_local_action`, and
  `C_deranged` (16,000 scalar-component evaluations);
- exactly eight non-mutating update-1 route-witness `autograd.grad` calls:
  four semantic and four combined-dynamics calls, and zero such calls at all
  later updates;
- 4,000 vectorized all-nine-action predictor calls, representing 144,000
  candidate-row successors but no extra presentations;
- 8,000 online encoder/lift training calls (current and next), 8,000 semantic-
  head calls, and 8,000 EMA encoder/lift target calls (next and fixed negative);
- 1,000 predictor updates, zero perception-only updates, zero predictor-only
  updates, zero separately trained predictor work, and no more than 30 active
  GPU minutes.

Observation and synthetic calls have separate counters and never enter those
totals. No caching observations into training, mixed precision, compile mode,
candidate reduction, retry, resume, schedule regeneration, or cap extension is
permitted.

Reservation precedes Torch/data/RGB/checkpoint/GPU access and consumes the sole
attempt. Public reservation, metrics, artifact, access, result, completed, and
complete failure receipts must record exact preregistration/source/review/
authorization/input/interpreter hashes; absent-root proof; process identity;
first failed gate and update; loaded/scheduled presentations; every forward,
objective, backward, optimizer, EMA, observation, warning, access, and write
count; last committed state hashes; and terminal control. Write-only
checkpoints may be emitted at updates 100, 400, and 1000, but may not be read,
resumed, selected, or used without later independent authority. Traces and
non-public payloads remain unopened by the terminal auditor.

Any failure consumes V1 and permits no retry, resume, alternate seed, depth,
width, action-query block, loss, normalization, schedule, threshold, data, or
same-root variant. A zero-presentation operational defect may be classified as
untested, but grants no automatic repair or replacement. A scientific failure
closes this Action-Query Spatial Successor mechanism. A complete pass qualifies
only the mechanism for independent terminal audit and separately planned
larger-scale work; it grants no checkpoint, navigation, G2, held-out, sealed,
production, promotion, or deployment authority.

| First terminal outcome | Decision |
|---|---|
| Source/U0 or first-update integrity failure | No scientific claim; V1 consumed/closed, no automatic replacement. |
| Update-100 gate failure | Early mechanism falsification; close V1 and this mechanism. |
| Update-400 gate failure | Intermediate mechanism falsification; close V1 and this mechanism. |
| Update-1000 conjunct failure or any cap breach | Complete mechanism failure; close V1 and this mechanism. |
| All update-1000 conjuncts pass | Mechanism-only qualification pending independent public-receipt audit; no downstream access or use. |

## Source/review/authority sequence

After this preregistration is reviewed and frozen: implement only the new
model/contract/runner/launcher/closure-checker/focused tests; run CPU-only
synthetic and source-closure checks; freeze a recursive source manifest and
source commit; obtain an independent science/source/custody review with zero
unresolved findings; obtain a distinct one-attempt machine authorization that
binds every source and runtime identity plus the requirement that the sole
output root be absent; have the launcher prove that absence immediately before
mode-`0700` reservation; execute once; then obtain a fresh independent audit
limited to public receipts. None of those later stages is authorized by this
document alone.

No generated input, dataset row, RGB/raster payload, N320 payload, checkpoint,
runtime output, trace, GPU, navigation, held-out, sealed, or rejected material
was opened to write this preregistration.
