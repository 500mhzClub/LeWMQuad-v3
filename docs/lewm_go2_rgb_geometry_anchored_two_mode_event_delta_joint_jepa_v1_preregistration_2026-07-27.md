# RGB geometry-anchored two-mode event-delta joint JEPA V1

Date: 2026-07-27

Status: exact scientific preregistration for document review only. This
document authorizes no source implementation, generated-input access,
checkpoint access, training, GPU use, execution, G2, navigation, held-out,
sealed, production, promotion, or deployment activity.

## Decision and repository goal

The repository goal remains a fully learned, RGB-only, perception-grounded
JEPA navigation stack, eventually validated on untouched externally custodied
held-out mazes.

Preregister one possible future probe that preserves the successful
geometry-anchored RGB representation and predicts a fixed-scale two-mode
energy distribution over its next-minus-current latent change:

1. `ZERO_EVENT` has mean exactly zero; and
2. `LEARNED_EVENT` has a learned state-and-action-conditioned mean.

A learned spatial event logit mixes those fixed-identity modes. There is no
learned variance, scale head, codebook, pair-posterior network, inverse
classifier, transport, flow, warp, correspondence, or third mode. The mixture
is a conditional energy model, not a claim of calibrated Gaussian or
aleatoric uncertainty.

## Frozen evidence and scientific distinction

The immediate predecessor is closed by:

`docs/lewm_go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_transport_joint_jepa_v1_terminal_audit_2026-07-27.json`

- file SHA-256:
  `38d65b46bd4ff83ab67924233badb96c37bd079f0b85b36c69512c905557f25e`;
- byte count: `19,570`;
- content SHA-256:
  `5a87a441560c8ab397a0cdfab5ca08fc1c7ad7b8294d323194f712ad834821ca`;
- frozen audit commit:
  `293a02e1cc99c3b5aac876efc68104093468506c`;
- classification: valid complete-budget scientific failure with no
  operational or integrity defect; and
- consumed work: 1,000 updates, 16,000 presentations, 600 joint updates, and
  600/600 passed shared-gradient gates.

At update 1,000, rigid transport produced action macro balanced accuracy
`0.1680282926443815`, action NLL `2.189596249358823`, and `0/8`
hardest-wrong family wins. It made action identity worse than V3 while generic
prediction remained strong. That audit permanently closes every transport,
warp, transform-bank, flow-bank, bound, padding, residual-depth, and
transport-loss variant.

This probe is not another deterministic delta or inverse-classification retry:

- Direct-BEV V5 predicted EMA current-to-next probability-state deltas. At
  update 100 it had action macro balanced accuracy `0.10911680911680913`,
  action NLL `2.1971453213932537`, and `0` hardest-wrong family wins.
- Patch-whitened residual V1-V5 covered residual prediction, action gain,
  all-action energy, and state-dependent latent flow. V5 improved coarse
  separation, but the hardest wrong action and action-free mean target still
  beat the executed prediction.
- Masked pair-tubelet V11-V13 was already a genuine joint RGB JEPA. At update
  400 it had action macro balanced accuracy `0.1284468596805172`, action NLL
  `2.1970160007476807`, hardest-wrong ratio `1.0802552700042725`, and a JEPA
  loss worse than initialization.
- Patch-whitened V6's inverse classifier remained exactly at `1/9` macro
  balanced accuracy with NLL above `log(9)`.

The sole new hypothesis is that deterministic prediction averaged together
no-visible-change and visible-change outcomes. The registered change is the
fixed-identity two-mode predictive state, coupled to change/static balancing
and explicit same-command context and state/action ablations. Merely changing
a delta loss, temperature, or predictor depth is not this mechanism.

No predecessor checkpoint, tensor, optimizer state, trace, observation,
registry, RNG state, receipt payload, or runtime output may be opened or
reused.

## Scientific question

Can an end-to-end jointly trained RGB encoder and two-mode event-delta
predictor distinguish executed action consequences from persistence, generic
temporal matching, and action-only templates while retaining geometry-grounded
perception?

Only the complete conjunctive update-1,000 gate answers this question. No
single loss, semantic, target-contrast, mode-use, or average action metric can
qualify the mechanism alone.

## Exact preserved envelope

Except for the predictor, predictive target, joint objective, and explicitly
registered observations below, preserve geometry-anchored V3 exactly:

- current RGB input `(B,3,112,112)`, float32; patch size `7`; encoder width
  `192`, depth `6`, and six heads;
- exact N320 encoder-state-only initialization;
- fixed level-camera projective anchors, four-sample bounded deformable lift,
  `64` latent channels, and `64 x 64` BEV lattice;
- inherited local lift refinement and
  `Conv2d(64,3,kernel_size=1,bias=True)` semantic head with class order
  `UNKNOWN`, `FREE`, `OCCUPIED`;
- online encoder/lift and requires-grad-false EMA encoder/lift target,
  hard-synced once and updated after every successful optimizer step at
  momentum `0.996`;
- exact nine-action vocabulary and order;
- exact train and checkpoint-selection roles, rows, endpoints, scene
  families, mapped negatives, loader, and schedule;
- only `current_rgb`, `next_rgb`, `fixed_negative_rgb`, and
  `raster_labels.u1` loaded per row; raster labels are the only supervision
  array and are never inference inputs;
- model seed `20260712`, schedule seed `20260713`, microbatch `4`, four
  microbatches per update, and effective batch `16`;
- 400 perception-only updates followed by 600 genuine joint updates;
- one update-zero AdamW instance, betas `(0.9,0.999)`, epsilon `1e-8`, weight
  decay `1e-4`, float32, learning rates `1e-4` for encoder and `3e-4` for
  lift/semantic and predictor groups;
- separate representation/semantic and predictor L2 clips of `1.0`;
- observations at updates `0`, `100`, `400`, and `1000`, with write-only
  checkpoints only at updates `100`, `400`, and `1000`; and
- caps of 1,000 updates, 16,000 presentations, and 30 active GPU minutes.

Construction is fresh. N320 encoder state is the sole pretrained state. No
data render, rebuild, filter, resample, remap, new scene, new label, history
frame, held-out role, or sealed role is permitted.

## Exact latent target

Use NCHW ordering. Let per-cell channel LayerNorm have no learned affine
parameters:

```text
N(Z) = layer_norm(Z.movedim(1,-1), (64,), eps=1e-5).movedim(-1,1)
x     = N(Z_online(current_rgb))
t0    = stopgrad(N(Z_EMA(current_rgb)))
t1    = stopgrad(N(Z_EMA(next_rgb)))
tn    = stopgrad(N(Z_EMA(fixed_negative_rgb)))
d     = t1 - t0
d_neg = tn - t0
```

`x`, `t0`, `t1`, `tn`, `d`, and `d_neg` each have exact shape
`(B,64,64,64)` and dtype float32. The predictor receives only `x` and the
commanded action. It never receives a future or target tensor, future RGB,
raster label, pose, depth, flow, odometry, map, or goal.

Every call to `N` uses `torch.nn.functional.layer_norm`, normalizes only the
last moved channel axis of length `64`, has no affine weight or bias, uses
exact epsilon `1e-5`, preserves float32, and is tested against an independent
reference. No spatial axis is normalized.

The target is therefore learned entirely from RGB through the stop-gradient
EMA branch. It is a latent event delta, not a privileged motion label.

## Exact predictor and tensor shapes

For one current state and one action:

1. `Embedding(9,16)` produces `(B,16)` and broadcasts to
   `(B,16,64,64)`.
2. Concatenate with `x` to form `(B,80,64,64)`.
3. Apply `Conv2d(80,64,kernel_size=3,stride=1,padding=1,bias=True)`, then
   GELU.
4. Apply two shared residual blocks, each exactly
   `v + Conv2(GELU(Conv1(v)))`; all four convolutions are
   `Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True)`.
5. Apply two heads to the same final trunk:
   - `event_mean_head` is
     `Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True)`; and
   - `event_logit_head` is
     `Conv2d(64,1,kernel_size=3,stride=1,padding=1,bias=True)`.

One-action outputs are:

```text
mu_event   : (B,64,64,64)
event_logit: (B,1,64,64)
```

The all-action vectorized outputs are:

```text
mu_event_all   : (B,9,64,64,64)
event_logit_all: (B,9,1,64,64)
```

Mode identities are exact and permanent:

```text
mu_ZERO_EVENT    = zeros_like(mu_event)
mu_LEARNED_EVENT = mu_event
```

`ZERO_EVENT` owns no trainable mean. Positive event logit always denotes
`LEARNED_EVENT`; labels may not permute. All weights are shared across actions
except the nine action-embedding rows. There is no scale parameter, action-
specific convolution, mode-specific trunk, coordinate channel, pooling,
attention, transport, warp, flow, grid sample, or alternate prediction path.

The predictor inventory is exactly `231,505` trainable parameters in exactly
`15` trainable tensors:

- action embedding: `144`;
- input projection: `46,144`;
- four residual-block convolutions: `147,712`;
- event-mean head: `36,928`; and
- event-logit head: `577`.

The exact ordered parameter names are:

```text
predictor.action_embedding.weight
predictor.input_projection.weight
predictor.input_projection.bias
predictor.residual_blocks.0.conv1.weight
predictor.residual_blocks.0.conv1.bias
predictor.residual_blocks.0.conv2.weight
predictor.residual_blocks.0.conv2.bias
predictor.residual_blocks.1.conv1.weight
predictor.residual_blocks.1.conv1.bias
predictor.residual_blocks.1.conv2.weight
predictor.residual_blocks.1.conv2.bias
predictor.event_mean_head.weight
predictor.event_mean_head.bias
predictor.event_logit_head.weight
predictor.event_logit_head.bias
```

## Exact fixed-scale event energy

For a target delta `y`, define cell energies with Smooth-L1 beta `1` and
channel mean:

```text
e0(y)[b,h,w] = mean_c(smooth_l1(y[b,c,h,w], 0, beta=1))
e1(y)[b,h,w] = mean_c(
    smooth_l1(y[b,c,h,w], mu_event[b,c,h,w], beta=1)
)
```

For all actions, the shape is `(B,9,64,64)`. Both terms are finite and
nonnegative. No learned scale, variance, temperature, or normalization may
alter them. Before mixing, remove only the known singleton channel:

```text
ell     = event_logit.squeeze(1)      # (B,64,64)
ell_all = event_logit_all.squeeze(2)  # (B,9,64,64)
```

Reject every other event-logit or cell-energy shape. In particular, implicit
right-aligned broadcasting between a singleton-channel logit and a
channel-free energy is forbidden.

Use the one frozen positive update-400 temperature `T400` defined below and
compute the mixture only in log space:

```text
log_p0 = logsigmoid(-ell)
log_p1 = logsigmoid( ell)
e_mix(y) = -T400*logaddexp(
    log_p0 - e0(y)/T400,
    log_p1 - e1(y)/T400
)
```

Never exponentiate probabilities before the log-add-exp. If `mu_event=0`,
then `e0=e1` and the identity `e_mix=e0` must hold within float32
`atol=2e-6, rtol=2e-6`, independently of the event logit. This exact
persistence identity is mandatory.

The analytic target-posterior event responsibility is observation-only and
uses the stable two-logit form:

```text
posterior_log_odds = (log_p1 - e1(y)/T400)
                     - (log_p0 - e0(y)/T400)
q_event(y) = sigmoid(posterior_log_odds)
```

It receives no separate target or loss and is never produced by a pair
encoder. Reject any nonfinite energy, responsibility, gradient, temperature,
or denominator before an optimizer step.

## Frozen update-400 temperature, change weighting, and baseline

Updates 1-400 perform no training predictor forward, objective, backward,
optimizer-state creation, or update. Observation-only calls are separately
counted. At the passed update-400 checkpoint-selection
observation, operate under `eval()` and `no_grad()` without mutating model,
optimizer, CPU RNG, or accelerator RNG.

This is the inherited authorized U400 calibration role already used by V3 to
freeze its joint-phase persistence normalization. It preserves the exact
predecessor checkpoint-selection population, order, and reduction, makes no
selection or promotion decision, and introduces no new data role. The only
change is that the target-derived calibration is split into raw cell
temperature `T400` and balanced persistence normalization `B400`.

For the exact 495 checkpoint-selection rows in frozen order, compute:

```text
d400[b,c,h,w] = N(Z_EMA(next))[b,c,h,w]
                - N(Z_EMA(current))[b,c,h,w]
e_persist400  = mean_c(smooth_l1(d400,0,beta=1))
T400          = mean_{b,h,w}(e_persist400)
w400          = e_persist400/(e_persist400 + T400)
```

Accumulate scalar summaries in float64 over float32 tensor values in exact
population, row, and spatial order. `T400` must be finite and strictly
positive. During joint training and later observations, compute only:

```text
e_persist = mean_c(smooth_l1(d,0,beta=1))
w = e_persist/(e_persist + T400)
```

`T400` is the raw, unbalanced selection mean of cellwise persistence energy.
It is never recomputed, adapted, annealed, or differentiated. The
correct-target `w` is reused unchanged for wrong actions, `d_neg`, context
swaps, and all ablations.

For any cell energy `e`, define per-row:

```text
E_changed(e,w) = sum_hw(w*e)/sum_hw(w)
E_static(e,w)  = sum_hw((1-w)*e)/sum_hw(1-w)
E_bal(e,w)     = 0.5*E_changed + 0.5*E_static
```

Each denominator must be finite and strictly above `1e-6`; otherwise stop.
No epsilon is added to a valid denominator. Equal changed/static weighting is
the registered defense against static-background domination.

After first forming each row's weighted spatial means, freeze the second and
only other target-derived scalar:

- `B400 = mean_rows(E_bal(e_persist400,w400))`.

`T400` sets only mixture temperature and soft change weighting. `B400` sets
only loss normalization and the terminal persistence comparison. Both must be
finite and strictly positive and are bound with one canonical joint content
hash before update 401. The actual small-nonzero initialized `mu_event` is not
used to define either scalar. When `mu_event=0`, the cell mixture equals
`e_persist` for any positive `T400`, so its population balanced energy equals
`B400` exactly; this independently witnesses that neither reference is
circular.

The inherited full-state target effective-rank, channel-variance, and spatial-
diversity baselines are also retained unchanged for representation gates, but
they do not parameterize the event energy or change weighting.

The observer must save and restore module modes and CPU/accelerator RNG states.
Pre/post online-model, target-model, optimizer, and RNG hashes must be exact.
It may not advance the schedule or presentation count.

## Exact objectives and genuine joint training

The inherited semantic term remains:

```text
A = 0.5*A_current + 0.5*A_next
S = A/log(3)
```

Each side is the equal-row mean of equal-present-final-class macro NLL over
`UNKNOWN`, `FREE`, and `OCCUPIED`.

Updates 1-400 use only:

```text
L_warmup = S
```

For updates 401-1000, evaluate all nine actions against correct `d`, apply the
frozen `w`, and let `E_a` be `(B,9)` balanced forward energies. Select
`E_exec` by executed action and define:

```text
scale_action = stopgrad(mean_a(E_a,keepdim=True)).clamp_min(1e-6) # (B,1)
P_event = mean_b(E_exec)/B400
raw_action_ce = cross_entropy(-E_a/scale_action, executed_action)
R_action = raw_action_ce/log(9)
```

All-action CE is derived mechanically and only from forward event-delta
energies. It owns no logits, classifier parameters, pair encoder, or inverse
head and is never a standalone causal mechanism.

Score the executed distribution against `d_neg`, retaining correct-target
`w`, to get `E_neg`:

```text
target_logits = stack(
    -E_exec/scale_action.squeeze(1),
    -E_neg/scale_action.squeeze(1),
    dim=1
) # (B,2)
raw_target_ce = cross_entropy(target_logits,label=0)
C_target = raw_target_ce/log(2)
```

The inherited fixed-negative mapping is immutable, same-role, same-scene, and
deranges the next endpoint without changing the row's executed action. For
the context control, only encode
`fixed_negative_rgb` through the online encoder/lift under `no_grad()`, apply
`N`, and detach the resulting `x_fixed`. The subsequent predictor forward
`predict(x_fixed,a_exec)` remains autograd-enabled, contributes gradients to
the action embedding, trunk, event-mean head, and event-logit head, and is
counted. Score it against original `d` and `w` to get `E_context_swap`:

```text
context_logits = stack(
    -E_exec/scale_action.squeeze(1),
    -E_context_swap/scale_action.squeeze(1),
    dim=1
) # (B,2)
raw_context_ce = cross_entropy(context_logits,label=0)
C_context = raw_context_ce/log(2)
```

Final action-NLL and target/context-NLL thresholds apply to the corresponding
raw cross-entropies, never to `R_action`, `C_target`, or `C_context` after
division by `log(9)` or `log(2)`.

The exact joint objective is:

```text
L_joint = S + P_event + R_action + C_target + C_context
```

Every coefficient is `1.0`. Add no mode-balance, entropy, inverse, variance,
flow, transport, HOLD, margin, adaptive-weight, annealing, or auxiliary loss.

The online RGB encoder, lift, semantic head, action embedding, shared trunk,
event-mean head, and event-logit head update together on every joint step. The
predictor is never trained behind a frozen encoder. The EMA target receives no
gradient and updates once after each successful online optimizer update.

## Exact initialization and RNG order

Use the inherited one CPU default-generator scope: save caller CPU RNG, seed
exactly once with `20260712` immediately before deformable-lift construction,
continue the stream through semantic head and predictor, then restore caller
CPU RNG. Make no accelerator RNG call during construction.

Encoder, lift, local refinement, null evidence, and semantic head must be
byte-identical to V3 before predictor construction. Construct predictor
modules exactly in this order:

1. `Embedding(9,16)`: allow its ordinary constructor draw, then overwrite
   weight with exact zeros without an RNG draw;
2. input projection with ordinary PyTorch Conv2d initialization;
3. residual block 0 `conv1`, then `conv2`, ordinary initialization;
4. residual block 1 `conv1`, then `conv2`, ordinary initialization;
5. event-mean head: allow its constructor draw, then overwrite weight with
   `normal_(mean=0,std=1e-3)` from the continuing CPU generator and bias with
   exact zero; and
6. event-logit head: allow its constructor draw, then overwrite weight and
   bias with exact zeros.

Then create EMA encoder/lift copies, hard-sync once, and set every target
parameter `requires_grad=False`.

At update zero:

- every action embedding is exact zero;
- all nine distributions are bitwise equal for a given current state;
- event logits are zero, so mode priors are exactly `0.5`; and
- the small nonzero mean head preserves a finite nonzero joint-loss path
  through the shared trunk into the online encoder at update 401.

Because real event energy requires the not-yet-frozen `T400`, update zero does
not compute a runtime event energy, action NLL, or action balanced accuracy.
Source-only tensors witness the energy identity, `log(9)`, and `1/9` for an
arbitrary finite positive synthetic temperature. The actual frozen-population
energy symmetry, NLL, and balanced-accuracy checks occur exactly once after
`T400`/`B400` freeze at update 400 and before update 401.

A source-only synthetic nonzero-delta witness must prove finite nonzero first-
joint-step gradients for online encoder/lift, action embedding, input
projection, all four residual-block convolutions, mean head, and logit head.
EMA target gradients remain absent. The witness uses constructed CPU tensors
only and performs no optimizer step.

## Exact observation-only ablations

All ablations use the exact checkpoint-selection population under `eval()` and
`no_grad()`, preserve model/RNG hashes, and never alter training.

### Matched single-mean delta

Collapse the two modes to their prior mean without new parameters:

```text
p_event = sigmoid(event_logit)
mu_match = p_event*mu_event
e_single = mean_c(smooth_l1(d,mu_match,beta=1))
```

Apply the same `w`. The mixture must beat this matched deterministic mean;
otherwise the second mode has added no predictive value.

### Component specialization and learned prior

For the executed action and correct target, retain the unmixed component cell
energies and the learned prior:

```text
p_event = sigmoid(ell_exec)
zero_changed  = E_changed(e0,w)
event_changed = E_changed(e1,w)
zero_static   = E_static(e0,w)
event_static  = E_static(e1,w)
prior_changed = E_changed(p_event,w)
prior_static  = E_static(p_event,w)
mu_changed_abs_row = E_changed(mean_c(abs(mu_event_exec)),w)
prior_mean_row = mean_hw(p_event)
prior_spatial_variance_row = mean_hw(
    (p_event-prior_mean_row)^2
) # population variance; no Bessel correction
prior_context_difference_row = mean_hw(abs(p_event-p_event_fixed))
mixture_overall_row = E_bal(e_mix_exec,w)
zero_overall_row = E_bal(e0,w)
event_overall_row = E_bal(e1,w)
```

These are observation metrics only; `w` remains the detached EMA-latent soft
weight and is not an event label or an auxiliary classification target. Also
record the corresponding aggregate and family values as the already
registered equal-row means. This distinguishes a learned event occurrence
prior from posterior separation created only after seeing the target.

### Same-command context swap

Use the already loaded `fixed_negative_rgb` as alternative current context,
hold the executed command fixed, and score original `d` and `w`. No new input
or mapping is introduced. Also record the mean absolute difference between
the true-context and fixed-context `p_event` maps; the predictor forward on
the detached alternative context is not itself detached.

### State-removed action template

All nine frozen action groups must be nonempty, with their row counts and
order asserted against the checkpoint-selection contract. For each group,
accumulate the full spatial normalized online-current tensor in float64 and
cast once to float32:

```text
x_template[a,c,h,w] = mean_{row:executed_action[row]=a} x[row,c,h,w]
```

Replace each row's `(64,64,64)` state by `x_template[a]`, apply its executed
action, and score its original `d` and `w`. No global scalar, pooled vector,
or cross-action mean is allowed. This preserves an action-specific spatial
template while removing row-specific state.

### Action-removed state-only mixture

Marginalize the nine already computed cell energies with a fixed uniform
action prior, without mutating embeddings:

```text
e_state_only = -T400*logsumexp_a(-e_mix_action/T400 - log(9))
```

Apply the same `w`. The executed-action distribution must beat this generic
state-only temporal mixture.

Together with hardest-wrong and HOLD controls, these ablations distinguish
action identity from generic temporal matching and action-only templates.

## Conjunctive gates

Stop at the first applicable failure. Ties fail strict comparisons.

Unless a metric explicitly names an ablation or negative target, event/action
metrics use the executed action and correct `d`. Cell values reduce to rows by
the registered `E_changed`, `E_static`, or `E_bal` formula; rows reduce to the
aggregate by equal-row mean; and a scene-family value is the equal-row mean
within that frozen family. A positive family win requires a strictly positive
mean margin; a tie fails. The hardest wrong is the minimum-energy non-executed
candidate, with the lowest frozen action index used only as deterministic
reporting tie-break and a zero margin still failing. Action balanced accuracy
is the unweighted mean of the nine per-class recalls over the frozen nonempty
action populations. Every denominator must satisfy its registered lower bound
and every input, intermediate, row result, aggregate, and family result must
be finite; otherwise the applicable gate fails closed.

An unqualified event energy means its registered `E_bal` row value. An
unqualified prior or responsibility mean means `mean_hw` for the row followed
by the same equal-row aggregate/family reduction. Changed/static qualifiers
always use `E_changed`/`E_static` instead.

### Update 0: structural, zero presentations

Require exact source/input declarations, inventories, shapes, parameter count,
RNG order, hard sync, mode identities, output-parameter action symmetry,
synthetic-positive-temperature stable energy, persistence identity, synthetic
`log(9)` NLL and `1/9` macro balanced accuracy, synthetic gradients, target
isolation, and zero forbidden-input/bypass counts. No actual-population event
energy is computed before `T400` exists. Training predictor work and predictor
optimizer-state counts are zero. Observation-only predictor calls are
separately counted and may not increment training, presentation, objective,
backward, or optimizer counters.

All nine candidate parameter outputs must be bitwise equal at update zero.
`p_event` must be bitwise equal to float32 `0.5`.

### Update 100: inherited perception health, 1,600 presentations

Copy the V3 update-100 perception gate unchanged: `A_100<A_0`, raster
`NLL_100<NLL_0`, balanced accuracy at least `0.60`, FREE recall at least
`0.55`, OCCUPIED recall at least `0.30`, FREE/OCCUPIED gap at most `0.50`,
rough balanced accuracy and rough OCCUPIED recall strictly improve,
paired-RGB margin improves with at least `6/8` family wins, online/EMA updates
equal 100, and every predictor-work count remains zero.

### Update 400: inherited perception gate and frozen references

Copy the complete V3 update-400 perception and anti-collapse gate unchanged.
Additionally require the exact read-only calculation, positivity, joint hash
binding, and model/optimizer/RNG preservation of the distinct `T400` and
`B400` scalars. Training predictor work and predictor optimizer-state counts
remain zero. After both scalars freeze and before update 401, evaluate the actual
checkpoint-selection action candidates once: all nine candidate energies must
be bitwise equal per row; raw action CE must match Python `log(9)` within
float32 `atol=2e-6, rtol=2e-6`; deterministic lowest-index `argmin` over the
frozen nonempty nine-class population must give macro balanced accuracy equal
to Python `1/9` within absolute tolerance `1e-12`; and event priors remain
bitwise float32 `0.5`. These are observation-only calls. Failure closes the
attempt before joint training.

### Update 401: joint integrity

Require the same optimizer identity and membership as update zero, exact unit-
weighted objective arithmetic, action logits mechanically derived from
forward energies, finite nonzero semantic and dynamics gradients to shared
online representation, finite nonzero gradients to every predictor submodule,
both semantic/dynamics gradient ratios in `[1/32,32]`, no target gradient, and
exactly one predictor/joint optimizer update.

### Update 1,000: complete qualification

Retain every inherited final perception, target-retention, integrity,
accounting, access, warning, and custody conjunct, including:

- `A_1000<=A_400`;
- raster NLL `<=min(0.38,NLL_400+0.01)`;
- raster balanced accuracy `>=max(0.80,BA_400-0.01)`;
- UNKNOWN recall `>=0.80`, FREE recall `>=0.75`, OCCUPIED recall
  `>=max(0.70,OCCUPIED_400-0.03)`, and FREE/OCCUPIED gap `<=0.25`;
- rough balanced accuracy `>=max(0.772,rough_BA_400-0.01)` and rough OCCUPIED
  recall `>=max(0.65,rough_OCCUPIED_400-0.03)`;
- paired-RGB margin positive and wins in `8/8` families; and
- target rank, channel variance, and spatial diversity each retain at least
  `75%` of update-400 values.

The event/action mechanism must also pass all of. References to action,
target, or context NLL below mean raw cross-entropy before objective
normalization:

1. mean executed `E_bal<=0.90*B400`;
2. action NLL strictly below `0.95*log(9)`;
3. action macro balanced accuracy strictly above `2/9`;
4. executed action beats hardest wrong in at least `6/8` families;
5. mean wrong-action energy strictly exceeds mean executed energy;
6. on non-HOLD rows, mean HOLD energy strictly exceeds executed energy;
7. correct/deranged target NLL strictly below `0.95*log(2)`, strict win rate
    at least `0.65`, and positive margins in at least `6/8` families;
8. true-context energy at most `0.95` times same-command swapped-context
    energy, true-context win rate at least `0.65`, and positive margins in at
    least `6/8` families;
9. true-state energy at most `0.95` times state-removed action-template
    energy and positive margins in at least `6/8` families;
10. executed-action energy at most `0.95` times action-removed state-only
    energy and positive margins in at least `6/8` families;
11. two-mode energy at most `0.98` times matched-single-mean energy and strict
    wins in at least `6/8` families;
12. learned-event component changed energy is at most `0.90` times zero-event
    changed energy, with strict event-over-zero wins in at least `6/8`
    families;
13. zero-event component static energy is at most `0.95` times learned-event
    static energy, with strict zero-over-event wins in at least `6/8`
    families;
14. the two-mode mixture strictly beats each unmixed component overall and
    beats ZERO_EVENT alone and LEARNED_EVENT alone in at least `6/8` families
    each;
15. changed-weighted mean absolute `mu_event` is strictly above `1e-4`;
16. learned-prior `p_event` changed mean exceeds its static mean by at least
    `0.05`, aggregate prior mean lies in `[0.10,0.90]`, mean per-row spatial
    population variance (the registered correction-`0`, no-Bessel formula) is
    at least `1e-4`, and the true/fixed-context prior maps differ by mean
    absolute value at least `0.02` with positive context-difference means in
    at least `6/8` families;
17. target-posterior mean `q_event`, using the registered soft
    `E_changed(q_event,w)` and `E_static(q_event,w)` reductions, has changed
    mean at least `0.10` above static mean;
18. aggregate target-posterior event responsibility `mean(q_event)` lies in
    `[0.10,0.90]`, and `q_event` and `1-q_event` each have mean at least
    `0.05` in at least `6/8` families;
19. all 600 joint updates pass both shared-gradient ratios in `[1/32,32]`,
    and the action embedding, shared predictor trunk, event-mean head, and
    event-logit head each have finite nonzero dynamics-gradient counts of
    exactly `600`; and
20. every model, optimizer, EMA, objective, backward, presentation, warning,
    state-hash, access, and custody count is exact.

Passing generic prediction, semantics, target contrast, mode use, or average
action separation cannot substitute for any other conjunct.

## Exact work accounting

One scheduled pair is one presentation. Observation and source-synthetic work
are not presentations and may not advance the schedule. A complete attempt is:

- 1,000 online optimizer updates and 1,000 EMA updates;
- 16,000 presentations;
- 4,000 microbatch objective evaluations and combined backward calls;
- 1,600 warmup and 2,400 joint microbatch objectives/backwards;
- 4,000 semantic-term evaluations and 2,400 evaluations each of `P_event`,
  `R_action`, `C_target`, and `C_context`, for 13,600 registered scalar-term
  evaluations while retaining exactly 4,000 combined objectives/backwards;
- 600 predictor/joint optimizer updates;
- 2,400 all-nine-action predictor calls during joint training;
- 2,400 executed-action context-swap predictor calls during joint training;
- 10,400 online encoder/lift training calls: two semantic calls in every one
  of 4,000 microbatches plus one detached fixed-context call in each of 2,400
  joint microbatches;
- 8,000 semantic-head training calls, two per microbatch;
- 7,200 EMA target encoder/lift training calls: current, next, and fixed
  negative once in each of 2,400 joint microbatches;
- zero training predictor work during updates 1-400;
- exactly 600 passed shared-gradient gates for completion; and
- no more than 30 active GPU minutes.

Keep scalar-term, combined-objective, backward, forward, observation,
synthetic, and training counters distinct. Every observation records its
pair/endpoint population, microbatch count, and online/EMA/predictor forward
counts separately; none enters the exact training totals above. Observation
passes may not be cached into training. Keep one combined backward per
microbatch.
No caching across scheduled rows, mixed precision, compile mode, reduced
candidate set, accumulation change, resume, or cap extension is permitted.

## Warning policy

Enable deterministic algorithms with warn-only scope, set cuDNN deterministic
true and benchmark false, and record every warning. Permit only `UserWarning`
with this exact base message:

```text
grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
```

It may have only one suffix of the exact form:

```text
 (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:<decimal-line>.)
```

Canonicalize only that suffix. Reject every other category or text. Record raw
and canonical counts and SHA-256 sets. Warning finalization must follow return
from the scientific callable and may not erase an already computed gate. A
warning never authorizes retry. The permitted warning originates only in the
preserved deformable RGB lift; the predictor has no grid sampler.

## Required source-only review before execution authority

Any future implementation must first pass independent source-only and CPU-
synthetic review proving:

- inherited encoder/lift/semantic identity before predictor construction;
- exact predictor inventory, names, shapes, order, initialization, parameter
  count, and update-zero symmetry;
- fixed mode identities and absence of a learned zero-event mean;
- independent Smooth-L1, stable log-mixture, posterior, weighting, baseline,
  objective, and ablation references;
- exact singleton-channel squeeze and per-row action-scale broadcasting, with
  shape-failure tests that would otherwise create a cross-batch axis;
- exact `mu_event=0` persistence identity;
- all-action permutation equivariance when embedding rows and one-hot columns
  are permuted together, with no action-name inspection;
- finite nonzero update-401 gradients to all online groups and no target
  gradient;
- an autograd-enabled predictor call on a detached fixed-context latent,
  proving context-control gradients reach every predictor submodule but not
  the detached alternative encoder path;
- constant-state/action-template fixtures fail context/state gates and
  action-ignored fixtures fail action/state-only gates;
- exact optimizer, clip, EMA, objective, backward, observation, warning, and
  scalar-safe state-hash accounting;
- no warmup predictor work or frozen-encoder predictor-fitting path;
- no scale head, inverse head, pair posterior, codebook, transport, warp,
  flow, correspondence, pose, depth, map, future-input, label-input, or
  checkpoint/trace read path; and
- source discovery and clean export obey repository custody rules.

Synthetic checks may use CPU Torch and constructed tensors only. They may not
open dataset rows, RGB, rasters, N320, generated inputs, predecessor outputs,
checkpoints, traces, GPU state, held-out, sealed, or rejected material.

## Attempt, receipts, and closure

Experiment ID:

`geometry_anchored_two_mode_event_delta_joint_jepa_v1`

Sole prospective output root:

`.generated/go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1/attempt_v1`

The root must be absent before a write-once mode-`0700` reservation. One fresh
attempt may be authorized only after preregistration freeze, recursive source
closure, independent review with zero findings, and a separate committed
machine authorization binding source, review, interpreter, runtime inputs,
schedule, and absent root.

Normal terminal receipts are exactly `reservation.json`, `metrics.json`,
`artifact.json`, `access.json`, `result.json`, and `completed.json`. An
operational exception writes complete `failure.json` and `completed.json`. A
normal scientific gate failure writes the normal result chain and no
`failure.json`.

Receipts must be canonical ASCII finite duplicate-safe JSON with one trailing
newline, content self-hash, exact byte/file bindings, and consistent attempt,
status, accounting, access, warning, checkpoint, and terminal controls.
Finalize files mode `0444` and root mode `0555`. Every terminal path records
the first failure, all work counts, active GPU seconds, source/input/state
bindings, warnings, access counts, checkpoint bindings, read-after-write
counts, and downstream-authority result.

Checkpoints at 100, 400, and 1,000 and the trace remain write-only. They cannot
be reopened unless a later independent terminal audit authorizes one exact
binding. Failure disqualifies all checkpoints.

There is no retry, resume, alternate seed, extension, same-root reuse, second
attempt, repair, threshold change, coefficient change, temperature change,
mode-count change, or integrity replacement authorized here. Stop at the first
scientific, numerical, integrity, accounting, custody, warning, or cap failure.

Failure after any scientific presentation permanently closes the two-fixed-
mode event-delta family, including event-logit, temperature, changed/static
weighting, head-depth, coefficient, and threshold variants. No larger
codebook, extra mode, pair posterior, or distributional successor follows
without materially new evidence and a separate argument. A zero-presentation
operational defect also grants no retry; even a science-identical replacement
requires new explicit preregistration and authority.

A complete pass authorizes only independent terminal audit and a later
decision. It does not authorize checkpoint reading, G2, navigation, held-out,
sealed, production, promotion, or deployment access.

## Registered risks

- One learned nonzero mode may still average distinct visible outcomes.
- One observed future per state-action pair cannot establish calibrated
  stochastic uncertainty.
- The zero-event component may dominate despite changed/static balancing.
- The event mean may become an action template; context, state-removal,
  state-only, and hardest-wrong gates are decisive.
- Joint event gradients may erode occupied-space perception.
- Extra EMA-current and context-control encodes increase work, but caps and
  schedule may not change.

These are falsification risks, not permissions for variants. Only the complete
conjunctive pass can earn the next source-only decision.
