# Go2 RGB memory-role factorized joint JEPA V1

Date: 2026-07-30

Status: exact scientific preregistration. This document authorizes source
implementation, source-only tests, metadata-only preflight, and preparation of
a reviewed narrow export. It does not by itself authorize training, GPU use,
checkpoint access, navigation, G2, held-out, sealed, benchmark, promotion,
production, or deployment activity. A later hash-bound one-shot authority is
required for execution.

## Decision and hypothesis

V28 is a valid terminal scientific failure at commit
`a5a25247d9804198125e17f000315a441089728d`. It completed exactly 400 joint
updates and 12,800 presentations, but its deterministic four-action endpoint
prediction remained `3.461591524447307` times worse than copying the current
EMA state. Wrong-scene discrimination passed, while persistence, wrong-plan,
tail-family robustness, and the frozen rough-depth gate failed. V28 published
no checkpoint and may not be retried, resumed, extended, or used as
initialization.

V1 tests a materially different division of labour. A navigation stack with a
separate learned memory does not require one perception state to be an exact
four-step future map. It does require perception to expose:

- a compact key that recognizes a repeated local observation for memory
  lookup; and
- a spatial state whose immediate change depends on the commanded action.

One shared RGB encoder is therefore trained jointly with two role projections
and two predictors. This is one JEPA optimization process, not a frozen encoder
followed by a separately fitted predictor. The probe deliberately contains no
recurrent memory: it first falsifies whether the interfaces needed by such a
memory and local controller can be learned at all.

## Exact mechanism

Reuse the reviewed V18 RGB-to-object-space state `z = phi(x)` with shape
`(64,64,64)` and its EMA target `phi_bar`. Add exactly:

- `h_place(z)`: global mean over a learned 64-channel projection followed by a
  learned 64-to-64 map and L2 normalization, producing `k` with shape `(64,)`;
- `g_place(k)`: a 64-to-128-to-64 normalized predictor with no action input;
- `h_local(z)`: two stride-two learned convolutions, producing `l` with shape
  `(32,16,16)`; and
- `g_local(l,a)`: a spatial predictor modulated by one exact one-hot action from
  the unchanged nine-action vocabulary.

The role factorizer has an exact frozen EMA copy. For one local row,

`p_local = g_local(h_local(phi(x_t)), a_t)`

`t_local = stopgrad(h_local_bar(phi_bar(x_t+1)))`.

The correct local energy is mean squared error. A cyclic wrong action
`(a_t + 1) mod 9` supplies a fixed `0.05` energy-gap hinge. The local loss is
correct energy plus that hinge.

For one place triplet, the anchor and positive are distinct registered RGB
images from the same scene, cell, and yaw bin, separated by another
environment/episode stream or at least four seconds. The hard negative has the
same scene and yaw bin but a different cell. Cell/yaw values select the triplet
offline and never enter the model. The prediction and targets are

`p_place = g_place(h_place(phi(x_anchor)))`

`t_positive = stopgrad(h_place_bar(phi_bar(x_positive)))`

`t_negative = stopgrad(h_place_bar(phi_bar(x_negative)))`.

Place energy is cosine distance. The loss is positive energy plus a fixed
`0.10` negative-minus-positive energy-gap hinge. This is a repeated-observation
key test, not a claim of rotation-invariant global localization or complete
loop closure.

Both new losses backpropagate through their online predictor, role projection,
the shared V18 object-space trunk, and the shared RGB encoder. Targets remain
stop-gradient EMA values. There is no endpoint-H4 head, recurrent state,
history input, pose/cell/yaw model input, explicit geometric memory, policy,
planner, or navigation loss.

## Frozen initialization and optimization

- Construct fresh from only the accepted N320 Camera checkpoint
  `.generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/checkpoint.pt`
  (file SHA-256
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`,
  content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`).
- Preserve V18 constructor seed `20260712`, object-space projection seed
  `20260729`, EMA momentum `0.996`, architecture, sweep masks, semantic,
  Camera, occupied-safety, survival/progress, and J24 physical objectives.
- Initialize the new role factorizer and predictors with seed `20260731` while
  preserving the caller RNG.
- Use one AdamW optimizer: RGB encoder learning rate `1e-4`; object-space,
  evidence, semantic-role, role-factorizer, and predictor learning rate
  `3e-4`; betas `(0.9,0.999)`; epsilon `1e-8`; weight decay `1e-4`.
- Preserve the inherited route-specific L2-to-one gradient scaling. Normalize
  local and place routes independently, sum their contributions at shared
  recipients, make exactly one optimizer step, then exactly one EMA step.
- No V19--V28, H4, rejected, or mutable runtime checkpoint, tensor, optimizer,
  RNG, or trace may be opened or reused.

## Frozen schedule and cap

Each update has microbatch size four and exactly:

- four physical microbatches: 16 scheduled examples;
- two immediate-local microbatches: 8 scheduled examples; and
- two place-triplet microbatches: 8 scheduled examples.

One scheduled row is one presentation, including one triplet row. Record RGB
decode and encoder work separately. Each update therefore accounts 32
presentations, 72 RGB decodes, 48 online RGB encodings, and 24 EMA-target RGB
encodings. The terminal cap is exactly 400 updates and 12,800 presentations:
6,400 physical, 3,200 local, and 3,200 place. Observe only updates 0, 100, and
400. Update 100 is diagnostic and cannot waive the terminal gate. There is one
attempt, no retry, no resume, no extension, and no alternate seed.

The physical route uses the frozen V25/V13 schedule and physical builder. The
local route uses the first 3,200 rows of the corrected causal H6 V2 training
index in frozen order, with current RGB `e2`, immediate-next RGB `e3`, and
action `actions[2]`. Its exact indexes remain:

| role | rows | bytes | SHA-256 |
|---|---:|---:|---|
| train | 16,000 | 10,328,000 | `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77` |
| validation | 2,048 | 1,317,888 | `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6` |

## Metadata-only place index

A metadata-only support census completed before this document with no RGB
open, RGB decode, GPU use, checkpoint open, protected-path open, or
probability-calibration referenced-path dereference. It joined only the 7,777
registered train and 924 checkpoint-selection endpoints for 80 allowed scenes
to their exact derived-label rows. It found 3,634 usable train anchors and 371
usable checkpoint-selection anchors.

The frozen, no-cycling selection uses measured support rather than fabricating
an equal quota:

| family | train | checkpoint selection |
|---|---:|---:|
| large_enclosed_maze | 242 | 32 |
| local_composite_motifs | 456 | 48 |
| loop_alias_stress | 325 | 32 |
| medium_enclosed_maze | 323 | 32 |
| open_obstacle_field | 528 | 64 |
| rough_local_dynamics | 527 | 64 |
| small_enclosed_maze | 387 | 20 |
| visual_sensor_stress | 412 | 28 |
| total | 3,200 | 320 |

Bind exactly:

| artifact | rows | bytes | file SHA-256 |
|---|---:|---:|---|
| `train.jsonl` | 3,200 | 4,687,348 | `72044c597286631be6133b45663ef975e222cd10d3f0cee1d0a9c038f0d422b6` |
| `checkpoint_selection.jsonl` | 320 | 473,508 | `a628a1047b6f15223a4fd7d30c5c87fa1914efef0955d70d9bd2f5330c77dcb0` |
| `manifest.json` | 1 | 42,308 | `a5997d93838419cabaaf8e262db70ed51f6f928195f1a312cadc4768f74ca6ca` |
| `receipt.json` | 1 | 779 | `37c6b497a304f02a5b159925934835337b55dc2eee19be1da01ca72525818307` |

The manifest content SHA-256 is
`cd3f894969422b924056a576e3acfb3b2be5e4d4d72e6ddd0e8d025618b14543`.
Train and checkpoint-selection scenes are disjoint. Runtime triplet batches
contain only three normalized RGB tensors; selection proofs are validated but
cannot cross the batch boundary.

## Exact update-400 gate

Every conjunct below must pass. No metric compensates for another.

1. Exact presentation/decode/encode, optimizer, EMA, gradient-recipient, source,
   role-access, finiteness, target-isolation, and immutable-observation-state
   integrity.
2. Place negative-minus-positive mean energy advantage at least `0.10`,
   deterministic bootstrap lower 95% strictly positive, and at least six of
   eight positive families.
3. Per-scene place retrieval over 40--64 unique candidates: equal-scene-mean
   recall@5 at least `0.40`, at least three times exact chance, and at least six
   of eight scenes strictly above chance.
4. Centered target place-key effective rank at least `4.0` at update 400 and at
   least `75%` of its update-0 value.
5. Local cyclic-wrong-minus-correct mean energy advantage at least `0.05`,
   deterministic bootstrap lower 95% strictly positive, and at least six of
   eight positive families.
6. On non-hold immediate transitions, correct-to-no-update energy ratio strictly
   below `1.0` and no-update-minus-correct bootstrap lower 95% strictly
   positive.
7. The compact physical floor: all twelve inherited causal-control checks true
   and strictly more than 72 of 189 physical margins passing, with structural
   integrity.

Rough-depth, total shortfall, tail, prior, same-cell/different-yaw, and full
loop-closure precision are diagnostic at this module stage. Memory-on versus
reset/reverse/shuffle, trajectory coherence, planning, collision, and
navigation success are explicitly deferred to a separately preregistered
memory-integration phase. They cannot fail or rescue this probe.

## Required source proofs and lifecycle

Focused source-only tests must prove exact model shapes and EMA binding, joint
gradient reach, action-only conditioning, strict batch schemas, the mixed
4+2+2 update, exact accounting/caps, one optimizer/EMA step, metadata-only pair
construction, role-before-path filtering, tensor-only runtime batches,
scene-disjoint bounded evaluation, thresholds, and denied-by-default lifecycle.

Freeze this document alone first. Then commit implementation and focused tests,
bind a recursive source closure, obtain an independent source review, and make
only an explicitly enumerated hash-bound clean export under a narrow
`AGENTS.md` exception. A later exact one-shot authority must bind the export,
runtime, hardware, N320 input, physical inputs, H6 indexes, and place indexes
before any RGB, checkpoint, GPU, training, or generated attempt output is
opened or created.

On FAIL, publish complete immutable receipts, publish no checkpoint, and close
this exact memory-role-factorized V1 mechanism. On PASS, publish only the
bounded update-400 development checkpoint and receipts. PASS authorizes only a
newly preregistered learned-memory integration gate; it does not authorize
navigation, probability calibration, G2, held-out, sealed, benchmark opening,
promotion, production, or deployment. The V4 30-scene sealed benchmark and all
externally custodied held-out mazes remain unopened.
