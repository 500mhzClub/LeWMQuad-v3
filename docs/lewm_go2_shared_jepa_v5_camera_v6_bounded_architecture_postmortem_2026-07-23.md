# Shared V5 Camera V6 bounded architecture postmortem

Date: 2026-07-23

Status: **same-architecture Camera line stopped; analysis only; no successor
or execution authority**

## Decision

Camera V6 is the terminal result for the existing protected Camera
architecture and training recipe. Do not launch a V7, extend the schedule,
change a seed or learning rate, rebalance a loss coefficient, refine the data,
relax the evaluator, or promote a closest checkpoint.

The only evidence-supported next investigation is a separately authorized,
zero-training diagnostic of the fixed soft evidence-decoding and rasterization
stage. It must not promote a checkpoint or open held-out material. A trained
architectural successor is not justified unless that diagnostic first isolates
a large, cross-scope failure in that stage.

## Terminal evidence

The immutable terminal audit is
`docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v6_terminal_audit_2026-07-23.json`,
committed at `f1c4e2efe948165004512ccc1882e721d8626d0b`. Its canonical content
SHA-256 is
`76727ada6442774412508b0ca96b1a50b5170bc75867235aecc132f28d1ac892`.

At update 8000:

- passed margins: `135/189`;
- total shortfall: `15.360492280690737`;
- worst normalized margin: `-2.9109309911727883`;
- complete physical scopes: `0/9`;
- qualified or selected checkpoint: none;
- G2, navigation, and held-out attempts: zero.

The late trajectory does not support more of the same:

| Update | Passed / 189 | Shortfall | Worst margin | Selection loss |
|---:|---:|---:|---:|---:|
| 4000 | 130 | 18.638109619170116 | -2.942538738250731 | 1.4113989913295875 |
| 6000 | 133 | 16.170685284493313 | -3.231825685501094 | 1.3773113556006658 |
| 8000 | 135 | 15.360492280690737 | -2.9109309911727883 | 1.3736735407839573 |

From update 6000 to 8000, only two additional margins passed, shortfall fell
about five percent, loss fell about 0.26 percent, and no scope closed.

The final misses are not mostly threshold noise. Of 54 negative margins, only
12 are within five percent of their threshold and they contribute 1.9 percent
of total shortfall. Eighteen margins miss by more than 25 percent and
contribute 75.5 percent of total shortfall.

## Architectural localization

Camera V6 resized RGB to `112x112`, encoded it as a `16x16` grid of
192-dimensional ViT tokens, and decoded a 36-channel dense feature map through
one non-overlapping stride-seven transpose convolution followed by one
`3x3` convolution.

The Camera head then produced:

- 64 ordered first-hit hazards and 64 within-bin offsets per image ray;
- calibrated clear/not-clear logits for five ground-support queries per
  source cell;
- a fixed differentiable composition of those predictions into a `64x64`
  unknown/free/occupied raster.

The shared encoder and Camera evidence head were trained. The BEV decoder,
occupancy head, JEPA predictor, target encoder, and target BEV decoder were
frozen. JEPA objective, JEPA backward, and EMA update counts were zero.

The final metric pattern separates two failure families.

### Soft decoding and rasterization are the leading non-rough hypothesis

At the aggregate scope, direct evidence passes:

- pixel first-hit balanced accuracy: `0.9771789001258041`;
- ground-clear balanced accuracy: `0.9747100416827836`;
- six ground distance accuracies: `0.9688127386413721` to
  `0.9796585405874831`.

The raster derived from that evidence fails:

- raster balanced accuracy: `0.9009460724448773`;
- free recall: `0.91637020862468`;
- occupied recall: `0.8059679976935274`;
- raster NLL: `0.18704089070408247`.

Across all nine scopes, raster balanced accuracy passes `0/9`, free recall
passes `1/9`, occupied recall passes `2/9`, and raster NLL passes `2/9`.

This localizes the failure to the evidence-decoding and rasterization stage,
but does not yet identify which operation is causal. One plausible amplifier
is the conjunctive free-evidence composition: five support probabilities are
multiplied for each source cell and four source cells are multiplied for each
output cell. Twenty support probabilities of `0.95` yield only about `0.358`
combined free probability. Occupied evidence is unioned first, and the
remaining mass is split between free and unknown.

This is localization, not yet causal proof. It does not authorize changing the
composition or evaluator.

### Rough motion is a distinct perception failure

`rough_local_dynamics` passes only `7/21` margins and contributes 47.3 percent
of total shortfall. Its final values include:

- pixel first-hit balanced accuracy: `0.8198594673963917`;
- ground-clear balanced accuracy: `0.647134926562893`;
- depth p95 error: `0.9777327477931971 m` against `<=0.25 m`;
- raster balanced accuracy: `0.7719525130620232`;
- occupied recall: `0.4319466882067851`.

Open-field depth p95 is also `0.8488615512847915 m`. These failures remained
large after the late loss plateau. The existing sidecars cannot distinguish
decoder resolution, representation invariance, temporal information, or
training distribution as the cause. They therefore do not justify a capacity,
data, or loss tweak.

## One proposed user decision

No diagnostic is currently authorized. If the user explicitly chooses this
architecture-level direction, a new source-free preregistration may define
exactly one zero-training development diagnostic:

1. Bind the V6 terminal audit and the immutable update-8000 checkpoint.
2. Open that checkpoint once and evaluate only the existing 495-pair
   checkpoint-selection role.
3. Preserve the predicted ray hazards, depths, and ground logits exactly.
4. Construct one diagnostic hard-evidence raster using only already-fixed
   decisions: finite-hit probability `>=0.5`; depth equal to the calibrated
   centre of the maximum-probability finite-hit bin plus that bin's predicted
   within-bin offset; and ground logit `>=0`. Use the same frozen camera
   geometry while replacing the soft distributional decoding, splatting, and
   probabilistic union/product operations by their hard MAP/Boolean diagnostic
   counterparts.
5. Perform no threshold search, calibration, gradient, optimizer step,
   checkpoint mutation, data mutation, or held-out access.
6. Publish one canonical diagnostic result and terminal audit.

The diagnostic supports a soft decoding/rasterization successor only if it
produces an absolute raster-balanced-accuracy gain of at least `0.05` in at
least six of the eight non-rough scopes and improves both aggregate free and
occupied recall by at least `0.05`, without changing any direct evidence
metric. Otherwise that broader hypothesis is rejected and this Camera/evidence
design should be abandoned. A PASS would not uniquely implicate the
twenty-factor free-product rule.

Even a diagnostic PASS is not Camera qualification. It would justify only a
new user decision about one structural evidence-decoding/rasterization
successor. It grants no training, checkpoint promotion, G2, navigation,
runtime, production, or held-out authority.

## Authority boundary

The V6 one-attempt execution authorization is consumed. Its terminal audit
explicitly denies retry, continuation, automatic successor, architecture
mutation, data refinement, frozen-camera JEPA, G2, navigation, promotion, and
held-out use.

The V6 preregistration requires a new user-directed architecture-level
decision after an update-8000 failure. This postmortem is evidence for that
decision; it is not the decision, a preregistration, source review, execution
authorization, or permission to inspect a checkpoint.
