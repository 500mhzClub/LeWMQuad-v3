# Go2 grounded dense-DINO joint-JEPA V1 preregistration

Date: 2026-08-04

Status: protocol fixed before implementation, training, or opening the
evaluation role for this attempt. This is one development-only mechanism
experiment. It authorizes no retry, held-out claim, safety claim, production
use, or deployment.

## Question

Does a dense action-conditioned JEPA become physically useful when its visual
representation, successor predictor, and embodiment-grounded outcome interface
are trained jointly, rather than predicting into a frozen representation and
applying a fixed image-goal cost afterward?

This changes the mechanism that failed the frozen-DINO same-patch-cost ceiling.
The planner will not consume token cosine distance. The model must predict
physical successor outcomes, and physical action ranking is the primary
scene-disjoint gate.

## Existing data only

Use exactly the existing matched-branch development bundle:

- 128 training states from 16 scenes and 128 evaluation states from 16
  scene-disjoint scenes;
- eight balanced families, two scenes per family and eight states per scene;
- nine executed candidate branches per state;
- three pre-action RGB frames and nine successor RGB frames per state;
- physical targets `[endpoint_dx_body_m, endpoint_dy_body_m,
  endpoint_dyaw_rad, path_length_m]` and the existing dense physical ranks.

The joined manifest is
`.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1/manifest.json`,
11,964 bytes, SHA-256
`87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e`.
Physical targets must be reconstructed from the separately bound raw state
receipts through the existing physical loader. The joined rows alone are not a
substitute.

No scene, action, RGB frame, physical trajectory, or successor will be
generated. No V-JEPA arm, architecture matrix, cost sweep, coefficient sweep,
or alternate seed is allowed.

The evaluation role is already development-exposed in this repository. Scene
separation prevents training leakage but does not make it fresh confirmation or
held-out evidence. All 2,304 branches have zero falls and zero tips, so safety
is not testable.

## Encoder and predictor

- Frozen source: local `dinov2_vits14`, repository commit
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- Checkpoint: `dinov2_vits14_pretrain.pth`, 88,283,115 bytes, SHA-256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- Exact 224x224 RGB and ImageNet normalization already bound by the bundle.
- Patch embedding, positional interpolation, and DINO blocks 0--9 are frozen.
  Their outputs may be computed once per opened RGB artifact and retained only
  as attempt-local trunk tensors.
- Online blocks 10--11 and the online final norm are trainable.
- The action-conditioned successor is the existing production
  `DenseDINOv2TemporalPredictorV1`: three 256x384 context grids, two executed
  15-value command tapes, and one requested 15-value candidate tape produce
  one normalized 256x384 successor grid.
- A new dense relational physical head consumes, per patch,
  `[current, predicted, predicted-current]`, learned patch position, and the
  existing 12 physical history/candidate scalars. Conditioned attention pools
  all 256 patches and predicts four standardized physical residuals.
- The physical outcome is decoded with training-only action means and residual
  scales. Goal-relative geometry is used only to turn predicted displacement
  into an action score; it is not a successor-predictor input.

Future executed commands, endpoint state, physical labels, dense ranks, true
successor tokens, and successor RGB are forbidden inference inputs.

## Three arms

1. `task_action_only`: refit the exact existing task/action ridge on the
   training role. Never load an old learned checkpoint.
2. `physical_only_matched`: online DINO tail, dense predictor, and physical
   head with exactly the same initialization, trainable parameter inventory,
   optimizer, batches, and physical losses as arm 3. It never opens successor
   RGB and receives no JEPA target loss.
3. `joint_jepa_grounded`: the same model plus a detached EMA copy of DINO
   blocks 10--11 and norm. Only this arm opens the 1,152 training successor RGB
   frames and receives dense successor losses.

Arm 2 is a matched physical-only direct-dynamics ablation, not a separately
architected recurrent SSM. This makes arm 3 versus arm 2 a clean intervention
on JEPA target supervision. Conventional recurrent comparison remains future
work only if this mechanism earns eligibility.

Run arm 2 first and write its durable checkpoint before opening any successor
RGB for arm 3. Evaluation RGB and evaluation physical receipts remain unopened
until both final training checkpoints are durable.

## Fixed optimization

- One model seed: `2026080405`.
- Candidate/state permutations: dedicated generator seed `2026080406`.
- Microbatch: two complete states (18 candidate branches).
- Accumulate four microbatches for effective batch eight states.
- Maximum 800 optimizer updates; train measurements at updates 0, 400, 800.
- AdamW, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`.
- Learning rate `3e-4` for predictor/head and `3e-5` for online DINO tail.
- Global gradient-norm clip `1.0`; float32 trainable state; deterministic
  algorithms; R9700 device 0.
- EMA momentum `0.996`; target parameters receive no gradient.

Both learned arms minimize:

- unweighted mean squared error over the four standardized physical residuals;
- `0.25` times an all-strict-pairs physical ranking loss. For true rank
  `i < j`, predicted lower-is-better costs use
  `softplus((cost_i - cost_j) / 0.05)`. Cost is predicted remaining planar
  goal distance plus `0.01 * relu(predicted_path_length)`.

Arm 3 additionally minimizes:

- mean patch cosine distance to the detached EMA true successor (`weight 1`);
- within-state nine-way predicted-action/true-successor InfoNCE with
  temperature `0.10` (`weight 0.10`).

No loss coefficient, optimizer setting, update count, encoder boundary, or
model width may change after the smoke or result.

## Train-only futility rule

At update 400, arm 3 continues to update 800 only if all are true:

1. all losses, gradients, parameters, predictions, and EMA targets are finite;
2. training normalized physical rank regret improves by at least `0.03` from
   update 0;
3. true-successor branch retrieval is at least `0.35` and at least `0.15`
   above update 0;
4. mean successor cosine error divided by the last-frame-persistence cosine
   error is at most `0.90`.

If arm 3 fails, write a train-capacity terminal and do not open the evaluation
role. Arm 2 remains a control and trains for all 800 updates unless a nonfinite
or infrastructure failure occurs.

## Evaluation and primary gate

Evaluation runs once after both checkpoints are durable. Each learned arm may
open only the three evaluation context RGB frames per state; neither may open
an evaluation successor RGB frame. Predict all nine actions, decode physical
outcomes, and apply the exact existing rank/tie evaluator.

Use the existing family-equal, whole-scene paired bootstrap with 10,000 draws,
seed `2026080407`, and 2.5/97.5 percentiles. The 16 evaluation scenes are the
uncertainty units; state or branch pseudo-replication is forbidden.

Arm 3 earns a development closed-loop experiment only if every item passes:

1. exact input/source/checkpoint provenance, role and scene disjointness,
   complete 128x9 evaluation, finite outputs, no forbidden input access, no
   evaluation successor RGB opens, oracle regret exactly zero, and deterministic
   repeated evaluation;
2. normalized physical rank regret is at most `0.13`;
3. regret is at least `0.02` below task/action-only and the paired bootstrap
   upper 95% bound for `(joint - task)` is below zero;
4. regret is at least `0.01` below the matched physical-only arm and the paired
   bootstrap upper 95% bound for `(joint - matched)` is below zero;
5. regret is below random expectation.

Task/action-only and the matched control remain in the report regardless of
whether they pass any candidate gate. Prediction MSE, action selection,
per-family effects, attention, and train JEPA metrics are diagnostics only.

## Stopping and claims

- Gate pass: freeze the checkpoint and separately preregister one
  development-only same-sensor closed-loop causal comparison.
- Gate fail or train futility: stop this partially trainable DINO-tail plus
  dense-predictor plus physical-head mechanism. Do not retry a seed, extend
  training, tune thresholds, open evaluation successors, or rescue the result
  with a different readout.
- Infrastructure failure before a scientific result consumes this attempt and
  authorizes no automatic replacement.

No outcome from this experiment alone establishes closed-loop navigation,
rollout composability beyond H1, persistent memory, safety, held-out
generalization, or deployment readiness.

## Source review and one-shot authority

Before execution, bind the model, runner, focused tests, this preregistration,
the exact input files, and the output root in a new independent source review
and one-shot execution authority. Previous authorities are consumed and must
not be reused.

