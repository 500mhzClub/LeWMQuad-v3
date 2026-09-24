# Go2 task-coupled recurrent physical dynamics V1 preregistration

Date frozen: 2026-08-04

Status: prospective development-only H1 mechanism experiment. No benchmark
role may be opened by this implementation until its source closure has passed
independent review and one exact execution authority has been issued.

## 1. Question and predecessor constraint

The experiment asks one narrow question:

> Does a compact conventional recurrent dynamics model extract
> scene-generalizing H1 branch-selection value from the ordered visual context,
> beyond the same model given only executed motion/action history and beyond the
> retained task/action analytic control?

It is not another JEPA run. It predicts four physical outcomes directly:
body-frame endpoint `dx`, `dy`, wrapped `dyaw`, and path length. Goal geometry is
used only by the frozen physical scorer and is not an input to the dynamics
model.

This intervention is required because the closest predecessors leave a real
gap:

- The matched physical-outcome screen flattened all three 4 x 4 DINO grids
  into one PCA16 vector and used a 532-parameter feed-forward MLP. Its visual
  arm was point-worse than its no-vision arm (`+0.00860305` normalized regret;
  interval crossed zero).
- The deterministic and stochastic recurrent screens pooled away spatial
  layout and predicted successor tokens, not task-coupled physical outcomes;
  both failed train-capacity gates.
- The grounded dense-DINO physical-only arm had no exact zero-vision
  counterpart. Its joint JEPA arm fit training but reversed on scene-disjoint
  evaluation.

Repository inventory found no existing source or result combining ordered
spatial recurrence, a direct physical/rank objective, and a parameter-matched
zero-vision intervention. A static MLP, another pooled successor predictor, or
another encoder swap would be scientifically redundant and is out of scope.

## 2. Fixed data and custody

The only scientific data are the existing H1 matched-branch development roles:

- train: 128 states, 16 scenes, eight families, nine executed branches/state;
- evaluation: disjoint 128 states, 16 scenes, eight families, nine executed
  branches/state;
- three pre-action RGB context frames/state;
- two executed historical command tapes and two measured body-frame odometry
  increments/state;
- one requested five-step, three-channel command tape/branch;
- four physical outcomes/branch and the existing dense physical rank.

The evaluation role has already been exposed by predecessor development work.
It is not held-out or final evidence. No `sealed_test.json`, `sealed/`,
`sealed_*`, legacy V4 final role, production role, or held-out role is
authorized.

No successor RGB or successor feature is permitted for either role. The runner
has no successor-reader branch: it opens exactly 384 train context artifacts,
writes the complete six-member checkpoint, then opens exactly 128 evaluation
receipts and 384 evaluation context artifacts. A semantic access ledger must
record zero train-successor and zero evaluation-successor opens. Data
generation, retry, resume, overwrite, and second-attempt authority are false.

## 3. Frozen visual representation

The DINOv2 ViT-S/14 source commit and checkpoint are the already bound
predecessor versions. All twelve transformer blocks and final norm are frozen.
For each 224 x 224 context RGB frame:

1. run the frozen full DINO encoder;
2. drop the CLS token;
3. L2-normalize each of the 256 final patch tokens;
4. reshape to 16 x 16 x 384;
5. average non-overlapping 4 x 4 patch blocks to a 4 x 4 x 384 grid.

One channel PCA `384 -> 16` is fitted in float64 across all 6,144 train context
cells (`128 states x 3 frames x 16 cells`). Component signs are fixed by making
the largest-absolute loading positive. Projected train channels are divided by
their train-only RMS. The frozen mean, components, scales, and signs are then
applied to evaluation. Spatial cell order and the three-frame temporal order
are retained. No successor tensor participates in fitting or projection.

## 4. Fixed arms and sole intervention

The primary arms are:

1. `task_action_only`: the unchanged analytic ridge control, refitted on the
   train role under the live numeric environment. Its coefficient identity is
   descriptive; its frozen behavioral witness is evaluation regret
   `0.17441406250000002`.
2. `no_vision_recurrent_direct`: the recurrent model below, with every
   projected visual cell set to exact zero.
3. `visual_recurrent_direct`: the identical recurrent model with the frozen
   projected DINO context cells.

Arms 2 and 3 have identical parameter names, shapes, model seed, initial bytes,
state batches, optimizer, update count, physical inputs, targets, and scorer.
The real-versus-zero visual tensor is the sole paired intervention.

## 5. Fixed model

For each of the 16 spatial cells, one shared PyTorch `GRUCell(34,16)` processes
three ordered steps. Its input is visual PCA16 plus transition18:

- `t0`: zero transition token;
- `t1`: measured body-frame odometry `dx,dy,dyaw` from frame 0 to 1 plus the
  exact preceding executed 15-value command tape;
- `t2`: the corresponding frame 1 to 2 increment plus command tape.

A learned 16-vector spatial-position embedding is added to each visual cell.
The candidate 15-value requested command tape is mapped by `Linear(15,16)`.
Its dot product with the 16 final recurrent cell states produces a
candidate-conditioned softmax attention pool. The attended state16 and action
query16 enter `Linear(32,16)`, `tanh`, and `Linear(16,4)`.

The standard PyTorch dual-bias GRU inventory gives exactly 3,604 trainable
float32 parameters. Xavier initialization uses a dedicated CPU generator;
biases are zero, position embeddings use truncated normal `std=0.02`, and the
final four-output layer is zero-initialized. Update zero therefore predicts the
train-only per-action physical mean for both learned arms. There is no encoder
tuning, JEPA loss, EMA target, successor decoder, semantic label, dropout,
reward, pose target, auxiliary head, or goal input.

## 6. Fixed optimization

- model seeds: `2026080411`, `2026080412`, `2026080413`;
- one shared state-permutation seed: `2026080414`;
- exactly 800 updates/member;
- complete nine-action states, batch eight states/update;
- traces at updates 0, 400, and 800;
- AdamW, learning rate `3e-4`, weight decay `1e-4`, betas `(0.9,0.999)`,
  epsilon `1e-8`, no fused/foreach path;
- global gradient norm clip `1.0`;
- deterministic float32 execution on the bound ROCm device;
- no checkpoint selection: decoded outcomes are averaged across all three
  fixed members before primary scoring.

The standardized target is the residual from the train-only mean for each of
the nine actions, divided by the joint train residual RMS for each of the four
outputs. The loss is fixed as:

`mean standardized residual MSE + 0.25 * all-strict-pair rank softplus`.

The rank term uses the existing differentiable cost
`remaining planar goal distance + 0.01 * relu(path_length)` and scale `0.05`.
True-rank ties are omitted. All nine candidates for every selected state are
present in every loss calculation.

There is no train-only early selection. All six finite members run the fixed
800 updates. Update-0/400/800 traces are descriptive capacity diagnostics and
cannot change evaluation eligibility, hyperparameters, or checkpoint choice.

## 7. Fixed evaluation and gates

Evaluation opens only after the complete checkpoint is durably written. The
three decoded member outcomes are averaged, then converted to the existing
one-centimetre-quantized H1 physical rank and action selection. Reports include
per-seed and ensemble regret, oracle-equivalent selection, target progress,
family directions, output RMSE, and train/evaluation trace contrast.

Paired comparisons use the existing family-equal, whole-scene bootstrap with
10,000 draws and seed `2026080407`. The visual arm passes only if all five
pre-existing grounded-H1 conditions pass:

1. provenance, deterministic repeat, complete role geometry, context-only
   access, and privileged physical oracle regret `0` / equivalence `1`;
2. visual normalized rank regret `<= 0.13`;
3. visual minus task/action regret `<= -0.02` and paired upper 95% bound `< 0`;
4. visual minus no-vision regret `<= -0.01` and paired upper 95% bound `< 0`;
5. visual regret is below random expectation.

The result is `PASS_TASK_COUPLED_RECURRENT_DYNAMICS_H1` only when all five pass;
otherwise it is `STOP_TASK_COUPLED_RECURRENT_DYNAMICS_H1`. No threshold can be
changed after result access. The no-vision-versus-task comparison is mandatory
but diagnostic, because it distinguishes embodiment priors from visual value.

## 8. Fixed interpretation and next action

- All gates pass: compact visual recurrence establishes development H1 value
  beyond both controls. This is not closed-loop or final world-model evidence.
  The only authorized next scientific proposal is a separately preregistered
  blind 1/2/4-step rollout comparison before planner integration.
- Visual significantly beats no-vision but fails task or absolute gates: real
  visual information exists but is not planning-useful enough; no rollout.
- No-vision beats task and visual does not beat no-vision: any gain is explained
  by odometry/action history; make no visual/world-model claim.
- Training improves but scene-disjoint evaluation fails: stop architecture
  tuning. The limiting hypothesis becomes scene diversity/generalization; size
  fresh matched counterfactual scenes before another model run.
- Training itself does not improve: this compact mechanism lacks train
  capacity. That is not evidence against JEPA, DINO, or more data generally.

Under every stop outcome, do not retry a seed, extend updates, tune PCA width,
change pooling, add a loss/head, relax a gate, inspect evaluation successors,
or integrate the planner. The 3 TB on-policy pool may later support
self-supervised appearance/temporal pretraining, but it does not create the
within-state untaken-action contrast required by this H1 evaluation.

## 9. Required durable outputs

The one-shot attempt root contains exactly:

- `reservation.json`;
- `checkpoint.pt` with all six update-800 states, fixed projections,
  normalizers, seeds, and traces;
- `result.json` with all reports, comparisons, gates, custody, provenance, and
  exact repeat evidence;
- `terminal.json`, which always sets retry/resume and navigation authority
  false.

An independent terminal review must rehash the authority, checkpoint, result,
and terminal, recompute the headline regret/comparison/gate logic, confirm zero
successor/protected access, and update the repository session handoff. No
scientific claim is final before that review.
