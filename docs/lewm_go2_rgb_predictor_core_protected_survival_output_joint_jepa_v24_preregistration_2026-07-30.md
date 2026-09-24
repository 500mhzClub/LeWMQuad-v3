# V24 predictor-core-protected survival-output joint-JEPA preregistration

Date: 2026-07-30

Status: preregistered fresh scientific successor only. No V24 source root,
output root, reservation, generated-input access, GPU work, training,
checkpoint, calibration, G2, navigation, held-out, or sealed access has
occurred or is authorized here.

## Frozen predecessor evidence

- The controlling predecessor evidence is
  `docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23_scientific_result_2026-07-30.json`.
  It is frozen in commit `04b0fa48c6c4e10868c2f302bc51100394e3907e`,
  with file SHA-256
  `753c91babd4f7116444654167d2507ffb52d22f970fc926c05d287683954c994`,
  byte count `20640`, and canonical content SHA-256
  `a5a6b8aa7312706d2ae3a5b53e39370462e9de6eda6b7a2ca4e2e0226a518ed8`.
- V23 is consumed and terminal at update 400. It passed structural and target
  integrity and all three physical thresholds. It passed all shuffled-action,
  wrong-RGB, and train-action-prior checks. Persistence had positive mean
  utility delta `0.032683944750712256`, but its bootstrap lower 95% bound was
  `-0.0028518293937047426` and only five of eight families were positive.
  V23 therefore passed ten of twelve causal checks and published no
  checkpoint.
- The V23 direct objective worked locally: output fit energy fell from
  `0.12122041545808315` at update 1 to `0.02193028014153242` at update 400,
  scene advantage reached `0.09103782911752832`, and prior advantage reached
  `0.047873977571725845`. In parallel, the inherited latent prediction loss
  `P` rose from `1.0` at update 1 to `9.32626724243164` at update 400; its
  consecutive 100-update means were `5.910986719727516`,
  `6.3907791584730145`, `8.823022493720055`, and `11.683673999905587`.
  This is evidence consistent with auxiliary interference in the dynamics
  core, not proof of negative gradient cosine.
- V24 is one fresh, evidence-led routing test from exact initialization. It is
  not a V23 retry, resume, extension, checkpoint continuation, delayed-onset
  variant, coefficient change, or conflict-projection experiment.

## Sole scientific change

V24 preserves the exact V23 output objective, labels, eligibility masks,
negative selection, action set, scaling, coefficient, and onset. The sole
scientific change is which online parameter tensors receive its auxiliary
gradient.

- The objective remains exactly `J24 = J23 = F + R`, where `F` is the mean
  Smooth-L1 survival-output fit over the eight non-HOLD actions and `R` is the
  single count-weighted softplus rank mean over eligible wrong-scene and
  frozen train-action-prior comparisons in V23's action-prior-residual
  coordinates.
- The computational graph remains the complete learned output path. J24 still
  passes through the latent predictor core when differentiating the current
  RGB representation and the swept-progress output.
- The J24 gradient accumulator contains exactly these 96 tensors and
  3,106,409 parameters:
  `encoder.*`, `bev_lift.evidence_head.*`,
  `bev_lift.point_projection.*`, `bev_lift.volume_block.*`, and exactly
  `predictor.swept_progress_head.output.weight` plus
  `predictor.swept_progress_head.output.bias`.
- J24 directly updates none of the other 13 `predictor.*` tensors, which
  contain exactly 259,008 parameters and form the latent dynamics core. Those
  tensors remain trainable and receive their normal inherited joint-JEPA
  gradient on every microbatch.
- J24 continues to exclude `semantic_head.*`, every `target_encoder.*` and
  `target_bev_lift.*` tensor, labels, metadata, and evaluator-only tensors.
  Its target-gradient count must remain zero.
- The one global J24 auxiliary L2 unit-norm cap is applied over the exact
  96-tensor subset before addition to the inherited routes. It is not applied
  per group. The inherited Camera, shared, representation, and complete
  15-tensor predictor routes retain their exact clipping and addition rules.

The architecture and total trainable parameter count do not change. This is
parameter-route protection, not freezing the predictor core globally, not
detaching the representation, and not training a downstream predictor after
the encoder.

## Joint-JEPA and accounting boundary

- The inherited joint-JEPA objective remains active on every microbatch. Its
  EMA-target latent-prediction and survival terms collectively train the
  online encoder, object-space representation, and all 15 predictor tensors;
  the 13-tensor latent core is trained by the prediction component, while the
  two swept-progress output tensors are trained by the inherited survival
  terms. The target encoder and target representation remain stop-gradient
  EMA copies updated exactly once after each optimizer step.
- J24 remains an auxiliary on the same all-action predictor forward. It adds
  no RGB read, encoder pass, target-encoder pass, predictor forward,
  presentation, graph, or optimizer step.
- Each completed update remains exactly four microbatch graphs, four
  all-action predictor forwards, four Camera-route gradient calls, four
  inherited joint-route gradient calls, four J24 gradient calls, twelve total
  autograd calls, eight predictor objectives, 32 camera-frame objectives, 16
  ordered presentations, one optimizer step, and one EMA step.
- The loss receipt is `L = N + C + J24`. Every inherited
  `S/P/U/R/O/C/N` definition and coefficient remains unchanged.

V24 is therefore still a fully learned, jointly optimized JEPA perception and
world-model test. No semantic oracle, geometric inference bypass, policy
supervision, privileged maze state, or separately trained navigation head is
introduced.

## Frozen identity

Except for the sole auxiliary parameter-route change and matching V24 receipt
names and lifecycle bindings, V24 preserves V23 and its V18 base exactly:

- learned RGB encoder, eight-height object-space representation, semantic and
  survival heads, local action-conditioned predictor, architecture, all
  parameter values at initialization, and total trainable parameter count;
- N320 initialization, constructor seed `20260712`, schedule seed `20260713`,
  experiment seed `20260728`, bootstrap seed `20260728`, projection seed
  `20260729`, float32 AdamW settings, learning rate, betas, epsilon, weight
  decay, EMA, inherited route clipping, parameter groups, and inherited
  joint-JEPA losses;
- V23's exact `F`, `R`, `J23` mathematics, eight non-HOLD action indices
  `(0,1,2,3,4,5,7,8)`, `1.5` m normalization, frozen train-action mean prior,
  deterministic wrong-scene row, eligibility rules, onset at update 1, and
  unit auxiliary norm cap;
- the 4262-pair schedule from presentation zero, four microbatches of four,
  train and checkpoint-selection roles, source data and labels, observation
  updates `(0,100,400,1000)`, terminal updates `(400,1000)`, eight-family
  registry, physical metrics, causal controls, and every threshold; and
- the maximum of 1000 updates and 16000 ordered presentations.

V24 starts once, in a new process, from exact initialization. No V23 model,
optimizer, EMA, RNG, schedule state, trace, metric, receipt, output, or mutable
runtime state may be opened or reused. The frozen V23 scientific-result
document may be used only as source-review identity evidence.

## Focused source acceptance

- Exact membership tests must prove the J24 route is 96 tensors and 3,106,409
  parameters, contains every intended perception and swept-progress output
  tensor, and contains none of the 13 latent-core, semantic, or target
  tensors.
- Gradient tests must prove finite nonzero J24 gradients through the encoder,
  evidence head, point projection, volume block, and both swept-progress
  output tensors; zero auxiliary accumulation into every latent-core tensor;
  and finite nonzero inherited joint gradients into all 15 predictor tensors.
- The V23 objective tensor tests remain unchanged and passing. A parity test
  must prove V23 and V24 produce bit-identical `F`, `R`, and total auxiliary
  loss from identical survival logits and metadata.
- One real CPU synthetic update must prove exact accounting, finite losses,
  one optimizer step, one EMA step, target isolation, and that each core
  predictor parameter's applied gradient contains only the inherited
  predictor route while each swept-progress output parameter contains the
  inherited plus J24 routes.
- Existing inherited model, evaluator, causal-control, custody, and source
  closure tests must remain passing. Recursive closure, independent review,
  narrow clean-export certification, and separate one-shot authority must be
  committed before reservation or execution.

## Falsification and scale gates

- Update 0 is informational and must pass structural, source, access,
  custody, accounting, target-isolation, exact route-membership,
  finite-gradient, and single-forward integrity.
- Every train row republishes V23's exact scene/prior eligible counts,
  advantages, energies, fit, and rank diagnostics under V24 identities. It
  additionally binds the 96-tensor included subset and 13-tensor protected
  core. Train-batch diagnostics never decide promotion.
- Update 100 is informational. It has no threshold, branch, restart,
  checkpoint, or early stop.
- At update 400, the unchanged conjunctive gate requires structural
  integrity; all twelve causal-control checks; physical margin count strictly
  greater than `72`; total physical shortfall strictly below
  `68.96954700805838`; and rough depth p95 strictly below
  `1.8582415819168085` m. Each of persistence, shuffled action, wrong RGB, and
  the frozen train-action prior retains its positive mean, positive bootstrap
  lower-95, and at-least-six-of-eight-family triplet.
- Any update-400 failure is terminal. No checkpoint is written or retained.
  Only a complete update-400 pass may continue in the same process without
  replaying or restarting its first 400 updates.
- Update 1000 is reached only after that pass and retains the exact inherited
  final gate: V12 full arm `24/24`; at least `112/189` physical margins;
  shortfall strictly below `33.05143763708337`; at least one complete physical
  scope; rough pixel balanced accuracy strictly above
  `0.8198594673963917`; rough ground balanced accuracy strictly above
  `0.647134926562893`; rough depth p95 strictly below
  `0.9777327477931971` m; and structural integrity.
- Only a complete update-1000 pass publishes the development checkpoint. No
  incomplete or failed run may publish one.

## Family-stop rule

- V24 is the final local routing variant of the V23 survival-output auxiliary.
  Any update-400 gate failure retires coefficient, onset, core-subset,
  head-subset, gradient-projection, and other local output-auxiliary variants.
  There is no V24 retry, resume, extension, replacement, alternate seed, or
  update-1000 continuation after a failed update-400 gate.
- A later proposal after V24 failure must change the learned temporal
  world-model mechanism materially. It may not spend another attempt tuning
  or rerouting J24. This preregistration does not authorize such a successor.

## One-shot and protected-access boundary

- Schema/evidence prefix:
  `lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24`.
- Fresh output root:
  `.generated/go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24/attempt_v1`.
- Fresh clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v24-core-protected-survival-source`.
- Both roots must initially be absent. There is exactly one attempt and no
  retry, resume, recovery, extension, replacement, coefficient search,
  alternate-onset run, or second attempt.
- Authority may cover only exact independently reviewed source plus frozen
  train and checkpoint-selection inputs on the reviewed runtime and hardware.
- Until a complete update-1000 pass, probability calibration, G2, navigation,
  held-out, sealed, promotion, production, and deployment remain forbidden.
  This preregistration grants none of those accesses and grants no source
  export, data, GPU, training, reservation, or execution authority.
