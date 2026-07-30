# V25 per-row persistence-contrastive temporal joint-JEPA preregistration

Date: 2026-07-30

Status: preregistered fresh scientific successor only. No V25 source root,
output root, reservation, generated-input access, recovery-state write, GPU
work, training, checkpoint, calibration, G2, navigation, held-out, or sealed
access has occurred or is authorized here.

## Frozen predecessor evidence

- The controlling predecessor evidence is
  `docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24_scientific_result_2026-07-30.json`.
  It is frozen in commit `2824c80c54fc7502b1413b3371fc87c9206f82a2`,
  with file SHA-256
  `f901d49eb9db0c39a068e67496b0b1cdaec954c9238edb40648140b924894e48`,
  byte count `22361`, and canonical content SHA-256
  `0349f41da529b0c8658bf14ae51d85892a6f21fb461a281a9e157c7e7ff571dc`.
- V24 is consumed and terminal at update 1000. It validly passed its
  update-400 gate with all twelve causal checks, 89 physical margins, total
  physical shortfall `52.292185477040775`, and rough depth p95
  `1.5603465557098388` m. Predictor-core protection therefore repaired V23's
  persistence failure strongly enough to earn the same-process continuation.
- V24 improved all three rough-motion metrics, passed physical-margin count,
  total physical shortfall, and worst physical margin through update 1000,
  while complete physical scope count remained zero. It failed the final
  development gate: it passed
  `94/189` rather than at least `112/189` physical margins, had shortfall
  `37.04639990568397` rather than strictly below `33.05143763708337`, rough
  depth p95 `1.2971169710159294` m rather than strictly below
  `0.9777327477931971` m, zero complete physical scopes, and `22/24` rather
  than `24/24` full-arm checks. The two lost checks were the frozen
  train-action-prior bootstrap and family-coverage checks.
- The inherited global persistence ratio `P` rose from `1.0` at update 1 to
  `8.644781708717346` at update 400 and `21.478042244911194` at update 1000.
  The complete predictor route's preclip gradient L2 rose from
  `1.9941233396530151` to `7.770937442779541` and
  `24.03590202331543`, while its applied clip scale fell to
  `0.041604429483413696` at update 1000. This is evidence of a scale/conflict
  pathology in the temporal objective, not proof of a unique cause.
- V24's family stop forbids another J24 coefficient, onset, subset,
  projection, or routing variant. V25 leaves J24 unchanged and instead makes
  the required material change to the learned temporal world-model
  objective. V25 is not a V24 retry, resume, extension, checkpoint
  continuation, alternate seed, or further local-output-auxiliary variant.

## Scientific hypothesis

V24's temporal term divides the microbatch mean executed-action prediction
energy by a detached microbatch mean persistence energy, clamped only at
`1e-6`. As the learned EMA representation makes consecutive observations
closer, that denominator can shrink and amplify the complete predictor route
even when the absolute executed-action error is improving.

V25 tests whether a row-local comparison against the same persistence
baseline can preserve the useful temporal target without inverse-baseline
gradient amplification or cross-row normalization. It asks each executed
action prediction to beat the no-motion persistence prediction for that same
row before averaging the row losses.

This mechanism is denominator-free and has a bounded scalar multiplier on
the executed-action energy gradient. It is not claimed to be globally scale
invariant. The unchanged channel-wise LayerNorm inside the frozen latent
energy reduces representation-scale sensitivity.

## Sole scientific change

Let the fixed microbatch size be `B = 4`. For row `i`, let `a_i` be its
executed action, let `z_hat_next_i[a_i]` be the corresponding online
action-conditioned predicted latent, and let `z_bar_current_i` and
`z_bar_next_i` be the current and next latents from the EMA target path.

The latent energy remains exactly the inherited per-row energy:

```text
E(x, y) = mean_all_cells_and_channels(
    SmoothL1(
        LayerNorm_channel(x),
        LayerNorm_channel(stopgrad(y)),
        beta=1.0
    )
)
```

For each row:

```text
e_pred_i    = E(z_hat_next_i[a_i], stopgrad(z_bar_next_i))
e_persist_i = stopgrad(E(z_bar_current_i, z_bar_next_i))
gap_i       = e_pred_i - e_persist_i
row_loss_i  = softplus(gap_i, beta=1, threshold=20) / log(2)
```

The replacement temporal objective is exactly:

```text
P25 = mean_i(row_loss_i), for i in {0, 1, 2, 3}
```

The batch mean occurs only after the four row-local softplus terms. There is
no global energy ratio, learned or empirical normalizer, denominator clamp,
margin, temperature search, coefficient, onset, family weighting, or
cross-row negative. `log(2)` is the fixed mathematical normalizer. When
`e_pred_i == e_persist_i`, that row contributes exactly `1`.

For PyTorch softplus with `beta=1` and `threshold=20`, the scalar multiplier
on each prediction energy is exactly:

```text
dP25 / d(e_pred_i) = sigmoid(gap_i) / (B * log(2)), when gap_i <= 20
dP25 / d(e_pred_i) = 1 / (B * log(2)),              when gap_i > 20
```

It is strictly greater than zero and at most `1 / (B * log(2))`; it cannot grow as the
persistence energy approaches zero. `e_persist_i` has exactly zero gradient.

V25 replaces only the scalar occupying the inherited `P` loss slot. Receipt
schemas may continue to call that slot `P`, but must bind its mechanism as
`P25` and must include it exactly once:

```text
N25 = S + P25 + U + R_inherited + O
L25 = N25 + C + J24
J24 = F + R_output
```

Here `R_inherited` is the unchanged swept-progress ranking term and
`R_output` is the unchanged V24/V23 survival-output rank term. No term is
duplicated or omitted.

## Gradient and target semantics

- `P25` differentiates through the online current-RGB encoder, learned
  object-space representation, and executed-action latent predictor path.
- The current and next EMA latents, `e_persist_i`, executed action indices,
  and all diagnostic values are stop-gradient. No target tensor may receive a
  gradient.
- `P25` remains inside the same inherited joint route and optimizer step as
  `S`, `U`, `R_inherited`, and `O`. The complete predictor remains jointly
  trained by the inherited objective.
- J24 remains bit-identical to V24. Its one capped auxiliary route still
  contains exactly 96 tensors and 3,106,409 parameters: the encoder,
  evidence head, point projection, volume block, and exactly the two
  swept-progress output tensors.
- The other 13 predictor-core tensors and 259,008 parameters remain protected
  from J24 only. They receive the normal inherited joint gradient containing
  `P25`; they are not frozen, detached, or trained separately.
- The target encoder and target representation remain stop-gradient EMA
  copies and update exactly once after each optimizer step.

V25 is therefore still a fully learned, jointly optimized JEPA perception
and world-model test. It adds no downstream frozen-encoder predictor,
semantic oracle, geometric inference bypass, policy supervision, privileged
maze state, or separately trained navigation head.

## Frozen identity

Except for replacing `P` by the equation above, matching V25 receipt names,
and the science-neutral recovery write described below, V25 preserves V24
exactly:

- learned RGB encoder, eight-height object-space representation, semantic and
  survival heads, local action-conditioned predictor, architecture, total
  trainable parameter count, and every initialization value;
- N320 initialization, constructor seed `20260712`, schedule seed `20260713`,
  experiment seed `20260728`, bootstrap seed `20260728`, projection seed
  `20260729`, float32 AdamW settings, learning rate, betas, epsilon, weight
  decay, EMA, parameter groups, route clipping, and gradient-addition rules;
- V24's exact `F`, `R_output`, `J24`, 96-tensor destination, protected
  13-tensor core, eight non-HOLD actions `(0,1,2,3,4,5,7,8)`, `1.5` m
  scaling, frozen train-action mean prior, deterministic wrong-scene row,
  eligibility masks, onset at update 1, and unit auxiliary norm cap;
- the 4262-pair schedule from presentation zero, four microbatches of four,
  train and checkpoint-selection roles, source data and labels, observation
  updates `(0,100,400,1000)`, terminal updates `(400,1000)`, eight-family
  registry, physical metrics, causal controls, and every threshold; and
- the maximum of 1000 updates and 16000 ordered presentations.

V25 starts once in a fresh process from exact initialization. No V24 model,
optimizer, EMA, RNG, schedule state, trace, metric, receipt, output, recovery
state, or mutable runtime state may be opened or reused. The committed V24
scientific-result document is source-review identity evidence only.

## Accounting and diagnostics

- Each completed update remains exactly four microbatch graphs, four
  all-action predictor forwards, four Camera-route gradient calls, four
  inherited joint-route gradient calls, four J24 gradient calls, twelve total
  applied-route autograd calls, eight predictor objectives, 32 camera-frame
  objectives, 16 ordered presentations, one optimizer step, and one EMA step.
- P25 uses the already computed predicted latent and EMA latents. It adds no
  RGB read, encoder pass, target-encoder pass, predictor forward, presentation,
  optimizer step, or applied gradient route.
- Every train-update receipt records detached, schedule-ordered per-row
  `e_pred`, `e_persist`, `gap`, and `row_loss` values for all 16 presentations,
  plus their finite count, sum, mean, minimum, maximum, and the count and
  fraction with `gap < 0`.
- Every receipt records the fixed `log(2)` normalizer, softplus beta and
  threshold, and an explicit assertion that no denominator is used.
- For comparison only, each microbatch records the detached legacy diagnostic
  `mean(e_pred) / clamp_min(mean(e_persist), 1e-6)` and the update receipt
  records its detached mean/minimum/maximum. This value must never retain a
  graph, enter a loss, alter a branch, or decide promotion.
- Existing applied-route receipts remain unchanged and report preclip L2,
  applied scale, tensor count, parameter count, and absent-gradient count for
  the Camera, joint shared, joint representation, complete predictor, and J24
  routes. V25 adds no diagnostic-only autograd call or gradient projection.
- Train-batch energy, gap, legacy-ratio, loss, and gradient diagnostics are
  observational only and never decide continuation or promotion.

## Science-neutral update-400 recovery write

- Only after a complete update-400 gate pass, with its metric durably written,
  and before any update-401 batch access, V25 writes exactly one full recovery
  snapshot at `recovery/update_400_training_state.pt` and its content-bound
  gate/write receipt at
  `recovery/update_400_training_state.binding.json` beneath the attempt root.
  The receipt binds the gate decision, snapshot SHA-256 and byte count, and
  exact trace-prefix identity before the uninterrupted process continues.
- The snapshot contains the complete online and EMA model state and buffers,
  optimizer state, EMA counter, V25 accounting, next schedule position,
  Python/NumPy/Torch CPU/visible-ROCm-device RNG states, and exact frozen
  source, authority, input, metric, and trace-prefix identities. It contains
  no dataset, RGB frame, label payload, evaluator output, held-out identity,
  or sealed material.
- The final path must initially be absent. Publication is atomic and
  exclusive, a second write is rejected, and the completed file is SHA-256
  and byte-count bound in a write receipt and made read-only.
- Serialization occurs only at the completed update boundary. It must not
  reseed, load any state, change model mode, advance the schedule or any RNG,
  mutate parameters, buffers, optimizer values, gradients, accounting, or
  metrics, or perform an additional model/data computation.
- The uninterrupted V25 process never opens or loads this snapshot. V25
  implements no recovery reader or resume path. This preregistration grants
  no resume authority; any later infrastructure-recovery proposal would need
  separate frozen review and authority and could not exceed the original
  update or presentation cap.
- An update-400 scientific gate failure writes no recovery snapshot. A
  required recovery-write failure after a gate pass is a fail-closed
  infrastructure failure before update 401, not permission to continue
  without the receipt.
- If V25 later fails scientifically at update 1000, the immutable update-400
  snapshot remains audit evidence only and grants no checkpoint, retry,
  resume, extension, or successor initialization authority. There is no
  update-1000 recovery-state write. Only the unchanged complete update-1000
  pass rule may publish the development checkpoint.

This operational write is not a second scientific change because it is
write-only, occurs after the gate at an optimizer/EMA boundary, is unread by
the uninterrupted run, and is required to prove zero training-state or RNG
mutation.

## Focused source acceptance

- Reference-tensor tests must prove the exact `E`, `e_pred`, `e_persist`,
  `gap`, row-softplus, and `P25` equations, including default softplus
  `beta=1`, `threshold=20`, normalization by `log(2)`, and value `1` at equal
  energies.
- Analytic-gradient tests must prove the stated bounded piecewise derivative
  in both the softplus and threshold-linear branches, exactly zero baseline
  and EMA-target gradients, finite nonzero gradients through the online
  encoder, object-space representation, and latent predictor core, and zero
  cross-row normalization dependence.
- Row-permutation and single-row perturbation tests must prove permutation
  invariance of the mean and that changing another row's persistence energy
  cannot change a row's prediction-energy derivative.
- Objective-composition tests must prove `P25` enters `N25` exactly once and
  that the detached legacy ratio cannot affect any loss, branch, gradient, or
  parameter update.
- V24 parity tests must prove bit-identical `F`, `R_output`, `J24`, parameter
  membership, route cap, and protected-core exclusion from identical logits
  and metadata.
- One real CPU synthetic update must prove exact accounting, finite losses,
  one optimizer step, one EMA step, target isolation, nonzero inherited joint
  gradients into the complete predictor, no J24 gradient in the protected
  core, and unchanged applied-route receipt semantics.
- Recovery tests must prove the writer is unreachable before a passed
  update-400 gate, rejects a pre-existing final path and any second write,
  contains every required state, and produces a bound read-only artifact.
  State and RNG identities captured immediately before and after the write
  must be identical. The production source must expose no recovery-state read
  or load path.
- Existing inherited model, evaluator, causal-control, custody, and source
  closure tests must remain passing. Recursive closure, independent review,
  narrow clean-export certification, and separate one-shot authority must be
  committed before reservation or execution.

## Falsification and scale gates

- Update 0 is informational and must pass structural, source, access,
  custody, accounting, target-isolation, exact temporal-objective,
  finite-gradient, route-membership, and single-forward integrity.
- Update 100 is informational. It has no threshold, branch, restart,
  checkpoint, or early stop.
- At update 400, the unchanged conjunctive gate requires structural
  integrity; all twelve causal-control checks; physical margin count strictly
  greater than `72`; total physical shortfall strictly below
  `68.96954700805838`; and rough depth p95 strictly below
  `1.8582415819168085` m. Each of coordinate-matched persistence, shuffled
  action, wrong RGB, and the frozen train-action prior retains its positive
  mean, positive bootstrap lower-95, and at-least-six-of-eight-family triplet.
- Any update-400 scientific failure is terminal. It writes no recovery state
  or development checkpoint. Only a complete pass writes the recovery
  snapshot and then continues in the same process without replaying or
  restarting its first 400 updates.
- Update 1000 is reached only after that pass and retains the exact inherited
  final gate: V12 full arm `24/24`; at least `112/189` physical margins;
  shortfall strictly below `33.05143763708337`; at least one complete physical
  scope; rough pixel balanced accuracy strictly above
  `0.8198594673963917`; rough ground balanced accuracy strictly above
  `0.647134926562893`; rough depth p95 strictly below
  `0.9777327477931971` m; and structural integrity.
- Only a complete update-1000 pass publishes the development checkpoint. No
  incomplete or failed run may publish one. The update-400 recovery snapshot
  is not a development checkpoint and grants no downstream access.

## Family-stop rule

- Any valid update-400 gate failure retires this per-row
  persistence-contrastive temporal family, including margin, temperature,
  coefficient, onset, reduction, denominator, baseline-mixture, clipping,
  and local gradient-routing variants.
- If update 400 passes but update 1000 fails, V25 is terminal and the same
  local family is retired. There is no retry, resume, extension, alternate
  seed, checkpoint continuation, or second V25 attempt.
- A later proposal after valid V25 failure must materially change the learned
  temporal target or world-model architecture. It may not tune P25 or J24.
  This preregistration does not authorize such a successor.

## One-shot and protected-access boundary

- Schema/evidence prefix:
  `lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25`.
- Fresh output root:
  `.generated/go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25/attempt_v1`.
- Prospective fresh clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v25-per-row-persistence-contrastive-source`.
- Both roots must initially be absent. There is exactly one fresh attempt and
  no retry, resume, recovery execution, extension, replacement, alternate
  seed, or second attempt.
- Authority may cover only exact independently reviewed source plus frozen
  train and checkpoint-selection inputs on the reviewed runtime and hardware.
  Source closure must exclude every runtime artifact and protected role.
- Until a complete update-1000 pass, probability calibration, G2, navigation,
  held-out, sealed, promotion, production, and deployment remain forbidden.
  The write-only recovery artifact grants none of those accesses.
- This preregistration grants no source export, generated-input access, data
  read, reservation, GPU use, training, execution, recovery read, checkpoint
  use, calibration, G2, navigation, held-out, sealed, promotion, production,
  or deployment authority.
