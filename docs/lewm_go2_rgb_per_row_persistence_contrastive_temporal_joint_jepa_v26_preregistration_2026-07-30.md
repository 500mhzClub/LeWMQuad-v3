# V26 per-row persistence-contrastive temporal joint-JEPA schema-integrity preregistration

Date: 2026-07-30

Status: preregistered fresh, science-identical schema-integrity successor only.
No V26 source root, output root, reservation, generated-input access, GPU work,
training, recovery write, checkpoint, calibration, G2, navigation, held-out, or
sealed access has occurred or is authorized here.

## Trigger and terminal predecessor

- The sole V25 attempt is consumed and terminal. It may not be retried or
  resumed.
- Its complete terminal audit is frozen at
  `docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25_terminal_failure_result_2026-07-30.json`
  in commit `26c8fd902319c06d4dbf25cab36a63ec2df44081`, with file SHA-256
  `5c8d6d80ce24c60900c49f6cf49979c3001024666a2156d945e526b396dd1596`,
  byte count `10380`, and canonical content SHA-256
  `59423f03ca153ca481d71ea4e88aaa625128ece4a15eb8b6253ae4f009272929`.
- V25 published a valid update-0 baseline and loaded the four scheduled
  update-1 microbatches, then stopped at `train_update_1` before a completed
  JEPA training-objective or predictor forward, backward call, optimizer step,
  EMA step, training update, or accounted training presentation. P25 was never
  evaluated and no recovery state or checkpoint was published. This is no
  evidence for or against the V25 scientific hypothesis.
- The exact exception literal was
  `V25 microbatch schema changed from frozen V24`. The V25 training wrapper
  incorrectly exposed its full V25 schema validator under the inherited V21
  and V23 projected-schema compatibility names. The unchanged executor
  projects the full batch to V21 keys before resolving
  `runtime.training_module._validate_microbatches_v21`; that valid projection
  was therefore rejected by the wrong validator.

V26 is a fresh science-identical one-shot successor, not a V25 retry, resume,
extension, alternate seed, or checkpoint continuation.

## Sole operational correction

The V26 training wrapper privately loads the frozen V25 training module and
delegates every scientific type, constant, objective, parameter partition,
optimizer operation, accounting operation, and update implementation to it.
Only the compatibility-validator aliases change:

```text
full_v25_validator = frozen_v25._validate_microbatches_v25
projected_v21_validator = frozen_v25._v24._validate_microbatches_v21
projected_v23_validator = frozen_v25._v24._validate_microbatches_v23

_validate_microbatches_v13 = full_v25_validator
_validate_microbatches_v21 = projected_v21_validator
_validate_microbatches_v23 = projected_v23_validator
_validate_microbatches_v24 = full_v25_validator
_validate_microbatches_v25 = full_v25_validator
_validate_microbatches_v26 = full_v25_validator
```

The frozen V25 module itself is not mutated. V26's executor, launcher, and
source checker are thin frozen-V25 adapters that change only V26 lifecycle
identity, preregistration/evidence bindings, schema and arm names, and fresh
source/output roots. V25's write-only update-400 recovery mechanism and its
ordering remain exact; V26 adds no reader or resume path.

No model code, tensor operation, architecture, parameter, initialization
value, dataset, role, RGB input, label, seed, schedule row, presentation,
loss, coefficient, onset, optimizer setting, EMA operation, gradient route,
clip rule, diagnostic, causal control, evaluator, threshold, gate, stopping
rule, recovery payload, accounting multiplier, or cap may change.

## Frozen scientific identity

V26 preserves the V25 preregistration in commit
`f00e20df3b429f9242516ac38f67fea587e04b22` and the independently reviewed
V25 scientific source in commit
`43231c689547b66de83f3cafbfac270455a7a234` exactly, including:

- the RGB encoder, eight-height object-space representation, semantic and
  survival heads, local action-conditioned predictor, EMA target path, model
  class, parameter counts, and all initialization values;
- the denominator-free per-row temporal objective
  `P25 = mean(softplus(e_pred_i - stopgrad(e_persist_i)) / log(2))`, with the
  same LayerNorm/SmoothL1 energy, detaches, beta, threshold, row ordering, and
  legacy diagnostic-only ratio;
- the unchanged jointly optimized composition
  `N25 = S + P25 + U + R_inherited + O`, `L25 = N25 + C + J24`, and
  `J24 = F + R_output`, with P25 entering exactly once;
- J24's exact 96-tensor, 3,106,409-parameter capped auxiliary route and the
  13-tensor, 259,008-parameter predictor core protected from J24 only while
  remaining inside the normal joint JEPA update;
- N320 initialization, constructor seed `20260712`, schedule seed `20260713`,
  experiment/bootstrap seed `20260728`, projection seed `20260729`, float32
  AdamW, learning rates, parameter groups, gradient addition and clipping,
  EMA, and one optimizer plus one EMA step per completed update;
- the exact 4,262-pair schedule from presentation zero, four microbatches of
  four, train and checkpoint-selection roles, data and labels, observation
  updates `(0,100,400,1000)`, terminal updates `(400,1000)`, eight-family
  registry, physical metrics, causal controls, and every threshold; and
- the maximum of 1,000 updates and 16,000 ordered presentations.

V26 starts once in a fresh process from exact initialization. It may not open
or reuse any V25 model, optimizer, EMA, RNG, schedule state, metric, trace,
recovery state, output, checkpoint, or mutable runtime state. Only committed
V25 source/evidence documents may enter the V26 source closure.

## Focused acceptance

- Source tests must prove all six compatibility aliases above by object
  identity and prove that every V26 scientific callable and type delegates to
  frozen V25.
- One regression must traverse the actual inherited
  V26/V25 -> V24 -> V23 -> V21 executor projection path using four ordered
  full-schema batches. It must show that V21 receives the exact projected V21
  keys, V23 uses its exact projected validator, and the full-schema boundary
  still uses the V25 validator. The regression must fail when the two V25
  incorrect aliases are substituted.
- Existing focused V25 tests, V21-through-V25 regressions, model/lifecycle
  tests, and latent-energy tests must remain passing. No generated scientific
  input or GPU is needed for source acceptance.
- Recursive source closure, independent source review, narrow clean-export
  certification, a fresh exact source root, and separate one-shot authority
  must be committed before reservation or execution.

## One-shot identity and gates

- Schema/evidence prefix:
  `lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26`.
- Experiment arm: `per_row_persistence_contrastive_temporal_v26`.
- Exact fresh attempt root:
  `.generated/go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26/attempt_v1`.
- Exact clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v26-per-row-persistence-contrastive-source`.
- There is exactly one fresh V26 attempt. Both roots must initially be absent.
  Retry and resume are false.
- Update 0 remains informational. Update 100 remains informational. The exact
  V25/V24 update-400 conjunctive falsification gate and update-1000 final gate
  are unchanged. Only a complete update-400 pass may write the unchanged
  write-only recovery snapshot and continue; only a complete update-1000 pass
  may publish the development checkpoint.
- Any valid update-400 scientific failure retires the per-row
  persistence-contrastive temporal family under V25's family-stop rule. Any
  V26 source, authority, reservation, custody, exception, or recovery-write
  failure is terminal and publishes no failed checkpoint. No further
  integrity successor is preregistered here.

Until a complete passing update-1000 result, probability calibration, G2,
navigation, held-out, sealed, production, promotion, deployment, recovery
read, retry, resume, and extension remain forbidden. The existing V4
30-scene sealed benchmark remains unopened.
