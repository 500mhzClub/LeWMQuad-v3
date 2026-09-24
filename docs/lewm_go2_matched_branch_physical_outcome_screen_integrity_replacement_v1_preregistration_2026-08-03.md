# Go2 matched-branch physical-outcome screen integrity replacement V1

**Frozen:** 2026-08-03, after the original `attempt_v1` ended in a
fail-closed reporting-adapter `AttributeError`, after an independent terminal
failure and replacement-admissibility audit, and before any replacement
scientific result is computed. **Status:** preregistration only; this document
is not execution authority.

## 1. Decision and claim boundary

This document permits the source review and later separate authorization of
exactly one science-identical integrity replacement for the frozen Go2
matched-branch physical-outcome screen V1. It is not a retry, resume,
completion, amendment, or reinterpretation of the consumed original attempt.

The scientific question remains exactly:

> Do retained pre-action odometric history and current visual context permit a
> small action-conditioned physical dynamics model to rank the nine executed
> branches better than the fixed task/action prior?

The replacement remains a development-only mechanism screen. It does not
train a JEPA, RSSM, Dreamer agent, policy, critic, reward model, or planner and
cannot establish navigation, rollout, safety, memory, deployment, or
production usefulness. A positive visual result permits only the separately
preregistered successor comparison already specified by the original
scientific contract.

## 2. Consumed predecessor and exact bindings

The original scientific contract remains the following immutable document:

| Document | SHA-256 | Bytes |
|---|---|---:|
| original preregistration | `6b758b33948ebd621698d47ec01a892c52f473fb6bec930fcdf1cb459fd8da3f` | 10,369 |
| original source review | `4b324b9d2d443d7d87beb043ca15eba9dfa8214b8a62219678eccb51366e61e0` | 6,100 |
| original execution authority | `b4ea0a0fd688543c5bfbfdc7c8d9f4db28bb7aa4025c08188d4cc084a250a696` | 13,631 |
| terminal failure and replacement-admissibility audit | `a3f889aa6494b67800ed5224f9ebe97bf266a6f40d444c8a36c90182332cf511` | 14,262 |

The audit is committed at
`446d6f902015a7e1aa2b27e88f92fb849114b559` and its exact path is
`docs/lewm_go2_matched_branch_physical_outcome_screen_v1_terminal_failure_and_replacement_admissibility_audit_2026-08-03.json`.
It grants no authority by itself.

The consumed original output root is
`.generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v1`.
Its exact inventory and bindings are:

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| `reservation.json` | `fba63ae369d73109c0d0e8287230c738dbeb75d6b4dc4e92954b83628d8a0c7a` | 514 |
| `physical_outcome_checkpoint.pt` | `90fa756cae37d7dda04d10a69fa9093b4f6447b55cb56c1f548909218510f3c7` | 2,544,111 |
| `terminal.json` | `c6c0a615639e55a4d7d2a513c769cd16a8085eaf0c0de7a1fbed15ad96c1ff10` | 489 |

The terminal status is `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`, its
`result_binding` is null, deterministic replay did not run, and it authorizes
neither retry nor resume. `evaluation.json`, `replay.json`, and `result.json`
do not exist. The original root and terminal must remain byte-identical.

The old checkpoint is quarantined. A source reviewer or authority validator
may stream its bytes solely to confirm the exact SHA-256 and byte count above.
It must never be deserialized, scored, inspected for training diagnostics,
used for model selection, warm-started, resumed, compared with replacement
weights, or supplied to the replacement runner, evaluator, or replay. It is
failure-lineage identity evidence, not a scientific input.

## 3. Truthful failure boundary and result independence

The original attempt completed all train-only fitting, checkpoint validation,
and durable checkpoint publication. It then loaded the evaluation cache,
built the evaluation dataset, and entered primary evaluation. First-learned-
member predictions and physical rank scores may have existed transiently in
the failed process before reporting raised:

`'_PhysicalLabelsV1' object has no attribute 'planar_clearance_proxy_min_m'`.

It would therefore be false to claim that evaluation never began. However,
the first candidate report did not complete; no arm report, paired bootstrap
comparison, scientific gate, evaluation artifact, replay artifact, result, or
scientific verdict was completed or published. Neither the independent audit
nor this preregistration computes, recovers, infers, or inspects any candidate
score.

The defect is independent of every possible prediction. Every branch uses the
same screen-local label adapter. The imported frozen reporter unconditionally
reads `planar_clearance_proxy_min_m` and then
`grid_recoverability_proxy`, while the adapter supplies neither attribute.
Thus every possible selected action and score matrix reaches the same first
missing attribute. These optional legacy nonphysical proxies are not model
inputs, targets, physical ranking keys, primary metrics, bootstrap variables,
or decision gates. The permitted correction below is dictated entirely by
the frozen interface and terminal exception; it is not selected in response
to model performance.

## 4. Sole permitted scientific-source correction

The screen-local `_PhysicalLabelsV1` adapter may add exactly these fields:

```python
planar_clearance_proxy_min_m: float | None
grid_recoverability_proxy: float | bool | None
```

Both must be populated as `None` for every branch because the admitted bounded-
branch receipts do not retain these legacy proxy values. The generic reporter
must consequently retain null proxy values in group rows and omit its optional
aggregate `nonphysical_proxy_metrics`. The two values may not be synthesized,
loaded from another artifact, used in selection, or promoted into any metric
or gate.

One focused regression test must exercise the real receipt-to-group adapter
through the imported selection reporter and verify all physical diagnostic
fields, both null proxy fields, and absence of the proxy summary. No other
model, data, projection, target, training, prediction, ranking, metric,
bootstrap, threshold, gate, verdict, or replay implementation may change.

Only necessary lifecycle plumbing may otherwise change: replacement
preregistration/source-review/authority bindings, the fresh output-root and
attempt identity, explicit predecessor-failure bindings, one-shot reservation
validation, and focused tests for those controls. Any replay edit must be
limited to accepting the new authority and root identity; its retraining,
comparison, reproduction fields, and verdict logic remain unchanged.

## 5. Frozen data and custody

The immutable panel remains 128 train states in 16 scenes and 128 evaluation
states in 16 disjoint scenes, balanced across the same eight families. Each
state has all nine physically executed branches. States and scenes, not the
1,152 branch rows per role, remain the independent units. The evaluation role
is already development-exposed and supports no fresh confirmation claim.

The replacement authority must reproduce all 15 original direct-input
bindings exactly:

| Input | SHA-256 | Bytes |
|---|---|---:|
| posthoc manifest | `87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e` | 11,964 |
| posthoc terminal | `a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56` | 1,250 |
| posthoc train JSONL | `edc6f88bb105c39575477fbfbb0224bf0312cf5ee3e90551f86a9c11c2ebb447` | 30,432,624 |
| posthoc eval JSONL | `531debbc431f2f8afc83a491b491b8822134c831b16ca4d283fe1e7f4ba07768` | 30,411,588 |
| posthoc terminal review | `bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669` | 2,844 |
| upstream physics result | `25caf0a5d4c69e99559a663aa4cae96fb23ef191ccf34486804c3f2243553314` | 183,320 |
| physics receipt check | `faeb50293bc684e35b6d725b027983ad0110e739db2d7b1aca1926e89a547dc6` | 892 |
| consumed collection terminal | `f7d2796139645892d22ad6bb99d26caffc2b5c3dcac2a655b1883b299d22bff4` | 12,949 |
| authorized collection plan | `8fe34054bb9ae709b6a8ecfea5fdae55c742d1b2e22af3c289d27a77f11c66ef` | 343,973 |
| calibration receipt | `58d1291ede7ee03a93d68eb7cec80c9322c47cd0b1d5fd1c41bf8f4b49ad484e` | 72,475 |
| DINOv2 train cache | `164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b` | 302,107,682 |
| DINOv2 train-cache receipt | `e94ec5d188811c44d4cc870e76d1888aa6f30ee6d423557ee9f3e2918a700994` | 1,770 |
| DINOv2 eval cache | `00a2e197d98effcd192392f50170648622a7210f954075002dc8b43110c636f8` | 302,106,281 |
| DINOv2 eval-cache receipt | `d3e928cc563beb4dd850f34ca41915b8e5974c6d0b1b182602f3e3f20828421c` | 1,770 |
| predecessor dense-DINO terminal review | `f6ed2d09a407a4cf70097eaa4b2dcffd223e598e4eb59cf8e751997459384020` | 27,120 |

All 256 upstream state receipts must again be rehashed through the same strict
direct derivation path. No RGB leaf may be opened, no encoder may execute, and
no protected, held-out, or sealed material may be named or accessed. No data
may be collected, rendered, generated, replaced, resplit, filtered, or scaled.

## 6. Frozen features, targets, arms, and training

The 12-scalar pre-action physical input remains, in order:

1. body-local `(dx,dy,wrapped_dyaw)` from context pose 0 to 1;
2. body-local `(dx,dy,wrapped_dyaw)` from context pose 1 to 2;
3. mean `(vx,wz)` for each of the two past executed `5 x 3` command blocks;
4. candidate requested `(vx,wz)`.

Absolute world pose, IDs, hashes, future executed commands, clipping,
trajectory samples, endpoints, labels/ranks, target RGB, and successor tokens
remain forbidden model inputs. Targets remain prebranch-body-frame endpoint
`dx`, `dy`, wrapped `dyaw`, and physical path length. Progress remains an
evaluation label, not a regression target.

The arms remain exactly:

- **A:** the frozen 27-coefficient, nine-head task/action-only ridge, identity
  `69895316b19bc179e35fdd76905aadbd50b6ad3e22e965b662ba59672c52886a`,
  refitted on the same train role only, with required evaluation regret
  `0.17441406250000002`;
- **B:** the 12 physical inputs followed by 16 zero-valued visual slots;
- **C:** the same physical inputs plus the frozen 16-dimensional current-
  context visual projection.

For C, each of the three context DINO grids remains `(16,16,384)`, mean-pooled
over non-overlapping `4 x 4` token blocks to `(4,4,384)`, then concatenated in
time/row/column/channel order to 18,432 values. The 16-component PCA remains a
train-only float64 column-centered thin SVD in descending singular order, with
the largest-absolute loading made positive using the smallest index on ties.
Evaluation uses the frozen train mean/components. Target or successor grids
remain forbidden.

B and C remain identical `28 -> 16 -> 4` tanh MLPs with 532 parameters per
member and seeds `2026080311`, `2026080312`, and `2026080313`. For each action,
the train-only mean physical outcome remains the base prediction and the MLP
predicts the standardized residual. Input population mean/scale and residual-
output population scales remain train-only, with zero or sub-`1e-8` scale
replaced by one. Initialization remains dedicated-CPU-generator Xavier-uniform
weights with zero biases.

Training remains CPU float32 with deterministic algorithms and one Torch
thread; complete nine-action state minibatches of 16 states; exactly 1,024
AdamW updates per member; learning rate `3e-4`; weight decay `1e-4`; betas
`(0.9,0.999)`; epsilon `1e-8`; gradient clip norm `1.0`; unweighted mean
squared standardized-residual loss; and the frozen seed-local permutation
schedule of 128 seed-local permutations with eight contiguous complete-state
batches per permutation. B and C retain identical initial states and schedules
for each matched seed. There is no early stopping, evaluation monitoring,
checkpoint selection, coefficient search, retry, or resume. Ensemble
predictions remain the arithmetic mean of the three decoded member outcomes.

The bound runtime remains Python `/usr/bin/python3.12`, Torch
`2.14.0.dev20260726+rocm7.1`, NumPy `1.26.4`, and Pillow `10.2.0`.

## 7. Frozen scoring, analysis, gates, and verdict

Predicted progress remains `||g|| - ||g - (dx,dy)||`; predicted path length is
clamped at zero only for physical scoring. The nine candidates remain ranked
using the existing one-centimetre contract: progress quantized descending,
then path length quantized ascending, then action ID. The zero-fall/zero-tip
panel remains `NOT_TESTABLE_ZERO_EVENT_SUPPORT`; no safety prediction or claim
is permitted.

The primary metric remains evaluation normalized physical rank regret, lower
is better. All candidate-minus-baseline intervals remain paired, equally
weighted across families, whole-scene resampled over the same 16 evaluation
scenes, with 10,000 draws and seed `2026080314`. Per-output RMSE and joint
standardized MSE versus zero motion and train-only action means remain
diagnostic only.

All seven gates remain exact:

1. Exact source/input/output bindings, no leakage, finite tensors, and no
   RGB, encoder, or protected access.
2. Privileged oracle regret `0` and oracle-equivalent rate `1`.
3. B-minus-A bootstrap upper 95% endpoint `< 0`, with every B seed regret
   below A.
4. C-minus-A bootstrap upper 95% endpoint `< 0`, with every C seed regret
   below A.
5. C-minus-B bootstrap upper 95% endpoint `< 0`, with every matched-seed
   C-minus-B regret negative.
6. Any arm used for advancement has point regret below random expectation.
7. A fresh process independently rebuilds the projection and PCA, retrains
   all six members, and exactly reproduces identities, predictions, selected
   actions, summaries, intervals, gates, and verdict.

Verdict precedence and strings remain exact:

- Gates 1, 2, 4, 5, 6(C), and 7 pass:
  `PASS_VISUAL_PHYSICAL_DYNAMICS_HEADROOM`.
- Otherwise gates 1, 2, 3, 6(B), and 7 pass:
  `PASS_ODOMETRY_ONLY_PHYSICAL_DYNAMICS_HEADROOM`.
- Otherwise:
  `STOP_RETAINED_INPUT_PHYSICAL_DYNAMICS_HEADROOM_NOT_ESTABLISHED`.
- Any contract or infrastructure failure:
  `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`.

No diagnostic can alter this decision and no threshold may be reinterpreted,
relaxed, supplemented, or tuned after the replacement begins.

## 8. Fresh full execution and replay

The only replacement identity is the initially absent root:

`.generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v2_integrity_replacement_v1`

Reservation must create that exact root exclusively and consume the sole
attempt before any scientific work. The runner must reject the original root,
any pre-existing replacement root, symlinks, alternate output paths, retry,
resume, overwrite, refill, or partial completion.

The replacement must start from the frozen inputs and seeds, rebuild the
train-only physical projection, PCA, outcome statistics, input statistics,
and task/action control, and fit all six members from their registered initial
states. It must write a new checkpoint before evaluation publication. The old
checkpoint cannot participate. It must then compute the primary evaluation,
launch exactly one fresh-process cache-only replay, and require the same exact
reproduction contract.

On success the exact output inventory remains:

- `reservation.json`;
- `physical_outcome_checkpoint.pt`;
- `evaluation.json`;
- `replay.json`;
- `result.json`;
- `terminal.json`.

On an exception, the runner must fail closed with truthful partial-artifact
accounting and `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`. No failure grants
retry, resume, overwrite, or a second integrity replacement.

## 9. Required implementation review and authority

The implementation and focused tests must be committed after this document.
An independent reviewer who authored neither the adapter correction nor the
lifecycle plumbing must compare the complete replacement source closure with
the original reviewed source commit
`7c0603440d27206f3c07789ff53274fa3a758f23` and bind every transitive source
and test path exactly.

The review must establish at least:

- the two null optional-proxy fields and focused regression are the only
  scientific-source change;
- model, input, target, PCA, training, scoring, bootstrap, gates, status
  precedence, and replay science are unchanged;
- all 15 direct inputs and all 256 receipt bindings remain exact;
- the original attempt inventory and terminal-failure audit remain exact;
- no code path deserializes or scientifically consumes the old checkpoint;
- the fresh root, reservation, failure terminalization, and one-shot boundary
  fail closed;
- the complete original focused suite plus replacement regression and
  lifecycle tests pass in the bound runtime; and
- compilation and whitespace checks pass.

Only after that review is committed may a separate execution authority be
created. It must bind this preregistration, the replacement source review and
reviewed commit, the original preregistration/source review/authority, the
terminal failure audit, the three original attempt artifacts as lineage
evidence, the exact replacement source closure, all 15 scientific inputs, the
original runtime, the unchanged scientific configuration, and the fresh root.

The authority must state explicitly that the replacement is science-identical,
the original attempt remains consumed, exactly one fresh full fit and replay
is authorized, and collection, RGB access, encoder execution, protected
access, old-checkpoint deserialization or reuse, retry, resume, and any second
replacement are unauthorized. Authority and the complete bound closure must
be rehashed before reservation, after reservation, and before final result
publication.

## 10. Terminal audit and stopping rule

After execution, an independent terminal review must bind the authority,
reservation, every replacement artifact, exact inventory, fresh-fit and replay
evidence, custody counts, gates, and terminal decision. It must separately
confirm that the old checkpoint was never deserialized or used scientifically
and that the original root is unchanged.

This is the final integrity replacement for this screen. If it ends in an
infrastructure failure, no second replacement is authorized and the screen
closes without a scientific conclusion. If it completes, its frozen PASS or
STOP decision is terminal for this mechanism screen. No outcome authorizes
navigation, deployment, protected evaluation, additional data generation,
3-TB scaling, threshold tuning, or another attempt.
