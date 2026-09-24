# Go2 dense V-JEPA 2.1 physical-interface ceiling V1 integrity replacement V1

**Frozen:** 2026-08-03, after the original attempt's terminal infrastructure
failure and its committed independent terminal review, and before replacement
implementation, authority, cache access, RGB decoding, encoder execution, or
training.

## 1. Scope and controlling records

This document authorizes at most one science-identical integrity replacement
for the original development-only physical-interface ceiling. It does not
change, retry, reinterpret, or extend the scientific experiment. These three
records are bound lineage:

| record | exact path | SHA-256 | bytes |
|---|---|---|---:|
| Original frozen preregistration | `docs/lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_preregistration_2026-08-03.md` | `ef5c687d509929169280a456618e92e92f2a072a646bc292be3d16850f801ad0` | 20,816 |
| Original infrastructure-failure terminal | `.generated/dev/go2_dense_vjepa2_1_physical_interface_ceiling_v1/attempt_v1/terminal.json` | `b8c04572ba67baccbd81d54bf9039398c490a451e70b7c83d1cc03f07565ebf7` | 1,801 |
| Committed independent terminal review | `docs/lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_terminal_review_2026-08-03.json` | `69f2a77962abcfec16331a2fb71b799521173781da102c5fac7af4a0246954bc` | 10,826 |

The terminal review is bound at commit
`55bfaac5aa400c904c8ee70e071936e2a11d577e`. The original terminal is an
infrastructure failure and contains no admissible scientific decision. No
provisional metric, gate value, model state, selection, or apparent direction
from that consumed attempt is evidence for this replacement.

The original preregistration is incorporated here in full by its exact byte
binding. It remains controlling for every scientific field. This document is
controlling only for the three integrity changes enumerated in section 3. If
an implementation, review, authority, or runtime record cannot satisfy both
documents without ambiguity, execution is prohibited.

## 2. Science-identical contract

The replacement preserves without modification every item below from the
original frozen preregistration:

- its purpose, interpretation boundary, development-only role, and prohibition
  on treating actual future tokens as deployable planner inputs;
- all scientific inputs and lineage witnesses, with the same exact paths,
  SHA-256 values, byte counts, classifications, and allowed access modes;
- the exact 128-state/16-scene train role, disjoint 128-state/16-scene
  evaluation role, eight families, two scenes per family per role, nine
  executed branches per state, 1,536 artifacts per role, 256 bound state
  receipts, and 1,536 bound evaluation RGB leaves;
- train role-plan identity
  `f6f94cf589ec44324fdefe0939aa7076e25543d984464d5b264a0b2f0ff9535b`,
  evaluation role-plan identity
  `5dbf9733fd245caff27ce5c5c2b3dc90a3fe9ca9e1bc894dc10a97d64dad9231`,
  and combined identity
  `99e60638634eff6ac244cff023cd2ae8f7aa0c53326263ba7a36fa6847386375`;
- the frozen `vjepa2_1_vit_base_384` EMA encoder, checkpoint and source
  closure, RGB decoding, 224-to-438 PIL-bilinear resize, 384 center crop,
  ImageNet normalization, `[3,1,384,384]` input, 24-by-24 to 16-by-16 area
  resampling, float32 token normalization, float16
  `[1536,256,768]` cache representation, and exact artifact order;
- the train-only PCA population, ordering, float64 covariance/eigendecomposition,
  tie break, sign convention, `K=8`, whitening floor, and bound PCA identity;
- the unchanged 245-parameter `DenseSharedSpatialReadoutV1`, its exact
  24-dimensional patch relation, four-dimensional conditioning, shared
  scorer, task/action ridge identity
  `69895316b19bc179e35fdd76905aadbd50b6ad3e22e965b662ba59672c52886a`,
  exact ridge evaluation regret `0.17441406250000002`, and residual physical
  rank target;
- all ten original arms and controls, including their construction, fitting
  status, wrong-scene involution, train-action-mean construction, action-ID
  tie breaking, retained-predecessor rows, report-only hold-constant arm, and
  exact random expectation;
- seeds `2026080303`, `2026080304`, and `2026080305`; matched initialization
  and state-order generators; float32 ROCm execution; AdamW at learning rate
  `1e-3`, weight decay `1e-2`, betas `(0.9,0.999)`, epsilon `1e-8`, no
  AMSGrad/foreach/fused path; gradient clipping at `1.0`; 256 epochs; eight
  complete-state batches of 16 states; exactly 2,048 updates per member; and
  arithmetic three-member ensembling;
- all reported metrics, per-seed/per-family/per-scene summaries, attention and
  finiteness diagnostics, zero-event-support interpretation, and physical-rank
  action selection;
- scene-unit, family-balanced paired bootstrap with 10,000 resamples, seed
  `2026080314`, percentile 95% intervals, and the convention that negative
  paired differences favor true-future V-JEPA;
- all ten qualification gates, their strict inequalities, exact oracle and
  random requirements, infrastructure/accounting requirements, and exact
  fresh-process replay equality requirements; and
- all prohibitions, lifecycle rules, output ordering, failure behavior,
  interpretation limits, and terminal decision meanings.

There is no outcome-informed change to data, split, representation,
preprocessing, PCA, readout, arm, control, target, seed, initialization,
optimizer, schedule, metric, bootstrap, gate, threshold, status, or reporting
rule. The replacement may not add an ablation, diagnostic, seed, epoch,
checkpoint choice, cache variant, model variant, or fallback behavior.

## 3. Exhaustive permitted integrity changes

Exactly these changes, and no others, are permitted relative to the reviewed
original implementation.

1. `_selected_actions_v1` becomes sentinel-safe and type-preserving. For every
   selected-action leaf it accepts only a value for which `type(value) is int`
   and the original action-validity checks pass, or the exact string
   `NOT_APPLICABLE`. An accepted integer remains the same integer; the sentinel
   remains the same string. Booleans, floats, numeric strings, `null`, other
   strings, and containers are rejected. This parser change may not alter
   scoring, selections, metrics, comparisons, gates, or serialization outside
   preserving the already-valid sentinel. Synthetic regression tests must
   prove unchanged integer behavior, exact sentinel preservation, and rejection
   of every disallowed type without opening any original-attempt scientific
   artifact.
2. Review, authority, runtime schemas, and identity strings may move only to a
   distinct `integrity_replacement_v1` namespace so that no original authority
   or output can authorize or satisfy this replacement. The independent source
   review must bind the exact replacement source/test closure and verify that
   the semantic diff is limited to item 1 plus these namespace/root changes. A
   separately committed, post-review, one-shot replacement authority must bind
   this preregistration, that review and commit, the unchanged scientific
   inputs, exact runtime/hardware, source closure, and an absent replacement
   root.
3. The only runtime output root is
   `.generated/dev/go2_dense_vjepa2_1_physical_interface_ceiling_v1/attempt_v2_integrity_replacement_v1`.

Mechanical references needed to use that root and namespace are permitted
only where the original implementation referred to `attempt_v1` or its
original schema identities. The successful inventory remains exactly, and in
the original lifecycle order: `reservation.json`, `vjepa2_1_eval.pt`,
`vjepa2_1_eval.json`, `ceiling_checkpoint.pt`, `evaluation.json`,
`replay.json`, `result.json`, and `terminal.json`.

## 4. Isolation and access contract

The replacement is a new one-shot execution, not a resume or retry. It must
perform the complete original primary workflow from immutable inputs:

1. atomically reserve the absent replacement root before the first token-cache
   deserialization or RGB decode;
2. rehash and load the original bound train V-JEPA cache;
3. decode all and only the 1,536 bound evaluation RGB frames exactly once;
4. execute the bound frozen encoder on all 1,536 frames and write a wholly new
   replacement evaluation cache and receipt;
5. fit PCA and all six networks from their fixed initial states and schedule,
   write the checkpoint before publishing evaluation results, and evaluate all
   fixed arms and gates; and
6. launch a fresh process that rehashes only the bound train cache and the new
   replacement evaluation cache, rebuilds PCA, reinitializes and retrains all
   six networks, and exactly reproduces every field required by original gate
   10 without RGB access or encoder execution.

The replacement must not open, hash for reuse, load, deserialize, copy, or
derive metadata or state from any original-attempt evaluation cache or receipt,
checkpoint, evaluation, replay, or result. In particular, all content under
the original `attempt_v1` root other than the exact bound failure terminal is
inadmissible. The committed terminal review is a lineage record only. The
replacement may not use any provisional selection, score, metric, interval,
gate, verdict, parameter, PCA state, action mean, or timing from the original
attempt.

Some implementers were exposed to provisional `attempt_v1` content while
diagnosing the integrity failure. That exposure is disclosed, does not turn
the content into evidence, and may not influence any scientific or execution
field. The only admissible response to the diagnosed failure is the narrowly
specified sentinel parser repair and regression above. Independent review must
confirm this constraint before authority.

Train RGB, every other encoder, collection, scene generation, the passive 3 TB
pool, protected, held-out, sealed, production, promotion, deployment, planner
integration, navigation, downstream training, and downstream-successor
execution remain forbidden. Symlinks, path escapes, unexpected roles, altered
artifact order, and any input/source/authority mismatch remain infrastructure
failures.

## 5. Gates, precedence, and terminal decisions

The original ten gates apply verbatim. In particular, all source/authority/
role/cache/order/RGB/access/finiteness checks, the exact oracle check, all six
strict upper-95% superiority comparisons, the strict random point comparison,
and the full fresh replay must pass. No loss, point estimate, seed, family,
attention view, provisional value, or post hoc analysis can override a gate.

Decision precedence is fixed:

1. Any incomplete execution, invalid input or source, access/accounting
   violation, non-finite value, failed exact-replay field, exception, or other
   infrastructure defect yields
   `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`. No scientific gate verdict may
   be published from that execution.
2. Only a complete infrastructure-valid execution reaches scientific
   precedence. If every original gate passes, the sole status is
   `QUALIFY_VJEPA_DENSE_INTERFACE_FOR_SEPARATE_BACKBONE_LEVEL_MATCHED_BRANCH_JEPA_PREREGISTRATION`.
3. If execution is complete and infrastructure-valid but any scientific gate
   fails, the sole status is
   `STOP_FROZEN_VJEPA_PHYSICAL_INTERFACE_NOT_ESTABLISHED`.

A qualification permits only a separately preregistered proposal; it grants no
training, predictor, planning, navigation, G2--G8, production, promotion, or
deployment authority. A STOP closes the frozen single-frame V-JEPA/readout
route under the original prohibitions. An infrastructure failure grants no
retry, resume, extension, second integrity replacement, or further attempt.
This replacement is consumed when its root is reserved, regardless of outcome.
