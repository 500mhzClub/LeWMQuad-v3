# Go2 Physical Micro-Overfit Protocol

Date refrozen: 2026-07-10, before authoritative panel or GPU output

This protocol adjudicates whether the current 112 px physical traversability
model is limited by the patch14/8x8 representation and whether the
patch7/16x16 alternative is sufficiently expressive and scene-general to earn
one full G2 training candidate. It is a train-role diagnostic. It does not open,
score, calibrate on, or select from checkpoint-selection, probability-
calibration, or G2-evaluation artifacts.

For this diagnostic, this document supersedes the earlier temporal-quartile,
early-stop, and single-seed wording in the 2026-07-09 generalization execution
contract. The broader G2/G3 ordering and isolation contract remains in force.

The causal intervention is the **patch/tokenization-resolution bundle**. Patch
size, token grid, patch-embedding tensor shape and parameter count, and
attention compute change together. The result must not be described as a
causal estimate of token resolution alone.

## Frozen Panel

An initial metadata-only eight-transition/single-scene pilot was rejected before
an authoritative panel was written. Its post-selection support audit found zero
medium-maze FREE support at 2-3 m and beyond 3 m in fit and cross panels, only
5 large-maze far-FREE fit cells, and 54 small-maze far-FREE cross cells. The
support gate therefore worked as intended. No threshold was weakened and no GPU
or model output was produced.

The superseding selection is label-independent and deterministic under seed
`go2_physical_microfit_patch7_v1`:

- Families: open obstacle field, rough local dynamics, small enclosed maze,
  medium enclosed maze, and large enclosed maze.
- Each family has exactly nine train scenes, ranked by SHA-256 under the frozen
  `pool-scene` namespace. The first four form the fit/same pool; the remaining
  five form the cross pool.
- Within each fit/same scene, streams are SHA-ranked under `pool-stream`; even
  ranks are eligible only for fit and odd ranks only for same-pool holdout. All
  streams in the cross pool are eligible only for cross-scene holdout.
- Panels are processed in `fit`, `same_scene_holdout`,
  `cross_scene_holdout` order and families in the canonical order above.
  Eligible `(scene, stream)` pairs are SHA-ranked under `panel-prefix`; rows
  within a stream are SHA-ranked under `row`.
- Every pool scene first contributes two metadata-valid transitions so the
  same-scene wrong-view control is defined. The remaining fixed prefix is then
  scanned until exactly 32 transitions per family are selected. A stream is
  skipped only for an already-used row or endpoint hash, or equal current/next
  endpoint hashes. Labels and model outputs never affect ranking or skipping.
- Each panel therefore contains 160 transitions and 320 frames; all three
  panels contain 480 rows and 960 globally unique endpoint hashes.
- At most one transition is selected per `(env_index, episode_id,
  reset_count)` stream.
- Fit and same-pool holdout use the same four scenes but disjoint episode
  streams. Cross-scene holdout uses the other five train scenes.
- Global rows and current/next image hashes are disjoint across all panels.
- The row contract records scene, family, global row, environment, episode,
  reset, current/next episode step, frame index, timestamp, primitive,
  relative SE(2), label-shard provenance, and image provenance.

The global JSONL parser temporarily materializes complete row objects,
including non-train path strings. No non-train path is emitted into the panel,
dereferenced, hashed, decoded, or passed through the model.

## Frozen Support Abort

Labels are inspected only after the panel is fixed. Every panel must satisfy:

- nonzero UNKNOWN, FREE, and OCCUPIED support in aggregate and in every family;
- at least 1,000 aggregate true-FREE cells in each gated distance bin:
  `[1,2) m`, `[2,3) m`, and `[3,+inf) m`;
- at least 100 true-FREE cells for every family in every gated distance bin.

The `[0,0.5) m` and `[0.5,1) m` bins are reported but not gated. Any support
failure aborts the experiment. There is no row replacement, panel reselection,
threshold adaptation, or bin adaptation.

The superseding N=32 rule was reproduced once to a non-authoritative `/tmp`
panel before source freeze. Its file/content SHA-256 values were
`ab74c8c14345dd19d42bd0f4ae68af299afbae77ecb6ed0735b69de143fef65d` /
`e319c67dd226911b2f0b51d52d0abd0dc0a2edc90d5dc11f9f8f432fb4a598a5`.
It contained 160/160/160 rows and passed without label-driven replacement: the
smallest aggregate gated-bin support was 20,551 cells and the smallest
family/bin support was 512 same-pool medium-maze far-FREE cells. One
metadata-unusable medium cross stream was skipped, and every selected pool
scene contributed at least two transitions. This feasibility read produced no
model output and is not the authoritative panel or a G2 result.

Selected source NPZ files are scene archives that also contain unselected
train rows and arrays. NPZ access materializes archive-level arrays. The
optimizer indexes only the selected fit rows; unselected rows never become
training examples or evaluation frames. These facts are disclosed in the
result ledger and are not described as row-level byte isolation.

## Matched Arms

- `patch14_8x8`: 112x112 RGB, patch size 14, 8x8 tokens, projective sigma 1.0
  token.
- `patch7_16x16`: 112x112 RGB, patch size 7, 16x16 tokens, projective sigma
  2.0 tokens.

Both arms use the same images, ordered minibatches, batch size, update count,
loss, center-projective column-attention lift, camera geometry, optimizer
stage, and normalized sigma/token-side ratio. All identically shaped initial
tensors are copied from patch14 into patch7. The artifact reports both
parameter counts, their delta, and every variant-specific tensor shape.

## Fixed Training Stages

The production-faithful stage is AdamW, learning rate `2e-4`, weight decay
`1e-4`, for exactly 2,000 updates. Evaluation occurs every 100 updates.

If either faithful arm fails its terminal fit gate, the ceiling stage is
mandatory for both arms, restarted from their original initial states. It is
AdamW, learning rate `1e-3`, weight decay `0`, for exactly 3,000 updates. A CLI
flag cannot suppress or enable this rule.

There is no independent early stopping. The first single pass and first run of
three consecutive passes are retained as diagnostics only. Authoritative stage
pass requires the aggregate fit gate and all five family fit gates to pass at
each of the final three evaluations:

- faithful: steps 1,800, 1,900, and 2,000;
- ceiling: steps 2,800, 2,900, and 3,000.

Per-arm expressivity is `faithful_pass OR ceiling_pass`. A faithful pass is
never erased by a ceiling result.

## Fit Gate

Every aggregate and family gate uses raw, uncalibrated probabilities and fixed
argmax decisions. It requires all of:

- raw hierarchical balanced NLL `<= 0.03`;
- UNKNOWN/KNOWN balanced accuracy `>= 0.99`;
- FREE/OCCUPIED balanced accuracy `>= 0.99`;
- UNKNOWN, FREE, and OCCUPIED recall each `>= 0.98`;
- true-FREE recall in each gated distance bin `>= 0.95`;
- cross-scene wrong-view minus correct-RGB balanced NLL `>= 0.25`;
- same-scene wrong-view minus correct-RGB balanced NLL `>= 0.25`.

Both wrong-view permutations are deterministic and have zero image and
transition matches. The cross-scene control also has zero scene matches. The
same-scene control stays within scene while changing transition.

## Holdout Comparison

Holdouts are compared only at a stage that both arms pass. The common stage is
faithful when both pass faithful; otherwise it is ceiling when both pass
ceiling. One arm's faithful checkpoint is never compared with the other arm's
ceiling checkpoint. No common passing stage means the comparison is
inconclusive.

All adjudicating holdout metrics are equal-weight macros across the five
families, not pooled-cell estimates. On both holdouts patch7 must satisfy:

- macro hierarchical NLL ratio to patch14 `<= 0.80`;
- macro far-FREE recall delta `>= +0.10`;
- every macro class-recall delta `>= -0.01`;
- no individual family/class recall delta below `-0.01`.

In addition, patch7 must have strictly lower NLL and strictly higher far-FREE
recall in all 5/5 cross-scene families and at least 4/5 same-scene families.
Ties fail. The all-positive cross-scene rule is the one-sided exact paired sign
result `p = 1/32`; a descriptive bootstrap cannot override it.

## Per-Seed Decision

The capacity branch provisionally supports patch7 only when patch14 fails every
attempted stage and patch7 passes at least one stage. The holdout branch
provisionally supports patch7 only when both arms are expressive and every
matched-stage holdout rule passes.

A single-seed artifact always emits
`patch7_full_train_candidate_licensed=false`, including a favorable seed. Seed
`20260711` is required after provisional support, faithful/ceiling gate
disagreement, a registered near-fit result, or a registered near-holdout
result.

Registered near-fit margins are NLL `<=0.033`, both balanced accuracies
`>=0.989`, every class recall `>=0.978`, every gated distance recall `>=0.945`,
and both wrong-view NLL deltas `>=0.225`. Registered near-holdout margins are
macro NLL ratio `<=0.85`, macro far-FREE delta `>=0.075`, every macro class
delta `>=-0.015` on both holdouts, or at least 4/5 favorable cross-scene
families.

## Two-Seed License

The result-artifact finalizer
`scripts/finalize_go2_physical_micro_overfit.py` is the only licensing path.
It requires precommitted file SHA-256 values for both inputs; verifies them
before deserialization, after reading, and after validation; rejects smoke or
noncanonical execution settings; validates complete stages/support/access
provenance; recomputes each stored decision from its faithful/ceiling stages;
and requires exact agreement with that recomputation. It then requires
identical panel, experiment contract, and source hashes before applying the
pure decision rule for exactly seeds `20260710` and `20260711`, including:

- both per-seed decisions provisionally support patch7;
- the same provisional branch (`causal_fit` or `matched_holdout`);
- the same favorable classification;
- the same qualifying optimizer stage;
- the same encoded support mechanism;
- neither per-seed artifact attempted to self-license.

Only then may the aggregate emit
`patch7_full_train_candidate_licensed=true`. Any discordance is
`two_seed_inconclusive`; there is no one-of-two vote.

## Artifact Integrity

The immutable panel and result bind input hashes, source hashes, git state,
initial/final state hashes, model configuration, initialization differences,
support counts, raw metrics, controls, learning curves, terminal gate steps,
decision checks, and a reconciled train-role access ledger. The authoritative
runner binds this protocol and
`docs/lewm_go2_generalization_execution_contract_2026-07-09.md` in its source
hashes. No GPU run or authoritative panel/result generation is part of the
protocol-correction change itself.

The authoritative runner accepts only batch 4, 2,000 faithful updates, 3,000
ceiling updates, and evaluation interval 100. A bounded
`--non-authoritative-smoke` mode defaults to three updates per stage and
interval one, emits schema `lewm_go2_physical_micro_overfit_smoke_result_v1`,
sets `authoritative=false` and `promotion_eligible=false`, and is rejected by
the finalizer. Thus a runtime smoke cannot be mistaken for evidence.
