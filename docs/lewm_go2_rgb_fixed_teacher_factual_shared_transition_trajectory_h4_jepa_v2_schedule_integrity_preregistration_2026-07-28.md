# Go2 RGB fixed-teacher factual shared-transition trajectory-H4 JEPA V2 schedule-integrity preregistration — 2026-07-28

## Decision and scientific question

- Preregister exactly one fresh, bounded **schedule-integrity replacement** of
  factual shared-transition trajectory-H4 JEPA V1. This document authorizes
  source-only adapter, runner, checker, and synthetic-test preparation. It does
  not by itself authorize index generation, RGB access, checkpoint access,
  training, GPU use, navigation, or held-out evaluation.
- V1 remains a terminal STOP on its exact schedule. None of its metric values,
  receipts, or custody findings is changed. Its action edges were nevertheless
  invalid for candidate planning: an edge labeled `p_i` began after `p_i` had
  already acted for one tick and ended after unseen `p_(i+1)` had acted for one
  tick.
- The question is narrow: does the **unchanged V1 model and science** pass when
  each requested action is instead paired with its reset-safe pre-command and
  post-block boundary states?
- This is the sole V2 integrity replacement. It is not a second V1 seed, a
  model revision, a loss repair, or authority to inspect or reuse V1 runtime
  artifacts. The proposed factorized conditional-increment JEPA is deferred;
  it may be reconsidered only after a clean V2 STOP on the unchanged gates.

## Frozen V1 scientific identity

- Normative V1 preregistration:
  `docs/lewm_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1_preregistration_2026-07-28.md`, commit
  `5c038f054f17d7d8928518723b12e1166db2d17a`.
- Frozen V1 implementation commit:
  `065bae4069d53a4d2c87f781df5ae9e29d5027a2`.
- The model source must remain byte-identical:
  `lewm/models/go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py`,
  21,734 bytes, SHA-256
  `38e264f8e18ffa3c3da4775fdd7d4a38549e8544f99cd863bfd2534999cd5b36`.
- The V1 runner is the normative science implementation:
  `scripts/run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py`,
  34,730 bytes, SHA-256
  `693cbea45b2a49f0f3edfb7cabce347b852a67af78df1ecf5462c65be48cd977`.
  A V2 wrapper may adapt only the reviewed index schema, frozen schedule
  hashes, output/receipt identity, and corresponding messages. Model
  construction, initialization order, forward path, objective, optimizer,
  observation, selection, and gate code must execute the V1 implementation
  unchanged or be proven source-identical by an independent semantic-diff
  checker.
- The existing V1 index source also remains untouched:
  `lewm/datasets/go2_recurrent_h4_rgb_sequences.py`, 27,386 bytes, SHA-256
  `3f8c2a89af2934e8225dd98447b952d9e5ce8bedac99a7f834118263957652e6`.
  V2 must be a small adapter around that reviewed source, not a copied rewrite.
- The only initialization input remains the accepted N320 checkpoint, file
  SHA-256
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`,
  content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`,
  13,777,100 bytes. A later, separately execution-authorized bound runner may
  open it once and copy only the reviewed `encoder.*` prefix into a fresh
  online encoder and a permanently fixed target encoder. No V1 predictor,
  checkpoint, tensor, trace, optimizer state, or runtime output may be listed,
  statted, hashed, opened, or reused.

## Correct requested-command boundary contract

- Let `F(i,j)` be the `j`th post-request frame of primitive block `i`, where
  `j=1:5`, `command_dt_s=0.1`, and the block duration is exactly 0.5 seconds.
  The recorded primitive is the requested command available to a future
  navigation policy. Executed/clipped commands, controller internals, pose,
  depth, flow, or other privileged state are not model inputs or targets.
- One edge labeled by requested primitive `p_i` is exactly:

  ```text
  F(i-1,5) --p_i--> F(i,5)
  ```

  `F(i-1,5)` is the real same-episode row immediately before `p_i` begins;
  `F(i,5)` is the fifth and final post-request row of `p_i`. No row or effect
  from destination primitive `p_(i+1)` belongs to this edge.
- One H6 row contains six consecutive complete blocks and seven shared
  boundaries:

  ```text
  e0=F(i-1,5), p0=p_i,   e1=F(i,5),
                 p1=p_i+1,e2=F(i+1,5),
                 p2=p_i+2,e3=F(i+2,5),
                 p3=p_i+3,e4=F(i+3,5),
                 p4=p_i+4,e5=F(i+4,5),
                 p5=p_i+5,e6=F(i+5,5).
  ```

  The notation `p_i+n` means the requested primitive of the nth consecutive
  block, not arithmetic on the nine-action vocabulary.
- All seven endpoints must have identical role, family, scene, source
  manifest, environment index, episode ID, and reset count. Each adjacent
  endpoint pair must differ by exactly five episode steps and five same-env
  ticks; with 48 interleaved environments its frame-index difference is
  exactly 240. Timestamp duration must be 0.5 seconds within the inherited
  `2e-4`-second tolerance.
- For every edge, all five post-source metadata rows, including the destination
  boundary, must be consecutive, strictly time ordered, and carry one stable
  command sequence ID, primitive, `block_size=5`, and `command_dt_s=0.1`. The
  destination boundary must carry the indexed primitive and be the fifth row
  of that block. The first row of the next block must never be used as its
  endpoint.
- A block without a real `F(i-1,5)` in the same episode/reset is ineligible.
  Missing boundaries may not be inferred, timestamp-shifted, duplicated,
  padded, or synthesized. Any step gap, reset, episode/context change,
  irregular tick, incomplete block, or destination-action contamination is a
  fail-closed rejection.

## Deterministic V2 index adapter

- New row schema:
  `lewm_go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity`.
  Serialized model fields remain exactly seven ordered RGB leaf names and six
  ordered action indices, plus the inherited public role/family/scene identity.
  No label, pose, motion, timing, reset, or command-array payload is serialized
  for the model.
- Preserve the exact V1 action vocabulary and ordering, source roles, scene
  split, preprocessing, packed transition-disjoint H6 policy, per-scene
  candidate cap 64, train quota 2,000 rows per family, validation quota 256
  rows per family, family interleave, train cap, and train/validation scene and
  manifest disjointness. Consecutive packed H6 groups may share their single
  boundary RGB while never sharing an action transition; this must be reported
  truthfully rather than described as RGB-row disjointness.
- Preserve seed string
  `go2_recurrent_h4_rgb_sequence_index_v1_20260727` and V1 rank values for each
  surviving logical six-block candidate. The adapter must enumerate the full
  deterministic candidate order before applying the valid-candidate cap,
  reject candidates without a real predecessor boundary without reranking the
  survivors, and continue the same per-scene round-robin/family interleave to
  the next ranked valid candidates until the exact quotas are restored.
- The adapter may not change primitive order, mine actions, rebalance by
  outcome, decode or select by RGB content, or use any model result. The
  inherited exact-leaf regular-file and eight-byte PNG-signature validation is
  permitted. Backfill is solely the deterministic consequence of reset-safety
  filtering.
- The V1 train and validation indexes remain immutable historical evidence:
  train SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`
  and validation SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`.
  They are not V2 runtime inputs.
- V2 train and validation index file SHA-256, content SHA-256, byte counts,
  exact row counts, scene counts, family counts, action-position coverage,
  rejected-predecessor counts, and source bindings must be generated once by
  the frozen reviewed adapter and committed before execution authority. No
  placeholder or locally regenerated hash may be accepted by the runner.

## Unchanged model and factual JEPA objective

- Inputs remain `e0:e2`, past actions `p0:p1`, and proposed future actions
  `p2:p5`. Fixed-teacher future targets remain `e3:e6`. Future RGB is visible
  only to the fixed target encoder under `no_grad` after all predictions have
  been formed.
- Retain exactly four coherent equal-mass particles, the fresh hidden-state
  initializer, one one-block spatial Transformer transition object shared over
  all six edges, one shared zero-initialized residual output head, and the V1
  factual-carrier insertion/open-loop rules. There is no model parameter,
  width, depth, bias, embedding, horizon, mode, recurrence, or initialization
  change.
- Retain the exact V1 proper equal-mass energy score: 50% coherent joint score
  plus 50% mean marginal-step score, including all 16 ordered atom pairs and
  their zero diagonal in the pair term.
- Retain the exact loss and weights:

  ```text
  L_prediction = 0.5 * ES_K4(all-six-edge local innovations)
               + 0.5 * ES_K4(cumulative open-loop H4 states)
  L_total = L_prediction
          + 1.0 * L_online_e0:e2_to_fixed_teacher_alignment
  ```

- Cyclic wrong action, all hold, reordered history, reset history,
  persistence, centroid, and particle-spread branches remain evaluation-only.
  No action/history margin, inverse classifier, candidate-action NLL,
  whitening, learned target, reconstruction, semantic, or navigation loss may
  be introduced.

## Unchanged optimizer, observations, and cap

- Preserve seed `20260727`, float32, no autocast, disabled cuDNN benchmarking,
  batch size 16, and the exact V1 parameter construction and RNG draw order.
- Preserve AdamW with encoder LR `1e-4`; every fresh transition, hidden,
  action, mode, spatial, and output parameter LR `3e-4`; weight decay `1e-4`;
  betas `(0.9,0.999)`; epsilon `1e-8`; and separate gradient-norm clipping at
  `1.0` for each inherited parameter group.
- Execute exactly 1,000 optimizer updates and exactly 16,000 ordered training
  presentations. Observe the complete fixed 2,048-row V2 validation schedule
  at updates `0,250,500,750,1000`, for 10,240 validation presentations. Active
  GPU time remains capped at 5,400 seconds.
- There is one fresh attempt only. Reservation, clean operational failure,
  scientific failure, or completion consumes it. There is no retry, resume,
  replacement of the replacement, second seed, longer run, threshold change,
  schedule regeneration, or observer rerun.

## Required source-only implementation and tests

- Preserve the three V1 file hashes above. Add a minimal V2 endpoint adapter
  and a thin V2 runner/checker; do not copy the model or scientific runner
  boilerplate into a new independent implementation.
- Synthetic tests must prove all of the following without opening a real
  index, RGB, label, checkpoint, test, held-out, or sealed input:
  - each edge uses `F(i-1,5) -> F(i,5)` and contains exactly five ticks of only
    the indexed requested primitive;
  - changing the destination primitive cannot change either endpoint of the
    preceding edge, and no destination first tick contaminates that edge;
  - all six durations are 0.5 seconds within tolerance, episode-step deltas
    are five, frame-index deltas are 240, and endpoints are shared exactly;
  - source, role, family, scene, environment, episode, reset, manifest,
    sequence, primitive, block-size, and timestep contexts are checked;
  - episode-start candidates lacking a real predecessor are rejected, reset or
    gap crossings fail closed, survivor order is unchanged, deterministic
    backfill restores quotas, and repeated runs are byte-identical;
  - V1 source hashes remain exact, and importing V2 source has no index
    generation, training, checkpoint, or other runtime-input side effect;
  - the V2 runner accepts only the frozen V2 schema/hashes and cannot accept a
    V1 row, a locally substituted index, a retry/resume input, or a prior
    predictor checkpoint;
  - the semantic diff from V1 is limited to index schema/hashes,
    output/receipt identity, and their validation messages; model inventory,
    initialization, losses, optimizer, observations, selection, thresholds,
    and decision gates are identical.
- Independent science and custody review must bind every new source/test path,
  the exact V1 witnesses, the generated V2 index hashes, and the semantic-diff
  report before any execution authorization.

## Selection and unchanged all-conjunctive PASS

- Update zero is informational and must reproduce exact persistence within
  `1e-5`, with zero action, hold, persistence, history, distribution-value,
  and spread gaps within `1e-5`, plus zero fixed-teacher state/rank drift.
- Among trained observations, select the eligible noncollapsed checkpoint with
  minimum validation cumulative combined normalized energy score. No tier,
  gate, family, or post-hoc selection is permitted.
- PASS requires the complete V1 conjunction, unchanged:
  - exact completion, finite observations, byte-identical fixed teacher with
    zero gradient/EMA updates, target and online effective-rank ratios at least
    `0.10`, and near-zero-variance fractions at most `0.05`;
  - cumulative combined and joint normalized scores each below persistence,
    marginal H1--H3 scores below persistence, and H4 score at most `0.90`;
  - `p0/p1` factual local-prior score below persistence, positive scene-
    bootstrap lower bound, positive gap in at least six families, and no family
    below `-0.02`;
  - positive H4 persistence bootstrap lower bound, positive H4 persistence gap
    in at least six families, and no family below `-0.02`;
  - combined distribution value at least `0.05`, positive bootstrap lower
    bound, positive in at least six families, and H4 normalized pairwise spread
    at least `0.05`;
  - validation-only cyclic H4 wrong-action gap at least `0.05`, positive
    bootstrap lower bound, nonnegative H1--H3 wrong-action gaps, positive H4
    gap in at least six families, and no family below `-0.02`;
  - H4 ordered-history gap at least `0.03`, positive bootstrap lower bound, and
    positive in at least six families;
  - positive H4 all-hold gap, positive in at least six families, and no family
    below `-0.02`.
- Control support counts must be recorded from the frozen V2 schedules before
  execution. They may not be used to change a control, threshold, breadth
  requirement, family floor, or selection rule.

## Custody, fresh identity, and terminal consequence

- Exact V2 output root:
  `.generated/go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity/probe_v1`.
  It must be absent before reservation and mode `0700` afterward.
- Receipt/schema prefix:
  `lewm_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity`.
  Reservation, metrics, artifact, access, result, completion, and independent
  terminal-audit documents must bind the frozen V2 source, schedule hashes,
  accepted N320 identity, counters, and one-attempt namespace.
- Runtime checkpoints and traces are write-only during the run and inaccessible
  afterward unless a later, separate authority explicitly permits a qualified
  checkpoint. No V1 or other stopped runtime root is an input.
- Only allowlisted development train/validation RGB leaves named by the frozen
  V2 indexes may be decoded by an authorized runner. Test, held-out, sealed,
  legacy V4 sealed material, labels, raw messages, navigation, G2--G8,
  production, promotion, and deployment remain unauthorized.
- PASS establishes only bounded development evidence that the unchanged
  factual shared-transition JEPA works on candidate-valid requested-action
  boundaries. It authorizes neither checkpoint opening nor downstream use.
- STOP closes this exact V2 integrity replacement with no retry. If prediction
  remains useful but action/history value still fails, the factorized
  conditional-increment H4 JEPA becomes the ordered, separately reviewed
  fallback. If generic prediction also fails, record that localization before
  choosing a materially different target/state category. In neither case may
  V2 be answered with another schedule version, scalar tweak, seed, extension,
  checkpoint reuse, or data-scale run.
