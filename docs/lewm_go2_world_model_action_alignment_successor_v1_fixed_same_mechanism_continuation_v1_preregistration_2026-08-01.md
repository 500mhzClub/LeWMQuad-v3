# Action-alignment successor V1 fixed same-mechanism continuation V1

Date: 2026-08-01

Status: **PREREGISTERED BOUNDED 200-UPDATE CONTINUATION; NOT EXECUTION AUTHORITY**

## Evidence that permits, but does not authorize, this continuation

The completed integrity-replacement comparison is frozen in
`docs/lewm_go2_world_model_action_alignment_successor_v1_integrity_replacement_v1_terminal_review_2026-08-01.json`.
Its terminal decision was
`MEANINGFUL_ALIGNMENT_IMPROVEMENT_INCOMPLETE`: the alignment arm beat its
concurrent baseline on the registered minimum-margin delta by
`0.0028507902419068927`, with shared-scene bootstrap q05
`0.0013970169908673067`. That crossed the registered relative-improvement
criterion and permits one separately preregistered continuation.

It did not show absolute repair. Only three of nine action-margin points and
three of nine q05 values were positive. The treatment's worst absolute margin
moved away from zero over the observed tail:

- u500: `-0.0061708136683418634`;
- u600: `-0.006452736979912198`;
- u700: `-0.00660276124845185`.

The concurrent baseline deteriorated faster, so baseline-relative delta grew
while the treatment's absolute worst-action result worsened. Treatment
persistence q05 was also negative (`-0.22601831547011703`) and worse than the
reproduced baseline (`-0.1829122861354923`). Balanced accuracy and prediction
rank improved, but those observations establish action identification rather
than untaken-action transition accuracy or planning usefulness.

This mixed result makes a long unconditional continuation or parameter search
unjustified. The purpose of this run is a short bounded budget-allocation gate:
test whether the absolute treatment direction reverses fast enough under the
already-running regime to match the pace implied by its remaining scheduled
horizon. This attempt never automatically authorizes more execution. Failure
closes this training branch. A pace-level, q05-positive absolute gain permits
only a new, separately preregistered and reviewed unchanged-mechanism block,
which implements the user's explicit instruction to pursue observed meaningful
per-run improvement. This prospective routing supersedes the predecessor
terminal review's recommendation to force closure after this block; it does
not alter any predecessor measurement. Neither outcome proves that every
possible action-conditional architecture or objective is incapable.
The prospective-only supersession is recorded explicitly, without changing
the historical review, in
`docs/lewm_go2_world_model_action_alignment_successor_v1_fixed_same_mechanism_continuation_v1_governance_correction_2026-08-01.json`.

## Exact continuation and hypothesis

Resume both completed u700 arms from their exact, SHA-256-bound terminal
snapshots, including each arm's own AdamW state. Reconstruct the same frozen
spatial substrate from the same predecessor. Continue absolute global updates
701 through 900 only. The model, frozen substrate, objective, coefficients,
margin, optimizer, clipping, masks, row-stable B32 candidate route, data,
runtime, and validation roles remain unchanged.

The primary hypothesis is not another concurrent-baseline contrast. It is:

> Over u701-u900, the alignment treatment's absolute shared-family/scene
> minimum action margin improves from its reproduced u700 value by at least
> `0.001298360001376009`, with paired bootstrap q05 greater than zero, while
> the frozen continuation-retention bundle passes. Persistence remains a
> routing measurement after action-alignment repair, not a retention gate.

The threshold is frozen before execution. The u700 residual is
`0.00660276124845185`. The sum of unchanged learning-rate fractions over
u701-u900 is `175.22190359794223`; the sum over all remaining scheduled updates
u701-u3000 is `891.0844401632202`; their ratio is
`0.19663894430234558`. Multiplying that share by the residual gives
`0.001298360001376009`. Operationally, this is the scheduled-horizon pace
needed to close the u700 deficit by u3000 if progress were proportional to
remaining learning-rate mass. It is an allocation gate, not a claim that
learning is linear in learning-rate mass. Recovering the exact
u500-to-u700 treatment loss, `0.0004319475801099867`, is recorded as a weaker
diagnostic and cannot keep the branch open.

## Bound inputs and restoration

The continuation uses the following terminal snapshots as paired
initialization inputs only:

- baseline u700: SHA-256
  `613693d06309f90b87a7ac3e836d6817eed8c1e473ed0063006eb88960bce770`,
  10,909,343 bytes;
- alignment u700: SHA-256
  `41435888521041aaa262db9a26eaa656d33a339998372ffd0b068d7c75679731`,
  10,909,343 bytes.

They may not be opened during source review. After a fresh attempt root is
atomically reserved, the authorized worker must read each exact file once,
verify its byte count and digest before deserialization, and use
`torch.load(..., map_location="cpu", weights_only=True)` exactly once. It must
validate the exact snapshot schema/status, arm/coefficient/update, authority,
reservation, substrate and schedule receipts; strict model keys, shapes,
dtypes and finiteness; exact optimizer groups, order, hyperparameters, moments
and step 700 for every trainable parameter. Baseline and alignment states may
not be crossed, reset, partially loaded, or supplemented.

The frozen spatial predecessor, V3 pack, train/validation indices, runtime,
and completed predecessor evidence remain separately bound. The predecessor
metric bundle is not a continuation input. The old attempt root remains
immutable.

Pre-authority SHA-256/stat reads of the metric bundle and/or the two snapshots
by the primary terminal reviewer, predecessor terminal-audit agent, and source
implementation agent are disclosed in
`docs/lewm_go2_world_model_action_alignment_successor_v1_fixed_same_mechanism_continuation_v1_preauthority_identity_read_disclosure_2026-08-01.json`.
No tensor was deserialized or semantically inspected and no artifact was
modified. Those actors are ineligible as the independent continuation source
reviewer. The worker's two-snapshot deserialization contract counts the one
authorized content read and deserialization of each snapshot inside the fresh
reserved attempt; historical identity-only reads remain separately bound
evidence.

Before update 701, the worker must replay no-gradient u700 validation from the
restored states and reproduce the frozen public u700 anchors, including both
arms' action margins, the treatment action-margin vector, balanced accuracy,
persistence, wrong-history, rank ratio, and concurrent delta. A mismatch is a
terminal pretraining integrity failure and consumes the sole attempt.
Bound u700 public scalar and vector anchors use `rel_tol=0, abs_tol=1e-12`;
snapshot identities, schemas, model state, optimizer state, schedule receipts,
and accounting remain exact. This tolerance is frozen and may not be tuned.

## Schedule and fixed budget

Construct the unchanged hash-ordered schedule through global update 900. Its
first 700 rows and prefix digest must equal the completed predecessor
schedule; train only `schedule[700:900]`. Learning-rate calls use absolute
updates 701 through 900 against the unchanged warmup 150 and cosine horizon
3000. Warmup, optimizer moments, and optimizer step counts may not reset.

The fixed continuation is:

- 16,000 existing training rows and 2,048 existing scene-disjoint validation
  rows;
- 200 additional updates per arm, batch 256, microbatch 32;
- 51,200 additional scheduled presentations per arm;
- ten head row-presentations per arm per scheduled row;
- read-only panels at reproduced u700, diagnostic u800, and terminal u900;
- full controls at the u700 replay and u900, and full-train fit only at u900;
- no early stopping, checkpoint selection, validation gradient, schedule
  extension, or mid-run launch decision;
- maximum 9,000 wall seconds and 7,200 GPU seconds; and
- zero RGB opens, data generation, network access, sealed access, held-out
  access, or protected-runtime access.

## Frozen measurements

At u700 replay and u900, compute the same nine-way action localization,
action-margin points/q05/shared-minimum q05, balanced accuracy, persistence,
wrong-history, effective-rank ratio, factual fit, contract checks, and
concurrent baseline-relative delta used by the completed comparison.
For rank at u700/u800/u900, the metric bundle stores the exact float64 centered
192-by-192 covariance sufficient statistics for the target and both arms. The
checker independently performs the eigendecomposition and entropy-rank ratio
calculation, checks it against the worker receipts, and uses only its
recomputed treatment ratios in terminal routing.

The primary continuation statistic is a shared-family/scene paired Bayesian
bootstrap of
`minimum_action_margin_alignment_u900 - minimum_action_margin_alignment_u700`.
It uses 10,000 positive exponential-weight draws, seed `20260812`, and sorted
indices 500, 5000, and 9499 for q05, q50, and q95. The exact same weights apply
to both updates. The concurrent baseline-relative delta remains diagnostic
only and cannot override a non-improving absolute treatment result.
Its q05 describes validation-scene reweighting conditional on this adaptively
selected continuation; it does not quantify training-seed uncertainty or
fresh-scene generalization.

The frozen u700 treatment descriptive anchors are:

- balanced-accuracy q05: `0.34701964075333114`;
- rank ratio: `0.47287848726118314`;
- persistence q05: `-0.22601831547011703`;
- wrong-history q05: `0.1406183675693852`;
- worst action-margin point: `-0.00660276124845185`;
- shared-minimum margin q05: `-0.0078111838906331724`.

They are reported again as u900-minus-u700 changes but are not brittle
point-estimate noninferiority gates. Continuation retention instead freezes
the registered floors: balanced-accuracy q05 above 1/9, wrong-history q05
above zero, rank ratio at least 0.25 at two of u700/u800/u900, preservation of
the u700-positive action-margin point and q05 IDs 0, 4, and 7, and all contract
and train-fit checks. Persistence remains a post-alignment routing measurement,
not part of this retention bundle.

## Terminal precedence

The checker independently recomputes all metrics and applies this order:

1. Any binding, custody, schedule, restoration, runtime, accounting,
   contract, finiteness, train-fit, or frozen continuation-retention failure
   closes the branch without a positive scientific conclusion.
2. If every one of the nine u900 treatment action-margin points and q05 values
   is positive, the shared-minimum q05 is positive, and retention passes,
   preserve the frozen localization routing. Five or more nonpositive
   per-action persistence q05 values gives
   `PASS_ACTION_ALIGNMENT_PROXY_REPAIR_PERSISTENCE_SYSTEMIC` and permits only
   separate preregistration of `PERSISTENCE_RESIDUAL_VS_MATCHED_BASELINE`.
   One to four such failures, or nonpositive aggregate persistence q05, gives
   `PASS_ACTION_ALIGNMENT_PROXY_REPAIR_PLANNING_WITH_PROXY_CAVEAT` and makes
   the u900 snapshot eligible for a separately preregistered planning gate
   with that caveat. No per-action persistence failures and positive aggregate
   persistence q05 gives
   `PASS_EXPLORATORY_ACTION_ALIGNMENT_AND_PREDICTOR_USEFULNESS_PROXY` and
   eligibility for a separately preregistered planning gate. None authorizes
   the next execution automatically.
3. If action alignment remains unrepaired, an absolute u900-minus-u700
   treatment gain of at least `0.001298360001376009` with paired q05 greater
   than zero and continuation retention intact is recorded as
   `MEANINGFUL_ABSOLUTE_PROGRESS_INCOMPLETE_CONTINUE_SAME_MECHANISM`. It
   permits only separate preregistration and review of the next fixed,
   unchanged-mechanism block; it grants no automatic execution authority.
4. A positive gain below that scheduled-horizon pace with q05 greater than
   zero is `POSITIVE_BUT_INSUFFICIENT_RATE_CLOSE_ALIGNMENT_BRANCH`.
5. A positive point with q05 at or below zero and q95 above zero is
   `INCONCLUSIVE_ABSOLUTE_CHANGE_CLOSE_ALIGNMENT_BRANCH`.
6. A nonpositive point or q95 at or below zero is
   `STALLED_OR_HARMFUL_CLOSE_ALIGNMENT_BRANCH`. Baseline-relative improvement
   alone is insufficient for every branch above.

No result automatically authorizes another alignment continuation,
coefficient change, optimizer reset, architecture tweak, identical
replication, integrity replacement, or follow-on. There is no
validation-driven extension inside this attempt past u900. The sole
alignment-unrepaired route that keeps training scientifically eligible is the
pace-level, q05-positive meaningful-progress outcome above, and it selects
`PREREGISTER_NEXT_FIXED_SAME_MECHANISM_BLOCK`. Every other unrepaired outcome
selects `NO_FURTHER_ALIGNMENT_TRAINING_OR_PLANNING_GATE`; only an
action-alignment repair reaches the frozen persistence/planning routing above.

## Lifecycle and claim boundary

The new identity is
`world_model_action_alignment_successor_v1_fixed_same_mechanism_continuation_v1/attempt_v1`
under `.generated/dev/`. Its campaign root must be absent and nonsymlinked.
Atomic reservation consumes the only V1 attempt. Retry, resume of this new
attempt, refill, overwrite, recovery, replacement, and an unregistered
extension of this attempt are false. Calling the operation a continuation
describes its scientific initialization; it is not runtime resume authority
for the consumed prior attempt. Only the registered meaningful-progress route
may motivate a separately named, separately preregistered successor block.

Source, tests, this preregistration and its machine-readable plan must be
committed first. A different-agent source-only review must be committed next.
One exact authority may then be committed separately. No authority is granted
by this document.

Even a proxy pass is optimized, development-only evidence on the existing
on-policy pool. It does not establish untaken-action causality, calibrated
transition probabilities, task regret, planning utility, navigation,
promotion, deployment, or production readiness.
