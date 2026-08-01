# Existing-pool action-alignment successor V1 preregistration

Date: 2026-08-01

Status: **source and scientific contract; not execution authority**.

## Question and frozen route

The completed aggregate-only V3 localization selected
`TEST_GLOBAL_ALIGNMENT_HYPOTHESIS`: all nine requested actions had negative
family/scene-equal factual-versus-best-wrong points and q05 values.  The
registered next comparison is therefore one global action-margin objective
against a concurrent factual-loss baseline.  Action reweighting, persistence,
architecture, data, schedule, coefficient search, and continuation are not
part of this attempt.

This is a development experiment.  It cannot establish requested/executed
equivalence, untaken-action causality, planning value, navigation value, or a
promotion/deployment claim.

## Exact design

Both arms are constructed fresh from the same frozen spatial predecessor and
have bit-identical trainable initialization.  They use the exact V3:

- 16,000 corrected H6 training rows and 2,048 scene-disjoint validation rows;
- 700 updates, batch 256, microbatch 32, seed `20260731`;
- 179,200-presentation hash-ordered schedule, masks, learning-rate schedule,
  optimizer, frozen encoder and frozen target encoder; and
- already-rendered immutable V3 train/validation pack.  No RGB is opened and
  no data is generated.

For each row let `a` be requested candidate action `actions[2]`, `E_a` its
normalized half-squared target-token energy, and `min_wrong E` the minimum over
the other eight requested actions, with lowest action ID resolving exact ties.
Both arms execute the same nine-way search and selected-wrong gradient route.
Their sole scientific difference is coefficient `c`:

`L = mean(E_a) + c * mean(relu(0.01 + E_a - min_wrong E))`.

- concurrent baseline: `c = 0.0`;
- alignment treatment: `c = 1.0`.

The memory-safe implementation scans every wrong row/action pair without
gradients in fixed chunks of at most 128 rows, selects the exact best wrong
action, then recomputes factual and selected-wrong energies with gradients. A
synthetic test must show value and parameter
gradient parity with a full differentiable nine-way computation when the
minimum is unique.  Both arms use the same path, query count, row order and
optimizer timing.

Validation is read-only at updates 500, 600 and 700.  There is no checkpoint
selection, early stopping, retry, resume, refill, coefficient change, or
mid-run decision.  Only terminal update 700 determines the outcome.  A
full-train factual panel at update 700 supplies finite train-fit accounting;
it makes no generalization claim.

## Inputs and custody

The authority must bind the exact V3 pack manifest and its six artifacts,
the exact predecessor checkpoint, train and validation indices, the V3 result
and terminal review, and the completed localization result and supervision
receipt.  The pack root is
`.generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1/pack`.

Pack payload reads, the predecessor checkpoint read, and H6 train/validation
metadata reads are authorized only inside the one reserved attempt.  Sealed,
held-out, protected, alternate checkpoint, alternate pack, RGB, network, and
3 TB corpus access remain forbidden.  Reusing the pack avoids all 72,192 V3
RGB leaf opens and all rendering cost.

The fresh attempt is
`.generated/dev/world_model_action_alignment_successor_v1/attempt_v1`.
Reservation consumes the only attempt. The worker fails closed unless exactly
one registered ROCm device is visible; the supervisor performs an independent
idle-device preflight and enforces a 9,000-second wall cap. The worker enforces
the 7,200-second GPU cap after every update, after each terminal panel, and at
terminal synchronization, so overshoot is bounded by at most one in-flight
update or evaluation panel rather than claimed as continuous supervisor timing.
Exactly 700 updates are authorized. Failure is terminal.

## Frozen metrics

At update 700, both arms are localized with the exact V3 validation metadata,
row→scene→family aggregation, 10,000 strictly-positive scene-cluster Bayesian
draws, and existing seeds for action margin, persistence, wrong history and
balanced action accuracy.

For arm `j`, define `M_{j,a}` as its family/scene-equal action-margin point and
`A_j = min_a M_{j,a}`.  The paired statistic is

`Delta_A = A_alignment - A_baseline`.

For every paired draw, seed `20260811` generates one shared family/scene weight
table used by both arms and all actions.  Sort 10,000 deltas and report indices
500, 5000 and 9499 as q05/q50/q95.

V3's point deficit is `dA = 0.009453551490358742`:

- meaningful point threshold: `0.25*dA = 0.0023633878725896856`;
- stall upper threshold: `0.10*dA = 0.0009453551490358743`.

Exploratory absolute alignment repair takes precedence when treatment has all
nine action points, all nine marginal q05s and the shared-minimum q05 strictly
positive, and all provisional retention checks pass.

Provisional retention requires:

- treatment balanced-accuracy q05 strictly above `1/9`;
- treatment wrong-history q05 strictly positive;
- treatment prediction/target effective-rank ratio at least `0.25` at two of
  updates 500/600/700;
- no treatment action point becomes nonpositive where the concurrent baseline
  point is positive; and
- every frozen contract and finite train-fit check passes.

Decision precedence is exact:

1. repaired plus retention → `PASS_EXPLORATORY_ACTION_ALIGNMENT_PROXY_REPAIR`;
2. any retention failure → `FAIL_RETENTION_CLOSE_ALIGNMENT_BRANCH`;
3. otherwise point `Delta_A >= 0.0023633878725896856` and paired q05 > 0 →
   `MEANINGFUL_ALIGNMENT_IMPROVEMENT_INCOMPLETE`;
4. otherwise paired q95 < `0.0009453551490358743` →
   `STALLED_CLOSE_ALIGNMENT_BRANCH`;
5. otherwise → `INCONCLUSIVE_ALIGNMENT_COMPARISON`.

A stall or retention failure ends alignment tweaking.  Inconclusive permits at
most one separately preregistered identical replication.  Meaningful but
incomplete permits at most one separately preregistered fixed same-mechanism
continuation.  Those options are mutually exclusive.  This result authorizes
neither automatically.

## Interpretation and next boundary

Because the treatment directly optimizes the identification proxy, even an
absolute repair is exploratory and cannot replace a fresh treatment/blind/
shuffled confirmation on a previously unconsulted role.  If alignment repairs,
the already-frozen persistence routing is applied next: systemic persistence
failure selects one separately preregistered residual comparison; localized or
aggregate uncertainty goes directly to a planning-usefulness gate with a proxy
caveat; persistence passage goes to that gate without the caveat.

Planning usefulness still requires physically executed or branched candidate
outcomes and independent utility/regret: top-action regret, pairwise ranking,
unsafe-action rate and calibration, followed for WM-D by a paired planner
on/off or score-permutation intervention on identical scenes and budgets.  No
latent-energy threshold is relabelled as planning evidence.
