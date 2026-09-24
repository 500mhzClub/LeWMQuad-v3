# LeWM Go2 dense V-JEPA 2.1 horizon diagnostic V1 preregistration

Date: 2026-08-03

## Purpose and claim boundary

This is one bounded, development-only, no-new-data diagnostic.  It asks only
whether the fixed dense spatial-token action-conditioned predictor was still
undertrained at 800 updates.  It does not revisit the completed four-arm
screen, authorize fresh rendering, inspect evaluation RGB, estimate
scene-disjoint generalization, or establish navigation usefulness.

The completed predecessor remains terminal
`STOP_BEFORE_FRESH_MATCHED_BRANCH_COLLECTION`.  All four predecessor arms
failed both capacity gates.  The dense V-JEPA arm alone showed enough late
progress from update 400 to 800 to leave its training-set capacity unresolved:

- error-to-persistence ratio improved from `1.0326830669810942` to
  `0.9164363539053353`;
- branch retrieval improved from `0.18229166666666666` to
  `0.2803819444444444` (`210` to `323` correct of `1,152`); and
- action-intervention margin improved from `0.01126912691526942` to
  `0.028666746492187187`.

The other three arms are excluded because even optimistic repeat-per-doubling
projections leave their 3,200-update retrieval near `0.18` to `0.27`, far
below `0.50`.  Repeating them would not plausibly change the predecessor's
collection conjunction.

## Frozen predecessor inputs

The execution authority must bind all of the following exact files:

- predecessor result: 21,377 bytes, SHA-256
  `a6caf2ed1950781815925ccc76b4dbbf40b0f331f4b14a5e60befc88f3aae605`;
- predecessor terminal: 510 bytes, SHA-256
  `bf3bf322c2f3db877be405ebf5ca1daf9dd1a5ffd667b769d44cccab22ede758`;
- independent predecessor terminal review: 4,991 bytes, SHA-256
  `c450baab14b50caed3469fa88f5812c92c02b04676059568e8dae3dc2e5bad83`;
- V-JEPA feature receipt: 1,822 bytes, SHA-256
  `5d4f8a82d10a33c21b41f1543d6f56b3a230a38f67b02d3f8e7330a8d30180f5`;
  and
- V-JEPA feature cache: 604,097,648 bytes, SHA-256
  `3549855ea857906dfe3a4b55fc817633b5114b2457f8facaa4fa87f9eddd798b`.

The runner must reconstruct the train-role index through the frozen
metadata-only posthoc loader and verify the predecessor index SHA-256
`b740e3efead2f79fd17337a9fa10784c91989e52e837d023b2cc02a2c19d018d`.
It must then verify the receipt, cache payload, artifact order, preprocessing
contract, tensor shape `[1536,256,768]`, float16 storage, finite values, and
per-token unit norms before training.

No PNG or other RGB leaf may be opened.  Both the metadata loader access audit
and the successor result must report zero RGB leaf opens.  Evaluation groups
may be used only to prove train/evaluation artifact-ID disjointness inside the
already reviewed metadata adapter; no evaluation target, label, or RGB value
may enter the predictor or any metric.

## Fixed model and optimization

Run only `dense_vjepa2_1`, using the exact frozen model implementation from
the predecessor source closure:

- three V-JEPA context grids, two historical requested action IDs, and one
  candidate requested action ID as input;
- two dense residual action-conditioned blocks, hidden width 128;
- the candidate successor V-JEPA grid as target;
- seed `2026080301`;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- batch size eight states with all nine candidate actions;
- global gradient-norm clipping at `1.0`;
- matched per-token cosine loss plus coefficient `0.25` nine-way within-state
  cross-entropy at temperature `0.1`; and
- a maximum of exactly 3,200 updates.

Initialization, state permutation generation, batch order, model operations,
loss, metrics, and threshold comparisons must be identical to the predecessor.
This is a horizon change only.  Training starts from the same seeded random
initialization because the predecessor checkpoint deliberately did not retain
optimizer state; it is not a resume.

The runner records traces at updates 0, 800, 1,600 and, only if allowed past
the futility check, 2,400 and 3,200.  It saves the update-1,600 terminal or
continuation checkpoint and, if reached, the update-3,200 checkpoint.  All
reported metrics use deterministic full-train-set evaluation.

## Drift witness at update 800

Before the result can be interpreted, the update-800 metrics must reproduce
the predecessor dense V-JEPA trace exactly under the bound environment:

- matched cosine error `0.06880560082693894`;
- persistence cosine error `0.07507951919817263`;
- error-to-persistence ratio `0.9164363539053353`;
- branch retrieval accuracy `0.2803819444444444`;
- cyclic deranged cosine error `0.09747234731912613`; and
- action-intervention margin `0.028666746492187187`.

Any mismatch consumes the attempt as a terminal infrastructure failure and
does not permit threshold interpretation or a retry under this version.

## Fixed update-1,600 futility rule

At update 1,600, continue to 3,200 only if all of the following hold:

- error-to-persistence ratio is at most `0.8582181769526677`, the midpoint
  from the predecessor terminal value to the unchanged `0.80` gate;
- branch retrieval is at least `0.3901909722222222`, the midpoint from the
  predecessor terminal value to the unchanged `0.50` gate;
- action-intervention margin is strictly positive;
- no loss, gradient, state, checkpoint, or metric is nonfinite; and
- deterministic repeated evaluation is exact.

Failure of any condition ends the run at update 1,600 with
`COMPLETE_FUTILITY_STOP`.  It forbids a 3,200- or 6,400-update continuation,
width change, loss change, alternate seed, retry, or fresh data collection
under this mechanism version.

## Fixed update-3,200 capacity gates

If the futility rule passes, training continues unchanged to update 3,200.
Training-set action-binding capacity passes only if:

- error-to-persistence ratio is at most `0.80`;
- branch retrieval is at least `0.50`;
- action-intervention margin is strictly positive;
- no nonfinite event occurred; and
- deterministic repeated evaluation is exact.

Failure produces `COMPLETE_CAPACITY_NOT_ESTABLISHED` and the same terminal stop
on further horizon or mechanism tuning.  Passing produces
`COMPLETE_TRAINING_SET_CAPACITY_ESTABLISHED`.

## Interpretation and next route

Neither possible terminal authorizes collection or evaluation RGB access.

A failure means the fixed dense V-JEPA mechanism did not establish even
training-set branch capacity under the bounded horizon.  The next scientific
route is a materially different representation mechanism, not another
optimization tweak.

A pass means only that the fixed dense V-JEPA mechanism can bind requested
actions to matched successors on the existing 16-scene training set.  It does
not reverse the predecessor four-arm stop, rescue the failed controls,
establish generalization, show correct physical action ranking, or prove
navigation.  Any later data collection or comparison requires a new explicit
preregistration and authority that addresses the failed control conjunction.

## Custody and execution authority

Before execution, freeze the new runner and focused tests, obtain an
independent source review with no open findings, commit the reviewed source,
and issue one exact caller-bound authority.  The authority grants one fresh
output root only.  It grants no dataset generation, RGB access, evaluation,
held-out, sealed, navigation, rollout, deployment, retry, resume, or
replacement-attempt authority.
