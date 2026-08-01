# Existing-pool three-arm V3 action-localization V1 preregistration

Date: 2026-08-01

Status: **source and read-only diagnostic contract; not execution authority**.

This document responds to the user's instruction to continue only while runs
produce meaningful improvement, to stop when progress stalls, and ultimately
to establish either the registered latent thresholds or useful planning
evidence. It authorizes no runtime input opening, artifact creation, training,
GPU work, data generation, retry, resume, promotion, held-out access, or sealed
access. A later exact authority may grant one CPU-only read-only diagnostic
after source freeze and independent review.

## 1. Purpose and predecessor

The completed integrity-replacement V3 experiment is consumed and immutable.
Its registered decision is `LOCALIZE_ACTION_ALIGNMENT_FAILURE`. Aggregate
training fit, scene-disjoint cross-arm generalization, balanced action
identification, wrong-history control, and rank passed. The hardest-action
margin failed; the later persistence gate was not reached by precedence and
would also fail.

The public u700 anchors are:

| Metric | Point | One-sided lower |
|---|---:|---:|
| Balanced action accuracy | `0.2469343816883539` | `0.23014452836846072` |
| Hardest wrong-action margin | `−0.009453551490358742` | `−0.01138311990101325` |
| Persistence log-energy advantage | `−0.14645548512800682` | `−0.1829122861354923` |
| Wrong-history log-energy advantage | `+0.12255093276460897` | `+0.11766703087321294` |

From update 500 to 700, balanced accuracy improved by approximately `0.0103171`
and persistence advantage improved by approximately `0.0472251`, while the
hardest-action margin worsened by approximately `0.00119195`. The hardest
margin has worsened continuously since its update-200 best. Consequently,
blind continuation of V3 is neither authorized nor the first-principles next
intervention.

## 2. Exact computational inputs

The diagnostic may consume exactly two runtime payloads:

1. Conditioned update-700 inert snapshot:
   `.generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1/arms/conditioned/snapshots/update_000700.pt`,
   SHA-256
   `df961a98ad148d6ba14bcdb03ddf13f3ec6edf73350ca60e1806af04281abe09`,
   212,616,145 bytes.
2. Frozen validation metadata:
   `.generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/val.jsonl`,
   SHA-256
   `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6`,
   1,317,888 bytes and 2,048 rows.

The snapshot contains the ordered validation indices, factual, persistence,
wrong-history and complete nine-way candidate-energy vectors. It also contains
model/optimizer state and tokens because it is one indivisible inert snapshot.
The authority must honestly allow the entire exact file to be loaded once with
`weights_only=True`; only the energy vectors may be consumed computationally.
No state may be restored, initialized, trained, forwarded, or emitted.

The validation index supplies requested action `actions[2]`, family and scene.
Its strict loader validates RGB path strings but follows none of them and
records `rgb_open_count: 0`.

All frame packs, RGB, train indices, other snapshots/checkpoints, predecessor
attempt roots, held-out roles, and sealed roles remain forbidden. The
insufficient `pack/val_meta.json` is not an input.

## 3. One-shot access and output contract

If later authorized, use fresh root
`.generated/dev/world_model_existing_pool_three_arm_v1_action_localization_v1/attempt_v1`.
It must be absent. Creating its reservation consumes the only attempt. Retry,
resume, refill, overwrite, V3 extension, and writes beneath the V3 root are
false.

The supervisor builds the attempt-plus-reservation in a private temporary
directory and atomically renames the complete campaign namespace into place.
A pre-rename failure removes only that supervisor-owned temporary directory and
does not consume the attempt; once the final namespace exists,
`reservation.json` already exists and the attempt is consumed. Any later
failure is closed by `terminal_supervision.json`.

Runtime caps and exact accounting:

- snapshot content opens: exactly one;
- validation-index content opens: exactly one;
- pack, RGB, train-index, other snapshot/checkpoint and network opens: zero;
- model forwards, training updates and optimizer steps: zero;
- GPU visibility and GPU seconds: zero;
- the external supervisor atomically materializes the reservation before
  launching the worker;
- worker outputs: aggregate localization or one terminal failure;
- receipt-only checker outputs: one checker receipt and reopens neither runtime
  payload; and
- the supervisor enforces a 1,800-second CPU wall cap covering authority
  validation through its pre-terminal decision, reserves 15 seconds for
  process-group termination/receipt closure, exact root inventories,
  exact worker/checker commands and exits, then writes one terminal-supervision
  receipt linking reservation, localization and checker bindings.

The snapshot is opened through a no-follow dirfd chain, read once, checked for
regular-file/inode/size stability and SHA-256 identity, and deserialized from
the already-hashed in-memory bytes with CPU `weights_only=True` loading.

The aggregate result must not emit row-level energies, tokens, state
dictionaries, optimizer state, RGB strings or scene IDs.

## 4. Frozen localization computations

All factual action IDs use the requested candidate position `actions[2]`.
Candidate prediction uses exact lowest-action-ID `argmin`, preserving V3.

The existing V3 metric implementation is reused to reproduce:

- the truth-row/prediction-column 9×9 confusion matrix;
- factual and predicted counts;
- row and scene/family-equal per-action recall;
- balanced accuracy and its original lower quantile;
- family-equal per-action hardest-wrong margin;
- hardest action ID and original global hardest-margin lower quantile;
- aggregate persistence and wrong-history log-energy comparisons.

For post-hoc per-action localization, define each row's action margin as

`min_{b != a}(E_b − E_a)`.

Within each family/action, average rows within scene, then scenes equally;
average the eight family values equally. Per-action lower/median/upper
quantiles use the same strictly positive scene-cluster Bayesian weighting
algorithm and seed `20260803` as V3, 10,000 replicates, sorted indices
500/5000/9499. These per-action intervals are diagnostics, not retroactive V3
gates and not frequentist coverage guarantees.

The scene weights are shared across all nine actions within each replicate.
The sorted q05 of the per-replicate minimum across actions must exactly
reproduce V3's registered global hardest-margin lower value
`−0.01138311990101325`. Routing uses both the nine marginal q05s and this
joint/global q05; nine positive marginal intervals alone do not repair the
registered multiple-action gate.

For persistence and wrong-history, the row value is respectively
`log(E_persistence) − log(E_factual)` and
`log(E_wrong_history) − log(E_factual)`. Apply the same scene-then-family-equal
per-action macro. New diagnostic Bayesian namespaces use seeds `20260807` and
`20260808`, with the same indices and interpretation. Aggregate controls must
exactly reproduce the original deterministic scene-clustered V3 results.

Also report, per action:

- validation rows and supporting scenes by family;
- frozen unique-train-row candidate-action counts and their descriptive
  inverse-uniform weights;
- row-weighted nine-way action-energy factual-rank histogram and MRR (distinct
  from the registered predictor target-rank-ratio gate);
- family/scene-equal factual energy and candidate spread;
- full family/scene-equal pairwise point-margin row; and
- the minimum pairwise-macro competitor ID and margin.

The count tuple below is independently witnessed by exact predecessor
`overlap_audit.json`, SHA-256
`ec2cfcd008059994d7803f1a14ede5d4ea3b76d50c36d0ca77532ae1deb8c2db`,
13,368 bytes, already included in the predecessor-evidence binding. These are
unique selected train rows summing to 16,000. They are not the 179,200 realized
scheduled presentations and therefore support a reweighting hypothesis, not a
causal conclusion that exposure caused the failure.

| ID | Action | Rows | Inverse-uniform weight |
|---:|---|---:|---:|
| 0 | arc_left | 2,959 | 0.6008 |
| 1 | arc_right | 1,197 | 1.4852 |
| 2 | backward | 1,075 | 1.6537 |
| 3 | forward_fast | 545 | 3.2620 |
| 4 | forward_medium | 4,303 | 0.4131 |
| 5 | forward_slow | 447 | 3.9771 |
| 6 | hold | 767 | 2.3178 |
| 7 | yaw_left | 2,893 | 0.6145 |
| 8 | yaw_right | 1,814 | 0.9800 |

## 5. Frozen post-localization routing

The localization result selects a successor class but authorizes none.

1. `UNCERTAINTY_LIMITED`: every alignment point is positive, but at least one
   marginal per-action q05 or the shared/global-minimum q05 is nonpositive. Do
   not change the model. First size a larger existing-pool scene-disjoint
   evaluation.
2. `TEST_ACTION_REWEIGHTING_HYPOTHESIS`: one or two alignment points are
   nonpositive and every failing action's descriptive unique-row
   inverse-uniform weight is strictly above `2.0`. Compare exact
   scheduled-presentation action weighting to a matched baseline. This name is
   deliberately noncausal; the diagnostic does not conclude exposure caused
   the failure.
3. `TEST_GLOBAL_ALIGNMENT_HYPOTHESIS`: at least three alignment points are
   nonpositive, or any failing action's unique-row weight is at most `2.0`.
   Compare one global action-margin objective to a matched baseline. This does
   not conclude that the failure is broad, that alternatives are physically
   different, or that the objective is causal.
4. Provenance is `UNAVAILABLE_WITHIN_BOUND_DIAGNOSTIC`. No executed/realized
   command join is an authorized input, so the result neither infers mismatch
   nor asserts repository-wide nonexistence and creates no data.
5. Alignment is repaired only when all nine points, all nine marginal q05s and
   the shared/global-minimum q05 are strictly positive. Before then persistence
   routing is deferred. Afterwards:
   - at least five nonpositive persistence q05s selects
     `PERSISTENCE_SYSTEMIC` and a later residual comparison;
   - one to four nonpositive q05s, or a nonpositive aggregate persistence lower
     bound, selects `PERSISTENCE_LOCALIZED_OR_AGGREGATE_UNREPAIRED` and a direct
     planning-usefulness gate with the proxy caveat before another model tweak;
   - no nonpositive per-action q05 and a positive aggregate lower bound selects
     `PERSISTENCE_PASSED` and the planning-usefulness gate.

No first successor may combine exposure, alignment, persistence, architecture,
data or schedule changes.

## 6. Meaningful-improvement and stall rules

Final registered latent-proxy repair retains the original thresholds:

- balanced-accuracy lower quantile strictly above `1/9`;
- hardest-action-margin lower quantile strictly above zero;
- persistence and wrong-history lower bounds strictly above zero;
- rank ratio at least `0.25` at two of the three registered tail observations;
- all contract, train-fit and scene-disjoint cross-arm gates pass.

For an alignment successor, V3's point deficit is
`dA = 0.009453551490358742`.

For each arm define `M_a` by the exact row→scene→family margin macro above and
`A = min_a M_a`. The paired point statistic is
`Delta_A = A_treatment − A_concurrent_baseline`. In each of 10,000 paired
replicates, seed `20260811`, generate one shared strictly-positive 52-bit
exponential scene-weight table and apply it to both arms and all nine actions;
then subtract the two replicate minima. Sort and use q05/q50/q95 indices
500/5000/9499.

- exploratory absolute action repair takes decision precedence only when the
  §5 alignment-repaired condition holds (all nine points, all nine marginal
  q05s and the shared/global-minimum q05 are strictly positive) and every
  provisional retention gate below passes;
- otherwise meaningful improvement requires point `Delta_A >= 0.25*dA =
  0.0023633878725896856` and paired q05 strictly above zero;
- otherwise stalled means paired q95 is strictly below `0.10*dA =
  0.0009453551490358743`;
- everything else is inconclusive.

For a later persistence successor, V3's point deficit is
`dP = 0.14645548512800682` log units.

- define `Delta_P = P_treatment − P_concurrent_baseline`, where `P` is the
  aggregate scene/family-equal log persistence advantage; use paired shared
  scene weights, seed `20260812`, 10,000 replicates and indices 500/5000/9499;
- meaningful improvement requires point `Delta_P >= 0.25*dP =
  0.036613871282001706` and paired q05 strictly above zero;
- otherwise stalled means paired q95 is strictly below
  `0.10*dP = 0.014645548512800683`;
- full repair still requires the absolute persistence lower bound above zero.

For this exploratory two-arm comparison, provisional retention means the
treatment itself keeps balanced-
accuracy q05 strictly above `1/9`, wrong-history q05 strictly above zero, the
registered target-rank ratio at least `0.25` at two of three tail observations,
all finite/contract/train-fit gates, and no newly nonpositive action point that
was positive in the concurrent baseline. Failure of any absolute retention
gate overrides “meaningful.” This is an absolute provisional rule, not a claim
of statistical noninferiority and not evidence that the treatment preserves
V3's blind/shuffled cross-arm gates. Those gates are unmeasured in two arms and
must be re-established in the later fresh confirmation before any statement
that the original thresholds passed.

Every training attempt must be fresh, equal-data/equal-budget against a
concurrent baseline, terminal-only with no checkpoint/validation selection,
and consumed regardless of outcome. The selected treatment receives one
matched comparison. A stalled result closes that treatment family immediately.
An inconclusive result permits exactly one separately preregistered identical
replication; if it is not meaningful, the whole branch closes. A meaningful
but incomplete result permits at most one separately preregistered fixed
same-mechanism continuation. The replication and continuation options are
mutually exclusive: this branch has at most two training comparisons total.
At that cap, only exploratory absolute repair proceeds to fresh confirmation;
every other outcome closes this plan, including meaningful-but-incomplete or
inconclusive progress. Further pursuit would require a new explicit
preregistration and authority rather than an automatic tweak. No coefficient,
schedule, architecture, data, or alternate-objective hop follows a stall under
this plan. One seed is only development evidence; a stronger claim requires a
fresh previously unconsulted scene-disjoint role and fixed multi-seed
confirmation.

## 7. Candidate successor mechanisms

The smallest successor is a fresh two-arm 700-update comparison on the same
16,000 train rows and 2,048 validation rows, same schedule/order/masks,
initialization, frozen substrate and compute-matched candidate evaluations:

- reweighting-hypothesis treatment: before optimizer creation, count exact
  scheduled presentations `m_a` across the registered 179,200-presentation
  order, require `sum_a m_a = 179200`, emit the count receipt, set
  `w_a = 179200/(9*m_a)`, and optimize the fixed-scale empirical mean
  `mean_i(w_{a_i} L_i)`. Never self-normalize by a batch's weight sum;
- alignment treatment: factual loss plus
  `mean relu(0.01 + E_factual − min_wrong E)`, coefficient `1.0`;
- there is no optimization/schedule treatment in this routing tree.

The actual treatment is chosen only by §5. A changed-objective success is
called `PASS_EXPLORATORY_ACTION_ALIGNMENT_PROXY_REPAIR`, not the original
learnability claim, because the alignment proxy was directly optimized and the
two-arm run lacks fresh blind/shuffled controls. A later confirmatory
treatment/blind/shuffled experiment on a previously unconsulted scene-disjoint
role must restore the original controls before any new factual-learnability
claim or statement that all registered thresholds passed.

The hinge forces the factual requested-action candidate away from every other
action even when two actions could have physically equivalent outcomes. It can
therefore repair an identification proxy by construction; it cannot establish
untaken-action causality or planning value. The consulted V3 validation panel
may select and assess this exploratory mechanism but cannot support its later
confirmatory claim.

## 8. Planning-usefulness boundary

Passing every latent threshold establishes factual requested-action alignment
and predictor usefulness for this protocol. It does not establish physical
outcomes for untaken actions or planning usefulness.

Planning evidence requires a separately preregistered scene-disjoint role with
physically executed or branched candidate outcomes and an independent utility
or regret target. It must report top-action regret, pairwise ranking,
unsafe-action rate and calibration against blind, shuffled and persistence
controls. Deployed WM-D evidence additionally requires a paired planner
on/off or score-permutation intervention on identical scenes, seeds and budgets,
with task improvement and safety noninferiority.

No planning threshold is invented from V3 latent-energy units. The existing
counterfactual action-regret evaluator is a later source seam, not authority to
open or create its runtime role.

## 9. Claim boundary

The read-only result is post-hoc development localization only. It makes no
requested-versus-executed equivalence, untaken-action, architecture-sufficiency,
planner-utility, navigation, WM-A, WM-D, G2-G8, promotion, deployment or
production claim. It grants no successor execution.
