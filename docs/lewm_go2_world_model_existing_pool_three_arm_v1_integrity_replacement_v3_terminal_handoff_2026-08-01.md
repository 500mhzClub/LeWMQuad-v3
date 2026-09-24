# Existing-pool three-arm world-model V3 terminal handoff

Date: 2026-08-01

Branch: `jepa-spatial-world-model-nav`

Attempt: `world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1`

Status: **execution complete; contract integrity passed; registered scientific
decision `LOCALIZE_ACTION_ALIGNMENT_FAILURE`**

This is a development-tier terminal record, not execution authority and not
citable as scientific evidence or promotion evidence. The attempt is consumed.
It must not be retried, resumed, refilled, overwritten, or reused.

The machine-readable terminal review is
`docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_terminal_review_2026-08-01.json`.
It directly binds the six outer terminal artifacts and all 24 measurement
receipts.

## 1. Bottom line

Yes: an actual GPU experiment ran to completion under the external supervisor.
The worker and checker both exited zero, all 700 registered updates completed
for each of the three arms, and the receipt checker passed.

The experiment found real factual requested-action signal in the existing Go2
pool. The aligned conditioned arm fit and generalized better than both the
candidate-blind arm and the marginal-preserving, cross-scene action-shuffled
arm. Its nine-way balanced action-identification accuracy was more than twice
chance.

It did **not** pass the full preregistered positive claim. At least one action
class failed the hardest-wrong-action alignment requirement. A later gate was
therefore not formally reached; if evaluated descriptively, the conditioned
predictor also remained worse than latent persistence.

The scientifically defensible conclusion is consequently narrower than either
of the two tempting extremes:

- the old claim that the on-policy pool provides no learnable action signal is
  refuted for factual, scene-disjoint prediction;
- the existing selected pool has not passed the complete factual-action
  learnability contract;
- immediate bulk counterfactual generation for this factual-action diagnosis
  is not justified;
- architecture incapacity is not supported, because the conditioned arm did
  use action information; and
- general architecture sufficiency is not established, because uniform action
  alignment, persistence usefulness, untaken-action validity, rollout, and
  navigation were not demonstrated.

## 2. What actually ran

The experiment used the existing non-protected Go2 main pool. It did not render
new simulator data and it did not train over the approximately 3 TB physical
pool.

| Item | Executed contract |
|---|---:|
| Existing train H6 rows | 16,000 |
| Existing validation H6 rows | 2,048 |
| Train / validation scenes | 1,000 / 150 |
| Scene overlap | 0 |
| Arms | conditioned, candidate-blind, action-shuffled |
| Updates per arm / total | 700 / 2,100 |
| Presentations per arm | 179,200 |
| Fresh packed frame bytes | 2,716,729,344 |
| GPU phase | 1,354.636 s |
| Supervisor wall time | 1,367.421 s |
| Peak GPU allocation | 6,081,984,512 B |

The frozen encoder and target encoder remained byte-identical from start to
finish, stayed in evaluation mode, accumulated no gradients, and received zero
EMA updates. This establishes exact substrate preservation for this run only;
it is not a functional physical or spatial-retention result.

The selected roles covered all nine visible requested actions and all 81
adjacent ordered action pairs. They covered 722 of 729 ordered triples; the
seven absent triples were registered as diagnostic-only, not a validity gate.
The shuffled arm used a global cross-scene bijection with exact action
marginals, zero fixed candidates, and zero same-scene donors.

## 3. Registered gate result

| Gate | Result | Terminal evidence |
|---|---|---|
| Contract integrity | **PASS** | checker `PASS`; exact accounting and frozen-substrate identity |
| Training fit | **PASS** | conditioned vs blind `+0.0118187`; vs shuffled `+0.0349968` log-energy advantage |
| Scene-disjoint generalization | **PASS** | u700 lower bounds `+0.00251384` and `+0.0215773` |
| Balanced action identification | **PASS** | `0.246934`; lower bound `0.230145`; chance `0.111111` |
| Hardest-action alignment | **FAIL** | margin `−0.00945355`; lower bound `−0.0113831`; required positive |
| Predictor usefulness / noncollapse | not reached | persistence would fail; wrong-history and rank would pass |

At update 700, the registered family-equal validation advantages correspond to
approximately 0.653% lower conditioned energy than blind and 2.598% lower than
shuffled. These are the controlled metrics. The unweighted raw validation means
were `0.125488`, `0.128231`, and `0.130497`, respectively, and should not be
substituted for the registered family-equal comparison.

The conditioned arm's balanced accuracy was `0.246934` with no exact ties. The
blind arm tied on all 2,048 rows and produced chance-balanced accuracy, which is
the expected negative control. The conditioned hardest-action margin improved
from `−0.08007` at update 0 to a best of `−0.00635` at update 200, then worsened
to `−0.00945` at update 700 even while average identification and cross-arm
advantages continued to improve. That trajectory does not justify blindly
continuing the consumed attempt.

The later predictor-health gate was masked by precedence. Its descriptive
values matter for diagnosis:

- persistence log-energy advantage: `−0.146455`, lower bound `−0.182912`;
  conditioned energy was approximately 15.77% above persistence;
- wrong-history log-energy advantage: `+0.122551`, lower bound `+0.117667`;
- prediction-to-target effective-rank ratios at updates 500/600/700:
  `0.445224`, `0.472739`, `0.468267`.

Thus the predictor used history and did not collapse, but average MSE training
did not yield a uniformly aligned or persistence-useful predictor under the
registered schedule.

The balanced-accuracy and hardest-margin lower quantiles use the frozen
deterministic positive-weight scene-cluster Bayesian bootstrap; they are not
frequentist coverage guarantees. Cross-arm, persistence, and wrong-history
lower bounds use the separately frozen deterministic within-family scene
resampling bootstrap. Only one training seed was run.

## 4. First-principles disposition of the earlier diagnosis

The July 31 critical-path argument conflated two questions:

1. Can observational data teach factual
   `next state | history, requested action` across related states and scenes?
2. Can it identify outcomes for actions not taken from the identical physical
   state?

This experiment answers the first question partially and positively: aligned
actions improved prediction beyond both controls on disjoint scenes. It does
not answer the second. Same-state action contrast remains necessary for a
strong untaken-action causal claim, but it was not necessary to expose factual
action signal.

The data/objective/architecture diagnosis is now:

- **Data absence as the immediate cause:** contradicted at the aggregate
  factual level. The selected existing rows contained usable action signal.
- **Bulk data scarcity:** untested and not a current explanation. The run used
  16,000 training rows, 0.885% of the 1,807,552 existing row-disjoint packed H6
  candidates. Train and validation together selected 18,048 rows, or 0.998%.
- **Architecture cannot condition on action:** contradicted at protocol level.
  The conditioned arm beat matched blind and shuffled controls.
- **Architecture is sufficient:** not established. One or more action classes
  remained misaligned and predictor usefulness failed descriptively.
- **Objective/optimization mismatch:** plausible, not uniquely proven. The
  average MSE objective improved aggregate energy and balanced accuracy while
  failing the worst-action and persistence criteria.
- **Requested-versus-executed mismatch or frozen-feature limitation:** still
  plausible. The permitted receipts do not identify the failing primitive or
  expose a complete executed-command join.

Diagnostic B remains only a small malformed-role training-capacity number. It
is superseded as evidence for factual scene-disjoint action learning and still
says nothing about counterfactual generalization. The old spatial-control
degradation interpretation also remains unverified; this run froze the encoder
and did not repeat a functional retention panel.

## 5. Status against the repository's actual goals

The repository goal is a deployed navigation system with persistent physical
and per-color beliefs, learned selection, and a world model causally present in
candidate or action scores. This experiment is one diagnostic below that
system-level boundary.

| Property | Status after V3 |
|---|---|
| Factual requested-action diagnostic | valid localized failure with partial positive signal |
| WM-A untaken-action ranking/regret | **unmeasured** |
| WM-C composability / latent rollout | **unmeasured**; direct re-entry remains structurally incompatible |
| WM-S physical/spatial retention | byte identity only; functional retention **unmeasured** |
| WM-D deployed causal presence | **unmeasured** |
| Learned target/frontier/route/motion selection | **unmeasured** |
| Formal G2–G8 or promotion | **not passed and not authorized** |

Accordingly, the execution goal for this bounded experiment is complete. The
broader world-model/navigation goal is not complete, and this result must not be
presented as though it were.

## 6. Data decision

V3 does not justify new bulk simulator data for this factual-action
localization now. Other repository workstreams and their data authorities are
outside this result.

The repository still has roughly 1.8 million existing packed H6 candidates
inside an approximately 3 TB physical pool. V3 used a deliberately small
16,000-row schedule to answer whether any controlled factual action signal was
learnable. It found that signal. More existing rows may or may not repair the
worst-action failure, but the current receipts do not diagnose row count as the
cause.

A small, scene-disjoint counterfactual evaluation role may become justified
later if factual alignment and predictor usefulness pass and the project still
needs an untaken-action claim. That would be a targeted evaluation role, not a
new multi-terabyte training corpus.

## 7. Next bounded work

The cheapest information-gaining operation is not another training run. Under
fresh, separate read-only authority, allowlist only the bound conditioned
update-700 snapshot and the exact bound validation-index metadata. Do not open
pack frames, RGB, or any other snapshot. Use those two inputs to extract:

1. the hardest failing action ID;
2. the nine-by-nine confusion matrix;
3. per-action and per-family hardest-wrong margins;
4. persistence advantage by action and family.

Once the failing action is known, derive its factual support from the permitted
receipt and bound validation metadata. Treat requested-versus-realized or
executed-command provenance as a separate source/metadata audit. Run it only if
a complete exact role-bound join already exists; otherwise record provenance as
unavailable and do not infer it or create data under this recommendation.

The current public receipts deliberately omit those values, so naming a culprit
without that extractor would be fabrication. No snapshot or checkpoint bytes
were opened during terminal review.

Choose a successor only after that localization:

- if support is sparse, preregister stratified resampling from the existing
  pool; if a complete role-bound command join exists and shows requested versus
  executed mismatch, isolate that provenance issue separately;
- if the failure is broad, compare one explicit action-ranking or
  persistence-residual objective against the same existing data;
- test more optimization only as a fresh preregistered ladder, never by
  resuming V3; and
- collect paired counterfactual evaluation data only after factual alignment
  and predictor usefulness are established.

This handoff grants none of those operations.

## 8. Terminal identity and custody

Frozen lifecycle:

- source: `20a7ad293fa1926e08e1a59db274f82898fc0f09`
- independent review: `12e93654db6da95a70f8f63a0c90e4bdcae57c83`
- authority and execution HEAD: `0ddace8846bedb4b268e4123dfdaf443a15e5d97`

Terminal artifact bindings:

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| reservation | `b217f94e8bce2dc91dc931644bc10f07d0bb979cd540200e241225fb8a2d4a12` | 17,048 |
| overlap audit | `ec2cfcd008059994d7803f1a14ede5d4ea3b76d50c36d0ca77532ae1deb8c2db` | 13,368 |
| shuffle audit | `892280a84967eb51d4a9f2733c79729361df2a0ac308191a1955c6f0a4a6a1bb` | 6,494,146 |
| result | `764ee61b7bb8b7e1221f01fc34ba0554d0ca681fde21e99b1a9f5585b3360bd4` | 26,054 |
| receipt check | `bf55ca3e69b87b2bd81abf4b33909fc2b2364302cc7e67a958b3c7abcdf4ff66` | 6,401 |
| terminal supervision | `313d449b3bd44c0e55e674195b548c444f817d837780faa45769f04b7286c8f4` | 4,824 |

Three independent terminal audits rechecked the receipt chain, all 24
measurement hashes and sizes, gate arithmetic, accounting, source closure, and
claim boundaries. All passed. The result JSON remains frozen at
`COMPLETE_PENDING_TERMINAL_REVIEW` with `scientific_verdict_emitted: false`;
this durable review records the terminal interpretation without mutating the
consumed runtime root.

Among attempt-root runtime artifacts, terminal review opened only the six
permitted outer JSON files and 24 measurement JSON receipts; source files and
durable documentation were separately inspected. The 31 pack/snapshot files
were identity-statted only. No pack payload, checkpoint/snapshot bytes, RGB,
held-out, sealed, or network resource was opened by terminal review. Runtime
artifacts remain uncommitted under `.generated/dev/**`.
