# Main-pool recurrent-JEPA temporal census result

Date: 2026-07-27

## Why this check was necessary

- The earlier recurrent-H4 census covered only the 5,172-row V5 development
  pair index. Its negative result was valid for that extraction, but it did
  not establish that the main `datagen_full` corpus lacked temporal support.
- This census therefore tested the main train/validation rollout metadata
  directly. Marginal primitive counts were not accepted as evidence of
  temporal feasibility.

## Scope

- Read only the literal `frames.jsonl` metadata leaves under the exact
  `datagen_full/rollout/train` and `datagen_full/rollout/val` roles.
- Covered all 1,150 scenes: 1,000 train and 150 validation, across all eight
  frozen scene families.
- Parsed 55,200,000 rows and 138,549,246,020 bytes.
- Ignored 38 non-sequence plan-root names at the name filter without statting
  or opening them; all 1,150 expected unique sequence leaves were present.
- Opened no RGB, labels, raw-message payloads, test roles, held-out material,
  sealed material, checkpoints, models, or benchmark outputs.
- Performed no training, GPU, navigation, qualification, promotion, or
  deployment run.

## Exact temporal unit

- One candidate contains six consecutive, reset-safe primitive transitions
  and seven observation endpoints.
- `p0,p1` join the three ordered context observations `e0,e1,e2`.
- `p2,p3,p4,p5` are the four future actions whose target endpoints are
  `e3,e4,e5,e6`, a nominal two-second prediction horizon.
- Transitions join only within the same scene, environment, episode, reset,
  split, and manifest context. Global interleaved frame order is never treated
  as environment continuity.
- Row-disjoint capacity packs each maximal transition path at offsets
  `0,6,12,...`; sliding H6 capacity reports every valid six-edge window.

## Execution correction

- V1 was interrupted after independent review caught an invalid fixed-shape
  assumption: real scenes can contain legitimate mid-scene resets, so a scene
  need not contain exactly 48 uninterrupted episode streams.
- The V1 failure receipt records `KeyboardInterrupt` and grants no authority.
- V2 changed only census integrity/accounting behavior: it permits any number
  of reset-separated streams, still forbids cross-reset joins, and requires
  exactly 1,000 rows for each of the 48 environments in every scene.
- V2 also requires unique scene and manifest identities within each role,
  zero train/validation identity overlap, strict scene/manifest schemas, and
  an ordered aggregate content commitment.

## V2 result

- Decision: `MAIN_POOL_H4_METADATA_FEASIBLE`.
- Failed predicates: zero.
- Temporal/row integrity violations: zero.
- Primitive transitions: 10,960,721.
- Sliding H6 windows: 10,614,345.
- Row-disjoint H6 windows: 1,807,552.
- Reset-safe streams: 79,279, of which 67,821 contribute at least one H6.
- Distinct identities: 1,000 train and 150 validation; no duplicate identity
  within a role and no train/validation scene or manifest overlap.
- Every one of the nine primitives occurs at each future position `p2` through
  `p5` in both roles and every family.
- Weakest train family is `rough_local_dynamics`: 459,863 sliding H6 and
  78,390 row-disjoint H6 windows from all 50 scenes. Its weakest exact
  action-position cell has 10,550 occurrences.
- Weakest validation family is `rough_local_dynamics`: 64,341 sliding H6 and
  10,969 row-disjoint H6 windows from all seven scenes. Its weakest exact
  action-position cell has 1,504 occurrences.
- These minima exceed the fixed V2 family, action-position, and scene-breadth
  thresholds by wide margins.

## Artifact bindings

- V2 receipt:
  `.generated/go2_recurrent_jepa_main_pool_census_v2/receipt.json`
- V2 receipt SHA-256:
  `aac85f1016dca12e57e0cf612cd51a745becb2941adf361c0b4a752fe10a5408`
- Ordered source-content commitment SHA-256:
  `0d5ce1c8aae3777a3e1c930959d5985817d92c28ec240ad03ed79121869d4696`
- V1 interrupted receipt:
  `.generated/go2_recurrent_jepa_main_pool_census_v1/receipt.json`

## Scientific conclusion and next step

- The main pool already contains ample reset-safe temporal material for the
  proposed recurrent, action-conditioned, multi-horizon JEPA. Fresh data
  collection is not the next step, and the small V5 index must not be used to
  argue otherwise.
- The next implementation should build a deterministic train/validation-only
  sequence adapter over these exact H6 units and bind their seven observation
  endpoints to the existing RGB corpus.
- The first experiment should train the context encoder, recurrent belief
  module, action-conditioned predictor, and EMA target encoder as one joint
  JEPA objective. It should not freeze the encoder and train a separate
  predictor, because that would not test the intended learned JEPA stack.
- Passing this metadata census authorizes no training by itself. Held-out and
  sealed benchmarks remain unopened.
