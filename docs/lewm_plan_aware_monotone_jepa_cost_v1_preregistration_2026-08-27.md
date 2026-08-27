# Plan-aware monotone JEPA cost V1 preregistration

Experiment: `PLAN_AWARE_MONOTONE_JEPA_COST_V1`.

This is a development-only, route-only ranker experiment. It supports no deployment-safety, material-hazard, learned-assurance, or closed-loop navigation claim.

## Preserved predecessor authority

- `RAW_LATENT_GOAL_COST_NO_GO`
- `TRUE_FUTURE_LATENT_GOAL_COST_NO_GO`
- `JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS_NOT_SUPPORTED`
- These IDs remain scoped to the predecessor raw token-wise goal-cosine assay; a successor plan-aware finding does not overwrite them.

- rollout training improves direct counterfactual future fidelity at H1–H4.
- rollout training improves selected action-specific retrieval metrics, strongest at H3–H4.
- predicted latents retain partial occupancy information but remain substantially below true-target occupancy.
- the registered proprioception interaction was broadly null.
- planning utility was not tested by the predictor qualification assay.
- Raw token-wise cosine distance between future V-JEPA latents and the virtual goal-view latent did not provide useful route ordering, even with true-future latents and oracle viability.
- The virtual goal renderer contained the floor plane but not the maze walls or landmarks. That result was therefore not a valid visual wall-avoidance test.

## Active policies

- `EVALUATION_FIRST_SINGLE_SEED`
- `ROW_LEVEL_EVIDENCE_PERSISTENCE`
- `DEVELOPMENT_MODE_END_TO_END_EXECUTION`
- `EXPLORATORY_SINGLE_SEED_FIRST`

## Frozen custody

- Source commit: `1d799eb24d8171cb6d90bc0d0e375d9e1b0cc4f0`.
- Required requirements ancestor: `b29eae1929725a4cc26a35d95662b545daee4553`.
- Panel: 48 frozen states, 12 candidates/state, H1-H3; 32 fit, 8 calibration, 8 observed development-heldout.
- TRUE/R1/RR tensors are already bound. P1/PR checkpoints exist, but 48-state P1/PR tensors and proprio-input custody are not yet bound.
- Route role is loaded from the bound 48-state candidate-invariant authority receipt; it is never inferred live from candidate outcomes.
- Contract-freeze commit subject: `Freeze plan-aware monotone JEPA route cost`.

## Rankers

Two final-only rankers are trained on fit rows: a 138-D no-latent kinematic/base MLP and a latent ranker with shared 1024-to-64 token projection, one 71-to-64 query, shared attention over current/H1/H2/H3, and a 394-to-256-to-128-to-1 readout. Both are residuals around the negative kinematic rank-cost anchor.
Every 768-by-1024 timepoint tensor is explicitly viewed as a 24-by-32-by-1024 spatial grid in row-major order (`flat_index = y * 32 + x`) and flattened in that same order before shared token LayerNorm, projection, and attention. This view/flatten contract does not permute token values.

AdamW uses lr 1e-3, weight decay 1e-4, 60 epochs and only the final checkpoint. Loss weights are pairwise 1, listwise 0.5 and residual L2 1e-3; listwise temperature is 1.
Pairwise targets are the sign of conditioned margin-Borda utility differences only when the absolute difference exceeds 1e-12; otherwise they are zero. The completion/0.03 m/5 degree direct comparator only constructs that Borda utility and is not a separate pairwise target.
Evaluation uses the same population-conditioned Borda authority for pairwise accuracy and ideal top-k/MRR/mean-rank metrics. Every candidate within 1e-12 of maximum Borda utility belongs to the oracle-best set; top-k succeeds for any member and rank uses the earliest model-ranked member. Candidate index resolves only deterministic representative/order ties. Route-progress Spearman/Kendall and maximum-progress normalized regret remain separate progress diagnostics.
A legitimately unavailable metric from an empty population, complete score tie, constant progress, or absence of Borda-ordered pairs remains `null` evidence. Any required gate criterion using it is false, never an execution exception. Stage A therefore classifies an all-tied true scorer as `PLAN_AWARE_JEPA_COST_NO_SIGNAL`; predicted, proprioception, and substitution gates likewise fail closed while preserving nullable descriptive deltas and factorial contrasts.
Condition-keyed seeds use the frozen SHA-256 derivation. Both models' base branches use the shared-base subkey for byte-identical initialisation; latent-only parameters use the exact latent-condition key.
Execution is no-retry by default. Only an archived failure in phase `TRAINING_SMOKE`, with zero completed full-training epochs, zero opened calibration/heldout rows, no final checkpoint, nonreusable partial artifacts, and nothing running, may authorize a wholly fresh corrected attempt. It must use a clean no-merge descendant correction freeze, bind every prior smoke archive in `prior_smoke_failure_custody`, reuse no artifact, and validate exact closure/authority. Because fit rows were opened before smoke, every archived pre-smoke source-closure snapshot must prove byte-identical preregistration, contract, output schema, fixture, and route-role authority; only closure-covered Python implementation/tests and the refreshed source-closure receipt may change. An empty or closure-only descendant is forbidden: at least one covered Python implementation/test byte must differ from every archived smoke snapshot. Same-freeze and every post-smoke retry are forbidden.

The target is the existing local-waypoint margin-Borda preference. Completion appears only in that ordering tuple. The prospectively frozen local route margin-Borda listwise target is the only utility target; there is no completion, safety, contact, viability, stuck, material-contact, or prior aggregate-scorer-utility head or target, and no unrestricted aggregate-utility head.
Training uses only the prospectively authorised oracle-viability-admissible candidate sets. Oracle viability is a disclosed row-level conditioning mask, never a score input or route target.
All 32 fit-state identities and every fit row remain in the evidence ledger. A fit state contributes to optimization only when at least two oracle-viability-admissible candidates exist. Zero- and singleton-admissible states are deterministically classified as `SKIPPED_ZERO_ADMISSIBLE` or `SKIPPED_SINGLETON_ADMISSIBLE`, contribute no pairwise, listwise, or residual loss, receive no optimizer step, and are excluded from the epoch-loss denominator. Epoch averages use `CONTRIBUTING` fit states only.
For every source and candidate population, evaluation separately reports the selected candidate's population-conditioned margin-Borda route utility per state and its aggregate sum and mean.
After both final checkpoints are locked and the evaluation/heldout barrier is open, the byte-bound predecessor 1,728-row candidate-evidence ledger is read once to re-reduce `RAW_TRUE_FUTURE_GOAL_COSINE`, `RAW_R1_GOAL_COSINE`, and `RAW_RR_GOAL_COSINE` from `-cost_h3` under these same successor Borda/rank/progress metrics. No cosine or predictor inference is rerun. Copied predecessor aggregates remain a separate historical, non-comparable context rather than matched evidence.

## Stages and stop rules

Stage A evaluates true-future candidate-specific route information. A failed true-future gate stops before new predictor inference and yields `PLAN_AWARE_JEPA_COST_NO_SIGNAL`; its incremental gate is persisted as classification `TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE_NOT_EVALUATED`, status `NOT_EVALUATED_TRUE_GATE_FAILED`, evaluated/pass false, reason `TRUE_FUTURE_GATE_FAILED`, and empty comparisons. Stage B is permitted only after that gate passes and substitutes R1/RR/P1/PR into the fixed latent ranker. Stage C runs only after `PROPRIOCEPTIVE_ROUTE_CONTRIBUTION` and uses strict substitutions/derangements without refitting.
The future-derangement result reports pairwise-accuracy loss, selected-progress loss in metres, selected-progress-ratio loss, normalized-regret worsening, and best-route-top-3 loss. The metre and top-3 losses are descriptive only; the frozen materiality triggers remain pairwise loss, progress-ratio loss, or regret worsening with no material principal reversal.
R1, RR, P1 and PR each receive the same frozen absolute preservation screen. `all predicted substitutions fail materially` means none passes. If any source passes while the complete RR gate fails, the primary classification remains unresolved and execution fails closed.
The complete RR gate additionally requires ALL_CANDIDATES selected immediate-contact, successor-nonviable, and stuck counts all to be no worse than R1. The incremental and proprioception gates retain their separate contact-plus-nonviability adverse checks.
The result persists the complete specify-only payload for its selected next experiment: oracle-admissible fixed-bank closed-loop MPC; one-seed route-consistency predictor training without a safety or protected-scope change; or a non-greedy obstacle-mediated local-subgoal assay with at least two geometrically plausible alternatives, oracle admissibility, and initially no topological memory or beacon layer.

All thresholds, primary classifications, next decisions, hashes, role barriers and output schemas are normative in the accompanying canonical JSON contract.

Contract SHA-256: `1667f325be2c835a6222dc90bb684f373a06b365d59b70e9746fd7adb052c382`.
Output-schema SHA-256: `e66798b015a2068f5be252dd3c0e4bc0b84aef4cf262d9b27b38089d65098338`.
Evaluator-fixture SHA-256: `440e21d4bb557a16e8d60a01eb7a0708438ff56b9f4c0947b371ed30a11478e3`.
