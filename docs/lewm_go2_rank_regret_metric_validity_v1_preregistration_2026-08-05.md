# Preregistration: rank-regret metric-validity study V1

Date: 2026-08-05
Branch: `jepa-spatial-world-model-nav`
Attempt identity: `go2_rank_regret_metric_validity_v1_attempt_v1`

Status: **development-tier, evaluation-only, non-confirmatory metric-validity
study.** It is not a world-model, dynamics, navigation, promotion, or safety
experiment. It trains no navigation candidate and promotes nothing. It runs no
new simulation. It opens no untouched, sealed, held-out, or V4 material.

This document is frozen before any correlation is computed.

---

## 1. Question

Does **normalized rank regret on the matched-branch panel** — the endpoint that
governed the fixed gates in handoff §11 and §13 — predict closed-loop planning
utility on this stack?

## 2. Why the question is open, and why it is not rhetorical

Two completed Aug-4 development results point in opposite directions about the
same representation:

- **Planner-oracle assay (H1 PASS).** Correct candidate ranking converts to
  materially better control: `+0.4320` m progress over deterministic shuffled
  scores, 95% interval `[0.3773, 0.4833]`, 24/24 scenes, oracle first-action
  regret exactly `0`. The planner seam is not the bottleneck.
- **DINO true-successor goal-cost (gate FAILED).** The same frozen-DINO cost
  reached **14/24 successes, exactly matching the geometric oracle's 14/24**,
  and `+0.29568` m over shuffled with interval `[0.10139, 0.47261]` — while
  **failing** its first-action geometric regret gate at `0.01574` m against a
  required `0.020` m, interval `[-0.04360, +0.01188]` crossing zero.

A scorer that is regret-gate-failing and closed-loop oracle-matching is direct
evidence that a one-step endpoint and the closed-loop endpoint can disagree.

That result also recorded, unprompted, the exact design caveat this study
resolves:

> scene regret is measured on each policy's own states after its trajectory has
> diverged (226 true-successor decisions versus 288 shuffled decisions). It is a
> policy-level physical-ranking comparison, not a pure same-state rank assay.
> ... a successor can preregister a same-state diagnostic separately.

## 3. The metric-chain design, and why it is split in two

There are **two distinct one-step metrics** in play, and the literature of this
repository has not so far distinguished them:

- **G — geometric first-action regret, in metres.** Progress lost relative to
  the best available branch. This is what the Aug-4 closed-loop harness measured.
- **R — normalized rank regret, dimensionless.** Dense-rank position of the
  selected branch over `max(1, max_dense_rank)`. This is what the §11 and §13
  gates used.

The frozen-DINO same-patch goal cost, the single most informative scorer, is
**not computable on the V3 matched panel**: that panel carries a relative target
*vector*, not a goal *image*, and the DINO cost is defined against a goal image.
Attempting to force it onto that panel would silently redefine the scorer.

The study therefore establishes the link in two measured parts rather than
asserting it in one:

- **Part B1 — does a one-step metric predict closed-loop utility?**
  Correlate G against closed-loop progress across the seven Aug-4 policies.
  All values are bound from completed, independently recomputed registered
  results. No new simulation.
- **Part B2 — do the two one-step metrics agree?**
  On the immutable V3 matched evaluation role, compute **both** G and R for every
  arm, on the identical 128 states and identical nine branches. This is a pure
  same-state assay and removes the trajectory-divergence confound named above.

The chain is valid only if both links hold, and the registered decision rule in
§6 requires both explicitly.

## 4. Arms and inputs

### 4.1 Part B1 scorers — bound closed-loop values, not recomputed

| scorer | closed-loop progress (m) | success | geometric regret G (m) |
|---|---:|---:|---:|
| `bearing` | `0.9000` | 24/24 | `0.02486` |
| `oracle_mpc` | `0.8151` | 14/24 | `0.00000` |
| `dino_true_successor` | `0.6494` | 14/24 | `0.05956` |
| `random` | `0.4968` | 0/24 | `0.07346` |
| `dino_true_successor_shuffled` | `0.3537` | 0/24 | `0.07530` |
| `dino_persistence` | `0.0000` | 0/24 | `0.13534` |
| `hold` | `0.0000` | 0/24 | `0.13534` |

Seven scorers spanning the full closed-loop range.

### 4.2 Part B2 arms — computed on the V3 matched evaluation role

The immutable collection at the registered SHA-256
`711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0`, evaluation
role, 32 scene-disjoint scenes, 128 states, 9 branches per state.

Arms are those whose rule is well defined on this panel: `physical_oracle`,
`geometric_endpoint` (rank by predicted final distance to target), `bearing`,
`task_action_only`, `context_only`, `dinov2_true_successor`,
`privileged_physical_successor`, `random_expected`, and `hold` (always the
zero-command branch). The learned arms reuse the observability-ceiling assay's
already-fitted scores; nothing is refitted here.

**G on this panel** is defined as
`max_j(physical_target_progress_m) - physical_target_progress_m[selected]`,
in metres, averaged over states — the same quantity the harness measures.
**R on this panel** is the unchanged registered normalized rank regret with the
`max(1, max_dense_rank)` denominator and complete-tie convention.

## 5. Statistics and their declared limits

- Spearman rank correlation, with Pearson as secondary.
- Uncertainty by scorer-level bootstrap, 10,000 resamples, seed `2026080504`.
- Ties handled by average ranks.

**Declared limits, stated before any result is seen.**

1. B1 has seven scorers and B2 has nine arms. Both are powered to distinguish
   "strong monotone relationship" from "no relationship" and **not** to estimate
   a correlation precisely. Neither may be reported as doing so.
2. B1's two endpoints come from the same 24-scene harness runs, so G there is
   measured on each policy's own diverged states. B2 is same-state by
   construction and has no such confound; that is precisely why the chain is
   split.
3. B1 and B2 use different scene panels. A scorer could in principle behave
   differently across panels, and the chain inherits that risk.

All three limits are reported with the result regardless of outcome.

## 6. Registered decision rule

Let `rho_1` be the Spearman correlation between G and closed-loop progress in
B1, and `rho_2` the Spearman correlation between R and G in B2. A valid proxy
requires `rho_1` strongly **negative** (less regret, more progress) and `rho_2`
strongly **positive** (the two one-step metrics agree).

- **VALID_PROXY.** `rho_1 <= -0.7` with bootstrap upper bound below `-0.3`,
  **and** `rho_2 >= 0.7` with bootstrap lower bound above `0.3`.
  Normalized rank regret is a usable cheap proxy. Keep it as the primary
  endpoint, governed by the ceiling-relative threshold the observability-ceiling
  assay derives.
- **INVALID_PROXY_AT_RANK_LINK.** `rho_1` passes but `rho_2 < 0.3` or its
  interval spans zero. A one-step metric does predict closed-loop utility, but
  **normalized rank regret is not that metric**. Replace the primary endpoint
  with geometric first-action regret in metres, and demote normalized rank
  regret to a diagnostic.
- **INVALID_PROXY_AT_CLOSED_LOOP_LINK.** `rho_1 > -0.3` or its interval spans
  zero. No one-step endpoint is established as predictive. Replace the primary
  endpoint with closed-loop progress against the oracle on development scenes,
  and demote both one-step metrics to diagnostics.
- **AMBIGUOUS.** Anything else. Report all endpoints separately; no endpoint
  change is made; every future gate must report all three.

No threshold may be relaxed, re-derived, or reinterpreted after any correlation
is observed. In every outcome, the §11 and §13 stops stand: this study can
change which endpoint *future* preregistrations use, and **cannot** reinstate,
rescue, or re-open any stopped mechanism.

## 7. Mandatory reporting

Regardless of outcome:

1. the full B1 and B2 tables of every endpoint;
2. the rank ordering under each endpoint, and every arm whose ranks disagree by
   two or more positions;
3. the three declared limits of §5;
4. the per-state scatter of R against G on the matched panel, and the count of
   states where the two metrics select different branches.

Item 4 is the mechanistic core: R and G can only diverge where the dense-rank
ordering and the metric progress ordering disagree, and the count of such states
localizes the disagreement.

## 8. Integrity gates

- the consumed V3 collection rehashes to
  `711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0`;
- `physical_oracle` R is exactly `0.0` and its G is exactly `0.0` on the matched
  panel;
- `random_expected` R reproduces the registered `0.4765170304232804`;
- every bound B1 value matches its source registered result exactly;
- byte-exact deterministic repeat of the complete analysis;
- exclusive output write; no overwrite, retry, or resume.

`geometric_endpoint` is **not** required to reach `R = 0`. It ranks by predicted
final distance, whereas the dense rank orders by `(fell, tipped, -progress,
path)`, so a nonzero value is expected and is itself part of the measurement.

## 9. What this does not authorize

No data generation, rendering, training, threshold relaxation, retry or resume
of any stopped mechanism, planner integration, deployment, promotion, or any
access to untouched, sealed, held-out, or V4 material.
