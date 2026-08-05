# Rank-regret metric-validity study V1 result

Date: 2026-08-05
Attempt: `go2_rank_regret_metric_validity_v1_attempt_v1`
Governing document: `..._preregistration_2026-08-05.md`

**Registered terminal: `VALID_PROXY`. Both registered links hold.**

Result identity SHA-256 prefix `70dec51fdfe22701c2405c6cbedb7d3a`.

---

## 1. Headline

Normalized rank regret **is** a valid proxy for closed-loop planning utility on
this stack. The suspicion that motivated this study — that the §11 and §13 gates
were taken on an endpoint that does not track what the project cares about — is
**not supported**. The endpoint choice is vindicated.

| link | statistic | Spearman | 95% interval | registered requirement | result |
|---|---|---:|---|---|---|
| B1 closed-loop | geometric regret G vs closed-loop progress | `-0.96364` | `[-1.00000, -0.69811]` | `rho <= -0.7`, upper `< -0.3` | **pass** |
| B2 rank | normalized rank regret R vs geometric regret G | `+0.98333` | `[+0.81651, +1.00000]` | `rho >= 0.7`, lower `> 0.3` | **pass** |

## 2. Part B1 — a one-step metric does predict closed-loop utility

Across the seven bound Aug-4 policies, geometric first-action regret in metres
predicts closed-loop progress at Spearman `-0.96364`, Pearson `-0.96516`, with
the whole interval below `-0.69`. Correct one-step candidate ranking converts to
closed-loop progress.

## 3. Part B2 — the two one-step metrics agree, same-state

On the identical 128 matched states and identical nine branches, with no
trajectory divergence:

| arm | R (rank regret) | G (metres) |
|---|---:|---:|
| `physical_oracle` | `0.00000` | `0.00056` |
| `geometric_endpoint` | `0.01665` | `0.00000` |
| `privileged_physical_successor` | `0.19001` | `0.00963` |
| `task_action_only` | `0.30036` | `0.01827` |
| `dinov2_true_successor` | `0.30884` | `0.02076` |
| `context_only` | `0.34569` | `0.02263` |
| `vjepa2_1_true_successor` | `0.35243` | `0.02374` |
| `bearing` | `0.42636` | `0.02734` |
| `hold` | `0.49851` | `0.04578` |

**The two metrics rank all nine arms identically.** No arm's position differs by
two or more places under the two orderings; the registered disagreement report is
empty.

The preregistered expectation for `geometric_endpoint` held: it scores `R =
0.01665` rather than zero, because the dense rank orders by
`(fell, tipped, -progress, path)` while G orders by progress alone. That
predicted, pre-declared non-zero is itself a check that the two metrics were
computed independently.

At state level the metrics disagree about the optimum on `14 / 128` states
(`10.9%`) — those where the rank-optimal and progress-optimal branch sets are
disjoint. Agreement is therefore high but not total, and the disagreement is
localized exactly where the theory says it must be.

## 4. What this resolves, and what it does not

**Resolved.** The DINO anomaly that motivated the study — a scorer
simultaneously failing its regret gate and matching the geometric oracle's 14/24
closed-loop successes — is **not** evidence of general metric invalidity. It was
a near-threshold miss (`0.01574` m against a required `0.020` m) on a
*policy-level* comparison over diverged states. The same-state analysis this
study performs shows the metrics agree at `rho = 0.983`. The Aug-4 result's own
caveat anticipated this, and the caveat was correct.

**Not resolved.** This study says nothing about whether the `0.13` absolute
threshold is *achievable*. A valid metric can still carry an unachievable
threshold. That question belongs to the observability-ceiling assay, whose
attempt failed its validity control and claimed no Outcome.

**Unchanged.** The §11 and §13 stops stand. This study changes only which
endpoint future preregistrations must use — and its answer is that the existing
endpoint was the right one.

## 5. Declared limits, reported as registered

1. B1 has seven scorers and B2 has nine arms. Both are powered to distinguish a
   strong monotone relationship from none, **not** to estimate a correlation
   precisely. The intervals are correspondingly wide, and B2's upper bound is at
   the `+1.0` boundary.
2. B1's geometric regret is measured on each policy's own diverged states; only
   B2 is same-state by construction.
3. B1 and B2 use different scene panels — the 24-scene development harness and
   the 32-scene V3 matched role.

One further limit, recorded because it is material: B2's learned arms
(`dinov2_true_successor`, `context_only`, `task_action_only`,
`privileged_physical_successor`, `vjepa2_1_true_successor`) are taken from the
observability-ceiling assay, whose registered Outcome was **void** under its
validity control. That voiding concerns the claim of a *ceiling*; it does not
affect the per-state selections used here, and B2's question — whether two
metrics agree across a diverse set of scorers — does not depend on any arm being
a valid ceiling. The arms serve only as spread along the metric range.

## 6. Integrity

All registered gates passed: collection rehash to
`711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0`,
`physical_oracle` R exactly `0.0`, `random_expected` R reproducing the registered
`0.4765170304232804` to `1e-12`, exclusive output write, no simulation, no
training, no encoder execution.
