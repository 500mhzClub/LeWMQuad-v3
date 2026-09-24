# Control-commitment-aligned contact filter V1

## Decision

**`CONTACT_SCORE_NO_GO_ACROSS_CONTROL_HORIZONS`**

Secondary decision: **`CONTINUATION_RISK_RANKING_NO_SIGNAL`**.

This is a post-outcome development diagnostic. The target is only
`SIMULATED_DISALLOWED_CONTACT_PROXY`: robot/environment contact included by
the frozen maze label. It is not a material-hazard, injury, human-safety,
property-damage, or fragile-infrastructure result, and it is not a closed-loop
safety guarantee.

The predecessor result remains exactly scoped as follows:

> No threshold on cumulative contact probability through H3 provides the
> required safety–mobility operating point for
> `WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1`.

## Frozen evidence and reproduction

The analysis used the immutable Stage-1 row ledger (3,456 rows; 15 ticks per
branch; SHA-256
`ab47eb7848b980947ced6ee6f10493ef12578ab7871ef8ebdb97b46122617e9c`).
Checkpoint SHA-256
`3e556531a0442df214d0667ad42110e42806ec3aa7aa240c2b2746d7c304af31`
was checked as lineage only and was not opened or executed.

The committed H3 result reproduced exactly: AUC 0.931459, AP 0.934783,
recall/FNR 0.942149/0.057851, contact-negative retention 0.592814, 21/24
states retaining an action, one selected contact, one false abstention,
0.099315 m selected progress, 0.686211 oracle-progress fraction, 0.347353
normalized regret, and 0.500000 best-negative top-3. Persisted admission and
selected-candidate identities also matched exactly.

## Control commitment and stopping status

The candidate consists of four five-tick primitive blocks, but the deployed
MPC returns only `best_sequence[0]`. The closed-loop runner executes this one
primitive block, obtains a new observation, and replans. Therefore:

- committed window: one block = five 0.1 s command ticks = 0.5 s (H1);
- each command tick contains five 0.02 s policy steps and each policy step ten
  0.002 s physics steps, for 250 physics steps per committed block;
- candidate blocks 2–4 are replaceable at the next planning cycle;
- a zero-command `hold` primitive is available next cycle;
- hold remains subject to the per-command-tick slew limits (0.25 m/s `vx`,
  zero `vy`, 0.35 rad/s yaw) and is not a validated emergency brake.

The relevant source bindings are `lewm/planning/local_mpc.py:163`,
`scripts/benchmark_lewm_closed_loop_mpc.py:2020`,
`lewm_genesis/lewm_genesis/rollout.py:1`,
`config/go2_platform_manifest.yaml:64`,
`config/go2_primitive_registry.yaml:14`, and
`scripts/dev_action_slew_reconstruction_v1.py:56`.

H2 is classified **`CONSERVATIVE_UNVALIDATED_STOPPING_PROXY`**. No validated
stopping-distance/time envelope or brake test binds the ten-tick horizon.
Consequently H2 is diagnostic, not an authorised hard-safety horizon.

## First-contact timing

| Split | By H1 | First H1→H2 | First H2→H3 | No contact through H3 |
|---|---:|---:|---:|---:|
| Calibration (288) | 31 | 39 | 21 | 197 |
| Held-out (288) | 53 | 39 | 29 | 167 |
| Combined (576) | 84 | 78 | 50 | 364 |

All four intervals were represented in every family. Full per-family and
per-candidate counts are retained in the machine-readable result.

## Horizon discrimination

| Split | Horizon | Positives | AUC | AP |
|---|---|---:|---:|---:|
| Calibration | H1 | 31 | 0.853897 | 0.683006 |
| Calibration | H2 | 70 | 0.929948 | 0.875673 |
| Calibration | H3 | 91 | 0.942210 | 0.923324 |
| Held-out | H1 | 53 | 0.938739 | 0.855748 |
| Held-out | H2 | 92 | 0.939496 | 0.920767 |
| Held-out | H3 | 121 | 0.931459 | 0.934783 |

Held-out score correlations (H1/H2/H3 order) were 0.675395 for H1–H2,
0.563525 for H1–H3, and 0.838979 for H2–H3. The H1 loop-family result was
weak (AUC 0.719807, AP 0.386089), despite perfect large-family H1
discrimination. This family heterogeneity matters to the no-go.

## Calibration

H1 and H2 temperatures and thresholds were fitted/selected using only the 24
calibration states. H3 was preserved and not refitted.

| Horizon | Temperature | Threshold | Calibration recall | Negative retention | States retained |
|---|---:|---:|---:|---:|---:|
| H1 | 3.416571 | 0.0346251 | 0.967742 | 0.369650 | 18/24 |
| H2 | 3.580197 | 0.0650308 | 0.971429 | 0.669725 | 22/24 |
| H3 frozen | 3.770689 | 0.0692084 | 0.956044 | 0.538071 | 20/24 |

Strict admission was `probability < threshold`; a tie was rejected. Complete
H1 and H2 calibration frontiers are bound in the evidence index.

## Held-out hard filters

| Metric | H1 committed | H2 diagnostic | H3 frozen |
|---|---:|---:|---:|
| AUC / AP | 0.938739 / 0.855748 | 0.939496 / 0.920767 | 0.931459 / 0.934783 |
| Recall / FNR | 1.000000 / 0 | 0.956522 / 0.043478 | 0.942149 / 0.057851 |
| Contact-negative retention | 0.434043 | 0.658163 | 0.592814 |
| Admitted negative / positive | 102 / 0 | 129 / 4 | 99 / 7 |
| States retaining negative | 21/24 | 22/24 | 21/24 |
| Selected hard-window contacts | 0 | 0 | 1 |
| False abstentions | 3 | 1 | 1 |
| Selected progress (m) | 0.116135 | 0.123307 | 0.099315 |
| Oracle-progress fraction | 0.752281 | 0.786542 | 0.686211 |
| Normalized regret | 0.311714 | 0.251790 | 0.347353 |
| Best-negative top-1 / top-3 | 0.291667 / 0.416667 | 0.304348 / 0.521739 | 0.227273 / 0.500000 |

H1 passed eight of thirteen gates. It failed AP, retention, progress fraction,
regret, and top-3. H2 passed ten of thirteen but failed progress fraction,
regret, and top-3; it is additionally not a validated stopping horizon.

Per-family H1 retention was 0.5692 large, 0.2800 medium, 0.3725 small, and
0.4638 loop. Selected progress was 0.2863, 0.0381, 0.2025, and -0.0468 m,
respectively. Thus the H1 formulation did not merely miss an aggregate gate:
medium retention was poor and loop route progress collapsed.

## Continuation-risk ranking

H1 hard filtering selected six branches that contacted only later: two first
contacts during H1→H2 and four during H2→H3. The frozen, weight-free H2/H3
risk tie-break selected the same 2+4 continuation contacts. It increased
progress slightly from 0.116135 to 0.119525 m and reduced regret from 0.311714
to 0.299102, but top-3 fell from 0.416667 to 0.375000. Because it did not
reduce continuation contact, `CONTINUATION_RISK_RANKING_SIGNAL` did not pass.

## Oracle upper bounds

| Condition | Abstain | Progress (m) | Regret | Top-3 | Selected stuck |
|---|---:|---:|---:|---:|---:|
| Oracle H1 filter + kinematic | 0 | 0.154377 | 0.172975 | 0.833333 | 9 |
| Oracle H2 filter + kinematic | 1 | 0.156771 | 0.144590 | 0.869565 | 11 |
| Oracle H3 filter + kinematic | 2 | 0.144730 | 0.191065 | 0.818182 | 12 |
| Oracle H1 + oracle continuation tie-break | 0 | 0.160489 | 0.134245 | 0.833333 | 9 |

The candidate bank and kinematic ranker can satisfy the route metrics under
oracle horizon labels. The learned H1 admitted set cannot.

## Interpretation and closure

The selected-contact part of the H3 failure was a decision-horizon mismatch:
the one H3-selected contact first occurred during H2→H3 and H1 selected no
committed-window contact. The complete H3 no-go was not rescued. H1 remained
below the discrimination, retention, route-progress, regret, and recovery
gates, and did not meet the preregistered “narrow miss” condition.

Therefore the exact next decision is to close
`WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1`; its candidate-conditioned predictor
is not authorised. A successor must change architecture, proxy target, or
sensor coverage.

Hard operational safety must be assessed over the action block actually
committed, while later contact remains relevant continuation guidance. A
candidate that contacts only under replaceable blocks is not necessarily an
immediate violation, but the planner must re-evaluate after every executed
block. Nothing here establishes safe closed-loop execution.

