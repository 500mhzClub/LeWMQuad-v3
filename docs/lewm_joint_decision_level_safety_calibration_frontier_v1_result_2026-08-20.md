# Joint decision-level safety calibration frontier V1

Date: 2026-08-20

Starting source: `03de8a59eaeb87b50644ae528016803c4ce4e399`

Status: `POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC`

Preserved terminal: `MECHANISM_SPECIFIC_SAFETY_COMPOSITION_NO_SIGNAL`

## Classification

`SPECIALIST_SCORE_FRONTIER_NO_GO`

Joint threshold selection materially improved mobility relative to independently calibrated component thresholds, but it did not pass the frozen gate. More decisively, no threshold pair on the post-hoc held-out oracle frontier passed the complete gate. The bottleneck is therefore the joint frontier of the two specialist scores, not merely the earlier independent calibration rule or the eight-state calibration selection.

This remains a post-outcome diagnostic on an already inspected held-out panel. It cannot support an independent scientific claim.

Machine-readable result SHA-256: `6a8bfba4549721f418408979ee819675b29e358a9edd46eedeb3674a98b6265f`.

## Frozen evidence and reproduction

No checkpoint was opened or executed. The reducer consumed only the immutable row ledger:

- ledger SHA-256: `a28be7a1254a77b553730c3024fb6ef24ed914a64ebf8bae3458142e3b0f8a08`;
- decoded content digest: `e4e7ae1b494b171dd8a623a5368045a07f315e4ff05a85921b7e004c7d55e9de`;
- 576 branches, 48 states, twelve candidates per state;
- 384/96/96 fit/calibration/held-out rows;
- exact state, candidate, split, family, route outcome, and aggregate/component label alignment.

Before frontier enumeration, the reducer reproduced the committed specialist composition exactly, including every selected candidate:

| Metric | Reproduced value |
|---|---:|
| Aggregate unsafe recall | 0.9828 |
| Aggregate FNR | 0.0172 |
| Safe retention | 0.2895 |
| States retaining safe | 6/8 |
| Selected unsafe rate | 0 |
| Mean selected progress | 0.1944 m |
| Normalized regret | 0.3788 |
| Best-safe top-3 | 0.1250 |

The fixed specialists were unchanged:

- contact: `ENHANCED_EMBODIED` cumulative-contact probability;
- stuck: `ACTION_CONTROL_ONLY` cumulative-stuck probability.

Their persisted temperature scaling was unchanged. Only the two admission thresholds varied.

## Calibration frontier and selected rule

The calibration enumeration covered all 5,016 distinct decision pairs induced by the 96 calibration rows, including strict-admission boundary values. Of these, 793 met the preregistered component and aggregate safety feasibility constraints.

The frozen lexicographic rule selected:

- contact threshold: `0.47424158453941345`;
- stuck threshold: `0.5235316753387451`.

Equality rejects. A branch is admitted only when both probabilities are strictly below their thresholds.

Calibration performance at the selected pair:

| Metric | Value |
|---|---:|
| Rows, unsafe/safe | 96, 72/24 |
| Contact recall | 0.9800 |
| Stuck recall | 1.0000 |
| Aggregate recall / FNR | 0.9861 / 0.0139 |
| Admitted safe/unsafe | 6/1 |
| Safe retention | 0.2500 |
| States retaining safe | 4/8 |
| States admitting only unsafe | 0 |
| Selected unsafe | 0 |
| False abstentions | 3 |
| Selected progress | 0.1461 m |
| Normalized regret | 0.2972 |
| Best-safe top-3 | 0.2857 |

The small calibration panel was already highly conservative: only 24 safe calibration branches were available.

## Held-out primary result

The selected calibration pair was applied once to the eight held-out states.

| Metric | Joint calibration | Frozen independent thresholds |
|---|---:|---:|
| Contact recall | 1.0000 | 1.0000 |
| Stuck recall | 1.0000 | 1.0000 |
| Aggregate recall / FNR | 0.9828 / 0.0172 | 0.9828 / 0.0172 |
| Safe retention | **0.4474** | 0.2895 |
| Admitted safe/unsafe | 17/1 | 11/1 |
| States retaining safe | 6/8 | 6/8 |
| States admitting only unsafe | 0 | 0 |
| Selected unsafe | 0 | 0 |
| False abstentions | **2** | 2 |
| Mean selected progress | **0.2773 m** | 0.1944 m |
| Normalized regret | **0.1636** | 0.3788 |
| Best-safe top-1/top-3 | **0.5000 / 0.5000** | 0.1250 / 0.1250 |
| Selected-progress/oracle-progress ratio | 1.2698 | 0.8900 |

The progress ratio uses the frozen evaluator convention: mean over non-abstaining selections. It can exceed one because the joint filter abstained on two difficult states while the oracle mean includes all eight. It is not a mission-level progress-retention estimate and does not erase the two false abstentions.

The held-out gate passed recall, FNR, safe retention, component recall, state availability, selected safety, progress, regret, and family-presence checks. It failed:

- false abstentions: `2 > 1`;
- best-safe top-3: `0.50 < 0.75`.

### Per state

| State | Family | Admitted safe/unsafe | Selected | Safe | Progress (m) | Heading improvement | Abstain |
|---|---|---:|---:|---|---:|---:|---|
| purpose-10 | large enclosed | 0/0 | — | — | — | — | yes |
| purpose-11 | large enclosed | 1/0 | 11 | yes | 0.0258 | 1.79° | no |
| purpose-22 | medium enclosed | 4/0 | 0 | yes | 0.5338 | 11.35° | no |
| purpose-23 | medium enclosed | 4/0 | 0 | yes | 0.3753 | 7.93° | no |
| purpose-34 | small enclosed | 1/1 | 2 | yes | 0.2369 | 14.24° | no |
| purpose-35 | small enclosed | 4/0 | 0 | yes | 0.4722 | −9.50° | no |
| purpose-46 | loop alias stress | 3/0 | 11 | yes | 0.0198 | 6.80° | no |
| purpose-47 | loop alias stress | 0/0 | — | — | — | — | yes |

### Per family

| Family | Recall | Safe retention | Safe-retaining states | Selected progress | Regret | Top-3 | False abstentions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Large enclosed maze | 1.0000 | 0.2500 | 1/2 | 0.0258 m | 0.3167 | 0.0000 | 1 |
| Medium enclosed maze | 1.0000 | 0.5333 | 2/2 | 0.4546 m | 0.0000 | 1.0000 | 0 |
| Small enclosed maze | 0.9286 | 0.5000 | 2/2 | 0.3545 m | 0.3325 | 0.5000 | 0 |
| Loop alias stress | 1.0000 | 0.3333 | 1/2 | 0.0198 m | 0.0000 | 0.5000 | 1 |

No family completely collapsed, but the small-maze family fell below 0.95 aggregate recall and the large/loop families retained little useful motion.

## Held-out oracle threshold frontier

The 5,220-pair held-out enumeration is explicitly `POST_HOC_ORACLE_FRONTIER_DIAGNOSTIC`. It was not used to select the deployed thresholds.

| Frontier limit | Value | Important accompanying failure |
|---|---:|---|
| Maximum safe retention at recall ≥0.95 | 0.6842 | 2 false abstentions; top-3 0.50 |
| Maximum safe-retaining states at recall ≥0.95 | 7/8 | retention 0.2105; regret 0.6012 |
| Maximum top-3 at recall ≥0.95 | 0.6250 | selected one unsafe; one state admitted only unsafe |
| Minimum regret at recall ≥0.95 | 0.0000 | retained only 1/8 states |
| Maximum selected progress with zero unsafe selection | 0.4722 m | retained only 1/8 states |

There were 68 nondominated risk–retention points and 19 nondominated risk–progress points. No held-out pair passed the complete development gate. Thus, a different calibration data set might choose a different trade-off, but no threshold composition of these two scores supplies the required safety, mobility, and route-recovery operating point on this panel.

The complete calibration and held-out threshold arrays are persisted in `joint_threshold_frontiers_v1.npz`:

- SHA-256: `ff22f37bbc0a1cded81d0faff0ec82b66150d31410d6f278425d99a28fe2852e`;
- decoded content digest: `cbd25194a291401bb8cafc2c7bcdd66063a47b86303679e0563f804036d5a6c4`;
- bytes: 39,866.

## Exact uncertainty and distribution shift

The selected pair missed one unsafe branch in calibration and one in held-out:

- calibration FNR: 1/72; exact two-sided 95% Clopper–Pearson interval `[0.00035, 0.07497]`;
- held-out FNR: 1/58; exact two-sided 95% interval `[0.00044, 0.09236]`.

Descriptive 10,000-draw state bootstrap intervals were:

- fraction of states retaining a safe action: `[0.375, 1.000]`;
- false-abstention fraction: `[0.000, 0.625]`;
- progress with abstentions assigned zero: `[0.0674, 0.3576] m`.

Even after zero observed misses, at least 59 independent unsafe examples would be required for a one-sided 95% upper confidence bound of 0.05. Here a miss was observed, and branches within states are not independent. No finite-sample safety guarantee is claimed.

Contact scores shifted materially between calibration and held-out (all-row KS statistic `0.3646`; negative-row KS `0.4016`), whereas stuck scores were more stable (all-row KS `0.0313`). This supports calibration fragility as a secondary concern, but the failed held-out oracle frontier rules it out as the primary bottleneck.

## Decision

A safety filter is not useful merely because it attains high unsafe recall. It must retain enough safe actions to permit the safety-related task to proceed. Reject-all and near-reject-all behavior is safety-related task failure, not a successful safety result.

The recommended next experiment is a prospectively designed `FACTORISED_MICRO_SAFETY_WORLD_MODEL_V1`:

- separate contact/impact and stuck/motion-shortfall temporal states;
- separate losses and calibration;
- row-level evidence persistence before reduction;
- a fresh frozen fit/calibration/held-out panel;
- the unchanged deterministic kinematic route ranker.

This diagnostic grants no authority to train that model automatically.

## Runtime and custody

- reducer runtime: approximately 2.51 s;
- generated frontier/result storage: approximately 76 KiB;
- focused tests: 6 passed;
- models trained or fine-tuned: zero;
- checkpoint/model inference: none;
- simulation, rendering, and encoding: none;
- JEPA predictor access: none;
- specialist assignment, probabilities, temperatures, labels, splits, and kinematic route ranker changed: none.
