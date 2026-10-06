# Decision-headroom audit V4.2 — final result, 2026-09-25

**Decision-table outcome: measurement/coverage/precision limitation; scientific verdict inconclusive.** Neither predeclared filter-attribution condition A nor B is supported in either exposure stratum. No primary regret quantity establishes command-history sufficiency, a learned benefit, or a representation/predictor defect. Inconclusive regret supports neither stopping nor continuing visual ego-motion research. The authorized audit is complete and stops at checkpoint (b).

The point estimates favour command history over both learned heads, but all primary regret intervals cross zero and the ±0.02-s practical-effect region. These are local 800-ms cost differences, not measured mission-time savings. The secondary-reference findings below are descriptive and do not override the primary result.

## Primary results

Means and frozen 99.6154% intervals (1 − 0.05/13), with the number of contributing layout clusters. Rates are percentage points; regret quantities are seconds. Intervals are printed untruncated, including negative bounds for nonnegative quantities and bounds outside rate support. Such wide t intervals diagnose poor precision. `D_old` and `D_maze` are learned minus command-history regret: negative would favour learned prediction.

| Quantity | Exposed layouts 00–03 | Runtime-unexamined-at-freeze layouts 04–07 |
| --- | --- | --- |
| G | 0.1050 [-0.5120, 0.7221]; n=2 | 0.1083 [-0.5987, 0.8153]; n=3 |
| H_scorer | 0.1057 [-0.5149, 0.7263]; n=2 | 0.1010 [-0.7825, 0.9845]; n=3 |
| H_motion | -0.0007 [-1.2383, 1.2370]; n=2 | 0.0073 [-0.2428, 0.2574]; n=3 |
| D_old | 0.1218 [-6.7738, 7.0174]; n=2 | 0.2357 [-0.2180, 0.6893]; n=3 |
| D_maze | 0.0960 [-4.9909, 5.1830]; n=2 | 0.1808 [-0.5604, 0.9220]; n=3 |
| paired_filter/R4/old_data-R2/excluded_safe | 0.8753 [-7.5124, 9.2631]; n=4 | 2.3392 [-19.5873, 24.2656]; n=3 |
| paired_filter/R4/old_data-R2/all_excluded_despite_safe | 2.8167 [-17.3332, 22.9666]; n=4 | 3.8707 [-58.3620, 66.1033]; n=3 |
| paired_filter/R4/maze_data-R2/excluded_safe | 0.6713 [-4.0695, 5.4120]; n=4 | 2.0496 [-12.1515, 16.2507]; n=3 |
| paired_filter/R4/maze_data-R2/all_excluded_despite_safe | 1.0806 [-10.9687, 13.1299]; n=4 | 1.2409 [-42.9626, 45.4444]; n=3 |
| paired_filter/R5c-R2/excluded_safe | -1.0864 [-7.0525, 4.8797]; n=4 | -1.1010 [-34.1663, 31.9642]; n=3 |
| paired_filter/R5c-R2/all_excluded_despite_safe | -5.4053 [-36.3171, 25.5065]; n=4 | -9.4073 [-157.1020, 138.2875]; n=3 |
| filter/R2/operating/excluded_safe | 56.0785 [35.4620, 76.6949]; n=4 | 40.3500 [-190.2804, 270.9805]; n=3 |
| filter/R2/operating/all_excluded_despite_safe | 40.8185 [5.3023, 76.3347]; n=4 | 22.2222 [-163.7164, 208.1609]; n=3 |


`G` is command-history regret against the reference; `H_scorer` is regret with true motion; `H_motion` is command history minus true-motion regret. They use their frozen applicable populations; the separate common-mask decomposition below is used for telescoping, not subtraction of differently masked primary means.

## Completed scope and validity

24 fixed source assignments closed: 23 completed and case 13 (layout 04/reactive) unresolved after `measured visual pose unavailable`. Its 24 sampled identities and failure remain in the panel with missing audit outputs. There were no retries or substitutions. 552/576 sampled states have audit records (95.8%). All 552 passed source decision-input reproduction, physical restoration and RGB qualification. Source-replay RGB matches: 26,496/26,496. Candidate-repeat comparisons: 276. The 5,244 branches include restoration checks and predetermined repeats; they are not 5,244 independent decision states.

The sampled modes are 300 route-target, 117 terminal-target and 135 target-free view-seeking states. The latter remain in the filter audit and have no invented positional regret.

All 3,312 first-repeat bank candidate outcomes (six per audited state; 2,760 movement candidates) were classified safe under both the 5-mm/contact hard criterion and the 20-mm operating margin. No disallowed contact was recorded in those outcomes. Therefore unsafe-candidate discrimination is untested: zero admitted-unsafe observations do not establish a gate's sensitivity to danger.

Clearance uses all 27 collision primitives at native 2-ms steps, with feet/legs/body included and ordinary support contact excluded. The FK endpoint-displacement robustness calculation also passed for these rows. As frozen in V4, this is native discrete ground truth plus a reported interval robustness convention; it is not certification of arbitrary unobserved continuous articulated motion. Original analytical fixture failures, erratum and 13 passing boundary cases remain in the earlier Phase 1 records; analytical tests are not substituted for these trajectories.

The frozen harm panel reports zero means and degenerate [0,0] layout-t intervals for the tested selected actions. Zero observed harm and zero between-layout variation do not prove zero risk or certify the 5% absolute / 1% excess-harm limits with adequate rare-event uncertainty. No safety or learned-benefit claim is made from those intervals.

## Reference coverage and denominators

There are 417 positional states. Primary reference availability is 337/417 (80.82%); secondary availability is 375/417 (89.93%). These are raw counts, not pooled substitutes for the declared weighted estimator. The frozen quantity-coverage threshold is 90%; multiple source cells fall below it. Primary regret has complete three-source support in only two exposed and three initially unexamined layout clusters. Layouts 02 and 03 have reactive samples entirely in view-seeking mode; layout 04 lacks its reactive audit. No missing source is reweighted away.

The secondary reference restores some physically safe endpoints inside the 0.46-m inflation region by the approved nearest-free-cell path convention. It does not change actual candidate safety, controller eligibility, or the primary reference. Remaining bank-cost unresolved states are retained. Operating-margin cost is zero on every available reference here because every bank candidate satisfied the margin; rejecting margin-satisfying candidates is a different cost and remains in per-row `excess_rejection_cost_s`.

Endpoint-clearance coverage below is a raw candidate count over positional states, not a weighted performance estimator. All these candidate trajectories passed the articulated operating margin; endpoint strata refer to the separate conservative 0.46-m reference inflation. Reference-available counts mean that the candidate belongs to a state with an available bank optimum.

| Reference endpoint stratum | Candidates | Primary finite endpoint cost | Secondary finite endpoint cost | Primary reference available | Secondary reference available |
| --- | --- | --- | --- | --- | --- |
| reference_free | 2339 | 2320 | 2320 | 2022 | 2142 |
| inside_046m_inflation | 133 | 0 | 133 | 0 | 108 |
| outside_inflation_below_5mm | 30 | 0 | 0 | 0 | 0 |

| Layout | Source | Audited | Positional | Primary available | Secondary available | Weighted coverage primary / secondary |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | command history | 24 | 23 | 23 | 23 | 100.0% / 100.0% |
| 00 | reactive feedback | 24 | 16 | 16 | 16 | 100.0% / 100.0% |
| 00 | learned / old head | 24 | 19 | 2 | 2 | 10.5% / 10.5% |
| 01 | command history | 24 | 23 | 18 | 20 | 78.3% / 87.0% |
| 01 | reactive feedback | 24 | 14 | 14 | 14 | 100.0% / 100.0% |
| 01 | learned / old head | 24 | 24 | 11 | 11 | 45.8% / 45.8% |
| 02 | command history | 24 | 23 | 23 | 23 | 100.0% / 100.0% |
| 02 | reactive feedback | 24 | 0 | 0 | 0 | N/A / N/A |
| 02 | learned / old head | 24 | 16 | 10 | 14 | 62.5% / 87.5% |
| 03 | command history | 24 | 20 | 20 | 20 | 100.0% / 100.0% |
| 03 | reactive feedback | 24 | 0 | 0 | 0 | N/A / N/A |
| 03 | learned / old head | 24 | 19 | 19 | 19 | 100.0% / 100.0% |
| 04 | command history | 24 | 22 | 22 | 22 | 100.0% / 100.0% |
| 04 | reactive feedback | 0 | 0 | 0 | 0 | N/A / N/A |
| 04 | learned / old head | 24 | 21 | 21 | 21 | 100.0% / 100.0% |
| 05 | command history | 24 | 24 | 24 | 24 | 100.0% / 100.0% |
| 05 | reactive feedback | 24 | 20 | 20 | 20 | 100.0% / 100.0% |
| 05 | learned / old head | 24 | 24 | 3 | 23 | 12.5% / 95.8% |
| 06 | command history | 24 | 22 | 18 | 22 | 67.4% / 100.0% |
| 06 | reactive feedback | 24 | 3 | 2 | 3 | 66.7% / 100.0% |
| 06 | learned / old head | 24 | 16 | 5 | 12 | 31.2% / 75.0% |
| 07 | command history | 24 | 22 | 22 | 22 | 100.0% / 100.0% |
| 07 | reactive feedback | 24 | 23 | 23 | 23 | 100.0% / 100.0% |
| 07 | learned / old head | 24 | 23 | 21 | 21 | 77.4% / 77.4% |


The full endpoint-cost and reference availability breakdown by clearance stratum, hard/operating status and case is in `analysis_v42.json:reference_coverage_by_clearance`. The full state/source/phase/localisation weighted denominators and exclusion counts are in `analysis_v42.json:cells`. No imputed rows or modified common masks are used.

## Filter attribution and binding rules

True-motion R2 still excludes 56.08% of safe movement candidates on exposed layouts and 40.35% on initially unexamined layouts (layout means). It excludes every movement despite an available safe movement on 40.82% and 22.22% of states respectively. These totals include observation-only view restrictions as well as motion-dependent rules; the unrestricted totals cannot establish condition B.

| Scope | A: old head | A: maze head | B | Restricted R2 binding level, % [99.6154% interval] |
| --- | --- | --- | --- | --- |
| exposed | False | False | False | 15.6250 [-38.4230, 69.6730]; n=4 |
| runtime_unexamined_at_freeze | False | False | False | 11.1111 [-125.3305, 147.5528]; n=3 |


Neither learned-head complete-exclusion contrast has a lower bound above τ=10 percentage points. Neither the restricted R2 motion-binding level nor the R5c contrast clears B. This is failure to establish the predeclared attribution, not evidence that either mechanism is absent. The exploratory Stage A hold analysis motivated this component; it did not determine cost weights or select a favourable comparison scope.

## Descriptive diagnostics

The following 95% intervals use the same equal-source/layout estimator on the already-written per-cell quantities. The chain uses the identical R2/R3/R4/R5c state mask and weights within each head. `L_readout` and `L_forecast` are arithmetic pathway differences, not causal attributions. Action-derangement and shrinkage comparisons retain their own masks.

| Secondary quantity, seconds | Exposed | Initially unexamined |
| --- | --- | --- |
| chain/old_data/G | 0.1050 [0.0577, 0.1524]; n=2 | 0.1083 [-0.0809, 0.2975]; n=3 |
| chain/old_data/H_scorer | 0.1057 [0.0581, 0.1533]; n=2 | 0.1010 [-0.1354, 0.3374]; n=3 |
| chain/old_data/H_motion | -0.0007 [-0.0957, 0.0943]; n=2 | 0.0073 [-0.0596, 0.0742]; n=3 |
| chain/old_data/D | 0.1218 [-0.4075, 0.6512]; n=2 | 0.2357 [0.1143, 0.3571]; n=3 |
| chain/old_data/L_readout | 0.0977 [-0.7796, 0.9750]; n=2 | 0.3637 [-0.1981, 0.9256]; n=3 |
| chain/old_data/L_forecast | 0.0234 [-0.2295, 0.2764]; n=2 | -0.1208 [-0.6628, 0.4213]; n=3 |
| A_action_old | 0.3370 [-0.2990, 0.9729]; n=2 | 0.1600 [-0.4558, 0.7758]; n=3 |
| shrinkage_old | 0.0016 [-0.2189, 0.2221]; n=2 | 0.0745 [-0.0637, 0.2127]; n=3 |
| chain/maze_data/G | 0.1050 [0.0577, 0.1524]; n=2 | 0.1083 [-0.0809, 0.2975]; n=3 |
| chain/maze_data/H_scorer | 0.1057 [0.0581, 0.1533]; n=2 | 0.1010 [-0.1354, 0.3374]; n=3 |
| chain/maze_data/H_motion | -0.0007 [-0.0957, 0.0943]; n=2 | 0.0073 [-0.0596, 0.0742]; n=3 |
| chain/maze_data/D | 0.0960 [-0.2945, 0.4865]; n=2 | 0.1808 [-0.0175, 0.3792]; n=3 |
| chain/maze_data/L_readout | 0.1049 [-0.0739, 0.2837]; n=2 | 0.0951 [-0.0842, 0.2745]; n=3 |
| chain/maze_data/L_forecast | -0.0096 [-0.3163, 0.2971]; n=2 | 0.0930 [-0.2088, 0.3948]; n=3 |
| A_action_maze | 0.3272 [-0.2316, 0.8861]; n=2 | 0.2769 [-0.4901, 1.0440]; n=3 |
| shrinkage_maze | -0.0511 [-0.0719, -0.0303]; n=2 | 0.0430 [-0.1289, 0.2149]; n=3 |


| Secondary-reference regret, seconds (95%) | Exposed | Initially unexamined |
| --- | --- | --- |
| G | 0.1141 [-0.0482, 0.2764]; n=2 | 0.1221 [-0.1230, 0.3672]; n=3 |
| H_scorer | 0.1081 [0.0912, 0.1250]; n=2 | 0.1367 [-0.0934, 0.3668]; n=3 |
| H_motion | 0.0060 [-0.1732, 0.1852]; n=2 | -0.0146 [-0.1418, 0.1126]; n=3 |
| D_old | 0.1175 [-0.3572, 0.5922]; n=2 | 0.1801 [0.0315, 0.3287]; n=3 |
| D_maze | 0.0911 [-0.2367, 0.4189]; n=2 | 0.1336 [0.0442, 0.2231]; n=3 |


In the secondary reference the initially unexamined learned-minus-command-history intervals are positive for both heads. This is a descriptive conditional result, not a replacement for the inconclusive primary family or a representation-defect claim. Common-mask chain details, reactive-row bank membership, optimal-set membership, unnecessary holds and costs of eligibility rejection remain in the versioned per-state panel. No off-bank branch was required in this run.

### Per-motion-source filter levels (descriptive 95%)

| Motion source / quantity, % | Exposed | Initially unexamined |
| --- | --- | --- |
| R2/excluded_safe | 56.0785 [48.0404, 64.1166]; n=4 | 40.3500 [-21.3694, 102.0695]; n=3 |
| R2/all_excluded_despite_safe | 40.8185 [26.9712, 54.6659]; n=4 | 22.2222 [-27.5371, 71.9816]; n=3 |
| R2/admitted_unsafe | 0.0000 [0.0000, 0.0000]; n=3 | 0.0000 [0.0000, 0.0000]; n=3 |
| R5c/excluded_safe | 54.9920 [47.0406, 62.9435]; n=4 | 39.2490 [-25.7613, 104.2593]; n=3 |
| R5c/all_excluded_despite_safe | 35.4132 [15.3261, 55.5002]; n=4 | 12.8150 [-42.3234, 67.9533]; n=3 |
| R5c/admitted_unsafe | 0.0000 [0.0000, 0.0000]; n=3 | 0.0000 [0.0000, 0.0000]; n=3 |
| R4/old_data/excluded_safe | 56.9538 [47.1598, 66.7478]; n=4 | 42.6892 [-24.4643, 109.8428]; n=3 |
| R4/old_data/all_excluded_despite_safe | 43.6352 [24.6311, 62.6394]; n=4 | 26.0929 [-36.7374, 88.9232]; n=3 |
| R4/old_data/admitted_unsafe | 0.0000 [0.0000, 0.0000]; n=3 | 0.0000 [0.0000, 0.0000]; n=3 |
| R4/maze_data/excluded_safe | 56.7497 [47.8234, 65.6761]; n=4 | 42.3997 [-22.4855, 107.2849]; n=3 |
| R4/maze_data/all_excluded_despite_safe | 41.8991 [24.5051, 59.2932]; n=4 | 23.4631 [-33.0105, 79.9367]; n=3 |
| R4/maze_data/admitted_unsafe | 0.0000 [0.0000, 0.0000]; n=3 | 0.0000 [0.0000, 0.0000]; n=3 |


### Localisation covariate

| Raw audited-state stratum | Count |
| --- | --- |
| position/le_20mm | 552 |
| yaw/le_5deg | 552 |


The following saved descriptive panels retain their own source/layout support. Covariate stratification is observational and does not identify a localisation effect.

| Scope / position stratum | G, seconds (95%) | Old-head complete-exclusion difference, pp (95%) | Maze-head complete-exclusion difference, pp (95%) |
| --- | --- | --- | --- |
| exposed/le_20mm | 0.1050 [0.0577, 0.1524]; n=2 | 2.8167 [-5.0395, 10.6729]; n=4 | 1.0806 [-3.6173, 5.7785]; n=4 |
| exposed/20_to_100mm | unavailable (n=0) | unavailable (n=0) | unavailable (n=0) |
| exposed/gt_100mm | unavailable (n=0) | unavailable (n=0) | unavailable (n=0) |
| exposed/unresolved | unavailable (n=0) | unavailable (n=0) | unavailable (n=0) |
| runtime_unexamined_at_freeze/le_20mm | 0.1083 [-0.0809, 0.2975]; n=3 | 3.8707 [-12.7835, 20.5249]; n=3 | 1.2409 [-10.5885, 13.0703]; n=3 |
| runtime_unexamined_at_freeze/20_to_100mm | unavailable (n=0) | unavailable (n=0) | unavailable (n=0) |
| runtime_unexamined_at_freeze/gt_100mm | unavailable (n=0) | unavailable (n=0) | unavailable (n=0) |
| runtime_unexamined_at_freeze/unresolved | unavailable (n=0) | unavailable (n=0) | unavailable (n=0) |


Yaw strata and route/terminal/view and mission-phase panels are retained in the saved analysis. Absence of sufficient layouts in a stratum remains unavailable; no strata are pooled after seeing results.

## Stage A holds and historical training provenance

Stage A completed unchanged before this audit. These are exploratory counts from different trajectories, not isolated readout effects:

| Run | Hold plans | No eligible movement | Eligible but outscored/tied | Override | Insufficient |
| --- | --- | --- | --- | --- | --- |
| 00/old | 987 | 898 | 88 | 1 | 0 |
| 00/maze | 979 | 875 | 19 | 84 | 1 |
| 02/maze | 1140 | 1138 | 1 | 1 | 0 |
| 02/old | 957 | 614 | 97 | 244 | 2 |


Observation-only / motion-clearance / stopping-projection flags were respectively 1/899/2, 837/959/0, 1131/1139/1, and 293/859/12 in the same run order. These overlapping flags do not sum to holds. See [the unchanged hold analysis](go2_stage_a_holds_exploratory_2026-09-23.md) for dispatch categories and retained-evidence limitations.

The separately approved historical check selected 16 training examples (four per population), inspected 32 source frames and rendered zero frames: all examples lacked a retained bound restore packet for the qualified path. Its status is `COMPLETE_WITH_UNRESOLVED_INPUTS`, not a bitwise pass or demonstrated corruption. It did not gate this audit and no historical physics or regeneration was performed.

| Historical population/result with residual uncertainty | Current predictor | Old-data readout | Maze-data readout |
| --- | --- | --- | --- |
| Factorial predictor / V1.2 ancestry | Inherited checkpoint/training ancestry | No direct image/weight edge identified; deployment receives predictor forecasts | Same deployment dependence |
| Older native geometry-progress and moving-action-switch training | Direct training inputs | Actual RGB pairs and inherited motion-head weights | Same old-data ancestry |
| Full-heading continuation | No new predictor fitting | Inherited mixed-data head and heading pairs | Same initialization and old pairs |
| Maze-view training/recovery | Frozen; no updates | New maze images not optimizer examples | Direct maze-view training pairs |
| Near-goal / stalled-turn / frozen-native comparisons | Evaluation, not fitting from these assays | Evaluation, not training populations from those diagnoses | Same distinction |
| Original failed pilot branch futures | No training or comparative audit dependency | None | None |


The scoped source-path conclusions and exact historical result identities remain in [the provenance report](go2_renderer_provenance_readonly_2026-09-23.md) and [training-dependency addendum](go2_remaining_phase1_provenance_addendum_2026-09-23.md). Training render validity for the current predictor and both readouts remains unverified. Consequently pathway loss cannot be specifically attributed to representation defects. That caveat does not invalidate this frozen-model audit with newly qualified inputs/outcomes. It establishes neither historical corruption nor proven absence of exposure.

## Resource use and retained assets

| Quantity | Measured | Cap |
| --- | --- | --- |
| Source attempts | 24 | 24 |
| Sampled / audited states | 576 / 552 | 576 |
| Attempted branches | 5244 | 6096 |
| Source simulated seconds | 9251.68 | 11556 |
| Branch simulated seconds | 4195.2 | 4876.8 |
| Wall hours | 33.027 | 72 |
| CPU core-hours | 48.898 | 1152 |
| Peak sampled RAM GiB | 11.911 | 32 |
| Peak sampled VRAM GiB | 5.208 | 8 |
| Retained GiB at resource closeout | 17.683 | 20 |


No resource stop or JSON writer failure occurred. Resource peaks are sampled observations, not OS-enforced hard limits. The handover preserved the original assignment prefix and cumulative budgets. The final converter receipt covers successor writes; predecessor immediate checks were unchanged, but its in-memory receipt was not transferred. Source navigation outcomes are descriptive, not a new navigation cohort result.

The versioned branch panel is `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_decision_headroom_phase2_v4_attempt_001/branch_panel_v42.json` (SHA-256 `9d34bc510ac659f594a1e661a9ca12a4f21f73bcc614b64467f5e974d45a9817`). It indexes all 576 sampled identities, including 24 missing-output rows, and links the retained source packets, physical trajectories, render checks, row masks and costs. Large runtime artifacts remain outside Git. [The compact closeout JSON](go2_decision_headroom_audit_result_2026-09-25.json) binds the analysis, panel, resources and historical check by SHA-256.

Authority: [V4.2 protocol](go2_decision_headroom_protocol_v42_2026-09-23.json), [explicit approval](go2_decision_headroom_v42_approval_2026-09-23.json), and [monitoring-only handover amendment](go2_decision_headroom_v42_monitor_handover_2026-09-24.json). [Decision memo](go2_decision_headroom_decision_memo_2026-09-25.md). No follow-up experiment, fitting, gate repair, sample expansion or artifact retirement is executed.
