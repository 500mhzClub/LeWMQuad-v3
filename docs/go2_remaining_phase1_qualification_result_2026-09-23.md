# Remaining Phase 1 qualification — final closeout, 23 September 2026

**The approved batch is complete and stopped with finite blockers. Phase 2 is not ready or authorized.** All four missing source/layout cells completed their fixed 84 branches. Source-decision reproduction, strict RGB restoration and candidate-repeat checks passed. Articulated clearance was measured at every native physics step, but continuous swept clearance remains unresolved. Five of the six source packets also lack the positional target required by the proposed reference cost. No comparative audit scores were computed.

This closes the approval attached to checkpoint commit `761ab695`. Its submitted protocol/configuration are preserved. The pre-execution successor [configuration](go2_decision_headroom_remaining_phase1_approved_v3_2026-09-23.json) has SHA-256 `3e468ee66d42938f558c882c7a90c8e44f1fbc28978f8036eda68d289f0a2578`. It fixes assignments, branch order, numerical tolerances, resource caps, 113 source bindings and 11 checkpoint/execution-input bindings. These were verified before execution. The restored renderer, controllers, models, 300-ms committed prefix and 800-ms branches remained unchanged. Only the approved qualification tooling and reporting were added.

## Completed population and strict results

| Layout / source | Frame | Own-source decision and current dispatch | New branches | Source RGB matches | Candidate repeat pairs | Positional reference target |
|---|---:|---|---:|---:|---:|---|
| 00 / command history, previous corrected state | 12 | Pass | 0 | Prior 48/48 | Prior 18/18 | Unavailable: initial view survey |
| 00 / reactive, previous corrected state | 132 | Pass | 0 | Prior 48/48 | Prior 18/18 | Unavailable: initial view survey |
| 00 / learned action | 132 | Pass | 21 | 48/48 | 18/18 | Unavailable: initial view survey |
| 02 / command history | 132 | Pass | 21 | 48/48 | 18/18 | Unavailable: footprint view seeking |
| 02 / reactive | 132 | Pass | 21 | 48/48 | 18/18 | Present: observed-map XY [1.225, 0.025] |
| 02 / learned action | 132 | Pass | 21 | 48/48 | 18/18 | Unavailable: footprint view seeking |

The two previous states received source-input replay only; their completed physics trials were not repeated. The four new collections each used exactly 161 camera frames, 805 policy steps, 17.6 simulated seconds including settling, and one frame-132 snapshot. Each then executed source replay 0, three repetitions of each of six candidates, and source replays 1 and 2. Eight new source replays therefore exercised restoration **after** candidate branches. There were no failed attempts, substitutions, retries, extended horizons or uncollected cells in this batch.

New totals: **192/192 source-replay RGB matches**, **72/72 candidate repeat pairs**, and **1,152/1,152 candidate-repeat image comparisons**. The maximum measured position and yaw discrepancies in the fixed restoration checks were both zero; contact flags agreed. Pixel MAE, RMSE, changed-pixel fraction and all reported absolute-error quantiles were zero. The recorded histograms put every channel in error bin zero. This establishes the strict image criterion on these assignments; it does not certify untested states or historical branch images. No additional feature diagnostics were run in this round.

The six-cell successor has 6/6 source-decision and corrected RGB coverage across two exposed layouts and three sources. This does **not** turn the original 24-slot pilot into a pass. Its eight-state physical evidence, 0/384 original RGB failure, other six states without corrected-path RGB qualification, uncollected slots and original coverage rule remain preserved.

## What source qualification establishes

Each packet was replayed through its own unchanged source selector. The comparison includes the complete recorded selection/eligibility and correction receipts, model input receipt, candidate requested/applied tapes, selected action, plan installation, snapshot-time dispatch and safety-limited command. Discrete decisions/schema had to be identical; numerical agreement used the predeclared `atol=1e-6, rtol=1e-6`. Only model wall-clock duration was excluded. All six reports have zero mismatches. Quantities not produced by a reactive final selector remain not applicable.

The check qualifies the captured packet, not only learned-source populations: the reactive packet can later support every applicable row, including R4. Only its own recorded source decision was reproduced here. No cross-method ranking, reference regret, new fitting or second-head comparison was executed.

Snapshot-time dispatch reproduction and replay of the recorded subsequent 800-ms applied tape establish different facts. The latter includes later command changes but does not regenerate subsequent replanning or obstacle vetoes from the initial packet. Such later decisions require their own causal observations. This limitation is explicit in each fidelity report; no future observation or true simulator pose was used to repair the packet.

Target inspection preserved the original packets. Five contain `NO_XY_ROUTE_TARGET`; their zero route vector must not be interpreted as a positional goal. The layout-02 reactive packet contains an observed-map target. Its presence alone does not qualify the physical reference, target-grid feasibility or scalar cost. No finite-cost comparison was attempted.

## Articulated geometry and unresolved intervals

All **27 native collision primitives**, including feet, legs and body, were bound to the existing URDF/FK support calculation in each new cell. Maximum binding errors were 1.31e-7 m position, zero dimension discrepancy and 1.86e-7 per rotation-matrix entry, below the predeclared 1e-5 identity tolerance. The native simulator used one internal substep per 2-ms recorded step.

For every robot primitive and maze wall, projection separation along wall-face axes gives a conservative lower bound on positive Euclidean separation. It is not an exact distance solver; overlapping projections would be unresolved rather than a demonstrated collision. Ordinary support ground is excluded according to the existing physical reader; feet against maze obstacles are included. Native disallowed-contact records and collision identities are retained independently.

| New cell | Traces, including original source trace | Native samples | Minimum sampled separation lower bound | Samples passing 5 / 20 mm | Unresolved between-step intervals |
|---|---:|---:|---:|---:|---:|
| 00 / learned | 22 | 8,822 | 176.735 mm | 8,822 / 8,822 | 8,800 |
| 02 / command history | 22 | 8,822 | 155.387 mm | 8,822 / 8,822 | 8,800 |
| 02 / reactive | 22 | 8,822 | 240.072 mm | 8,822 / 8,822 | 8,800 |
| 02 / learned | 22 | 8,822 | 200.794 mm | 8,822 / 8,822 | 8,800 |
| **Total** | **88** | **35,288** | **155.387 mm** | **35,288 / 35,288** | **35,200** |

There were zero native disallowed-contact samples, zero contact-flag/identity discrepancies and zero clearance/contact disagreements in these traces. This is a cross-check on non-contact trajectories, not empirical validation on a colliding articulated trajectory. The original analytical disallowed-contact case remains a separate test.

**None of the 35,200 between-step intervals is certified.** There is no independently verified upper bound on intervening base translation, body rotation and articulated point speed, nor an existing verified swept calculation covering them. Sampled velocities, endpoint displacement and repeat agreement cannot supply that guarantee. Large sampled gaps remain useful measured evidence but are not silently promoted to continuous acceptance. The two earlier corrected snapshots likewise have no articulated swept certificate. No general collision library or further repair cycle was started.

## Resource closeout

| Quantity | Measured approved execution | Binding limit |
|---|---:|---:|
| New sources / branches | 4 / 84 | 4 / 84 |
| Simulated seconds, including settling | 137.6 | 137.6 |
| Owner wall time | 766.85 s (12.78 min) | 3,600 s |
| Aggregate CPU | 1,131.65 s (0.314 core-hours) | 16 core-hours |
| Peak sampled aggregate RAM | 13.446 GiB | 16 GiB |
| Peak sampled total device VRAM | 5.064 GiB | 8 GiB |
| Retained output at owner closeout | 0.350 GiB | 2 GiB |
| Peak sampled additional footprint, including cache growth | 0.351 GiB | 3 GiB |
| Minimum free RecoveryStorage / workspace | 35.568 / 5.499 GiB | At least 12 / 4 GiB |

There was one execution owner, no resource stop and no error. Memory and footprint are sampled measurements, not OS-enforced instantaneous maxima. Device VRAM includes other users. “Peak additional writes” follows the approved meter's retained-output plus cache-growth footprint; it is not cumulative block-device write traffic. Small closeout/report writes follow owner metering. Read-only preparation and final report consolidation are outside the owner wall/CPU totals. No artifact retirement occurred.

Source collection alone cost 635.18 wall seconds; the remaining 131.67 seconds covered branch work and qualification overhead. At the slowest measured source rate, the former 24 × 60-s future source proposal would consume approximately 4.04 hours **before** branches. The [conditional sampling/budget closure](go2_remaining_phase1_reference_sampling_closure_2026-09-23.md) therefore proposes 30-s sources under the same four-hour maximum. Rough scaling gives 2.02 hours for sources and 0.77 hours for 1,776 branches plus this round's amortized qualification overhead, leaving about 1.21 hours for additional audit encoding/scoring/persistence. Those are forecasts, not measured Phase 2 costs or a qualified launch budget.

## Explicitly retained Stage A and provenance analyses

Stage A remains complete and unchanged: four runs, 1,198 selected plans each, 480.32 simulated seconds each, no goal/home arrivals, disallowed contacts or pipeline faults. The existing read-only hold analysis is exploratory:

| Layout / head | Holds | Movement unavailable by eligibility | Movement eligible but outscored/tied | Recovery/dispatch override | Insufficient evidence |
|---|---:|---:|---:|---:|---:|
| 00 / old data | 987 | 898 | 88 | 1 | 0 |
| 00 / maze data | 979 | 875 | 19 | 84 | 1 |
| 02 / maze data | 1,140 | 1,138 | 1 | 1 | 0 |
| 02 / old data | 957 | 614 | 97 | 244 | 2 |

No exact score ties occurred in the classified score-loss group. Overrides were stopping projection in 1/84/0/12 cases and blocked latched recovery turns in 0/0/1/232 cases, respectively.

| Layout / head | Observation-only scan/view restriction | Prediction-dependent memory-clearance exclusion | Stopping-projection exclusion |
|---|---:|---:|---:|
| 00 / old data | 1 | 899 | 2 |
| 00 / maze data | 837 | 959 | 0 |
| 02 / maze data | 1,131 | 1,139 | 1 |
| 02 / old data | 293 | 859 | 12 |

These second-table flags overlap. Stopping flags count the preceding scan-eligible candidate subset, while an override can act outside it. They therefore need not equal override totals. Other observation-validity exclusions are not fully reconstructible. Different trajectories prevent an isolated causal readout interpretation. These counts document diagnostic coverage and did not tune the reference cost. The [original hold report](go2_stage_a_holds_exploratory_2026-09-23.md) preserves dispatch counts and input bindings.

The [historical provenance addendum](go2_remaining_phase1_provenance_addendum_2026-09-23.md) identifies residual uncertainty and training/evaluation edges explicitly:

- The factorial RGB rollout checkpoint at seed 2026080901/epoch 21 is an **inherited training ancestor of the current predictor**. Historical population-wide rendering/source binding remains unresolved. The September 36-context frozen-native branch assay evaluates those predictors; that assay is not a fitting population.
- Geometry-progress and moving-action-switch collections supply **current predictor training and actual image-pair training for both readouts**. This bounded review did not re-establish every historical runtime/render binding.
- Full-heading collections supply **both readout initializations/continuations**, not current predictor updates. The inspected fresh, forward-only collector has no identified rewind path, but this is not population-wide historical certification.
- Maze-view examples train **the matched maze-data head**, not the old-data arm or frozen predictor. Shared image caches do not establish an optimizer-input edge to the old-data head. Complete historical rendering qualification remains unresolved.
- Near-goal and stalled-turn/local-control results are **diagnostic evaluation, not fitting inputs** to the current heads. The failed original pilot's branch futures supplied neither fitting data nor comparative audit evaluation. The known August waypoint floor-only defect has no identified edge to the inspected current recipes; absence of an identified edge is not universal exclusion.

The original [renderer provenance review](go2_renderer_provenance_readonly_2026-09-23.md) retains path-specific restoration/reset classifications, including current source-image collection. Uncertainty is neither demonstrated corruption nor proven absence of exposure. Potentially compromised inherited training would confound attribution of a loss to representation defects; it would not automatically invalidate an audit of the frozen deployed model on newly qualified inputs/outcomes. No historical rerun or regeneration occurred.

## Protocol closure and finite blockers

The [reference and sampling closure](go2_remaining_phase1_reference_sampling_closure_2026-09-23.md) preserves the original cost formula/weights and separates disallowed contact plus the 5-mm articulated hard-clearance criterion, a common true-geometry 20-mm operating margin, and controller eligibility. It specifies separate margin cost and excess rejection cost. The former 0.48-m center-distance proxy does not certify the articulated margin. Original failed sanity assertions and their erratum remain beside the 13 passed analytical boundary cases; those qualify stationary proxy predicates, not moving articulated clearance.

Observation-only exclusions remain frozen. Future motion-dependent masks must be recomputed using each row's substituted motion and the unchanged deployed rule. They are distinct from the shared state sets/weights for paired statistics. R4's final candidate mask is not copied to other rows. Numerical equality is 1e-8 s; practical near-ties are 0.10 s (20 mm at 0.2 m/s), with a separate 0.025-s repeat-range limit and δ=0.25 s usefulness threshold. Continuous cost differences remain unchanged.

Conditional Phase 2 quotas are eight layouts, three sources each, four representative plus at most two diagnostic states per cell, at most 144 unique states and 1,776 branches. The proposed 30-s windows give at most 2,176.8 simulated seconds including settling. Predetermined repeat spot-checks, fixed reservoir/seed rules and the 4-hour/64-core-hour/24-GiB-RAM/8-GiB-VRAM/16-GiB-retained/20-GiB-peak envelope are documented. These are proposed design bounds, not execution authority. With four layouts per exposure stratum, precision is explicitly exploratory; original coverage, harm and inconclusive rules remain, with no effect-dependent expansion.

| Blocker | Affected rows / required quantities | What remains valid |
|---|---|---|
| **B1: No continuous articulated clearance certificate** | R1 physical-and-margin optimum; all reference regrets, margin/rejection costs and primary/decomposition cost contrasts. All four new cells have unresolved intervals; the earlier two lack this certificate too. | Measured 2-ms conservative geometry, native contact records, source decisions, physical and RGB restoration. No-contact observations are not a hard-clearance pass. |
| **B2: Positional reference target absent in 5/6 packets** | Finite travel/geodesic reference and corresponding regret for those five view-seeking packets; route-following reference coverage remains limited. | Faithful source decisions and original target semantics. The reactive layout-02 packet has a positional target, but B1 still applies; it is not a qualified optimum. |
| **B3: Phase 2 executable package/reduced-cost-repeat evidence incomplete** | Audit launch: unexamined layout identities and row adapters are not frozen; scalar-reference repeat validity is unavailable under B1/B2. | The fixed conditional design, explicit masks/thresholds, measured resource forecast and exploratory precision assumptions. Zero pose/pixel repeat variation is preserved without extrapolating full determinism. |

R3 is **not independently blocked by RGB restoration on these six states**. No method rows were scored; R3-dependent L_readout/L_forecast and other reference-cost contrasts are unavailable because the physical/reference prerequisites remain unqualified, not because missing rows were imputed or all sources were restricted to learned-origin states. A renderer-only failure would not automatically invalidate otherwise qualified rows, but the present physical-reference blocker prevents claiming a qualified primary comparison.

No additional qualification, renderer patch, controller repair, source substitution or budget extension follows this closeout. All snapshots, source packets, command tapes, camera/asset/software identities, physical trajectories, comparisons and unresolved-interval records are retained. Later rendering is potentially recoverable, not guaranteed, and would require its own fidelity checks.

[Machine-readable results and evidence hashes](go2_remaining_phase1_qualification_result_2026-09-23.json) identify the single output root `go2_decision_headroom_phase1_remaining_v2_attempt_001` under the existing RecoveryStorage development-artifact directory. This is the requested finite-blocker endpoint. **Stopped; no Phase 2 collection or comparative scoring.**
