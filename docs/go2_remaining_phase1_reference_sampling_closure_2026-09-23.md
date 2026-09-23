# Reference, masks and sampling closure after remaining Phase 1

This document specifies interpretation and a conditional design. It does not authorize Phase 2, compute comparative selections/regret, change the approved four-cell batch, or qualify a physical reference that lacks a swept-clearance bound. The numerical source-replay rules were frozen in `go2_decision_headroom_remaining_phase1_approved_v3_2026-09-23.json` before the first comparison.

## Source fidelity and distinct masks

For each captured packet, qualification reruns **that source's own** deployed selector with the frozen old-data head, captured native RGB/body/command history, observed map/pose, target and pre-selection controller state. All recorded scores, selection/eligibility receipts, correction receipts, six requested/limited tapes and the selected action are compared. Discrete/schema decisions must be identical; numerical tolerance is absolute 1e−6 plus relative 1e−6, fixed before execution. Only model wall-clock duration is excluded as non-decision metadata. The same deployed plan-installation and current-time request/limiter logic is called; no later observation or true pose repairs the packet.

Current-time dispatch fidelity is not a replay of every later replanning decision. The 800-ms source restoration separately executes the **recorded applied command trace**, including later changes. Those two tests establish different facts. A later veto based on a newly acquired observation cannot be inferred from the initial packet; claims about that later dispatch need its own causal log. Future audit branches retain the declared fixed candidate tapes, prefix and horizon rather than silently substituting subsequent source replanning.

A qualified reactive-source packet is a valid source population for every applicable later row, including R4. It is not restricted to a reactive row. The present exception reproduces only its recorded reactive decision; it does not compare a counterfactual learned choice there. R4 on learned-source packets must reproduce deployed learned selection; its second-head/substituted-motion evaluations remain Phase 2 work.

Freeze observation-only exclusions and source state. For each future row, run the **unchanged** deployed motion-dependent clearance, recovery and stopping checks on that row's substituted motion. Save each candidate's mask and reason. Never copy R4's final mask to R2/R2b/R3/R4s. True motion or actual-future features are limited to the specified substitution; true map, future contact and true future clearance never enter the controller scorer. R1 alone uses true geometry as the independent reference.

Separately, statistical paired masks select common **states**, with the same original sampling weights. For each head the R2/R3/R4/R5c decomposition uses one shared state intersection and G is recomputed on it. Row-specific candidate eligibility is not a statistical pairing rule. Missing rows are neither holds nor imputed estimates; every excluded denominator/reason is reported. If only R3 is unavailable, L_readout/L_forecast are unavailable, while independently qualified direct contrasts can remain. A missing physical reference affects every regret-based primary comparison.

## Physical definitions and cost decomposition

Report three distinct outcomes:

- **Physical harm:** attributed disallowed contact, including feet against maze obstacles and disallowed body/ground contact under the existing reader; ordinary permitted support-ground contacts are excluded. The separate declared hard-clearance criterion is at least **5 mm** between all robot collision surfaces and maze geometry throughout the branch. Unknown continuous clearance remains unknown, not a no-harm certificate.
- **Operating margin:** a common **20-mm articulated surface-to-maze clearance** throughout the branch, applied equally to all candidates and evaluated against true swept geometry. This carries forward the proposal's 20-mm allowance beyond its nominal 0.46-m disk. The former 0.48-m center-distance convention is an alignment proxy, not proof about moving legs/body. The controller's intended 0.45-m nominal radius / 0.03-m translation reserve and recovery exceptions remain unchanged and are reported separately.
- **Controller eligibility:** the candidates admitted by observed geometry, forecasts and deployed rules. Neither admission nor rejection establishes true physical acceptability or the operating margin.

Let P be physically acceptable candidates, M the candidates satisfying the common operating margin, E a row's controller eligibility, and C the unchanged finite non-safety reference cost. The primary reference is min C over P∩M. Report margin cost `min(P∩M) − min(P)` separately from excess rejection cost `min(P∩M∩E) − min(P∩M)`. Use one fixed C and identical state inputs for these differences. Any empty set, unresolved physical clearance or unavailable scalar cost makes that quantity undefined. A margin-violating selected action is explicitly classified as such and is not credited with a favourable primary regret against a more conservative optimum. Contact/harm and operating-margin violation rates retain their wider declared populations and denominators.

The reference's travel/heading/braking/settling formula, weights, 20-mm grid and original initial-frame target anchoring remain unchanged. Its disk-grid feasibility is a separate approximation and may make C unavailable even for physically harmless motion. Missing positional route targets are not repaired with simulator truth or an invented goal. All original 24-case failures and the two-label post-execution erratum remain alongside the 13 passed cases. The latter test stationary analytic wall distances at 0.45/0.46/0.465/0.48 m center distance and a contact flag; **they do not validate articulated trajectories or between-step clearance**.

Keep continuous C and regret differences unchanged. Numerical cost equality is 1e−8 s; practical near-tie is **0.10 s**, motivated by 20 mm at 0.2 m/s or 0.045 rad at 0.45 rad/s. Scientific usefulness remains δ=0.25 s. Legitimate repeat ranges are reported separately; uncertainty crossing the practical classification boundary makes it indeterminate. A repeat-cost range above 0.025 s prevents the proposed reduced-repeat qualification; no tolerance is widened after a failure. Zero repeated pose/pixel variation does not eliminate practical near ties, nor establish scalar-cost repeat validity where the physical reference is unavailable.

## Fixed conditional audit quotas and maximum budget

Retain **eight layouts**, separated into four exposed dense-cohort layouts and four previously unexamined layouts from the same generator. Retain three source controllers per layout, **four representative plus at most two diagnostic states per cell**, at most **144 distinct states**. No new layout is generated by this closure. The V2 construction/physics/appearance seeds remain 2026092307 / 2026102800 / 2026102900; exact generated identities still need freezing before any Phase 2 collection, with no runtime-based rejection/substitution.

Revise the proposed source window from 60 to **30 seconds after 1.5 seconds settling**, solely for the measured resource budget. The four approved source collections cost 162.42, 158.49, 156.85 and 157.43 wall seconds for 16.1 post-settling simulated seconds, before branch work. Linear conservative scaling of 24 sources at the slower rate would consume about **4.04 hours for 60-second sources alone**, exceeding the existing four-hour cap before branching/scoring. The 30-second choice leaves about two hours for branches, encoding, references, persistence and headroom. This is a runtime design decision, not an effect-dependent sample change. It may reduce mission/target coverage; those cells remain missing rather than triggering extensions.

Eligible planning frames become 12,16,…,292: **71 reservoir opportunities per source, 1,704 total**. Keep the original seed 2026092309 and streaming uniform representative reservoir (n=min(4,N), probability n/N, weight N/n), plus separate one-slot translation-history and turn/reversal-history reservoirs selected only from observable history. Freeze observations, clearance-bin labels and mission phase before branch outcomes; report overlap and branch each retained state once. No extra rollouts fill empty bins. Retain at most six snapshot/packet objects per source in RAM, each at most 64 MiB serialized, writing only final members. There is no artifact retirement.

Reduced physics repeats are only a **conditional proposal**: every sampled state gets three source traces (one before, two after candidates); ordinary states get one branch per candidate. The first representative state by timestamp in each layout/source cell is predetermined for three repetitions of all six candidates and any separate off-bank reactive tape. The approved Phase 1 batch still uses all three repeats unchanged. No saved repeats add states. Any spot-check failure stops without retry or repeat expansion. These sampled checks do not prove universal determinism, and unresolved scalar-reference repeat qualification prevents promoting this to an executable audit.

| Conditional Phase 2 bound | Fixed proposed maximum |
|---|---:|
| Layouts / source windows / state union | 8 / 24 / 144 |
| Branch attempts, including source and off-bank spot-check repeats | 1,776 |
| Total simulated seconds including settling | 2,176.8 |
| Execution wall / aggregate CPU | 4 hours / 64 core-hours |
| Aggregate RAM / total device VRAM | 24 GiB / 8 GiB |
| Retained / peak additional writes including caches | 16 GiB / 20 GiB |
| Filesystem reserves | RecoveryStorage ≥12 GiB; workspace ≥4 GiB |

The original storage envelope remains 9 GiB snapshots/packets, 3.5 GiB branch records, 3 GiB source recordings and 0.5 GiB scalar/report evidence. Shorter source windows reduce projected writes; they do not enlarge another category. No raw depth or dense feature tensors are retained. Snapshot, command, camera, asset/software and physical trajectory evidence remains available for possible later work, without guaranteeing replayability.

## Precision and inconclusive rule

This is explicitly **exploratory**. Four independent layouts per exposure stratum cannot generally support the desired δ/2=0.125-s precision. With assumed layout standard deviations 0.25 / 0.5 / 1 s, a 99% t interval with three degrees of freedom has approximate half-width 0.73 / 1.46 / 2.92 s. These are assumptions, not observed pilot method-regret variance. Never pool exposed and unexamined layouts to conceal uncertainty, or count correlated decisions/repeats as independent layouts.

Keep five primary contrasts with 99% individual intervals and the declared shared state/weight rules, 95% descriptive secondary intervals, 90% coverage overall **and per layout/source cell**, and 90% reference/paired-score coverage among restored states. With four representative slots, per-cell coverage effectively requires four valid slots. Keep the 5% absolute harm and 1-percentage-point paired excess limits. Four layouts with zero observed layout-any-harm events still have a one-sided 95% binomial upper bound about **52.7%**; that is a layout-event quantity, not a per-decision risk bound. Inadequate precision or harm bounds remain inconclusive. No effect-dependent expansion is allowed.

This closes the declared choices without pretending the physical-reference blocker is solved. The qualification result must finish with a finite blocker list, specific affected rows/quantities and what remains valid. No additional repair or qualification cycle follows automatically.
