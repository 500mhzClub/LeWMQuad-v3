# Decision-headroom protocol V4 — frozen approval submission

**Execution requires Andrew's explicit approval of this version.** V4 implements the decisions on B1–B3 and replaces the conditional 30-second design. It preserves the original handoff's scientific question, frozen controllers/models, six-candidate bank, committed prefix, 800-ms horizon, source fidelity and strict RGB criteria. It introduces no new qualification category. No Phase 2 collection, comparative result analysis or historical training re-render has run.

## Interpretation and quantity-specific validity

Native simulator states at every **2-ms internal physics step** are the ground truth for this simulator audit. Positive continuous-time clearance between steps is not a prerequisite for the whole audit. Each candidate/state/quantity receives its own available, unsafe or unresolved status and reason. Missing R3 affects its contrasts, missing positional objectives affect regret, missing candidate motion affects that candidate's filter record, and an unresolved bank optimum affects that state's regret. Other states and quantities continue. Resource exhaustion and code/identity violations remain execution stops, not scientific failures of every row. No failures cause retries, state replacements or expansion.

The evaluator uses all 27 collision primitives, including feet/legs/body, from the frozen native asset and existing FK/support implementation. At every native step, wall-face projections give separation **lower bounds**, and points belonging to each primitive give separation **upper bounds**. A lower bound above a threshold certifies the sampled criterion; an upper bound below it establishes a sampled violation. An intervening bracket is unresolved, rather than pretending a conservative lower bound is the exact distance. Ordinary permitted support-ground contact is excluded by the existing reader; foot–maze and disallowed body contacts are retained. Contact identities and any conflict with positive maze separation are reported; ground and maze attribution stay distinct.

Report three distinct facts:

1. Disallowed native contact, and the separate **5-mm articulated hard-clearance** criterion.
2. The common **20-mm articulated operating margin**, evaluated against true maze geometry, independently of controller admission.
3. The unchanged controller eligibility under its observed geometry, objective, memory and substituted motion.

For the requested robustness check, let primitive k have world center c, world rotation Q and circumradius r. For each consecutive native step, compute `D_k = ||c_(t+1)-c_t|| + r_k * angle(Q_t^T Q_(t+1))`. Body rotation and joint motion both enter through the FK world transforms. Radii are half the box diagonal, sphere radius, or `sqrt(radius²+(length/2)²)` for cylinders. Store `min(sampled_lower_t, sampled_lower_(t+1)) - D_k` per primitive and interval. A candidate that passes the sampled threshold but fails only this interval check is **unresolved for that criterion**, not unsafe and not safe. This endpoint displacement envelope is the requested robustness convention; it does not claim an independently bounded unobserved continuous trajectory. No collision library is introduced.

Native contact is an observed fact even on incomplete branches. Positive safety requires a complete 401-sample horizon. Technical restoration uncertainty remains separate from observed contact. Failed comparisons and unresolved intervals stay in the panel; they do not halt otherwise supported quantities.

## Active objectives and reference regret

The source packet's **active objective** is either positional route following, positional terminal approach/settling, or target-free view seeking. Mission phase is separately outbound exploration, goal approach/settle, or return. Original captured target coordinates are preserved and anchored through the source's initial physical frame; no current true-pose correction, true-map target repair or future-observation substitution is allowed.

Regret is computed **only for positional objectives**, reported separately by objective mode, mission phase, source and layout exposure. View-seeking states have no scalar reference cost in V4 and enter the filter audit below. Exclusion of view states from regret is structural applicability, not a failed reference or a reason to drop those states from other quantities.

Keep the existing finite cost components and weights: true-grid geodesic travel at 0.2 m/s, heading alignment at 0.45 rad/s, opposing linear/angular braking at 0.4 m/s² and 1 rad/s², and phase-appropriate arrival settling. Keep the 20-mm grid and declared 0.46-m geodesic inflation as scalar-reference approximations. They do not replace articulated safety. Invalid targets, unreachable grid endpoints or incomplete required costs remain unresolved for scalar regret only.

Let P denote candidates passing the hard criterion, M the common operating margin, E a row's controller eligibility and C the finite non-safety cost. R1 minimizes C over P∩M. Report `min C(P∩M) − min C(P)` as operating-margin cost and `min C(P∩M∩E) − min C(P∩M)` as excess eligibility-rejection cost. Empty or unresolved sets produce an unavailable quantity. Unknown candidates that could change the optimum make this state's optimum unresolved; do not silently optimize a favourable known subset. A margin-violating selection has an explicit physical/margin classification, not a favourable regret against a conservative optimum. Harm and margin denominators are broader than finite-regret denominators.

The original strict sanity failures, post-execution two-label erratum and **13 passed analytic boundary cases** remain preserved. They establish their original stationary proxy predicates, not new articulated trajectory evidence. No cost or controller margin is tuned against the six check states.

## Primary filter audit

The exploratory Stage A logs motivate this addition: many holds had no eligible movement, including 898/987, 875/979, 1,138/1,140 and 614/957 holds in the four runs. Other holds arose from score choice or overrides. Those trajectories differed, so this is diagnostic motivation, not an isolated readout effect or evidence that excluded motion was safe.

At every representative state, recompute the deployed gates under **R4 predicted motion for each matched head, R5c command-history motion, and R2 true branch motion**. Hold observed map/pose, active objective, controller/recovery state, candidate tapes and observation-only scan/view restrictions fixed. Recompute the unchanged memory-clearance/reserve-recovery and stopping-projection rules for each substituted motion. Never copy R4's final mask across rows. The full selector is preserved separately. A movement actually admitted by an explicit deployed selection override is counted as admitted in the effective filter mask, while its base-gate exclusions and override are retained. A nominally empty mask followed by an admitted recovery movement is not reported as complete movement exclusion. This does not infer admission of other unchosen override alternatives. True contact, true clearance and true map never enter these scorers.

Every one of the five movement candidates receives its articulated hard and operating-margin classification, admission status, all exclusion reasons and binding rule. Binding order for exclusion accounting is observation scan/view subset, memory clearance (including the deployed reserve-recovery rule), then stopping projection. Record overlapping reasons too. Override/score effects are not relabelled as physical gate exclusions.

For each motion source/head and threshold, report weighted counts, unresolved counts and:

- **Excluded-but-safe rate:** excluded safe movement candidates / resolved safe movement candidates.
- **Admitted-but-unsafe rate:** admitted unsafe movement candidates / admitted movement candidates with resolved physical status. Also report admitted unsafe / all resolved unsafe and safe / all resolved excluded, so both conditioning directions are visible.
- **Complete movement exclusion despite safety:** states with all five movement candidates excluded and at least one safe / states where this event is determinate. An admitted movement proves the event false; otherwise missing gates or lack of a known safe candidate with unresolved candidates makes it unresolved.

Report binding-rule counts, observation-only versus motion-dependent exclusions, and mode/phase strata. “Physically safe but excluded” is not “useful for the view objective”: this round assigns no view utility. The primary filter panel uses the 20-mm operating margin; the 5-mm/contact panel is a separately labelled hard-safety diagnostic. Neither is an alternate definition of controller eligibility.

## Frozen rows and implementation

The audit-only adapters are `lewm/decision_headroom_v4_development.py`; collection, layout loading, branching and analysis have separate V4 source identities. They instantiate a private copy of the captured controller state and call the unchanged deployed selector. Only the stated motion tensor is substituted at the correction boundary. The source/model implementations remain unchanged.

| Row | Frozen implementation |
|---|---|
| R0 | Uniform among R5c-eligible candidates; per-state SHA-derived seed from 2026092308; unresolved if empty |
| R1 | Independent true-geometry physical-and-operating-margin optimum; canonical tie order |
| R2 | Native branch XY/yaw at eight horizons in the original body frame; no true contact/map enters selector |
| R2b, each head | R4 direction/yaw retained, translation norm replaced by true norm; norms below 1e-6 m or missing true motion remain unavailable |
| R3, each head | Same frozen encoder/preprocessing and readout on actual primary-camera branch futures |
| R4, each head | Same frozen context encoder, action-conditioned predictor and matched motion head |
| R4s, each head | Fixed suffix derangement [1,2,3,4,5,0]; common three-step prefix unchanged |
| R5c | Original fitted command-history motion on the exact source candidate command tapes |
| R5r | Original reactive selector; its projected tape compared against the bank at absolute 1e-7/component; separately branch an off-bank tape and report its signed gap |

All rows use canonical order hold, forward, left_arc, right_arc, left_turn, right_turn. R3/R4 share feature normalization and the current `512×384` encoder preprocessing of native `640×480` primary RGB. Dense features are transient. Frozen old-data and maze-data heads are both reported; no preferred head is selected. Reactive-source packets support R4 as well as all other applicable rows.

The six qualified states were used only for implementation checks: all six own-source selections reproduced; recomputed R4 old-head motion matched captured forecasts within 1e-6 absolute/relative; both heads' feature adapters had valid shapes; suffix derangement preserved the shared prefix; observation-only masks stayed identical; FK lower/upper ordering and interval formulas passed; quantity-local missingness left other rows/candidate gates available. No pilot comparative selections, regrets, filter rates or rankings were retained as scientific results. One initial missing observational model hook in the audit adapter was corrected; its failed implementation check is preserved. No deployed controller repair or new physics was performed. Mission-wide collection itself has not been run before approval.

## Frozen layouts and mission-wide sampling

The [layout manifest](go2_decision_headroom_v4_layouts_2026-09-23.json) contains exact scene geometry, asset/appearance/physics seeds, topology identities, exposure and future roles. Layouts 00–03 are the existing exposed dense cohort; 04–07 are newly constructed by the same generator, unexamined at runtime. Construction seed is 2026092307, new physics seeds 2026102800–03 and appearance seeds 2026102900–03. The generator's fixed structural rules reject duplicate topology/embedding against its explicit registry, including readout training and transfer layouts; no runtime outcome or model result enters construction. All structural rejections are retained. No later seed search/substitution is allowed. These are development audit layouts, not sealed final benchmarks.

Future panel roles: exposed 00–01 fit-eligible, exposed 02–03 selection, new 04–07 evaluation-only. V4 fits nothing. Runtime inspection during this audit ends any claim that the new layouts remain untouched for later model development.

Run each of three unchanged source controllers—command history, reactive, learned with the old-data head—once on each layout, in layout-major order: **24 mission-length sources**. Each runs to mission termination or **480 seconds after 1.5 seconds settling**. Record source navigation outcomes descriptively, never as a navigation cohort result. The existing 2-mm depth-noise treatment is unchanged; its four fixed recipe indices are assigned as audit layout index modulo four (0–3 repeated for new layouts), and original depth packet/noise hashes are retained.

Sample **24 representative states per source**, at most 576 overall. No additional diagnostic rollout or oversampling quota is added; motion-history, reversal and clearance tags describe these same states. Candidate decision frames are 12,16,…,4788, at most 1,195/source and 28,680 total. The last fixed frame leaves the full 800-ms source-image window before the 480-s source cap. Earlier mission termination can still truncate a sampled future; preserve that missing quantity without replacement.

Use a deterministic hash-priority reservoir (seed 2026092309, run identity, frame), keeping at most 24 provisional members per phase. Initial phase quotas are min(8,N_phase). Distribute unused slots one at a time in fixed outbound/goal/return order, up to available phase counts, until min(24,N_total) slots are assigned. This guarantees 24 when at least 24 eligible decisions exist without fabricating an unvisited return phase. Within phase, choose the smallest priorities; inclusion probability is n_phase/N_phase and weight N_phase/n_phase. Save every decision's phase, priority, admission and final membership. Failed serialization/restoration remains a sampled failure, not an opportunity to refill.

Phase labels use only captured mission state and observed terminal-approach/settling flags. Within a layout, source cells have equal weight; phase prevalence within source is restored by inclusion weights. Exposed and newly constructed strata remain separate. Diagnostic/mode subgroup counts are not silently pooled as balanced prevalence.

Snapshots are captured only for provisional reservoir admissions and held losslessly compressed in RAM (maximum 72 provisional members, 24 per phase). Only final members are written. Limit each combined snapshot/packet to 64 MiB raw and 16 MiB compressed. Native RGB acquisition and source inputs are unchanged; only final selected source windows are persisted, losslessly, with camera hashes/transforms. Full physical trajectories, contact identities, source command tapes and decision/mission logs remain. Discarding never-persisted provisional buffers is not artifact retirement. Existing artifacts are untouched.

## Branches, fidelity and local failures

Every sampled state receives three original source-trace replays, one before and two after candidates, plus one branch per candidate. The earliest selected representative frame in each layout/source cell is the predetermined spot-check: three repeats of every candidate and any off-bank reactive tape. Spot-check failures make the affected state/quantity unresolved; they do not trigger expansion or block the remaining audit. The identity is fixed before branch outcomes. This checks repeated execution without claiming universal determinism.

At each source packet, reproduce its own logged discrete selection, scores, gates and snapshot-time final limited command with the existing 1e-6 absolute/relative numeric tolerances. Preserve N/A quantities for reactive selectors. Current-time fidelity and replay of the subsequently recorded source tape are distinct; no later replan is invented from an earlier packet.

Keep strict source/repeat physical tolerances (1 mm and 0.1 degree with matching contact flags), and 48/48 source RGB bitwise comparisons per state. RGB failure disables actual-future feature quantities; physical restoration failure disables outcomes dependent on that restoration. Preserve valid original source-input evidence separately. Candidate-repeat variability is retained; on positional spot-checks, a repeated scalar-cost range over 0.025 s makes the reference/repeat quantity unresolved there. No tolerance widens after failure.

## Delta, masks, inference and inconclusive results

Replace δ=0.25 s with **δ=0.02 s** for primary regret/difference classifications. A mission has roughly 1,200 decisions at 400-ms cadence and about 600 non-overlapping 800-ms windows. Under the strong assumptions of sustained local loss, useful subsequent execution and no compensating/overlapping effects, 0.25 s per opportunity corresponds to 150–300 s, an implausibly permissive “negligible” threshold relative to 480 s. The proposed 0.02 s corresponds to 12–24 s (2.5–5% of the budget), or 4 mm at the reference travel speed. This motivates a stricter local margin; **summing regret is not a proven mission-time prediction**.

Keep numerical cost equality 1e-8 s and the existing **0.10-s per-decision practical near-tie** (20-mm reference grid scale). A local set-membership tolerance and a mean-effect equivalence margin answer different questions. The smaller δ does not make the grid/reference capable of resolving every 4-mm consequence; inadequate precision or sensitivity remains inconclusive. Continuous costs/differences are unchanged. Zero repeated variation is not evidence that microscopic effects matter.

Retain nominal G, H_scorer, H_motion, D_old and D_maze, with positional applicability and explicit paired state masks. The head-specific R2/R3/R4/R5c chain uses one shared population and identical weights; nominal G is also reported on its own broader supported population. Missing R3 makes L_readout/L_forecast unavailable on those states; it does not disable direct R4–R5c comparisons. Candidate eligibility is not a shared statistical mask.

The primary family has **17 quantities**: five regret contrasts and three operating-margin filter rates for four motion/head entries (R2, R5c, R4-old, R4-maze). Use layout-level estimates, equal source-cell weights, and two-sided t intervals with Bonferroni confidence `1−0.05/17 = 99.705882%` for the declared overall primary panel. Report all four layout values per exposure stratum. Mode/phase panels and hard-safety diagnostics use 95% descriptive intervals and are not selectively promoted. Rates retain raw counts and unclipped calculated intervals (bounded interpretation [0,1]). No decisions/repeats count as independent layouts.

Coverage is quantity-specific: report valid/applicable denominators and every unresolved reason. Retain 90% restoration/input coverage and 90% supported quantity coverage overall and per source/layout cell for claims about that quantity; structurally target-free decisions are not part of the regret denominator. A failed coverage condition limits that claim only. An absent mode is reported absent. Absolute hard-harm acceptance remains 5%, paired excess hard harm 1 percentage point, assessed with 95% layout-level intervals; zero observed harms do not prove a low risk. Report contacts and operating-margin violations separately. Filter counts diagnose rejection/admission behavior without being a learned-benefit claim. No general superiority conclusion follows from conditional regret alone.

With only four layouts per exposure stratum, this remains exploratory despite many more within-mission states. The prior δ=0.25-s precision scenarios were already weak; the smaller δ is harder to resolve. Inadequate precision yields an inconclusive result for that quantity, not an enlarged audit. No effect-dependent sample expansion, fitting or recommendation implementation is allowed.

## Proposed execution budget

Measured Phase 1 sources cost 156.85–162.42 wall seconds for 16.1 post-settle simulated seconds. Scaling at the slowest rate to 24×480 seconds gives **32.28 wall hours for source collection**. The new six-state adapter check took 97.97 seconds including frozen model loading, both heads' R3/R4 feature paths, selector checks, one FK trace/state and serialization; scaling that workload to 576 states is about 2.61 hours, not a measured complete-audit cost. The earlier 84 branches plus qualification overhead took 131.67 seconds beyond source collection, suggesting about 2.65 hours at 6,096 attempts. Allow the remaining approximately 10.5 hours for longer-run map costs, snapshot handling, full candidate FK bounds, reference grids, persistence and reporting. These estimates motivate the cap; they do not guarantee completion.

| Resource | Frozen proposed cap |
|---|---:|
| Layouts / source missions / representative states | 8 / 24 / 576 |
| Branch attempts, including source replay and off-bank spot checks | 6,096 |
| Source physics including all settling | 11,556 s |
| Branch physics / total physics | 4,876.8 / 16,432.8 s |
| Execution wall / aggregate CPU | **48 hours / 768 core-hours** |
| Aggregate RAM / total device VRAM | 32 GiB / 8 GiB |
| Retained output / peak additional footprint including caches | 20 GiB / 22 GiB |
| Filesystem reserves | RecoveryStorage ≥12 GiB; workspace ≥4 GiB |
| Concurrent native/GPU owners | 1 / 1 |

Maximum branches: `576×(3 source+6 bank+1 off-bank) + 24×(12 bank extras+2 off-bank extras) = 6,096`. Unused branches never buy additional states. New raw depth and dense-feature file allowance is zero. The owner reuses the existing budget meter, counts failed attempts, and preserves partial closeouts. Proposed retained envelope: at most 9 GiB final compressed bundles, about 5.25 GiB branch records at the maximum measured per-branch size, with 5.75 GiB for articulated records, selected source images, full source traces, logs and reports. Six combined packets measured 8.13–8.74 MiB compressed (38.88–41.79 MiB raw); later packets may be larger, hence explicit size limits. Compression affects newly written lossless storage only, not controller observations or historical artifacts.

At preparation, RecoveryStorage had about 35.55 GiB free and the workspace 5.50 GiB. The proposed 22-GiB peak plus 12-GiB reserve fits narrowly; recheck actual capacities at execution admission and keep monitoring. No deletion, storage expansion or alternate filesystem is implicitly authorized. The hard stop remains binding if the forecast proves insufficient.

The JSON configuration binds all caps, source/checkpoint identities, exact layout manifest and new adapters. Launch only after approval, using `scripts/run_go2_headroom_v4_development.py --approval <record.json>` under the existing Genesis environment. The approval record must carry the exact protocol JSON SHA-256, `phase2_explicitly_approved: true` and the user's approval text. No such record has been created. The audit root is exclusive-create and cannot restart completed assignments.

## Training-render provenance and proposed historical check

Carry **render-unverified training provenance** for the current predictor and both readouts. The [bounded provenance addendum](go2_remaining_phase1_provenance_addendum_2026-09-23.md) identifies legacy factorial predictor ancestry, older native predictor/readout populations, full-heading readout inputs and maze-view training. It distinguishes training from evaluation dependencies and does not claim demonstrated corruption or universal cleanliness. L_readout/L_forecast remain frozen-model pathway contrasts, not causal attribution to a representation/readout/predictor defect. Newly qualified audit inputs do not resolve training provenance.

Propose, **do not execute**, a separate historical bitwise re-render check: four hash-seeded training examples from each of (1) legacy factorial ancestor, (2) current predictor native training, (3) full-heading readout training and (4) maze-view readout training. Seed 2026092311; select by SHA of population/example ID from retained training manifests, before rendering or inspecting errors. At most **16 examples, 32 primary RGB frames** (current/future), zero physics advancement, zero regenerated datasets, no retries/substitutions. Require retained scene/asset/software/camera identities and sufficient recorded physical/render state. Missing reconstruction inputs are unresolved; do not replay a historical trajectory to obtain them. Restore through the corrected visual-cache path and use the identical capture/resize/colour/lossless-save pipeline without changing source images. Preserve bitwise equality, pixel distributions and all missing/failure records. A passing small sample does not certify all historical training data.

Proposed separate ceiling: 45 minutes wall, 12 core-hours, 16 GiB RAM, 8 GiB VRAM, 128 MiB retained, 256 MiB peak writes, the same 12/4-GiB reserves. This proposal has **no execution authority**, and its work is not charged to or silently included in Phase 2. It adds no current qualification gate.

## Completion and approval boundary

After approval, execute the fixed audit, report every assigned cell/state/row and quantity-specific failure, produce the versioned branch panel and one decision memo recommending a single next step, then stop. Source-run navigation outcomes remain descriptive. Neither successful implementation checks nor this freeze authorize Phase 2, historical re-rendering, fitting, controller changes, artifact retirement or the memo's recommendation.
