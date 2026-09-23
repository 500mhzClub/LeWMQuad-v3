# Decision-headroom protocol V2 — checkpoint (a)

**Frozen proposal, not execution authority. Phase 2 is not yet technically qualified.** This revision incorporates the user’s two-state recheck amendment and supersedes unresolved choices in the earlier protocol draft. The final handoff remains authoritative. This document and its adjacent hashed JSON identify the proposed next work; no task below has been launched by writing or committing them.

The completed amendment comprises the [strict recheck](go2_decision_headroom_rgb_recheck_result_v2_2026-09-23.md), [renderer provenance review](go2_renderer_provenance_readonly_2026-09-23.md), [exploratory Stage A holds](go2_stage_a_holds_exploratory_2026-09-23.md) and [13-case boundary panel](go2_decision_headroom_clearance_boundary_result_v2_2026-09-23.json). Stage A remains complete and unchanged. No fitting, controller repair, candidate expansion, additional navigation cohort, sensor change or artifact retirement is part of this proposal.

## What is and is not qualified

The original pilot attempted eight states from command-history and reactive sources on layout 00. All passed its physical pose/contact restoration and candidate-repeat checks; the original RGB restoration failed. The corrected recheck passed on command-history/frame 12 and reactive/frame 132, including source replay after candidate branches. These two states are members of the original eight, not two extra independent states.

Missing corrected-path coverage includes the action-conditioned source and every layout-02 source. The other six original states have not had their branch RGB qualified under the correction. The original 24-slot pilot therefore remains incomplete/failed against its original 90% rule. We do not retrospectively lower that rule, relabel uncollected slots as passes, or claim full determinism from the first eight states. A separate successor qualification would have its own explicit denominator.

The remaining limiting issue is not another proposed renderer patch. It is source/layout coverage and the physical meaning of the clearance reference. A Phase 2 approval should not be inferred from a renderer pass while those remain open.

## Clearance, harm and reference cost

The current saved clearance measures a **0.46-m disk about the base XY origin** against true wall boxes. It samples native physics every 2 ms. Its segment lower bound is the smaller endpoint clearance minus half the endpoint displacement: the distance field is 1-Lipschitz along the straight interpolated center segment. The reference-cost routine uses the minimum sampled clearance minus half the largest step, a more conservative whole-trace version of that bound.

This does **not** certify an articulated robot. The disk is not yet shown to enclose every leg, body pitch/roll excursion or posture along these branches. The straight-segment argument also does not bound the actual continuous trajectory between simulator samples. Saved evidence explicitly marks `continuous_between_physics_steps_qualified=false`. Native disallowed-contact events are independent physical observations, not a proof of positive continuous clearance.

Proposed qualification reuses the existing URDF collision-primitive/FK helper and retained base pose, joints, velocities, native geometry identity and contacts. Verify all collision shapes and joint ordering against the native asset. Bound each link's world-space envelope at the 2-ms samples, including body orientation and legs. For between-sample clearance, a valid point-speed bound V implies an endpoint expansion of V×Δt/2, plus any geometry/pose uncertainty. Sampled velocities or endpoint displacement alone do not establish V: use an independently justified bound on intervening motion, or retain the explicit unqualified status. No controller margins are altered to compensate. If a valid bound cannot be established from retained evidence within the proposed budget, stop and report that limitation; do not advertise a full physical-clearance certificate.

Keep three separate facts for every branch:

1. **Physical harm/validity:** complete horizon, attributed disallowed contacts (support contacts separate), fall/stop, and any qualified articulated intersection/clearance evidence. Unknown continuous clearance remains unknown.
2. **Common operating margin:** a declared 0.48-m center-to-obstacle distance against true geometry, equivalent to the reference disk plus 20 mm, applied equally to hold, translations and turns. This is an evaluator convention, not the controller's eligibility mask or a claim that the disk is the actual body.
3. **Controller eligibility:** the original observation-only restrictions, nominal 0.45-m threshold, translation reserve to 0.48 m, recovery exceptions and stopping/dispatch rules, evaluated unchanged on their own inputs. A candidate can satisfy the common true-geometry margin yet be rejected by these rules, or be allowed by the controller while violating it.

The primary R1 comparator is the lowest cost candidate satisfying both qualified physical acceptability and the common margin. Also retain the unrestricted physical comparator and the margin-plus-controller-eligible comparator. Use the same finite reference cost C for all three. Report separately:

- operating-margin cost = min C over physical-and-margin candidates minus min C over physical candidates;
- excess eligibility-rejection cost = min C over physical-and-margin-and-eligible candidates minus min C over physical-and-margin candidates.

An empty set, unqualified physical clearance or unavailable finite reference cost is **undefined**, with its count and reason. Never substitute the actual controller mask for physical acceptability or assign unsafe candidates an invented finite penalty. Contacts remain reportable when scalar costs are unavailable. Reference validity and physical acceptability are distinct masks; disk-grid inability to score a physically harmless state is a reference limitation.

The non-safety cost formula and weights remain those of the original analytical evaluator: geodesic distance / 0.2 m/s, heading / 0.45 rad/s, opposing linear/angular braking terms with accelerations 0.4 m/s² and 1 rad/s², and phase-appropriate arrival settling. The existing 20-mm grid, 0.46-m inflation and 5-mm reference path gap remain explicit approximations. A scoring adapter must separate those scalar components from the additional common-margin and articulated-validity masks; it must not modify controller code or silently change reference weights. Invalid targets, inaccessible grid endpoints and missing components remain missing. Original targets use the single initial physical-frame anchor, never a current-truth correction or convenient goal snap.

## Numerical precision, practical ties and repeats

Continuous costs, differences and regret are unchanged by classification tolerances. Numerical equality tolerance is **1e−8 s**. Practical near-tie tolerance is **0.10 s**, corresponding to 20 mm at the reference 0.2 m/s speed (one reference grid cell), or 0.045 rad (2.58 degrees) at 0.45 rad/s. This is a declared practical resolution, not a tolerance estimated from zero variation in two repeats. The scientific usefulness threshold remains **δ=0.25 s** (5 cm at the same speed), separately from near ties.

Keep numerical-optimal and practical-near-optimal sets separate. Legitimate repeat variation is reported as absolute component/cost ranges on a fixed state/tape, separately from source replay error. Repeat uncertainty crossing the practical-tie boundary makes that classification indeterminate; it does not enlarge the 0.10-s definition. A maximum repeated-cost range above 0.025 s triggers reference/repeat qualification failure for the proposed reduced-repeat design. Systematic source-replay errors are never absorbed into a tie tolerance. Geometry classification has a 1e−9-m numerical band only; it is not an allowance for physical uncertainty.

The added panel contains below/at/above cases (100 μm offsets) for 0.45, 0.46, 0.465 and 0.48 m center clearance, plus disallowed contact despite ample clearance: **13/13 pass**. Integer micrometre expectations were saved before execution. It checks the unchanged strict controller full-clear predicates, original 5-mm reference predicate and proposed inclusive common-margin predicate. Reserve-recovery paths need separate trajectory evidence; these stationary cases do not certify them. The original 24-case strict failure and its two-label post-execution erratum remain intact. No weights or margins were changed to make either panel pass.

## Proposed remaining Phase 1 qualification — explicit approval required

Use a new, separately identified successor root. Existing completed source trajectories and corrected recheck assignments are not restarted. Collect only the four missing layout/source cells: layout 00/action, layout 02/command-history, layout 02/reactive, layout 02/action, in that order. Use the same fixed controllers and old-data collection head. Each is bounded to the original 17.6 simulated seconds including settling; retain exactly frame 132 with its complete 800-ms source trace. No outcome-based replacement or extra source rollout.

At each of these four states perform the same 21-branch sequence as the passing recheck: source replay 0, three repetitions of each of six candidates, then source replays 1 and 2. **No repeat reduction is proposed for these currently unqualified cells.** Stop on any strict source RGB/physical or candidate-repeat failure; no retry or automatic budget extension. These four plus the two already corrected states form a six-cell successor qualification population, requiring 6/6 coverage and unchanged strict criteria. This does not turn the original 24-slot pilot into a pass.

The same bounded phase includes read-only articulated/source-input qualification: original PNG/hash/capture/preprocessing identity, causal context and command timestamps, native joint/asset identity, target-frame anchoring, contact attribution and the clearance bounds described above. No model fitting or comparative method selection is permitted. If provenance or between-sample geometry cannot be qualified, stop with explicit missing evidence. Do not automatically patch the renderer or controllers.

| Additional resource | Proposed hard cap |
|---|---:|
| New source trajectories / retained new states | 4 / 4 |
| Branch attempts | 84 |
| Simulated seconds, including all settling | 137.6 |
| Wall / aggregate CPU | 1 hour / 16 core-hours |
| Aggregate RAM / total device VRAM | 16 GiB / 8 GiB |
| Retained / peak additional writes including caches | 2 GiB / 3 GiB |
| Filesystem reserves | RecoveryStorage ≥12 GiB; workspace ≥4 GiB |

The corrected 42-branch run measured 104.90 wall seconds including 160 optional encodes, 138.50 core-seconds and 54.01 MiB retained; branch execution was approximately half a second per 800-ms tape. Source snapshot+packet sizes were approximately 39–42 MiB. These measurements support the finite cap; they do not guarantee runtime for the missing action source or layout. Admission and live metering must enforce the cap. No automatic compression or retirement is part of admission.

## Proposed audit sampling and conditional reduced repeats

The following is an **identified design proposal**, not a fully qualified launch package. Exact new layout manifests and the completed qualification must be frozen and submitted before approval of Phase 2 execution. No new layout or source trajectory has been generated for this design.

Use eight layouts: the four exposed dense-cohort layouts, separately from four previously unexamined layouts from the unchanged static-maze generator. Freeze construction seed 2026092307, physics seed base 2026102800 and appearance seed base 2026102900; use the existing topology novelty rule against the exposed/prior/training/transfer graphs. Generate four, with no runtime rejection or substitution. Geometry-only identity rejection follows the existing generator and must be recorded before collection; absence of four eligible identities is a design failure, not permission to search different seeds. Never call these a sealed final benchmark.

For each layout collect the three fixed source controllers for 60 simulated seconds after the 1.5-s settle, with the same source readout. At decision frames 12,16,…,592 consider complete, causally valid packets that allow the following 800-ms trace within that window. Capture an outcome-blind streaming uniform reservoir of **four representative states** and separate reservoirs of **one translation-history and one turn/reversal-history diagnostic state**. All eligibility and diagnostic labels use source-observable inputs, not branch outcomes or model performance. A representative cell with N eligible states has n=min(4,N), probability n/N and weight N/n. Record every frame's inclusion status/reason and the RNG stream. Incomplete horizons/terminated sources reduce coverage; never replace a source or borrow another cell's quota.

Diagnostic history bins use the last five acknowledged 100-ms commands: translation if any |vx|>1e−6 m/s; otherwise turn if any |yaw rate|>1e−6 rad/s. Reversal is a recorded sign change within the turn history, not an extra quota. Annotate observed clearance bands (≤0.45, (0.45,0.48], >0.48 m, unknown) and mission phase (exploration/approach-settle/return); report empty bands rather than adding rollouts. Diagnostic sampling has its own n/N, never pooled into representative prevalence. Record overlap and execute the union once, at most six states/cell, 144 total.

Maintain at most six selected snapshot/packet objects in RAM during collection; write only final reservoir members. Record at most 146 reservoir admission opportunities per source, at most 3,504 overall. A selected serialized snapshot+packet has a 64-MiB admission limit. Bound the small original RGB/command source recording separately. This avoids an unbudgeted full physical-snapshot trajectory and does not retire any written artifact. Fail admission if actual object/write sizes exceed caps.

Conditional on completed qualification, use three full source replays at **every** sampled state, one before and two after candidate branches. Each candidate gets one branch, except the first representative state by timestamp in **every layout/source cell**, whose entire bank gets three repeats. Spot-check identity is fixed before branch outcomes. For any off-bank reactive tape execute its separate branch; at those same spot-check states repeat it three times. Exact projected-tape equality within 1e−7 per command component defines bank equivalence. No new action enters R1's six-action bank. Maximal attempts are 144×(3+6+1) + 24×(12+2) = **1,776**, with overlap reducing work rather than adding substitute states. Failure of a spot-check stops the audit; it does not trigger automatic repeat expansion. Qualification at two or six pilot states is not claimed to prove all future determinism.

| Conditional audit resource | Proposed hard cap |
|---|---:|
| Layouts / sources / retained state union | 8 / 24 / 144 |
| Branch attempts, including source replays and off-bank reactive | 1,776 |
| Total simulated seconds including 24 settling windows | 2,896.8 |
| Wall / aggregate CPU | 4 hours / 64 core-hours |
| Aggregate RAM / total device VRAM | 24 GiB / 8 GiB |
| Retained / peak additional writes, including caches | 16 GiB / 20 GiB |
| Filesystem reserves | RecoveryStorage ≥12 GiB; workspace ≥4 GiB |

Storage admission envelopes: 9 GiB selected snapshots/packets, 3.5 GiB branch records, 3 GiB source recordings and 0.5 GiB scalar/report evidence. This is a cap-based forecast using the measured recheck, not measured audit usage. No raw depth arrays or dense feature tensors are retained. Images, state, command tapes, camera definitions, asset/software identities and physical trajectories are preserved. If these caps prove insufficient, stop without expanding or retiring artifacts. CPU/RAM headroom does not justify concurrency until native source workers and metering are covered by the approved implementation.

## Rows, masks, estimands and uncertainty

Retain R0, R1, R2, R2b, R3, R4, R4s, R5c and R5r as defined in the earlier draft/final handoff, with **both matched heads** wherever a head is used. R0 seed is 2026092308. Canonical tie order is hold, forward, left_arc, right_arc, left_turn, right_turn. R4s uses cyclic suffix permutation [1,2,3,4,5,0] after the identical committed 300-ms prefix. R2b translation direction is undefined below 1e−6 m; preserve those missing entries rather than manufacture a direction. R3 is retained in the proposal following the strict recheck, conditional on per-state RGB qualification. No audit rows have been executed.

Reference regret uses the common operating-margin acceptable bank. Report physically harmful selections, harmless margin violations and excess eligibility rejection separately. Finite regret requires a complete qualified outcome, valid original source packet/target, finite common reference and a selected candidate in that acceptable set. Harm and margin violation rates use their wider identifiable populations, not only finite-regret cases. Report all mask intersections, sample weights, missing targets, failures, invalid clearance, unavailable images and empty acceptable banks by row/layout/source/stratum. Unavailability is not converted to hold. Off-bank reactive choices retain their signed bank-relative gap, with the same physical/margin qualification.

For each matched head, R2/R3/R4/R5c decomposition uses one identical common mask, and G is recomputed on that mask. Nominal G is also shown on its own valid R5c population. L_readout and L_forecast require qualified R3; where unavailable, both are unavailable and the remaining direct paired contrasts use their explicitly reported pairwise masks. No missing rows are imputed or telescoped across different populations.

Primary contrasts remain G, H_scorer, H_motion, D_old and D_maze, with δ=0.25 s. Weight source cells equally within layout and layouts equally within exposure stratum; show representative and diagnostic panels separately. Required coverage is ≥90% of planned representative slots overall **and** ≥90% within every layout/source cell, plus ≥90% reference and primary paired-score coverage among restored states. With four representative slots per cell this effectively requires four valid slots; sparse cells remain failures, not opportunities for more collection.

Use layout-level paired estimates, t intervals at 99% individually (Bonferroni five-primary family), reported separately for exposed and unexamined strata; secondary intervals are 95%, descriptive, with no selective promotion. With only four layouts per stratum, normality-based intervals are fragile: publish every layout value and label the study exploratory. The earlier blinded precision scenarios already show that eight layouts generally cannot reach δ/2 precision except under unusually low variability; four per stratum is weaker still. More within-layout states do not fix that. No effect-dependent expansion is allowed.

Harm: report weighted state rates, paired differences and layout-level intervals, plus the probability of any harm on a sampled layout using an exact binomial interval. Zero observed harms do not produce a zero upper bound; that layout-any-harm interval is **not** a per-decision risk interval. Proposed decision limits are 5% absolute harm and 1 percentage-point excess over command control. If this small panel cannot bound those quantities adequately, harm noninferiority remains inconclusive. Do not claim useful decision quality without both regret and harm evidence; inconclusive is an allowed endpoint.

## RGB fallback and stopping rule

The user’s failure-contingent fallback was not activated: the amended recheck passed. Nevertheless, any later RGB qualification failure would leave R3 and L_readout/L_forecast unavailable. R0/R1/R2/R2b/R4/R4s/R5c/R5r may remain analysable only where their physical outcomes, clearance, original source inputs and own required features independently qualify. Such a reduced panel needs explicit approval of its changed masks and coverage; it is not an automatic fallback execution. Physical agreement alone is not source-image validity. Preserve existing snapshots/tapes/trajectories for potentially recoverable, **not guaranteed**, later rendering, which needs its own fidelity checks.

**Stop here at checkpoint (a).** The recheck and both read-only reports are complete. Recommended approval scope is the finite remaining Phase 1 qualification above, followed by another identified checkpoint submission; **Phase 2 is not ready for execution approval yet**. Its layout identities, full physical/source-input qualification and implementation bindings remain outstanding. Neither this frozen proposal, its commit nor the RGB pass authorises collection or comparative scoring. After a separately approved fixed audit, deliver the result, versioned branch panel and one-step decision memo, then stop without implementing that recommendation.
