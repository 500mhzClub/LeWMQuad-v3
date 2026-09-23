# LeWMQuad-v3: final agent handoff — decision-headroom audit

**Prepared:** 23 September 2026  
**Research owner and approver:** Andrew Knowles  
**Workspace:** `/home/andrewknowles/Workspace/LeWMQuad-v3`  
**Suggested repository path:** `docs/go2_decision_headroom_agent_handoff_final_2026-09-23.md`

**Status and precedence:** This is the consolidated execution handoff. It replaces `go2_decision_headroom_audit_brief_2026-09-23.md` and supersedes Stages B–D of section 10 in `LeWMQuad_JEPA_World_Model_Progress_Report_2026-09-23.md`. Stage A continues unchanged. Stage E (timing, sensing and hardware) remains deferred. Historical goals and superseded roadmap stages do not authorise additional work.

**Immediate authority:** Phase 0 and the bounded, blinded Phase 1 pilot, including necessary audit tooling. **Phase 2 is conditional:** stop at checkpoint (a), submit the frozen protocol and measured budget, and obtain Andrew's explicit approval before comparative audit execution. At the end of Phase 2, deliver the decision memo and stop. No recommended intervention is authorised by this handoff.

## Start here

1. Check the actual status of the existing four-run Stage A comparison using its retained plan and live/completed records. Do not assume report-era PIDs are still live, launch a duplicate owner, restart a completed trial, or change an active trial. If Stage A is already complete, consolidate its existing results rather than rerunning it.
2. Finish Stage A under its existing controller, budgets, resource limits and physical readers. Preserve every result and failure. Do not let an unfavourable result trigger follow-up fitting.
3. Record finite Phase 1 collection, compute and storage caps before new pilot physics. Then implement and qualify the smallest audit tooling required below. Fresh source rollouts are permitted only as bounded audit-data collection, not as new navigation-performance cohorts.
4. Complete the blinded technical pilot, commit the proposed protocol and hashed configuration, and **stop for Andrew's explicit approval at checkpoint (a)**. A commit or a passing pilot is not permission to start Phase 2.
5. After approval, execute the fixed audit, produce one result report and one decision memo, and stop at checkpoint (b). Do not implement the memo's recommendation.

Do not restart the historical results inventory, reopen the research strategy or add another review cycle before carrying out the work authorised here. Follow applicable repository instructions and permission controls; do not bypass a denied operation.

---

## 1. Why the plan is changing

This is a JEPA-based PhD about useful feature representations for general robotics. The Go2 maze stack is a testbed for that question, not the thesis.

The 23 September report establishes that action-conditioned prediction in frozen V-JEPA 2.1 feature space is real. At 700 ms, feature MSE is 0.234 for the action-conditioned predictor, against 0.312 without future actions and 0.402 for persistence.

It also shows two limits:

- In the full-maze controller, that prediction is used only to forecast short-horizon ego-motion through an XY/yaw readout. Command history already predicts that quantity to roughly 5–15 mm at 700 ms on executed windows.
- The latest dense cohort completed 1/4 round trips, against 3/4 for command history.

Errors on executed windows do not bound errors on unexecuted alternatives. Average motion error also does not determine decision quality. Closed-loop success can mask poor action ranking (ARC-Bench, arXiv 2609.05461), so decisions must be audited directly.

Further readout or predictor iterations could reduce motion error without establishing that any decision changes. The next step is therefore to:

1. measure decision headroom directly;
2. localise where the learned pathway gains or loses decision quality;
3. let that measurement choose the next intervention.

## 2. Research goal (reset)

### Primary question

When do JEPA representations carry decision-relevant information that generalises to held-out environments? "JEPA representations" here means frozen V-JEPA 2.1 features and action-conditioned predictions in that space. "Decision-relevant" means information beyond nominal command-based motion, current geometry and explicit memory. The question also asks what a controller must do to use that information.

### What counts as useful

A representation, or a prediction in it, is useful for a decision when both of these hold:

- the readout and scoring budget is fixed;
- using the representation reduces counterfactual selection regret or harmful selections, relative to strong non-learned alternatives, on held-out layouts.

Feature MSE, probe accuracy and branch retrieval remain supporting evidence, consistent with the evidence levels in section 2 of the report. They are not the criterion.

### Navigation success and the retired goal

Navigation success (verified goal and home arrival) becomes a qualification check for any candidate controller, not the scientific endpoint.

The 7 September objective was reliable closed-loop navigation through unseen mazes, using the learned model to evaluate candidate actions. It must not be used to justify further component work.

### No claims about the training objective

The current configuration cannot isolate the contribution of the JEPA training objective, for two reasons:

- the encoder is a frozen foundation model;
- the encoder screen is confounded by resolution.

Make no claims about the objective from this work. The audit is designed so that a later matched-representation comparison can run on identical decisions (section 8).

## 3. Authority

### Authorised

1. **Phase 0:** finish Stage A unchanged, or consolidate its completed outcomes if it has already finished.
2. **Phase 1:** a bounded technical pilot for faithful physics branching, repeat variability, reference-cost qualification and resource measurement. No comparative method selections, rankings or regrets.
3. **Phase 2, only after checkpoint (a) approval:** execute the headroom audit under the approved frozen protocol, then produce a decision memo that recommends, but does not execute, the next step.
4. **Tooling:** the necessary snapshot/restore mechanism, branch runner, independent reference-cost evaluator and analysis scripts. Reuse existing machinery where possible. Preserve the controller and model code under test; audit-only adapters must have separate identities.
5. **Fresh source rollouts:** bounded collection with fixed controllers and declared snapshot quotas. Their purpose is obtaining restorable decision states, not estimating navigation performance. They do not expand the task family, sensing, action bank or deployment scope. Phase 2 source collection starts only after checkpoint (a).

### Not authorised without explicit approval

- Any fitting or fine-tuning of the encoder, predictor, readout or goal metric; any alteration of controller scorer weights or eligibility rules. The only cost specification permitted is the independent evaluator-only reference cost developed in Phase 1 and frozen at checkpoint (a).
- New environment families, including terrain or dynamic obstacles.
- Sensor changes, including RGB-only.
- Changes to the candidate bank or horizon.
- Replacing the default controller.
- New navigation-performance or prospective confirmation cohorts beyond Stage A. This does not prohibit the bounded source-rollout collection expressly authorised above.
- Any sealed benchmark material. Legacy V4 remains development-only and ineligible for final evaluation.
- The camera-stabilisation / nuisance-variance diagnostic. Propose it in the memo only if the audit finds a failure it would help localise.

### Approval checkpoints

- **(a) End of Phase 1 — mandatory approval stop.** Submit the committed protocol, hashed config, pilot validity report, fixed sampling design and measured resource budget. Obtain Andrew's explicit approval of the identified version before Phase 2 collection or comparative scoring. Record that approval with the experiment identity. Silence, a commit, passing tests or the original roadmap is not approval.
- **(b) End of Phase 2 — decision stop.** Deliver the result and a memo recommending one next step. Do not execute it. Any repair, extra collection, fitting, benchmark expansion or task redirect requires new approval.

Routine work inside the recorded Phase 1 caps or approved Phase 2 budget does not require repeated permission requests. Exceeding those bounds or changing scientific scope does.

## 4. Phase 0: close out Stage A

Run the four already assigned trials to terminal outcome, with their fixed budgets and physical readers. If they have completed, use their completed evidence. Report goal/home arrivals, disallowed contacts, simulated duration, holds and executed-window motion errors for every assigned outcome, as specified in Stage A of the report. Compare the matched readouts with the retained initial-head outcomes without treating reused predecessors as new independent trials.

Label the result a development intervention on exposed layouts, with one run per cell. It cannot establish a readout effect on navigation, and its outcome triggers no follow-up fitting.

Its decision records may contribute source states to the audit, but only if they pass the restoration checks in 5.2.

## 5. Phase 1: technical pilot

The pilot has four jobs:

- prove that branches are physically faithful;
- measure legitimate variability between repeats;
- fix the reference cost;
- measure the compute and storage budget.

It ends by proposing a frozen protocol and stopping at checkpoint (a).

**Pilot caps:** before any new pilot physics, record numerical ceilings for source rollouts, layouts, sampled states, branch/repeat attempts, compute time, RAM/VRAM, retained bytes and per-filesystem reserve. Use existing measured rates where available; the initial pilot must be small and finite. The sanity-panel size below is guidance, not authority for unbounded collection. Stop if the caps are reached; do not extend them to make the pilot pass. Planning estimates are not resource caps.

**Blinding rule:** do not compute the audit rows' selections, regrets or rankings on pilot branch states. The exception is the independent reference-cost ranking of the sanity panel against its prewritten expected answers. Feature/predictor/readout components may run for timing without producing comparative selection results. Do not estimate paired method-regret variance in the pilot.

Frozen source controllers necessarily make their ordinary online choices while collecting snapshots; that is permitted. It is not permission to compare their pilot mission outcomes, use those outcomes to tune the reference cost, or select states to favour a method. The source applied-command trace may also be replayed for restoration fidelity without scoring it as a comparative selection. This blinding rule concerns the audit, not a prohibition on operating the source controller.

Pilot states are for the pilot only. Draw the audit sample afresh. Record pilot layout exposure and keep those layouts out of any subsequently claimed never-examined layout subset.

### 5.1 Source states and restoration

Prefer fresh source runs with native full-state snapshots at decision points. Reconstructing historical trajectories is the fallback, because deterministic generation recipes do not guarantee bitwise reconstruction of historical closed-loop runs (report, section 11). Historical states may be used only if they pass 5.2.

A snapshot must capture everything that affects the branch:

- base pose and twist;
- joint positions and velocities;
- contact and solver state the simulator exposes;
- locomotion-policy internal state, where present: recurrent state, gait phase or clock, action history;
- command-limiter state and any pending command queue;
- commands already committed at decision time;
- RNG state;
- the planner's exact candidate tapes, canonical ordering and eligibility information at that state.

Retain a **frozen decision-input packet** alongside each physical snapshot: decision timestamp; the exact RGB context consumed, preprocessing and observation timestamps; causal applied-command history; recorded committed prefix; observed pose and map; active route target; routing and mission state; recovery/selector state; scorer configuration; candidate definitions; and eligibility masks with exclusion reasons. Retain anything else actually consumed by the decision path, rather than reconstructing a different observation later. Source identities and digests must connect the packet to the physical snapshot.

Physical ground truth and future branch outcomes are stored separately as evaluator-only evidence. Restoring physics does not authorise substituting privileged observations into the method inputs.

The limiter is part of the predictor's physical contract, because future commands are projected through it. Its state is not bookkeeping.

**Branch semantics.**

1. Set `t = 0` at the source decision timestamp and restore the complete state and decision-input packet.
2. Each candidate branch spans exactly `[0, 800] ms`, **including** the already-committed prefix. Reproduce that prefix unchanged. A candidate may differ only where the source execution contract permits replacement.
3. Execute the declared candidate continuation through the same limiter and execution machinery for the remainder of that interval. Do not append another 800 ms after the prefix.
4. Record outcomes at `t = 100, 200, ..., 800 ms` relative to that same decision timestamp. Use the current dense controller's exact six tapes and eight horizons; do not substitute the separate historical three-pulse assay or change action magnitudes/durations.
5. Where candidates have not yet diverged because of the committed prefix, label action-discrimination quantities as inapplicable. Shared-prefix motion may still inform the physical cost. Freeze any repeat/terminal-event handling in the protocol; do not fabricate missing future frames or outcomes.

For each branch, record:

- physical state at all eight horizons (100–800 ms);
- disallowed obstacle contacts, using the existing physical reader's contact definition, separately from ordinary support contacts;
- swept-footprint clearance;
- rendered camera frames at every time the encoder or readout consumes, at native input resolution.

### 5.2 Restoration fidelity and repeat variability are separate quantities

- **Restoration fidelity.** Re-execute the complete recorded applied-command trace over `[0, 800] ms`, including later command changes from the source run. Do not hold only the first logged action and call disagreement a restoration failure. Compare pose, yaw and the existing disallowed-contact readout with the recorded outcomes at every horizon. Counterfactual candidate branches instead execute their declared tapes and are not expected to reproduce commands selected by the original controller's later replanning.
- **Repeat variability.** Replay the same state and action several times. If simulator and policy are deterministic, verify near-identical outcomes. If not, use at least three repeats and characterise the spread.

An invalidly reconstructed state is not a noisy valid state. Suppose repeats agree with each other but systematically disagree with the source recording. That is a restoration or configuration failure. When it happens:

- exclude the state from ranking comparisons;
- report it by stratum;
- do not widen tolerances to absorb it.

Only legitimate repeat variability, with the declared numerical precision of the measurement, may inform near-tie tolerances. Systematic restoration disagreement must not enlarge them. Record repeat handling and action acceptability under variability before Phase 2. Treat a restoration failure as a failed technical observation, not as a failed navigation method or a near-tie.

### 5.3 Declared physical reference cost (evaluator-only)

The reference cost uses simulator ground truth. Only the independent reference evaluator and R1 use that true geometry to construct the reference optimum. It is never exposed to deployed selection rows. It is a declared physical reference, not the true optimal cost-to-go. The other privileged diagnostic substitutions are limited to those explicitly specified in section 6.3.

**Requirements.**

- **Fixed route target.** Use the target active in the source run at that state. Routing is not audited.
- **Safety as a constraint, not a weight.** A candidate is acceptable if it causes no **disallowed obstacle contact** under the existing physical reader's definition and its swept footprint maintains the declared minimum clearance. Ordinary foot-ground support contact is not a disallowed obstacle contact. Selecting an unacceptable candidate when an acceptable alternative existed is a harmful selection, reported separately. Freeze the handling of stochastic repeats and terminal events. Never make an unsafe action appear favourable by giving it a low progress cost or an arbitrary finite safety penalty.
- **Progress** as reduction in geodesic distance to the active target, on the true occupancy map inflated by the robot footprint. Not Euclidean endpoint distance.
- **Terminal heading and motion,** where they affect the next manoeuvre. For example, penalise misalignment with the geodesic descent direction at the endpoint. This stops a necessary preparatory turn being scored as useless.
- **Phase-appropriate stopping.** Reward settling only where the mission phase calls for it (goal or home arrival). Never reward holding universally.
- **Physical units.** Express the non-safety cost in a physical unit, preferably time-equivalent seconds of travel, so thresholds can be justified physically.
- **Components kept.** Keep every component. Aggregates use frozen weights and are always reported alongside components.

**Alternative formulation.** Value each endpoint by a short, fixed, privileged continuation, applied identically to all candidates. Choose one formulation in the pilot, not both. If chosen, declare the continuation policy and duration and include its cost in the resource budget. This is an evaluator-only endpoint valuation; it does not change the controller's candidate tapes or its 100–800-ms prediction horizons.

**Sanity panel.**

1. Assemble a small set of physically obvious cases, on the order of 20–30. Cover:
   - wall ahead;
   - open corridor;
   - junctions needing a preparatory turn;
   - near-goal settling;
   - near-wall grazing;
   - reversals.
2. Write the expected correct ranking, or optimal set, for each case before computing any cost.
3. Freeze the smallest formulation that ranks all of them correctly.

**Flag, don't score, two kinds of state:**

- states where the active target is physically invalid;
- states where no candidate is both acceptable and useful.

These are router, perception or candidate-bank limitations, not ranking failures. Retain and count them by source controller, layout and stratum. Keep any observed disallowed contacts in the raw evidence even where scalar ranking is undefined. Define "useful" in the frozen reference-cost protocol, including the value of preparatory turns and required settling.

### 5.4 Budget

Measure per-state branch cost (candidates × repeats × horizon, plus rendering and encoder passes) and extrapolate.

Storage rules:

- Prefer the existing RecoveryStorage artifact destination. The report's approximately 41 GiB there and 5.5 GiB on the workspace are dated observations, not live admission evidence. Verify free space on the actual resolved output/temp/cache paths with per-path filesystem checks, accounting for shared devices only once.
- Declare every input, output, temporary, cache and log path. Do not assume a path's mount from its name or silently write beside read-only inputs. Set a job-specific reserve and peak-write budget; do not import a mission-specific storage threshold as a repository-wide policy.
- Store RGB losslessly at native model input resolution, together with the decision packets, snapshot data, poses, disallowed contacts, clearances, command tapes and reference-cost components.
- Do not retain raw depth or dense feature tensors for this audit. Recompute features from stored frames. If faithful branching requires an additional retained input that conflicts with this rule, stop and request a scoped amendment rather than silently omit it.
- Check free space before each batch and include peak writes plus the declared reserve. Do not fill the filesystem. No new deletion or depth-retirement scope is authorised by this handoff.
- Measure CPU/GPU time, RAM/VRAM and useful concurrency. Do not compete with an active Stage A owner or change its timing/resource conditions. Reuse existing admission tooling; do not create a general infrastructure programme.

### 5.5 Freeze the protocol

Before any comparative scoring, commit a protocol document plus a hashed, machine-readable config. It must contain:

- source/controller/checkpoint identities, decision-packet definition, exact candidate tapes, ordering, limiter/prefix semantics and all coordinate/time conventions;
- restoration acceptance tolerances, repeat counts and handling, exclusion rules and minimum acceptable restoration/scoring coverage;
- the near-tie definition, informed by legitimate repeat variability, and the numerical threshold below which translation direction is undefined for R2b;
- the chosen physical reference cost, its component units/weights, minimum clearance, usefulness rule, sanity-panel evidence and any evaluator-only continuation;
- **δ**, the smallest practically meaningful regret difference, justified physically. A mission-duration heuristic may motivate its scale, but summing local regrets is not a proved forecast of mission-time savings; state the assumptions;
- the absolute harmful-selection acceptance limit and its uncertainty rule; the criterion for acceptable paired excess harm when claiming a learned benefit; and all coverage requirements;
- the sampling procedure, inclusion probabilities, stratum boundaries, source-controller quotas and treatment of unfilled strata;
- the layout list, exposure histories and fit/selection/evaluation roles; keep never-examined layouts separate from development and pilot exposure;
- the layout/state counts, branch/repeat counts, source-rollout quotas, resource/time caps, output paths, measured costs and peak storage plus reserve;
- head-specific primary and secondary quantities, common scoring populations/masks and weighting rules (6.4), bootstrap/hierarchical interval method, multiplicity treatment where applicable, and the inconclusive rule;
- the fixed candidate derangement and tie-breaking/random-row procedure; treatment of unavailable selections, row failures, off-bank reactive actions and missing outcomes;
- source/config hashes, preflight checks and attempt/output identities. Do not pick a preferred readout after seeing Phase 2 results; report both existing heads under the predeclared analysis.

**Sizing.** Keep the pilot blinded. Use it to measure restoration validity, repeat variability, branch cost and storage needs, not comparative performance. Propose layout/state counts using those measured costs plus explicitly declared conservative variance assumptions or precision scenarios. Do not estimate paired method-regret variance from the blinded pilot.

Target confidence-interval precision relative to δ (for example, half-width about δ/2), rather than selecting a sample size to obtain significance. State which assumptions make the target plausible. More states may improve each layout estimate but do not substitute for independent layouts. If an affordable design cannot credibly target the required precision, label Phase 2 exploratory and preserve an inconclusive outcome. Freeze sample size before comparative scoring; no effect-dependent expansion.

**Checkpoint (a): stop here.** Submit the protocol, machine-readable config, pilot evidence and measured budget for Andrew's explicit approval. Phase 2 is not authorised merely because these files have been committed.

## 6. Phase 2: the audit

### 6.1 Question

On the current static-maze task, with route, memory, sensor observations and candidate bank held fixed:

- how much decision regret remains in strong simple controls?
- where does the learned pathway introduce or remove regret?

Results are scoped to the tested task, bank, horizon and layouts.

### 6.2 Sampling

**Source controllers:**

- command history;
- reactive feedback;
- the current action-conditioned dense configuration.

Report every result by source controller, because each controller visits different states.

**Layouts:** include cohort/development layouts and same-family layouts never used for training or tuning. Record each layout's exposure history.

**Strata:**

- clearance margin;
- motion history (moving, turning, holding);
- turn reversals;
- mission phase (exploration, goal approach and settle, return);
- source controller.

**Two samples, analysed separately:**

- **Representative sample:** decision states drawn with known inclusion probabilities. Used for prevalence-weighted estimates.
- **Diagnostic panel:** deliberately oversampled near-wall, transition, reversal and hold states, to find where small errors matter. Never pool it with the representative sample into prevalence estimates without weights.

Fix the sampling procedure and draw the states before any branch outcome is scored.

### 6.3 Rows

Attempt every row on every sampled state and preserve any undefined/failed row. Do not quietly substitute different states for different methods. Scalar comparisons use the explicit shared scoring populations in 6.4, rather than assuming all selections are acceptable.

Except for the privileged reference R1, rows retain the source observed map, active route target and controller state. All current-scorer rows retain its weights, tie-breaking and eligibility rules. Upstream observation-only exclusions remain fixed. Where unchanged downstream eligibility checks depend on the substituted motion, record their resulting masks and reasons rather than changing those checks to make a row pass. R5r retains its own frozen reactive selection rule.

**Only the independent reference evaluator and R1 use true geometry.** R2 receives true branch motion, R2b receives only the specified true translation magnitudes, and R3 receives actual future features, solely as evaluator-only substitutions. These do not permit true-map, future-contact or future-clearance leakage into the current scorer. No diagnostic row is deployed as a navigation controller.

| Row | Selection rule | Role |
| --- | --- | --- |
| R0 | Uniform random eligible candidate, using a fixed recorded sampling rule and seed | Scale reference, not an additional independent experiment |
| R1 | Reference optimum under 5.3; also report the optimum within the controller-eligible subset | Best available within bank and horizon |
| R2 | Current scorer, given true branch motion | Whether the scorer exposes the physical opportunity |
| R2b | Current scorer, given R4's decoded motion with translation magnitude replaced by the true magnitude at the corresponding horizons; direction and yaw unchanged | Evaluator-only test of progress shrinkage; undefined direction handled below |
| R3 | Current scorer, readout applied to actual future features from the branch frames | Observed-representation and readout pathway |
| R4 | Current scorer, readout applied to predicted features, as deployed | Learned forecasting pathway |
| R4s | As R4, but each candidate's predicted features use another candidate's action inputs (fixed derangement) | Whether action conditioning contributes to decisions |
| R5c | Current scorer, given command-history motion | Deployed nominal control |
| R5r | Reactive controller's own selection; branch it separately if its action is off-bank | Prediction-free control |

Notes on the rows:

- Run R3 and R4 with the matched readout pair that already exists: the old-data control and the maze-data head. Run R2b and R4s against their corresponding R4 head, without mixing heads inside a comparison. No fitting or outcome-based checkpoint selection. This tests Stage A's readout question at decision level; additional states are not additional independent layouts.
- R4s uses a fixed derangement of candidate suffix/action inputs, while retaining the identical history and committed prefix. Keep the physical branch labels correct. It is an information intervention, not a refitted control. Do not derange committed prefixes or select a favourable permutation after scoring.
- **R2b zero-vector rule:** if a required predicted translation norm is below the frozen numerical threshold, the magnitude-only substitution is undefined there. Report the candidate/state and exclude it from the prespecified shrinkage-comparison population; do not invent a direction or use true direction. Keep this diagnostic's coverage explicit.
- Omit the no-future-action predictor as a selection row. Its identical candidate forecasts make it an unsuitable competitive selection control here; this does not deny its value in prior prediction assays.
- A reactive action is off-bank only under the frozen tape-equivalence definition, not because of its label alone. Branch an off-bank selection separately from the same restored state and prefix. Do not add it to R1's bank or substitute it for a bank member. For an acceptable off-bank selection, report its **signed bank-relative cost gap**, which can be negative; do not call it ordinary within-bank regret. If it beats the bank optimum, flag a candidate-bank limitation.
- R2 removing a hold implicates the motion-forecast/readout pathway, not shrinkage alone. R2b is the narrower magnitude-only diagnostic; changes in direction, yaw and clearance must not be silently included in it.

### 6.4 Metrics, acceptability and scoring populations

Let `B(s)` be the fixed full candidate bank, `A(s)` its physically acceptable subset, and `C_ref(s,a)` the declared non-safety physical reference cost. For a valid target and a state with at least one acceptable, useful candidate, define:

`C*(s) = min_{a in A(s)} C_ref(s,a)`.

The candidate bank, acceptance rule and usefulness rule are frozen before Phase 2. The reference optimum is relative to that bank, horizon and cost, not an unrestricted optimal policy.

**Harm and finite regret are separate outcomes.**

- Evaluate harmful selections on **all restoration-valid sampled states with a valid reference and an acceptable alternative**. If a selected action is unacceptable, mark harm and retain the physical outcome; do not assign arbitrary finite scalar regret or a synthetic safety penalty.
- For an acceptable in-bank selection, report `r_m(s) = C_ref(s,a_m(s)) - C*(s)`. This is nonnegative. Near-tie tolerance affects optimal-set/rank classification, not the underlying continuous cost difference. R1 has zero regret by definition.
- For an acceptable off-bank reactive selection, report `gap_R5r(s) = C_ref(s,a_R5r(s)) - C*(s)`. This is a signed bank-relative gap and may be negative. Retain separate in-bank and off-bank results.
- Report unacceptable selections, unavailable/undefined selections, pipeline/branch failures and reference-invalid states explicitly. A row failure is not silently a hold, a safe action, or a missing case that can be ignored. No acceptable/useful candidate is a distinct bank/target limitation. Preserve raw contacts even when ranking cannot be defined.

**Denominators and common populations.**

1. Report validity, acceptability, harm and finite-regret coverage for every row, layout, source controller and stratum. Include counts and sampling weights. Regret conditional on acceptable selection is not overall performance.
2. Report the primary nominal headroom `G` on all scoreable states where R5c selects an acceptable action, with R5c harm on the full valid denominator beside it. Near-optimality can only be claimed for this declared scoring scope and only if the frozen harm and coverage requirements pass.
3. Each paired finite-regret comparison uses the same state set and weights for its two rows: both must have valid, acceptable selections. Freeze this mask rule before outcomes; retain rather than hide excluded cases. Report the corresponding paired harm comparison on the full valid population, not only the both-safe subset.
4. For the telescoping chain, use one explicit common population for R2, R3, R4 and R5c, for each readout head, with identical weights. Recompute all chain components and the chain-specific `G` on that population. Do not claim a telescoping identity between means calculated on different masks. Report nominal `G` on its broader scoring population separately.
5. The informative/supported populations may differ across diagnostics. Label these differences and their coverage. If safe/common coverage is too limited or selective for the intended conclusion, the result is inconclusive; do not infer overall benefit or sufficiency from favourable conditional means.

**Per-state secondary outcomes:** unnecessary hold when an acceptable candidate beats the executed hold by more than the near-tie tolerance; rank agreement/optimal-set membership; every cost component; eligibility exclusions; and repeat variation. A harmful hold is additionally counted as harm, not just inefficiency.

**Primary quantities:** layout-level weighted means, paired where applicable.

| Quantity | Definition | Interpretation |
| --- | --- | --- |
| `G` | `mean r_R5c` on the declared nominal scoring population | Direct decision headroom beyond nominal command-history selection; not inferred by separately thresholding its components |
| `H_scorer` | `mean r_R2` | What the current decision rule loses with true motion supplied |
| `H_motion` | `mean (r_R5c - r_R2)` on their common population | Signed effect of replacing nominal motion by true motion under the current scorer |
| `D_learned` | `mean (r_R4 - r_R5c)` on their common population, reported per head | Negative values favour the learned pathway; accompany with harm difference and coverage |

`H_motion` is not an upper bound on any predictor's possible decision benefit. An imperfect forecast can compensate for a scorer flaw. A learned benefit can therefore be present even if `H_motion` is negligible or negative; investigate rather than rule it out by definition.

**Secondary diagnostic quantities:**

- `L_readout = mean (r_R3 - r_R2)`.
- `L_forecast = mean (r_R4 - r_R3)`; it may be negative.
- `A_action = mean (r_R4s - r_R4)`; a positive value means deranging action inputs worsens selection.
- **Shrinkage:** unnecessary holds and regret under R4 versus R2b on their declared, valid shared population, with harm alongside.
- **Eligibility:** report whether any reference-near-optimal full-bank candidate was eligible, and the cost gap to the best acceptable controller-eligible candidate. This distinguishes equivalent optima and tests the clearance-hold hypothesis without changing eligibility rules.
- **Reactive:** paired in-bank regret or off-bank signed gap, always with harm and bank-membership coverage.

On an identical common population and weights, `G_chain = H_scorer + H_motion` and `D_learned = L_forecast + L_readout - H_motion` telescope arithmetically. They are **not** an additive causal allocation of blame. Do not apply these identities to differently masked primary/secondary summaries.

### 6.5 Analysis and interpretation rules

**The unit of inference is the layout.** Compute weighted per-layout summaries, take paired differences across layouts, quantify uncertainty with the frozen layout-cluster bootstrap or equivalent hierarchical method, and display every layout. Decisions and overlapping branches within a trajectory are not independent generalisation units. A small number of layouts may leave intervals insufficiently reliable or wide; state that limitation rather than hiding it with more frames.

Analyse the representative sample and diagnostic panel separately. Do not turn oversampled near-wall outcomes into prevalence estimates without the declared inclusion weights. Maintain source-controller and exposure strata. Report both existing readout heads as planned; do not select a winner after seeing results or silently change the primary comparison.

Let `[L_Q, U_Q]` denote the frozen-convention interval for quantity `Q`:

| Classification | Rule |
| --- | --- |
| Negligible nonnegative regret/headroom | `U_Q < delta` |
| Signed difference practically equivalent to zero | Entire interval inside `[-delta, +delta]` |
| Material positive signed difference | `L_Q > +delta` |
| Material negative signed difference | `U_Q < -delta` |
| None of the preceding classifications is supported | Inconclusive for that classification; report the interval |

For decision-table conditions that require **ruling out a practically meaningful benefit**, require `L_D > -delta`; do not use failure to reach significance as that evidence. Practical equivalence is stronger and requires the full interval within both margins. Distinguish these statements in the memo.

Declare interval convention, confidence level and treatment of multiple primary contrasts in the protocol. For example, a two-one-sided-test equivalence convention may use a 90% interval, but the choice must precede comparative scoring. Every "small", "large", "material" or "negligible" conclusion must reference an explicit rule, not an impression from a point estimate.

**Harm and coverage requirements:** freeze the absolute harm limit, paired excess-harm criterion and their uncertainty conventions. A lower finite regret with unacceptable harmful-selection behaviour is not an improvement. Zero observed harms alone is not proof of zero risk. Nominal near-optimality requires its direct `G` interval, satisfactory nominal harm and adequate scoring/restoration coverage. Paired learned-benefit claims require their declared harm and coverage criteria as well as `D_learned`.

Exploratory/post-hoc analyses remain labelled and do not select a decision-table row. No effect-dependent addition of layouts, states, repeats or hypotheses is authorised.

### 6.6 Decision table for the memo

Apply validity, harm and coverage requirements first. Report every supported diagnostic classification, then recommend **one** next step. If apparently conflicting rows arise because of different populations or heads, explain them; do not select the favourable scope after the fact. Nothing in this table authorises the recommended intervention.

| Finding under the frozen analysis | Recommendation |
| --- | --- |
| Restoration/reference validity or scoring coverage fails its frozen requirement; no useful candidates dominate; or intervals cannot support a classification | Report the specific measurement, bank or precision limitation. Do not attribute it to JEPA. Recommend one bounded measurement/design remedy, not an automatic sample expansion. |
| `U_G < delta` on the declared nominal scoring population, nominal harm satisfies its criterion, and restoration/scoring coverage is adequate | Command history is near-optimal within the declared tolerance on this tested scope. Recommend stopping further visual ego-motion optimisation for this task/interface and considering the section 9 redirect under a separate approval. Do **not** infer this from `H_scorer` and `H_motion` each being below delta. |
| `D_learned` is materially negative and the harm/coverage criteria pass | Report a learned decision benefit regardless of `H_motion`. Recommend a predeclared, precision-justified closed-loop confirmation before further fitting. Investigate whether the benefit reflects improved forecasting, compensation for scoring errors or another diagnosed interaction. |
| `H_motion` is practically equivalent to zero, `H_scorer` is materially positive, and a meaningful learned benefit is ruled out by `L_D > -delta` | True-motion substitution provides no material gain under the current scorer. Prioritise one shared scorer investigation or an explicitly new consequence interface. Do not claim motion prediction can never help. |
| `H_motion` is materially positive, `L_D > -delta`, `L_readout` is practically equivalent to zero, and `L_forecast` is materially positive | Recommend one targeted predictor intervention with matched controls, or another diagnostic only if needed to specify it. No intervention starts under this handoff. |
| `H_motion` is materially positive, `L_D > -delta`, and `L_readout` is materially positive | Recommend a representation/readout-interface investigation, with the existing forecast diagnostic reported alongside. Later representation comparisons in section 8 require separate approval. |
| A signed difference is materially negative in an unexpected diagnostic, several mechanisms remain inseparable, or none of the preceding rows is supported | Report the interaction or unresolved result explicitly. Do not force the audit into a repair-versus-redirect conclusion. Recommend at most one justified next step. |

In all cases report `L_readout`, `L_forecast`, `A_action`, shrinkage, harm and eligibility diagnostics. Loss localisation remains useful even when there is little baseline headroom. A two-component decomposition is not permission to bypass the direct `G` test.

### 6.7 Deliverables

1. **Result report:** `go2_decision_headroom_audit_result_<date>.md`.
   - Lead with the decision-table row and the primary intervals.
   - Put detail in appendices, with per-state records, per-layout summaries and bootstrap outputs in machine-readable form.
2. **Decision memo** (two pages at most):
   - the decision-table row;
   - supporting intervals;
   - the single recommended next step;
   - a proposed pre-registered evaluation for it.

   Do not execute the recommendation.
3. **The branch panel** as a versioned asset (section 8), with source/decision packets, physical outcomes, sampling roles, reconstruction checks and exclusion flags.
4. **Closure:** identify every assigned state/row and any failures, unscorable outcomes or exclusions; report actual resource use; link the frozen protocol and its explicit approval. Deliver the memo and stop at checkpoint (b). Do not promote a controller or silently continue the superseded roadmap.

## 7. Discipline and stop conditions

**Discipline:**

- One question per phase. No scope expansion.
- Continue existing practice: fresh output roots, source and checkpoint hashes, preserved failures, no overwriting.
- Simulator ground truth and actual future features are evaluator-only.
- After approval of the protocol freeze, do not change the reference cost, thresholds, samples, strata, masks, model/head identities, action bank or analysis in response to results.
- Ground truth and actual futures must remain separated from deployed/source-controller inputs; diagnostic substitutions are allowed only in the specified rows.
- If a protocol defect is found: stop affected work, document it, preserve the failed version and outputs, propose a successor protocol and affected reruns, and obtain approval for the correction before executing it. Report both versions. A code fix that changes the scientific calculation is not a routine continuation.
- Keep validation/scratch outputs separate from scientific attempt roots. Retain exclusive-create/overwrite protections. Routine checks should be limited to establishing the authorised treatment and valid evidence, not broad repeated infrastructure reviews.

**Stop and report, rather than improvise, if:**

- restoration cannot be achieved for a usable fraction of states within the pilot budget;
- no small reference-cost formulation passes the sanity panel;
- projected compute or storage exceeds the budget;
- any required change would alter the controller under test;
- branching, state reconstruction or scorer input requirements cannot be met faithfully;
- the next operation crosses an approval checkpoint or permission boundary.

Do not launch training, change the task or widen a budget as an improvised workaround. Routine progress reports should state completed evidence, current phase/budget and the next bounded step; do not count saved records as independent experiments or live progress as a completed result.

## 8. The branch panel as a representation benchmark

The thesis claim is about useful feature representations, so build the audit to be reused.

**Store per branch:**

- rendered frames at every consumed time;
- physical outcomes at all horizons;
- reference-cost components;
- post-limiter commands;
- restoration and variability flags;
- the linked decision-input packet, full restoration trace, source controller, layout exposure/role, sampling weight and candidate eligibility information.

**Record** the encoder checkpoint, preprocessing and resolution used for R3 and R4, so other encoders can later run on identical frames.

**Freeze layout-level roles for the panel before collection:** fit-eligible, selection and evaluation-only. Phase 2 fits nothing. These roles constrain any later approved readout fitting.

Evaluation-only data remain excluded from fitting, but they cannot remain an untouched final evaluation once their outcomes guide design choices. Record that exposure and obtain a fresh final population for later claims as needed. Do not relabel pilot/development/legacy V4 data as sealed or fresh final evidence.

The reason for these requirements (the comparisons themselves are not authorised now): the same frozen decisions can later score alternative representations under matched readout budgets:

- V-JEPA 2.1;
- a DINOv2/v3 baseline at matched resolution and token density;
- a small end-to-end JEPA trained on training-role data (LeWM-style);
- a supervised baseline.

The panel supports decision-level comparisons of representations. Attribution to the JEPA training objective additionally requires a separately approved design controlling architecture, training data, optimisation and other relevant differences; the listed encoder substitutions do not isolate that objective by themselves. Reuse of the audit method in another task family or embodiment is possible later, not authorised here.

## 9. After the memo (context only; not authorised)

If the memo recommends a redirect, the pre-agreed direction is an environment-dependent task. The representation should have a job the nominal model cannot do: anticipation before contact.

**Design requirements:**

- **Physical properties that change achievable motion,** such as friction or drag.
- **Visual cues whose relationship to those properties generalises.** Use property classes with procedurally varied appearance, and evaluate on held-out instances where the cue remains informative. Two things are forbidden:
  - random remapping between appearance and property;
  - a single memorisable texture per property.
- **A fair baseline** that adapts from proprioception after contact, with the same body sensing. Vision's legitimate advantage is anticipation before contact.
- **Decision points where anticipation can change a choice.** At current speeds, the 100–800-ms local horizon covers only on the order of 10 cm of travel. The design must therefore state whether forecasts act locally (slowing or turning at a boundary) or feed traversability costs to the router.
- **Nominal model unchanged.** Command history remains the nominal motion model. The learned pathway contributes corrections or consequence predictions, and every override is scored for benefit and for harm.

**Deferred,** each as a separately motivated configuration:

- RGB-only sensing;
- latency work: distilled V-JEPA 2.1 models, batching, fewer horizons;
- hardware.

## 10. Execution and resource plan

Use measured costs and hard caps, not elapsed-time estimates as execution authority.

| Phase | Execution bound | Required closeout |
| --- | --- | --- |
| Phase 0 | Existing four assignments, original budgets and resource limits; no duplicates or extensions | Every assigned outcome and physical reader, with development exposure retained |
| Phase 1 | Finite numerical pilot/source/sanity-panel, branch/repeat, compute and storage caps recorded before collection | Validity, variability, cost qualification, measured throughput and storage; proposed frozen protocol and budget |
| Checkpoint (a) | No Phase 2 collection or comparative scoring yet | Andrew's explicit approval of the identified protocol/config and budget |
| Phase 2 | Approved fixed layouts/states/rows, repeat counts and resource caps | Complete evidence accounting, paired layout-level analysis, versioned panel and decision memo |
| Checkpoint (b) | End of authority | Stop; any next intervention requires a new decision |

Recheck actual free space and live ownership when work starts; report-era PIDs and capacities are not current facts. Preserve failures and provenance. No silent retries with altered science, no effect-driven expansion, and no additional deletion permission are implied.

## Source documents and provenance

This handoff consolidates `go2_decision_headroom_audit_brief_2026-09-23.md` and the final review corrections into one execution specification. Its research evidence comes from `LeWMQuad_JEPA_World_Model_Progress_Report_2026-09-23.md`, especially sections 3, 6–10 and 11. Follow the report's existing source manifests for the actual experiment/checkpoint identities; do not invent missing IDs from names or historical status text.

The headroom rationale references ARC-Bench (`arXiv:2609.05461`) as in the source brief. No further literature review is required to carry out this authority, and that reference does not predetermine the audit's result.

**Final sequence:** finish Stage A unchanged → bounded blinded pilot → explicit protocol/budget approval → fixed decision audit → one decision memo → stop.
