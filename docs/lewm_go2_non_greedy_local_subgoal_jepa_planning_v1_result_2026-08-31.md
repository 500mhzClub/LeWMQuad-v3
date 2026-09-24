# NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1

Development-only exploratory result.

> JEPA route selection under oracle admissibility.

This is not JEPA safety and does not establish deployment safety, learned contact avoidance, physical Go2 safety, hidden beacon discovery, topological localisation, persistent memory, or complete maze navigation.

Primary classification: `NON_GREEDY_REACTIVE_OR_KINEMATIC_BASELINE_DOMINANT`.
Secondary classifications: `CANDIDATE_FUTURE_DERANGEMENT_MATERIAL`, `FUTURE_TIME_ORDER_DERANGEMENT_MATERIAL`.
Exact next experiment: `move the JEPA evaluation to topological or longer-horizon subgoal selection, rather than training another local route ranker.`.

## Panel

This is a constructed non-greedy challenge set, not an estimate of natural task prevalence.
The panel contains 96 unique scene/episode identities with role totals {'CALIBRATION': 16, 'DEVELOPMENT_HELDOUT': 16, 'FIT': 64}. All registered adequacy checks passed.

Adequacy: obstructed rays 96/96; two-or-more admissible 1.0000; direct/geodesic top-1 disagreement 1.0000; weak-or-negative direct progress for oracle-best 1.0000.

## Stage A

| Condition | Pairwise | Spearman | Kendall | Top-1 | Top-3 | MRR | Regret | Oracle fraction | Selected geo (m) | Selected Euclid (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DETERMINISTIC_KINEMATICS / KINEMATIC | 0.4981 | 0.0242 | -0.0039 | 0.0000 | 0.0000 | 0.1333 | 0.4365 | 0.3986 | 0.0795 | 0.0796 |
| NO_LATENT_NON_GREEDY_RANKER / NO_LATENT | 0.6117 | 0.2893 | 0.2235 | 0.0000 | 0.5000 | 0.3333 | 0.4365 | 0.3986 | 0.0795 | 0.0796 |
| CURRENT_VISUAL_REACTIVE_RANKER / CURRENT_VISUAL | 0.9479 | 0.9148 | 0.8958 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3206 | -0.0222 |
| TRUE_FUTURE_JEPA_TRAJECTORY_RANKER / TRUE_FUTURE | 0.9593 | 0.9541 | 0.9186 | 0.9375 | 1.0000 | 0.9688 | 0.0040 | 0.9935 | 0.3193 | -0.0241 |
| FUTURE_TRAJECTORY_DERANGEMENT / TRUE_FUTURE_DERANGED_CANDIDATE | 0.7642 | 0.6495 | 0.5284 | 0.3125 | 0.5625 | 0.4801 | 0.2616 | 0.5181 | 0.1862 | 0.0102 |
| FUTURE_TIME_ORDER_DERANGEMENT / TRUE_FUTURE_DERANGED_TIME | 0.8816 | 0.8352 | 0.7633 | 0.7500 | 0.9375 | 0.8385 | 0.0185 | 0.9727 | 0.3099 | -0.0221 |

Stage-A classification: `NON_GREEDY_REACTIVE_OR_KINEMATIC_BASELINE_DOMINANT`; predictor substitution authorized: `False`.
Candidate-trajectory derangement material: `True`; time-order derangement material (descriptive): `True`.

## Conditional Stage B

Stage B did not run because the frozen Stage-A gate did not pass.

## Scientific interpretation

The machine-readable metrics and independently regenerated raw score ledgers are authoritative. True-future trajectories are an observability upper bound; R1/RR are frozen predictor substitutions and no predictor was trained.

The true-future trajectory ranker passed every absolute route-signal check and materially outperformed deterministic kinematics: pairwise accuracy improved by 0.4612, normalized regret fell by 0.4324, and oracle-progress fraction improved by 0.5949. Candidate-trajectory derangement reduced pairwise accuracy by 0.1951, increased normalized regret by 0.2576, and reduced oracle-progress fraction by 0.4754, showing that candidate-specific future information affected the learned ordering.

However, true-future trajectories did not add registered route-selection value over the current-state visual representation. Against CURRENT_VISUAL, true future improved pairwise accuracy by only 0.0114 while regret was worse by 0.0040 and oracle-progress fraction was worse by 0.0065; zero of the three incremental criteria passed. CURRENT_VISUAL selected the oracle-best route in 16/16 states, while TRUE_FUTURE selected it in 15/16. The correct interpretation is therefore that current visual geometry was sufficient on this local challenge set; realised future trajectories were used, but did not improve selected actions over the reactive visual ranker.

The candidate-trajectory and time-order derangement secondary labels are descriptive diagnostic dispositions. Stage B did not run, so whether R1 or RR preserves the true-future route information—and whether rollout improves over one-step prediction—was not evaluated.

### Descriptive adverse selections over ALL_CANDIDATES

| Condition | Immediate contact | Nonviable successor | Stuck | Dead end |
|---|---:|---:|---:|---:|
| Deterministic kinematics | 0 | 0 | 0 | 7 |
| No-latent ranker | 0 | 0 | 0 | 7 |
| Current visual ranker | 0 | 0 | 0 | 0 |
| True-future ranker | 0 | 0 | 0 | 0 |
| Candidate-trajectory derangement | 0 | 0 | 3 | 1 |
| Time-order derangement | 0 | 0 | 0 | 0 |

These are descriptive simulated outcomes, were not training targets, and do not establish deployment safety.

## Prior development binding

The authoritative recovered predecessor remains `DEVELOPMENT_SCIENTIFIC_PAYLOAD_RECOVERED` / `KINEMATIC_BASELINE_DOMINANT`. Its artifacts were governance-only and supplied zero rows, tensors, or checkpoints to this experiment.

The previous local waypoint benchmark was largely solvable from immediate action kinematics and was not an adequate task on which to demonstrate incremental world model value.

The preserved predecessor classifications are `TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL`, `TWO_STEP_PLAN_AWARE_JEPA_COST_NO_SIGNAL`, `PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED`, `RAW_LATENT_GOAL_COST_NO_GO`, and `TRUE_FUTURE_LATENT_GOAL_COST_NO_GO`; `TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE` and `JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS` remain unsupported. The separate safety workstream remains `REQUIREMENTS_ACQUISITION_REQUIRED`, `PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED`, `SIMULATED_CONTACT_PROXY_SCOPE_ONLY`, `REPLANNING_INTERFACE_UNRESOLVED`, and `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`.

## Prohibited-action counters

All counters are zero: predictor training, safety-model training, closed-loop control, topological memory, graph input to the ranker, novelty, beacon discovery, custom Python audit hooks, and custom startup or forensic frameworks.
