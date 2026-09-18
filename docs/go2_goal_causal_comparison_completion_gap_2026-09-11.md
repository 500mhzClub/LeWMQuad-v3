# Causal-comparison completion gap

This is an assessment of goal coverage, not an execution protocol or selection
of a new study. The independent-layout population remains unexecuted and its
eight-diagnostic final review remains pending.

Section 10.D of `NOVEL_MAZE_WORLD_MODEL_NEW_THREAD_HANDOFF_2026-09-07.md`
requires online rollout on/off with the same frozen predictor, cost and action
interface. Section 10.F separately requires evidence for JEPA training, online
planning and memory contributions. Successful source checks or even successful
navigation by every arm in the current 32-case population cannot alone satisfy
that complete causal requirement.

`lewm/independent_round_trip_comparison_study_development.py` assigns four arms:
persistent JEPA, persistent supervised rollout, reactive, and current-pair JEPA.
The first two share their controller and differ in their assigned trained model.
The current-pair comparison changes the planning map while retaining tracking,
contact, mission, residual and other history. Its scope is persistent planning
map information, not every form of memory.

The reactive factory constructs `ReactiveFloorTransportController` without a
model or learned residual. It uses current geometry instead of forecast-based
candidate feasibility and learned scoring. Its source explicitly labels this a
whole-method comparison. It supplies a nonpredictive baseline but does not
isolate online rollout under the same predictor/cost/action interface.

## Consequence for the next study decision

The final policy review must distinguish the questions answered by the original
32-case definition from the additional same-interface rollout intervention
needed for the full goal. If the original definition is executed, retain that
causal limitation in its results and leave the broader requirement open.
If treatments, budget or model assignments change, define and check a prospective
successor before consuming independent-layout sensor data or outcomes. The
existing fixed-definition verifier does not authorize such changes by accepting
a different rationale string.

A same-interface intervention needs an explicit operational definition of
rollout off, including the data used by the unchanged cost and feasibility
interfaces. It must account for final-goal, intermediate-waypoint, reentry,
first-interval correction, hold recovery and empty feasible sets. The branch
inventory in `go2_extended_budget_worker_launched_and_ranking_scope_review_2026-09-11.md`
identifies why replacing only a final action or one waypoint score is incomplete.
A ranking-only intervention retaining forecast vetoes answers a conditional
ranking question and must not be relabelled as fully nonpredictive planning.

The operational definition, implementation and matched physical executions for
that additional comparison are still missing. No new treatment or controller
was selected here. Preserve the current native diagnostics and their original
controllers; use their completed outcomes in the pending policy review. No
navigation, independent-generalization, causal-advantage, real-time or hardware
qualification follows from this assessment.
