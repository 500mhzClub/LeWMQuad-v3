"""Memory modules for the topological navigation stack (v3).

``topological_memory`` defines the abstract ``Memory`` interface and the
``KeyframeMemory`` baseline (the goal image is the sub-goal — behaviourally
identical to the v2 planner). The learned topological memory (BeliefEncoder +
loop closure + top-k Bayes filter) is a later stage; see
``docs/v3_topological_nav_plan.md`` §5 and
``docs/lewm_topological_nav_stage1_retrieval_2026-06-09.md`` (Stage 1 decided to
build the BeliefEncoder).
"""
