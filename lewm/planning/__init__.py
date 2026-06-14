"""Planning modules for the topological navigation stack (v3).

Extracted from ``scripts/benchmark_lewm_closed_loop_mpc.py`` (Stage 0 refactor).
The seam is: ``primitive_bank`` (candidate action library) -> ``costs`` (pure
goal-image cost over rollouts) -> ``local_mpc.LocalMPC`` (Level 3 controller) ->
``hierarchical_planner.HierarchicalPlanner`` (Levels 1-3 orchestrator). All
modules here are genesis-free so they can be unit-tested with a fake model; see
``docs/v3_topological_nav_plan.md`` §4.1.
"""
