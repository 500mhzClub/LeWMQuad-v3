# NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1

Development-only, exploratory, one-seed, scene-disjoint non-greedy local-subgoal experiment.

> This is a constructed non-greedy challenge set, not an estimate of natural task prevalence.

The frozen question is whether current, true-future, R1, and RR V-JEPA trajectories change route selection when immediate Euclidean progress and command kinematics are insufficient. The primary target is H3 geodesic progress. Contact and successor viability define reported oracle populations and are never learned targets. Positive wording is **JEPA route selection under oracle admissibility.** It is never **JEPA safety.**

The panel has 96 unique scenes: 24 per WALL_DETOUR, U_ESCAPE, DEAD_END_LURE, and OFFSET_PASSAGE; each family contributes 16 fit, 4 calibration, and 4 development-heldout states. The complete candidate-blind eligible pool is fixed before role assignment and before encoder/model opening.

Exactly three rankers use seed 2026083101, AdamW 1e-3/1e-4, 60 epochs, final epoch only, one seed. Stage A and conditional Stage B gates are the exact contract literals. No predictor is trained. No closed-loop control, safety model, memory, routing graph input, novelty, or beacon system is run.
