# Maze2 repeats the contact-penalty turn-selection mechanism

Completed read-only56970/PID2586362 exited0. Result SHA
9e2e78192de73340e9dbed6f1b2a47d9432176b89b1d572f9056800ea04a956d
under go2_supervised_maze02_turn_scores_v1_attempt_001; launch SHA
a459d8c001f10dccb69f7fe44759322047a6240fa7929f399a5d59dc97f9182b.
All3014 observations and3000 actual selections were included. Every complete
score output reconstructed exactly with the frozen original scoring function.

All3000 selections chose turns:1500 right and1500 left, with995 consecutive
direction changes. All six actions were feasible at every selection. No
translation phase, surface or segment veto occurred. At every turn, forward
offered greater geometric potential progress than the selected turn, but lower
utility after the original800ms contact penalty. The same reversal occurred for
left arc on2998 selections and right arc on2982. The first terminal was frame3003,
MISSION_TICK_BUDGET_EXHAUSTED, matching the complete original audit.

Mean forward geometric potential progress was19.718193664848524mm, including
both distance and alignment terms. Its mean uncalibrated contact score was
0.02095180695836963; at the original coefficient1.2 this subtracts
25.142168350043555mm from utility. Mean forward utility was therefore
-5.423974685195026mm. These are predicted score components, not measured motion
or calibrated contact probabilities. The analysis reconciles saved phase,
surface and eight-segment flags; it does not independently recompute geometry.

The finding extends the maze1 score diagnosis to every completed maze2 selection.
It supports the rationale for the already frozen100ms commitment-contact native
test, without changing that test or inferring an alternative physical outcome.
The original maze2 still has no arrivals or crossings and fails strict visibility
at three primary-camera frames. This readout neither repairs those failures nor
establishes full-cohort JEPA advantage, reliable navigation or deployment readiness.

Runner source SHA
24b26c6a56282d48d99c59005c4fc60c74f4c199343d0cd0cb240de6cb63c937;
protocol SHA
d5615de1bb73470a3e2c5db888f8eaf9e5e92a235663c6d8e0e75fe79a681bbf.
The numerical analysis directly reuses the frozen maze1 runner's summarize and
reconstruct functions, bound at
8c7119b3711bd27541275d58d746f48c8763073d187e04dc2c7e91b1f3941dc0.
Initial preflight51785 caught a string-versus-Path error in the new admission
wrapper before output creation. The wrapper was corrected before freezing or
execution. Corrected preflight2417 exited0 with1675 sources and18124 input bindings.
No original attempt was changed or rerun.

Hardware preflight:16 physical/32 logical CPUs, all affinity, CPU3.5%,
78,127,984,640 available RAM bytes, GPUs idle,690,267,357,184 artifact-free bytes
and21,357,699,072 workspace-free bytes. One CPU analysis process used an8GiB RAM
admission and16MiB exclusive output allowance above40GiB reserve; these are not
OS limits. The prior completed maze1 analysis supplied the representative
workload. During execution analysis RSS was approximately1.093GB, while the
single original maze3 worker progressed from tick208 through280 and onward.
Hardware after execution remained within available capacity and is saved in
the result. Reported211.0549084818922s includes final binding verification and
hardware inspection; it is not controller latency.

Independent read-only44216 exited0: verified both result/launch identities,
all1675 source bindings before/after, all18124 input bindings, the exact original
worker/audit identities, population/action/terminal agreement, all aggregate
feasibility/reason counts and mean-utility arithmetic. This independent check
did not repeat all3000 numerical score reconstructions, the raw controller audit,
or the original transitive runtime verifier. The completed analysis itself did
the full score reconstruction and source/input verification before/after.

No model was loaded, no native scene was launched, and no controller, horizon,
coefficient, command or queue was changed. Aggregate remains25 completed
raw-audited native episodes and zero verified round trips. Original handles19047,
37343 and14895 remain live; maze3 worker2585353/start tick131683685 is collecting.
The contact-horizon native pilot remains after the existing fixed queue.
