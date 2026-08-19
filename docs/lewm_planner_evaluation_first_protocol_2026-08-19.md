# Evaluation-first protocol

## Stage 0 — freeze and evaluator

Freeze a scene-disjoint state/branch panel, waypoint construction, candidate bank, label definitions, safety contract, split, and result schema. Run synthetic perfect, reversed, tied, all-unsafe, and abstention fixtures. Run action-only and oracle reducers before training.

## Stage 1 — true-future planner gate

Fit the factorised ranker on exact-reset branches from training scenes. Qualify motion MAE/RMSE, heading error, safety AUROC/ECE, clearance calibration, pairwise safe-progress accuracy, waypoint regret, selected progress, success, and abstention on held-out true futures. Stop if geometry or safety is invalid; do not blame the JEPA predictor.

## Stage 2 — one predictor seed

Freeze the planner. Evaluate only seed `2026080901`, one-step and two-step, with no model-specific calibration. H3 is primary, H1/H2 diagnostics. Report gaps to true-future upper bound and all per-family values. A positive exploratory result requires two-step improvement in waypoint progress/success without material safety regression.

## Stage 3 — bounded closed loop

Only after Stage 2 passes, run SAFE_MAZE_ROUTE_SEGMENT_V1 with oracle local intents, 4–8 scenes, a fixed 12-cycle budget, identical controller and safety guard, and one seed. At each cycle predict H1–H3, apply admissibility, execute one block, observe, and replan. Do not replicate until the complete single-seed result is promising.

## Proposed gates

| Gate | Requirement | Failure disposition |
|---|---|---|
| Evaluator | deterministic complete schema | fix implementation only |
| Geometry | direction cosine ≥0.50; heading MAE ≤45°; endpoint error ≤ median edge | stop planner qualification |
| Safety | AUROC ≥0.75; ECE ≤0.10; no family collapse | stop predicted evaluation |
| True-future planning | success ≥0.70; normalized regret ≤0.25; pairwise ≥0.65 | stop; planner insufficient |
| Two-step offline | progress +0.05 or success +0.10 vs one-step; unsafe +≤0.02 | no closed loop |
| Closed loop | success +0.10 vs action-only; collision/fall +≤0.02; 3/4 families | exploratory positive only |

Thresholds are frozen before held-out scoring and are not significance claims.

## Minimum data plan

Use 16–24 decision states, 12 candidates each, 8/12/4 scene-family split for fit/calibration/test where available, H1–H3 labels, and atomic exact-reset branches. This is 192–288 branches, substantially smaller than a confirmatory multi-seed study. Reuse existing branches only where waypoint and safety labels are exactly compatible; otherwise collect a purpose-built local-intent panel in a later authorised pass.

## Compute and storage envelope

For 192–288 branches with RGB, dense H1–H3 targets, poses, safety and waypoint labels, reserve roughly 20–40 GB for compressed metadata/latents and 0.5–2 hours for branch collection on the existing simulator host (a measurement, not a promise). Planner fitting is expected to be minutes to tens of minutes on one GPU because the predictor and encoder remain frozen; reserve <2 GB incremental VRAM for the ranker. Calibration and evaluation should be CPU/GPU post-processing over the same shards. Closed-loop is capped at 4–8 scenes and 12 cycles per episode. No multi-seed or large-corpus compute is justified before this gate passes.

## Implementation sequence

1. Freeze panel, waypoint contract, labels, safety definitions, and thresholds.
2. Implement synthetic evaluator and action-only/oracle reducers.
3. Collect or validate exact-reset fit/calibration/test branches.
4. Fit and independently qualify motion, safety, and progress heads.
5. Apply the true-future planner gate.
6. Run one-step and two-step seed `2026080901` with the frozen planner.
7. Only after a positive result run the bounded receding-horizon maze segment.
