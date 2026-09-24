# Design decision memo

## Decision

Proceed, in a later implementation pass, with `SAFE_LOCAL_WAYPOINT_TRAVERSAL` and `FACTORISED_RISK_CONSTRAINED_TRAJECTORY_RANKER`. Do not resume place-head work or claim global navigation.

1. **Intermediary goal:** reach a body-relative local waypoint safely.
2. **Why occlusion-safe:** the waypoint is route intent, not a visible beacon image; it can lie around a corner.
3. **Observation:** recent RGB/context, applied-control history, waypoint vector, and the twelve frozen candidates.
4. **Prediction:** H1–H3 motion, progress/completion, safety/clearance, and uncertainty for each candidate.
5. **Learned:** small factorised task heads and calibrated residual/support estimates; the JEPA stays frozen.
6. **Frozen:** encoder, one-/two-step predictors, action bank, control reconstruction, labels, safety contract, split, and thresholds.
7. **Unsafe rejection:** calibrated risk/clearance/support/runtime veto is an admissibility gate, never offset by progress.
8. **Uncertainty:** ensemble/seed disagreement where available, support distance, horizon disagreement, and calibrated residual intervals; abstain/brake if no candidate is admissible.
9. **Ranking:** lexicographic LCB distance, completion probability, LCB progress, uncertainty, control variation, candidate index.
10. **True-future upper bound:** apply the frozen planner to true H1–H3 latents and require geometry, safety, ordering, regret, and waypoint-success gates.
11. **One-seed test:** freeze the planner and compare seed `2026080901` one-step versus two-step; H3 primary.
12. **Closed-loop justification:** only a positive predicted-latent offline result with no material safety regression authorises the bounded maze segment test.
13. **Future memory/novelty:** known beacon routes through memory to local intents; unknown beacon uses count/frontier novelty first, then switches to route mode on detection.
14. **Novelty:** potentially novel is the tested combination and assurance sequence; JEPA planning, latent costs, topological memory, and novelty exploration individually have clear precedent.
15. **Prior art:** DINO-WM, V-JEPA, LeWorldModel, temporal-cost JEPA work, ViNG/ViNT/NoMaD, Neural Topological SLAM, Plan2Explore, Dreamer/TD-MPC2/MuZero.

## Stop rule

If true-future geometry or safety fails, stop without predictor attribution. If two-step does not improve local planning without safety regression, close this planner design and preserve the positive rollout/action-sensitivity findings as representation evidence only.

## Claim boundary

The eventual manuscript may say that the project evaluates whether action-conditioned JEPA futures support safe, body-relative local waypoint selection. It may not say that the current predictor solves beacon discovery, global route planning, safe navigation, or physical Go2 transfer.
