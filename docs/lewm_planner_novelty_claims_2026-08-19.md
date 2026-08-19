# Conservative novelty analysis

| Proposed element | Classification | Safe wording |
|---|---|---|
| JEPA latent prediction for planning | CLEAR_PRIOR_ART | “builds on action-conditioned latent world models.” |
| Learned temporal/progress cost | CLEAR_PRIOR_ART | “uses a learned progress signal.” |
| Twelve-candidate exhaustive selection | KNOWN_COMPONENT_NEW_COMBINATION | “evaluates the complete fixed candidate bank.” |
| Exact-reset same-state counterfactual supervision | KNOWN_COMPONENT_NEW_COMBINATION | “uses matched branch outcomes to supervise action ordering.” |
| Body-relative waypoint with hidden beacon | POTENTIALLY_NOVEL | “tests an occluded local-waypoint decomposition on a quadruped.” |
| Factorised geometry/progress/safety/uncertainty | POTENTIALLY_NOVEL | “proposes a factorised ranker with independent qualification gates.” |
| Safety admissibility and abstention | KNOWN_COMPONENT_NEW_COMBINATION | “treats safety as a constraint and supports abstention.” |
| Memory/novelty layer supplying local intent | KNOWN_COMPONENT_NEW_COMBINATION | “separates global subgoal production from local control.” |
| Complete combination as a first | REQUIRES_MORE_SEARCH | Never say “first JEPA navigator,” “first latent MPC,” or “first topological JEPA planner.” |

The defensible contribution is an experimentally gated integration: action-conditioned JEPA futures on a quadruped, exact-reset candidate supervision, body-relative occluded intent, factorised geometry/progress/safety/uncertainty, and a later memory/frontier interface. It becomes a scientific contribution only if true-future gates pass and the one-seed predicted planner shows a reproducible directional result; it is not supported by the current predictor evidence alone.
