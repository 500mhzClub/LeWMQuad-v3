# Planner prior-art matrix

| Component | Representative primary source | Status for this project | Boundary |
|---|---|---|---|
| JEPA visual prediction | [V-JEPA 2](https://arxiv.org/abs/2506.09985), [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | Frozen project evidence | Prediction is not planning. |
| Action-conditioned latent dynamics | [DINO-WM](https://arxiv.org/abs/2411.04983), LeWorldModel | Established component | No priority claim. |
| Learned temporal/progress cost | [Temporal-Distance-JEPA](https://arxiv.org/abs/2607.25337), [Path-Aware cost](https://arxiv.org/abs/2608.14125) | CLEAR_PRIOR_ART | Our factorisation and safety gates remain to be tested. |
| Latent MPC/search | Dreamer, TD-MPC2, MuZero | CLEAR_PRIOR_ART | First experiment uses 12 exhaustive candidates, not CEM/MPPI. |
| Hierarchical subgoals | [PiJEPA](https://arxiv.org/abs/2603.25981), [hierarchical latent planning](https://arxiv.org/abs/2604.03208) | CLEAR_PRIOR_ART | Oracle local intent is a decomposition, not a hierarchy result. |
| Visual goal-conditioned navigation | [ViNT](https://arxiv.org/abs/2306.14846), [NoMaD](https://arxiv.org/abs/2310.07896) | CLEAR_PRIOR_ART | Beacon discovery is explicitly out of scope initially. |
| Topological memory/routing | [Neural Topological SLAM](https://arxiv.org/abs/2005.12256), ViNG/SPTM/ViKiNG lines | CLEAR_PRIOR_ART | Existing memory interface failures are preserved. |
| Frontier/novelty exploration | [Plan2Explore](https://arxiv.org/abs/2005.05960), frontier methods | CLEAR_PRIOR_ART | Start with visitation/frontier novelty. |
| Exact-reset same-state branch supervision | Project-specific evidence | KNOWN_COMPONENT_NEW_COMBINATION | Requires a documented comparison to prior counterfactual control work. |
| Body-relative waypoint intent under visual occlusion | Project proposal | POTENTIALLY_NOVEL | Only if the exact task/embodiment combination is absent after a fuller search. |
| Factorised geometry/progress/safety/uncertainty ranker | Project proposal | POTENTIALLY_NOVEL | Novelty is the combination and qualification protocol, not each head. |
| Safety as admissibility plus abstention | Safety-constrained control prior art | KNOWN_COMPONENT_NEW_COMBINATION | Must be calibrated and deployment-contract valid. |
| Future memory/novelty layer supplying local intent | Topological and exploration prior art | KNOWN_COMPONENT_NEW_COMBINATION | Do not claim a new memory algorithm. |

Restrained wording: “We evaluate a factorised, risk-constrained local planner that consumes action-conditioned JEPA futures and an oracle body-relative waypoint, with exact-reset candidate supervision and explicit true-future gates.”
