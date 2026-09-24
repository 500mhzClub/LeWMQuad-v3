# Primary literature review: learned JEPA planning (2026-08-19)

This review uses primary papers and project sources only. It is a design input, not evidence that any cited method has been reproduced here. Links are stable arXiv records unless an official project source is noted.

## Latent world models and JEPA planning

| Work | Core setting and relevance | Goal / cost / search / safety | Implication here |
|---|---|---|---|
| [DINO-WM, arXiv:2411.04983](https://arxiv.org/abs/2411.04983) | Visual latent dynamics for goal-conditioned control. | Latent prediction is paired with a planning objective; safety is not the central contribution. | Supports separating representation learning from the local cost and validating the cost on true futures. |
| [V-JEPA 2, arXiv:2506.09985](https://arxiv.org/abs/2506.09985) | JEPA video prediction and action-conditioned robotics transfer. | Predictive latent objective; downstream control is a separate stage. | Establishes the frozen target/predictor boundary used by this project. |
| [V-JEPA 2.1, arXiv:2603.14482](https://arxiv.org/abs/2603.14482) | Updated V-JEPA family for physical-world video prediction. | Representation is not itself a route planner. | Justifies retaining dense tokens and testing a small task head. |
| [LeWorldModel, arXiv:2603.19312](https://arxiv.org/abs/2603.19312) | JEPA-style world modelling for embodied navigation. | Demonstrates the relevance of action-conditioned latent futures; planning interface remains a distinct design choice. | Motivates action-conditioned quadruped evaluation, without claiming priority. |
| [What Drives Success in Physical Planning with JEPAs, arXiv:2512.24497](https://arxiv.org/abs/2512.24497) | Empirical analysis of physical planning with JEPA representations. | Highlights that representation quality and planner/objective choices jointly determine success. | Supports independent geometry, safety and ranking gates. |

## Learned trajectory costs and latent planning

[Temporal-Distance-JEPA](https://arxiv.org/abs/2607.25337), [Path-Aware World-Model Planning via Latent Trajectory Cost](https://arxiv.org/abs/2608.14125), [Latent World Models with Monotone Planning Costs](https://arxiv.org/abs/2608.09073), [Progress-Aware Hyperbolic World Models](https://arxiv.org/abs/2608.01926), and [Temporal Straightening for Latent Planning](https://arxiv.org/abs/2603.12231) all motivate learning a temporally meaningful or progress-sensitive cost rather than treating raw latent distance as a sufficient objective. They are prior art for learned progress/cost structure, not evidence that the proposed factorised safety-constrained ranker is novel. Their cost assumptions, embodiments, and safety contracts must be compared experimentally rather than imported wholesale.

## Search, hierarchy, and goal decomposition

[Policy-Guided World Model Planning / PiJEPA](https://arxiv.org/abs/2603.25981), [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208), [FF-JEPA](https://arxiv.org/abs/2606.09311), and [Delta-JEPA](https://arxiv.org/abs/2606.31232) cover policy guidance, hierarchy, or alternative latent transition structure. The proposed first task deliberately supplies an oracle local intent, making it a low-level qualification rather than a claim to solve hierarchical search.

## Visual and topological navigation

[ViNG](https://arxiv.org/abs/2012.09812), [Neural Topological SLAM](https://arxiv.org/abs/2005.12256), [ViNT](https://arxiv.org/abs/2306.14846), and [NoMaD](https://arxiv.org/abs/2310.07896) establish strong precedent for visual goal-conditioned navigation, temporal-distance or waypoint-like prediction, topological memory, and exploration-oriented control. Related NRNS, SPTM, and ViKiNG lines should be cited in a final manuscript after checking the exact task and memory contract. These works make “topological memory + local control” and “goal-conditioned visual navigation” clearly prior-art components.

## Exploration and novelty

[Plan2Explore](https://arxiv.org/abs/2005.05960) provides a primary reference for uncertainty-driven exploration in learned world models. NoMaD includes goal-masked exploration. Frontier and information-gain exploration are established robotics paradigms; the first implementation here should therefore prefer simple visitation/frontier novelty over raw prediction error.

## Broader comparisons

Dreamer-style actor-critic imagination, TD-MPC2, and MuZero provide established alternatives: value-equivalent planning, actor/value learning, and latent model-predictive control. They reinforce the distinction between a predictive representation, a decision cost, and a safety mechanism. None licenses a claim that the present JEPA predictor already supports planning.

## Design conclusions

1. A latent predictor is necessary but not sufficient: true-future planner qualification must precede predicted-future evaluation.
2. Goal-conditioned local intent is a defensible decomposition for an occluded beacon; the global memory/exploration layer can be added later.
3. Learned progress and safety should be factorised and calibrated separately.
4. Exact-reset, same-state candidate comparisons are the appropriate supervision for action ranking.
5. Novelty should be phrased as a tested combination and assurance protocol, not as a first-ever JEPA navigator or latent-cost planner.

## Required-paper record

The following compact records make the review auditable. “Not central” means the primary paper does not make that element its main claimed mechanism; it is not a claim that the implementation lacks all such machinery.

| Paper | Task / embodiment | Representation and goal | Cost/search | Safety, uncertainty, memory/exploration | Relation |
|---|---|---|---|---|---|
| DINO-WM | Visual control; embodied agents | DINO latent; goal-conditioned | Latent distance and model-based planning | Safety not central; no required topological memory | Prior latent planning |
| V-JEPA 2 | Robot video/action prediction | JEPA video latent; downstream task goals | Separate downstream controller | Transfer-oriented, not a safety planner | Frozen encoder/predictor precedent |
| V-JEPA 2.1 | Physical-world video prediction | JEPA dense latent | Downstream use | Not a beacon-search planner | Target encoder precedent |
| LeWorldModel | Embodied navigation/world modelling | JEPA-style predictive latent | Task-specific planning/control | Memory and safety depend on task | Embodied JEPA precedent |
| Physical-planning JEPA analysis | Physical planning | Joint-embedding predictive representations | Studies planner-sensitive success | Highlights representation/planner interaction | Supports independent gates |
| Temporal-Distance-JEPA | Temporal goal/progress prediction | Latent temporal distance | Learned temporal cost | Calibration/safety not the central novelty | Prior progress signal |
| Path-Aware latent cost | Model-based path planning | Latent trajectory encoding | Learned path-aware cost | Cost-centric, not this admissibility contract | Prior learned trajectory cost |
| Monotone Planning Costs | Latent planning | Monotone learned cost | Cost ordering/search | Monotonicity is the key constraint | Prior cost structure |
| Progress-Aware Hyperbolic WM | Goal-directed latent planning | Hyperbolic/progress representation | Progress-aware objective | Not a quadruped safety contract | Prior progress geometry |
| Temporal Straightening | Latent planning | Temporally straightened latent paths | Search over simplified latent trajectories | Safety not central | Prior trajectory geometry |
| PiJEPA | Policy-guided world-model planning | JEPA latent plus policy guidance | Guided search | Hierarchical guidance | Prior hierarchy/search |
| Hierarchical latent planning | Hierarchical embodied tasks | Latent subgoals | Multi-level planning | High-level decomposition | Oracle local intent is a deliberately smaller slice |
| FF-JEPA | Future-focused prediction/planning | JEPA future representation | Future-aware decision process | Task-dependent | Recent JEPA planning context |
| Delta-JEPA | Action/state change prediction | Delta latent | Change-based planning | Task-dependent | Supports explicit motion outputs |
| ViNG | Visual navigation | Visual goal/waypoint representation | Learned navigation policy | Robot navigation, not this safety factorisation | Clear visual-nav prior |
| Neural Topological SLAM | Visual mapping/navigation | Learned topological representation | Graph/topological inference | Memory-centric | Clear topology prior |
| ViNT | Goal-conditioned visual navigation | Goal-conditioned visual representation | Distance/waypoint-like policy | Local navigation | Local waypoint prior |
| NoMaD | Visual navigation and exploration | Goal-conditioned visual representation | Diffusion/policy-style action generation | Goal-masked exploration | Exploration and visual-goal prior |
| Plan2Explore | World-model exploration | Ensemble disagreement | Intrinsic information gain | Uncertainty-driven exploration | Later novelty layer |
| Dreamer / TD-MPC2 / MuZero | General model-based RL | Latent dynamics/value representations | Actor-critic, MPC, or value-equivalent search | Each has its own uncertainty/safety assumptions | Broader baseline families |
