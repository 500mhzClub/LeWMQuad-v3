# LeWM Navigation Architecture Reset Review

Date: 2026-06-14

## Research-Scope Clarification

This review diagnoses why the current system does not reliably choose and follow
safe paths. Its proposed geometry-first deployment architecture optimizes for
solving that navigation task.

The project's primary objective is instead to test and extend the limits of
JEPA-style world models for embodied navigation. For that objective, explicit
geometry remains a teacher, oracle, probe, and evaluation instrument, but should
not replace the learned world model in the final control loop.

The JEPA-first research direction that supersedes the deployment recommendation
in this document is recorded in:

`docs/lewm_jepa_navigation_next_steps_2026-06-14.md`

## Decision

Stop adding heuristics to the current v43 / ego-depth navigation controller.
Freeze it as a research artifact and baseline.

Do not treat the current LeWM as the central planning representation for reliable
perception navigation. Keep it only where the evidence supports it:

- visual place/view recognition;
- short-horizon predictive features;
- optional retrieval/localization inputs;
- a baseline for future structured-representation experiments.

Reset the navigation architecture around an explicit spatial belief and a local
path tracker:

1. perception produces an egocentric free-space / occupancy belief with
   uncertainty;
2. onboard odometry registers that belief over a short temporal window;
3. a geometric local planner selects a collision-free centerline and tracks it
   using execution-calibrated primitive envelopes;
4. a topological memory handles long-range routing and goal-image retrieval;
5. a learned world model is optional and auxiliary unless a redesigned spatial
   model passes direct navigation gates.

This is a pivot in the role of the world model, not a rejection of the useful
engineering and representation-learning results already obtained.

## Bottom Line

The current base world model is a useful view/place encoder and a partially useful
forward predictor. It is not an appropriate sole substrate for the task we actually
care about: reliably choosing safe paths from perception.

The final demo does not validate the LeWM predictor as a planner. During the
successful seek, the main controller:

- routes with BFS over a graph built during a privileged scene-graph DFS tour;
- aligns primarily using mapping-time IMU bearings or raw-frame cosine;
- walks using hand-written heading hold and DWA-like clearance heuristics;
- avoids obstacles using rendered depth registered with simulator ground-truth
  pose;
- detects the final object with a pixel-space hue gate;
- performs final approach with a pixel-space colour-centroid servo.

The LeWM rollout planner is used only as a fallback. The successful ego-depth
artifact reports `fallback_blocks = 0`, so the predictor had no fallback decisions
to make in that run.

The demo therefore proves that the learned view-recognition/localization components
can participate in a heavily engineered route-replay system. It does not prove that
the base world model learned navigation, safe-path selection, or a generally useful
planning state.

## Scope Reviewed

This review traces the architecture and evidence across the relevant project
documents, code, and artifacts, including:

- the LeJEPA identifiability and strategy reviews;
- the full-checkpoint, A2/A3, scaled SIGReg/source-mix, and yaw-invariance results;
- rollout-prediction and receding-MPC experiments;
- nav-cost, energy-head, pose-aux, IDM, first-action, and task-aligned-head
  experiments;
- retrieval, BeliefEncoder, loop-closure, filter, GoalAdapter, and routing gates;
- Stage 4a/4b and wide-maze physical navigation findings;
- perception-depth occupancy, runtime-contract, and trajectory-safety findings;
- the final ego-depth v43 demo video, JSON artifact, logs, and controller code.

Primary final-demo artifact:

`.generated/topo_nav/ego_depth_v43_demo.json`

## What The Final Demo Actually Shows

### Endpoint result

The ego-depth v43 demo reaches the selected goal:

| metric | result |
|---|---:|
| initial distance | 5.41 m |
| final distance | 0.28 m |
| evaluation success | true |
| perceptual stop correct | true |
| falls | 0 |
| subgoals reached | 8 |

That is a real integration result. It is not a compelling reliable-navigation
result.

### Trajectory quality

| metric | ego-depth demo | privileged-grid reference |
|---|---:|---:|
| path length | **17.36 m** | 8.79 m |
| direct displacement | 5.41 m | 5.41 m |
| path/direct ratio | **3.21x** | 1.62x |
| yaw primitives | **215** | 21 |
| alignment blocks | **127** | 29 |
| head corrections | **84** | 21 |
| veto escapes | **64** | 4 |
| edge realigns | **6** | 1 |
| heading RMS | **0.337 rad** | 0.148 rad |

The video matches these numbers: long periods are spent rotating, re-aligning,
skimming walls, or recovering a route bearing. Success is obtained through
persistence and recovery budgets, not through consistently selecting a safe,
efficient path.

### Privilege and hand-engineering still present

The successful artifact explicitly records:

- `deployment_valid = false`;
- simulator-rendered depth;
- simulator ground-truth pose for occupancy registration;
- mapping-time IMU bearings for graph edges;
- a privileged DFS tour over the known scene graph;
- setup-time goal selection constrained to a route-valid, forward-routable,
  colourful terminal spur.

The final controller is a benchmark scaffold for component diagnosis. It is not
yet an autonomous perception-navigation policy.

## Question 1: Is The Base World Model Appropriate?

### Answer

Not as the central navigation/planning representation in its current form.

It remains appropriate as an auxiliary representation and scientific baseline.

### Evidence that it is useful

- Frozen seq4 raw place retrieval reaches about `R@1 = 0.43`, `R@5 = 0.64`.
- It recognizes salient, visible, goal-facing beacons well enough for local
  servoing in controlled open-space tests (`0.73-0.92` success).
- The predictor is action-conditioned.
- Multi-step rollout training substantially improves prediction over persistence
  for the seq11 track.
- The BeliefEncoder improves retrieval and supports useful passive-replay
  localization.

### Evidence that it is unsuitable as the nav core

- Projected latent distance vs graph distance has median Spearman rho about
  `0.03`.
- Same/adjacent and far-away places are nearly equally distant in latent space.
- Yaw is strongly decodable (`R2 about 0.81`) while position is weak
  (`R2 about 0.16`).
- A different heading at the same place is almost as far away as a different
  place.
- Raw latent L2 is anti-metric for fine local action ranking.
- The learned energy head, pose head, IDM path, first-action rankers, and
  task-aligned frozen heads all fail their deployed decision gates.
- Pose supervision improves decodability without improving first-action ranking.
- Predicted rollout features do not add enough collision/clearance information to
  replace geometric safety.
- The successful final controller deliberately avoids `plan_cost` between
  interior nodes because it is flat in corridors.

The model predicts a compressed global view representation. It does not retain
the explicit local geometry required to answer:

- which side of the corridor has clearance;
- whether an arc will sweep the body into a wall;
- where the free-space centerline lies;
- whether an obstacle is getting closer;
- which safe action makes the best local progress.

Those are the load-bearing questions for reliable navigation.

## Question 2: Are The Latent Representations Suitable?

### Answer

They are suitable for recognition and weak localization. They are not suitable
as a unified state for metric planning, safety, and control.

### The representation is overloaded

The current pooled 192-D latent is expected to serve all of these roles:

- view identity;
- place identity;
- heading;
- action-conditioned prediction target;
- goal cost;
- local metric;
- loop closure;
- collision/clearance state;
- planning state.

The experiments show that these roles conflict. The latent primarily preserves
what makes the next global view predictable and recognizable. Fine spatial
layout and action-level affordances are secondary.

### Single pooled vectors are the wrong bottleneck for local geometry

The task needs spatial correspondence. A pooled CLS-like vector discards where
in the image an obstacle, opening, or goal lies. The patch-token and coarse
spatial screens used so far did not solve retrieval/action ranking, but that does
not establish that spatial representations are unhelpful. It establishes that
the current from-scratch ViT features and shallow readouts do not expose the
required geometry.

The correct target is not "make the existing pooled vector slightly better."
It is an explicit structured representation, such as:

- egocentric occupancy/free-space with uncertainty;
- depth or traversability;
- local BEV features;
- spatial patch-token dynamics;
- recurrent belief over geometry and egomotion.

### Partial observability is real but not the main local-control blocker

History helps retrieval modestly, and the BeliefEncoder is useful. However:

- its registered retrieval gate narrowly failed;
- pairwise loop closure remains weak;
- its replay-filter pass has about `20%` false-merge rate and about `4x`
  fragmentation;
- filter coherence is measured on passive contiguous replay, not autonomous
  exploration and recovery;
- local safe-path selection still fails even when localization is adequate.

A better belief model is useful, but it cannot substitute for local geometry.

## Question 3: Is SIGReg Working Against Us?

### Answer

SIGReg is not the single root cause, and reducing its weight is not the fix.

The scaled factorial already answered that:

| setting | projected rho | retrieval@1 | eval std | action sensitivity |
|---|---:|---:|---:|---:|
| uniform SIGReg 0.09 | 0.030 | 0.276 | 0.695 | 0.0033 |
| uniform SIGReg 0.03 | 0.025 | 0.238 | 0.533 | 0.0003 |
| uniform SIGReg 0.01 | 0.025 | 0.172 | 0.371 | 0.0000 |

Lowering SIGReg:

- did not recover metric geometry;
- degraded recognition;
- reduced held-out dispersion;
- erased action sensitivity in the scaled screen.

Therefore, do not spend more time tuning `sigreg_lambda` on this architecture.

### The deeper objective mismatch

SIGReg still contributes to a structural tension. It pushes a negative-free
global representation toward an isotropic Gaussian marginal. Maze geometry,
free space, contact boundaries, cyclic heading, and topology are highly
structured and non-Gaussian. More importantly, the training objective rewards
predicting the next pooled view, not preserving a navigation-sufficient state.

An isotropic anti-collapse objective can coexist with useful navigation features,
but it does not create them. In this system it appears to help recognition and
training stability while leaving navigation geometry absent.

The correct response is to separate representation roles:

- keep anti-collapse regularization on an appearance/place branch if useful;
- do not require a geometry/occupancy branch to be an isotropic Gaussian;
- directly supervise or self-supervise spatial geometry, egomotion, and
  traversability;
- gate any representation on closed-loop action ranking and safety, not only
  prediction or retrieval.

## Question 4: Are The Learned Heads And Planner Appropriate?

### Learned heads

The general strategy of adding small heads to frozen pooled features is exhausted.

Repeated pattern:

| component | own/offline result | deployed result |
|---|---|---|
| GoalEnergyHead | ranking accuracy about 0.98 | worse nav than latent L2 |
| pose auxiliary | encoded metric decodability improves | first-action ranking fails |
| IDM | within-corpus decodability | no held-out transition gain |
| first-action ranker | some signal | regret/collision gates fail |
| task-aligned frozen heads | beat random on parts | fail minimum gates |
| final-block adapter | small improvement | still misses promotion |
| GoalAdapter | improves cell R@5 slightly | view R@1 regresses |
| LoopClosureHead | belief better than baselines | pairwise precision/recall gate fails |

Heads can reweight information that exists. They cannot recover local geometry
discarded by the representation or repair a mismatched training/deployment
contract.

Future learned heads are appropriate only when:

1. their input representation explicitly contains the required spatial state;
2. their target exactly matches the deployed decision;
3. the gate is closed-loop and scene-disjoint;
4. a simple geometric or action-only baseline is included;
5. the head is stopped immediately if it fails the registered gate.

### Global topological planner

Topological routing is still appropriate for long-range image-goal navigation,
but the current result is narrower than the name suggests.

The graph is built from a privileged DFS tour over the true scene graph. Seek
mostly retraces that recorded chain. Goal selection uses setup-time privilege to
choose a route-valid, forward-routable, alias-safe target. This demonstrates
goal-image retrieval and route replay, not autonomous mapping or novel path
planning.

The offline routing gate itself did not pass its registered margin versus a
random-local baseline. Its adopted fixed `k=8` lookahead is a density-specific
heuristic. Later traversal bypasses the learned local planner by replaying
edge direction.

Keep topological memory as a global layer, but rebuild its contract around
executed local paths and explicit uncertainty. Do not ask it to compensate for a
missing local spatial planner.

### Local planner

The current local controller does not plan a safe path. It attempts to hold a
single stored bearing for each edge and reactively avoids predicted collisions.
That is why it skims walls and oscillates when depth occupancy is noisy.

The next local controller should:

- plan through the local occupancy/free-space belief;
- track a centerline or short lookahead path;
- score continuous swept-body clearance;
- use execution-calibrated primitive envelopes;
- retain collision as a hard constraint;
- use learned rollout/value features only as a tie-breaker after safety.

## Why Continuing The Current Heuristic Path Is Low Value

The current progression has repeatedly followed this loop:

1. a component fails its registered gate;
2. a downstream state machine or constraint works around the failure;
3. the single v43 scenario improves;
4. the demo becomes less representative of the learned architecture;
5. another behavior defect appears on video.

Examples include:

- failed pairwise loop closure rescued by strict filter dynamics;
- flat corridor `plan_cost` replaced by align-and-forward edge traversal;
- failed goal identity by latent cosine rescued by pixel hue gating;
- lagging localization at the goal bypassed by perceptual latching;
- weak final approach replaced by colour-centroid servo;
- noisy ego-depth behavior rescued by larger persistence/recovery budgets.

These are valid diagnostics and engineering fixes. They are also evidence that
the architecture is no longer converging toward a simple, general learned
navigation system.

## Recommended Architecture

### 1. Explicit perception state

Primary local state:

`B_t = {egocentric occupancy/free-space, uncertainty, recent egomotion}`

Inputs:

- monocular RGB if preserving the current platform contract;
- onboard odometry/proprioception;
- optionally real depth only after an explicit platform-contract change.

Training may use simulator depth, geometry, contact, and pose as privileged
labels. Runtime may not consume them directly.

For monocular RGB, train the perception module to predict local traversability or
depth/occupancy explicitly. Do not rely on a pooled latent to contain it
implicitly.

### 2. Geometric local planning and tracking

Use the belief to produce a short local path or centerline. Select primitives
using:

- hard collision feasibility;
- minimum and low-percentile swept clearance;
- progress along the local path;
- heading/curvature cost;
- steering hysteresis;
- calibrated physical execution envelopes.

This layer owns "do not hit walls" and "pick a safe local path."

### 3. Topological global memory

Use visual place recognition to:

- retrieve the goal node from a goal image;
- localize probabilistically among known nodes;
- route over edges produced by actual traversals.

Each edge should store:

- executed local path / action trace;
- local-map snippets or path corridor;
- uncertainty and traversal quality;
- directionality.

The graph proposes a nearby route target. The local spatial planner decides how
to reach it safely.

### 4. Optional redesigned world model

If a learned world model remains a central research objective, redesign it around
the actual task:

- spatial patch tokens or local BEV, not only a pooled global vector;
- recurrent belief conditioned on image history, actions, and proprioception;
- explicit factorization into place/appearance, geometry/traversability, and
  motion/egomotion state;
- multi-step free-running prediction from the start;
- direct occupancy/depth/contact/egomotion auxiliaries;
- task-aligned goal-conditioned value or policy trained on counterfactual local
  actions;
- anti-collapse regularization applied only where compatible with the branch's
  semantics.

The redesigned model should not be promoted because its prediction loss or
retrieval improves. It must beat a geometric baseline on safe first-action
selection and closed-loop navigation.

## What To Keep

Do not discard:

- the scene corpus and held-out splits;
- the rendering and rollout pipeline;
- the physical gait and primitive registry;
- the evaluation harness and artifact discipline;
- the runtime local-obstacle interface;
- the ego-depth occupancy implementation as an upper-bound/debug source;
- seq4 LeWM as the recognition baseline;
- the BeliefEncoder/filter as a localization baseline;
- the topological graph and BFS utilities;
- the extensive negative results and registered gates.

The work has produced a strong diagnostic platform. The reset should use it.

## What To Stop

Stop immediately:

- tuning v43-specific recovery thresholds;
- adding more state-machine branches to improve the single demo;
- treating endpoint success on v43 as architecture validation;
- adding small heads to the same pooled latent without a new spatial substrate;
- tuning SIGReg strength as a route to metric navigation;
- using LeWM latent rollout probabilities as a hard safety mechanism;
- claiming the current ego-depth result as pure-perception navigation;
- investing in a larger generalization sweep of the current controller before
  fixing the local-planning architecture.

## Reset Gates

Run these gates before committing to a large new model.

### Gate A: Is local planning the dominant blocker?

Build the proposed centerline/clearance local planner first with privileged local
occupancy, while keeping global route targets fixed.

Evaluate across at least 8 held-out scenes x 2 goals x multiple physical seeds.

Pass criteria:

- zero contacts/falls;
- path/direct ratio materially below the current ego-depth `3.21x`;
- yaw/align/escape blocks reduced by at least 50%;
- route success preserved;
- no per-scene controller tuning.

If this fails, the main blocker is gait/path tracking rather than perception or
the world model. Fix that before any representation research.

### Gate B: Can perception support the local planner?

Replace privileged occupancy with ego-depth, then with the intended RGB-derived
belief.

Measure against privileged geometry:

- free-space precision/recall;
- false-free rate near obstacles;
- false-occupied rate on the centerline;
- swept-path collision prediction;
- occupancy stability during rotation and gait motion;
- downstream path quality and contacts.

Do not promote a perception model on pixel/depth error alone.

### Gate C: Does a learned spatial representation add value?

Compare, on identical held-out decisions:

1. geometric belief only;
2. geometric belief plus current LeWM pooled latent;
3. geometric belief plus a spatial/recurrent world-model feature.

The learned feature must improve path progress or efficiency without increasing
contacts. If it does not, keep it out of the deployed planner.

### Gate D: Is topological memory useful beyond route replay?

Remove privileged DFS route construction and setup-time route-valid goal
selection. Build the memory from an autonomous or fixed non-privileged
exploration policy.

Evaluate:

- goal-node retrieval;
- localization under recovery and revisits;
- graph connectivity and false shortcuts;
- success on goals not selected for forward-routability;
- improvement over odometry-only route replay.

If this fails, the topological memory remains a recognition experiment, not a
navigation component.

## Recommended Execution Order

1. Freeze and document the current v43/ego-depth controller as the baseline.
2. Implement continuous-clearance centerline local planning on privileged
   occupancy.
3. Validate physical path tracking across held-out scenes.
4. Swap in ego-depth and measure the perception gap.
5. Build the intended RGB-derived local belief with privileged training labels.
6. Re-evaluate topological memory only after local navigation is reliable.
7. In parallel, prototype a structured spatial world model, but keep it outside
   the deployed control loop until it passes Gate C.

## Final Verdict

Going back to the drawing board is justified.

The correct reset is not "throw away LeWM and start over from nothing." It is:

> Stop asking the current pooled LeWM latent and a growing set of heads to solve
> metric geometry, local safety, value estimation, localization, and routing at
> once. Use explicit spatial perception and planning for the parts that require
> geometry, retain LeWM where it has demonstrated value, and require any future
> learned world model to earn a larger role through direct closed-loop gates.

The final demo is useful because it makes this conclusion visible. The system can
eventually reach a selected goal, but it cannot yet reliably choose and follow a
safe path. More v43 heuristics would improve the artifact while moving further
away from the intended research claim.
