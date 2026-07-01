# Potential Paper Plan

Date: 2026-06-17

Status: planning / audit synthesis. This document captures an audit conversation
about whether the LeWMQuad JEPA navigation program can become a publishable
paper, how novel it is relative to prior art, what a credible Go2 result would
require, and which JEPA architecture improvements are worth pursuing. It does
not register new gates or claims. It references the authoritative records:

- `docs/lewm_foundational_jepa_research_plan_2026-06-15.md` (mainline plan +
  full Phase 3A ledger);
- `docs/lewm_jepa_navigation_next_steps_2026-06-14.md` (failure analysis +
  redesign directions).

## 1. One-paragraph framing

The research question is not "can a robot reach a goal in a maze." It is whether
a JEPA-style learned world model, trained only on robot pixels, action history,
and (optionally) proprioception, can acquire a structured predictive belief
state that supports safe navigation action selection **without** pretrained
visual features (DINO) or privileged geometry (depth, occupancy, pose, route
breadcrumbs) at runtime. Privileged geometry, oracles, depth, and DINO are
allowed as teachers, labels, probes, and ceilings; they are forbidden as
deployed runtime inputs to the component making the navigation decision. That
boundary is the entire discipline of the project and the reason several
"successful demos" are explicitly logged as not counting.

## 2. The research arc (what happened, honestly)

### 2.1 The Go2 phase: real successes that did not validate the world model

The original system (LeWM = Learned World Model) encoded each observation into a
single globally pooled latent, trained mostly to predict the next observation's
latent under the logged action, with uniform SIGReg anti-collapse. On top of it
were built a topological navigation stack (recognition memory + hierarchical
planner, SPTM lineage) and a perception/occupancy depth controller.

These produced genuine end-to-end successes in simulation (wide-maze v27
blue-beacon goal, final 0.46 m, zero falls; Stage 4b goal-image-only maze, 4.00 m
over 5 hops to 0.36 m final; ego-depth v43 maze, final 0.28 m, perceptual stop,
cosine 0.99). They did **not** validate the learned world model:

- the route came from a privileged scene-graph DFS tour;
- execution was dominated by mapping-time bearings and reactive heading logic;
- local obstacle handling used simulator depth and ground-truth pose;
- the final object approach used a pixel-space colour servo;
- the best artifact reported `fallback_blocks = 0` — the LeWM rollout planner
  made zero decisions;
- the trajectory needed 215 yaw primitives, 127 alignment blocks, 64 veto
  escapes, path/direct ratio 3.21x.

The deeper diagnosis (Phase A gate): the frozen LeWM latent is a good
place-recognition code (retrieval@1 ~0.42, ~21x lift) but a poor metric code
(correlation with true geometry rho ~0.03; reachability head ~baseline). It
encodes appearance and heading strongly, metric position and free space weakly.

Six coupled structural causes (the redesign brief):

1. global pooling discarded action-relevant spatial structure;
2. the representation was an observation encoder, not a belief state;
3. observational prediction did not teach counterfactual action consequences;
4. the predictive horizon was too weak for planning (rollouts lose to
   persistence beyond the first step);
5. one latent was asked to satisfy conflicting roles (recognition wants
   heading-invariance; control wants heading-sensitivity);
6. uniform SIGReg prevented collapse without building navigation geometry
   (lowering it degraded recognition without recovering metric structure).

### 2.2 The first redesign attempt (still on robot data): Phase 2D -> 2AA

Before pivoting, the objective was repaired on the existing Go2 counterfactual
data. Almost everything failed the executable gate, but informatively:

- 2D/2E/2F/2G (spatial tokens, slot geometry, consequence head, action-utility
  head): collapsed and/or lost to persistence by large factors (24.96x, 229.69x,
  117.98x);
- 2H audit: the 81-way utility target is genuinely source-conditioned, not
  action-sequence bias — the target is fair;
- 2I-2N (FiLM, interaction-only, soft, class-balanced rankers): collapsed to a
  single global first move, did not use the source observation;
- 2O/2Q (factorized affordance target + true-factor ceiling): the **ceiling
  passes** (primitive match 0.867, regret ~0.001) — the target/selector contract
  is coherent; the problem is the learned state, not the task;
- 2R/2S (geometry bridges): **2S swept-geometry affordance passes** (match
  0.531) — first learned affordance-state diagnostic to pass; the missing
  variable is action-conditioned swept consequence geometry;
- 2T-2Z (RGB / occupancy / ray bridges back to that state): all failed the
  deployed action gate; single-frame RGB, dense single-frame occupancy, and
  local rays cannot recover the action-relevant state;
- 2W: found a cleaner privileged teacher target (drop explicit second-action
  identity, add light ranking) that passes;
- 2AA: DINOv2 patch features set up as a ceiling/control only, not the mainline.

Takeaways: (a) when handed the right consequence summary the system chooses
correctly, so the task is solvable and the scoring is right; (b) a single frame
does not contain enough — memory and motion over time are required.

### 2.3 The pivot: a 2D positive control (Phase 3A)

Decision: stop running expensive quadruped JEPA sweeps until the objective works
in a controlled setting. Build a deterministic 2D grid/maze world rendered to
pixels, expose only pixels + actions to the model, keep privileged state only for
labels/probes/oracles, generate same-source counterfactual branches for every
candidate action, include collisions, dead ends, turns, and aliased corridors.
Explicitly reject pivoting the mainline to a DINO-WM replication; DINO stays a
ceiling/control.

### 2.4 Phase 3A ledger (condensed; full detail in the foundational plan)

- 3A.1-3A.6: getting any learned action-sensitivity. Early runs lost to
  persistence and collapsed. The decision-token branch (3A.5) was the first to
  kill collapse, pass effective-rank, beat persistence at step 1, and pass
  zero + hard-negative margins.
- 3A.7: first narrow positive-control pass (medium 2D split), first-primitive
  receding-horizon; enabled by an observation beacon (ego-frame goal
  direction + topological distance). DINO unused.
- 3A.8/3A.9: memory. 3A.9 hidden-goal alias split is the first clean
  memory-dependence result — memory-on passes, paired no-memory ablation fails.
- Online-memory honesty check: a non-privileged egocentric marker-memory control
  claims 11/12 (and 14/14 reachable, and 16/16 with the explicit
  odometry-frontier planner). This proves the task is solvable from pixels +
  actions alone; every later failure is "our learner isn't good enough," never
  "the task is impossible."
- Learned spatial belief map: first learned-memory success on the tiny alias
  split (11/12).
- v4 novelty_then_claim + closed loop: 13/24 claimed; strong ablations
  (no-history claim 0.023, marker-colour-removed 0.000, shuffled candidates
  0.023, spatial-memory-disabled 0.000). Critical negative: deployed phase
  metrics were identical at steps 128-512, so the win was dominated by a direct
  RGB colour prior.
- v5 randomized palette (removes the colour shortcut): closed-loop collapsed to
  1/16, then 3/16 with larger memory; more history / h12 / current-marker all
  negative. Explicit odom-frontier planner hits 16/16, re-confirming the
  interface is sufficient and the learned policy is the gap.
- 3A.13-3A.21: replace scaffolds with learned parts one at a time. Learned
  JEPA-token map head, action-rolled egocentric memory (no global x,y), trained
  recurrent egocentric memory updater, multi-step value-field planner (the key
  abstraction the one-step heads missed), learned target-field head, learned
  extractor-mode head. 3A.21 became the selected artifact: learned perception +
  learned recurrent memory + learned target field + learned extractor; only
  value propagation/action extraction remained fixed.
- 3A.22-3A.28: remove the last hand-written pieces. One-step action distillation
  failed (1/16, collisions); dense value-map planner reached 11/16; ensembles
  13-15/16; learned marker-action return closed strict learned 16/16 (max96);
  learned route gate / learned router / counterfactual router closed strict
  learned 16/16 at max68 — but proven on a single validation set.
- 3A.29 + latest: broad disjoint split (seed 20260701, 8 train / 8 validation
  scenes, 64 source/goal groups) exposed the overfit; the later nine-seed broad
  aggregate improved but remains unsolved.

### 2.5 Where we are now (the broad-generalization result)

| Controller | Broad split (of 64) |
| --- | --- |
| Old selected strict controller | 37 / 64 |
| Exact odometry frontier ceiling | 62 / 64 |
| Exact learned latent-recurrent frontier ceiling | 60 / 64 |
| Structured latent-memory diagnostic, explicit fixed marker target | 61 / 64 |
| Best strict learned controller (current) | 59 / 64 |

The latest iteration added `--trace-output` for per-step failed-episode traces,
a `--latent-value-map-turn-oscillation-breaker` (a generic progress guard, not a
colour/side-wall shortcut, but still a hand-written readout rule) which recovered
two failures, and matching DAgger rollout support in the value-map planner
trainer. Verification: py_compile clean; 63 focused Phase 3A tests pass; no stray
training processes.

Honest interpretation: the learned JEPA-latent map and recurrent memory are now
close to the exact ceiling (60-62/64) — the map is stored well. The remaining gap
to 59/64 is **strict learned action selection**; the misses are mostly wrong
learned exploration/return actions, not failure to store the map. Strict learned
broad generalization is improved (37 -> 59) but not solved, against a real
ceiling of ~62.

The broader headline after extending to nine randomized-palette broad seeds is:

| Controller | Nine-seed broad aggregate |
| --- | --- |
| Old strict baseline | 518 / 576 = 89.9% |
| action05 comparator | 513 / 576 = 89.1% |
| Best strict learned aggregate found so far | 525 / 576 = 91.1% |

One bounded follow-up on 2026-06-19 tested a trace-action-preference router:
`models/checkpoints/phase3a_explore_claim/phase3a_v5_value_map_router_old_to_action05_trace_actionpref_oracle_train8_val47_pw2_512.pt`.
The label was positive only when the action05 fallback planner matched the trace
oracle and the old strict primary planner did not. This was negative: only
`98 / 10382` train examples and `22 / 1407` validation examples were positive,
and the trained router produced `0` validation true positives at threshold
`0.5`. A strict seed-`20260747` smoke eval stayed at `55 / 64` with
`fixed_marker_return=false`, `explicit_side_wall_fallback=false`, and
`fallback_after_step=999`.

Practical implication: do not optimize the claim around hitting `100%` on this
one random set. More broad seeds are expected to reveal more failures, and the
privileged planner/ceiling is itself not always perfect. The publishable
question is whether the learned JEPA state and learned readout improve
calibrated robustness across multiple held-out randomized splits under the
strict no-fixed-marker-return/no-explicit-side-wall-fallback contract.

A later 2026-06-19 threshold-calibration sweep on the best trace-outcome router
did not improve the nine-seed aggregate:

| Router threshold | Nine-seed strict aggregate |
| --- | --- |
| 0.50 | 525 / 576 = 91.1% |
| 0.80 | 518 / 576 = 89.9% |
| 0.95 | 518 / 576 = 89.9% |

The higher thresholds recover some of seed `20260739` but lose episodes on
other seeds. This reinforces the interpretation that the remaining problem is
not just scalar router calibration; it is learned action/readout behavior around
closed-loop failure states.

Claim discipline (unchanged): this is a controlled 2D positive control showing
the mechanism can be learned, with hand-written readout rules still in the loop.
It is not "JEPA learned navigation" and not "Go2 is solved."

### 2.6 Current Go2 memory bridge update, 2026-06-19

The Go2 translation path now has a concrete aggregate memory bridge, not just a
paper plan. The first fixed-slot recurrent RGB probe was rejected because it
could match or beat the normal result when recurrent state was reset, so it did
not prove temporal memory. The stronger matched-current-view causal probe asks:
for the same hidden current view, does prior visibility change the answer?

The query-conditioned slot probe trained on old causal rows plus the new medium
min4 shard and green top-up reached:

- normal matched-view balanced accuracy: `0.764`;
- recurrent-state reset: `0.462`;
- reversed input history: `0.437`;
- normal minus best ablation: `0.302`;
- validation rows: `74`;
- current validation queries: `87`.

This is enough to support the next engineering step: wire a Go2 hidden-target
memory controller and evaluate memory-on/off/shuffled closed loop. It is not
paper-grade yet. Red and yellow are strong, green is now measurable after a
validation top-up, but blue validation recall is still `0.000`. The result
therefore demonstrates translatability of the mechanism in aggregate, while
identifying an object-specific failure that must be either fixed or bounded.

Controller-facing update, 2026-06-20: the selected checkpoint now has an offline
target-selection gate, not just a per-query probe. The report
`.generated/go2_hidden_target_memory/go2_causal_memory_target_gate_slot_thr080_val_v2_plus_green_report.json`
asks whether the memory state would select the correct hidden target object or
abstain on matched current-view frames. At threshold `0.80`, normal recurrent
history reaches `0.783` balanced frame accuracy, `0.815` target-selection
precision, `0.857` negative-frame abstention, `0.710` positive-frame recall,
and `0.310` separation over the best corrupted-history control. Reset,
reversed-history, and shuffled-hidden-state controls are all materially worse.
This strengthens the translatability bridge, but it is still offline event-slice
evaluation. It does not replace the required Go2 command-block/closed-loop
memory-on/off/shuffled evaluation, and blue positive recall remains `0.000`.

Future-claim update, 2026-06-20: the target-gate evaluator can now join selected
targets back to full derived labels. On repaired validation, normal memory
selects `22` positive target frames but `0` of those selected targets have a
future claim/approach in the existing route-teacher labels. On medium-min4
training scenes the same metric finds `59` selected-positive future claims,
including `21` hidden claims, so the metric can detect closure when the rollout
contains it. Therefore the current Go2 bridge demonstrates learned memory
recognition and target selection, not return-policy closure. The next
publishable step is command-block execution or validation rollouts that actually
contain post-memory-activation return opportunities.

The extended causal audit explains the bottleneck: existing medium-validation
labels contain only `3` ambiguous seen-before rows with any future claim
opportunity, all in the blue group, and `0` ambiguous hidden-future-claim rows.
Because the current probe has `0.000` blue recall, this validation split cannot
prove learned future-claim closure without either fixing blue transfer or adding
return-capable validation rows for other factors.

This update changes sequencing: do not return to 2D latent-ordering research
before the Go2 closed-loop memory-on/off/shuffled demonstration. Latent ordering
is the next paper-version research loop after translatability is shown, because
it answers how much of the external learned-memory scaffold can move into the
representation itself.

JEPA-substrate correction, 2026-06-20: the Go2 CNN+GRU memory bridge above is
now explicitly treated as a baseline, not as JEPA transfer. A minimal
Go2 JEPA-style path has been added and documented in
`docs/lewm_go2_jepa_substrate_memory_update_2026-06-20.md`. The compact
action-conditioned Go2 latent encoder is non-collapsed and above chance on
next-latent retrieval (`0.177` vs `0.013` chance). A frozen-JEPA recurrent query
probe reaches matched-current-view balanced accuracy `0.626`, versus `0.424`
with reset recurrent state and `0.336` with reversed history (`+0.202` over the
best ablation). This is sufficient for the bounded claim that learned Go2 memory
can operate on a frozen JEPA-style substrate.

Later same-day follow-up strengthened the controller-facing frozen-JEPA bridge.
Adding a contrastive next-latent term to the Go2 JEPA trainer, training a direct
frame-level target gate, and selecting checkpoints against reset/reversed/
shuffled-history controls produced a usable offline controller proxy. At margin
`0.2`, the direct frozen-JEPA gate reaches `0.781` balanced frame accuracy,
`0.775` recall, `0.788` abstention, `0.816` precision, and `+0.209` over the
best corrupted-history control. Paired with a contrastive frozen-JEPA geometry
readout, the two-stage proxy reaches target recall `0.775`, false-claim rate
`0.212`, target-steering pipeline success `0.700`, and `+0.200` over the best
corrupted-history control.

Therefore the paper claim boundary moves one notch: "frozen-JEPA Go2 memory can
support an offline target-selection plus target-direction proxy under strict
history ablations." It still does **not** support "Go2 JEPA memory solves
hidden-target return" because no command-block execution or memory-on/off
closed-loop return evaluation has been run.

## 3. Novelty assessment

### 3.1 Similarity to Neural SLAM (high)

The Phase 3A pipeline is, structurally, a JEPA-substrate re-derivation of the
Neural SLAM / cognitive-mapping paradigm:

- **Cognitive Mapping and Planning (CMP), Gupta et al., CVPR 2017** — learned
  mapper writes first-person views into a latent egocentric map; a planner
  implemented as a Value Iteration Network (value iteration as conv +
  channel-wise max-pool) reads it; the map emerges unsupervised; plans under
  partial observability. This is essentially the 3A.14-3A.23 architecture. The
  3A.18 finding ("the missing abstraction was multi-step value propagation, not
  a better one-step head") rediscovers why VINs exist.
- **Active Neural SLAM (ANSL), Chaplot et al., ICLR 2020** — learned SLAM module
  -> 2xMxM occupancy map + pose; global policy proposes a long-term goal;
  analytical planner -> short-term goal; local policy acts.
- **Goal-Oriented Semantic Exploration / SemExp, Chaplot et al., NeurIPS 2020** —
  frontier exploration until the target object is found, then go to it. This is
  the explore-then-claim task.
- **Neural Topological SLAM, Chaplot et al., CVPR 2020** — maps onto the earlier
  topological nav stack.

Differences worth claiming: ANSL/CMP use a pose sensor and (ANSL) an analytical
planner; this project removes runtime pose/odometry and aims for learned (not
analytical) planning on a self-supervised JEPA substrate.

### 3.2 Other examples of the implementation pattern

Learned spatial memory + planner for navigation is a populated field: Neural Map
(Parisotto and Salakhutdinov 2017), MapNet (Henriques and Vedaldi 2018),
Differentiable SLAM-net (Karkus et al. 2021), MERLIN (Wayne et al. 2018), and
the Dreamer/RSSM line (recurrent latent world model + planning, but it
reconstructs, so not JEPA). The components are prior art; the contribution must
be the specific combination and constraints.

### 3.3 Learned memory on a JEPA world model

Recurrent latent memory in world models is old (Dreamer/RSSM). JEPA-specific
memory is just appearing. The current JEPA-navigation cluster almost universally
uses instantaneous latent state + MPC, no persistent memory:

- DINO-WM (Zhou et al. 2024): frozen DINOv2 patches + action-conditioned
  predictor + latent MPC; zero-shot goal reaching incl. mazes; no persistent map.
- PLDM, V-JEPA 2 / V-JEPA 2-AC (Meta 2025): action-conditioned latent +
  planning; no accumulated spatial memory.
- "What Drives Success in Physical Planning with JEPA-WMs" (arXiv 2512.24497,
  late 2025): beats DINO-WM and V-JEPA-2-AC; latent-space planning, not a memory
  contribution.
- "Hierarchical Planning with Latent World Models" (arXiv 2604.03208, 2026):
  hierarchical planning over PLDM/DINO-WM/V-JEPA2 for larger unseen mazes — the
  closest "make JEPA nav generalize to unseen mazes" framing, but via planning.
- HanoiWorld (arXiv 2601.01577, 2026): a JEPA world model that adds RSSM-style
  recurrent latent memory for partial observability (AV control), with a flat
  recurrent state, not a learned spatial map + frontier exploration.
- PiJEPA: instruction-conditioned goal nav with a JEPA WM + MPPI planning;
  frozen encoder, no spatial memory.

Net: the precise cell — self-supervised JEPA latent perception -> learned
persistent egocentric spatial memory -> learned planner -> frontier-explore-then-
claim, no runtime pose/geometry/pretrained features — is genuinely under-occupied,
but it is an intersection of two mature literatures (Neural SLAM spatial memory
and JEPA latent planning), i.e. combinatorial novelty, which reviewers weight
lower than conceptual novelty.

### 3.4 Verdict

Moderate-to-low novelty as currently scoped, concentrated in the wrong place for
a methods paper. The architecture largely re-derives CMP + ANSL + SemExp with a
JEPA front-end. The diagnostic discipline and negatives (recognition-strong/
metric-weak rho ~0.03, counterfactual action-identifiability gates, persistence
failures, colour-shortcut/memory-leak catches, ceiling-vs-learned 59 vs 60-62/64)
are the most original content, but a pure diagnostic paper is explicitly not the
target (see 4.3). The novelty must instead come from synthesis + property +
regime, defined next.

### 3.5 The components are saturated; novelty must be synthesis + property + regime

Decomposing the design space, every individual ingredient is already published,
mostly in 2024-2026:

- JEPA + recurrent memory: HanoiWorld (2026); the Dreamer/RSSM line generally;
- metric/topology-preserving latent: Quasimetric RL; "Repairing Latent World
  Models with Reachability Metrics" (2026);
- equivariant spatial memory: Flow Equivariant World Models (2026), EgoMap,
  Neural Map;
- explore-to-find-then-return: SemExp (2020);
- explore by where the world model is uncertain (latent-disagreement / intrinsic
  exploration): Plan2Explore (Sekar et al., 2020).

Consequence: there is no novel *component* left to discover in this design space.
Any plan of the form "find the one clever new module" will reinvent one of the
above. The contribution must therefore be one or more of:

1. **Synthesis + demonstration** — a system that does what none of the above do
   individually, proven on real hardware under the strict no-privilege
   constraints (Section 5), with causal ablations.
2. **Property claim** — show that a *self-supervised* JEPA (no rewards, no pose,
   no pretrained features) yields locally-metric / globally-topological latent
   structure, and that this structure is *causally responsible* for transfer to
   novel scenes. The novelty is the demonstration + causal ablation, not the
   existence of metric latents (already known).
3. **Regime** — target the discover-then-return POMDP (goal initially hidden,
   must be found, then routed back to), not the given-goal-image latent-MPC
   regime that DINO-WM / PLDM / V-JEPA2 address. See 6.1.

This is how many strong CoRL/RSS systems papers work: known parts, but a system
and a regime that are new, with the baselines carrying the argument. The
baselines (Section 5.4) are not garnish; they *are* the contribution's evidence.

## 4. Publishability

### 4.1 Replacing hand-cranked parts with learned rules: necessary, not sufficient

Removing the turn-oscillation breaker, fixed value propagation, and
phase-conditioned extractor closes the project's own claim discipline but does
not by itself create a contribution. Reviewer blockers:

1. it is a 2D rendered gridworld (the biggest blocker); CMP/ANSL solved the
   photorealistic version years ago;
2. heavy prior-art overlap with CMP/ANSL/DINO-WM;
3. no competitive peer baselines yet (only odom/oracle ceilings).

### 4.2 What would make it publishable, by leverage

- transfer off the gridworld (Habitat/photorealistic or physical Go2) — highest
  value, converts "sandbox" to "system";
- an ablation isolating the JEPA substrate's value vs a from-scratch encoder,
  DINOv2 patches, and a supervised mapper, under matched compute and no runtime
  pose, on novel-appearance generalization (the randomized-palette axis is ideal);
  if JEPA does not win, the honest "JEPA is not necessary here" is also a finding;
- lead with the negatives as a diagnostic contribution (TMLR / RLC / workshop),
  achievable without leaving the gridworld.

### 4.3 Realistic placement

As-is (all parts learned, 2D only, good ablations): workshop or TMLR/RLC, not a
top-tier main track. A main conference needs transfer or a head-to-head win over
DINO-WM/ANSL-class baselines.

Decision (made 2026-06-17): target the **system paper**, not a diagnostic study.
This commits to transfer (physical Go2, Section 5) plus competitive baselines and
the JEPA-substrate ablation (Section 5.4), under the synthesis + property + regime
framing (3.5). The existing negatives still appear, but as ablations supporting
the system claim, not as the headline.

## 5. The Go2 target ("Physical Go2, strict constraints, dozens of novel mazes")

### 5.1 The decisive fork: which Go2 demo is it

The 2024 ego-depth Go2 run already solved novel mazes and counted for nothing
(privileged route, GT pose, sim depth, colour servo, `fallback_blocks = 0`). The
only question that matters: does the Go2 result hold the project's boundary
conditions at runtime?

1. no runtime pose/odometry from the simulator (real IMU + leg odometry, which
   drifts, is allowed; Genesis ground-truth (x,y,yaw) is not);
2. no privileged depth/occupancy — perception from the JEPA latent over onboard
   RGB;
3. no pretrained features, no colour servo, no hand-route;
4. learned planning actually in the loop (the Go2 equivalent of
   `fallback_blocks > 0`, with explicit gates/breakers off).

If any leak, it is the 2024 demo again with a better narrative.

### 5.2 Real Go2 vs Go2-in-Genesis are different papers

- Go2 URDF in Genesis = embodiment change, still simulation; useful, not
  decisive.
- Physical Unitree Go2 = a genuinely different claim and the hard one (sim-to-real
  perception gap on a model trained on textured renders, plus a locomotion stack
  under the navigator). Memory flags the current perception path as
  "deployment-INVALID (sim depth+pose)"; closing that is most of the remaining
  work and where the publishable surprise lives.

### 5.3 A demo is not an evaluation

"Solve a novel maze" (N=1) is a teaser. A result needs success rate + SPL over
many mazes x goals x seeds; a defined novelty axis (layout? appearance? scale?);
and failure analysis (use `--trace-output`).

### 5.4 What must ship alongside the demo

1. peer baselines on the same robot/task (DINO-WM, an ANSL-style learned-map +
   analytical-planner, an end-to-end RL/Dreamer agent) under matched inputs;
2. the JEPA-substrate ablation (swap JEPA latent for from-scratch supervised
   encoder and for DINOv2 patches, memory+planner fixed);
3. all hand-cranked readouts off, reported as such.

### 5.5 Verdict by scenario

- physical Go2, strict constraints held, ~dozens of novel mazes with
  success/SPL, >=2 baselines, JEPA ablation: strong CoRL / RSS / ICRA paper,
  plausibly NeurIPS/ICLR with the self-supervised-memory framing — the version
  worth building;
- Go2-in-Genesis, strict constraints, quantitative eval + baselines: solid
  mid-tier / workshop-to-conference; capped by "still sim";
- Go2 single-video demo with any lingering privileged input: not publishable as
  a learned-world-model claim.

The trap to name: the failure mode is not "can't finish a maze" (already done).
It is finishing while the learned components actually drive, under onboard-only
sensing. That distinction is the entire difference between a demo and a
contribution.

## 6. Architecture improvements worth pursuing

Sequencing update, 2026-06-19: the architectural ideas below are still the
right route to novelty, but they should not block the Go2 memory-controller
translation. The immediate target is a concrete Genesis-Go2 hidden-target
controller with memory-on/off/shuffled/reversed controls. Once that works, return
to 2D as the fast lab for proving latent ordering, then re-test the successful
latent objective on Go2.

### 6.1 Reframe: the thesis, the regime, and why the JEPA earns its place

In the current pipeline the JEPA is a perception feature extractor and
memory + planning are bolted on, inviting "the contribution is an ANSL stack; the
JEPA is incidental." To stand out, move memory and prediction inside the JEPA and
make the JEPA earn its place by providing **two things from one self-supervised
objective** that the alternatives split or lack:

1. a **locally-metric, globally-topological latent** -> the memory/map and the
   return-phase latent planning;
2. a **predictor whose uncertainty signals where to explore** -> the exploration
   phase, replacing the hand-written frontier rule with a principled learned
   signal.

Thesis sentence:

> A single self-supervised JEPA yields both the metric spatial structure needed
> to plan a return and the predictive uncertainty needed to drive exploration;
> this lets one model discover-then-return in novel mazes, onboard-only, where
> given-goal latent-MPC models (DINO-WM/PLDM) cannot operate and supervised-map
> systems (ANSL) need pose and labels.

**The regime is the differentiator.** Latent MPC (gradient descent / CEM / MPPI
through the predictor toward a goal latent) — what DINO-WM, PLDM, V-JEPA2 do —
only works when the goal latent is known and reachable within the reliable
prediction horizon; they assume a *given goal image* in a *largely observable*
setting. This project's task — goal hidden behind walls, must be discovered, then
returned to — is a POMDP. No latent structure lets you gradient-descend toward a
latent you have never observed; that is information-theoretic, not a
representation flaw. The only valid architecture is therefore structured-latent
perception + memory + (exploration objective for the unseen-goal phase, latent
planning for the return phase). Needing exploration and memory is not a
regression; it is the correct and unavoidable structure for the discover-then-
return regime, and that regime is what given-goal latent-MPC papers do not
address.

**Topology vs metric.** "Preserve topology" (which places connect to which) is
sufficient for routing but not for control (clearance, collision). The target is
**locally metric, globally topological** — which is exactly what a map is. The
contribution is not inventing that object; it is claiming the self-supervised
JEPA latent + memory *is* that map and *causes* generalization.

**How generalization actually works (and where it breaks).** Generalization
comes from a split between scene-general operators learned once and a per-episode
map built fresh:

- perception (observation -> local latent geometry): "walls look like walls,
  openings like openings" is scene-general; learned once, transfers;
- dynamics (how the latent changes under an action): scene-general; transfers;
- the map is rebuilt every episode from those operators, so a novel maze yields
  a novel map from the same operators;
- equivariance handles rotations/translations structurally instead of memorizing
  them, covering unseen configurations.

It breaks if perception/dynamics overfit to training-maze statistics (textures,
corridor widths, lighting), in which case the operators do not transfer. This is
precisely what the randomized-palette + disjoint-seed splits test; the 37 -> 59/64
jump is the partial-but-incomplete answer. Generalization is thus an empirical
claim to be won, not a guarantee the architecture provides.

**Residual risk to state plainly.** Plan2Explore already did latent-uncertainty
exploration (in a reconstructive world model) and Quasimetric RL already did
metric latents, so reviewers will ask what is new beyond combining them. The only
valid answers are (a) self-supervised + no privilege + real robot + novel-maze
transfer, and (b) the discover-then-return regime those works do not address. If
the system cannot beat DINO-WM / ANSL / Plan2Explore-style baselines in that
regime, there is no paper.

### 6.1a What "encode geometry into the latent" actually means

The representation contribution is often stated as "stop recovering geometry from
the latent with supervised heads; encode it directly." That is right at the high
level but contains a trap the project already fell into, so it must be stated
precisely. Geometry currently lives in three places, and only two are the gap:

| Geometry | Where it lives now | Verdict |
| --- | --- | --- |
| 1. Local per-frame ("what is around me") | Decoded by a supervised head (`Phase3ALatentMapHead`) from JEPA tokens, ~100% acc | Fine; keep decoding. The info is in the latent and a readout is unavoidable (even CMP/ANSL decode a map). |
| 2. The accumulated map ("the layout") | External egocentric grid, rolled by odometry/actions | Hand-built, outside the latent |
| 3. Metric/relational structure ("how far, which way, reachable") | External value propagation / BFS; latent itself is rho ~0.03 here | Hand-built, outside the latent |

So the gap is not "we decode geometry with a head." Decoding local labels is fine
and permanent. The gap is that #2 (the map) and #3 (the metric) live in hand-coded
machinery, not in the learned representation. The latent has the *content* (a wall
is readable) but not the *structure* (latent distance says nothing about traversal
distance).

Critical refinement: "encode geometry into the latent" is NOT "add supervised
geometry targets so the latent predicts occupancy." That naive reading is exactly
Phase 2R/2T/2Z, which added occupancy/geometry supervision and all failed the
action gate. Decodability of geometry is not navigational structure. The useful
meaning is to shape the latent's *structure* via the objective, not via more
decode heads:

- metric: latent distance approximates traversal/reachability cost (quasimetric/
  reachability objective), so "how far is the goal" is a latent property, not a
  BFS over an external grid;
- equivariant: an action transforms the latent predictably (SE(2)), so
  accumulating observations into a map is a structural latent operation, not an
  odometry-driven grid roll with a geometric prior;
- counterfactual: action-conditioned prediction in that latent is reliable, so
  "what if I go left" is evaluated in the latent, not by replaying a hand-coded
  transition.

Why this is the through-line of the whole project: the scaffolds the project keeps
trying to delete (odometry rolling, value propagation, BFS, side-wall gate,
turn-oscillation breaker) exist *because* the latent lacks metric/equivariant
structure — something has to do that job, so it was hand-coded. They cannot be
cleanly removed while the latent is rho ~0.03, because then nothing computes
distances or integrates space. Put the structure into the latent via the
objective and the scaffolds become redundant rather than removed: planning
collapses to reading latent distances, memory collapses to accumulating
structured latents, and generalization follows because the structure is a
scene-general property learned once.

What does not go away: memory. The POMDP forces accumulation (a whole maze does
not fit in one frame's latent). But the memory's content and operations become
latent-native (a field/set of structured latents, queried by latent geometry)
instead of a hand-built occupancy grid with odometry. Memory stays; the hand-coded
geometry of the memory is what moves into the representation.

### 6.2 Landscape caveat (position, do not collide)

The two most natural improvements are no longer greenfield in 2026:

- equivariant spatial memory: Neural Map (effectively equivariant), EgoMap
  (action-conditioned, Spatial-Transformer, approximately equivariant), and
  Flow Equivariant World Models (arXiv 2601.01075, 2026) formalizing
  memory-under-group-symmetry;
- reachability/metric latents: Quasimetric RL, contrastive temporal distances,
  and "Beyond Euclidean Proximity: Repairing Latent World Models with
  Horizon-Matched Reachability Metrics" (arXiv 2605.22164, 2026) — essentially
  this project's rho ~0.03 / "L2 is anti-metric" finding with a proposed fix.

This validates the diagnosis and provides off-the-shelf tools; novelty must live
in the integration on a JEPA substrate, on real hardware, under strict
constraints, which those papers do not do.

### 6.3 Ranked bets

Tier 1 — fix the root cause, become a JEPA paper

1. Reachability/quasimetric-structured latent (highest leverage). Objective so
   latent distance approximates time/cost-to-reach, not appearance. Direct cure
   for the central documented failure. Converts the latent from a
   place-recognition code into a planning metric. Beat/cite Quasimetric RL,
   contrastive temporal distances (2406.17098), 2605.22164. Risk: must work in a
   self-supervised JEPA from pixels, not a reward-based GCRL setup — that is the
   novel slice.
2. Fold the egocentric memory into the JEPA as an SE(2)-equivariant recurrent
   belief. Today the memory is hand-rolled by the action (3A.14) — a manual SE(2)
   transform. Make it the JEPA's learned equivariant action-conditioned
   prediction over map-latent tokens. Unifies perception + memory + world-model
   into one learned state (kills the "incidental JEPA" critique); equivariance is
   what gives novel-layout/appearance generalization. Beat/cite Flow Equivariant
   WM (2601.01075), EgoMap, Neural Map. Risk: equivariance is crowded, so it must
   be a component of the metric-map story, not the standalone claim.

Tier 2 — required for "the JEPA drives the decision" and real-robot credibility

3. Counterfactual action-identifiability as the primary objective (finish Phase
   3B). The gates already encode it (hard-negative/zero-action margins beating
   persistence). Without it the planner is memory-driven, not JEPA-driven.
4. Epistemic uncertainty (ensemble or variational JEPA; see Variational JEPA,
   arXiv 2601.14354). Provides the safety story a physical-robot paper needs;
   "uncertainty predicts rollout failure" is a clean figure.

Tier 3 — horizon/abstraction, only after Tier 1

5. Temporal abstraction in the predictor (multi-step jumps) for planning depth;
   connects to Hierarchical Planning with Latent World Models (2604.03208).
   Caveat: do not build a hierarchy-of-predictors on a non-metric base —
   recognition-not-metric propagates up a level. Fix the metric first.

### 6.4 The result that makes reviewers care

Not "16/16 on a Go2." The causal ablation figure: metric structure (rho, or
reachability-correlation) and novel-maze SPL move together, and knocking out
either the reachability objective or equivariance collapses both. This shows
*why* it generalizes — an emergent metric cognitive map from self-supervision —
which is a NeurIPS/ICLR-flavored claim, not just a CoRL system.

### 6.5 What to stop doing

- more one-step policy/action-head distillation sweeps (3A.17/3A.22 already
  showed the wrong abstraction);
- equivariance as a headline (it is a means to the metric-map end);
- any hierarchy-of-JEPAs before the metric latent works.

### 6.6 If one bet: the reachability/quasimetric latent

It is the documented root cause, it turns the work from "ANSL with a JEPA
encoder" into "a JEPA that learns navigable geometry," and the 2026 repair paper
both proves the problem is real and gives a baseline to beat. Layer equivariant
memory-folding on top once it moves rho; that pair is the distinctive paper.

## 7. Recommended sequencing

Update, 2026-06-19: this sequence was too strict about solving latent ordering
before Go2 re-entry. The corrected milestone order is:

1. Treat the 2D learned-memory result and cheap-probe audit as complete for the
   current bridge milestone.
2. Translate the learned-memory scaffold to Genesis-Go2 under the strict runtime
   boundary: RGB, executed command blocks, and onboard-like proprioception only.
3. Use Go2 memory-on/off/shuffled-history ablations to prove the 2D mechanism
   transferred.
4. Then return to 2D as the fast lab for reachability/quasimetric and
   equivariant-memory objectives, requiring latent structure metrics and
   closed-loop SPL/success to move together.
5. Re-test successful latent-ordering variants on Go2.

Progress note, 2026-06-20: the bridge has started but is not finished. Durable
Genesis-Go2 hidden-target labels, rendered event slices, RGB/label joins, and a
supervised memory-probe trainer now exist. The first fixed-slot probe was
rejected because reset-state ablations matched or beat it. The later
query-conditioned matched-current-view causal probe is aggregate-positive:
balanced accuracy `0.764` versus reset `0.462` and reversed-history `0.437` on
repaired validation. The offline target-selection gate is also
aggregate-positive: balanced frame accuracy `0.783`, precision `0.815`, and
normal-minus-best-corrupted `0.310`. A label-backed future-claim check is
negative on repaired validation (`0/22` selected-positive future claims), while a
training-scene sanity check finds future claims when the rollout contains them.
The closure bottleneck is concentrated in validation blue: only `3` ambiguous
seen-before rows have future claims, all blue, and the current model misses blue.
Therefore this is evidence that temporal working memory is translating as a
mechanism, but not yet a paper Go2 result: blue positives are still missed and
the state has not yet driven command-block execution or return-policy closure.

Progress note later on 2026-06-20: the strict offline Go2 hidden-return gate is
now met. A new validation shard with hidden future-claim opportunities was
created under
`.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen`.
Existing repaired-validation checkpoints failed it, so we generated targeted
Go2 train top-ups rather than reopening 2D optimization. The offset-12 top-up
added green hidden-return coverage, and targeted rendering produced `352` valid
strict hidden-return training rows.

The conservative controller-facing checkpoint
`.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_topup_seed20260621_thr050_lr5e4.pt`
passes the strict scene-disjoint offline target-selection gate with balanced
frame accuracy `0.7625`, positive recall `0.525`, negative abstain specificity
`1.000`, precision `1.000`, false claims `0`, wrong-object selections `0`,
selected-positive hidden future claims `21/21`, and normal-minus-best-corrupted
`0.261`. It covers both strict validation colors: blue `5/8` and green `16/32`.

This materially strengthens the "2D learned memory translates to Go2" bridge,
but the paper boundary remains unchanged: this is still an offline memory-gate
result, not a closed-loop Go2 return controller. The next paper-relevant step is
to show that this conservative learned memory can drive command-block return
behavior with memory-on/off/reset/reversed/shuffled controls. Only after that
should we return to 2D to test whether reachability/quasimetric objectives can
move latent ordering metrics and closed-loop behavior together.

Progress note later still on 2026-06-20: the command-block bridge forced a
claim-boundary correction. The Go2 row aux vector includes the current command
primitive and velocity block. That is a label leak for same-row primitive
prediction, and it makes the full-aux strict target gate too generous for a
runtime controller claim. We added scrubbed command-aux target-gate evaluation,
a frozen-memory primitive head, and a target-relative geometry memory probe.

Results:

- full-aux conservative target gate remains valid as offline bridge evidence:
  `0.7625` balanced frame accuracy, `1.000` precision, `0` false claims;
- scrubbed no-slot target gates are only partial: about `0.65` balanced frame
  accuracy, `0.60` to `0.625` recall, `0.667` to `0.697` abstention, and `10`
  to `11` false claims;
- primitive imitation from frozen memory is rejected as a controller result:
  best oracle-target accuracy `0.475` versus `0.300` majority, but reset/shuffle
  histories are equal or better;
- target-relative geometry memory is a more plausible bridge but still partial:
  best scrubbed no-slot run gets `50.3` deg mean bearing error, `0.37 m` range
  MAE, `0.675` steering-bucket accuracy, with reset `0.300`, reverse `0.425`,
  and shuffle `0.550`.

Paper implication: the current evidence supports "learned working-memory
signals transfer from 2D to Go2 event slices"; it does not yet support "the
learned memory drives a Go2 return controller." The next novel implementation
target is a scrubbed target-geometry memory and command extractor. Latent
topological ordering remains the later paper-version research question after
Go2 translatability is concrete.

Additional follow-up on 2026-06-20: after supervising geometry on
`first_visible_evidence` frames, the best geometry-only command extractor
reaches target-direction accuracy `0.700` and target-primitive proxy accuracy
`0.575`, but this is only modestly above the target-direction majority baseline
of `0.625`. The full two-stage scrubbed hybrid remains below the controller
claim bar: target recall `0.625`, false-claim rate `0.333`, and positive-frame
target-steering pipeline success `0.450` with a `0.125` corruption gap.

This sharpens the paper boundary. The current Go2 result is a credible offline
translation signal, not yet a controller demonstration. The next novel
implementation should target a safer scrubbed hybrid on better-balanced
observe-to-hidden Go2 trajectories. The route-teacher primitive itself should
not be the primary supervision target for this bridge: on these rows,
route-teacher steering agrees with direct target-bearing steering only `0.25`
of the time, because it follows waypoints rather than direct bearing-to-target.

Additional frozen-JEPA follow-up on 2026-06-20: the offline controller proxy is
now strong enough to justify execution work. The selected artifacts are:

- JEPA substrate:
  `.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt`;
- selector:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260630_contrast02_reset05_shuffle05_img64_lat96_h128.pt`;
- geometry:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_frozen_jepa_contrast02_seed20260629_img64_lat96_h128.pt`;
- hybrid report:
  `.generated/go2_hidden_target_memory/go2_direct_gate30_geometry_contrast02_margin02_report.json`.

The hybrid result is target recall `0.775`, false-claim rate `0.212`, zero
wrong-object selections, target-steering pipeline success `0.700`, and a
`0.200` target-steering gap over the best corrupted-history control. This is the
first controller-facing frozen-JEPA Go2 translatability result that should be
carried into command-block execution. It is not yet a paper result; the next
necessary figure is memory-on/off/reset/reversed/shuffled execution on strict
hidden-return episodes, plus matched frozen-random/CNN/DINO substrate baselines.

Replayed command-block follow-up on 2026-06-20: the selected frozen-JEPA
controller was converted into actual Go2 primitive command blocks using the
project command registry and safety adapter. The replay report is
`.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate30_geometry29_margin02_report.json`,
with emitted command records in
`.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate30_geometry29_margin02_commands.jsonl`.
At margin `0.2`, replay reaches target recall `0.775`, false-claim rate
`0.212`, target-selection precision `0.816`, target-steering pipeline success
`0.700`, target-primitive proxy success `0.575`, and `+0.200` over the best
corrupted-history command replay. Memory-off and reset emit only hold commands;
reversed and shuffled histories emit non-hold commands but have far higher false
claim rates (`0.455`) and lower target-steering success (`0.225` / `0.500`).

Strict runtime-aux correction later on 2026-06-20: that command-scrubbed replay
is no longer accepted as the Go2-comparable evidence standard. The Go2 aux
vector still carried `clearance_m` and `traversability_forward_m`, which are
scene-derived debug fields rather than runtime RGB-controller inputs. After
adding `_scrub_runtime_aux` and retraining/evaluating the frozen-JEPA gate and
geometry heads under that stricter boundary, target identity memory still
transfers but controller replay does not.

Strict runtime-aux results:

- direct gate, best checked margin `-0.5`: balanced frame accuracy `0.780`,
  recall `0.650`, abstention `0.909`, precision `0.897`, false claims `3 / 33`,
  and `+0.159` over the best corrupted-history control;
- geometry: `62.9 deg` mean angle error, `0.42 m` range MAE, `0.750`
  steering-bucket accuracy, and `+0.125` corrupted-history steering gap;
- command replay: margin `0.0` reaches recall `0.625`, false-claim rate
  `0.061`, target-steering success `0.400`, and gap `+0.150`; margins `-0.2`
  and `-0.5` raise recall to `0.650` but target-steering remains `0.400`.

This changes the paper boundary. The current result supports "learned target
memory over a frozen Go2 JEPA-style visual substrate transfers under strict
runtime-like aux." It does not support "the learned memory drives a comparable
Go2 return controller." The command registry/safety-adapter replay path is
useful engineering scaffolding, but the next novel implementation must repair
runtime-available target geometry/action and add a camera-conditioned Genesis
collector before live memory-on/off/reset/reversed/shuffled rollouts are
paper-relevant.

GPU strict-replay repair later on 2026-06-20: after unsetting
`HSA_OVERRIDE_GFX_VERSION`, the ROCm PyTorch environment used GPU0
(`AMD Radeon AI PRO R9700`) successfully. A positive-frame-weighted frozen-JEPA
selector repaired the strict target-selection cap:
`.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260633_runtimeaux_pos125_m-15_gpu.pt`
at margin `-1.5` reaches recall `0.825`, false-claim rate `0.212`, precision
`0.825`, and corrupted-history balanced-frame gap `+0.252`.

Pure frozen-JEPA geometry still did not pass the strict replay bar after
steering-head, slot-query, threshold, and broader-data ablations. The best
strict frozen-JEPA geometry replay found was target-steering `0.625` with weak
memory gap, while the stronger-gap frozen-JEPA replay stayed near `0.600`.

The first strict Go2 command-replay implementation milestone is therefore a
mixed-substrate result, but it is not 2D-comparable under the later `0.90+`
target-success bar:

- selector:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260633_runtimeaux_pos125_m-15_gpu.pt`;
- geometry:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_trainablecnn_runtimeaux_seed20260647_img64_h128.pt`;
- replay:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate33pos125_trainablegeom47_runtimeaux_m-15_arc010_report.json`;
- command trace:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate33pos125_trainablegeom47_runtimeaux_m-15_arc010_commands.jsonl`;
- target recall `0.825`;
- false-claim rate `0.212`;
- target-steering pipeline success `0.725`;
- target-primitive pipeline success `0.525`;
- normal-minus-best-corrupted target-steering gap `+0.275`;
- memory-off/reset emit only hold.

This is a concrete implementation milestone and can support the engineering
claim that the 2D learned-memory scaffold has a strict Go2 command-replay path.
It does not yet prove comparable Go2 working-memory performance: target-steering
is `0.725`, below the `0.90+` 2D bar. Follow-up direct-controller runs can reach
`0.900-0.950` positive target-steering with runtime object geometry, but only
with false-claim rates above `0.60` and weak corruption gaps. The paper novelty
target should now split cleanly: first repair calibrated Go2 memory selection,
then research how to move the trainable geometry/action leg into the JEPA latent
substrate.

The older sequence below is retained as the paper-version representation agenda,
not as a prerequisite for the first Go2 memory translation.

1. Paper decided (2026-06-17): system paper under the synthesis + property +
   regime framing (3.5, 6.1). This requires transfer + baselines + the
   JEPA-substrate ablation; the negatives become supporting ablations, not the
   headline.
2. Implement the reachability/quasimetric latent objective in Phase 3A; measure
   whether rho and broad-split SPL move together (the money figure, in 2D first).
3. Fold the egocentric memory into the JEPA as an equivariant recurrent belief;
   re-run the broad-split ablation with/without equivariance and with/without the
   reachability objective.
4. Add counterfactual action-identifiability as the primary objective and
   epistemic uncertainty for the safety story.
5. For the representation-paper version, re-test successful latent-ordering
   variants on Go2 under the strict constraints in 5.1, with the baselines and
   ablations in 5.4, and quantitative eval (5.3).

## 8. References

- Cognitive Mapping and Planning for Visual Navigation (Gupta et al., CVPR 2017):
  https://arxiv.org/pdf/1702.03920
- Learning to Explore using Active Neural SLAM (Chaplot et al., ICLR 2020):
  https://arxiv.org/abs/2004.05155
- Neural Topological SLAM for Visual Navigation (Chaplot et al., CVPR 2020):
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Chaplot_Neural_Topological_SLAM_for_Visual_Navigation_CVPR_2020_paper.pdf
- Object Goal Navigation using Goal-Oriented Semantic Exploration (Chaplot et al.,
  NeurIPS 2020):
  https://proceedings.neurips.cc/paper/2020/file/2c75cf2681788adaca63aa95ae028b22-Paper.pdf
- Differentiable SLAM-net (Karkus et al., 2021): https://arxiv.org/pdf/2105.07593
- EgoMap: structured egocentric memory for Deep RL: https://arxiv.org/pdf/2002.02286
- Plan2Explore: self-supervised exploration via latent disagreement (Sekar et al.,
  2020): https://arxiv.org/abs/2005.05960
- DINO-WM (Zhou et al., 2024): https://arxiv.org/abs/2411.04983
- V-JEPA 2 (Meta, 2025): https://arxiv.org/abs/2506.09985
- What Drives Success in Physical Planning with JEPA World Models? (2025):
  https://arxiv.org/html/2512.24497v2
- Hierarchical Planning with Latent World Models (2026):
  https://arxiv.org/html/2604.03208v1
- HanoiWorld: JEPA world model with RSSM recurrent memory (2026):
  https://arxiv.org/html/2601.01577v1
- Flow Equivariant World Models (2026): https://arxiv.org/pdf/2601.01075
- Beyond Euclidean Proximity: Repairing Latent World Models with Reachability
  Metrics (2026): https://arxiv.org/html/2605.22164v1
- Learning Temporal Distances: Contrastive Successor Features (2024):
  https://arxiv.org/pdf/2406.17098
- Offline GCRL with Quasimetric Representations (2025):
  https://arxiv.org/html/2509.20478v1
- Variational JEPA as Probabilistic World Models (2026):
  https://arxiv.org/pdf/2601.14354
