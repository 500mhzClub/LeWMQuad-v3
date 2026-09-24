# Snapshot determinism, pilot status, and predictor-side scoring feasibility

Date: 2026-08-10
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.**

Frozen and untouched: confirmatory `443e591` / `60b0bb2d…`, audit baseline
`cdecdae`, spatial commit `2ff1761`, probe package `b8f05e57…`, spatial result
`04454ffe…`, counterfactual design `e3176d93…`.

## Recorded spatial conclusion

> The qualified post-confirmatory H=1 occupancy assay found no detectable rollout
> effect or proprioception-by-rollout interaction. It neither establishes spatial
> improvement nor spatial degradation. No H=2–4 spatial claim is available from
> the existing temporal corpus.

---

# 1. Snapshot schema and digest contract — `b18c49eae0405fc2697ffac36ccb2cfb3395f05b31a17227a0cf443b67f80589`

## The decisive empirical result

Genesis 0.3.14 exposes `Scene.save_checkpoint(path)` / `Scene.load_checkpoint(path)`,
which pickle the **full physics state**. Tested directly: a CPU rigid scene settled
30 steps, checkpointed, then continued 25 steps three ways.

| comparison | max abs difference |
|---|---:|
| replay 1 vs replay 2 | **0.0** |
| original continuation vs replay | **0.0** |
| bit-identical | **yes** |

**The solver half of deterministic replay is available and exact.** That is the
single biggest risk in the counterfactual plan, and it is retired. The branch
corpus does not need a bespoke solver-state serialiser.

## Schema — three layers, only one of which Genesis owns

| layer | owner | covers | verified |
|---|---|---|---|
| solver state | Genesis `save_checkpoint` | root pose/orientation, root linear+angular velocity, all joint positions/velocities, contact and collision solver state, environment-object state, internal caches | **bit-identical** |
| harness state | this repository | **previous applied command (the slew-limiter carry)**, slew-limiter internals, episode id / reset count / step counter, beacon and task state, termination and safety flags, command-block sequence id and tick index, controller mode | not built |
| controller state | locomotion policy | policy buffers and any observation history fed to the policy | **unconfirmed** |
| RNG state | split | Genesis physics RNG, numpy, torch, environment samplers | not built |

**Completeness is established empirically, never assumed.** Setting position and
velocity tensors is explicitly not accepted as sufficient; any field that diverges
under repeated restore identifies missing snapshot state.

**Digest contract.** Canonical snapshot digest = sha256 over the serialised solver
checkpoint bytes concatenated with canonicalised harness-state JSON. Separate
compatibility digest over {Genesis version, backend, scene id and geometry hash,
controller id and weights hash, sim options, dt, substeps, configuration digest}.
Restoration **refuses** on compatibility-digest mismatch, snapshot-digest failure,
or schema-version mismatch.

**A risk worth naming now:** if `genesis_contract_ppo` is not feedforward — if it
carries a recurrent state or an observation-history buffer — that buffer is
snapshot state, and omitting it would silently change a replayed trajectory while
passing a solver-only check. This must be confirmed before qualification, and it
is exactly the failure mode a zero-action replay cannot detect.

# 2. Deterministic replay qualification — **NOT RUN**

Qualification requires the harness layer above (previous applied command,
slew-limiter internals, task/termination state, RNG capture) integrated with scene
loading, the controller, and the rendering path. **That integration does not
exist**, so the four required sequence types (sign reversal, sustained turn,
contact-adjacent, slew-limited) × 3 repetitions could not be executed, and neither
could branch-order invariance.

Per design v1.1 I stopped rather than approximate branches by resetting a subset of
state — which is precisely the shortcut the design forbids.

# 3. 20-state oracle pilot — **NOT RUN**

Gated on qualification, which did not run. No predictor checkpoint was opened at
any point in this work.

---

# 4. Predictor-side candidate-scoring feasibility audit (read-only)

No scoring head was trained.

## Why the confirmatory metric cannot be the score

The 32 predictors emit predicted token grids, not utilities. The confirmatory
metric is cosine to the **true** future latent — a realised-outcome quantity. It is
disqualified as a planning score by definition, not by convenience.

## Existing heads, and why each is or is not compatible

| head | verdict | reason |
|---|---|---|
| `lewm/models/energy_head.py` (GoalEnergyHead) | **incompatible** | empirically rejected in earlier work (plan_cost beat it); trained under a different goal-image contract, not the factorial action/control/target-latent contract |
| `lewm/planning/costs.py::plan_cost` | **incompatible** | belongs to the topological-nav stack; operates on a different representation and a goal-image convention; also shown flat between same-corridor views (distance concentration) |
| `lewm/models/pose_head.py`, `idm_head.py` | **incompatible as a utility** | predict pose/action, not oracle components; an IDM score would rank by action recoverability, which is not utility |
| depth/occupancy/traversability heads in `lewm/planning/` | **partially usable as inputs** | model-independent geometry, but none maps to progress/safety/completion under the frozen weights |
| the H=1 occupancy probe frozen here (`b8f05e57…`) | **usable as a component** | already qualified, fitted on train-split true latents only, disjoint from selection rows; but it decodes occupancy, not utility |

**None is reusable as a candidate score merely because it exists.**

## Label availability for a shared decoder

The corpus carries per-step labels at 10 Hz, 48,000 rows per scene, that map onto
the frozen oracle components:

| oracle component | label source | coverage |
|---|---|---|
| progress | `bfs_distance_to_landmark` per landmark (Δ over the branch) | **48,000/48,000 non-null** per scene |
| safety | `clearance_m`, `stuck_label`, contact/fall events | present |
| completion | landmark `bfs_distance_cells` reaching terminal, `landmarks[].range_m/bearing_body_rad` | present |

| property | status |
|---|---|
| rows / clusters available | 3,922 train rows, 80 scenes, ~2,101 episode clusters corpus-wide |
| family coverage | all 8 |
| horizon coverage | H=1 labelled everywhere; **H=2 only 81/475 selection, H=3–4 none** |
| labels match design-v1.1 oracle definitions | **partially** — BFS distance is per-landmark and needs a goal binding to become "progress" |
| inputs available at planning time | yes if inputs are (predicted latent, goal specification); labels may be privileged, inputs may not |
| overlap with the 475 selection rows / pilot / evaluation states | avoidable — fit on train rows only, exactly as the occupancy probe did |
| goal/task context present | yes, via `landmarks[]` bearing/range and the landmark id |
| privileged fields required | **as labels only**; BFS distance must never be an input |

## Three strategies compared

| strategy | feasible | assessment |
|---|---|---|
| **A.** one shared multi-head utility decoder on true target latents | **yes** | trainable on train-split true latents with the three oracle labels; applied to *predicted* latents at scoring time, which is leakage-free. Needs a goal binding to define progress |
| **B.** separate progress / safety / completion decoders combined by the frozen weights | **yes, preferred** | same data, but each head is separately qualifiable and separately reportable, so a failure of one component does not silently corrupt the composite. Matches the frozen `1.0/−2.0/0.5` weighting exactly |
| **C.** non-learned score from already-frozen probes only | **no** | the only qualified frozen probe is occupancy at H=1. Occupancy alone cannot express progress or completion, and the H=2–4 label gap means it cannot be extended |

**Recommendation: strategy B**, conditional on the leakage controls below. It is
recommended *only* because it can be trained and calibrated on data disjoint from
the final counterfactual evaluation and applied identically to all 32 predictors.

**The future model-side score must never use** the realised branch outcome, a true
future latent, future proprioception, simulator ground-truth pose or velocity
unavailable in deployment, or model-specific calibration on the final evaluation
branches.

## Blocker, stated precisely

**No leakage-free candidate-scoring contract exists today.** Strategy B is
feasible but unbuilt, and it requires a goal binding that the current temporal
corpus does not define per row. Until it is built and frozen, **rank regret cannot
be attributed to the predictors**. The oracle branch corpus remains independently
useful for characterising the candidate-utility landscape.

---

# 5. Scorer-fit corpus — design only, not generated

Required because the existing temporal corpus lacks (a) a per-row goal binding and
(b) any H≥2 label, both of which strategy B needs.

| item | specification |
|---|---|
| states | **120**, 15 per family across all eight |
| branches per state | 6 (a reduced bank; fitting needs coverage, not ranking resolution) |
| horizon | 4 blocks, matching evaluation |
| scenes | a **third disjoint pool**: not the 80 factorial scenes, not the 20 pilot development scenes, not the future 200 evaluation scenes |
| fit / calibration split | 96 states fit / 24 states calibration, split by scene, never by row |
| goal binding | one landmark designated per state at snapshot time and recorded; progress is Δ BFS to *that* landmark |
| labels | the three oracle components, recorded identically to design v1.1 |
| leakage controls | scene-disjoint from all three other corpora; calibration by scene; no predictor output used in fitting; frozen before any evaluation branch is scored |

## Updated compute and storage

| item | estimate |
|---|---:|
| snapshot/replay qualification (4 sequences × 3 reps + order invariance) | ~1 h |
| 20-state oracle pilot (240 branches) | ~2.5 h, ~1.5 GB |
| scorer-fit corpus (120 states × 6 branches) | ~9 h, ~5 GB |
| scorer training + calibration | ~2 h |
| 200-state evaluation corpus (2,400 branches) | ~20 h, ~15 GB |
| scoring 32 checkpoints over the evaluation corpus | ~6 h |
| **total to a defensible rank-regret result** | **≈ 40 h, ~22 GB** |

That is up from the ~26 h in v1, because the scorer-fit corpus and its training
did not exist in that estimate.

---

# 6. Blockers

1. **Harness snapshot layer unbuilt** — Genesis solver replay is exact, but the
   previous applied command, slew-limiter internals, task/termination state and RNG
   capture are not integrated. Qualification and the pilot could not run.
2. **Controller statefulness unconfirmed** — `genesis_contract_ppo` must be shown
   feedforward, or its buffers become snapshot state. A zero-action replay cannot
   detect this.
3. **No leakage-free predictor-side score** — strategy B is feasible but unbuilt,
   and needs a per-row goal binding the temporal corpus does not carry.
4. **H=2–4 labels absent**, which is why the pilot's evaluation-only spatial labels
   matter and why no multi-horizon spatial claim is available today.

## Stopping condition

Nothing is running. No 200-state corpus, no utility decoder trained, no predictor
checkpoint scored, no predictor retrained, no frozen result altered.
