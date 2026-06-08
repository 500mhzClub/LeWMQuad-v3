# Post-IDM Review Decision and Next Experiment Gate (2026-06-06)

## Decision

Do not launch an IDM fine-tuning ladder or resume the pose-aux ladder as the
main navigation direction.

Keep the implemented IDM head as experimental tooling, but require a cheap
frozen-latent closure diagnostic before considering it again. Even a positive
IDM decodability result is not a promotion result: IDM predicts which logged
command occurred, while the deployed problem is choosing which counterfactual
command makes progress toward a goal.

The main navigation path returns to the registered v3 decomposition. The
executed P1 projected-feature and P3 pooled raw-feature screens below both
failed their gates, so the immediate next work is recognition/topology plus
spatial/history substrates rather than more pooled-feature head tuning:

1. LeWM remains the local action-conditioned forward model.
2. A goal-aligned first-action benchmark measures the missing local-control
   capability directly and is reused across representation substrates.
3. Recognition/retrieval plus explicit topological memory handles routing and
   partial observability.
4. A spatial/history substrate screen is the fallback if frozen LeWM features
   cannot support the local action ranker.

## Evidence

### The forward model already uses actions

The scaled seq11 checkpoint has `zero-free@h10=+0.201` and MPC recorded-action
win-rate versus zero of about `0.66`. The predictor also already injects action
conditioning through AdaLN at every transformer block. The remaining failure is
not simply "the model ignores actions."

### Decodability has repeatedly failed to transfer to action selection

The 300-session pose-aux screen improved encoded metric decodability
monotonically, but:

- predicted-to-goal correlation only tied the continuation control;
- first-action Spearman stayed approximately zero or negative in every cell;
- metric-cost navigation stayed at or below random.

The partial 1000-session rerun strengthens the distinction. Only F0 completed,
with predicted-to-goal correlation `+0.164` but first-action correlation
`-0.050`. C0/C1/C2 did not produce checkpoints, and the ladder process exited.

### IDM is not a goal-conditioned or counterfactual objective

The implemented head predicts the requested active command block from an
observed consecutive latent pair. It has no goal input, does not compare
candidate actions from the same state, and does not supervise realized progress,
collision outcome, or reachability. It can therefore improve through behavior
policy, heading, or optic-flow cues without improving planning.

### Literature correction

- PLDM includes IDM, but its reported ablation does not establish IDM as the
  load-bearing mechanism. Removing IDM leaves Two-Rooms at `98.0%` and lowers
  Diverse Maze to `75.5%`; removing variance/covariance regularization is much
  more damaging.
- DINO-WM explicitly plans without a pre-learned inverse model and attributes
  precise control performance to frozen spatial patch features.
- The recent JEPA-WM design study focuses on pretrained encoders, multistep
  rollout, context/proprioception, predictor design, and CEM. It does not present
  IDM as the general solution.
- These successful benchmarks are mostly fully observed or use richer state
  information. They do not remove the single-camera POMDP limitation here.

## Closure Diagnostic

Run `scripts/probe_idm_decodability.py` on the current checkpoint before any IDM
fine-tune. It trains deterministic held-out-scene ridge readouts in both raw and
projected latent spaces. Ridge strength is selected on a training-only validation
split; the held-out scenes are used only for the reported result.

- `state`: predict the command from `z_t` alone;
- `true_pair`: predict from the real `(z_t, z_t+1)` pair;
- `shuffled_next`: predict from `(z_t, shuffled(z_t+1))`;
- `delta`: predict from `z_t+1 - z_t`.

A true-pair gain over both `state` and `shuffled_next` demonstrates
transition-specific action information. It does not demonstrate actionable
goal geometry.

Default interpretation:

- pair gain `< 0.05` over either control: close IDM; do not train it;
- pair gain `>= 0.05` over both controls: IDM remains scientifically plausible,
  but may only proceed in a small cell whose primary gate is first-action
  regret/correlation, never IDM R2.

### Executed closure result

The closure probe was run on the current scaled seq11 source checkpoint
`lewm_seq11_e3.pt` using a bounded 1,000-session corpus, a 10% held-out-scene
split, 20,480 training transitions, and 10,240 evaluation transitions. The
strict JSON result is stored at
`models/idm_closure_20260606/e3_idm_decodability.json`.

| latent | state R2 | true-pair R2 | shuffled-next R2 | pair gain vs state | pair gain vs shuffled | delta R2 |
|---|---:|---:|---:|---:|---:|---:|
| raw | -21.652 | -25.393 | -25.684 | **-3.741** | +0.290 | -0.300 |
| projected | -17.211 | -48.186 | -15.890 | **-30.976** | -32.297 | -0.295 |

The true pair fails the required gain over the state control in both spaces.
Although the training-only selection scores show some within-corpus
decodability, it does not generalize to held-out scenes. **Close the current IDM
branch: do not calibrate `--idm-lambda` and do not launch IDM fine-tuning.**

## Next Experiments

### P0: Freeze the current result

- Preserve commit `870a146` as implemented-but-not-promoted tooling.
- Do not calibrate `--idm-lambda`.
- Do not restart the incomplete 1000-session pose ladder.
- Report the pose/IDM branch as a negative transfer result: improving
  decodability did not improve action selection.

### P1: Goal-aligned local-action dataset and ranker

From identical start states, evaluate every primitive against a local subgoal
and record:

- realized first-block progress toward the subgoal;
- collision/invalid outcome;
- final distance after the block;
- current observation/history, goal observation, and primitive ID.

Train a frozen-feature goal-conditioned action ranker or value head. Its primary
offline gate is first-action regret no more than half the random-pick regret,
with positive rank correlation. Promote to closed-loop only after that gate.

#### Implemented P1 v0

- `scripts/build_first_action_dataset.py` renders a fixed start/visible-beacon
  goal pair, randomizes start heading, and evaluates every primitive under the
  benchmark kinematic/collision model from that identical state.
- Bounded all-family screens use round-robin family ordering rather than taking
  the first lexical family.
- `lewm/models/action_ranker.py` and `scripts/train_first_action_ranker.py` train
  a frozen projected-feature ranker. The report includes a train-set action-only
  prior control.
- Promotion requires all of: regret ratio `<= 0.5`, positive first-action
  Spearman, selected collision rate `<= 0.05`, improvement over the action-only
  prior, at least 32 held-out scenes, and at least 256 held-out groups.

#### Executed P1 v0 result

Artifacts are under `.generated/first_action_p1_v0/`. The balanced corpus has
264 training groups from 33 scenes and 328 held-out groups from 41 scenes,
covering all eight families. Each group scores seven primitives. Setup failures
are explicit in the dataset summary JSON rather than silently omitted.

| seed | first regret (m) | regret / random | first Spearman | selected collision | gate |
|---:|---:|---:|---:|---:|---|
| 20260606 | 0.0489 | 0.707 | +0.163 | 13.4% | fail |
| 20260607 | 0.0498 | 0.722 | +0.161 | 12.2% | fail |
| 20260608 | 0.0507 | 0.734 | +0.161 | 12.5% | fail |

Mean regret ratio is `0.721`, mean first-action Spearman is `+0.161`, and mean
selected collision rate is `12.7%`. The action-only prior has regret ratio
`0.992`, so the projected features contain some goal-conditioned local-action
signal, but the ranker remains far short of the registered regret and collision
gates.

**P1 v0 decision:** close the pooled projected single-frame ranker cell. Do not
tune the head or run closed-loop physics. Reuse this dataset/gate for P3
spatial/history substrate comparisons.

### P2: Minimal recognition/topological path

Implement the smallest useful v3 path before the full BeliefEncoder:

1. frozen-LeWM retrieval embedding/head;
2. GoalAdapter into the same retrieval space;
3. explicit transition graph over visited places;
4. subgoal selection over the graph;
5. local action ranker for subgoal pursuit.

Add history only if it improves held-out same-place retrieval or loop-closure
precision by a registered margin.

#### Registered P2.0 retrieval-head gate

The detailed preregistration is in
`docs/lewm_phase_b_minimal_retrieval_plan_2026-06-06.md`.

P2.0 trains a small metric head on frozen pooled **raw** LeWM features. It uses
same-cell positives and non-adjacent same-scene negatives, then evaluates with
the existing A3 same-cell retrieval benchmark on 32 held-out `test_id` scenes
and three fixed seeds. Promotion requires mean Recall@5 to improve over the
frozen raw baseline by at least 15 percentage points, every seed to improve,
and mean Recall@1 not to regress.

The explicit graph is the next implementation only if this offline retrieval
gate passes. Reachability and graph distance will come from observed edges and
BFS, not from a learned two-frame reachability regressor.

#### Executed P2.0 result

P2.0 failed on 32 train scenes, 32 held-out scenes, and all three registered
seeds. Frozen raw Recall@1/Recall@5 was `0.3420/0.5491`; the learned head mean
was `0.3213/0.5379`, a regression of `-0.0206/-0.0112`. Mean graph-distance
Spearman also did not improve.

**P2.0 decision:** close pooled-feature retrieval-head tuning and do not build
the online graph on this substrate. Execute the registered P3.1 spatial
patch-token retrieval screen next.

#### Executed P3.1 result

P3.1 also failed on 32 held-out scenes. Current-LeWM patch mean Recall@5 was
`0.5327` and the spatial pyramid was `0.5284`, both below raw CLS at `0.5491`.
Recall@1 also regressed slightly.

**P3.1 decision:** do not build on the current from-scratch ViT patch tokens.
The next registered cell is P3.2, the identical held-out retrieval screen using
a strong frozen pretrained DINOv2 substrate. If that fails, move to short
history rather than continuing single-frame feature engineering.

#### Executed P3.2 result

P3.2 failed. DINOv2 CLS and patch mean improved Recall@1 slightly to
`0.3525/0.3536` from raw LeWM's `0.3420`, but both reduced Recall@5 to
`0.5432/0.5409` from `0.5491`. The DINOv2 spatial pyramid was worse.

**P3.2 decision:** close single-frame retrieval substrate work. Execute the
registered P3.3 short-history retrieval screen; if it fails, stop the current
retrieval/topology branch rather than building online memory.

#### Executed P3.3 result and final branch decision

P3.3 failed on 32 held-out scenes. H8 mean was the best history descriptor,
improving Recall@1 by `+0.0197` and Recall@5 by `+0.0198`; the registered
Recall@5 margin was `+0.05`.

**Final retrieval/topology decision:** stop. Do not build BeliefEncoder,
GoalAdapter, loop closure, or online memory on the current data/objective.
Learned pooled heads, current-LeWM patch tokens, DINOv2 features, and short
history all fail to produce the required held-out retrieval improvement.

The active next program is task-aligned navigation learning. The 2026-06-08
data review confirmed that new collection is not required: mine branch-choice
and collision-recovery decision points from the existing 69.6M aligned
rendered/label rows, train with privileged simulator labels but pixel/action
inputs, and gate only on first-action regret/collision followed by closed-loop
goal success. See `docs/lewm_task_aligned_data_readiness_2026-06-08.md`.

That handoff has now executed on balanced 32-scene train and validation
subsets. Each produced 16,384 branch/recovery decisions with joined RGB,
requested/executed actions, outcomes, target frames, and privileged labels.
Offline all-primitive scoring found a 30.4% logged-action optimal rate on
validation, establishing direct policy-learning headroom without new
collection. The next cell is a pixels/history/actions candidate scorer with
separate collision, progress, heading, and clearance heads. IDM remains
closed.

That frozen-base candidate-scorer program has now executed. Pooled raw,
2x2-spatial, and four-frame-history descriptors all failed the minimum gate in
all three seeds. Stop frozen-head and inference-search variants. The next
controlled escalation is task-aligned adaptation of only the final two
vision-encoder blocks; do not resume IDM or collect new rollouts.

### P3: Substrate screen if P1 fails

P1 v0 failed, so compare:

- pooled LeWM raw/projected features;
- frozen DINOv2/DINOv3 patch features;
- two-frame and short-history variants.

Use the same first-action dataset and gates. Do not compare substrates only on
prediction or decodability metrics.

#### Executed pooled-feature baseline

The dataset builder now stores identical pooled `raw` and `proj` LeWM features,
and `train_first_action_ranker.py --latent-space {raw,proj}` applies the same
head, outcomes, splits, seeds, and gates to either space. Reports are under
`.generated/first_action_p1_v0/ranker_{raw,proj}_seed*.json`.

| substrate | mean regret / random | mean first Spearman | mean selected collision | three-seed gate |
|---|---:|---:|---:|---|
| action-only prior | 0.992 | +0.087 | 0.0% | control |
| pooled projected LeWM | 0.721 | +0.161 | 12.7% | fail / fail / fail |
| pooled raw LeWM | **0.596** | **+0.282** | **9.7%** | fail / fail / fail |

Raw encoder features preserve substantially more local-action signal than the
projected planning space, but still miss both the regret threshold (`<= 0.5`)
and collision threshold (`<= 5%`) in every seed. **Stop pooled single-frame
rankers. The next P3 cells must add spatial patch structure and/or history.**

## Stop Conditions

- Stop IDM permanently if the closure probe lacks transition-specific gain.
- Stop any auxiliary objective whose first-action metric does not beat its
  continuation control after one bounded proxy cell.
- Stop raw latent-L2 metric navigation on the current pooled latent; that path is
  already exhausted.
- Stop pooled single-frame local-action rankers after the executed raw/projected
  screen; projector/head tuning is not the next lever.
- Stop the current retrieval/topology branch after P2.0/P3.1/P3.2/P3.3 all fail
  their registered held-out retrieval gates.
- Do not build BeliefEncoder, GoalAdapter, loop closure, or online memory until
  a new task-aligned substrate demonstrates direct navigation improvement.
- Do not run physics navigation until the kinematic first-action and closed-loop
  gates pass on at least 32 scenes and 3 seeds.
