# 2D Memory Probe Audit and Go2 Translation Plan

Date: 2026-06-19

Status: decision record and execution plan.

## Decision

Do not run another pre-Go2 2D probe campaign for marker memory, egocentric
position structure, transition smoothness, or latent distance/geodesic
correlation. Those questions have already been tested in substance. The current
2D evidence is sufficient to proceed to Genesis-Go2 translation.

The bar was drifting from "demonstrate learned working memory in 2D" toward
"solve latent topology and delete every scaffold." That is too high for the
current milestone. Latent ordering is a paper-version research target, not a
prerequisite for the Go2 translation milestone.

## What 2D Has Already Shown

The 2D result is valid under this bounded claim:

- pixels plus action history can feed a learned recurrent egocentric memory;
- the memory persists hidden marker information after observation;
- the memory supports discover-then-return behavior in a controlled 2D POMDP;
- memory ablations and scaffold-removal attempts identify which parts are
  learned and which parts remain fixed.

The 2D result does not yet prove:

- end-to-end learned navigation;
- metric/topological latent organization;
- replacement of all geometry, value propagation, or action priors;
- Go2 transfer.

## Cheap Probe Audit

| Proposed check | Already tested by | Answer | Decision |
| --- | --- | --- | --- |
| Can a linear/simple probe recover marker memory? | Phase 3A marker-memory, latent map, recurrent memory, and value-field reports in `docs/lewm_jepa_navigation_next_steps_2026-06-14.md` | Yes, stronger than a linear probe: marker readouts reached perfect validation in the recurrent-memory setting and closed-loop 2D runs reached the gate. | Complete; do not block Go2. |
| Does latent memory preserve egocentric position-ish structure? | Latent map head, Phase 2Z occupancy review, recurrent egocentric memory update, pose/yaw probes | Local structure is decodable and usable when accumulated in an external/action-rolled egocentric memory. The pooled latent itself is heading/view dominated. | Complete; use as Go2 design constraint. |
| Do nearby states/actions have smoother latent transitions? | IDM decodability, rollout-horizon, pose-geometry, action-sensitivity, task-aligned rollout-safety probes | Some transition/action signal exists, but it does not reliably become a closed-loop action-ranking mechanism. One-step policy/action-head routes are not the right abstraction. | Complete; do not restart IDM/action-head sweeps. |
| Does latent distance correlate with grid/geodesic distance? | A2 aliasing, final checkpoint benchmark, nav-cost diagnosis, latent metric decodability | No. Projected latent distance has rho around 0.03; yaw-matched variants improve but remain insufficient; bare L2 can be anti-metric. | Complete; this motivates later latent-ordering work. |

## 2D Working-Memory Requirements

For the current milestone, 2D "working memory works" means all of the following
are demonstrated and documented:

1. The runtime state uses learned visual evidence plus action/history, not a
   hidden global cell id or route breadcrumb.
2. A recurrent memory state persists target evidence after the target leaves the
   camera view.
3. Closed-loop discover-then-return success is materially better than
   memoryless, shuffled-history, update-disabled, marker-removed, or corrupted
   egomotion controls.
4. The report tags every remaining scaffold: local map decoding, geometric
   action prior, fixed marker/frontier target construction, value propagation,
   safety vetoes, and thresholds.
5. Failures are categorized as perception, memory storage, reachability/readout,
   exploration, or collision/safety failures.

The current Phase 3A evidence meets the first-pass 2D working-memory standard.
It is not a final paper result by itself because it still relies on explicit
planning and fixed geometric structure.

## Go2 Translatability Requirements

The next milestone is a Genesis-Go2 hidden-target task that preserves the 2D
memory claim while replacing the toy grid embodiment.

Concrete contract/audit entry point:

```bash
python3 scripts/audit_go2_hidden_target_memory_contract.py \
  <raw-rollout-or-derived-labels>/derived_labels/labels.jsonl \
  --out .generated/go2_hidden_target_memory/<run>_audit.json
```

This audit marks a rollout episode as memory-relevant when a landmark is seen,
then hidden for a configurable contiguous window, and later approached again by
BFS distance or metric range. It is a data/evaluation contract, not a learned
controller score.

Smoke on existing Go2 route-teacher data, 2026-06-19:

```bash
python3 scripts/derive_raw_rollout_labels.py \
  .generated/genesis_bulk_rollouts/route_teacher_small_maze_8env_80block_20260517_raw/small_enclosed_maze_f4820a9f5483 \
  --family small_enclosed_maze \
  --topology-seed 12346982286301639415 \
  --split train \
  --out /tmp/lewm_go2_hidden_target_labels_small80

scripts/audit_go2_hidden_target_memory_contract.py \
  /tmp/lewm_go2_hidden_target_labels_small80/labels.jsonl \
  --out /tmp/lewm_go2_hidden_target_labels_small80/hidden_target_audit.json
```

Result: the derived-label pass produced 3,200 labels across 8 env episodes with
0 missing command joins. The hidden-target audit found 7/8 episodes with a
seen-then-hidden landmark and 3/8 episodes with a seen-hidden-return candidate.
That means the existing route-teacher data already contains the shape needed for
the first Go2 memory contract; the next step is to turn this from an offline
data audit into a controller/evaluator baseline.

Offline oracle-label baseline evaluator:

```bash
scripts/evaluate_go2_hidden_target_memory_baselines.py \
  .generated/go2_hidden_target_memory/train_small240_labels/labels.jsonl \
  --out .generated/go2_hidden_target_memory/train_small240_hidden_target_baselines.json
```

The evaluator compares:

- `memory_on`: landmark was seen, then hidden, then later approached;
- `visible_only`: the later close-approach step is directly visible;
- `hidden_claim`: the later close-approach step is still hidden;
- `shuffled_memory`: remembered landmark id is deterministically rotated within
  the episode;
- `no_memory`: no stored target after disappearance.

Canonical durable Go2 memory artifacts generated on 2026-06-19:

| rollout/slice | split | eligible episodes | memory_on | visible_only | hidden_claim | shuffled | no_memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small maze 80-block route-teacher | train | 5 | 5 | 0 | 5 | 4 | 0 |
| small maze 160-block route-teacher | train | 5 | 5 | 2 | 5 | 5 | 0 |
| small maze 240-block route-teacher | train | 6 | 6 | 4 | 6 | 6 | 0 |
| small maze LOS-v2 route-teacher | train | 4 | 4 | 3 | 4 | 3 | 0 |
| medium maze 100-block route-teacher | train | 2 | 0 | 0 | 0 | 0 | 0 |
| medium maze 80-block route-teacher | val | 4 | 2 | 2 | 0 | 1 | 0 |

Interpretation: the Go2 hidden-target contract is real, but the existing
rollouts are not yet sufficient for the final translatability claim. The
small-train slice has strong hidden-claim events, but the shuffled landmark
control also succeeds because the scene/trajectory aliases the landmarks. The
medium validation slice is scene-disjoint and has memory-on > shuffled/no-memory,
but it has no strict hidden-claim successes. This is enough for a supervised
memory-probe bridge, not enough for a closed-loop controller result.

Provenance note: earlier scratch results used manually regenerated
family/topology-seed labels under `/tmp`. The durable artifacts below use labels,
render plans, RGB, and datasets under `.generated/go2_hidden_target_memory` and
are the canonical numbers for this decision record.

Rendered-RGB/data-join event slices:

```bash
python3 scripts/derive_raw_rollout_labels.py \
  .generated/genesis_bulk_rollouts/route_teacher_small_maze_8env_240block_metric_arrival_20260517_raw/small_enclosed_maze_f4820a9f5483 \
  --scene-corpus .generated/scene_corpus/acceptance \
  --scene-id small_enclosed_maze_f4820a9f5483 \
  --out .generated/go2_hidden_target_memory/train_small240_labels

scripts/plan_bulk_render_replay.sh \
  .generated/genesis_bulk_rollouts/route_teacher_small_maze_8env_240block_metric_arrival_20260517_raw/small_enclosed_maze_f4820a9f5483 \
  --out-root .generated/go2_hidden_target_memory/train_small240_render_plan_full \
  --backend cpu \
  --camera-hz 10.0

scripts/select_go2_hidden_target_memory_render_frames.py \
  .generated/go2_hidden_target_memory/train_small240_render_plan_full/000000_small_enclosed_maze_f4820a9f5483/render_replay_plan.json \
  .generated/go2_hidden_target_memory/train_small240_hidden_target_baselines.json \
  --out .generated/go2_hidden_target_memory/train_small240_render_plan_events \
  --context-steps 1

HOME=/tmp/lewm_go2_render_home \
XDG_CACHE_HOME=/tmp/lewm_go2_render_cache \
MPLCONFIGDIR=/tmp/lewm_go2_mplconfig \
PYTHONPATH=$PWD/lewm_genesis:$PWD/lewm_worlds \
  .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/render_replay_genesis.py \
  .generated/go2_hidden_target_memory/train_small240_render_plan_events/render_replay_plan.json \
  --backend cpu \
  --out .generated/go2_hidden_target_memory/train_small240_render_events \
  --store-resolution training \
  --no-depth \
  --overwrite

scripts/build_go2_hidden_target_memory_dataset.py \
  .generated/go2_hidden_target_memory/train_small240_render_events \
  .generated/go2_hidden_target_memory/train_small240_labels/labels.jsonl \
  --out .generated/go2_hidden_target_memory/train_small240_render_events/dataset.jsonl
```

Event dataset results:

| dataset | frames selected | valid rows | invalid skipped | label misses | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| train small80 | 30 | 27 | 3 | 0 | Strong hidden-claim/visible-only separation, but shuffled remains high. |
| train small160 | 42 | 36 | 6 | 0 | Adds longer small-maze event coverage. |
| train small240 | 54 | 48 | 6 | 0 | 22 valid rows have no currently visible landmark. |
| train small LOS-v2 | 42 | 24 | 18 | 0 | Many invalid camera frames; usable as supplemental small-maze coverage only. |
| train medium100 | 12 | 9 | 3 | 0 | Adds medium visuals but no return-success labels. |
| val medium80 | 40 | 35 | 5 | 0 | Scene-disjoint validation slice; 3 valid rows have no currently visible landmark. |

The joined rows carry `visible_landmark_ids`, `hidden_landmark_ids`, and
`go2_hidden_target_memory_selection` event metadata. This proves the durable
data path exists:
raw rollout -> derived labels -> memory baselines -> render plan -> event
selection -> RGB render -> RGB/label dataset.

Supervised Go2 RGB memory-probe bridge:

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_go2_hidden_target_memory_probe.py \
  .generated/go2_hidden_target_memory/train_small240_render_events/dataset.jsonl \
  --validation-datasets .generated/go2_hidden_target_memory/val_medium_4env_80block_render_events/dataset.jsonl \
  --output .generated/go2_hidden_target_memory/go2_hidden_target_memory_probe_train_small240_val_medium80.pt \
  --report-output .generated/go2_hidden_target_memory/go2_hidden_target_memory_probe_train_small240_val_medium80_report.json \
  --epochs 20 \
  --hidden-dim 64 \
  --image-size 64 \
  --device auto
```

Result: the original recall-only probe trained on 48 small240 event rows and
evaluated on 35 medium-validation event rows reached hidden-memory recall 0.586
over 29 hidden-memory targets, while the memoryless current-visibility control
was 0.0. After adding stricter precision/F1 and recurrent-state ablations, this
is not sufficient evidence of temporal memory: the same model had hidden-memory
precision 0.258, F1 0.358, false-positive rate 0.790, and identical hidden
metrics when the recurrent state was reset at every frame.

Expanded small-maze training rows from small80, small160, small240, and LOS-v2
improved target specificity on the scene-disjoint medium-validation slice:
hidden-memory precision 0.875, recall 0.241, F1 0.378, and false-positive rate
0.016. However, the reset-state ablation reached precision 0.714, recall 0.345,
and F1 0.465. That means the current supervised Go2 RGB bridge proves the
render/label/train/eval path and a target-specific non-visible detector, but it
does not yet prove causal recurrent working memory. The next Go2 iteration must
create validation episodes where success requires updating memory from earlier
visibility, not relying on static scene/trajectory cues.

Matched-current-view causal audit and probe:

```bash
python3 scripts/audit_go2_causal_memory_pairs.py \
  .generated/go2_hidden_target_memory/train_small80_labels/labels.jsonl \
  .generated/go2_hidden_target_memory/train_small160_labels/labels.jsonl \
  .generated/go2_hidden_target_memory/train_small240_labels/labels.jsonl \
  .generated/go2_hidden_target_memory/train_small_los_v2_labels/labels.jsonl \
  --max-examples-per-group 20 \
  --out .generated/go2_hidden_target_memory/train_small_all_causal_memory_pairs.json
```

This audit finds same `scene_id`/`cell_id`/`yaw_bin`/hidden-object rows where
the current view is matched but prior visibility differs. The source-aware audit
found 33 ambiguous current-view groups in the combined small-train labels and 9
groups in the medium-validation labels. This gives a sharper causal-memory
probe: for positive rows the selected render plan includes the first-visible
evidence frame plus the matched hidden current frame; for negative rows it
includes the matched hidden current frame before the object has been seen.

Causal render/probe results:

| causal slice | train rows | validation rows | normal causal balanced acc | reset-state acc | reversed-history acc | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v1, small80+small160 -> medium80 | 58 | 20 | 0.536 | 0.250 | 0.473 | First sign that recurrent state can help on matched-view rows, but too close to chance. |
| v2, expanded context/examples | 180 | 61 | 0.498 | 0.545 | 0.398 | More data did not improve transfer; reset-state was better. |
| v3, v2 plus medium100 causal train rows | 257 | 61 | 0.493 | 0.489 | 0.481 | Adding the available medium-train causal rows did not solve the causal transfer. |

Conclusion at that point: the causal matched-view machinery was in place and it
correctly rejected the original fixed-slot RGB GRU as a Go2 working-memory
result.

Query-conditioned causal probe update, 2026-06-19:

The fixed output-slot formulation was too brittle because it asked the model to
learn one classifier per landmark slot and the early training split had poor
object coverage. `scripts/train_go2_causal_memory_query_probe.py` reframes the
test as: given the recurrent visual state and a queried landmark/color, answer
whether that queried object was seen before the matched hidden current view.
This is still a supervised RGB probe, not a closed-loop controller, but it is a
cleaner Go2 memory bridge.

The old causal train split was object-limited:

- small80 and small160 train causal rows covered only red;
- medium100 train rows covered only blue;
- medium validation covered red, blue, green, and yellow.

A new balanced medium-train shard was generated from
`.generated/scene_corpus/minimum_20260520T080420Z`:

```bash
GENESIS_ROCM_PYTHON=.generated/venvs/genesis_render_vulkan/bin/python \
JOBS=auto scripts/run_mass_datagen.sh \
  --scene-corpus .generated/scene_corpus/minimum_20260520T080420Z \
  --out .generated/go2_hidden_target_memory/go2_medium_min4_8env80_20260619_datagen \
  --split train \
  --family medium_enclosed_maze \
  --scene-limit 4 \
  --n-envs 8 \
  --n-blocks 80 \
  --backend cpu \
  --collector-mix route_teacher=1.0 \
  --quality-profile raw_training \
  --no-render
```

The raw causal audit found 50 ambiguous matched-current-view groups: red 23,
blue 16, green 4, yellow 7. Rendered causal training data from these scenes
produced 401 valid rows before targeted top-ups. Green validation was then
repaired with a held-out top-up because the existing medium validation render
contained green negatives but no green positives, despite the raw causal report
having green positives.

Targeted selector support was added via:

```bash
python3 scripts/select_go2_causal_memory_pair_render_frames.py \
  <render_replay_plan.json> \
  <causal_memory_pairs.json> \
  --object-id landmark_02_landmark_green \
  --source-index <n> \
  --max-groups 100 \
  --max-examples-per-bucket 10 \
  --context-steps 1 \
  --out <selected-plan-dir>
```

Best query-conditioned causal-memory bridge so far:

| probe | train rows | validation rows | validation current queries | normal balanced acc | reset-state acc | reversed-history acc | normal minus best ablation | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| color query, medium min4, threshold 0.75, original val | 401 | 61 | 75 | 0.785 | 0.496 | 0.499 | 0.286 | Good aggregate but validation had no green positives. |
| slot query, old+medium+green top-up, threshold 0.65, repaired val | 759 | 74 | 87 | 0.764 | 0.462 | 0.437 | 0.302 | Best repaired-validation bridge; recurrent history is causally useful in aggregate. |
| slot query, old+medium+green+blue top-up, threshold 0.65, repaired val | 1025 | 74 | 87 | 0.684 | 0.478 | 0.471 | 0.206 | Blue top-up over-weighted the train distribution and hurt transfer; do not adopt. |

Per-color validation for the selected repaired-validation bridge:

| color | validation positives | validation negatives | balanced acc | recall | specificity | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| red | 8 | 13 | 0.962 | 1.000 | 0.923 | Strong. |
| yellow | 16 | 19 | 0.921 | 1.000 | 0.842 | Strong. |
| green | 2 | 16 | 0.781 | 1.000 | 0.562 | Measurable after validation repair, but positive count remains small. |
| blue | 7 | 6 | 0.250 | 0.000 | 0.500 | Fails; this is the remaining measured weakness. |

This is enough to say the Go2 learned-memory mechanism is translating in the
aggregate matched-view probe: normal recurrent history beats reset and
reversed-history controls by a material margin on scene-disjoint validation.
It is not enough to claim a paper-grade Go2 result. The paper-grade bar still
requires either a stronger all-colors causal probe or a closed-loop controller
where memory-on beats memory-off/shuffled-history under the strict runtime
boundary.

Offline target-selection gate update, 2026-06-20:

`scripts/evaluate_go2_causal_memory_target_gate.py` converts the
query-conditioned probe into the next controller-facing decision: at each hidden
current-view frame, should the memory controller select a remembered target
object, and if so which object? This is still an offline rendered-event-slice
evaluator, not command execution, but it is stricter than per-query reporting
because it includes abstention, false-claim, and wrong-object behavior.

Best current target-gate artifact:

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/evaluate_go2_causal_memory_target_gate.py \
  .generated/go2_hidden_target_memory/val_medium_4env_80block_causal_pair_render_events_v2/dataset.jsonl \
  .generated/go2_hidden_target_memory/val_medium_4env_80block_causal_pair_render_events_green_topup/dataset.jsonl \
  --checkpoint .generated/go2_hidden_target_memory/go2_causal_memory_query_probe_slot_thr065_train_old_plus_medium_green_topup_val_v2_plus_green.pt \
  --threshold 0.8 \
  --out .generated/go2_hidden_target_memory/go2_causal_memory_target_gate_slot_thr080_val_v2_plus_green_report.json
```

Result on 66 current-view frames:

- normal target-gate balanced frame accuracy: `0.783`;
- positive-frame recall: `0.710`;
- negative-frame abstain specificity: `0.857`;
- target-selection precision: `0.815`;
- false claims: `5`;
- wrong-object selections: `0`;
- reset recurrent-state balanced frame accuracy: `0.473`;
- reversed-history balanced frame accuracy: `0.401`;
- shuffled-hidden-state balanced frame accuracy: `0.336`;
- normal minus best corrupted-history control: `0.310`.

Threshold calibration was cheap and did not change the qualitative picture:

| threshold | frame bal acc | positive recall | negative abstain | precision | normal minus best corrupted | false claims | missed positives | blue positive recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.716 | 0.774 | 0.657 | 0.667 | 0.199 | 12 | 7 | 0.000 |
| 0.55 | 0.716 | 0.774 | 0.657 | 0.667 | 0.247 | 12 | 7 | 0.000 |
| 0.60 | 0.716 | 0.774 | 0.657 | 0.667 | 0.247 | 12 | 7 | 0.000 |
| 0.65 | 0.744 | 0.774 | 0.714 | 0.706 | 0.296 | 10 | 7 | 0.000 |
| 0.70 | 0.757 | 0.742 | 0.771 | 0.742 | 0.280 | 8 | 8 | 0.000 |
| 0.75 | 0.755 | 0.710 | 0.800 | 0.759 | 0.282 | 7 | 9 | 0.000 |
| 0.80 | 0.783 | 0.710 | 0.857 | 0.815 | 0.310 | 5 | 9 | 0.000 |

The green+blue-top-up checkpoint was also evaluated at the same `0.80`
controller threshold and was worse: frame balanced accuracy `0.726`, precision
`0.710`, false claims `9`, and normal-minus-corrupted `0.255`. It still had
blue positive recall `0.000`. Therefore the blue top-up route should not replace
the selected checkpoint. The measured weakness should be carried forward as a
coverage/factor caveat, not used to reopen unbounded tuning.

Label-backed future-claim check:

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/evaluate_go2_causal_memory_target_gate.py \
  .generated/go2_hidden_target_memory/val_medium_4env_80block_causal_pair_render_events_v2/dataset.jsonl \
  .generated/go2_hidden_target_memory/val_medium_4env_80block_causal_pair_render_events_green_topup/dataset.jsonl \
  --checkpoint .generated/go2_hidden_target_memory/go2_causal_memory_query_probe_slot_thr065_train_old_plus_medium_green_topup_val_v2_plus_green.pt \
  --threshold 0.8 \
  --labels .generated/go2_hidden_target_memory/val_medium_4env_80block_labels/labels.jsonl \
  --out .generated/go2_hidden_target_memory/go2_causal_memory_target_gate_slot_thr080_val_v2_plus_green_with_claims_report.json
```

This preserved the target-gate result but found `0/22` selected-positive future
claims on repaired validation, and `0` hidden future claims. A training-scene
sanity check on the medium-min4 plus green-top-up rows found the join does work:
normal recurrent memory selected `214/234` positive frames and `59` of those
had future claims, including `21` hidden claims, versus `4` selected-positive
future claims for reset state. Interpretation: the current validation causal
slice demonstrates learned memory recognition and target selection, but it does
not yet demonstrate return-policy closure. The next Go2 dataset/control step
must include validation episodes where the selected hidden target can actually
be approached/claimed after memory activation.

The causal-pair audit was extended to count future-claim opportunities:

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/audit_go2_causal_memory_pairs.py \
  .generated/go2_hidden_target_memory/val_medium_4env_80block_labels/labels.jsonl \
  --max-examples-per-group 20 \
  --out .generated/go2_hidden_target_memory/val_medium_4env_80block_causal_memory_pairs_with_claims.json
```

The existing validation labels have only `3` ambiguous seen-before rows with any
future claim opportunity, all in the blue group, and `0` ambiguous seen-before
hidden-future-claim rows. That directly explains the current closure failure:
the only validation causal rows that can prove future claim closure are in the
same blue factor where the current model has `0.000` recall.

Calibration diagnostic: lowering the global threshold or lowering only the blue
threshold is not a clean solution. With the base threshold at `0.80` and only
blue lowered to `0.10`, the gate recovers blue recall `0.571` and `4`
selected-positive future claims, but false claims increase to `11`, negative
selected future claims increase to `6`, target-selection precision falls to
`0.703`, and normal-minus-best-corrupted falls to `0.262`. This is useful
diagnostically because it shows blue evidence is present in the scores, but it
is not an acceptable final thresholding rule for the Go2 claim.

GPU note: this smoke ran on CPU because the managed sandbox does not expose the
ROCm device (`torch.cuda.is_available()` was false even though the TinyQuadJEPA
environment is ROCm-enabled). The probe script uses `--device auto` and will use
the GPU in an unsandboxed environment where ROCm is visible.

Runtime inputs:

- RGB observations;
- executed Go2 command blocks from the existing primitive contract;
- onboard-like proprioception/egomotion only;
- no simulator pose, depth, occupancy grid, global route, target id, or
  privileged map at runtime.

Evaluation-only labels and ceilings:

- simulator pose;
- target visibility and distance;
- traversability/occupancy;
- privileged route/frontier planner;
- exact-map ceiling and memory-oracle ceiling.

Task shape:

- randomized rooms/corridors/textures;
- visually discoverable target object or marker;
- exploration until discovery;
- return/claim from memory after the target is no longer directly visible;
- scene-disjoint train/validation/test splits;
- injected odometry/noise/drift before claiming transferability.

Minimum Go2 pass:

1. A memory-on controller beats memoryless and shuffled-history controls.
2. It closes a meaningful fraction of hidden-target episodes under the strict
   runtime boundary.
3. It remains below privileged ceilings but narrows the gap enough to show the
   2D learned-memory mechanism transferred.
4. Failure reports separate perception, egomotion drift, memory update,
   locomotion/primitive execution, exploration, and return-policy failures.

## Paper Novelty Path

The strongest paper path is staged:

1. Demonstrate learned external working memory in 2D.
2. Translate that learned-memory scaffold to Genesis-Go2 under strict runtime
   inputs.
3. Then return to 2D as the fast laboratory for latent-ordering objectives:
   reachability/quasimetric distance, action equivariance, and latent-native
   memory update.
4. Re-test on Go2 only after the 2D latent-ordering objective moves both latent
   structure metrics and closed-loop behavior.

The novelty is not "we tuned a 2D planner to 100%." The novelty target is:

- a JEPA-derived persistent belief state for hidden-goal discover-then-return;
- a clear causal memory ablation;
- Go2 translatability under realistic runtime inputs;
- later, a representation result showing that learned latent structure can
  replace some hand-coded reachability/geometric scaffolds.

## Immediate Plan

1. Freeze the 2D cheap-probe audit as complete.
2. Stop broad 2D controller/router/action-head sweeps unless they test a new
   causal hypothesis.
3. Keep the negative Phase 3B reachability-conditioned planner as evidence that
   decodable reachability is not the same thing as a useful closed-loop latent
   structure.
4. Start the Genesis-Go2 hidden-target/memory task design and data contract.
5. Port the 2D memory scaffold concept to Go2 with explicit scaffold tags:
   learned local perception, recurrent memory update, command-block egomotion,
   target memory, and fixed action extraction.
6. Run Go2 memory-on versus memory-off/shuffled-history baselines before any
   new latent-ordering campaign.
7. After Go2 translatability is demonstrated, return to 2D for the novel latent
   ordering objective and require rho/reachability metrics to move with
   closed-loop SPL/success.

## 2026-06-20 Strict Hidden-Return Gate Update

The earlier repaired-validation target gate proved recognition/selection, but
not hidden-return closure because the selected validation targets had no
post-activation hidden claim opportunities. A stricter validation shard now
exists:

- labels: `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/labels`;
- audit: `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/causal_memory_pairs_with_claims.json`;
- rendered strict datasets:
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/datasets/hidden_claim_source{0,1,2}.jsonl`.

This validation shard contains `299` seen-before hidden future-claim rows and
`265` unseen-before hidden future-claim rows. The strict rendered evaluation
set has `40` positive hidden-return target frames and `33` matched negative
frames. Existing checkpoints failed this stricter test: the earlier
green-top-up checkpoint at threshold `0.50` reached only `0.517` balanced frame
accuracy, `0.125` positive recall, and `-0.008` normal-minus-corrupted
separation; blue-top-up checkpoints selected nothing at their trained
thresholds.

The fix was not another 2D campaign. We generated label-only Go2 training
top-ups, found the missing color coverage, and rendered only strict
hidden-return event slices:

- offset-4 train top-up: blue/red/yellow hidden-return cases, no green;
- offset-12 train top-up: red `135`, blue `66`, green `126`, yellow `381`
  seen-before hidden future-claim rows;
- targeted rendered top-up datasets: `352` valid JSONL rows across offset-4 and
  offset-12.

Three bounded query-probe retrains were run. Two useful operating points now
pass the offline Go2 hidden-return memory gate:

- conservative controller gate:
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_topup_seed20260621_thr050_lr5e4.pt`
  with report
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_causal_memory_target_gate_hidden_claim_topup_seed20260621_thr050_lr5e4_report.json`;
  balanced frame accuracy `0.7625`, positive recall `0.525`, negative abstain
  specificity `1.000`, target precision `1.000`, false claims `0`,
  wrong-object selections `0`, selected-positive hidden future claims `21/21`,
  and normal-minus-best-corrupted `0.261`;
- high-recall diagnostic gate:
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_topup_seed20260620_thr050.pt`
  with report
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_causal_memory_target_gate_hidden_claim_topup_seed20260620_thr050_report.json`;
  balanced frame accuracy `0.781`, positive recall `0.775`, negative abstain
  specificity `0.788`, target precision `0.816`, false claims `7`,
  wrong-object selections `0`, selected-positive hidden future claims `31/31`,
  and normal-minus-best-corrupted `0.242`.

The conservative gate is the better handoff to command-block control because it
does not select unseen impostors on the strict shard. It still covers both
strict validation colors: blue `5/8`, green `16/32`. This is enough to move to
closed-loop Go2 command selection. It is not yet a paper-grade Go2 navigation
result because the memory state has not executed a return policy.

## 2026-06-20 Controller-Boundary Correction

While wiring command-block selection we found an important boundary issue:
`_aux_features` includes the current row's `command.primitive_name` and velocity
block. That is acceptable only if the row is interpreted as already executed
action history. It is not acceptable for predicting the same row's next
primitive. For controller-facing command experiments, current command aux must
therefore be scrubbed.

Implemented follow-up tooling:

- `scripts/train_go2_memory_command_policy.py` trains a frozen-memory primitive
  head and reports oracle-target plus learned-gate pipeline controls;
- `scripts/train_go2_memory_target_geometry.py` trains a scrubbed recurrent
  target-relative bearing/range probe;
- `scripts/train_go2_causal_memory_query_probe.py` and
  `scripts/evaluate_go2_causal_memory_target_gate.py` now support
  `--scrub-command-aux`.

Runtime provenance: host ROCm sees two idle AMD GPUs, but
`/home/andrewknowles/TinyQuadJEPA/bin/python` reports
`torch.cuda.is_available() == False` and `torch.cuda.device_count() == 0`, so
these follow-up probes ran on CPU.

Strict scrubbed target gate:

- best no-slot query checkpoints:
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_scrubbed_seed20260627_current_pos05_noslot.pt`
  and
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_scrubbed_seed20260628_current_pos025_noslot.pt`;
- controller reports:
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_causal_memory_target_gate_hidden_claim_scrubbed_seed20260627_current_pos05_noslot_thr050_report.json`
  and
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_causal_memory_target_gate_hidden_claim_scrubbed_seed20260628_current_pos025_noslot_thr050_report.json`;
- best operating points are partial, not a pass: balanced frame accuracy
  `0.646` to `0.648`, positive recall `0.600` to `0.625`, negative abstain
  specificity `0.667` to `0.697`, precision `0.694` to `0.706`, and `10` to
  `11` false claims on `33` negatives;
- query-level reset/reverse controls are strong (`~0.25` balanced-accuracy
  drop), but shuffle remains too close at the frame level (`0.515` to `0.560`),
  so the scrubbed gate is not yet a reliable runtime memory selector.

Primitive imitation bridge:

- best report:
  `.generated/go2_hidden_target_memory/go2_memory_command_policy_hidden_return_scrubbed_seed20260627_noslot_report.json`;
- oracle-target primitive accuracy improves over majority (`0.475` vs `0.300`),
  but corrupted histories do as well or better (`reset 0.600`, shuffle `0.525`);
- learned-gate pipeline success is only `0.25` of positive frames.

This rejects the immediate "identity gate -> command primitive" bridge. The
failure is coherent: object identity memory is not enough to choose a Go2 return
primitive without a target-relative pose or waypoint memory.

Target-relative geometry bridge:

- best no-slot geometry reports:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_hidden_return_scrubbed_seed20260620_noslot_report.json`
  and
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_hidden_return_scrubbed_seed20260622_noslot_report.json`;
- the strict rendered slices include real evidence events, not only current
  hidden frames: `52` validation `first_visible_evidence` events and `270`
  train `first_visible_evidence` events survived rendering/join;
- the stronger memory-dependent run gets mean bearing error `50.3` degrees,
  range MAE `0.37 m`, steering-bucket accuracy `0.675`, with reset `0.300`,
  reverse `0.425`, and shuffle `0.550`;
- the lower-angle run gets `46.5` degrees and `0.30 m`, but has weaker
  corruption separation.

Interpretation: Go2 translatability is real at the offline memory-gate level,
but controller-readiness now requires a learned target-geometry memory state.
That is still in the Go2 translation lane, not the later paper goal of proving
topological latent ordering.

## 2026-06-20 Geometry Command Bridge Follow-Up

The target-geometry trainer now supervises `first_visible_evidence` geometry,
not only later hidden-current rows. This makes the strict event slices closer to
the actual observe-then-recall mechanism.

New tooling:

- `scripts/evaluate_go2_memory_geometry_command_extractor.py` evaluates
  predicted bearing/range as command-direction choices under reset, reverse,
  and shuffled-history controls;
- `scripts/evaluate_go2_hybrid_memory_geometry_controller.py` evaluates the
  two-stage scrubbed pipeline: learned target selector then learned
  target-geometry command extraction.

Best geometry-only command extractor:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_hidden_return_scrubbed_seed20260630_evidence_img96_h128.pt`;
- report:
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_geometry_command_extractor_seed20260630_evidence_img96_h128_nohold_selectall_report.json`;
- oracle-target command-direction accuracy `0.700`, target-primitive proxy
  accuracy `0.575`, with best corrupted target-direction accuracy `0.575`
  (`+0.125`);
- target-direction majority baseline is `0.625`, so this is only modestly
  above the validation distribution;
- route-teacher primitive/steering agreement is not the right pass criterion on
  these rows: route steering matches target steering only `0.25`, because the
  route teacher follows waypoints rather than direct bearing-to-target.

Best scrubbed hybrid pipeline:

- selector:
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_scrubbed_seed20260627_current_pos05_noslot.pt`;
- geometry:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_hidden_return_scrubbed_seed20260630_evidence_img96_h128.pt`;
- report:
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_hybrid_selector27_geometry30_nohold_report.json`;
- target recall `0.625`, target-selection precision `0.694`, false-claim rate
  `0.333`, positive-frame target-steering pipeline success `0.450`, with a
  `0.125` gap over the best corrupted-memory control.

Selector threshold sweeps did not fix the hybrid: threshold `0.8` reduced
false-claim rate only to `0.273` and kept target-steering success at `0.425`.
A full-aux selector diagnostic removes false claims (`0.000`) but still reaches
only `0.425` target-steering pipeline success, so the remaining blocker is not
just thresholding.

Conclusion at this pre-JEPA controller-bridge stage: the Go2 bridge demonstrates partial, memory-dependent
target-direction information, but the controller requirement is not met. The
next requirement is a stronger scrubbed selector plus target-geometry memory on
better-balanced observe-to-hidden Go2 slices, especially right-turn and
hard-negative cases. Closed-loop command execution should wait until the
offline hybrid reaches a materially safer operating point.

JEPA-substrate correction, 2026-06-20: the Go2 bridge above was a direct
supervised CNN+GRU baseline, not JEPA transfer. A minimal frozen Go2 JEPA-style
substrate now exists; see
`docs/lewm_go2_jepa_substrate_memory_update_2026-06-20.md`. The frozen-JEPA
causal query probe reaches `0.626` matched-current-view balanced accuracy with a
`+0.202` gap over reset/reversed-history controls, satisfying the first
Go2 JEPA-substrate memory requirement. The controller-facing target gate
(`+0.070` corrupted-history gap) and geometry readout (`+0.025` steering gap)
remain preliminary for this first frozen-JEPA checkpoint; the follow-up below
supersedes that controller-facing status.

Frozen-JEPA controller-proxy update later on 2026-06-20: the preliminary
controller-facing result above was improved by adding an optional in-batch
contrastive next-latent loss to the Go2 JEPA trainer, training a direct
frame-level target gate, fixing checkpoint selection to include shuffled-history
controls, and retraining target-relative geometry on the same contrastive frozen
encoder. The selected direct gate at margin `0.2` now reaches `0.781` balanced
frame accuracy, `0.775` positive recall, `0.788` negative abstention, `0.816`
target precision, `7 / 33` false claims, `0` wrong-object selections, and
`+0.209` normal-minus-best-corrupted balanced frame accuracy. The two-stage
selector + geometry proxy reaches target recall `0.775`, false-claim rate
`0.212`, target-steering pipeline success `0.700`, target-primitive proxy
success `0.575`, and `+0.200` normal-minus-best-corrupted target-steering
success.

This meets the offline Go2 frozen-JEPA controller-proxy bar. It still does not
meet the closed-loop Go2 controller bar: no command blocks have been executed
from this selector/geometry pair, and no memory-on/off/reversed/shuffled
closed-loop return evaluation has been run.

Replayed command-block update later on 2026-06-20:
`scripts/evaluate_go2_frozen_jepa_command_replay.py` now expands the selected
frozen-JEPA selector + geometry output into Go2 primitive command blocks through
`config/go2_primitive_registry.yaml`, applies platform safety limits from
`config/go2_platform_manifest.yaml`, and writes a command JSONL trace. This is
execution-facing replay, not live physics feedback.

Selected replay:

- report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate30_geometry29_margin02_report.json`;
- command trace:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate30_geometry29_margin02_commands.jsonl`;
- margin `0.2`;
- target recall `0.775`;
- false-claim rate `0.212`;
- target-selection precision `0.816`;
- target-steering pipeline success `0.700`;
- target-primitive proxy success `0.575`;
- normal-minus-best-corrupted target-steering success `+0.200`;
- safety-clipped command blocks `10 / 73`, from yaw/arc rate limiting;
- replay gate pass: true.

Memory controls in replay:

| replay mode | target recall | false-claim rate | target-steering success | non-hold commands |
| --- | ---: | ---: | ---: | ---: |
| normal | `0.775` | `0.212` | `0.700` | `38` |
| memory-off abstain | `0.000` | `0.000` | `0.000` | `0` |
| reset recurrent state | `0.000` | `0.000` | `0.000` | `0` |
| reversed history | `0.575` | `0.455` | `0.225` | `38` |
| shuffled hidden states | `0.600` | `0.455` | `0.500` | `39` |

Margin `0.0` fails on false claims (`0.303`); margin `0.4` fails on recall
and target steering (`0.475` recall, `0.450` steering success). This confirms
margin `0.2` as the command-replay handoff.

Strict runtime-aux recheck later on 2026-06-20:
the command-replay handoff exposed one more boundary issue. The aux vector used
by the Go2 probes still included `clearance_m` and `traversability_forward_m`.
Those fields are not available to a learned RGB controller at command-selection
time. The new `_scrub_runtime_aux` path zeros them in addition to the current
command aux fields, and the frozen-JEPA gate/geometry trainers and evaluators
now carry an explicit `--scrub-runtime-aux` contract.

Under that stricter boundary, target selection survives but the controller
handoff does not:

- strict runtime-aux direct gate, best checked margin `-0.5`: balanced frame
  accuracy `0.780`, positive recall `0.650`, negative abstention `0.909`,
  target precision `0.897`, false claims `3 / 33`, and `+0.159`
  normal-minus-best-corrupted balanced frame accuracy;
- strict runtime-aux geometry: `62.9 deg` mean angle error, `0.42 m` range MAE,
  `0.750` steering-bucket accuracy, and `+0.125` corrupted-history steering
  gap;
- strict runtime-aux command replay fails at all checked margins:
  margin `0.0` reaches target recall `0.625`, false-claim rate `0.061`,
  target-steering success `0.400`, and gap `+0.150`; margins `-0.2` and `-0.5`
  raise recall only to `0.650` while target-steering remains `0.400`.

Decision: do not count the earlier command-scrubbed replay as
2D-comparable Go2 evidence. Count it as an engineering artifact showing that
the command registry/safety adapter path works. The current runtime-plausible
evidence says the memory selector transfers, but the target-direction/action
readout is not viable enough to justify live Go2 claims without a runtime RGB
bridge and a better geometry/action objective.

GPU/runtime repair later on 2026-06-20:
the active ROCm PyTorch environment sees GPU0 only when
`HSA_OVERRIDE_GFX_VERSION` is unset. The working training prefix is
`env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES=0 ...`; with that prefix,
GPU0 (`AMD Radeon AI PRO R9700`) was used for the selector and geometry
ablations.

The strict selector shortfall was repaired with a positive-frame-weighted
frozen-JEPA direct gate:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260633_runtimeaux_pos125_m-15_gpu.pt`;
- selected margin `-1.5`;
- recall `0.825`, false-claim rate `0.212`, precision `0.825`, corrupted
  balanced-frame gap `+0.252`.

Frozen-JEPA geometry remained below the comparable replay bar despite steering
heads, object-slot queries, threshold sweeps, and broader-data training. Best
strict frozen-JEPA geometry replay found in this pass reached target-steering
success `0.625`, but with only `+0.025` memory gap; the best stronger-gap
frozen-JEPA replay stayed around `0.600`.

A trainable-CNN geometry control did pass the strict replay gate when paired
with the repaired frozen-JEPA selector:

- geometry checkpoint:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_trainablecnn_runtimeaux_seed20260647_img64_h128.pt`;
- replay report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate33pos125_trainablegeom47_runtimeaux_m-15_arc010_report.json`;
- command trace:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate33pos125_trainablegeom47_runtimeaux_m-15_arc010_commands.jsonl`;
- strict replay gate pass: true;
- target recall `0.825`;
- false-claim rate `0.212`;
- target-selection precision `0.825`;
- target-steering pipeline success `0.725`;
- target-primitive pipeline success `0.525`;
- normal-minus-best-corrupted target-steering gap `+0.275`.

Decision correction: the strict mixed-substrate replay is not 2D-comparable
under the actual 2D learned-memory bar. It remains a useful engineering bridge:
frozen-JEPA learned memory for target selection plus trainable-CNN geometry for
target-relative action reaches target-steering success `0.725`, but the 2D
target is approximately `0.90+`. The pure frozen-JEPA paper claim is still not
met for geometry/action.

Follow-up direct-controller GPU runs tested a stricter Go2 bridge:

- direct memory-to-steering controller;
- strict runtime-aux scrubbing;
- runtime-query geometry features without geodesic/BFS map fields;
- exclusive prefix memory before current-frame ingestion;
- object-level candidate BCE auxiliary loss;
- object-slot identity.

Best positive-target results reached the 2D target-success range but failed the
full working-memory gate:

- `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_exclusive_short_seed20260662_h128_report.json`:
  target-steering `0.950`, false-claim rate `0.606`, corruption gap `+0.150`;
- `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_exclusive_short_pos2_seed20260663_h128_report.json`:
  target-steering `0.900`, false-claim rate `0.636`, corruption gap `+0.175`.

Best cleaner memory-dependency results stayed below the 2D target-success bar:

- `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_seed20260653_h128_report.json`:
  target-steering `0.825`, false-claim rate `0.121`, corruption gap `+0.350`;
- `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_exclusive_seed20260660_h256_report.json`:
  target-steering `0.850`, false-claim rate `0.091`, corruption gap `+0.275`.

Current decision: Go2 has not yet met the 2D-comparable working-memory bar. The
remaining problem is not steering geometry alone; it is calibrated memory
selection. High target recall is achievable, but it currently admits too many
false memory claims.

## Requirements Tracker

| Requirement | Current status | Next action |
| --- | --- | --- |
| 2D learned memory works under bounded claim | Met for the first-pass milestone | Freeze and cite the Phase 3A evidence. |
| Cheap diagnostic probes complete | Met | Do not rerun before Go2. |
| Go2 hidden-target task contract exists | Met as an offline label/audit contract on train and validation route-teacher rollouts | Generate longer/more validation rollouts with strict hidden-claim successes. |
| Go2 offline oracle-label memory baselines run | Met on durable train/val artifacts; train small240 has hidden-claim events but shuffled also succeeds, val medium80 has memory_on 2/4 vs shuffled 1/4 but no hidden-claim successes | Treat current baselines as bridge diagnostics, not final evidence. |
| Go2 rendered RGB/label data path exists | Met on durable event slices: train small80 27 valid rows, train small160 36, train small240 48, train small LOS-v2 24, train medium100 9, val medium80 35, all with 0 label misses | Scale render slices after better route-teacher rollouts. |
| Go2 learned recurrent RGB memory probe runs | Met as an engineering bridge but not as a memory result: expanded small training -> medium val gives precision 0.875 / recall 0.241 / F1 0.378, but reset-state F1 is 0.465 | Keep the bridge code; replace the weak model/data with a causal matched-view objective. |
| Go2 learned-memory target-selection gate runs | Met under the stricter runtime-aux boundary: strict frozen-JEPA gate reaches `0.780` balanced frame accuracy, `0.650` recall, `0.909` abstention, `0.897` precision, `3 / 33` false claims, and `+0.159` corrupted-history gap | Keep as evidence that learned target memory transfers; do not optimize thresholds further until geometry/action is repaired. |
| Go2 learned-memory future-claim closure | Partially met offline: selected positives can be recovered under strict runtime aux, but recall drops from the command-scrubbed `0.775` to `0.650` | Improve data coverage and geometry/action readout before claiming executed return behavior. |
| Go2 target-relative geometry memory | Mixed: pure frozen-JEPA strict geometry remains below the replay bar; trainable-CNN strict geometry reaches `46.3` deg mean bearing error, `0.50 m` range MAE, and `0.700` steering-bucket accuracy | Keep frozen-JEPA geometry as the open representation problem; use trainable geometry only as the current implementation bridge. |
| Go2 geometry-to-command extraction | Met under strict runtime aux only with trainable-CNN geometry: selected replay reaches target-steering success `0.725` and target-primitive success `0.525` | Next repair target is frozen-JEPA geometry/action, not selector thresholding. |
| Go2 hybrid selector + geometry controller proxy | Useful but not 2D-comparable: frozen-JEPA selector plus trainable-CNN geometry reaches recall `0.825`, false-claim rate `0.212`, target-steering success `0.725`, and `+0.275` corruption gap | Carry this only as a runtime-bridge artifact; do not claim 2D-level performance. |
| Go2 replayed command-block memory controller | Not met for 2D-comparable command replay. Mixed-substrate replay exists, but target-steering is `0.725` vs the `0.90+` bar | Keep the generated command JSONL as a handoff artifact; next work is calibrated memory selection plus runtime bridge. |
| Go2 primitive command imitation from memory | Rejected: best oracle-target command head reaches `0.475` accuracy vs `0.300` majority, but reset/shuffle histories do as well or better | Do not wire this head as a controller. Use target-relative geometry instead. |
| Go2 closed-loop learned-memory controller runs | Not met. Replay exists, but the current Genesis collector interface does not provide RGB to a learned controller at command-selection time | Add a camera-conditioned collector/runtime bridge before live claims. |
| Go2 causal memory ablations pass | Met for strict target selection and mixed-substrate strict replay; pure frozen-JEPA geometry remains open | State the boundary explicitly: memory identity transfers through frozen JEPA, but action geometry currently needs a trainable encoder. |
| Go2 frozen-JEPA memory substrate | Met for memory selection: non-collapsed Go2 JEPA-style encoder, frozen-JEPA causal query gap `0.202`, repaired strict direct target-gate gap `0.252`; not met for target geometry | Treat as translatability evidence for memory, with frozen geometry as the next representation target. |
| Latent topology/order objective | Future paper work | Revisit after Go2 translatability. |
