# Phase 3B Memory Gate and Reachability Plan

Date: 2026-06-19

Status: execution plan. This document turns the Phase 3A audit and paper-plan
review into concrete next work. It intentionally stops treating 100% closed-loop
success on one 2D split as the main target.

## Decision

The Phase 3A evidence is sufficient to continue past controller perfection
sweeps. The learned JEPA latent map and recurrent egocentric memory are close to
the exact broad-split ceiling, while the remaining gap is mostly strict learned
readout/action selection.

Update, 2026-06-19: the 2D cheap-probe audit is complete enough for the current
milestone. Phase 3B reachability/quasimetric work remains important, but it is
no longer a prerequisite for starting Genesis-Go2 translation. The next
blocking milestone is Go2 translatability of the learned-memory scaffold; latent
ordering becomes the paper-version research loop after that bridge is working.

Second update, 2026-06-19: the Go2 matched-current-view causal probe now has an
aggregate positive bridge result, but not a final Go2 or paper result. The
selected query-conditioned slot probe trained on old causal rows plus the new
medium min4 shard and green top-up reached `0.764` balanced accuracy on repaired
scene-disjoint validation, versus `0.462` with recurrent state reset and `0.437`
with reversed history. That is enough to continue into Go2 memory-controller
wiring. It is not enough to claim solved translatability because blue validation
recall is still `0.000` and there is no closed-loop memory-on/off/shuffled Go2
controller yet.

Third update, 2026-06-20: the Go2 bridge now has a controller-facing offline
target-selection gate. `scripts/evaluate_go2_causal_memory_target_gate.py`
evaluates whether the recurrent memory state would select the right hidden
target object or abstain. The selected checkpoint at controller threshold `0.80`
reached `0.783` balanced frame accuracy, `0.815` target-selection precision,
`0.857` negative-frame abstention, and `0.310` normal-minus-best-corrupted
separation against reset, reversed-history, and shuffled-hidden-state controls.
This is enough to move from probe evaluation to command-block selection. It is
still not a closed-loop Go2 result, and it still misses blue positives.

Fourth update, 2026-06-20: optional full-label future-claim scoring was added to
the same evaluator. It is a useful guardrail against overclaiming: on repaired
validation, normal recurrent memory selects `22` positive target frames but
`0` of those selected targets have a future claim/approach in the existing
route-teacher labels. On medium-min4 training scenes the same metric finds
`59` selected-positive future claims, including `21` hidden claims, so the join
works. The validation problem is therefore data/control closure, not a broken
metric. The next Go2 gate must create or execute return behavior after memory
activation.

The causal-pair audit now also counts future-claim opportunities. The existing
medium validation labels have only `3` ambiguous seen-before rows with any
future claim opportunity, all in blue, and `0` ambiguous seen-before hidden
future-claim rows. Since the current model has `0.000` blue recall, the current
validation split cannot show learned future-claim closure without either fixing
blue transfer or adding return-capable validation rows for other factors.

For representation work, the next target is therefore not another router,
threshold, or one-step action head. The next target is a
reachability-structured memory: the latent belief should encode which cells can
be reached through remembered free space, how far targets/frontiers are, and
which action consequences change that reachability.

## Current Claim Boundary

Allowed claim:

- In a controlled randomized-palette 2D POMDP, pixels plus action history can
  support a persistent egocentric memory that approaches the exact planner
  ceiling, and memory ablations show the state is causally useful.
- In Genesis-Go2 RGB event slices, a query-conditioned recurrent memory probe
  can use history on matched-current-view hidden-landmark rows better than reset
  or reversed-history controls in aggregate.
- The same Go2 recurrent memory state can drive an offline target-selection gate
  better than reset, reversed-history, and shuffled-hidden-state controls in
  aggregate.

Not yet allowed:

- JEPA has learned general navigation.
- Go2 navigation is solved.
- The learned state has replaced planning geometry.
- The result is end-to-end free of hand-written readout structure.
- The Go2 result is paper-grade across all landmark colors or closed-loop
  command selection.
- The target-selection gate is a robot controller; it is not yet wired to
  command-block execution.

## 2D Working-Memory Gate

The 2D benchmark is considered "working" when the selected strict controller and
memory representation satisfy all of the following:

1. Multi-seed broad randomized-palette evaluation is within a small margin of
   the exact available ceiling, not necessarily 100%.
2. Memory-on beats no-memory, shuffled-history, marker-removed,
   memory-update-disabled, and egomotion-corrupted controls by a material margin.
3. The result holds on scene-disjoint layouts and at least one harder axis:
   longer horizons, larger maps, noisier egomotion, or unseen render palettes.
4. Any scaffold is explicitly tagged in reports: geometric prior, turn breaker,
   fixed marker return, side-wall fallback, router, or hand-written frontier
   rule.
5. Failures are categorized as perception, memory storage, reachability/readout,
   exploration, or collision/safety failures.

Recommended reporting metrics:

- claim/success rate;
- collision rate;
- steps or SPL-like path efficiency;
- marker-seen and latent-marker-seen rates;
- memory localization/top-k when the marker was observed;
- reachability correlation or distance error;
- ceiling gap against exact odometry and exact learned-latent frontier planners.

## Phase 3B Implementation Target

Implement reachability/quasimetric latent structure before further controller
sweeps.

This remains a paper-version representation target, not the next blocker. The
immediate Go2 implementation target is lower level:

- feed the query-conditioned Go2 memory state into a command-selection scaffold;
- use the offline target-selection gate as the memory-conditioned target source;
- evaluate memory-on, memory-reset, shuffled-history, and reversed-history
  command-block runs on hidden-target episodes;
- keep all scaffolds explicitly tagged: fixed primitive set, route-teacher data,
  query construction, memory probe, action extraction, safety vetoes;
- diagnose the blue causal-probe failure without reopening the 2D optimization
  campaign.

Initial implementation:

- add a reusable reachability target builder for egocentric memories;
- expose dense maps for reachable cells, current-cell distance, target/frontier
  distance, discounted target value, target mask, and frontier mask;
- expose pairwise shortest-path targets for sampled cells as the future
  quasimetric objective surface;
- add a trainable reachability head and trainer while keeping current Phase 3A
  controller behavior unchanged until the target is tested and audited.

Runnable entry point:

```bash
scripts/phase3b_rocm_train_reachability.sh \
  --train-data .generated/jepa_phase3a/<train>.jsonl \
  --validation-data .generated/jepa_phase3a/<validation>.jsonl \
  --base-checkpoint models/checkpoints/phase3a_explore_claim/<base>.pt \
  --latent-map-head models/checkpoints/phase3a_explore_claim/<latent-map>.pt \
  --latent-memory-updater models/checkpoints/phase3a_explore_claim/<memory>.pt \
  --output models/checkpoints/phase3a_explore_claim/<phase3b-reachability>.pt \
  --memory-size 31 \
  --target-mode marker_or_frontier \
  --save-best
```

Initial smoke, 2026-06-19:

```bash
scripts/phase3b_rocm_train_reachability.sh \
  --train-data .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/train_phase3a_positive_control.jsonl \
  --validation-data .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/validation_phase3a_positive_control.jsonl \
  --base-checkpoint models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt \
  --latent-map-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_broad_multiseed_0109_1723_31_train_2048.pt \
  --latent-memory-updater models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_broad_map_broad_seed20260701_2048.pt \
  --output /tmp/phase3b_reachability_smoke.pt \
  --memory-size 31 \
  --max-train-episodes 1 \
  --max-validation-episodes 1 \
  --max-steps 2 \
  --optimization-steps 1 \
  --batch-size 2 \
  --hidden-dim 8 \
  --device cpu \
  --log-every 1
```

Result: completed end-to-end, built 2 train and 2 validation examples, loaded the
base model, latent-map head, and latent-memory updater, trained one step, and
wrote a `jepa_phase3b_reachability_training_report_v0` checkpoint. The smoke is
only an interface check; the metrics are not meaningful.

Conditioned value-map planner entry point:

```bash
scripts/phase3b_rocm_train_reachability_value_map_planner.sh \
  --train-data .generated/jepa_phase3a/<train>.jsonl \
  --validation-data .generated/jepa_phase3a/<validation>.jsonl \
  --base-checkpoint models/checkpoints/phase3a_explore_claim/<base>.pt \
  --latent-map-head models/checkpoints/phase3a_explore_claim/<latent-map>.pt \
  --latent-memory-updater models/checkpoints/phase3a_explore_claim/<memory>.pt \
  --latent-value-field-head models/checkpoints/phase3a_explore_claim/<value-field>.pt \
  --latent-value-extractor-head models/checkpoints/phase3a_explore_claim/<extractor>.pt \
  --reachability-head models/checkpoints/phase3a_explore_claim/<reachability>.pt \
  --output models/checkpoints/phase3a_explore_claim/<phase3b-value-map>.pt \
  --memory-size 31 \
  --save-best
```

Initial conditioned-planner smoke, 2026-06-19:

```bash
scripts/phase3b_rocm_train_reachability_value_map_planner.sh \
  --train-data .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/train_phase3a_positive_control.jsonl \
  --validation-data .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/validation_phase3a_positive_control.jsonl \
  --base-checkpoint models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt \
  --latent-map-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_broad_multiseed_0109_1723_31_train_2048.pt \
  --latent-memory-updater models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_broad_map_broad_seed20260701_2048.pt \
  --latent-value-field-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_broad_map_memory_1ch_2048.pt \
  --latent-value-extractor-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_extractor_broad_map_memory_1024.pt \
  --reachability-head /tmp/phase3b_reachability_smoke.pt \
  --output /tmp/phase3b_reachability_value_map_smoke.pt \
  --memory-size 31 \
  --max-train-episodes 1 \
  --max-validation-episodes 1 \
  --max-steps 2 \
  --optimization-steps 1 \
  --batch-size 2 \
  --hidden-dim 8 \
  --device cpu \
  --log-every 1
```

Result: completed end-to-end, loaded the Phase 3B reachability checkpoint,
constructed reachability-conditioned value-map examples, trained one step, and
wrote a `jepa_phase3b_reachability_value_map_planner_training_report_v0`
checkpoint. This is also only an interface check.

Closed-loop reachability-conditioned eval entry point:

```bash
scripts/export_jepa_phase3a_closed_loop_demo_mp4.py \
  --validation-data .generated/jepa_phase3a/<validation>.jsonl \
  --checkpoint models/checkpoints/phase3a_explore_claim/<base>.pt \
  --output .generated/jepa_phase3a/<eval>.mp4 \
  --report-output .generated/jepa_phase3a/<eval-report>.json \
  --score-source latent_recurrent_reachability_value_map_planner \
  --latent-map-head models/checkpoints/phase3a_explore_claim/<latent-map>.pt \
  --latent-memory-updater models/checkpoints/phase3a_explore_claim/<memory>.pt \
  --latent-value-field-head models/checkpoints/phase3a_explore_claim/<value-field>.pt \
  --latent-value-extractor-head models/checkpoints/phase3a_explore_claim/<extractor>.pt \
  --latent-reachability-head models/checkpoints/phase3a_explore_claim/<reachability>.pt \
  --latent-reachability-value-map-planner-head models/checkpoints/phase3a_explore_claim/<phase3b-value-map>.pt \
  --exact-online-memory-size 31
```

Initial closed-loop eval smoke, 2026-06-19:

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python scripts/export_jepa_phase3a_closed_loop_demo_mp4.py \
  --validation-data .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/validation_phase3a_positive_control.jsonl \
  --checkpoint models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt \
  --output /tmp/phase3b_reachability_value_map_eval_smoke.mp4 \
  --report-output /tmp/phase3b_reachability_value_map_eval_smoke_report.json \
  --score-source latent_recurrent_reachability_value_map_planner \
  --latent-map-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_broad_multiseed_0109_1723_31_train_2048.pt \
  --latent-memory-updater models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_broad_map_broad_seed20260701_2048.pt \
  --latent-value-field-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_broad_map_memory_1ch_2048.pt \
  --latent-value-extractor-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_extractor_broad_map_memory_1024.pt \
  --latent-reachability-head /tmp/phase3b_reachability_smoke.pt \
  --latent-reachability-value-map-planner-head /tmp/phase3b_reachability_value_map_smoke.pt \
  --exact-online-memory-size 31 \
  --max-episodes 1 \
  --max-steps 2 \
  --skip-video \
  --device cpu
```

Result: completed end-to-end and wrote
`/tmp/phase3b_reachability_value_map_eval_smoke_report.json` with
`score_source=latent_recurrent_reachability_value_map_planner` and two action
selections through that path. The smoke used one-step toy checkpoints, so
`claimed 0/1 episodes` is expected and not a behavioral result.

Execution note: the managed Codex sandbox hides `/dev/kfd` and `/dev/dri`, so
ROCm PyTorch falls back to CPU inside the sandbox even though the TinyQuadJEPA
environment is HIP-enabled. Heavy training/eval commands must run unsandboxed
with explicit approval. Unsandboxed, the same interpreter sees two HIP devices
and reports `AMD Radeon AI PRO R9700`. Closed-loop exporter evals are
GPU-underutilized because the rollout loop is Python/control heavy, so running
two or three eval ablations concurrently is reasonable after checking VRAM and
CPU. Do not assume the same for training jobs without a resource check.

Real broad-split Phase 3B result, 2026-06-19:

- Reachability head:
  `models/checkpoints/phase3a_explore_claim/phase3b_reachability_broad_seed20260701_marker_or_frontier_2048.pt`
  trained for 2048 steps and selected step 1792 by `target_top1_match`.
  Validation reached `target_top1_match=0.9516`, `reachable_precision=0.9903`,
  `reachable_recall=0.9887`, and `action_match=0.6968` on the 16-episode
  validation cap. This shows the auxiliary reachability target is learnable.
- Reachability-conditioned value-map planner, total-action selected:
  `phase3b_reachability_value_map_planner_broad_seed20260701_explicit_frontier_broadonly_2048.pt`
  selected step 256 with `action_match=0.8839`,
  `broad_action_match=0.8961`, and `sparse_action_match=0.8540`.
- Reachability-conditioned value-map planner, broad-action selected:
  `phase3b_reachability_value_map_planner_broad_seed20260701_explicit_frontier_broadmetric_2048.pt`
  selected step 1792 with `broad_action_match=0.9240`,
  `action_match=0.8394`, and `sparse_action_match=0.6309`.
- Strict 64-episode closed-loop eval:
  both reachability-conditioned planner variants claimed only `2/64` episodes.
  The broad-metric report was
  `.generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/phase3b_reachability_value_map_broadmetric_strict_max68_report.json`.
  The total-metric report was
  `.generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/phase3b_reachability_value_map_totalmetric_strict_max68_report.json`.
- Strong unconditioned Phase 3A baseline:
  `phase3a_v5_latent_value_map_planner_expandedmap_frontiertrace_selected_allstates_multiseed35_1536.pt`
  claimed `58/64` episodes under the same strict max-68 eval, with report
  `.generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/phase3a_unconditioned_expandedmap_frontiertrace_allstates_multiseed35_strict_max68_report.json`.

Interpretation: the first Phase 3B implementation is a negative closed-loop
result. It proves the reachability target can be predicted from recurrent
memory, but the current way of feeding those features into the value-map planner
destroys closed-loop behavior. This should not be tuned with broad threshold
sweeps. The next hypothesis must explain the mismatch between offline
reachability/action metrics and rollout failure.

Next trainer integration:

1. Diagnose the reachability-conditioned rollout collapse by comparing
   trajectory-level selected actions, marker observation, latent marker memory,
   target fields, and reachability features against the unconditioned baseline.
2. Add a no-reachability/capacity-matched ablation, or a zero/shuffled
   reachability-feature ablation, before training more variants.
3. Add a contrastive or regression loss so latent distances preserve
   shortest-path/reachability ordering.
4. Re-run broad multi-seed closed-loop evaluation with the same strict flags as
   the current best controller.
5. Compare against a no-reachability-objective ablation with identical memory
   and controller capacity.

Pass condition:

- reachability metrics improve and closed-loop success/SPL improves or remains
  near ceiling;
- removing the reachability objective degrades both latent reachability metrics
  and closed-loop behavior;
- the effect survives broad multi-seed randomized-palette evaluation.

## Go2 Translation Gate

The Go2 bridge should start in Genesis, not on physical hardware.

Strict Genesis-Go2 Phase 3B conditions:

- runtime inputs are RGB plus executed command blocks and onboard-like
  proprioception/egomotion;
- simulator pose, depth, occupancy, and global route are labels/evaluation only;
- command blocks use the existing Go2 primitive contract;
- drift/noise is injected before claiming transferability;
- hidden-goal discover-then-return is retained so memory is necessary.

The first Go2 target is a 2.5D analog of Phase 3A:

- randomized rooms/corridors/textures;
- a visually discoverable target object;
- exploration until discovery, then return/claim;
- broad scene-disjoint train/validation splits;
- exact privileged planners used only as ceilings.

Update, 2026-06-19: the Go2 bridge is now partially instantiated. Durable
hidden-target label, render-selection, RGB dataset, and supervised memory-probe
artifacts exist under `.generated/go2_hidden_target_memory`. The initial
recall-only result was too permissive: train small240 -> val medium80 reached
hidden-memory recall `0.586` versus memoryless `0.0`, but precision was only
`0.258`, false-positive rate was `0.790`, and resetting the recurrent state at
every frame produced identical hidden metrics.

After expanding small-maze event training rows and selecting checkpoints by
hidden-memory F1, the best scene-disjoint supervised bridge has precision
`0.875`, recall `0.241`, F1 `0.378`, and false-positive rate `0.016` on medium
validation. That is a better target-specific non-visible detector, but it is
still not a temporal-memory result because the reset-state ablation reaches F1
`0.465`. Remaining blockers are therefore concrete: route-teacher
train/validation slices must force memory update causally, the probe must beat
reset/reversed-history controls, and only then should the memory state be ported
into a closed-loop command-selection controller.

Follow-up on 2026-06-19: a source-aware matched-current-view causal audit now
exists (`scripts/audit_go2_causal_memory_pairs.py`). It found 33 ambiguous
small-train groups and 9 medium-validation groups where the same current
scene/cell/yaw/object view can have different prior visibility histories. The
first causal probe on rendered matched-view rows reached balanced accuracy
`0.536` versus reset-state `0.250` and reversed-history `0.473`; an expanded
v2 slice reached `0.498` while reset-state reached `0.545`; adding 77 valid
medium-train causal rows reached `0.493` while reset-state reached `0.489`.
This is a useful negative: the causal evaluator is now strong enough to reject
the current tiny RGB GRU bridge, and the next Go2 requirement is to pass this
matched-view memory test before wiring a controller.

Follow-up on 2026-06-20: the query-conditioned causal bridge passes an offline
target-selection gate. The report
`.generated/go2_hidden_target_memory/go2_causal_memory_target_gate_slot_thr080_val_v2_plus_green_report.json`
uses the selected green-top-up checkpoint and repaired validation set. It scores
66 current-view frames: normal balanced frame accuracy `0.783`, positive recall
`0.710`, negative abstain specificity `0.857`, target-selection precision
`0.815`, false claims `5`, wrong-object selections `0`, reset-state balanced
frame accuracy `0.473`, reversed-history `0.401`, shuffled-hidden-state `0.336`,
and normal-minus-best-corrupted `0.310`. A blue-top-up checkpoint at the same
controller threshold was worse (`0.726` frame balanced accuracy and `0.255`
normal-minus-corrupted) and still missed blue positives. The next requirement is
therefore not more probe tuning; it is command-block execution with the same
memory-on/off/corrupted controls.

The label-backed future-claim report
`.generated/go2_hidden_target_memory/go2_causal_memory_target_gate_slot_thr080_val_v2_plus_green_with_claims_report.json`
adds a sharper boundary: validation target selection is positive, but future
claim closure is `0/22` selected-positive targets. The corresponding
medium-min4 train sanity report finds `59/214` selected-positive future claims
and `21` hidden claims, so the metric can detect closure when the rollout
contains it. This means the current repaired validation split is good for
matched-view memory recognition, not for return-policy proof.

The extended causal audit
`.generated/go2_hidden_target_memory/val_medium_4env_80block_causal_memory_pairs_with_claims.json`
shows why: only `3` ambiguous seen-before validation rows have any future claim
opportunity, all are blue, and none are hidden claims.

Follow-up later on 2026-06-20: a strict hidden-return validation shard and
targeted train top-ups close the offline memory-gate gap. The strict validation
audit
`.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/causal_memory_pairs_with_claims.json`
contains `299` seen-before hidden future-claim rows and `265` unseen-before
hidden future-claim rows. Existing green/blue top-up checkpoints failed this
strict shard, confirming this was not a calibration-only issue.

The offset-12 train top-up added the missing color coverage: red `135`, blue
`66`, green `126`, and yellow `381` seen-before hidden future-claim rows.
Targeted rendering produced `352` valid strict hidden-return training rows.
After three bounded query-probe retrains, the conservative controller-facing
gate
`.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_topup_seed20260621_thr050_lr5e4.pt`
passes the strict offline target-selection test:

- report:
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_causal_memory_target_gate_hidden_claim_topup_seed20260621_thr050_lr5e4_report.json`;
- balanced frame accuracy `0.7625`;
- positive recall `0.525`, including blue `5/8` and green `16/32`;
- negative abstain specificity `1.000`;
- target-selection precision `1.000`;
- false claims `0`, wrong-object selections `0`;
- selected-positive hidden future claims `21/21`;
- normal-minus-best-corrupted balanced accuracy `0.261`.

The high-recall sibling checkpoint reaches `0.781` balanced accuracy and
`0.775` positive recall, but makes `7` false claims. Use the conservative
checkpoint for command-block control. This satisfies the offline Go2
hidden-return memory-gate requirement; it still does not prove closed-loop Go2
return navigation.

Follow-up later on 2026-06-20: command-block wiring exposed a stricter runtime
boundary. The rendered row aux vector includes the current command primitive and
velocity block. For command prediction this is a label leak, and for a
controller-facing memory selector it is at least an overly generous action
history assumption. We therefore added scrubbed command-aux training/evaluation
paths.

Strict scrubbed no-slot target gates are only partial:

- checkpoint
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_scrubbed_seed20260627_current_pos05_noslot.pt`
  scores balanced frame accuracy `0.646`, positive recall `0.625`, negative
  abstain specificity `0.667`, precision `0.694`, false claims `11`, reset
  `0.376`, reverse `0.392`, shuffle `0.515`;
- checkpoint
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_hidden_return_scrubbed_seed20260628_current_pos025_noslot.pt`
  scores balanced frame accuracy `0.648`, positive recall `0.600`, negative
  abstain specificity `0.697`, precision `0.706`, false claims `10`, reset
  `0.389`, reverse `0.382`, shuffle `0.560`.

The frozen-memory command head confirms that object identity is not enough for
Go2 primitive selection. The best scrubbed/no-slot primitive report
`.generated/go2_hidden_target_memory/go2_memory_command_policy_hidden_return_scrubbed_seed20260627_noslot_report.json`
gets oracle-target primitive accuracy `0.475` versus `0.300` majority, but
reset-state accuracy is `0.600` and shuffle is `0.525`; learned-gate positive
pipeline success is only `0.25`.

The next bridge is target-relative geometry memory, not more primitive-head
tuning. Initial scrubbed geometry probes are partial but useful:

- `.generated/go2_hidden_target_memory/go2_memory_target_geometry_hidden_return_scrubbed_seed20260620_noslot_report.json`:
  mean bearing error `50.3` deg, range MAE `0.37 m`, steering-bucket accuracy
  `0.675`, reset `0.300`, reverse `0.425`, shuffle `0.550`;
- `.generated/go2_hidden_target_memory/go2_memory_target_geometry_hidden_return_scrubbed_seed20260622_noslot_report.json`:
  mean bearing error `46.5` deg, range MAE `0.30 m`, steering-bucket accuracy
  `0.650`, but weaker corruption separation.

These strict event slices are not full episodes, but they are also not
single-frame-only labels: rendering/join preserved `270` train and `52`
validation `first_visible_evidence` events for the selected current hidden
targets.

Follow-up after evidence-frame supervision: target-geometry improved but still
does not meet the controller proxy. The best geometry-only command extractor
uses
`.generated/go2_hidden_target_memory/go2_memory_target_geometry_hidden_return_scrubbed_seed20260630_evidence_img96_h128.pt`
and report
`.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_geometry_command_extractor_seed20260630_evidence_img96_h128_nohold_selectall_report.json`.
It reaches oracle-target target-direction command accuracy `0.700` and
target-primitive proxy accuracy `0.575`, but the target-direction majority
baseline is already `0.625`; the best corrupted-history control reaches
`0.575`.

The two-stage scrubbed hybrid proxy also remains below the execution bar. Best
report:
`.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/go2_hybrid_selector27_geometry30_nohold_report.json`.
It reaches target recall `0.625`, target-selection precision `0.694`,
false-claim rate `0.333`, positive-frame target-steering pipeline success
`0.450`, and `0.125` normal-minus-best-corrupted pipeline gap. Selector
thresholds up to `0.8` did not fix this. A full-aux selector diagnostic removes
false claims but still reaches only `0.425` target-steering pipeline success.

Immediate next requirement: improve the scrubbed selector and target-geometry
memory on better-balanced observe-to-hidden slices before closed-loop command
execution. The needed data pressure is specific: more right-turn hidden-return
cases, hard unseen-return negatives, and longer evidence-to-hidden trajectories.
Do not claim closed-loop Go2 return from the full-aux target gate alone.

## Paper Novelty Gate

The paper target is valid only if the system demonstrates a property, not just a
component stack.

Minimum novelty evidence:

- persistent JEPA belief state, not single-frame perception;
- hidden-goal discover-then-return, not given-goal image servoing;
- reachability/topological structure measured directly in latent memory;
- strict runtime boundary without DINO, depth, pose, occupancy, or route
  breadcrumbs;
- baselines against DINO/DINOv2 features, supervised map heads, and
  Neural-SLAM/CMP-style learned-map plus analytical planning.

Without Go2 transfer this is a controlled-memory systems result. With strict
Genesis-Go2 transfer it becomes a stronger robotics simulation paper. With
physical Go2 under the same runtime contract it becomes the main systems claim.

## Immediate Next Steps

1. Treat the current broad Phase 3A aggregate as the frozen baseline.
2. Stop broad router/threshold sweeps unless they test a new causal hypothesis.
3. Treat the current reachability-conditioned planner as failed in closed loop:
   `2/64` versus the strong unconditioned `58/64` baseline.
4. Add exporter progress logging to every long eval command, for example
   `--progress-every-episodes 8`.
5. Diagnose the Phase 3B failure on paired episodes before retraining: selected
   actions, target field, sparse target, reachability feature channels, marker
   observation, and latent marker memory.
6. Add a capacity-matched no-reachability or zero/shuffled-reachability ablation
   before claiming the reachability conditioning is causal.
7. Do not run broad Phase 3B retuning as a pre-Go2 blocker. Preserve the current
   negative result and use it as motivation for later latent-ordering work.
8. Continue the Genesis-Go2 hidden-goal bridge under scrubbed command aux. The
   next iteration is data/model improvement for the hybrid selector plus
   target-geometry proxy. Only after the offline hybrid proxy is materially
   safer should we run command-block execution with memory-on, reset,
   reversed-history, and shuffled-history controls. The Go2 milestone is
   translatability of the learned-memory scaffold, not proof that latent
   topology has already emerged.
