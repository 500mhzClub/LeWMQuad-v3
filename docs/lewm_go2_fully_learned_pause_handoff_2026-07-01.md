# Go2 Fully Learned Demo Pause Handoff - 2026-07-01

## Pause State

Paused at 2026-07-01 20:38 BST after stopping the active v216 GPU fine-tune.

Goal is still open: produce a clean generalized Go2 fully learned demo that succeeds on held-out mazes with no privileged runtime components, no route tables, no pose-topology features, and no oracle explorer calls.

No clean success has been achieved yet. The best current candidates either explore enough but collide/stall, or stay clean but stop progressing after the first target claim.

## Runtime Contract

The intended demo contract remains:

- Runtime inputs allowed: RGB/JEPAs, proprioception/command history, learned local policy outputs, learned primitive outcome/body-risk heads, and online egomotion/local-map state built only from executed actions and local observations.
- Runtime inputs forbidden for the final claim: oracle explorer calls, hand-authored route memory, route tables, pose-topology features, privileged target/maze geometry, and any scene-specific held-out labels.
- Generalization requirement: train on the scene-disjoint train set and evaluate on held-out `medium_enclosed_maze_eeb8320a6934`.

The strict checker to pass is still:

```bash
.generated/venvs/genesis_render_vulkan/bin/python scripts/check_go2_fully_learned_demo.py \
  --result <result> \
  --max-ticks 700 \
  --max-contact-like-stalls 0 \
  --max-hard-stalls 0 \
  --max-body-violations 0 \
  --require-generalized-runtime-contract \
  --forbid-route-memory \
  --forbid-pose-topology-features \
  --train-scenes medium_enclosed_maze_000c67a65968,medium_enclosed_maze_0100925f9754,medium_enclosed_maze_01732aabc542,medium_enclosed_maze_03fa030348c7,medium_enclosed_maze_42ec4c74ee43,medium_enclosed_maze_595b349fbbf7,medium_enclosed_maze_5ae240e5e391,medium_enclosed_maze_62c21394102b,medium_enclosed_maze_70b773fdffb2,medium_enclosed_maze_a25fd29d9278,medium_enclosed_maze_abb9ac953e00,medium_enclosed_maze_b8c906fc9e8e,medium_enclosed_maze_df939d2d7b68
```

## GPU Training State

GPU availability was confirmed earlier through the TinyQuadJEPA environment:

- CUDA visible through ROCm shim: true.
- Devices: `AMD Radeon AI PRO R9700`, `AMD Radeon Graphics`.
- During v216 training, `rocm-smi` showed GPU 0 around 95-100 percent busy.

The active v216 run was stopped on pause request. Last flushed training line:

```text
epoch=60 loss=0.1419 val_acc=0.939 val_macro_f1=0.939
```

The checkpoint was updated at 20:38:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v216_frontier_finetune_20260701/generalized_learned_local_mapcnn_v216_frontier_v133init_h384.pt
```

No report JSON exists for v216 because the run was terminated before normal end-of-training report write. The checkpoint metadata confirms:

- schema: `lewm_go2_closed_loop_learned_local_policy_v0`
- model type: `map_cnn`
- feature variant: `clock_state_online_map_edge_v1`
- online map size/channels: `21`, `8`
- hidden dim: `384`
- batch size: `8192`
- device used by command: `--device cuda`

Synthetic v216 datasets remain:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v216_frontier_finetune_20260701/v216_frontier_synth_train.npz
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v216_frontier_finetune_20260701/v216_frontier_synth_val.npz
```

Disk is tight: about `4.0G` free on the workspace filesystem at pause time.

## Main Artifacts

Primary exploration policy before v216:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v133_postclaim_dagger_20260630/generalized_learned_local_mapcnn_v133_balanced_postclaim_v132init_h384.pt
```

Paused v216 frontier fine-tune:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v216_frontier_finetune_20260701/generalized_learned_local_mapcnn_v216_frontier_v133init_h384.pt
```

Post-claim policy:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v176_postclaim_oracle_dagger_20260630/generalized_learned_local_mlp_v188_masked_visualreadout_state_h384.pt
```

Best target policy currently used:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v191_green_target_20260630/generalized_learned_local_mapcnn_v213_green_standoff_escape_nomapmask_h384.pt
```

Learned safety/outcome heads:

```text
.generated/go2_wallaware_learned/primitive_outcome_jepa_broad_train24_block080_v95.pt
.generated/go2_wallaware_learned/primitive_body_clearance_jepa_v106_train32val32_obstacle_margin002_afterstart_h192.pt
.generated/go2_wallaware_learned/current_body_risk_jepa_body040_half024_margin006_v1.pt
```

## Latest Held-Out Results

Best clean-ish prior held-out candidate:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v191_green_target_20260630/heldout_v213_frontier_recovery_rerank_nohold_combinedretry3_seed1_fixedorder_result.json
```

- success: false
- claimed: `red`
- contact-like stalls: `0`
- hard stalls: `0`
- body clearance violations: `0`
- final distance: `4.486 m`
- issue: stays clean but does not cover enough maze after red.

Current-code coverage candidate:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v216_frontier_finetune_20260701/heldout_v133_current_origcommit_seed1_fixedorder_result.json
```

- success: false
- claimed: `red`
- saw `yellow` at tick `486`
- final distance: `1.006 m`
- contact-like stalls: `234`
- hard stalls: `229`
- body clearance violations: `0`
- issue: explores, but only because unsafe forced single-candidate commits are allowed.

Current-code clean but frozen candidate:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v216_frontier_finetune_20260701/heldout_v133_recovery_rerank_combinedretry3_hard082_seed1_fixedorder_result.json
```

- success: false
- claimed: `red`
- contact-like stalls: `0`
- hard stalls: `0`
- body clearance violations: `0`
- body clearance hard vetoes: `637`
- forward executions: `14`
- wall vetoes: `640`
- visited online-map cells: `7`
- issue: the hard body-clearance threshold is too conservative and blocks coverage.

v216 epoch-20 held-out eval:

```text
.generated/go2_memory_closed_loop/generalized_learned_local_suite_v216_frontier_finetune_20260701/heldout_v216_frontier_recovery_rerank_combinedretry3_seed1_fixedorder_result.json
```

- success: false
- claimed: `red`
- contact-like stalls: `2`
- hard stalls: `2`
- body clearance violations: `0`
- forward executions: `28`
- frontier noops: `437`
- issue: v216 synthetic frontier fine-tune did not improve coverage and made the run less clean.

v216 epoch-60 checkpoint has not been evaluated yet after the pause.

## What Changed In This Iteration

Touched implementation files:

- `scripts/train_go2_closed_loop_learned_local_policy.py`
  - Filters update-only rows out of supervised labels while still allowing metadata-driven online-map replay updates.
  - Adds online-map reset support and guard-blocked primitive marking for training features.

- `scripts/synthesize_go2_visual_bearing_aug_dataset.py`
  - Adds synthetic visual-bearing contexts for target standoff escape and target servo approach.
  - Adds guard-probe/update-only rows for online-map feature training.

- `scripts/run_go2_generalized_learned_local_eval.sh`
  - Adds env passthroughs for frontier recovery reranking, preserve-turn/backward settings, and current-body-risk knobs.

- `scripts/benchmark_go2_memory_closed_loop.py`
  - Adds recovery-rerank candidate override support in the learned local action guard.
  - Separates escape forcing from frontier recovery reranking so reranked candidates are not scored as forced escape.
  - Adds parser/result plumbing for the new frontier guard recovery settings.

Validation already run before the pause:

```text
.generated/venvs/genesis_render_vulkan/bin/python -m py_compile scripts/benchmark_go2_memory_closed_loop.py
bash -n scripts/run_go2_generalized_learned_local_eval.sh
```

No final strict checker pass exists.

## Current Diagnosis

The held-out blocker is now narrow and concrete:

- Forced frontier commits give enough coverage to see later targets, but they create contact-like/hard stalls.
- Recovery-reranked frontier commits remove stalls, but the learned body-clearance hard veto becomes too conservative and traps exploration around the first claimed target.
- Raising the hard-veto threshold to `0.82` made the freeze worse.
- Increasing the clearance margin to `0.10` allowed a little more motion but reintroduced a hard stall and still did not improve coverage.
- v216 synthetic frontier fine-tuning improved validation metrics but did not translate to held-out closed-loop coverage.

The most likely next code change is not another broad training run. Add a runtime-contract-clean guard calibration knob so the body-clearance hard veto only overrides a translating frontier recovery action when the replacement action is clearly safer. For example, require the replacement primitive's learned clearance blocked probability to be below a cap such as `0.70-0.72` before vetoing a requested translating primitive. This keeps the runtime learned/online-only while avoiding cases where a forward action at `0.79` is replaced by a yaw action at `0.74`, which removes motion without buying meaningful safety.

After that, rerun the held-out fixed-order eval with:

- v133 explore policy first, because it still has better coverage behavior than v216.
- frontier recovery rerank enabled.
- combined blocked retry after 3 noops.
- body hard veto threshold around `0.78`.
- new replacement-safety cap sweep around `0.70`, `0.72`, and `0.75`.

Only if that produces a clean held-out run with coverage should v216 epoch-60 be evaluated or retrained further.

## Resume Checklist

1. Confirm no training is running:

```bash
pgrep -af '[t]rain_go2_closed_loop_learned_local_policy.py'
```

2. Avoid generating large new datasets until disk is cleared; only about `4.0G` was free at pause.

3. Implement the body-clearance replacement-safety cap in:

```text
scripts/benchmark_go2_memory_closed_loop.py
scripts/run_go2_generalized_learned_local_eval.sh
```

4. Compile/syntax check.

5. Run the v133 held-out eval sweep first.

6. Run `scripts/check_go2_fully_learned_demo.py` only after a result both succeeds and has zero contact-like, hard, and body-clearance violations.

