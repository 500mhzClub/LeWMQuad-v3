# Topological Nav — RESUME / Handoff (2026-06-09)

Pick-up point for the topological-navigation build (formerly "H-JEPA"). Read this
first, then the per-stage docs linked at the bottom. Authoritative spec:
`docs/v3_topological_nav_plan.md`; build sequence:
`docs/lewm_topological_nav_implementation_plan_2026-06-09.md`.

## TL;DR — where we are

- The architecture is **one flat JEPA (LeWM, frozen) + topological recognition
  memory + hierarchical planner** (SPTM lineage), NOT an H-JEPA. (Renamed
  2026-06-09; "H-JEPA" kept only as a deferred research bet, plan §8.)
- **Stage 0 (planner refactor): DONE + committed** (commit `696f6f4`).
- **Stage 1 (recognition re-validation + H2 cheap test): DONE.** Decision: build
  the BeliefEncoder.
- **Stage 2 (BeliefEncoder): DONE.** H2 is **supported** — a learned history
  encoder beats naive pooling + single-frame (v6: eval R@5 0.637 vs 0.593 vs
  0.582, +0.0436, 3/3 seeds, R@1 +0.042). The registered +0.05 gate **formally
  failed by 0.006** on an unsaturated data curve; decision (registered before
  running): the Stage-2→3 call moves to the **Stage 3a consumer-side gate** —
  LoopClosureHead recall at 99% precision on v6-belief vs single-frame vs
  mean-pool (spec §5.3 bars + belief ≥ +5 pp recall over single-frame, 3/3
  seeds). See the Stage 2 doc "v6 + decision".

## Environment (READ — non-obvious, will bite on resume)

- **GPU torch:** `~/TinyQuadJEPA/bin/python` (torch 2.10 dev, ROCm 6.3, `cuda
  True` on the AMD R9700). Use `--device cuda`.
- The scripts reference `.generated/venvs/genesis_rocm` — **that venv is GONE.**
  `.generated/venvs/genesis_render_vulkan` exists but is **CPU-only torch**
  (2.12+cu130, `cuda False`) and is for genesis/vulkan rendering, not GPU compute.
- `tqdm` was pip-installed into the TinyQuadJEPA venv (the probe scripts need it).
- The retrieval/belief scripts read **pre-rendered PNGs** + encode through the
  frozen LeWM — pure torch, **no genesis needed** — so they run in TinyQuadJEPA.
  Only the closed-loop benchmark needs the vulkan venv + `--apply-textures
  --backend vulkan`.
- **Base checkpoint:** `models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9.pt` (frozen).
- **Data roots (defaults in the scripts):** rollout `.generated/datagen_full/rollout`,
  render `.generated/datagen_full/render_textured_v03`, corpus
  `.generated/scene_corpus/minimum_tex_20260520T211541Z`. `test_id` = held-out eval split.
- All Stage-1/2 artifacts live in `.generated/topo_nav/` (gitignored).

## Stage 0 — planner refactor (DONE, committed `696f6f4`)

Benchmark planner extracted into a genesis-free seam under `lewm/`:
`planning/{primitive_bank,costs,local_mpc,hierarchical_planner}.py`,
`memory/topological_memory.py` (abstract `Memory` + `KeyframeMemory` baseline).
`scripts/benchmark_lewm_closed_loop_mpc.py` delegates to it. Behaviour-lock:
`~/TinyQuadJEPA/bin/python lewm/tests/test_planning_refactor.py` (6/6).
**`HierarchicalPlanner` + `Memory` is the seam Stage 3 plugs the learned memory into.**

## Stage 1 — recognition re-validation + H2 cheap test (DONE)

Doc: `docs/lewm_topological_nav_stage1_retrieval_2026-06-09.md`. On frozen seq4, 32 eval scenes:
- Place recognition re-validated: **R@1 0.43, R@5 0.64, lift ~21×, graph-ρ 0.08**
  (recognition-not-metric confirmed on seq4; 0.42 was not a seq11 artifact).
- Learned single-frame supcon head: **fails** (adds nothing → lever is history).
- Naive frozen-history pooling: **+0.014 R@5, fails the recall gate** (cheap
  shortcut closed).
- History-disambiguability AUC on aliased pairs: **0.81→0.86** (+0.04–0.055 over
  single-frame) → signal present & history-separable → **build the BeliefEncoder.**

## Stage 2 — BeliefEncoder (IN PROGRESS)

Doc: `docs/lewm_topological_nav_stage2_belief_encoder_2026-06-09.md`.
Code: `lewm/models/belief_encoder.py` (small Transformer + attention-pool over H=8
frozen latents → L2-normalized place embedding; **pure supervised contrastive,
NO anti-collapse regularizer** — negatives suffice; the repo's SIGReg is for the
negative-free world-model objective, not a contrastive head).
Script: `scripts/train_belief_encoder.py`. Tests: `lewm/tests/test_belief_encoder.py` (6/6).
Gate: beat naive-pooling R@5 (0.593) by +0.05 across 3 seeds, R@1 non-regression.

Run arc (all seq4, scene-disjoint train/eval; naive bar R@5=0.593, single-frame ~0.576):

| run | json (`.generated/topo_nav/`) | config | train R@5 | eval R@5 | Δ naive | note |
|---|---|---|---:|---:|---:|---|
| v1 | `belief_encoder_seq4_e9.json` | +VICReg 1.0 | n/a | 0.574 | −0.019 | **confound** (mis-applied VICReg, train_loss stuck 1.05) |
| v2 | `..._v2.json` | supcon, 32 tr, big | n/a | 0.564 | −0.029 | fits (loss 0.11) but **overfits** cross-scene |
| v3 | `..._v3_train16.json` | supcon, 127 tr, big | 0.923 | 0.593 | parity | 4× data → naive parity; train≫eval |
| **v4** | `..._v4_small.json` | **small+reg, 127 tr** | 0.854 | **0.621** | **+0.028** | **capacity was the limiter; beats naive 3/3** |
| v5 | `..._v5_reg.json` | small+more reg | 0.877 | 0.617 | +0.024 | reg saturated; v4 is the sweet spot |
| v6 | `..._v6_train32.json` | **small + 2× data** | 0.830 | **0.637** | **+0.0436** | gate failed by 0.006; curve unsaturated; 3/3 seeds; → Stage 3a consumer gate |

**v4 config (the winner):** `--hidden 128 --n-layers 1 --embedding-dim 64
--dropout 0.3 --weight-decay 3e-3 --epochs 80`. Saved encoders:
`.generated/topo_nav/belief_encoder_seq4_e9_v4_small_encoders/belief_encoder_seed*.pt`.

**Settled:** H2 supported; the **DINOv2 substrate-fork (§6) is NOT indicated** —
the encoder already beats the free baseline. Two confounds (VICReg term; then
over-capacity/under-data) each masqueraded as failure and were removed first.

### How to read v6 when it finishes
```
~/TinyQuadJEPA/bin/python - <<'PY'
import json; d=json.load(open(".generated/topo_nav/belief_encoder_seq4_e9_v6_train32.json"))
print("eval R@5 across seeds:", d["learned_summary"]["retrieval_at_5_across_seeds"]["mean"])
print("Δ vs naive R@5:", d["learned_summary"]["mean_recall5_improvement_vs_naive"])
print("train R@5 (last seed):", d["train_recall_last_seed"]["retrieval_at_5"]["mean"])
print("gate passed:", d["gate"]["passed"])
PY
```
**Decision tree:**
- v6 clears naive +0.05 across seeds → **gate passes** → adopt config, go to Stage 3.
- v6 plateaus ~+0.03–0.04 → H2 still supported (modest, real). Choose: (a) proceed
  to Stage 3 with the v4/v6 encoder (the topological memory needs recognition, not
  a record R@5), or (b) one more architecture pass (e.g. mean-pool head, H=12,
  per-family balancing). **Not** a substrate fork either way.

### Bank caches (reuse to skip the ~17–30 min re-encode)
- `.generated/topo_nav/belief_banks_seq4_e9_train16.pt` — 127 train + 32 eval. Fast
  iteration on model/reg: `--bank-cache <that> ...` (skips model load + encode).
- `.generated/topo_nav/belief_banks_seq4_e9_train32.pt` — being built by v6 (bigger).

Example fast iteration (no re-encode):
```
~/TinyQuadJEPA/bin/python scripts/train_belief_encoder.py \
  --checkpoint models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9.pt \
  --output .generated/topo_nav/belief_encoder_seq4_e9_<tag>.json \
  --bank-cache .generated/topo_nav/belief_banks_seq4_e9_train16.pt \
  --hidden 128 --n-layers 1 --embedding-dim 64 --dropout 0.3 --weight-decay 3e-3 --epochs 80 \
  --device cuda
```

## Uncommitted work to commit (Stage 2 unit)

New: `lewm/models/belief_encoder.py`, `scripts/train_belief_encoder.py`,
`lewm/tests/test_belief_encoder.py`,
`docs/lewm_topological_nav_stage2_belief_encoder_2026-06-09.md`,
`docs/lewm_topological_nav_stage1_retrieval_2026-06-09.md` (if not already in
`696f6f4` — it was committed there), this RESUME doc.
Modified: `docs/lewm_topological_nav_implementation_plan_2026-06-09.md` (Stage-0/1
marked done, VICReg→SIGReg correction at §5 Stage 2).
**Excluded as before:** `scripts/build_task_aligned_feature_dataset.py`,
`pose_aux_watcher.log`, `scripts/train_goal_localization_head.py`,
`scripts/train_patch_cross_attention_head.py` (prior unrelated work).
Note: user prefers commits **without** a Co-Authored-By footer.

## Next steps (in order)

1. ~~Read v6; finalize the "## v6 + decision" section of the Stage 2 doc.~~ DONE.
2. ~~Commit the Stage 2 unit (no co-author footer).~~ DONE.
3. **Stage 3a (consumer gate, registered in the Stage 2 doc):** LoopClosureHead
   (`lewm/models/loop_closure.py` + `scripts/train_loop_closure_head.py`) on the
   cached banks — v6-belief vs single-frame vs mean-pool; deployment threshold
   from calibration scenes at precision ≥ 0.99; Platt + ECE; gate = §5.3 bars +
   belief ≥ +5 pp recall over single-frame 3/3 seeds. Result decides adopt-v6 vs
   one action-token/motion-aux pass vs reassess.
4. **Stage 3** (`v3_topological_nav_plan.md` §5–6, plan §5 Stage 3): wire the
   BeliefEncoder into the `Memory`/`HierarchicalPlanner` seam — online node
   commitment with **goal-facing** `representative_observation` (hard constraint
   from the Stage-1/nav findings), calibrated `LoopClosureHead`, top-k Bayes
   filter; then recognition-based `ReachabilityHead` (6 buckets, false-loop
   negatives) and the 3-level planner (Level 3 = seq4 + `plan_cost` LocalMPC,
   already built in Stage 0). GoalAdapter maps a goal image into belief space
   (goal-facing keyframes).
5. Stage 4: end-to-end closed-loop eval (learned subgoals replace the privileged
   breadcrumbs) via `benchmark_lewm_closed_loop_mpc.py --apply-textures --backend vulkan`.

## Detailed docs
- Spec: `docs/v3_topological_nav_plan.md`
- Build plan: `docs/lewm_topological_nav_implementation_plan_2026-06-09.md`
- Stage 1: `docs/lewm_topological_nav_stage1_retrieval_2026-06-09.md`
- Stage 2: `docs/lewm_topological_nav_stage2_belief_encoder_2026-06-09.md`
- Nav-base synthesis (why seq4 + plan_cost): `docs/lewm_nav_base_synthesis_2026-06-09.md`
- Memory index: `MEMORY.md` → "Topological-nav" entries.
