# Predictor capacity: 24×1024×16 shape-matched AdaLN successor vs the 17.2M control

Date: 2026-08-07
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** No manifest or authorization
status is inherited. `probability_calibration`, `evaluation`, `untouched` and
sealed data were never opened. The encoder never moved and was never executed in
the training run.

Artifacts: `/home/andrewknowles/.cache/lewm_go2_temporal_v03/temporal_action_jepa_v1/`
(`arm_frozen_official_scale/`, `capacity_curves/`)

---

# DECISION

> ## REJECT CAPACITY AS SUFFICIENT
>
> **Predictor capacity alone is insufficient within the current one-step AdaLN
> architecture.**

This does **not** reject the official 305M action-token predictor architecture,
which was not tested here.

Which gates failed, stated precisely:

- **Latent prediction improved slightly; geometry stayed below persistence.**
  Selection dense L1 0.36496 → 0.36232 and interface MSE 0.32035 → 0.31847, both
  in the large model's favour — while predicted-future occupied IoU went
  0.2694 → 0.2494, *away* from the persistence gate of 0.3133.
- **Action sensitivity did not regress, and did not improve.** Margin
  0.0499 → 0.0493, still below the +0.0586 gate. 6 of 8 scenes marginally lower.
- **Open-obstacle-field remained unresolved.** 0.1193 → 0.1201, still below
  persistence 0.1329 and far below the true-future reference 0.2452.
- **Canonical-interface compatibility passed but is not reassuring.** The
  capacity model is slightly closer to the true future in aggregate token error,
  but it still loses local occupied structure under the canonical true-future
  spatial readout.

## SCOPE CORRECTION (2026-08-08)

A later continuation of the one-step line to epoch 11 showed that **six epochs was
undertrained**: step-one occupied IoU kept climbing to `0.3354` (control, epoch 10)
and crossed the persistence gate of `0.3128`, which no six-epoch run had done.

Two consequences for this document:

1. **What survives unchanged.** At a matched six-epoch budget, a 26.6x larger
   predictor produced **no early or sample-efficiency gain** — the arms tracked
   within ~0.003 on every selection metric from epoch 2 onward. That comparison
   is sound and is unaffected.

2. **What must be narrowed.** The stronger reading — that capacity does not help
   *at convergence* — is **not supported**, because neither arm was converged. The
   convergence guard did not fire at epoch 5 (delta −0.0102) but that reflected a
   local dip, not a plateau: the same curve later rose by more than 0.04. Every
   conclusion below is therefore **scoped to the fixed six-epoch schedule**, and
   the possibility that capacity helps under a longer schedule is untested.

## Conclusion, recorded narrowly

> A 24×1024×16, 457,309,184-parameter AdaLN predictor — 26.6× the 17.2M control —
> optimised successfully but did not improve canonical future geometry or action
> sensitivity. **Predictor capacity alone is insufficient within the current
> one-step AdaLN architecture.**

The official 305M action/proprioception-token architecture is **not** rejected.
It was not tested. The predictor-capacity line is closed; no further capacity
run, longer schedule or encoder movement is authorised.

## Optimisation was successful — this is not an undertraining result

The predeclared convergence guard was checked first and did **not** fire. The
epoch-4→5 change in selection occupied IoU was **−0.0102**, against a material
improvement threshold of **+0.005**. The model is past its peak, not still
climbing, so `CAPACITY TEST INCONCLUSIVE` does not apply.

Train loss fell monotonically for all six epochs and ended *below* the control
(0.35156 vs 0.35661). The large model optimised correctly; it simply did not
convert that into future geometry.

---

## 1. Architecture

| | control | capacity successor |
|---|---|---|
| name | 17.2M dense-L1 / output-normalised predictor | **24×1024×16 shape-matched AdaLN capacity successor** |
| blocks | 6 | **24** |
| width | 384 | **1024** |
| heads | 6 | **16** |
| MLP ratio | 4 | 4 |
| token dim in/out | 1024 | 1024 |
| **parameters** | **17,198,080** | **457,309,184** |
| ratio to control | 1.0× | **26.6×** |
| action conditioning | AdaLN | AdaLN, unchanged |
| positional scheme | learned spatial 768 + temporal 3+1 | unchanged |

**Not parameter-matched to the official 305M V-JEPA robot predictor.** The
official model is smaller because it inserts actions and proprioception as
*tokens* rather than through a 1024→6144 AdaLN projection in every block; that
projection alone is ~151M of the 457M here. Preserving AdaLN is correct for a
capacity-only intervention — changing it would have been a conditioning change.

Everything else held from the control: frozen official V-JEPA 2.1 ViT-L encoder,
context `t−480, t−240, t`, target `t+240`, action = command block `t → t+240`,
4,075 train / 491 checkpoint_selection sequences, all eight families, seed
`2026080651`, ordered rows, AdamW, lr 3e-4, wd 0.01, clip 1.0, effective batch 4,
6 epochs, final-epoch selection, dense L1, LayerNorm on target and output,
one-step teacher-forced only. No rollout, proprioception, action tokens or extra
context.

## 2. Feasibility receipt

| | value |
|---|---:|
| parameter count | 457,309,184 |
| peak VRAM | **16.414 GiB** of 31.86 |
| microbatch / accumulation | 4 / 1 (none needed) |
| activation checkpointing | not needed |
| precision | bf16 autocast, **unchanged** from the control |
| step time | 1.46 s |
| steps/epoch | 1,019 |
| estimated / actual full run | 2.48 h / ~2.4 h |
| loss finite | ✅ 1.0381 |
| gradient norm finite | ✅ 0.8658 |
| blocks receiving gradient | **24 / 24** |
| encoder gradients | **0** |

The encoder is frozen by construction, not by flag: this run never instantiates
it, consuming cached frozen-encoder tokens, so no encoder graph, EMA or
fine-tuning path exists.

## 3. Training and selection curves

| ep | train L1 ctrl | train L1 cap | sel L1 ctrl | sel L1 cap | margin ctrl | margin cap | occ IoU ctrl | occ IoU cap | occ frac ctrl | occ frac cap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.45960 | 0.46725 | 0.40944 | 0.41758 | +0.0176 | +0.0042 | 0.0684 | 0.0600 | 0.00236 | 0.00197 |
| 1 | 0.39045 | 0.39240 | 0.38830 | 0.38855 | +0.0262 | +0.0227 | 0.1042 | 0.1150 | 0.00434 | 0.00376 |
| 2 | 0.37430 | 0.37143 | 0.37776 | 0.37424 | +0.0446 | +0.0459 | 0.1746 | 0.2046 | 0.00676 | 0.00894 |
| 3 | 0.36621 | 0.36199 | 0.37145 | 0.36961 | +0.0482 | +0.0479 | **0.2676** | **0.2651** | 0.01157 | 0.01222 |
| 4 | 0.36063 | 0.35585 | 0.36686 | 0.36259 | +0.0505 | +0.0504 | 0.2520 | 0.2597 | 0.00880 | 0.00947 |
| 5 | **0.35661** | **0.35156** | **0.36496** | **0.36232** | **+0.0499** | **+0.0493** | **0.2694** | **0.2494** | 0.01120 | 0.01173 |

The two arms track each other to within ~0.003 on every selection metric from
epoch 2 onward. A 26.6× capacity increase moved selection dense L1 by 0.0026 and
occupied IoU by −0.0200.

## 4. Prediction, final epoch

| | control | capacity |
|---|---:|---:|
| correct−shuffled margin | +0.0499 | +0.0493 |
| gate | +0.0586 | +0.0586 |

Per-scene margin, all eight:

| scene | family | control | capacity |
|---|---|---:|---:|
| `large_enclosed_maze_d78318b1e87b` | large_enclosed_maze | +0.0635 | +0.0600 |
| `local_composite_motifs_811b818f1914` | local_composite_motifs | +0.0575 | **+0.0583** |
| `medium_enclosed_maze_f30352cb052e` | medium_enclosed_maze | +0.0550 | **+0.0564** |
| `loop_alias_stress_aeb36ab10bc1` | loop_alias_stress | +0.0499 | +0.0497 |
| `visual_sensor_stress_dc440a3fb679` | visual_sensor_stress | +0.0478 | +0.0450 |
| `small_enclosed_maze_16b0fc2c449b` | small_enclosed_maze | +0.0388 | +0.0384 |
| `rough_local_dynamics_0e631dbfbd46` | rough_local_dynamics | +0.0364 | +0.0360 |
| `open_obstacle_field_25cc6fe2de4f` | open_obstacle_field | +0.0346 | +0.0336 |

Capacity is higher in 2 of 8 scenes, lower in 6, by margins of ≤0.003. No scene
reaches the gate.

## 5. Canonical future geometry, fixed true-future probe

Denominators: 491 frames, 2,011,136 cells, 263,239 observable, 14,098 occupied;
occupied is **5.36%** of observable cells and **0.701%** of all cells.

| | occ IoU | occ P | occ R | UNKNOWN IoU | free IoU | occ fraction |
|---|---:|---:|---:|---:|---:|---:|
| true future (reference) | 0.4970 | 0.6398 | 0.6900 | 0.9794 | 0.9551 | 0.01602 |
| **persistence (gate)** | **0.3133** | 0.5076 | 0.4501 | 0.9648 | 0.9025 | 0.01643 |
| control 17.2M | 0.2694 | 0.5649 | 0.3400 | 0.9680 | 0.9358 | 0.01120 |
| **capacity 457M** | **0.2494** | 0.5112 | 0.3276 | 0.9676 | 0.9294 | 0.01173 |
| *(all-free free-IoU baseline)* | | | | | *0.9464* | |
| *(target occupied fraction)* | | | | | | *0.00701* |

Both predictors sit below persistence and far below the reference. Note free IoU
for both predictors (0.9358, 0.9294) is **below** the all-free baseline of 0.9464,
so free space carries no signal here, as in every previous stage.

## 6. Per-family predicted occupied IoU

| family | control | capacity | persistence | true future |
|---|---:|---:|---:|---:|
| `small_enclosed_maze` | 0.4963 | 0.4487 | 0.6219 | 0.7728 |
| `rough_local_dynamics` | 0.3945 | 0.3775 | 0.4095 | 0.4954 |
| `medium_enclosed_maze` | 0.2913 | 0.2588 | 0.2658 | 0.5134 |
| `large_enclosed_maze` | 0.2877 | 0.2730 | 0.3216 | 0.5640 |
| `visual_sensor_stress` | 0.2429 | 0.2233 | 0.3326 | 0.5602 |
| `local_composite_motifs` | 0.2415 | 0.2252 | 0.3815 | 0.5840 |
| `loop_alias_stress` | 0.1864 | 0.1736 | 0.2040 | 0.3651 |
| **`open_obstacle_field`** | **0.1193** | **0.1201** | 0.1329 | 0.2452 |

The control beats persistence in 1 of 8 families (`medium_enclosed_maze`); the
capacity model in 0 of 8.

### `open_obstacle_field` headline

| | occ IoU | precision | recall |
|---|---:|---:|---:|
| control 17.2M | 0.1193 | **0.2880** | 0.1691 |
| **capacity 457M** | **0.1201** | 0.2500 | 0.1877 |
| persistence | 0.1329 | 0.1930 | 0.2993 |
| true future | 0.2452 | 0.3502 | 0.4498 |

Change relative to the 17.2M predictor: **+0.0008 IoU** — noise. Recall rose
0.1691 → 0.1877 while precision fell 0.2880 → 0.2500. Still below persistence,
still less than half the reference.

## 7. Occupied-volume diagnostics

| | predicted occ. fraction | ×target | precision |
|---|---:|---:|---:|
| target | 0.00701 | 1.00× | — |
| true future | 0.01602 | 2.29× | 0.6398 |
| persistence | 0.01643 | 2.34× | 0.5076 |
| control | 0.01120 | 1.60× | 0.5649 |
| capacity | 0.01173 | 1.67× | 0.5112 |

Gate 4 passes: neither predictor achieves anything through diffuse
over-prediction — both predict *less* occupancy than persistence, at precision at
or above it. The deficit is recall (0.3276 / 0.3400 against persistence 0.4501
and reference 0.6900): both predictors erase occupied structure rather than
smearing it.

## 8. Verification

**Evaluation path.** Both checkpoints evaluated through the same verified fp16
frozen-encoder caches, the same fixed true-future probe, the same frozen
changed-token mask (threshold 0.76190, 94,540 of 377,088 tokens) and the same
derangement seeds (11, 23, 37). The encoder is never executed in evaluation.

**Determinism, verified.** Both checkpoints evaluated twice under identical
inputs produced **bit-identical** predictions: `prediction_max_abs_diff = 0.0`,
changed-cosine delta `0.0`, occupied-IoU delta `0.0` for each arm. The 0.0200 IoU
and 0.0006 margin differences between arms are therefore signal, not evaluation
noise. Checkpoint hashes: control `0858a4a5…`, capacity `c05e56da…`; fixed probe
`f053b4e2…`.

**Derangement identity, verified.** The three shuffled-action permutations over
491 rows are seed-reproducible from (11, 23, 37), fixed-point-free, genuine
permutations, pairwise distinct and identical across both arms — hashes
`545b2701…`, `da7a93d0…`, `974379c8…`. The changed-token mask (threshold
0.7618998289108276, 94,540 of 377,088 tokens) and the fixed probe are the same
objects for both arms.

**Consistency anchor.** The control's epoch-5 row reproduces the earlier
standalone successor evaluation exactly — margin +0.04986, occ IoU 0.26942,
occ fraction 0.011205 — confirming the identical-path requirement holds.

**Cache-parity deviation, recorded.** The 17.2M control *trained* on live float32
encoder features; the capacity successor trained on the float16 caches from the
same frozen encoder. Measured parity on a fixed 24-row subset: feature max |Δ|
0.0156, mean |Δ| 2.0e-4, relative 1.76e-4, derived cosine/error agreeing to 1e-6.
Eliminating it would have required retraining the baseline, which was excluded.

## 9. Defects recorded during this work

- **Non-faithfully-resumable checkpoints (`arm_frozen_official_scale`).** Verified
  from the bytes of `checkpoint_epoch1.pt` (SHA-256 `1e46de9b…`, 1,829,327,779
  bytes): top-level keys are only `['predictor', 'encoder_trainable']`. Absent:
  epoch, global step, optimizer state (no `exp_avg`/`exp_avg_sq`), Python/NumPy/
  torch-CPU/torch-CUDA RNG and the data-order generator. No scheduler is used in
  this run, so its absence is correct rather than a defect. I had wrongly claimed
  the run was covered by resume support added mid-flight; a running interpreter
  does not pick up source edits. Recorded in `resumability_note.json`; the run was
  not restarted or warm-started.
- **Per-scene recomputation inefficiency.** The final-epoch per-scene block
  recomputed the three shuffled-action predictions inside the 8-scene loop — 24
  full 491-row passes at 457M parameters where 3 suffice, ~4.6 h of avoidable
  compute. Deterministic and therefore correctness-neutral; my ETA statements
  were wrong for this reason. Recorded in `procedural_defect_note.json` and fixed
  in `eval_dev_v03_capacity_curves_v2.py`.
- **Checkpointing rebuilt.** `scripts/dev_checkpoint_v1.py` now stores model,
  optimizer, scheduler (or an explicit absence reason), epoch, global step, seed,
  model config and all RNG streams including the data-order generator; writes via
  temp file → `fsync` → atomic `replace` → directory `fsync`, then reloads to
  verify; refuses to write incomplete state or to resume from it. Wired into both
  training runners for future invocations.

## 10. Next

The next isolated official-recipe discrepancy is the **two-step autoregressive
rollout objective** (`auto_steps: 2`, `jloss + sloss`), not encoder movement.
**Not launched and not authorised by this document.**

Remaining untested official-path differences, for the record: 8-frame context
(vs 3), 2-step rollout, proprioception, and token-wise action injection in place
of AdaLN.

The operational gate is unchanged: a frozen-encoder predictor must beat
persistence (0.3133) on future occupied geometry under the fixed true-future
probe before encoder movement is reintroduced.
