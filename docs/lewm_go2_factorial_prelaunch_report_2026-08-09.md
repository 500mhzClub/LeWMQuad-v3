# Four-cell factorial: modality contract, driver, evaluator and pre-launch state

Date: 2026-08-09
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** The four-cell experiment has
**not** been launched. Only toy, fixture and dry-run checks were executed. No
24-epoch cell was trained, no seed quadruplet was started, no scientific selection
outcome was evaluated, and the sealed benchmark was not inspected.

---

# 1. Final modality contract

## 1.1 Audit: the previous applied commands were not duplicated, and not shared

Measured on the manifest: the stored previous-applied history covers steps
`s−15 … s−1`; the action covers `s … s+4`. They are **disjoint by construction**.
They coincide only when the command happens to be steady, in **676 of 2,000** rows
sampled — a coincidence of constant commands, not a redundancy.

They were also **not available to the RGB cells at all**. Leaving them in the
proprioceptive tensor would therefore have confounded the proprioception factor
with control history: any effect could have been the efference copy rather than
sensed state.

**Resolution:** they move to an explicitly named **`control_history`**
(efference-copy) input, consumed **identically by all four cells**.

## 1.2 The three model-facing tensors

| tensor | shape | dims | present in |
|---|---|---:|---|
| **action** | `(B, 10)` per transition | 10 = 5 ticks × (vx, yaw) | all four cells |
| **control_history** (efference copy) | `(B, 3, 5, 2)` → 10 per slot | 2/sample | **all four cells** |
| **proprio** (the experimental factor) | `(B, 3, 5, 30)` | 30/sample | proprio cells only |

`proprio` is now **sensed physical state only**:

| channels | dims |
|---|---:|
| projected gravity (offset feature) | 3 |
| body angular velocity (gyro) | 3 |
| joint positions | 12 |
| joint velocities | 12 |
| **total** | **30** |

Excluded: previous applied command (moved to `control_history`), lateral command
`vy` (identically zero), body linear velocity, absolute yaw, world pose, `/odom`,
camera extrinsics, foot contacts, joint effort, IMU linear acceleration.

**Control history at rollout** needs no validity mask. The appended slot of a
predicted frame is exactly the previous action block —
`control_slot_from_action(a_{h-1})` — so the efference copy is a deterministic
function of the action plan, involves no measurement and no future observation,
and stays available at every horizon. Proprioception keeps its absence token and
the 3 → 2 → 1 → 0 observed-slot schedule.

## 1.3 Parameter counts per cell

| model | parameters |
|---|---:|
| frozen predictor (12-D primitive action, pre-existing) | 17,198,080 |
| **RGB cells** (`rgb_one_step`, `rgb_rollout`) | **17,201,920** |
| **Proprio cells** (`proprio_one_step`, `proprio_rollout`) | **17,214,592** |

| delta | value |
|---|---:|
| corrected action + control history vs frozen | **+3,840** |
| proprioception vs the RGB cells | **+12,672** (0.074 %) |

Shared in all four cells: `control_in.weight` 3,840, `control_in.bias` 384,
`control_modality` 384. Proprio-only: `proprio_in.weight` 11,520,
`proprio_in.bias` 384, `proprio_modality` 384, `proprio_absent` 384.

The two arms differ by the proprioceptive path and nothing else; every other
parameter is bit-identical at a shared seed, verified per cell by the driver and
by test.

---

# 2. Projected-gravity normalisation

The stored feature is

```
g_feature = g_body − (0, 0, −1)
```

with **no corpus-derived scaling**. The three components keep one shared physical
scale and their mutual geometry. In the frozen statistics the gravity slice
carries `mean = [0, 0, 0]` and `std = [1, 1, 1]` as **fixed constants**, recorded
as such, not as corpus estimates.

Audited values and tests (all passing):

| audit | result |
|---|---|
| projected-gravity norm error, `‖g_feature + (0,0,−1)‖ − 1` | **< 1e−9** over 1,500 rows × 15 samples |
| component ranges | `x, y ∈ [−1, 1]`; `z ∈ [0, 2]` |
| finiteness | every proprioceptive value finite |
| gravity uses no corpus statistic | `mean == [0,0,0]`, `std == [1,1,1]`, string asserts FIXED CONSTANTS |
| all other channels use training-only statistics | `source == "train split only"`, 59,490 train samples |
| frozen statistics reused exactly at selection/planning | content hash re-derived and matched against the recorded `sha256`, and against the manifest's `normalisation_sha256` |

---

# 3. Hardened action and manifest contracts

The 10-D five-tick post-slew action is preserved. Hard failures now raise and
**refuse to write a manifest**:

| condition | mechanism |
|---|---|
| non-zero requested or applied `vy` in a manifest | `ContractViolation` during scene build |
| a planning candidate carrying lateral motion | `LateralMotionRejected` in `reconstruct_block` (also rejects a lateral previous state) |
| a scene not covered by action-reconstruction validation | set-difference check against the recorded `scene_ids` |
| a sequence crossing a reset | proprioceptive window and action span both checked |
| reconstruction differing from the logged post-limiter trace | `ContractViolation` on any mismatched block |

Action-reconstruction validation now covers **exactly the 80 manifest scenes**:
**768,000 / 768,000 blocks (100 %)**, tick-exact 100 %, recorded with the scene
list in `action_reconstruction_validation.json`.

## Rows kept and dropped, by split and family

**4,444 of 4,566 kept — 3,966 train / 478 selection.** No row was dropped for
action verification.

| reason code | total | by split/family |
|---|---:|---|
| `proprio_history_absent` (window starts before the episode) | 90 | train: visual_sensor_stress 22, local_composite_motifs 17, large_enclosed_maze 10, loop_alias_stress 7, medium_enclosed_maze 7, rough_local_dynamics 7, small_enclosed_maze 6, open_obstacle_field 3 · selection: rough_local_dynamics 4, large_enclosed_maze 2, medium_enclosed_maze 2, loop_alias_stress 1, open_obstacle_field 1, visual_sensor_stress 1 |
| `proprio_history_crosses_reset` | 32 | train: local_composite_motifs 17, rough_local_dynamics 7, small_enclosed_maze 3, open_obstacle_field 2, visual_sensor_stress 1 · selection: local_composite_motifs 2 |

## Frozen hashes

| artefact | sha256 |
|---|---|
| **configuration (current, frozen)** | `582e7088c2230963fa9b5a0acde4e3de0a863d4c55af74dd7c53d5c1eb18497a` |
| `proprio_rows.jsonl` | `7b79d12830f12175c591a87982a20e5df7a8d64cfc40e99dd9cee2dc1ae2543e` |
| normalisation statistics | `f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab` |
| seed registry | recorded in `factorial_v1/seed_registry.json` |

> **`f410df7989fd639761b7177c00cc6d12fb9db15a1a6c46d9898a4c7bd6f7e0c8` is marked
> `SUPERSEDED_PRE_RUN_CANDIDATE` and must never be used for scientific training.**
> It carried previous applied commands inside the proprioceptive tensor and
> z-scored projected gravity. The supersession is recorded in
> `SUPERSEDED_CANDIDATE_CONFIGURATIONS` and asserted by test.

---

# 4. The single matrix driver

`scripts/run_dev_proprio_factorial_driver_v1.py` — one driver, four cells,
differing only in `use_proprio` and `rollout`.

**Pre-registered ten-seed registry** (fixed before seed 1; the capped pilot
decides how many are *used*, never which or what they are):

```
2026080901  2026080902  2026080903  2026080904  2026080905
2026080906  2026080907  2026080908  2026080909  2026080910
```

**Balanced cell-order rotation** — each cell occupies each of the four serial
positions equally often across the ten seeds (verified by test):

| seed index | order |
|---:|---|
| 0 | rgb_one_step, rgb_rollout, proprio_one_step, proprio_rollout |
| 1 | rgb_rollout, proprio_one_step, proprio_rollout, rgb_one_step |
| 2 | proprio_one_step, proprio_rollout, rgb_one_step, rgb_rollout |
| 3 | proprio_rollout, rgb_one_step, rgb_rollout, proprio_one_step |
| 4–9 | the same 4-cycle repeats |

**Per quadruplet:** one shared base-weight artefact carrying a content digest;
every cell loads it, the digest is re-checked (a corrupted artefact is a
predeclared technical invalidity), and every shared parameter is verified
bit-identical before a single optimisation step. Modality parameters come from a
separate keyed stream at `seed + 7919`.

**Stochastic-operation audit.** All randomness comes from **named stateless
generators** keyed by `(seed, purpose, epoch)` — never the global stream — so a
rollout cell's extra predictor call and a proprio cell's extra modules cannot
shift another cell's batches or masks. This is tested directly: building a
proprio model, running a 4-step rollout and churning the global stream leaves the
batch plan identical. **Dropout is asserted disabled** (70 modules checked); an
injected dropout module makes the assertion fail.

**Budget:** exactly 24 epochs, fixed **epoch-21** checkpoint, no metric-based
selection, no exclusion for being poor or trending. Technical invalidity is
limited to: hash/manifest mismatch, NaN/infinite values, incomplete training from
infrastructure failure, corrupted checkpoint, implementation failure.

**Launch guard:** invoking the driver with `--seed-index` exits non-zero with
*"training is not authorised"*. Only `--dry-run` proceeds. This is tested.

---

# 5. Evaluation harness and exact metric aggregation

`scripts/eval_dev_proprio_factorial_v1.py`. All cells share the hash-verified
**478-row** selection manifest, identical target encodings, identical frozen
masks and one shuffled-action assignment (`DERANGEMENT_SEED = 11`).

## Primary estimator — the only one used for the interaction

1. **within a row** — mean cosine over valid (changed-token) positions;
2. **within an episode cluster** — mean over that episode's rows;
3. **within a family** — mean over that family's episode scores;
4. **across families** — the **unweighted** mean of the eight family scores.

```
I_s = (PropRoll_s − PropOne_s) − (RGBRoll_s − RGBOne_s)
```

Verified on a hand-computed fixture: rows `[1,3]`→episode 2.0, `[10]`→10.0,
family A = 6.0; family B = 5.0; equal-family = **5.5**, whereas a token-pooled
mean would be 4.833 — materially different, so the two schemes cannot be confused.
A family absent from the split raises rather than silently averaging seven.

Corpus-weighted / token-pooled values are computed and reported in a **separate
secondary block**; the interaction is never formed from them.

Also reported: H=1–4 correct-future cosine, correct-versus-shuffled margin,
occupied spatial metrics, per-family values, terminal-window mean/dispersion/slope
(flagged `used_for_selection: false`, `used_for_exclusion: false`), and the
prospectively declared `local_composite_motifs` diagnostic.

**Co-outcomes.** The frozen configuration declares `non_inferiority_margins:
None`, so occupied spatial metrics and the shuffled-action margin are reported as
**mandatory co-outcomes** and the harness sets
`formal_non_regression_claimable: false`. No formal non-regression claim may be
made from them, and the code enforces that rather than relying on the write-up.

---

# 6. Capped seed re-estimation

`scripts/dev_seed_reestimation_v1.py`.

`interim(values)` accepts exactly five interaction values and returns **only**
the sample sd, its bound, the power curve and `n_final`. It computes no mean,
sign, interval or family breakdown, and a test asserts none appears in the record.
A second test shifts every interaction by +0.5 and confirms `n_final` and the sd
are unchanged — the decision cannot depend on the observed effect.

```
sigma_U = s_I * sqrt(4 / chi2_{0.10, 4})            90 % one-sided upper bound
power   = exact noncentral-t, one-sample, df = n-1, ncp = sqrt(n)*delta/sigma_U
n*      = smallest n in [5, 10] with power >= 0.80 at delta = 0.005, alpha = 0.05
```

Verified against `scipy.stats.nct` directly, and power confirmed monotone in n.
Small sd → `n = 5`; large sd → `n = 10` with `precision_limited: true`.

`final()` is a separate function and reports all individual `I_s` values, the
mean, the sample sd, the 95 % t-interval, the final seed count and the
re-estimation decision, with the replication unit recorded as the training seed
quadruplet and episode bootstrapping explicitly noted as not replacing it.

---

# 7. Test results

**84 passed**, across four files, in the CPU-torch venv
(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, the ROS `launch_testing` plugin otherwise
breaks collection).

| file | tests |
|---|---:|
| `test_proprio_action_contract.py` | 44 |
| `test_proprio_factorial_driver.py` | 26 |
| `test_horizon_sequence_frame_action_mismatch.py` | 8 |
| `test_planning_refactor.py` | 6 |

Requested coverage: shared-weight identity; seed and RNG-stream isolation;
identical batch/augmentation plans; fixed epoch-21 checkpoint; deterministic
resume; cell-order independence; exact primary estimator; interim suppression of
comparative means; capped sample size; failure on lateral commands and on
manifest/hash mismatch — plus gravity norm/range/finiteness, training-only
statistics, frozen-statistics reuse, control-history presence in every cell, and
drop-count reporting.

One defect the tests caught: the shared-weight bit-identity check compares the
model against the artefact it loaded, so both move together and a **corrupted
artefact was undetectable**. A content digest is now written with the artefact and
re-verified on every load.

---

# 8. Dry-run artefacts

`factorial_v1/dry_run.json`, `factorial_v1/seed_registry.json`, and base-weight
artefacts for seeds 1–2. For both seeds:

| check | result |
|---|---|
| shared parameters bit-identical across all four cells | **true** |
| batch plan identical across cells | **true** |
| proprio parameters identical within a seed | **true** |
| control-history parameters present in the RGB cells | **true** |
| dropout | disabled, asserted, 70 modules checked |

No cell was trained; `"trained": false` is recorded.

---

# 9. Remaining blockers

1. **The driver's training loop has never executed on real cached features.** The
   feature-loading path is written against the existing cache layout
   (`frozen_train_ctx0/1.f16`, `frozen_current.f16`, `frozen_train_future.f16`,
   `frozen_train_step2.f16`) but is reached only from the code path the launch
   guard blocks, so it is unexercised. Those caches are indexed over the original
   4,566-row set; the 4,444-row manifest needs an index map by `pair_sha256`, and
   **that map is not yet built**. This is the one piece standing between the
   current state and a runnable seed 1.
2. **No selection-split feature cache exists for the corrected manifest.** The
   evaluator's `main()` deliberately exits; its estimator is exercised only on
   fixtures. Wiring it to real tensors needs the same index map.
3. **Device-rotation policy is declared but untested** — the dry run is
   single-device. If cells are ever spread across GPUs, the rotation needs a real
   multi-device check first.

None of these blocks the current deliverable; all three block launching seed 1.

---

## Stopping condition

Nothing is running. The four-cell experiment has not been launched and awaits
approval.
