# Proprioceptive implementation: code, validation and frozen configuration

Date: 2026-08-09
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** No cell of the factorial has been
trained. Only deterministic shape, fixture, invariant and overfit smoke tests were
run. No corpus was regenerated, no image re-rendered, no frozen result modified,
and the sealed benchmark was not inspected.

The system built here is a **reference-informed quadruped adaptation**, not an
official upstream configuration. No upstream config combines a V-JEPA 2.1 encoder
with an action-conditioned predictor; the context stays a fixed sliding
three-frame window, positional treatment stays learned-absolute, and the target
stays a single endpoint frame.

---

# 1. Code diff

New files only. `run_dev_v03_temporal_action_jepa_v1.Predictor` and every frozen
artifact are untouched, so the frozen H=1–4 result remains exactly reproducible.

```
 lewm/tests/test_proprio_action_contract.py          | 424 ++++++++++++++
 scripts/build_dev_v03_proprio_action_manifest_v1.py | 374 +++++++++++
 scripts/dev_action_slew_reconstruction_v1.py        | 255 ++++++++
 scripts/dev_proprio_experiment_config_v1.py         | 202 +++++++
 scripts/dev_proprio_predictor_v1.py                 | 260 ++++++++
 5 files changed, 1515 insertions(+)
```

| file | role |
|---|---|
| `dev_action_slew_reconstruction_v1.py` | the deterministic post-slew reconstruction + corpus validator |
| `build_dev_v03_proprio_action_manifest_v1.py` | builds and verifies the corrected-action + proprioception manifest |
| `dev_proprio_predictor_v1.py` | predictor with the 10-D action and optional proprio tokens; the rollout path |
| `dev_proprio_experiment_config_v1.py` | the frozen factorial configuration and the pilot sample-size function |
| `test_proprio_action_contract.py` | 29 leakage / alignment / contract / smoke tests |

---

# 2. Action-reconstruction validation

## 2.1 The identified limiter

```
applied[k] = prev + clip(requested[k] - prev, -rate, +rate)
prev       <- applied[k]                       # carried across block boundaries
prev       <- (0, 0, 0)                        # at every respawn
rate       =  vx 0.25 m/s,  yaw 0.35 rad/s     per 0.1 s tick
```

Inputs: the requested primitive and the previous **applied command**. Measured body
motion is never consulted, so the identical pure function serves hypothetical
planning actions — `reconstruct_block(primitive, previous_applied)`.

## 2.2 Corpus validation — exact

116 scenes across all splits and all eight families:

| quantity | value |
|---|---:|
| blocks compared | **1,113,600** |
| ticks compared | **5,568,000** |
| **block-exact** | **1,113,600 / 1,113,600 = 100.0000 %** |
| **tick-exact** | **5,568,000 / 5,568,000 = 100.0000 %** |
| sign-reversal blocks, exact | **38,488 / 38,488 = 100 %** |
| first block after a reset, exact | **7,481 / 7,481 = 100 %** |
| clipped blocks, exact | **397,198 / 397,198 = 100 %** |
| reset restarts exercised | 7,716 |

Record: `proprio_v1/action_reconstruction_validation.json`.

## 2.3 How the reset case was found, and why it mattered

The first reconstruction carried the applied command across the whole run and
reached 99.914 %. The residual 0.086 % concentrated in the first ~6 blocks of each
episode (i=1: 2.95 %, i=2: 4.25 %, decaying to 0.028 %), which I initially read as
a controller startup transient. It was not. `episode_step` **restarts at a
respawn**, and many envs respawn a few steps after spawn — so those were
post-reset blocks where the limiter had returned to a standing command while my
carry did not.

Two consequences, both now fixed and both worth recording because either would
have silently corrupted the action channel:

1. **Keying**: `(env, episode_step)` is *not* unique within a scene. Keying on the
   global step derived from `frame_index` is. The first manifest build, keyed on
   `episode_step`, silently merged two episodes and reported 10.4 % action
   mismatch; that build was discarded.
2. **Reset semantics**: the limiter restarts from `(0,0,0)` at each respawn.

With both corrected the reconstruction is exact everywhere, including the reset
and sign-reversal cases the brief called out. A sign reversal is real: requested
`+0.45`, applied `−0.10` on the first tick while the controller decelerates out of
the previous command.

## 2.4 The constant channel

`vy` is identically zero in every requested and executed block of every scene, and
none of the nine primitives commands lateral motion. Under the stated exclusion of
constant fields it is dropped from the model-facing action (15 → **10 dims**) and
from the previous-applied-command proprioceptive channel (33 → **32 dims**). The
limiter arithmetic retains all three channels, so a future corpus with lateral
commands needs only `ACTIVE_CHANNELS` widened.

---

# 3. Proprioceptive tensor contract

| | |
|---|---|
| shape per row | `(3 slots, 5 samples, 32 channels)` |
| window | **trailing**: steps `[s−14 … s]`, three slots tiling contiguously |
| alignment | the newest sample of each slot **is** that slot's observation step |
| rate | 10 Hz, verified gapless (all inter-sample deltas exactly 100 ms) |
| entry | one token per slot, predictor only; encoder untouched |
| target | **visual only** — no proprioceptive target, no auxiliary loss |

| channels | dims | note |
|---|---:|---|
| projected gravity | 3 | from roll/pitch only; yaw cancels exactly, verified unit-norm |
| body angular velocity | 3 | gyro, body frame |
| joint positions | 12 | manifest order, with a verified bijection to Unitree SDK order |
| joint velocities | 12 | as above |
| previous applied command | 2 | `applied[k−1]` on (vx, yaw); strictly historical |
| **total** | **32** | |

Excluded: lateral command `vy` (constant), body linear velocity (simulator ground
truth — **no synthetic noise model is used**), absolute yaw, world pose, `/odom`,
camera extrinsics, foot contacts, joint effort, IMU linear acceleration.

## Manifest

| | |
|---|---|
| rows kept | **4,444** of 4,566 — **3,966 train / 478 selection** |
| dropped | 90 `proprio_history_absent` (episode start), 32 `proprio_history_crosses_reset` |
| dropped for action verification | **0** |
| blocks verified in-build | **768,000 / 768,000** |
| action blocks per row | 4 for 4,378 rows; 3 for 26; 2 for 40 |
| `proprio_rows.jsonl` sha256 | `c825f72aee4e346f93fcf7674cd0be6d0c89fa6725dbbe7baf6f576da6d53f2e` |
| normalisation sha256 | `67a7f5712f291aaf4df8b1d9101da7f4481f2068f63ae5ce5a29013082fc795f` |

Rows are only ever dropped, never added, and the identical row set is used by all
four cells. Normalisation statistics come from the **train split only** (59,490
samples) and are frozen and hashed.

## Rollout handling

The proprio window slides exactly as the image window does: the oldest slot is
dropped, the appended slot holds a *prediction* and is marked invalid. Observed
slots therefore fall **3 → 2 → 1 → 0** across H=1…4. The tensor only ever loses
entries, so no future proprioceptive value needs to exist and none can be read.

---

# 4. Test results — 29/29, plus 15 pre-existing = 44 passed

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 …/genesis_render_vulkan/bin/python -m pytest`

| requirement | test | result |
|---|---|:--:|
| randomising/deleting future proprioception leaves predictions unchanged | `test_injected_future_proprioception_is_inert` | pass |
| no horizon input has proprio timestamps later than the observation | `test_no_proprio_timestamp_later_than_its_observation` | pass |
| observed-proprioception perturbations **can** affect predictions | `test_observed_proprioception_can_affect_predictions` | pass |
| future-proprioception perturbations **cannot** | `test_invalid_slot_content_cannot_affect_predictions` | pass |
| reset boundaries do not mix proprioceptive history | `test_reset_boundaries_do_not_mix_proprioceptive_history` | pass |
| image / action tick / proprio sample temporal alignment | `test_image_action_and_proprio_share_one_temporal_origin`, `test_action_block_alignment_to_the_transition`, `test_proprio_history_is_contiguous_and_trailing`, `test_proprio_samples_are_10hz_and_gapless` | pass |
| reconstructed post-slew trace matches logged values | `test_manifest_records_verification_against_logged_values`, `test_slew_ramp_and_sign_reversal`, `test_reset_start_is_zero` | pass |

Also covering: seed pairing (shared weights bit-identical across cells),
proprio-parameter determinism, access counting (`3,2,1,0`), gradient reachability,
absence-token gradient, constant-channel exclusion, joint-permutation bijection,
projected-gravity unit norm and yaw-freedom, shape rejection, and overfit smoke for
both cells.

## Two implementation defects the tests caught

**AdaLN-Zero makes every block the identity at initialisation.** The first
perturbation tests passed *vacuously* — at step 0 nothing in the context, image or
proprioceptive, can reach the output. All perturbation tests now run after
discarded warm-up steps.

**Masking by multiplication is not inertness.** `0 × NaN = NaN`, so a masked slot
containing NaN or inf would still poison the output. The input is now hard-gated
with `torch.where` *before* the projection, which makes "invalid slot content is
inert" an exact bitwise identity rather than an approximate one.

## A correction to my own test design, worth stating

My first formulation asserted that deleting proprioception leaves **H=4**
unchanged. That was wrong, and it is precisely the error the brief warned about:
observed proprioception legitimately changes H=1–3, and those predictions are the
context of H=4, so H=4 moves for entirely sound reasons. The tests were
reformulated to decide leakage from **access and inertness** only —
`test_injected_future_proprioception_is_inert` injects values into predicted slots
during the real rollout and requires all four horizons bit-identical — and a
companion test now asserts the opposite direction, that an observed perturbation
*should* propagate to H=4, so no later reader mistakes propagation for leakage.

---

# 5. Parameter counts

| model | parameters | delta |
|---|---:|---:|
| frozen predictor (12-D primitive action) | 17,198,080 | — |
| **corrected-action RGB cell** | **17,197,312** | **−768** vs frozen |
| **corrected-action + proprioception cell** | **17,210,752** | **+13,440** vs the RGB cell |

The −768 is the action head narrowing from 12 to 10 inputs (2 × 384). The +13,440
is the entire proprioceptive path: `proprio_in.weight` 12,288, `proprio_in.bias`
384, `proprio_modality` 384, `proprio_absent` 384 — **0.078 %** of the model. The
two cells differ by this and nothing else; all other parameters are bit-identical
at a shared seed, verified by test.

---

# 6. Frozen experiment configuration

`scripts/dev_proprio_experiment_config_v1.py`, sha256
**`f410df7989fd639761b7177c00cc6d12fb9db15a1a6c46d9898a4c7bd6f7e0c8`**, persisted
to `proprio_v1/frozen_experiment_config.json`.

**Cells** — `rgb_one_step`, `rgb_rollout`, `proprio_one_step`, `proprio_rollout`;
objectives `e1` and `1.5·e1 + 0.5·e2`; the corrected 10-D action in all four.

**Pairing** — shared initialisation from `torch.manual_seed(seed)` before any
shared parameter; proprio parameters from a separate generator at `seed + 7919`;
one data-order generator seeded from the seed giving identical row order in all
cells; no train-time augmentation; no dropout or stochastic depth.

**Budget and checkpoint** — fixed 24 epochs, no extension; **evaluate the fixed
epoch-21 checkpoint for every technically valid run**; no per-run best-epoch
selection. Terminal-window (19–23) mean, sd and the OLS slope over epochs 14–23
are reported as stability diagnostics; **no run is excluded for improving or
deteriorating**.

**Endpoints** — primary: equal-family H=2 correct-future cosine. Principal
estimand: `I_s = (PropRoll_s − PropOne_s) − (RGBRoll_s − RGBOne_s)`.
Corpus-weighted results secondary. H=3 = beyond-trained-horizon transfer,
H=4 = longer-horizon diagnostic. **No combined H=2–3 endpoint.** Mandatory
non-regression outcomes: occupied spatial information and correct-versus-shuffled
action margin — a cosine gain with material loss of either is not an unqualified
success.

**Seed design** — capped internal pilot. Stage 1 = 5 paired quadruplets (20 runs).
Before any comparative mean, sign, interval or family plot, compute **only** the
seed-level sd of the H=2 interaction, take a conservative upper confidence bound,
and size for a 0.005 interaction at α=0.05 / 80 % power. Bounded to [5, 10];
escalation on the observed mean, direction or significance is forbidden; if the
requirement exceeds 10, complete 10 and label the study **precision-limited**.
Replication unit = the **training seed quadruplet**; episode bootstrap quantifies
within-seed evaluation uncertainty only and may not substitute for it.

Sample size implied by the pilot (one-sample paired, upper-bounded sd):

| observed sd of `I_s` | 80 % upper bound | n required | n used |
|---:|---:|---:|---:|
| 0.001 | 0.0016 | 1 | 5 (floor) |
| 0.002 | 0.0031 | 4 | 5 (floor) |
| 0.003 | 0.0047 | 7 | 7 |
| 0.004 | 0.0062 | 13 | 10, **precision-limited** |
| 0.005 | 0.0078 | 20 | 10, **precision-limited** |

**Family diagnostic** — `local_composite_motifs` is prospectively declared as a
family-level diagnostic, not a primary endpoint; tuning to it is prohibited;
equal-family reporting is preserved so corpus weighting cannot hide it.

---

# 7. What is not built, and open items

Not implemented, by scope decision: growing context, RoPE, all-frame target
supervision, proprioceptive prediction target, auxiliary proprioceptive loss,
encoder movement, counterfactual branch corpus.

Not yet written (needed before the first scientific run): the four-cell training
driver that consumes this manifest, and the evaluation harness that computes the
equal-family H=2 endpoint plus the two non-regression outcomes per seed. Both are
mechanical given the existing rollout evaluator, but neither exists yet.

Open items I did not decide unilaterally:

1. **`projected_gravity[2]` has sd 0.0012** about a mean of −0.9988. It is not
   constant, so it stays, but z-scoring amplifies it ~800×. If you would rather it
   were dropped or left unscaled, that is a config change before the first run.
2. **The 10-D action has no lateral channel at all**, so any future planner that
   wants strafing needs new data, not just new code.
3. **122 of 4,566 rows were dropped** (2.7 %), all at episode starts or reset
   crossings. The row set is identical across cells, so the factorial is unaffected,
   but the manifest is no longer row-identical to the frozen RGB-only study.

---

## Stopping condition

Implementation, manifests, validation and tests are complete. **The four-cell
training experiment has not been launched** and awaits approval.
