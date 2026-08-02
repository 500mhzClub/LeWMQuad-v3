# Go2 bounded WM-A branch experiment V1

Date: 2026-08-02
Status: **parity source correction under re-review; first invocation did not consume the attempt; no bounded execution authority**

This document is not execution authority, a scientific result, a retry or
resume grant, a held-out-access grant, or a deployment grant. The first parity
supervisor invocation stopped before root creation because its pre-reservation
chain rehash incorrectly required the deliberately absent fresh root to be an
existing reserved directory. The one-shot attempt therefore remains
unconsumed. The minimal correction is not executable until it is committed,
independently re-reviewed, and bound by a new immutable authority. No bounded
branch experiment or model-panel evaluation has been launched.

## Current disposition

The original parity source, source review, plan, and execution authority were
frozen in commits `7d5b6c5`, `406658e`, and `f0a9405`. The authorised command
failed in about 2.2 seconds with `VisualDomainParitySupervisionError: parity
output root is not the canonical reserved directory`. Control never reached
the fresh-root check or atomic root creation: no reservation, RGB, depth,
physics, model evaluation, or output tree was produced, and the root remained
absent. This is a pre-consumption source defect, not a consumed parity result
and not a retry opportunity. The old authority is no longer reusable because
the supervisor and its bound runtime test must change. The unchanged exact
plan may be rebound by a successor authority after the corrected source closure
passes independent review. The immutable incident record is
`docs/lewm_go2_world_model_visual_domain_parity_v1_preconsumption_launch_failure_and_authority_supersession_2026-08-02.json`
(3,930 bytes, SHA-256 `3e82a468222b51212d6f427982abd553295899e9ac63728caa248a0220994656`).

The original 160-branch calibration consumed its one attempt and failed on a
valid low-information near-wall observation. That failure did not show render
corruption and did not justify screening near-wall or low-texture navigation
states. Calibration V1 remains immutable and cannot be retried or resumed.

Calibration V2 consumed its one attempt and failed after producing physically
distinct branches whose 13 local RGB frames were byte-identical 678-byte
uniform-gray PNGs. Its terminal is 3,766 bytes (`292f6e...`) and physics
receipt is 26,094 bytes (`f00e8f...`). It grants no retry, resume, plan, or
authority.

The failure exposed a more fundamental input-domain error. Historical training
RGB under `.generated/datagen_full/render_textured_v03` used
`scripts/render_replay_v03.py`: native/stored 224×224, raw `fov_deg=78.323`
passed directly as Genesis `yfov`, the historical `build_scene`/texture
helpers/default options, nominal platform camera mount, walls/obstacles/
landmarks, and no manifest lighting, distractors, or extrinsic jitter. The
failed collector used native 640×480, converted horizontal FOV to about
62.837° vertical FOV, then downsampled to 224 square. Adding textures alone is
not domain parity.

Before a bounded plan can exist, a successor calibration must pass under the
exact historical textured-v03 contract and independently reviewed
implementation equivalence must bind renderer, texture helper, all selected
texture leaves, every derived mesh, collector/helper source, source RGB, and
two independently rendered candidate RGB leaves for identical
scene/base-pose/camera identities. This is a deterministic qualification gate,
not a statistical sample: exactly one ordinary TRAIN scene per family and four
pre-bound poses per scene produce 32 reference/candidate/duplicate triples.
All 32 reference/candidate pairs and all 32 candidate/duplicate pairs must be
pixel-exact, with no missing or extra row. SSIM and normalized L1 are retained
only as diagnostics; no confidence interval or relaxed image threshold can
pass the gate. Missing or failed evidence is `STOP_NO_GENERATION_AUTHORITY`; a
same-sensor flag or distribution-only test is insufficient.

The progression comparison and offline analysis completed. The result is
122,056 bytes,
`f4981f210e54a6a6b26a06b14eb8a543a78ff855ab61266506682d986a687720`;
the analysis is 32,958 bytes,
`45c876c2c0156f4788cc9862060a0025db892ceb16692a28e6041c3245b2e19c`.
Its decision is `DELTA_PROXY_NOT_MEANINGFUL`. A fresh 90-minute run merely to
add receipts to `result.json` is not required. The frozen analyzer hashes all
12 fixed terminal snapshots without
deserializing tensor payloads. The bounded consumer reopens `analysis.json`,
reopens its exact `input_result`, reruns the analyzer, and independently
rehashes the pack, predecessor, provenance indices, six training sources, and
all 12 `terminal_snapshot_bindings`. Any mismatch fails closed.

The frozen progression source identities used by that analyzer are:

- runner: 44,968 bytes,
  `0cb15c6414d7deeda6c206981457c72a45558905ea695cdae924a844702d49e0`;
- analyzer: 18,632 bytes,
  `37fb6a306bc2e1370581e968094cdf9453a0e9df2d8814079dc2a92368ca6f31`.

## Exact claim-bearing experiment

The causal branch experiment is fixed at the preregistered scale. Calibration
may stop authority because the projection is unsafe; it may not silently
shrink this design.

| Property | Exact value |
|---|---:|
| scene roles | train and eval |
| families per role | 8 |
| scenes per family per role | 2 |
| total scenes | 32 |
| states per scene | 8 |
| train states | 128 |
| eval states | 128 |
| total states | 256 |
| requested actions per state | 9 |
| physical candidate branches | 2,304 |
| sentinel branches | 0 |
| context RGB frames | 768 |
| target RGB frames | 2,304 |
| total stored RGB frames | 3,072 |

Each state uses one of eight fixed two-action histories. Every candidate then
branches from the exact synchronized state using one requested `(5,3)` action
block. The model receives only `requested_action_id`; the future executed
command tape remains an outcome/audit field and is never candidate input.

### Deterministic scene selection

The panel is not caller-picked. The scene-panel builder enumerates every
direct ordinary `corpus.json` campaign under `.generated/scene_corpus`, keeps
ordinary `train` scenes, deduplicates scene IDs by a fixed hash rule, and
removes the union of:

- calibration V2 scene IDs;
- progression train and validation scene IDs; and
- predecessor train, validation, and place checkpoint-selection scene IDs
  reopened from the fixed terminal provenance receipt, exact place manifest,
  and all three exact JSONL indices.

It then takes the lowest four fixed SHA-256 ranks in each family. A separate
fixed role hash assigns two to readout training and two to evaluation. The
plan builder independently rederives the complete panel and rejects a
caller-substituted scene, role, history, target, or corpus inventory. Targets
come from each exact ordinary scene manifest. The scene-panel binding, complete
corpus bindings/inventory, selection contract, exclusion hash, selected rows,
and deterministic per-scene floor/wall/obstacle texture-leaf bindings travel
through plan, authority, generation terminal, and evaluation identity.
Near-wall or low-texture content is not an exclusion criterion.

### Hard visual-domain prerequisite

The gate and authority carry one exact `visual_domain_parity_freeze`. It binds
the parity result, the consumed successful supervisor terminal, the independent
review, historical source-RGB reference, candidate pixel
panel, `render_replay_v03.py`, `textures.py`, the candidate collector, the
fixed parity evaluator, all evidence scene IDs and poses, raw-pixel hashes,
historical summary/plan/frames/Genesis lineage, complete corpus-selection
bindings, the selected floor/wall/obstacle texture leaves, and every derived
structural mesh. The dedicated source plan deterministically selects one
complete historical scene per family by fixed SHA rank and the first four
bound frame records. It re-derives both base-pose encodings, the nominal camera
pose, the frame-record hash, and the historical RGB raw-pixel hash before any
mutable output reservation. The candidate must invoke the shared exact
historical-pose RGB-only helper twice independently for each pose: exactly 64
RGB calls, zero depth calls, and zero physics steps. The plan independently
rejects any native/stored resolution other than 224×224, any
horizontal-to-vertical FOV conversion, downsampling, scene-extrinsic jitter,
lighting/distractor substitution, or geometry/texture contract other than the
historical textured-v03 path. The old 640×480/downsample renderer is ineligible
even if a document labels it `PASS`.

## Successor-calibration-derived execution caps

Only measurements from a future successful, exact textured-v03 successor
calibration may be copied into the gate and used to derive explicit authority
caps. Failed V1/V2 measurements cannot mint authority:

- minimum wall cap: `max(3,600 s, 20 × calibration wall seconds)`;
- hard wall cap: `28,800 s`;
- stored-RGB cap: `max(512 MiB, 20 × calibration stored PNG bytes)`;
- hard stored-RGB cap: `2 GiB`;
- selected-device VRAM cap: calibration global baseline plus `4.5 ×` its
  measured peak delta; and
- VRAM hard cap: 95% of the exact measured device total.

The authority also exposes the exact render and branch counts: 32 scenes, 256
states, 2,304 branches, 3,072 render calls/stored frames, 172,800 lane policy
steps, and 1,728,000 lane physics steps. The authority builder stops if any
calibrated projection exceeds a hard cap. It does not downsample, drop scenes,
reduce actions, or silently substitute a smaller experiment.

The successor calibration keeps deterministic repeatability and physical
outcome equivalence as separate claims. All 16 repeat controls must replay
executed command tapes, physical trajectories, and stored RGB exactly; the
`1e-6 m` value is only the numerical floor recorded for that exact-repeat
diagnostic. Candidate physical-outcome classes use fixed preregistered `0.01 m`
progress and path-length rounding bins. Those bins have boundary artifacts and
must not be described as pairwise-distance-`<=1 cm` equivalence.

Calibration must contain exactly two states from each of the eight families.
For requested-action query `a`, let its alternatives be every action whose
fixed `0.01 m` physical-outcome class differs from `a`. Query `a` is jointly
eligible only when that set is nonempty and **every** such alternative differs
from `a` in both executed-tape class and decoded stored-RGB raw-pixel class.
The calibration must contain at least 72 eligible queries out of 144 overall
and at least nine out of 18 in every family, yielding minimum coverage `0.5`
both overall and per family. Per-state executed-tape, physical-class, and RGB
unique-count histograms remain useful collapse diagnostics, but their old
aggregate nontrivial-state rule is not a gate. The separate sum of
physical-outcome class counts divided by 144 is also diagnostic only. Failure
is `STOP_INSUFFICIENT_JOINT_COUNTERFACTUAL_DISCRIMINATION_SUPPORT`, not
authority to tune, replace, or refill the calibration panel.

Raw-pixel class identity is never accepted from a receipt declaration alone.
The calibration analyzer invokes the checker's one-file-descriptor,
no-symlink PNG reader for every textured frame and requires an exact
single-frame 224×224 RGB decode whose contiguous C-order bytes reproduce the
declared pixel SHA-256. Declaration tampering and a changed PNG with a rebound
file hash therefore fail before a calibration receipt can freeze the contract.

Disk capacity is checked before reservation, without treating 30 GiB as a
required residual floor. The complete parity + calibration + bounded sequence
contains 3,296 stored RGB frames (64 + 160 + 3,072). Across the 32 fixed
historical parity references, the largest exact 224×224 PNG is 51,387 bytes,
which projects 169,371,552 bytes for all 3,296 frames; even uncompressed
224×224×3 RGB is 496,140,288 bytes. The bounded stored-PNG hard ceiling remains
2 GiB, and the combined projected-new-output ceiling plus explicit safety
margin must not exceed 4 GiB. The measured 31,470,034,944 available bytes at
source preflight clears that budget. This is a capacity calculation, not
permission to delete, relocate, or interrupt existing experiment output.

The external supervisor alone atomically creates the fresh attempt root after
preflight. Root creation is the irreversible attempt-consumption event, and
the subsequent nonce/PID-bound reservation records that already-consumed
attempt. A crash after root
creation therefore consumes the attempt; an existing attempt root is terminal
and can never be interpreted as permission to resume. The collector must be
the supervisor's direct child and consume that exact nonce/PID-bound
reservation. Stored RGB bytes are checked incrementally. A 20 ms global VRAM
monitor is active while the collector runs and terminates it on cap violation
or counter failure. Any collection, receipt, cap, or render failure is
terminal: no refill, overwrite, retry, or resume path exists.

Before the bounded joiner can emit the frozen manifest, it repeats that exact
decoded-pixel verification for every bounded frame and for the bound
calibration collection. The evaluator repeats it once more while reading each
model input. Thus the joint tape/physical/RGB eligibility signatures are
grounded in decoded leaves, not merely mutually consistent JSON receipts.

## Preregistered evaluation

Latent normalization uses only the 128 train states through one fixed
`masked_plain/seed_2026080201` reference checkpoint shared by all 12 members;
the aggregate rejects differing standardizer identities. Physical readout
fitting uses only the 128 train states. Outcome equivalence is never fitted in
learned latent space: it is the frozen physical-oracle dense-rank class under
the preregistered `0.01 m` progress/path bins. These are fixed rounding bins
with boundary artifacts, not a claim of literal pairwise `<= 0.01 m`
equivalence. Every generalization measurement uses only the 128 evaluation
states. Pilot train/eval scenes are disjoint from progression and all
predecessor train/validation/place observational scenes.

The fixed model panel is the Cartesian product of:

- arms: `masked_plain`, `masked_delta`, `full_plain`, `full_delta`; and
- seeds: `2026080201`, `2026080202`, `2026080203`.

These are the exact 12 update-700 snapshot bindings in progression
`analysis.json`. Branch results cannot select a model or checkpoint. The
one-shot panel runner reserves one fixed output root before opening a model or
RGB leaf, writes all 12 immutable measurement reports, and calls the aggregate
only after rereading all 12 reports. A failed or partial panel is terminal and
cannot be aggregated.

### Direct branch fidelity and outcome equivalence

The true-future descriptor is the target encoder's mean/std descriptor over
four fixed masked-token rows, standardized by the one fixed train-only
reference above. Exact requested-action matched error remains exact even when
two actions share a physical outcome class. Equivalence-aware margin and
retrieval use only equality of the frozen physical oracle dense rank; latent
distance cannot define, merge, or split an equivalence class.

Margin and retrieval do not falsely penalize two requested actions whose
physical oracle outcomes are equivalent: they compare the nearest
equivalent future with the nearest non-equivalent future and score retrieval
against the equivalence-adjusted chance rate. The report persists each
action-ordered executed-tape, physical-class, and decoded stored-RGB raw-pixel
signature and recomputes the same universal joint eligibility rule used by
calibration. Margin and retrieval are scored only for eligible queries. The
eligible action IDs and signatures must be identical across forecast,
shuffled, and HOLD-blind controls and across all 12 fixed checkpoints. If
eligible-query coverage is below `0.25` overall, below `0.25` within any fixed
family, or either of the two fixed evaluation scenes in any family has no
eligible query, the checkpoint is `CHECKPOINT_MEASUREMENT_INCONCLUSIVE_DATA`;
discrimination gates cannot be silently waived.

Controls are action-shuffled forecasts and HOLD-blind forecasts. Future
executed commands are never model inputs.

### Physical planning utility

Nine independent train-only ridge heads map each arm's features to normalized
dense physical rank regret. Reports retain forecast, current-state/action,
task/action-only, HOLD-blind, action-shuffled, true-future-ceiling, and
random-expected controls. Primary physical measurements are:

- normalized dense physical rank regret (lower is better);
- `fell OR tipped` selected-action rate; and
- physical target progress in metres.

### Uncertainty

Every interval is paired by state and clustered by whole scene within the
fixed family stratum. Bootstrap resampling preserves equal weights for all
eight families, uses 10,000 resamples, and seed `20260802`. It therefore cannot
let a family with more rows dominate the estimate. With only two evaluation
scenes per family and three fixed training seeds, this remains bounded
development evidence, not deployment or population uncertainty.

### Per-checkpoint gates

Each member emits `CHECKPOINT_MEASUREMENT_PASSES_PREREGISTERED_GATES`,
`CHECKPOINT_MEASUREMENT_FAILS_PREREGISTERED_GATES`, or the explicit
`CHECKPOINT_MEASUREMENT_INCONCLUSIVE_DATA`. It cannot emit a global usefulness
verdict. The two plain candidate arms each use two-sided alpha 0.025, giving a
Bonferroni-controlled one-sided family alpha 0.025 for the arm-agnostic
"either plain arm" decision. Delta controls use ordinary 95% descriptive
intervals and cannot establish usefulness.

| Gate | Requirement |
|---|---|
| evaluator sensitivity | ceiling rank-regret reduction vs current: `upper <= -0.05` |
| direct matched error | forecast reduction vs shuffled: `upper <= -0.02` |
| direct non-equivalent margin | if applicable, forecast gain vs shuffled: `lower >= 0.02` |
| equivalence-aware retrieval | if applicable, advantage over adjusted chance: `lower >= 0.05` |
| physical rank regret | forecast reduction vs current: `upper <= -0.05` |
| safety noninferiority | unsafe-rate increase vs current: `upper <= 0.02` |
| absolute safety | forecast unsafe-rate upper bound `<= 0.05` |
| target progress | forecast gain vs current: `lower >= 0.01 m` |
| absolute target progress | forecast progress lower bound `>= 0.01 m` |
| falsification controls | forecast rank regret beats task/action-only, HOLD-blind, shuffled, and random (`upper < 0`) |

### Aggregate-only routing

The aggregate first recomputes every summary, paired comparison, gate, and
status from embedded per-state rows. Claimed pass strings or empty/fabricated
gate maps are rejected. It requires one generation/evaluation analysis/result/
12-checkpoint freeze, one standardizer, and identical model-independent
physical separability across all members.

There is no presumed `full_delta` primary. The confirmatory candidate family
is `masked_plain` and `full_plain`. Only the complete panel may emit
`USEFUL_SCENE_DISJOINT_PLANNING_EVIDENCE_DEVELOPMENT_ONLY`, and only if at
least one Bonferroni-controlled plain arm clears every absolute/relative gate
in all three seeds. If neither does, usefulness is not established; inadequate
physical separation routes to explicit inconclusive data. Delta arms remain
negative/mechanism controls under the frozen `DELTA_PROXY_NOT_MEANINGFUL`
decision and cannot substitute for a failed plain candidate, authorize
observational scaling, or become rollout eligible.

Secondary 2×2 mechanism effects use state-paired, equal-family scene intervals
on the shared direct-error surface. Delta and spatial main effects must favor
treatment in every seed with `upper-95 <= -0.02`. Full-grid noninferiority is
`FP−MP <= +0.02` and `FD−MD <= +0.02` in every seed; interaction is descriptive.
Neither practical mechanism routes to
`STOP_OBSERVATIONAL_MECHANISM_TUNING`, matched-branch training data, and
conventional/Dreamer baselines. Physical regret remains a primary per-arm
planning-utility gate, not an extra mechanism-treatment prerequisite.

One checkpoint, one selected seed, a partial panel, or a delta-only success can
never satisfy usefulness. Even a passing aggregate is not closed-loop
navigation evidence, held-out evidence, a safety certificate, checkpoint
promotion, G2 authority, or deployment evidence.

## Required sequence

1. Bind the completed fixed progression result/analysis identities above; do
   not launch a fresh run solely to embed snapshot receipts. Preserve
   `DELTA_PROXY_NOT_MEANINGFUL` as a binding negative route.
2. Build the dedicated exact 8-scene × 4-pose parity plan, independently
   review its complete source/runtime/texture/mesh/corpus closure, and issue a
   one-shot RGB-only authority. The supervisor must atomically create one fresh
   attempt root, thereby consume the attempt, then record its nonce/PID-bound
   reservation, render every pose twice, and emit leaf, generation, and
   candidate-panel receipts. Reopen every bound leaf and run the fixed
   evaluator; all 64 pixel comparisons must be exact. Independently
   review the consumed successful terminal and result before their exact
   result/terminal/review triple can enter `visual_domain_parity_freeze`.
3. Treat calibration V2 as consumed terminal failure. Only after the exact
   parity result, consumed successful terminal, and independent review are
   frozen, implement and independently review a new one-shot calibration under
   that bound historical textured-v03 render contract. Require exact replay as
   a separate technical gate, fixed `0.01 m` planning-outcome bins, at least
   72/144 jointly eligible queries overall and 9/18 in every family,
   `FREEZE_PILOT_CONTRACT`, successful supervision, and independent terminal
   review. Otherwise stop with
   `STOP_INSUFFICIENT_JOINT_COUNTERFACTUAL_DISCRIMINATION_SUPPORT`.
4. Reopen and rehash progression pack, predecessor, terminal provenance,
   source closure, result, analysis, and all 12 snapshots. Freeze their union
   of observational scene exclusions before branch data exists.
5. Derive the deterministic complete-corpus 32-scene panel and build the exact
   2,304-branch plan plus gate witness.
6. Freeze the source-and-review commit. Generate a non-passing review template;
   an independent reviewer must replace it with a passing review and no open
   findings.
7. An explicit authorizer may bind the exact plan, gate, review, model panel,
   calibrated resource caps, calibration collection, and exact parity
   result/terminal/review triple in one execution authority. Issuing this
   document does not consume the bounded attempt; only the supervisor's later
   atomic creation of the fresh attempt root does.
8. Invoke only the bounded external supervisor with exact caller-supplied
   authority SHA-256 and byte count. Independently review its terminal and
   frozen joined manifest before any evaluator reads RGB leaves.
9. Invoke the exact one-shot 12-member evaluation-panel runner. Independently
   review its terminal, member reports, and aggregate before citing the bounded
   development conclusion.

No step grants held-out, sealed, navigation, G2, production, promotion,
deployment, retry, resume, refill, or adaptive experiment-selection authority.

## Source and test inventory

New or modified bounded-pilot files in this source-only tranche:

- `scripts/build_go2_world_model_bounded_branch_scene_panel_v1.py`
- `scripts/build_go2_world_model_bounded_branch_experiment_plan_v1.py`
- `scripts/build_go2_world_model_bounded_branch_experiment_authority_v1.py`
- `scripts/collect_go2_world_model_bounded_branch_experiment_authorized_v1.py`
- `scripts/run_go2_world_model_bounded_branch_experiment_authorized_v1.py`
- `scripts/evaluate_go2_world_model_bounded_branch_experiment_v1.py`
- `scripts/run_go2_world_model_bounded_branch_evaluation_panel_v1.py`
- `scripts/build_go2_world_model_visual_domain_parity_plan_v1.py`
- `scripts/build_go2_world_model_visual_domain_parity_authority_v1.py`
- `scripts/run_go2_world_model_visual_domain_parity_authorized_v1.py`
- `scripts/evaluate_go2_world_model_visual_domain_parity_v1.py`
- `scripts/check_go2_world_model_counterfactual_pilot_v1.py`
- `scripts/analyze_go2_world_model_counterfactual_calibration_v1.py`
- `scripts/join_go2_world_model_counterfactual_pilot_v1.py`
- `lewm/datasets/go2_world_model_counterfactual_pilot_v1.py`
- `lewm/tests/test_go2_world_model_bounded_branch_experiment_v1.py`
- `lewm/tests/test_go2_world_model_bounded_branch_lineage_v1.py`
- `lewm/tests/test_go2_world_model_bounded_branch_runtime_boundary_v1.py`
- `lewm/tests/test_go2_world_model_bounded_branch_evaluation_panel_runner_v1.py`
- `lewm/tests/test_go2_world_model_visual_domain_parity_plan_v1.py`
- `lewm/tests/test_go2_world_model_visual_domain_parity_authorized_v1.py`
- `lewm/tests/test_go2_world_model_counterfactual_textured_v03.py`
- `lewm/tests/test_check_go2_world_model_counterfactual_pilot_v1.py`
- `lewm/tests/test_analyze_go2_world_model_counterfactual_calibration_v1.py`
- `lewm/tests/test_go2_world_model_counterfactual_consumers_v1.py`
- `lewm/tests/test_join_go2_world_model_counterfactual_pilot_v1.py`
- this runbook.

The progression runner and its six-file source closure are frozen inputs, not
changes in this tranche. The calibration analyzer, checker, joiner, and bound
dataset consumer are intentionally edited and must be rebound in the reviewed
source closure.

Focused verification after the final integration pass:

```text
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest -q \
  lewm/tests/test_go2_world_model_counterfactual_pilot_v1.py \
  lewm/tests/test_check_go2_world_model_counterfactual_pilot_v1.py \
  lewm/tests/test_analyze_go2_world_model_counterfactual_calibration_v1.py \
  lewm/tests/test_go2_world_model_counterfactual_consumers_v1.py \
  lewm/tests/test_join_go2_world_model_counterfactual_pilot_v1.py \
  lewm/tests/test_build_go2_world_model_counterfactual_calibration_plan_v1.py \
  lewm/tests/test_build_go2_world_model_counterfactual_calibration_authority_v1.py \
  lewm/tests/test_run_go2_world_model_counterfactual_calibration_authorized_v1.py \
  lewm/tests/test_run_go2_world_model_counterfactual_smoke_authorized_v1.py \
  lewm/tests/test_analyze_go2_world_model_progression_v1.py \
  lewm/tests/test_go2_world_model_counterfactual_textured_v03.py \
  lewm/tests/test_go2_world_model_visual_domain_parity_plan_v1.py \
  lewm/tests/test_go2_world_model_visual_domain_parity_authorized_v1.py \
  lewm/tests/test_go2_world_model_bounded_branch_experiment_v1.py \
  lewm/tests/test_go2_world_model_bounded_branch_lineage_v1.py \
  lewm/tests/test_go2_world_model_bounded_branch_runtime_boundary_v1.py \
  lewm/tests/test_go2_world_model_bounded_branch_evaluation_panel_runner_v1.py

Final integrated verification after the pre-reservation lifecycle correction:
`315 passed` in `7.66 s`. The prior `313 passed` result did not exercise the
fresh-root versus reserved-root chain rehash and is no longer current source
evidence. The earlier `141 passed` result predates the visual-domain, lineage,
place-provenance, joint-gate, decoded-pixel, and arm-hierarchy corrections and
is also not current completion evidence.
```
