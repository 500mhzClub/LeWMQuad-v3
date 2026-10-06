# Research artefact migration plan, 6 October 2026

Andrew, 6 October: "a hard stop after this round of experimentation … consolidate where we are, what's in place, and tidy the repo into something more focused, and consumable as a research artefact … a new repo with a more appropriate name, as well as all of the scripts/artefacts to recreate", then "create a migration plan to produce a clean repository that implements everything required for this paper".

This plan is for review. Nothing has been copied or created yet.

## 0. Decisions (Andrew, 6 October)

| | Decision | Consequence for the plan |
|---|---|---|
| Scope | "all results comparing the different harnesses once we established they could all complete the mazes" | Read as the controller comparisons after the capability gate: R1, R3, R4, R5 and R6, with R2 as the method behind the compared C3. The capability-qualification path itself (V4 gate, validation) is background. **To confirm.** |
| Repository | **Go2-JEPA-Navigation**, public | The first push needs a pre-publication review by Andrew: licences, secrets, absolute paths, personal data, and a name-only sealed scan. Nothing is pushed before it. |
| Shipped data | yes, as proposed | Checkpoints, maze sets (excluding sealed), per-mission result summaries, gate replay inputs. Raw runs and caches are archived and rebuildable. |
| Training | ship checkpoints and their lineage, and **all code to retrain each component from scratch** | Every learned component must retrain from shipped data with a seed parameter. **Multiple seeds per component** will be run later, so published results are not one lucky seed. See section 3a. |
| Platform | ROCm for now | Pin the ROCm stack. Replace the `.pth` bridge with one locked environment. Docker is optional. |
| Extras | ship them, for later review | C1A, C1R, the forecast-degradation hooks, demo textures and hero video, the marker probe and the unmarked control go under `experiments/dev/` with their results. |

## 1. What the paper rests on

Everything here is from the navigation-capability programme (25 September onwards), and all of it is labelled **preliminary** (development mode).

| # | Result | Record |
|---|---|---|
| R1 | Testbed and capability: C0–C4 on 60 preliminary-test mazes, 332 missions, recovery on and off | `go2_navigation_preliminary_results_2026-10-02.md` (+ tables, budget rescore) |
| R2 | C3 decoder fix: forecast accuracy by movement type, and C3 vs C4 at matched data | `CURRENT_RESEARCH_BRIEF.md` (decoder progress), `dev_decoder_fits/` |
| R3 | Forecast sensitivity (dose-response of degraded C1 forecasts) and calibrated margins | `go2_navigation_forecast_sensitivity_2026-10-02.md`, `…_tables`, `go2_navigation_calibrated_margin_results_2026-10-02.md` |
| R4 | Harness v4 → reserve-exit v2: re-run results and known limitations | `go2_navigation_reserve_exit_rerun_results_2026-10-03.md`, `go2_navigation_harness_v4_known_limitations_2026-09-29.md` |
| R5 | Dynamics stage 1: uniform low friction, zero-shot | `go2_navigation_dynamics_stage1_v2_results_2026-10-05.md` |
| R6 | Dynamics stage 2: marked slip strips, matched refits, offline marker control, closed-loop C1 family | `go2_navigation_dynamics_perturbation_plan_2026-10-02.md` (stage-2 sections), `go2_navigation_stage2_strip_stall_diagnosis_2026-10-05.md` |
| V | Videos: one per controller, plus the textured hero demo | `<capability root>/videos/`, `~/Videos/LeWMQuad_JEPA_navigation_hero_textured_maze30.mp4` |

**Decision A (Andrew): which of R1–R6 and V are in the paper, and what is the headline claim?**

The programme question is "when do JEPA representations carry decision-relevant information for navigation beyond kinematic prediction, reactive control and explicit memory". The current evidence answers it mostly in the negative:
- in this testbed, C3 ≈ C4 ≈ C1 in outcomes (R1);
- outcomes are insensitive to forecast error over a wide range (R3);
- vision-driven refits recognise slip but barely anticipate it (R6).

Earlier programmes (decision headroom, May–August LeWM/topological navigation) are out of scope. They are history, cited if needed.

## 2. What is in place (measured)

### Code

`scripts/trace_go2_research_artifact_closure_2026_10_06.py` follows imports and repository-path string constants from the entry points behind R1–R6 and V. It reads filenames only for sealed material, and opens no sealed file.

| | Files |
|---|---|
| Tracked Python in the repository | ~5,980 |
| **Static closure of the paper's entry points** | **1,617** (lewm 957, scripts 607, lewm_genesis 38, lewm_worlds 15) |
| Non-code files the closure reads at runtime | 365 (357 under `docs/`: protocols, preregistration, budgets and bindings that the frozen harness loads and often hash-checks; 3 config, 3 assets, 1 model) |

**Why the closure is so large:**
1. **Layered development modules.** Each harness or controller version imports its predecessor and extends it (`previous.study.previous.reference…`). The deployed system is the top of a deep stack of mixins, composed at runtime from fix names (`dev_harness_fixes_development.compose`) and the pinned harness version (`reserve_exit_v2`).
2. **Runtime-bound documents.** Protocols, preregistration and budgets live in `docs/*.json` and are read, often hash-verified, at mission time.
3. **The model loader imports its training scripts to locate checkpoints,** so the whole training lineage is in the static closure.

The static closure over-approximates. Many imported modules contribute one constant or a base class.

### Runtime models

All are kept (storage review, 4 October):

| Model | Location | Size |
|---|---|---|
| V-JEPA 2.1 ViT-L encoder (external, Meta) | `~/.cache/vjepa2_1_vitl_dist_vitG_384.pt` | 5.2 GB |
| Frozen action-conditioned predictor | workspace drive `go2_horizon_dense_predictor_v1_attempt_001/action_final.pt` | small |
| Maze-view readout | `go2_maze_view_readout_v1_attempt_003` | small |
| C3 decoder and matched C4 | `dev_decoder_fits/p3_large_past_frames_s2026093011.pt` | in 2.5 GB folder |
| C1 command model | `go2_short_pulse_command_control_v1_attempt_001/command_only.npz` | KB |
| Locomotion policy | `models/tier_a_go2_locomotion/20260516_contract_ppo` (in repo) | 4.4 MB |
| Proprio normalisation | temporal cache `proprio_v1` | KB |
| Stage-2 refits | `stage2_decoder_fits/`, `stage2_c1_refit_v1/` | 347 MB |

### Training lineage

All inputs are still on disk (storage review, "Training lineage"):
- **Predictor:** a 4-stage chain. The August temporal model was trained on cached V-JEPA features of 18,690 May-corpus frames in 80 scenes. It then went through native adaptation and balanced-start predictor stages to reach the horizon dense predictor. Five families of training recordings feed it.
- **Maze-view readout, C1, and C3/C4 decoders:** recordings and feature-cache manifests as recorded.

### Results and data in the capability root

| Item | Size | Contents |
|---|---|---|
| Mission runs | 193 GB | 1,890 runs: logs, traces, planning; RGB frames in some |
| Stage-2 feature cache | 63 GB | rebuildable in about 6 h of GPU |
| Unmarked eval cache | 3 GB | rebuildable |
| Replays | 2.7 GB | |
| Videos | 1.9 GB | |

**Maze sets:** dev, validation, round, `prelim_test_v1` (declassified), stage2 fit, held-out and eval. **`sealed_test_v2` stays sealed and is not migrated.**

### Environment

Genesis 0.4.6 plus torch 2.12 / ROCm 7.2 (through a `.pth` bridge into `world_model_rocm_7_2_1_v1`). Single AMD R9700. CPU physics is deterministic; GPU encoder numerics depend on batch composition.

## 3. Strategy: two artefacts, the first the oracle for the second

### A. Frozen reproduction snapshot (faithful, not pretty)

- **Contents:** the exact closure (1,617 + 365 files) at a frozen commit.
- **How it is built:** exported by an **explicit, SHA-256-checked file manifest**, the only form AGENTS.md permits. No clone, archive, worktree or recursive copy. Legacy sealed blobs are in this repository's history, so the new artefacts start with **fresh history**. A provenance manifest maps every exported file to (source commit, path, sha256).
- **Purpose:** an exact reproduction of every logged result, and the oracle for B. It is published only if wanted, as an appendix or archival record.

### B. Clean repository (the research artefact)

- **Contents:** a flat, readable implementation of the final system only: harness `reserve_exit_v2` + coverage fix, recovery switch, controllers C0–C4 (+ C1A, C1R), perturbation hooks, the training pipeline, and the experiment and report scripts for R1–R6.
- **What goes:** no version stacks and no runtime-bound docs. Configuration lives in versioned YAML with hashes.
- **Behaviour is proven equal to A,** not assumed.

**Name, Decision B** (suggestions): `go2-jepa-nav`, `jepa-nav-testbed`, `vjepa-go2-navigation`. The current name, LeWMQuad-v3, reflects the earlier LeWM world-model programme.

**Proposed layout:**
```
<name>/
  go2nav/
    sim/          Genesis scene from a maze spec, cameras and sensor packets, depth noise,
                  locomotion-policy wrapper, friction field and patches, demo textures
    mazes/        generator, exclusion registry, set registration and verification
    perception/   RGB-D tracker (SIFT + LK), registration, occupancy map, obstacle vetoes
    planning/     candidate primitives and tapes, scoring, reserve-exit harness, dispatch,
                  safety limits, recovery behaviours
    models/       V-JEPA encoder wrapper, horizon-conditioned predictor, readouts and C3 decoder,
                  C4 supervised predictor, C1 command ridge, C1A adaptive travel
    controllers/  C0 oracle, C1, C1A, C1R, C2 reactive, C3 JEPA, C4 supervised
    data/         recordings, verified replay, context rules, feature caches
    training/     predictor lineage (4 stages), readout, decoder and C4 fits, C1 fit, refits
    eval/         mission runner, cohort runner, episode evaluation, scoring, edge bins
  configs/        platform manifest, harness, controllers, protocol values (from docs/*.json)
  experiments/    one driver + config per result R1–R6 and V
  paper/          table and figure generation from result files
  artifacts/      manifest (URLs + sha256 + licence) and fetch script
  tests/          unit tests + equivalence tests against snapshot A
  docs/           reproduction guide, results map, limitations, custody note
```

### 3a. Retraining from scratch and multi-seed runs

**Component graph** (each node gets a `seed` and records a lineage manifest):

| # | Component | Trained on | Parent |
|---|---|---|---|
| T1 | temporal action predictor (August) | V-JEPA 2.1 features of 18,690 May-corpus lineage frames (80 scenes), `proprio_v1` | fresh weights |
| T2 | native adaptation | native Go2 recordings | T1 |
| T3 | balanced-start predictor | moving-action-switch, geometry-progress, short-pulse, balanced-start recordings | T2 |
| T4 | horizon dense predictor (**the frozen JEPA predictor**) | balanced-start horizon-action recordings | T3 |
| T5 | full-heading readout → maze-view readout | readout recordings and frame lists | T4 features |
| T6 | C1 command ridge | family, switch and short-pulse contexts | closed form, no seed |
| T7 | C3 decoder + matched C4 | feature cache from T4 (old, maze and on-policy contexts) | T4, T5 |
| T8 | stage-2 refits (C3/C4, C1R) | stage-2 cache | T7, T6 |

**Training data to ship** (measured):

| Data | Size |
|---|---|
| Predictor-stage recordings | ~8.5 GB (switch family 5.9, geometry-progress 1.6, short pulse 0.36, balanced start 0.14) |
| Predictor checkpoints | ~0.8 GB |
| Readout and transfer data | ~0.3 GB |
| C3/C4 decoder data, rest/turn recordings and gap replays | ~2.6 GB |
| Stage-2 recordings and replays | ~2.7 GB |

The **T1 inputs** are the 18,690 lineage frames (size to be measured) plus `proprio_v1`. Their 33 GiB of cached V-JEPA features are rebuilt by a shipped script rather than shipped. Feature caches for T7/T8 are likewise rebuilt, not shipped. The total is in the tens of GB, which suits Hugging Face Hub datasets.

**Lineage manifest per checkpoint:** parents, with their sha256; the training-data manifest sha256; code commit; config; seed; metrics. The shipped checkpoints get retrofitted manifests from the existing records (plans, results and hashes in `docs/`).

**Multi-seed protocol.** This is designed now, run later in the rigorous phase:
- Seeds come in **full chains**: seed *k* retrains T1→T8 end to end. That measures end-to-end variance, not just the last stage's.
- C1 is closed form, so its variance comes from data, not a seed.
- Then closed-loop evaluation of every seed's C3 and C4 on the rigorous-phase set.
- The number of seeds and the evaluation set are fixed before any seed result is seen.
- Compute is estimated in P1 from the original stage timings. Closed loop dominates: C3 is about 10 GPU h per 20 missions.

**Retraining gate (G5):** each component, retrained in B with its original seed and data order, reproduces its logged metrics. It reproduces the checkpoint hash exactly where the training is deterministic, and stays within a stated tolerance where GPU kernels are not.

### Equivalence gates (B must reproduce A)

The repository already has the needed machinery: verified replays that assert every hash, decision, dispatch command and trace value, and fit-record reproduction (the stage-2 cache reproduced the decoder fix's metrics with 0 difference across 14 sets).

| Gate | Check | Tolerance |
|---|---|---|
| G1 component | tracker, map, forecasts (C1, C3, C4) and vetoes on recorded inputs | bit-exact on CPU paths; GPU encoder at recorded batch composition |
| G2 mission replay | B replays logged missions from snapshot A: one per controller, recovery on and off, friction, patches (marked), stall stop, and the final-frame-failure case | identical decisions, dispatches and native traces |
| G3 offline | B rebuilds a cache subset and refits; it reproduces the logged fit records and the stage-2 tables | exact (same seeds and data order) |
| G4 fresh closed loop | a small cohort (e.g. 5 mazes × C1, C3, C4) run in A and in B | identical outcomes and traces on CPU controllers; C3 within batch-numerics tolerance |

## 4. Phases

| Phase | Work | Engineering | Compute |
|---|---|---|---|
| P0 | Decisions A–F (below) | – | – |
| P1 | **Runtime trace:** run one short mission per controller and condition, plus one cache, fit and report each, under coverage. This gives the executed functions, which become the porting list. The static closure (1,617 files) should shrink to a few hundred functions. | 0.5 day | ~1 day CPU/GPU, parallel |
| P2 | **Snapshot A:** manifest, hash-checked export, then a bit-exact replay smoke from the snapshot (paths remapped by config) | 0.5–1 day | hours |
| P3 | **Clean repo skeleton:** configs (from the docs bindings), artefact manifest and fetch script, CI with unit tests | 0.5 day | – |
| P4b | **Training pipeline port:** T1–T8 as seed-parameterised stages with lineage manifests; data manifests and fetch; gate G5 on each stage with its original seed | 1 week | GPU: the original stage timings, once each |
| P4 | **Port bottom-up with gates:** sim → perception → planning/harness (hardest: the mixin stack and fixes) → models and controllers → data/training → eval/reports. G1 per layer, G2/G3 at the end. | **1.5–2.5 weeks** | replays: hours |
| P5 | **Artefact packaging:** checkpoints, maze sets, result tables and per-mission JSON summaries, selected replay inputs | 1 day | upload time |
| P6 | **Docs:** README, paper-results map (each table and figure → command → expected numbers), limitations, custody note | 1 day | – |
| P7 | **Fresh-machine verification:** clean environment, fetch artefacts, run G2/G3 and regenerate the paper tables | 0.5 day | ~0.5 day |

**Total:** about 4–5 weeks of engineering, with P4 and P4b dominating, plus GPU time for the G5 retraining checks. The multi-seed runs belong to the rigorous phase, after the repository is gated.

## P1 result: runtime trace (6 October, 13:00)

The porting list is in `docs/go2_research_artifact_p1_runtime_trace_2026-10-06.json`, with per file and per function the executed lines.

**Method.** Coverage, installed outside the pinned environment (`scripts/trace_go2_p1_runtime_development.py`).
- The frozen harness verifies its environment exactly: Python distributions, `PYTHONPATH` and render variables. So entries are booted under coverage by rewriting the launcher's subprocess commands, with the environment untouched.
- Spawned workers get coverage through a spawn-executable shim.

**What was traced.**
- 11 short missions (cohorts `p1t3_*`):
  - C0, C1, C2, C3 and C4, plus C1A and C1R;
  - recovery on and off;
  - uniform friction, marked patches with the recording stall stop;
  - the v2 launcher's forecast degradation and calibrated margins, with the pessimistic-unknown map.
- 10 offline steps: reports, scoring, context counts, the C1 refit check, the cache collector, the decoder baseline and the marker control.
- The two rewritten result files (decoder baseline, marker control) reproduced their originals exactly. The originals are kept in `stage2_decoder_fits/original_before_p1_trace/`.

**Outcomes** matched expectations. 9 of 11 completed a round trip. The two failures:
- **C4 on maze 45 with recovery on:** the same failure as the preliminary run (the latch release).
- **C1 with 40-mm forecast noise:** a degraded condition.

**Size of what actually runs:**

| | Count |
|---|---|
| Files in the static closure | 1,617 |
| Files with any executed line | 918 (24,417 lines; most are import-time `def`, `class` and constant lines) |
| **Files with an executed function body** | **350** (lewm 239, scripts 88, lewm_genesis 22, lewm_worlds 1) |
| **Executed functions** | **1,126**, with 8,467 executed lines in their bodies |

So the clean repository's runtime core is about 1,100 functions and about 10k statements, not the static closure's 1,617 files. The rest is version layering, alternatives that were never selected, and import scaffolding.

**Not traced** (ported from static reading instead):
- **Training lineage T1–T5:** checked by gate G5.
- **Data and set generation:** data collection and the capability maze generator.
- **Rebuild and render steps:** feature-cache encoding, verified replay, the unmarked re-render, the marker probe, and video rendering.
- **Rare runtime branches:** contact handling and some recoveries. Gate G2, run over many logged missions, catches these.

**Effect on the estimate.** P4 (porting the runtime core) looks like the lower end of its 1.5–2.5 weeks. P4b (training) is unchanged.

## P2 and P3 (6 October, 14:30)

Andrew: "do it. all code for maze gen, RL locomotion training, data generation, training, inference for each harness type, and experiments that produce comparisons between the harness should be ported".

**Closure extended to the whole pipeline.** `scripts/trace_go2_research_artifact_closure_2026_10_06.py` now covers maze and scene generation, locomotion RL training (`train_genesis_go2_locomotion_contract`, the upstream Genesis example fetcher, the contract check), May-corpus data generation (shell drivers followed), the T1 temporal predictor and its cache builders, T2–T5 training, all training-data collection, the C1 lineage, the historical C3/C4 stages, every pinned launcher, and the experiment reports.
- It also follows repository paths named inside JSON/YAML bindings. The first snapshot missed a hash-bound file that only a docs binding names.
- `docs/go2_research_artifact_closure_2026-10-06.json`: 2,313 code and binding files, plus 540 data files.

**Lineage scripts.** `scripts/discover_go2_lineage_scripts_2026_10_06.py` reads the script hashes recorded in each lineage artefact (`docs/go2_research_artifact_lineage_scripts_2026-10-06.json`). Every recorded hash matches the HEAD version, so one frozen commit serves as the source of every component.

**Snapshot A exported:** `~/Workspace/LeWMQuad-v3-snapshot-2026-10-06`.
- 2,751 files (45 MB) from commit `935f923a`, by `scripts/export_go2_research_snapshot_2026_10_06.py`: an explicit list, per-file SHA-256 against the commit blob, sealed names refused before reading.
- Fresh git history; `SNAPSHOT_MANIFEST.json`.

**Snapshot check (passed).** C1 on prelim maze 30, recovery off (`snapA_smoke2`), against the same mission from this repository (`p1t3_base`):
- the physics trace (51,010 samples), dispatch requests, poses, model calls and outcome are identical;
- `planning.json` differs only in one wall-clock timing field.

**P3: clean repository skeleton** at `~/Workspace/Go2-JEPA-Navigation`. Local only; nothing is pushed before Andrew's review.
- The `go2nav` package layout and the ROCm 7.2 environment lock (111 packages, with `rsl-rl-lib` 5.4.1 for locomotion retraining).
- A custody test (no sealed names, no absolute home paths in code).
- `docs/PORTING_MAP.md` (350 traced files with targets).
- `artifacts/manifest.yaml` (14 checkpoints with SHA-256).
- `docs/MIGRATION_STATUS.md`, the resume point for the port.

## 5. Decisions needed

- **A. Paper scope:** which results (R1–R6, V) and the headline claim. This sets the porting list in P1.
- **B. Name**, and whether the repository is public or private, and its licence.
- **C. What ships as data:**
  - **Proposed:** checkpoints, maze sets (excluding sealed), per-mission result summaries, and the replay inputs for the gate missions. Raw runs (193 GB) and caches (66 GB) stay as archive and are rebuildable.
  - **Hosting:** Hugging Face Hub or Zenodo.
- **D. Training from scratch:**
  - **Proposed:** ship the predictor and readout checkpoints and document the 4-stage lineage, with its scripts ported but not CI-gated.
  - **Alternative:** gate a full re-train. That needs the 33 GiB feature caches and the lineage recordings shipped, and days of GPU.
- **E. Platform:** keep ROCm-pinned, or make it CUDA/ROCm-agnostic (the code is torch-generic; the pins are environment-only).
- **F. Development-only material:**
  - the C1A / C1R arms, the forecast-sensitivity degradation hooks, and the demo textures: keep as experiments, or drop;
  - the decision-headroom and earlier programmes: excluded, with a pointer to this archive.

## 6. Risks

- **Behaviour drift while flattening the harness.** Mitigated by gates G1–G4. Every logged mission is a test case.
- **Licences.**
  - V-JEPA 2.1 weights: Meta's licence; may need a download-from-source step instead of re-hosting.
  - Go2 URDF: from Genesis assets.
  - Textures: CC0.
  - Locomotion policy: ours.
- **Custody:**
  - `sealed_test_v2`, its backup and the legacy sealed files are excluded by name and never read.
  - The rigorous phase regenerates a sealed set with B's generator and an exclusion list covering every graph used here.
  - The export tooling itself must honour the filename-only rule.
- **Provenance:** run records cite commit hashes of this repository (launch pins). This repository must stay intact, and the provenance manifest links to it.
- **Determinism:** GPU numerics depend on batch composition, so C3 gates use the recorded batch composition or a stated tolerance.
