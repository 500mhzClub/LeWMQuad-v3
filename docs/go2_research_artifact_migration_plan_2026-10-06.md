# Research artefact migration plan, 6 October 2026

Andrew, 6 October: "a hard stop after this round of experimentation … consolidate where we are, what's in place, and tidy the repo into something more focused, and consumable as a research artefact … a new repo with a more appropriate name, as well as all of the scripts/artefacts to recreate", then "create a migration plan to produce a clean repository that implements everything required for this paper".

This plan is for review. Nothing has been copied or created yet.

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
| P4 | **Port bottom-up with gates:** sim → perception → planning/harness (hardest: the mixin stack and fixes) → models and controllers → data/training → eval/reports. G1 per layer, G2/G3 at the end. | **1.5–2.5 weeks** | replays: hours |
| P5 | **Artefact packaging:** checkpoints, maze sets, result tables and per-mission JSON summaries, selected replay inputs | 1 day | upload time |
| P6 | **Docs:** README, paper-results map (each table and figure → command → expected numbers), limitations, custody note | 1 day | – |
| P7 | **Fresh-machine verification:** clean environment, fetch artefacts, run G2/G3 and regenerate the paper tables | 0.5 day | ~0.5 day |

**Total:** about 3–4 weeks of engineering. Phase P4 dominates, and it scales with how much of R1–R6 is kept.

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
