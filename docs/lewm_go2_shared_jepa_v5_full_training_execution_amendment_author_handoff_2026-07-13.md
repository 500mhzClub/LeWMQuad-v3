# Shared JEPA V5 full-training amendment author handoff

Date: 2026-07-13

Author: `/root/raw_plan_v2_qa`

Status: **AUTHOR COMPLETE; DIFFERENT-AGENT REVIEW REQUIRED BEFORE IMPLEMENTATION**

## Frozen artifact

| Artifact | SHA-256 |
|---|---|
| `docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_2026-07-13.md` | `b21d01d062543cc7b7f3f5281f66ac40df76726c678a9364f7a4e451b035a4a7` |

The amendment is a source/data-read-free preregistration. It creates no model,
trainer, checkpoint, calibration, G2 authority, result, output directory, or
promotion license.

## Reviewed parents

The author rehashed and reviewed the following already-passed source boundary:

| Artifact | SHA-256 |
|---|---|
| V5 model/output/loss source | `b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9` |
| V5 model tests | `848aa8be369b89c973a4da916f9c7abeff47eca12aceb4304cf612ed4d53227b` |
| staged lifecycle core | `62a19f3028e9152120af990528752431b996f56b4bc9b62db32eba47ae235a1f` |
| staged fixed launcher | `7f273649fa6c8b4256c552359927fc20bb59d1bfbd5b47194a3f5a941c5b8958` |
| G2/G3 runner wrapper | `37402f0f75a7a4f475539e269e77aeae072ce80b0af0bcb4147e2ec1b33ff57a` |
| G2/G3 finalizer wrapper | `f0426201f5344d0eb1d43e183e4755ac8fd7aecdc9af6e5b7c19076af3f5dc34` |
| candidate/full publisher wrapper | `4e045365dadb28bd37cdbb49808bef7528d4e5cb0c3e77ff5aae678559174fab` |

The model source fixes one online encoder, exact V4 fit migration, the
production config, current/next complete four-equal V4 loss, established JEPA
package, EMA update boundary, deployment state filtering, and checkpoint-v5
provenance. The separately passed lifecycle consumes only an already-qualified
checkpoint and retains all six production identities as `None`.

## Scientific freeze summary

The amendment fixes these choices before learned output:

- both V4 seeds must pass every `N=5,16,32,320` rung, but only the preregistered
  primary seed `20260710` N=320 state may migrate;
- one CPU FP32 V5 initialization with seed `20260712` is migrated once and
  copied exactly into promoted-JEPA and matched-no-JEPA arms;
- train role only: 72 scenes, 4,262 pairs, 7,777 unique endpoints, eight
  families, no subsampling;
- concatenated CPU `randperm` schedule seed `20260713`, 8,000 updates,
  effective batch 16, microbatch four, accumulation four, 128,000 pair
  presentations;
- AdamW, a fixed warmup/cosine schedule, FP32, global clip 1.0, and one EMA
  update after each optimizer step;
- GPU0 exact R9700 only, at least 32 GiB; Raphael/GPU1/iGPU is forbidden;
- promoted backward uses the exact established JEPA total plus both endpoint
  V4 packages, each containing all four equal V4 components;
- eight fixed checkpoint steps; checkpoint-selection role alone determines
  eligibility and a total lexicographic ranking under mandatory physical and
  JEPA-health gates;
- the ablation is evaluated only at the already selected promoted update and
  can never replace it;
- all 759 unique calibration endpoints, fixed six-parameter positive diagonal
  vector scaling, fixed LBFGS settings, and a fixed conservative probability
  grid produce one global threshold tuple per arm;
- aggregate and all eight family development gates must pass; and
- exclusive immutable output, a content-chained actual-open ledger, independent
  reconstruction, terminal no-retry semantics, and one role-global eight-scene
  G2 attempt are mandatory.

## Exact metadata assumptions

The role universe is not inferred from payloads. It comes from frozen reviewed
documentation and commitments:

- paired manifest file SHA-256:
  `ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180`;
- row index file SHA-256:
  `187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac`;
- assignment SHA-256:
  `016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02`;
- metadata-plan content/pairs/endpoints SHA-256:
  `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`,
  `76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`,
  and `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`;
- roles: train `72/4262/8524/7777`, selection `8/495/990/924`,
  calibration `8/415/830/759` for scenes/pairs/endpoint instances/unique
  endpoints; and
- untouched G2: eight scenes, 469 parent pairs, set commitment
  `0c9d5cfb6fdeec9be17a1afa8aed13fb62848a06594782c98933e1db8a2e1402`.

The exact G2 endpoint population was intentionally not derived or opened. The
staged G2 runner must reconstruct its own already-governed role input only after
the qualified-checkpoint boundary.

## Open prerequisites and assumptions

1. The final V4 full-panel two-seed ladder does not yet exist. The amendment
   binds its required structure and deterministic primary-seed rule, not future
   checkpoint hashes.
2. The full development raw-supervision artifact does not yet have an
   independently accepted builder manifest/audit. The trainer must bind those
   future exact hashes without changing this scientific contract.
3. The `4 x 4` accumulation choice is sized conservatively for the discrete
   34 GB R9700 from source-shape analysis, not a hardware run. A reviewed
   payload-free synthetic GPU0 smoke is mandatory before exact reservation. A
   failed smoke requires a new pre-output amendment.
4. The fixed physical thresholds reuse the frozen V4 N=320 development gate;
   the fixed JEPA-health thresholds reuse the established persistence,
   noncollapse, rank, and counterfactual boundaries already present in the
   repository plan/source.
5. The vector-scaling and conservative-threshold arithmetic is fully stated in
   the amendment so trainer implementation cannot inherit an unreviewed mutable
   utility by name alone.
6. PyTorch/ROCm/runtime versions, complete transitive trainer sources, the
   generated schedule hash, initial state hash, parameter/state inventory, and
   future dataset/V4 artifact hashes must be frozen in an implementation
   manifest before any exact run.

## Access statement

This authoring pass read repository Markdown and Python source only. It did not
open any `.generated` dataset, row index, sidecar, source scene, frames file,
render plan/summary, RGB, raw label, model/checkpoint payload, calibration
payload, G2/G3 role payload, held-out/sealed input, runtime/navigation result,
physical executor/reset input, accelerator, hardware input, or production
artifact. It executed no Torch model construction, inference, optimization,
calibration, or GPU command.

## Required independent review

A reviewer other than `/root/raw_plan_v2_qa` must rehash the amendment and all
bound parent sources/documents, independently verify every count and formula,
and challenge at least:

1. second-seed substitution, averaging, missing-rung, or partial-gate migration;
2. nonidentical arm initialization, RNG, schedule, microbatch, or optimizer
   state;
3. raster-only, single-endpoint, reweighted, or partial JEPA/V4 loss paths;
4. train/selection/calibration leakage and ablation-driven checkpoint choice;
5. failing-checkpoint ranking, rounded ties, late threshold changes, per-family
   calibration, or rare-class backfill;
6. R9700/iGPU confusion, adaptive OOM fallback, extra EMA updates, or hidden
   mixed precision;
7. mutable/replayed output, incomplete access events, self-certified metrics,
   retry/resume, and post-result escalation; and
8. G2 opens before role-global reservation, a second G2 namespace, checkpoint
   substitution after contact, or premature G3 binding.

The reviewer must issue a separate PASS or BLOCK record. A PASS licenses only
additive trainer/verifier/publisher implementation against this preregistration.
It does not license data construction/use, V4 execution, model training,
selection, calibration, checkpoint use, G2/G3, held-out, runtime, hardware,
navigation, production, or promotion.
