# V4 N5 full-panel V1 implementation handoff

Date: 2026-07-13

Status: **source implementation complete; exact execution blocked pending a different-agent review**

Implementation author: `/root/v4_execution_successor_review`

## Frozen authority

The implementation binds these records before any train RGB decode, model
construction, GPU query, attempt reservation, or output creation:

- preregistration:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_successor_preregistration_2026-07-13.md`,
  SHA-256
  `0ad13e3897c70f90df6705538f4d86262ec53d3e096618a69563acdf63567c01`;
- structural-trigger amendment:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_structural_trigger_amendment_2026-07-13.md`,
  SHA-256
  `1e08aac0ace734d2cbcce9e965b10a7031a94764dd7b47114d38e33944990262`;
- terminal invalidation JSON:
  `docs/lewm_go2_observable_camera_ray_fit_v4_prepublication_structural_invalidation_2026-07-13.json`,
  file/content SHA-256
  `1744a50badd6c9f5c1ef4c8c3cbd05f8c0fc8acff4fbbf066e40e1f7de24f560` /
  `7bdaae6ebb13b7d90290dfe07f5d48f403d29cad977f4a56c9ac7b8cfbcb8602`.

It also rehashes the frozen V2 numerical/data dependencies and exact target
partition sources before accepting the new source-review record.

## Source closure

- policy and artifact validator:
  `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`,
  `875edc86efbe25d246b24c2ef2467cc7956b1b3bb90e6d8d1e03e4a9c5b11d88`;
- isolated launcher:
  `scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`,
  `3cb9ff782a15bc97dd3cca2cc25705e006d6af19a7dbef6d27dee893d9b570c8`;
- trainer:
  `scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`,
  `48ac856c080906a8d73d5a9b97d1dcf7fe21f5bc99217cce669c43b9c091acca`;
- independent inference verifier:
  `scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`,
  `00c62cec39e1eb05bf23a96a9153aa8ff350235c2e5c6662f6148934ab9d85b0`;
- fail-closed finalizer:
  `scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`,
  `1d4471381a6c3b29f0b077e44e3126f956281ff105d4e38aa8e0f6ba18675b8b`.

The required review must bind all five exact files. It must also reproduce the
frozen dependency map, authority bindings, experiment contract, output root,
and licenses exposed by
`go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`.

## Exact experiment

The source has no CLI surface for changing the scientific configuration. It
fixes:

- seed `20260710`, the same frozen `N=5` subset and targets;
- fresh `ObservableCameraRayEvidenceV4Model` initialization;
- AdamW, 400 optimizer updates, batch size 5;
- 2,000 total frame exposures;
- seeded concatenated-randperm schedule SHA-256
  `62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`;
- learning rate and weight decay `1e-4`;
- float32, no autocast, global gradient clip `1.0`;
- the same four losses weighted `0.25` each;
- final update only, with evaluation batch size one;
- matched and wrong-RGB controls;
- GPU0 R9700 only, Raphael/iGPU rejected;
- native thread caps of one and 1 to 5 RGB decoder workers;
- one exclusive attempt at
  `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/seed_20260710/n5`.

The trainer recomputes aggregate evaluation totals from the four aggregate
components, then validates them with the unchanged `1e-9` frozen structural
tolerance before publication. The verifier independently loads and rehashes
the checkpoint state, decodes the five selected RGBs again, reruns batch-one
matched and wrong-RGB inference, requires exact equality to the result, and
calls the unchanged frozen metric and threshold functions. It permits no
metric repair, commitment exception, or threshold weakening.

## Attempt lifecycle

The reservation is built and fsynced in an exclusive sibling staging
directory, then atomically renamed to the canonical attempt. A pre-claim fault
removes staging and leaves the canonical attempt unclaimed. Every caught fault
after the atomic claim removes partial checkpoint/result/completion artifacts
and publishes a terminal `failed.json` with retry authority false.

After training and complete input revalidation, all five selected RGB paths
are rehashed against their exact commitments before checkpoint serialization.
Tests demonstrate that a mutated RGB is rejected. A successful attempt writes
the checkpoint and result exclusively and writes `completed.json` last.

A structurally valid numeric failure is terminal. A pass authorizes only
design and different-agent review of a later-rung schedule. Neither outcome
directly authorizes N16 execution, another seed, V5, G2, held-out, selection,
calibration changes, runtime, hardware, production, or promotion.

## CPU verification

The focused and predecessor closure command passed 48 tests:

```bash
env PYTHONPATH=/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages \
  python3 -m pytest -q \
  lewm/tests/test_launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py \
  lewm/tests/test_train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py \
  lewm/tests/test_verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py \
  lewm/tests/test_finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py \
  lewm/tests/test_diagnose_go2_observable_camera_ray_fit_v4_structural_invalidation.py \
  lewm/tests/test_train_go2_observable_camera_ray_fit_v4_v2.py \
  lewm/tests/test_launch_go2_observable_camera_ray_fit_v4_v2.py \
  lewm/tests/test_verify_go2_observable_camera_ray_fit_v4_metrics_v2.py \
  lewm/tests/test_finalize_go2_observable_camera_ray_fit_v4_ladder_v2.py
```

Coverage includes the 400-by-5 schedule, strict loss arithmetic, source-review
mutation, selected-RGB mutation, reservation-license mutation, pre/post atomic
claim fault injection, sealed access ledgers, independent verifier source, and
unchanged N5 thresholds. No GPU or exact-data execution occurs in these tests.

## Required independent review

Execution remains unavailable until an agent other than
`/root/v4_execution_successor_review` reviews the exact source closure, reruns
the CPU command above, and exclusively creates canonical JSON at:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_independent_review_2026-07-13.json`

The record must use schema
`lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_source_review_v1`, set
status to `different_agent_review_passed_exact_full_panel_v1`, name the distinct
reviewer, and match `expected_source_review_core(...)` exactly before adding its
canonical `content_sha256`. Author self-review is rejected in code.

Review command:

```bash
env PYTHONPATH=/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages \
  python3 -m pytest -q \
  lewm/tests/test_launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py \
  lewm/tests/test_train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py \
  lewm/tests/test_verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py \
  lewm/tests/test_finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py
```

Only after that review is frozen may the different agent authorize this exact
command. The implementation author must not run it:

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  HIP_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py \
  --source-review \
  docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_independent_review_2026-07-13.json \
  --source-review-sha256 DIFFERENT_AGENT_FROZEN_FILE_SHA256 \
  --rgb-workers 5
```

At this handoff, the canonical source-review file and the entire
`n5_full_panel_v1` output tree are absent. No exact attempt, reservation, RGB
decode, model construction, checkpoint use, or GPU query has run.
