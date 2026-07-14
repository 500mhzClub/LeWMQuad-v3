# Independent source review: Camera V9 hierarchical first-hit successor

Date: 2026-07-13

Reviewer: `/root/raw_v8_builder_reviewer`

Implementation author: `/root/coordinator_v2_qa`

Authority amendment SHA-256: `ccc8097b4d3bd70aabf3c701226928e360fafb04a12a452c4fd406e9bba3db0a`

Verdict: **PASS**

## Scope and access boundary

This is an independent source-closure review of the frozen nine-file Camera V9 candidate. The reviewer differs from both the amendment author and implementation author. Candidate files were not modified.

The review did not access exact-attempt outputs, datasets, `.generated`, RGB evidence, checkpoints, GPU state, G2, held-out results, runtime hardware evidence, navigation, production, or promotion paths. Prior V8 terminal evidence was treated as identity-only evidence. No numeric V5/V6/V7/V8 payload and no V8 checkpoint was inspected.

## Frozen bindings

| Role | Path | SHA-256 |
|---|---|---|
| loss | `lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py` | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `00e0cbc796d83ce9137f95f853d6262cac4a464782540ecd05276927267c8be1` |
| trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `af8baa9a4aac7f0de19caa55f43e6120010e7d6765e0dceaa7cb18e95a88888f` |
| verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `43142be57b105bacf90124223c67d93372482ae0eeb64f4e9a8658f5a951909e` |
| executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `94cbe45f290f92a2a5ffaf7e87063e78e1aec17ba8d4fcae9e799e2235374246` |
| synthetic proof | `lewm/tests/n5_hierarchical_first_hit_v9_synthetic_execution.py` | `fd12a7dd1d877e507a0d332e4d96e684cc989fe0242fe1ee6ac61598d5702d3e` |
| loss tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `5bb9e1c31e26ef4d4490013b9d377db161fa5ecde7471d4fa9ca4eb44a6a227b` |
| lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_lifecycle.py` | `d7a7048d2242be98aec9f7e2d66d4121d0e5f67e65c9d51292c08b311e7053ee` |
| handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_implementation_handoff_2026-07-13.md` | `50e22a56d2cb49e3b449aa760883c22dec1521abbd0d1b43fdbd0a69c5f374f2` |

All bindings were rehashed after testing and matched the amendment. All nine files are ASCII.

## Scientific review

The new loss implements the approved two-level hierarchical likelihood from first principles. Ordered hit and no-hit probabilities are normalized in log space. Presence loss gives equal weight to represented hit and no-hit target groups. Conditional distance loss renormalizes within hit mass and gives equal weight to represented target hit bins. Empty groups retain a differentiable zero and all normalization and finiteness preconditions are checked.

Independent arithmetic reproduced the implementation on hand-constructed states and showed that the retired per-ray group weighting yields a different value. The loss is invariant to batch/ray replication and target ordering. Logits at `+/-10000` remain finite and differentiable with finite gradients.

The trainer creates one fresh base Camera V4 model and never loads a prior state. The reviewed contract is AdamW, learning rate `1e-4`, weight decay `1e-4`, 4000 updates, five frames per update, gradient clip 1, 41 fixed diagnostic points, and final-update-only checkpoint selection. The four top-level losses have exact weight `0.25`. Matched and cyclic wrong-RGB panels are both evaluated.

An independent `torch.randperm` reconstruction reproduced the complete 4000-update schedule and its SHA-256 `fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`: 20,000 frame exposures, every one of five frames used exactly once per update. Mutations to update count, schedule, or checkpoint selection are rejected.

The retained model, evaluation panel, raster contract, frozen gate, and threshold families remain bound. Independent synthetic evaluation exercised six distance checks, five family checks, and three class checks and reproduced the exact 26-check gate.

## Execution and lifecycle review

The V9 executor is a mechanical V8 lifecycle successor outside the four reviewed V9 adaptation functions: reservation core, frozen training adapter, verifier adapter, and CPU smoke. The synthetic lifecycle proof is AST-identical to V8 after version/path normalization. Trainer and verifier differences from the retained V1 implementation are limited to the approved loss, metadata, evaluation, and compute-only verifier changes.

The verifier loads only the bound V9 checkpoint inside the isolated child, reconstructs and strictly validates metadata/state bindings, uses a fresh base model with strict state loading, and recomputes evaluation, accumulator, and frozen gate. It has no direct publication authority and no in-process fallback.

The executor uses an isolated `-I -B` child, a sanitized environment, GPU 0 binding only for an authorized exact attempt, and removes `HSA_OVERRIDE_GFX_VERSION`. Request/response nonce and source/artifact bindings, timeout, stderr, nonzero exit, and extra-output rejection are present. Parent-only metric publication, journal poisoning, descriptor-retained filesystem authority, no-follow traversal, owned-directory cleanup, and terminal single-use failure behavior are covered by retained and independent adversarial tests.

## Executed QA

CPU-only trainer contract smoke: PASS. CPU-only executor contract smoke: PASS. Both reported 4000 updates, 20,000 exposures, the schedule hash above, 41 diagnostic points, final-only checkpoint selection, and the four new loss names with total `0.265` from exact `0.25` weights.

Author suites plus independent QA: **157 passed in 5.82 seconds** with GPU visibility empty, BLAS/OpenMP threads capped at one, plugin autoload disabled, and the system ROS Python path supplied. Independent QA alone: **11 passed in 0.79 seconds**.

Nine Python files compiled successfully with `PYTHONPYCACHEPREFIX` directed to `/tmp`. `pyflakes` and `ruff` were not installed; no dependency installation was performed. AST comparison, source scans, import execution, and the 157-test suite provide the static and executed evidence used for this verdict.

Reviewer-owned QA: `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_independent_qa.py`, SHA-256 `8efaaecc2cea0815b31dc883b179d39e65bbd59337c5c9607ca02b2a9ed31119`.

## Authorization conclusion

The frozen Camera V9 source closure satisfies the amendment and is approved for one exclusive fresh exact attempt. This review does not authorize retry, infrastructure replacement, navigation, production, or promotion. Exact execution remains subject to the canonical policy preflight and all bound authority checks.
