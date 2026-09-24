# Camera-ray N5 hierarchical-first-hit V9 terminal verifier failure diagnosis

Date: 2026-07-14

Author: `/root`

Status: **read-only terminal diagnosis; no retry or execution authority**

## Exact outcome

The sole reviewed Camera V9 attempt completed its 4,000 optimizer updates and
20,000 five-frame exposures on the discrete AMD Radeon AI PRO R9700. The
fresh isolated verifier subprocess then returned nonzero. The lifecycle owner
refused fallback, removed its owned `checkpoint.pt`, `result.json`, and
`completed.json`, and published a terminal failure receipt.

Only these attempt artifacts remain:

| Role | Path | File SHA-256 | Content SHA-256 |
| --- | --- | --- | --- |
| reservation | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v9/attempts/seed_20260710/n5/reservation.json` | `184628c4518f0a3e7411561ee7f9ed83da1f89c9af7d729ed3e6ffe76ce0f1a2` | `1ad75999d8d88e9fa3599bec97fe9b18c2d8b893c372cb263dd8a0fa748449e0` |
| terminal failure | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v9/attempts/seed_20260710/n5/failed.json` | `285c7bf38975a1ca13063d7b7ca36b31aa1b966cd206e0a418c07198c0719a3a` | `1c22542a02e9e2707872df36c72bc790ca5fe06e57b0e03c63c40c2f6c2ebf7a` |

The receipt fixes `failure_stage=verification`, `failure.class=runtime`,
`failure.code=execution_failure`, intact owned-directory journal integrity,
successful cleanup, and `retry_authorized=false`. No metric receipt, gate,
checkpoint, result, completion, or numerical score survives.

## Frozen source and review

| Role | Path | SHA-256 |
| --- | --- | --- |
| V9 amendment | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_preimplementation_amendment_2026-07-13.md` | `ccc8097b4d3bd70aabf3c701226928e360fafb04a12a452c4fd406e9bba3db0a` |
| loss | `lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py` | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `00e0cbc796d83ce9137f95f853d6262cac4a464782540ecd05276927267c8be1` |
| trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `af8baa9a4aac7f0de19caa55f43e6120010e7d6765e0dceaa7cb18e95a88888f` |
| verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `43142be57b105bacf90124223c67d93372482ae0eeb64f4e9a8658f5a951909e` |
| executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `94cbe45f290f92a2a5ffaf7e87063e78e1aec17ba8d4fcae9e799e2235374246` |
| synthetic support | `lewm/tests/n5_hierarchical_first_hit_v9_synthetic_execution.py` | `fd12a7dd1d877e507a0d332e4d96e684cc989fe0242fe1ee6ac61598d5702d3e` |
| loss tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9.py` | `5bb9e1c31e26ef4d4490013b9d377db161fa5ecde7471d4fa9ca4eb44a6a227b` |
| lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_lifecycle.py` | `d7a7048d2242be98aec9f7e2d66d4121d0e5f67e65c9d51292c08b311e7053ee` |
| author handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_implementation_handoff_2026-07-13.md` | `50e22a56d2cb49e3b449aa760883c22dec1521abbd0d1b43fdbd0a69c5f374f2` |
| independent review | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_independent_review_2026-07-13.json` | `20d5abd9327267c5e40a66b464fd6589d30704ee8be7b919cadfd52b30350016` |
| independent report | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_independent_source_review_report_2026-07-13.md` | `0e930ef2bd0d0753f4928c69a462de2c05bf13d3e62139a0079e12a66e815522` |
| independent QA | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_independent_qa.py` | `8efaaecc2cea0815b31dc883b179d39e65bbd59337c5c9607ca02b2a9ed31119` |

The review content SHA-256 is
`8d7edcefce04d85a042558aa7ccc638c8da8e0690fcc36d9cff15e99bc6a0347`.

## Proven diagnosis

The exact child exception is unrecoverable from the frozen artifacts:

1. the parent captured child stdout and stderr in memory;
2. on nonzero return it discarded the return code, signal interpretation,
   stdout, stderr, sizes, and hashes and raised a new generic `RuntimeError`;
3. terminalization reduced that exception to generic `execution_failure`;
4. terminalization deleted the only checkpoint/result/completion bytes before
   publishing the diagnostic-free failure receipt.

The real isolated child path was also absent from the reviewed test closure.
Lifecycle tests replaced `subprocess.run` with a fake for both success and
failure. They therefore did not jointly execute request parsing, exact bundle
validation, checkpoint reload, input reload, matched/wrong-RGB recomputation,
response serialization, and parent response validation in a fresh
`python -I -B` process. The exact attempt was the first such execution.

Two independent read-only postmortems reproduced all available CPU science
checks in genuine isolated processes:

- V9 checkpoint metadata, serialization, state manifest, and strict fresh-
  model load validate;
- canonical result JSON roundtrip validates;
- trainer and verifier evaluation are exactly equal on synthetic five-frame
  inputs after a strict state roundtrip; and
- every frozen source/proof hash remains unchanged.

These checks do not identify the exact runtime exception and do not establish
a numerical result. They only rule down a deterministic CPU loss/checkpoint
disagreement. Possible exact-only loci include bundle/input/resource
validation, GPU recomputation, exact equality after GPU reload, or response
construction. No OOM, GPU fault, signal, or numerical evidence was retained.

## Scientific interpretation

Camera V9 is a terminal infrastructure failure with an **unobserved numerical
outcome**. It is neither a camera-head pass nor a camera-head numerical fail.
Its checkpoint may not be reconstructed, reused, selected, promoted, or
treated as evidence. The V9 attempt may not be retried.

The evidence justifies only an additive lifecycle successor with fresh
training and unchanged science. It does not justify changing the model,
hierarchical loss, five-frame panel, seed, 4,000-update schedule, optimizer,
wrong-RGB control, metric implementation, or any of the 26 numeric thresholds.

## Required successor direction

A V10 lifecycle successor must:

- preserve the complete V9 scientific treatment and rerun fresh in a new
  one-attempt namespace;
- emit phase-tagged, bounded, canonical child error envelopes;
- preserve return code/signal, bounded stdout/stderr metadata, request and
  artifact hashes, and sanitized exception context in an immutable diagnostic;
- durably publish and fsync that diagnostic before owned scientific-artifact
  cleanup, then bind it from the final failure receipt;
- retain no tensor, RGB, prediction, metric, or other numerical payload in a
  failure diagnostic;
- keep success fail-closed with no fallback and no stderr; and
- pass a real fresh-process CPU synthetic success/failure protocol, not only a
  mocked subprocess test, before source review may authorize one fresh V10
  attempt.

This diagnosis grants no V10 source, exact, data, checkpoint, GPU, retry,
later-rung, training, selection, calibration, G2, held-out, runtime, hardware,
production, promotion, or deployment authority.
