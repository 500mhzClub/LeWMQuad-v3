# Go2 N32 camera-pose projection audit result

Date: 2026-07-11

Status: completed; orders the next N32 experiment but cannot pass N32, G2, or
runtime promotion.

## Provenance

- result path: `.generated/go2_n32_pose_projection_audit/v1/result.json`
- result file SHA-256:
  `2c7efba897054ea0067db58f020e70dc5f3c5804785c74cbda4a8b76e0210b9d`
- canonical content SHA-256:
  `6a9d05a0fb92289334cf39bb6947a2022a05a7c1892e8bb1c5a7156f9ca227f4`
- V2 implementation manifest SHA-256:
  `62375f9116843418e3812078ea23a8ed870a6bec0e4fe42580427c411d5df3bf`
- sole authorization / role-namespace amendment SHA-256:
  `ae17eb856c5329e8c5dfa5e4339306ef19e60c53c5f67d43746b268be9cc3370`
- runner source SHA-256:
  `3b422299eaa8d81c2397301e4981ce92a0c60731c39179d52bf7645868b674a3`
- pure geometry source SHA-256:
  `8835fbecc798c1cc3dd7a17b07821677a893854e1a1af0c3073c1bded9a07ac6`

The canonical content hash was independently recomputed after publication and
matched. The first V1 attempt had failed closed on an ambiguous legacy/current
role name and wrote no result. The V2 role amendment retained the original
balanced current physical-training panel before this result.

## Scope and access

The audit matched exactly 320 unique pose records, 64 per registered family,
from 160 current `physical_dataset_role=train` transitions. Legacy source split
counts were `train=244`, `test_hard=14`, `test_id=32`, and `val=30`; they were
hash-bound provenance only and did not alter inclusion.

The exact access reconciliation passed. It records zero original-monolithic
panel, RGB, label-shard, checkpoint, model-output, G2, physical-nontrain, and
sealed-payload opens. This was camera/base pose metadata analysis, not model
evaluation.

## Frozen ordering result

The preregistered dynamic-pose decision required both:

1. rough-local-dynamics median per-frame p50 displacement `>= 0.5` token;
2. rough minus pooled-nonrough median `>= 0.25` token.

Observed:

| Quantity | Result | Threshold | Pass |
| --- | ---: | ---: | --- |
| rough median per-frame p50 | 0.2553285781 token | >= 0.5 | no |
| pooled non-rough median | 0.2814419708 token | diagnostic | n/a |
| rough minus non-rough | -0.0261133926 token | >= 0.25 | no |

Therefore `material_dynamic_pose_mismatch=false` and the immutable next
intervention is `explicit_hierarchical_output`.

Across all 320 frames, the pooled query displacement p50/p95 was
`0.25351/0.78578` token, `18.56%` of jointly valid queries moved at least half a
token, and validity flips were `0.8839%`. Individual frames can still have large
pose deviations: the aggregate maximum displacement was `18.17` tokens and the
largest outlier occurred outside the rough family. These are retained as
diagnostics and motivate later deployment-valid pose handling, but the frozen
rough-versus-nonrough test does not license dynamic projection as the next N32
change.

## Consequence

The next candidate separates the currently entangled decision into one KNOWN
versus UNKNOWN logit and one OCCUPIED versus FREE conditional logit, then
constructs analytic joint UNKNOWN/FREE/OCCUPIED log probabilities. It keeps the
width-24 encoder/token projection, width-64 context, registered fixed
projection, factorization, full-ray dilations, data, initialization comparison,
exposure, optimizer schedule, and gates unchanged.

No holdout, second seed, full training, calibration, G2, shared-JEPA, or runtime
work is licensed unless this candidate passes the same complete N32 ordering
ladder.
