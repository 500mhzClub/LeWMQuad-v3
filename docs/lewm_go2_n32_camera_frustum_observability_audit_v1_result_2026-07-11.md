# Go2 N32 camera-frustum observability audit v1 result

Date: 2026-07-11

Status: immutable runner result; scientifically negative and non-authoritative
because the independent finalizer rejected one exact floating-point evidence
value. This result authorizes no representation implementation, training, G2,
holdout, runtime, or promotion access.

## Frozen artifacts

- execution binding SHA-256:
  `96eb4b9eb11b0924056ffb89590ecf13bb20ffcc72c6aca5e6cb51e92bb8132e`
- human implementation manifest SHA-256:
  `25dbb153a72aac3d9eeaf899d7582216d78d2839ddb2edf68f67271d217001bd`
- machine implementation manifest SHA-256:
  `9350fe37c6953adb48793abe275cd0b7fbaa713b3390829d5ad7829e5882b47e`
- runner result:
  `.generated/go2_n32_camera_frustum_observability_audit/v1/result.json`
- runner result file SHA-256:
  `6ec4ac68828f6b7b8928489267951f78f30533ca52458ed59526596ce107d223`
- runner result canonical content SHA-256:
  `41c82454f5328bf19fc82428fbffb5a48bb03a2d11479a015248c51587b4c3c7`

The result is immutable. It must not be edited, replaced, treated as finalized,
or used to authorize a model.

## Runner outcome

The fit-only runner completed successfully over 320 canonical train-role
frames and 20 committed label shards. It reported:

- source hashes, provenance, access reconciliation, target reconstruction,
  camera-centered mapping, and all 320 fixed-camera-mount composition checks:
  pass;
- reconstruction mismatches: 0 cells in 0 frames;
- mapping collisions: 0;
- rendered/collision target ambiguity: false;
- denied or unexpected accesses: 0;
- RGB decodes/byte opens, model/checkpoint/output opens, G2, holdout,
  physical-nontrain, runtime, seed-20260711, and sealed opens: 0.

The representation decision was false solely because the frozen
known-target-coverage gate failed. Of the old body-local supervised targets:

| Class | Supported | Total | Unsupported |
| --- | ---: | ---: | ---: |
| FREE | 118,792 | 118,793 | 1 |
| OCCUPIED | 9,856 | 10,228 | 372 |
| UNKNOWN | 508,152 | 1,181,699 | 673,547 |

The gate permits unsupported UNKNOWN cells but requires every FREE and
OCCUPIED target to lie in the proposed camera-centered support. It therefore
failed. All five registered families contained unsupported known targets:

| Family | Unsupported FREE | Unsupported OCCUPIED |
| --- | ---: | ---: |
| open obstacle field | 1 | 14 |
| rough local dynamics | 0 | 96 |
| small enclosed maze | 0 | 86 |
| medium enclosed maze | 0 | 99 |
| large enclosed maze | 0 | 77 |

This is a first-principles result: a current forward camera view cannot be
trained to reproduce every known cell in a body-local target that extends
outside its geometric frustum. The online memory may retain such cells, but a
current-frame visual observation head requires a frustum-scoped target or an
explicit observation-versus-memory factorization.

## Finalizer rejection

The independent standard-library finalizer rejected:

```text
FinalizationError: frame_reports[13].camera_mount_composition differs from
independent full-quaternion composition
```

The only differing values were the quaternion norm and its residual:

| Value | Runner | Finalizer |
| --- | ---: | ---: |
| quaternion norm | 1.0000000510408722 | 1.0000000510408724 |
| absolute residual | 5.104087219542919e-08 | 5.104087241747379e-08 |

Python 3.12's `sum()` accumulation in the finalizer differed by one ULP from
the runner's frozen explicit expression
`sqrt(qx*qx + qy*qy + qz*qz + qw*qw)`. Both values are far inside the
`1e-5` semantic tolerance, but the result correctly remained non-authoritative
because compact evidence is compared exactly.

The narrow implementation correction is to use the same explicit four-term
expression in the independent finalizer and retain `sum()` for the
three-vector norms whose runner computation also uses `sum()`. The reproduced
quaternion is now an exact runner/finalizer compatibility fixture.

## Consequence

A fresh binding and source freeze must move the immutable output to v2, record
this v1 result and finalizer failure, and rerun preparation, runner, and
finalizer from zero. The v2 audit is expected to preserve the scientific
coverage failure; its purpose is to make that negative independently
authoritative. No successor representation or model output is authorized
until v2 finalization completes.
