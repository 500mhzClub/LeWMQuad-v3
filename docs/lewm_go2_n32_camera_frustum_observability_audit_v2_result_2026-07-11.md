# Go2 N32 camera-frustum observability audit v2 result

Date: 2026-07-11

Status: independently finalized authoritative fit-only negative. This result
authorizes no camera-frustum representation, model output, G2, holdout,
runtime, or promotion access.

## Frozen artifacts

- execution binding SHA-256:
  `c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9`
- human implementation manifest SHA-256:
  `ff8ee60755f8355de756cd2cf710a791569e98d11f2a721557d28e0af2ed08b4`
- machine implementation manifest SHA-256:
  `93b59cc38338857f01160b1cc048071ab7f32d0d2cfc2ab0a30b64a0e5a40380`
- machine canonical content SHA-256:
  `5811cc732ed4a0af53f70e099721c5d1854f49ccd8891cce8570ff9231ab70df`
- runner result:
  `.generated/go2_n32_camera_frustum_observability_audit/v2/result.json`
- runner result file SHA-256:
  `7725ecddf2fa77bb762733fd35df2efd2fb60d4f9aa8ab6fdf2bee660522909e`
- runner result canonical content SHA-256:
  `11420607d2c4f8e79af9214d43bbc6259669ee84c9ccc0aaefd4167cc1d809a1`
- finalizer stdout evidence SHA-256:
  `adcde9063a5cb5853adf401f3d01110c3a7cf455abf5b5f10c3550d433deb784`

The runner result is immutable. Future work must bind this dated report and
must not use the generated result as a model-selection or training input.

## Final decision

The independent finalizer reproduced the runner decision:

| Gate | Result |
| --- | --- |
| source hashes | pass |
| provenance | pass |
| exact target reconstruction | pass |
| access reconciliation | pass |
| camera-centered mapping | pass |
| rendered/collision ambiguity | false |
| fit known-target coverage | **fail** |
| camera-frustum implementation authorized | **false** |

All training, G2, holdout, seed-20260711, runtime, and promotion licenses
remain false.

## Observability result

The proposed camera-centered map is deterministic and injective:

- 1,990 supported Cartesian cells mapped to 1,990 unique polar bins;
- 0 collisions, partial mappings, out-of-range indices, or support mismatches;
- 0 target-reconstruction mismatches across 320 frames;
- all 320 full-quaternion camera-mount checks pass;
- no rendered/collision target ambiguity.

The old whole-body local target is not fully observable from one current
forward-facing camera frame:

| Class | Supported | Total | Unsupported |
| --- | ---: | ---: | ---: |
| FREE | 118,792 | 118,793 | 1 |
| OCCUPIED | 9,856 | 10,228 | 372 |
| UNKNOWN | 508,152 | 1,181,699 | 673,547 |

Unsupported UNKNOWN is expected and permitted. The frozen gate requires every
supervised FREE and OCCUPIED target to be inside current-camera support, so the
1 unsupported FREE and 372 unsupported OCCUPIED cells fail it. Every
registered family contains at least one unsupported known target.

This is not a capacity or optimizer failure. It also does not prove that the
observable-physical-v3 labels are privileged or invisible. Those labels use
visible ground samples and first-surface obstacle witnesses. A visible witness
can occupy part of a 0.10 m output cell even when that cell's center lies
outside the proposed center-point support. The failed object is therefore the
preregistered Cartesian-center-to-one-polar-bin decoder geometry, not the
audited target.

## Access result

The three-phase preparation, runner, and finalizer reconciliation passes:

- 20 exact label shards opened once per label-reading phase;
- 320 selected label rows and 320 supervision rows read per phase;
- 960,000 committed source JSONL records scanned per phase;
- 103 unique source-geometry paths reconciled;
- RGB/image, checkpoint/model output, G2, holdout, physical-nontrain, runtime,
  seed-20260711, and sealed accesses: 0;
- denied and unexpected path attempts: 0.

## Required successor

The observable-physical-v3 target and observation-versus-memory separation
remain the correct semantics. The next intervention is a new geometry/decoder
binding:

1. **Current-frame physical evidence:** keep the calibrated full-rectilinear-
   frustum, visible-ground, and conservative first-surface obstacle target with
   zero configuration inflation.
2. **Cell-footprint or ray-witness support:** replace one body-cell center to
   one camera-polar-bin gathering with a source-grounded construction that can
   represent every output cell receiving visible evidence. Candidate geometry
   must be defined from closed cell footprints and/or camera-ray hit support,
   not tuned to V4 errors.
3. **Persistent online memory:** transform and fuse admitted observations over
   time with explicit pose, uncertainty, age, contradiction, and observation
   provenance.
4. **Configuration map:** derive robot-sized free/blocked state only after
   physical multi-view fusion using the frozen asymmetric 0.47 m morphology.

A dated geometry binding must freeze the camera-to-output support relation,
decoder aggregation, injectivity/collision alternative, label compatibility
proof, camera calibration, hashes, N32 ladder, and access boundary before any
successor model output. If a proposed geometry still cannot support every fit
FREE/OCCUPIED label, it fails before training. The existing target may be
amended only if separate source-grounded ambiguity is proved; this audit found
none.
