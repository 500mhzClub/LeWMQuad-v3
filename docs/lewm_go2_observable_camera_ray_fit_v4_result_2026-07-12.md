# Observable camera-ray fit V4 exact result

Date: 2026-07-12  
Status: exact train-fit dataset build and independent audit complete; model,
G2, held-out, runtime, and promotion remain unauthorized

## Scope and verdict

The reviewed V4 builder completed the exact fit-only build, and the separate
auditor completed its exact comparison against the registered legacy
physical-v3 train labels.

The immutable result covers:

- 320 registered train-role endpoint frames;
- 20 scene shards;
- 4,096 raster cells per frame;
- 1,310,720 audited cells in total;
- 181 dataset files with a 66M on-disk footprint.

The 181-file count excludes the audit receipt written after dataset
publication. The files have 68,008,459 logical bytes; filesystem allocation is
reported as 66M. Internal hash and raster determinism checks passed.

This is a target-construction and fit-dataset result. No learned model was
trained or evaluated, and it is not evidence of G2 or held-out generalization,
runtime readiness, beacon claiming, or promotion.

## Immutable authorization and results

| Artifact | File SHA-256 | Canonical content SHA-256 |
| --- | --- | --- |
| Reviewed implementation authorization, `docs/lewm_go2_observable_camera_ray_fit_v4_implementation_manifest_2026-07-12.json` | `aa882ae7cc7b038028acf73e4addc049e030a7d3fe7fd1ceb0ff9ded1e464e0e` | `17440ae679d1e730f8f37b2fe62de9bef5029e69198b4969d6ff8990bd38d90b` |
| Dataset manifest, `.generated/go2_observable_camera_ray_fit_v4/v1/manifest.json` | `2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85` | `9be0c1539897bd731d4dfaf96e03b5d5c1d31d8cb8c723a2b77ffde57baf2812` |
| Audit result, `.generated/go2_observable_camera_ray_fit_v4/v1/audit_result.json` | `2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c` | `a922114b7e42552043a487bae527c35fb511804d4e8683c5a3f64a2bf499cf76` |

The reviewed implementation source-map SHA-256 is:

`a22989bcd64c2e79fbb2a06743622fd7ca14332d0b4715f2c982af083a2061bd`

The three file hashes were recomputed from the files. Each canonical content
hash was independently recomputed from the canonical JSON object after
removing its top-level `content_sha256` field. The source-map hash was
independently recomputed over its canonical ordered entry list. Every
recomputed value exactly matched the declared value.

## Exact legacy comparison

V4 intentionally asks what current calibrated camera evidence supports. The
legacy physical-v3 target also encoded a finite physical-world boundary that
is not visible in the current camera observation. Therefore equality with the
legacy target is diagnostic, not the definition of V4 correctness.

| Class | Legacy physical-v3 | V4 observable-camera raster | Difference |
| --- | ---: | ---: | ---: |
| UNKNOWN | 1,181,699 | 1,072,012 | -109,687 |
| FREE | 118,793 | 228,477 | +109,684 |
| OCCUPIED | 10,228 | 10,231 | +3 |
| Total | 1,310,720 | 1,310,720 | 0 |

There are 109,687 differing cells across 284 of 320 frames. With legacy labels
as rows and V4 labels as columns, the complete confusion matrix is:

| Legacy / V4 | UNKNOWN | FREE | OCCUPIED |
| --- | ---: | ---: | ---: |
| UNKNOWN | 1,072,012 | 109,684 | 3 |
| FREE | 0 | 118,793 | 0 |
| OCCUPIED | 0 | 0 | 10,228 |

V4 therefore covers every legacy FREE cell and every legacy OCCUPIED cell
exactly. All differences start from legacy UNKNOWN: 109,684 become V4 FREE and
three become V4 OCCUPIED. V4 does this intentionally because it accepts
current camera-supported free ground and visible first-surface hits without
reintroducing the invisible finite-boundary prior. Repairing these differences
with physical-free, collision, or global scene-boundary information would
violate the V4 visual-evidence contract.

## Access closure

The build selected exactly 320 authorized train frames. It opened no RGB bytes
and no fit-label payloads. The independent auditor subsequently opened the 20
authorized train-fit label shards and selected exactly 320 label and
supervision rows for the diagnostic comparison. That access was explicitly
authorized and was not a G2 or held-out evaluation.

Both build and audit receipts report zero for every forbidden access class,
including:

- RGB byte opens and RGB decodes;
- G2 and held-out label or geometry opens;
- sealed, runtime, and seed-20260711 opens;
- checkpoint-selection or probability-calibration opens;
- model checkpoint or model-output opens;
- physical non-train role opens;
- denied attempts and unexpected path attempts;
- derivative shard or cache writes.

The auditor also verified the manifest-bound RGB receipt by metadata join
without opening RGB. All G2, held-out, model-output, runtime, and promotion
license booleans remain false.

## Resource policy

The exact build used the reviewed CPU-only execution contract: a bounded pool
of six spawned workers, one native numerical thread per worker, canonical
merge order, and no GPU. The audit receipt independently records
`gpu_used=false`.

GPU 1 remains forbidden for model training and inference. It is the
approximately 2 GB Raphael integrated GPU and has already failed the N32 V4
smoke with `hipErrorInvalidDeviceFunction`. Under the execution resource
policy it may be used only for non-data hardware identification or an
explicitly isolated backend-compatibility diagnostic. Any later substantial
JEPA or learned-head training is authorized only on GPU 0 and only after the
corresponding stage authorization.

## Remaining work and authorization boundary

Only the V4 fit dataset and its independent exact audit are complete. The
following stages remain outside this result and unauthorized:

1. Training or selecting the observable-camera evidence model.
2. Reading RGB bytes for an authorized trainer.
3. G2 evaluation or any held-out evaluation.
4. Runtime integration, persistent-memory evaluation, or planner evaluation.
5. Checkpoint promotion or any sealed evaluation.

The next stage must obtain its own reviewed authorization and bind to the exact
implementation, dataset, and audit hashes above. This result alone grants no
model, G2, held-out, runtime, or promotion license.

## 2026-07-13 N5 preflight incident

A later narrowly authorized N5 development-fit invocation did not start an
attempt. It stopped while validating the frozen dataset-manifest metadata
because the trainer contained `...863c040...` where the builder,
implementation manifest, and immutable dataset manifest contain
`...863d040...` in the source-geometry manifest SHA-256.

Only authorization/source metadata and the dataset manifest were inspected.
No audit or target-shard payload, RGB, checkpoint, result, metric receipt, GPU,
model inference, or training operation was reached. No reservation or attempt
directory exists, so this incident produced no fit result and consumed no
ladder attempt.

The candidate remediation changes the trainer commitment to the exact frozen
value and adds a pre-reservation provenance regression. It does not widen the
reviewed development-fit or metric licenses and does not authorize G2,
held-out, runtime, promotion, or aggregation. N5 remains unlaunched pending a
different-agent byte review of the regenerated source binding.

## Superseding V1 terminal record and V2 successor

The prior paragraph describes the provenance-preflight incident only. After
that repair was reviewed, V1 N5 was reserved and later terminated because the
warning parser rejected PyTorch's exact `Context.cpp:157` source-location
trailer on an otherwise allowlisted deterministic-kernel warning.

The immutable V1 terminal record is reservation
`115e3a4e0ad7db7f5bd6b01c7ddde29d79563600ffb84ef77a0c585f009e854e`
(content `ca458f9371a211017f1b7a710b41508e2219a1afe19516ace2553a8eaa4d15dd`)
and failure
`6eb1becc195165e5fb49c1d222cac301f4169f301a48245d23a2b8213363af48`
(content `7c1fe8f1ea73d8caef33debd9076bc3ddcacfaf337ec2a0000cec64f678c21e4`).
No V1 checkpoint, result, completion, gate, or metric receipt exists.

The frozen ladder-v3 amendment creates a separate `development_fit_v2`
successor with fresh one-attempt budgets and an exact V1-failure lineage in
every reservation. It changes no rung, seed, step, data, target, model,
threshold, or license. V2 N5 remains unlaunched pending different-agent review
of the 43-entry source map
`eb8c97dae6f3ef3839a886cac200774c87dfb6e452f71c13e75557eb8c9feac3`.
