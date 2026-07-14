# N5 full-panel V6 lifecycle-recovery amendment (2026-07-13)

## Status and scope

This document is an additive, pre-implementation amendment to the frozen N5
full-panel experiment. It authorizes source construction and independent review
of one V6 infrastructure-replacement lifecycle. It does not itself authorize
execution. Exact execution remains forbidden until a different-agent source
review passes the complete frozen V6 source closure.

The V5 scientific attempt is consumed and terminal. V6 is not a numeric or
scientific retry. It is one infrastructure replacement for a run that completed
training but failed closed before publishing any numeric result or checkpoint.
No V5 numeric result or checkpoint survived, was admitted, or was inspected.
Nothing learned from V5 training may select, tune, calibrate, or otherwise alter
V6.

## Bound V5 terminal evidence

The only V5 exact-attempt payloads admitted by this amendment are the lifecycle
receipts below:

- reservation path:
  `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/seed_20260710/n5/reservation.json`
- reservation file SHA-256:
  `f8062f2ed2bdb1589ca806fb9331ce7f1ec0675d4466e96c0a78530080ea501a`
- reservation content SHA-256:
  `1427a5524cbc7e72ac24d78c221775bab3c943d36967b88df6e780743faafc15`
- reservation byte count: `4532`
- failure path:
  `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/seed_20260710/n5/failed.json`
- failure file SHA-256:
  `7ead760085f5365ac83ebfc8875910cbc076437fa972d48d008aa3b2127e50af`
- failure content SHA-256:
  `84cfa81aa2db9fa7cd7233e314e7d3da50b4fc23af863ab38e9ab948ac51358b`
- failure byte count: `802`

The V5 failure receipt is authoritative for terminal state: `status=failed`,
`failure_stage=training`, `retry_authorized=false`, no owned partial artifacts
required cleanup, and every later-rung/runtime/promotion license is false. The
absence of `checkpoint.pt`, `result.json`, `completed.json`, metric verification,
and gate artifacts is part of the recovery premise.

## Infrastructure diagnosis

V5 retained full directory metadata, including timestamps and size, for every
ancestor from the filesystem root through the claimed output. During training,
a concurrent retained test created and removed a private temporary directory
directly under the shared repository `.generated` directory. That unrelated
operation changed `.generated` metadata at
`2026-07-13 13:20:26.803623735 +0100`. V5 checked the retained chain before its
first success write at approximately `13:20:28` and correctly failed closed.
The V5 output subtree itself was not replaced or mutated by that operation.

V6 changes only the lifecycle ancestry predicate. It must retain component-wise
`O_NOFOLLOW` directory descriptors and verify stable directory identity and
security attributes for every ancestor. It must reject symlinks, rename/alias
swaps, inode replacement, type changes, owner/group changes, and permission
changes. For shared ancestors above the V6-owned exclusive subtree, it must not
bind directory link count, size, modification time, or change time, because
unrelated direct child creation/deletion legitimately changes those fields.
At and below the V6-owned exclusive root, full metadata remains bound and only
explicit executor-owned mutations may refresh it.

## Frozen science

V6 must preserve the V5/V1 numerical experiment exactly:

- seed `20260710`;
- fit size `N=5` using the same five frozen train frames and target partition;
- fresh model initialization, with no V5 state or checkpoint input;
- `ObservableCameraRayEvidenceV4Model`;
- AdamW, learning rate `1e-4`, weight decay `1e-4`;
- 400 optimizer updates, batch size 5, and 2,000 frame exposures;
- schedule SHA-256
  `62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`;
- float32, autocast disabled, gradient clipping norm `1.0`;
- four frozen loss components weighted `0.25` each;
- final-update-only checkpoint selection;
- matched-RGB and wrong-RGB-with-target-calibration evaluations;
- the unchanged independent metric verifier and final gate;
- GPU0 only: AMD Radeon AI PRO R9700;
- Raphael iGPU forbidden, and native math threads fixed to one per process.

No hyperparameter, frame, target, metric, gate, calibration, threshold, model,
evaluation control, or hardware role may change.

## One-use V6 lifecycle

The new canonical output root is frozen as:

`.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v6`

This namespace must be absent before the sole exact V6 claim. The executor may
create it only as part of its one isolated end-to-end operation. It must use a
unique private staging directory, process-death-safe lock, atomic no-replace
claim, retained claimed-directory descriptor, immediate parent fsync, exclusive
artifact publication, self-hashed terminal receipts, and owned-artifact cleanup
before failure terminalization. No caller-controlled production path, partial
stage entry point, lifecycle token, mutable registry, or reusable capability is
permitted.

V6 may perform exactly one fresh infrastructure-replacement attempt after a
different-agent PASS review. It may not read any V5 numeric payload. It may not
retry V6, run a second seed, run N16, inspect holdout or G2 data, select or tune
from later-rung evidence, change calibration, enter runtime/hardware/production,
or promote a checkpoint. All such licenses remain false regardless of V6
outcome.

## Required adversarial evidence

Before review, V6 must include tests proving:

1. concurrent direct create/delete churn under the shared `.generated`
   ancestor does not invalidate an otherwise owned V6 claim;
2. symlink, rename, restored-alias, inode replacement, permission, and ownership
   changes to ancestor or owned components are rejected;
3. the exclusive V6 subtree still detects unowned direct-child mutation;
4. V1-V5 security, source, science, schedule, GPU0, single-use, cleanup, and
   terminalization regressions remain passing;
5. imports and CPU contract smokes open no production data/output and cannot
   execute the exact attempt.

The source author must freeze the V6 policy, executor, production-ineligible
synthetic model, author tests, and implementation handoff by SHA-256. A different
agent must review those exact bytes. No exact execution or numeric-payload
inspection is authorized during implementation or review.
