# Shared JEPA V5 raw-supervision builder V1 independent review

Date: 2026-07-13

Reviewer: `/root/camera_v5_independent`

Verdict: **BLOCK**

The frozen V1 candidate was reviewed without modification. Its synthetic
construction, reviewed-V4 conversion, deterministic layout, and retained-parent
publication checks pass, but its exact authorization validator opens a
caller-selected repository path before the authority source map is known to be
structurally complete. That violates the preregistered no-payload-before-valid-
authority boundary. No exact build, exact source payload, RGB, legacy label,
G2 payload, checkpoint, held-out input, accelerator, or canonical output was
opened.

## Frozen candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v1.py` | `3bc1559776e2f8471bb6a7a1ddd8808b1f1224687dedf280fd2300820afe25ec` |
| `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v1.py` | `df5fd60b50ba852d44fd6fe0034c7e763fc08030875488be3850e774906ceeb3` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v1.py` | `15767446ba45851a7f5774560db8e8f6f87d831a51fde7585acffa028f3ba2e4` |
| author handoff | `9d9aee5f636069d8beef2362bcc43b9be0063207d9ffe17d9045f99e3c30d28c` |
| independent BLOCK reproducer | `306b02e9ebaec7eb4a0649e65bff203582a9dba99a43d708c9adfd962d332104` |

All frozen metadata V5 parents and reviewed V4 source hashes also rehashed
exactly.

## Blocking finding

`_validate_authorization_payload()` validates each `source_map` row and
immediately calls `_read_bound_regular_file()` for that row. Only after the
loop does it compare the observed role set with the exact required nine roles.

The independent reproducer supplied a canonical, self-hashed authorization
with both PASS flags but only one source-map row:

```text
role: builder_source
path: arbitrary/referenced_frames.jsonl
sha256: 1111...1111
```

The candidate called the bound-file opener for that arbitrary path and only
then raised `RawSupervisionBuildError` because the other eight roles were
missing. The authority is structurally invalid, yet a caller-selected
repository file has already crossed the source-opening boundary. The path
could name metadata or a referenced source payload rather than a review file.

This is not the allowed rehash of hard-coded frozen parent sources. It is an
open controlled by an authority record that has not passed its own complete
structural validation.

## Required successor

An additive successor must use two phases:

1. validate the complete authorization object, content hash, exact nine-role
   set, uniqueness, strict role-to-path policy, relative canonical paths, and
   all cross-bindings without opening any source-map target; and
2. only after phase one succeeds, open and rehash the nine frozen review/source
   targets through the descriptor-bound reader.

The independent regression must instrument every metadata, source-map, and
referenced-source opener and prove zero calls for absent, malformed, duplicate,
missing-role, extra-role, wrong-role/path, and wrong-cross-binding authority
records.

## Passing evidence

The following parts passed and can be retained by an additive successor:

- absent authorization rejects before metadata and source openers;
- metadata V5 exact identities remain 5,172 pairs, 10,344 references, 9,460
  unique endpoints, and 88 scenes with the five frozen inventory hashes;
- the worker uses the reviewed V4 evidence and raster functions once per
  scheduled unique endpoint and rejects duplicate/orphan joins;
- the eight arrays include scalar `[N]` ground plane storage and the reviewed
  `64x64` three-state raster;
- the exact ledger fixes 9,460 raycasts, 10,344 references, and zero RGB,
  parent-label, G2-payload, held-out, model, runtime, hardware, and production
  opens;
- the author one-worker/six-worker test is byte-identical;
- the source loader retains full-RPY object parity and does not dereference RGB
  or parent label paths;
- publication uses a retained no-follow parent descriptor and
  `renameat2(RENAME_NOREPLACE)`; and
- occupied destinations, replaced staging, foreign cleanup targets, and
  existing failure receipts are preserved.

## Verification

All commands disabled external pytest plugins, capped the four native thread
families to one, and hid CUDA, HIP, ROCr, and GPU ordinal visibility.

```text
author focused builder suite:                    15 passed in 1.24s
author V4 + metadata V5 matrix:                 103 passed, 2 deselected
independent BLOCK suite:                          8 passed, 1 failed
py_compile (candidate plus independent test):   PASS
git diff --check:                                PASS
```

The two matrix deselections are the documented stale predecessor assertions
about the now-authorized V4 implementation manifest. An unchanged run
reproduced `103 passed, 2 failed`; the explicitly deselected run reproduced
`103 passed, 2 deselected`.

The canonical exact output
`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`
remains absent. This BLOCK grants no exact build, dataset use, training,
selection, calibration, G2, held-out, runtime, hardware, navigation,
production, or promotion authority.
