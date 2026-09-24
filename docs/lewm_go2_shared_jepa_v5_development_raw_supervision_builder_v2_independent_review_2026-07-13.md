# Shared JEPA V5 raw-supervision builder V2 independent review

Date: 2026-07-13

Reviewer: `/root/camera_v5_independent`

Verdict: **BLOCK**

The frozen V2 candidate was reviewed without modification. Its literal
two-phase validator fixes the V1 early-open defect: malformed or structurally
invalid authority is rejected before any metadata, reviewed target, parent, or
referenced source opener. Its strict machine-review parser also correctly binds
canonical JSON, duplicate-key rejection, self-hash, PASS verdict, distinct
reviewer, fixed author, ordered candidate, and the exact narrow all-false
downstream authority map.

The candidate remains blocked because its imported execution surface still
contains three authority bypasses. It exposes the independently blocked V1
exact entry through `builder._v1`, accepts a caller-supplied worker function in
an authorization-named pool without validating the supplied authorization, and
temporarily replaces the process-global V1 authority validator with an
accepting callback inside every compatibility bridge. The production phase-two
validator also exposes caller-controlled reader/root/parent-rehash test seams.

No exact build, authorization, metadata plan, development frame, scene
manifest, render plan, render summary, RGB, legacy label, G2, held-out,
checkpoint, model output, accelerator, canonical dataset, or failure receipt
was opened or created during review.

## Frozen candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v2.py` | `0ae5ddd836802ced1fcf7524b67970247dccace6787fd0acc7268cbae4d3e71c` |
| `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v2.py` | `c11396874677c3cd3d0ef76353ea7de1449ef610d35f0b4256530a4f62b1d303` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2.py` | `6755044af535dc0c2de93f0f5bd79b01b140da33bc8ff2ec5b003ef592b50339` |
| author handoff | `7f278c5c24a8e9d89c6b0e3ecb9252acd0edec5729bd9fdde5d72231848bc04f` |
| independent BLOCK reproducer | `2c34fec949ea43e03b3f7f3c97b8d8ddba0aad1c9192dfd8b00d3f646dd03d43` |

The frozen V1 candidate and its independent BLOCK reproducer also rehash to
the identities bound by V2. The V2 source, CLI, author test, and handoff match
the hashes supplied for review exactly.

## Blocking findings

### 1. Imported V1 exact fallback and mutable authority callback

`go2_shared_jepa_v5_raw_supervision_builder_v2.py` retains the complete V1
module as `_v1`. A normal V2 import therefore exposes the callable
`builder._v1.execute_exact_build_v1`, even though V1 is independently blocked
and `execute_exact_build_v1` is merely omitted from V2's `__all__`.

More importantly, `_call_v1_load_exact_scene_job`,
`_call_v1_revalidate_exact_scene_sources`, and
`_call_v1_load_parent_contracts` each assign a local accepting callback to the
process-global `_v1._require_exact_authority`, call into V1, and restore the
original afterward. During that interval, any same-process V1 exact helper or
the exposed V1 exact entry sees the accepting callback rather than either the
V1 or V2 fixed-file validator. The callback returns the already parsed V2
authority mapping for the matching string digest.

The independent behavioral reproducer substituted only a synthetic V2 gate
result and a no-I/O V1 probe. Inside the bridge, the probe observed that the
global V1 validator had changed and that it returned the synthetic authority.
Restoration after the call worked, but restoration does not remove the
authority window.

This violates the required absence of mutable worker authority, V1 exact
fallback, callback bypass, and import bypass.

### 2. Caller callback runs without authority validation

`_run_authorized_scene_pool(function, argument_rows, workers,
authorization_sha256)` accepts an arbitrary caller function. The
`authorization_sha256` parameter is never read. The helper installs the worker
environment and submits the supplied function immediately.

The independent reproducer replaced only `ProcessPoolExecutor` with an inline
no-I/O executor and passed an invalid digest plus a marker callback. The marker
executed and the helper returned normally instead of rejecting authority. In
production, the same reachable helper can submit any picklable importable
function under the builder's worker process configuration.

This violates the requirement that every reachable worker path internally
revalidate the fixed authorization before caller-selected work can run.

### 3. Production phase two retains reader and parent-skip seams

`_validate_authorization_phase_two` is reachable after import and accepts a
caller `repository_root`, a caller `reader`, and
`rehash_frozen_parents=False`. A caller can construct a syntactically valid
fixed-role phase-one capsule from a mapping and have phase two accept synthetic
review/source bytes without reading the fixed repository targets or frozen
parents. The returned object is an accepted authority mapping.

The public exact entry does not directly accept that mapping, so this seam is
not by itself the exact-build exploit. It nevertheless contradicts the frozen
requirement that production have no caller-supplied authority mapping or
reader callback, and it compounds the mutable V1 callback bridge above. Test
injection must be isolated in a production-ineligible module rather than
parameters on the production validator.

## Passing evidence

The following requested properties passed and can be retained by a successor:

- phase one requires exactly six top-level fields and nine unique source rows
  in the frozen order, with exact role-to-literal-path mapping and canonical
  relative POSIX paths;
- malformed, missing, extra, duplicate-role, duplicate-path, wrong-order,
  wrong-role, wrong-path, noncanonical-path, malformed-entry, wrong-author,
  wrong-review-path, wrong-candidate, wrong-cross-binding, and wrong self-hash
  records reach zero target, metadata, parent, and referenced-source openers;
- a phase-two capsule is strict-JSON reparsed, phase-one revalidated, and
  compared by exact frozen dataclass equality before its first reader call;
- the two review records are duplicate-key-safe canonical JSON and require
  exact schema, PASS, reviewer, implementation author, ordered candidate,
  canonical self-hash, and source-only approval with all exact-build, audit,
  dataset-use, training, selection, calibration, G2, held-out, runtime,
  navigation, hardware, production, and promotion flags false;
- the main V2 exact path gates before output-container, metadata, inventory,
  parent, or development-source access;
- both fixed worker functions invoke a V2 bridge that revalidates the fixed
  authority in their spawned process before entering V1, although the bridge
  then creates the blocking global callback window;
- V2 retains the V1 construction engine and eight-array layout, a six-worker
  ceiling, 5,172 pairs, 10,344 endpoint uses, 9,460 unique endpoint raycasts,
  88 scene jobs, deterministic ordering, second source pass, retained-parent
  no-replace publication, and inode-owned cleanup; and
- the canonical output, terminal failure receipt, and exact authorization file
  remain absent.

## Verification

All commands disabled external pytest plugins, capped OMP, OpenBLAS, MKL, and
NumExpr threads at one, and hid CUDA, HIP, ROCr, and GPU ordinal visibility.

```text
V2 author authority/synthetic suite:                 27 passed
V1 + V2 + applicable V1 review closure:              50 passed, 1 deselected
V1 frozen BLOCK reproducer:                           8 passed, 1 failed
V2 independent BLOCK suite:                           9 passed, 4 failed
independently authored auditor V2 synthetic suite:   25 passed
two transient shared-/tmp ancestry failures:          rerun 2 passed
py_compile (candidate, author test, review test):      PASS
git diff --check (review test):                        PASS
```

The V1 deselection is its known frozen early-open BLOCK assertion. The four V2
failures are exactly the imported V1 fallback, unauthenticated caller callback,
mutable global V1 authority callback, and production phase-two injection seams
described above.

The first combined retained run had two unrelated failures because concurrent
pytest activity changed metadata on shared `/tmp` ancestors retained by the V1
publication test. Both affected construction tests passed alone immediately,
and the complete 50-test retained command then passed. No candidate behavior or
file was changed.

## Required successor

An additive successor must preserve the passing phase-one and review-record
validation while removing every production injection seam:

1. do not retain or expose a module object containing the blocked V1 exact
   entry; import only reviewed data types and pure construction primitives;
2. never overwrite `_v1._require_exact_authority` or any module-global
   validator, even temporarily;
3. replace V1 exact compatibility calls with V2-native fixed-source readers
   whose authority check and target operation are closed in one function;
4. make the worker pool select only literal internal worker functions and have
   each worker independently read and validate the fixed authorization before
   any other operation;
5. remove reader, repository-root, parent-skip, authority-mapping, and callback
   parameters from production validators; place synthetic seams in a separate
   production-ineligible test module; and
6. retain the exact two-phase structure, review cross-bindings, construction
   science, layout, worker ceiling, and publication behavior already passing.

This BLOCK grants no exact build, audit, dataset use, training, selection,
calibration, G2, held-out, runtime, hardware, navigation, production, or
promotion authority.
