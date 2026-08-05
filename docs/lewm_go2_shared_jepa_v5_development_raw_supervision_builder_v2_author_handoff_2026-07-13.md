# Shared JEPA V5 raw-supervision builder V2 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_builder_arch`

Status: **AUTHOR COMPLETE; DIFFERENT-AGENT REVIEW REQUIRED; EXACT BUILD NOT RUN**

## Scope

V2 is an additive authorization-boundary successor to the frozen, independently
blocked builder V1. V1 and its BLOCK reproducer remain byte-identical. V2 reuses
the frozen V1 evidence conversion, raycast, raster, array layout, pair/endpoint
joins, six-worker CPU construction, deterministic merge, source revalidation,
and retained-parent no-replace publication. It replaces only the exact-build
authorization boundary.

No exact authorization exists. No exact metadata build, development frame,
scene manifest, render plan, render summary, RGB, legacy label, G2, held-out,
checkpoint, model output, runtime result, hardware, accelerator, canonical
dataset, or failure receipt was opened or created during V2 authoring.

## Frozen V2 candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v2.py` | `0ae5ddd836802ced1fcf7524b67970247dccace6787fd0acc7268cbae4d3e71c` |
| `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v2.py` | `c11396874677c3cd3d0ef76353ea7de1449ef610d35f0b4256530a4f62b1d303` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2.py` | `6755044af535dc0c2de93f0f5bd79b01b140da33bc8ff2ec5b003ef592b50339` |

The V1 source, CLI, author test, author handoff, and independent BLOCK
reproducer rehash exactly to the identities recorded in the V1 independent
review. The canonical exact output
`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`
and its failure receipt remain absent.

## Required two-phase boundary

Phase one is pure validation and has no reader callback. It requires:

1. the exact V2 authorization schema and exact six top-level fields;
2. a lower-case SHA-256 for the authorization file and a correct canonical
   self-hash over the other fields;
3. exactly nine source rows in the frozen order, each with exactly
   `role`, `path`, and `sha256`;
4. unique roles and paths, canonical relative POSIX paths, and the literal
   role-to-path mapping below;
5. exact builder/auditor review-binding schemas, PASS verdicts, fixed
   implementation authors, distinct reviewers, review path/file/content
   hashes, and exact ordered candidate maps; and
6. different builder/auditor authors and different builder/auditor reviewers.

Only a frozen `PhaseOneAuthorizationV2` capsule can enter phase two. Before its
first read, phase two duplicate-key-parses the capsule's canonical payload,
reruns phase one from scratch, and requires exact dataclass equality. A caller
cannot manufacture a capsule containing an arbitrary target.

Phase two then descriptor-opens exactly the nine fixed targets, validates their
file hashes, and duplicate-key-parses the two canonical machine review JSON
records. Each review must have the exact schema, PASS verdict, reviewer/author,
ordered candidate, canonical self-hash, and narrow source-approval authority.
Every exact-build, exact-audit, dataset-use, training, selection, calibration,
G2, held-out, runtime, navigation, hardware, production, and promotion flag in
the review is false. Only after both reviews pass are the frozen metadata/V4
parent sources rehashed. Metadata and referenced development sources remain
unreachable before this complete boundary.

Every compatibility adapter that can call a frozen V1 exact helper accepts the
authorization SHA rather than a caller-supplied authority object. It reruns the
complete V2 fixed-file gate in that same process immediately before installing
a short-lived V1 authority callback and invoking one fixed V1 helper. There is
no mutable verified-authority global, unrestricted module fallback, or exported
V1 exact entry point.

## Nine authority roles

The authorization `source_map` is an ordered list of exact
`{role,path,sha256}` objects:

| Order | Role | Literal relative path |
| ---: | --- | --- |
| 1 | `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v2.py` |
| 2 | `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v2.py` |
| 3 | `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2.py` |
| 4 | `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_author_handoff_2026-07-13.md` |
| 5 | `builder_review` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_independent_review_2026-07-13.json` |
| 6 | `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v2.py` |
| 7 | `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v2.py` |
| 8 | `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v2.py` |
| 9 | `auditor_review` | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v2_independent_review_2026-07-13.json` |

The nine-role contract is retained literally from the frozen preregistration
lineage. It includes the builder handoff but not an auditor handoff. The auditor
machine review commits the auditor source, CLI, and test; an explanatory auditor
handoff may exist but is not a machine authority role. Adding it would silently
change the frozen role cardinality and source-map contract.

## Machine review contract

The authorization schema is
`lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v2`. Its nested
review bindings use
`lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v2` and
have exactly:

```text
schema, review_schema, verdict, reviewer, implementation_author,
path, file_sha256, content_sha256, candidate
```

The builder review schema is
`lewm_go2_shared_jepa_v5_raw_supervision_builder_v2_independent_review_v1`.
Its candidate rows are builder source, CLI, test, and handoff in that order.
The auditor review schema is
`lewm_go2_shared_jepa_v5_raw_supervision_auditor_v2_independent_review_v1`.
Its candidate rows are auditor source, CLI, and test in that order.

Each canonical review JSON has exactly:

```text
schema, verdict, reviewer, implementation_author, candidate,
authority, content_sha256
```

The builder and auditor implementation authors are fixed respectively to
`/root/raw_builder_arch` and `/root/raw_auditor_author`. Review files are
canonical JSON plus one newline and reject duplicate keys.

## Retained science and output

V2 preserves the exact V1 dataset path and schemas. It still constructs 5,172
pairs, 10,344 endpoint uses, 9,460 unique endpoints, and 88 scene shards from
the metadata V5 plan. Every unique endpoint is raycast once. The eight raw
arrays, including scalar `[N]` ground-plane storage and the `64 x 64`
three-state raster, are unchanged. All dataset-use and downstream licenses in
the dataset manifest remain false.

Scene work uses at most six spawned CPU workers. All four native thread
variables are fixed to one and CUDA, HIP, ROCr, and GPU ordinal visibility are
empty. V2 does not use a GPU. The second metadata/source pass, private `0700`
staging directory, retained canonical parent descriptor, inode-owned cleanup,
and `renameat2(RENAME_NOREPLACE)` publication remain the frozen V1 code.

## Adversarial coverage

The V2 suite proves zero metadata, source-map-target, parent, and referenced
source calls for absent, malformed, duplicate-key, duplicate-role,
duplicate-path, missing-role, extra-role, wrong-role, wrong-path,
noncanonical-path, malformed-entry, reordered-role, wrong-author,
wrong-review-path, wrong-cross-binding, and wrong authorization content-hash
records. It also:

- passes an adapted copy of the exact V1 BLOCK reproducer with zero opens;
- rejects a fabricated phase-one dataclass before any read;
- rejects direct calls to every V1 compatibility bridge with fabricated state
  before metadata, parent, or referenced-source access;
- opens exactly nine fixed targets for a fully valid injected synthetic phase
  two and validates both machine review records;
- rejects duplicate keys in authorization and review JSON; and
- reproduces byte-identical synthetic V1 construction output.

## Verification

Every pytest command disabled external plugins, fixed all four native thread
families to one, and hid all accelerator variables.

```text
Focused builder V2 authority/synthetic suite:             27 passed
V1 + V2 + nonblocking V1 independent review checks:       50 passed, 1 deselected
Metadata V5 + reviewed V4 + V1/V2 regression matrix:     228 passed, 2 deselected
py_compile (V2 source, CLI, test):                         PASS
git diff --check (V2 source, CLI, test):                   PASS
ASCII check (V2 source, CLI, test):                        PASS
```

The single deselection is the frozen V1 BLOCK reproducer; its unchanged failure
remains the evidence that V1 cannot be authorized, while the adapted V2 test
passes. The two broader-matrix deselections are the already documented stale
V4 predecessor assertions that require its now-authorized implementation
manifest to remain unauthorized. An unfiltered run reproduced 228 passes and
only those same two failures.

The independently authored auditor V2 reports that it mirrors every shared
authorization, role/path, candidate, review, and false-authority literal; its
current synthetic cross-contract suite reports 18 passed. That is coordination
evidence, not an independent PASS for this frozen builder candidate.

## Required independent review

A reviewer other than `/root/raw_builder_arch` must, without exact execution:

1. rehash this source, CLI, test, handoff, frozen V1/BLOCK lineage, metadata V5,
   reviewed V4 sources, and independently authored auditor V2 closure;
2. independently replay every phase-one adversary and instrument all metadata,
   source-map, parent, and referenced-source openers to prove zero calls;
3. attempt fabricated phase capsules, review bindings, mappings, globals, and
   direct compatibility-helper calls before any source access;
4. validate exact nine-target phase-two reads, canonical duplicate-key-safe
   review parsing, PASS/candidate/authority cross-bindings, and parent-read
   ordering;
5. reproduce unchanged V1 construction science, one/six-worker byte identity,
   eight-array layout, worker isolation, and strict publication behavior; and
6. issue a machine-readable PASS or BLOCK review at the literal builder-review
   JSON path above.

Even a builder review PASS grants no exact build. A later dual-review machine
authorization must bind all nine final hashes before the V2 gate can open any
metadata or development source.
