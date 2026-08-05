# Shared JEPA V5 raw-supervision auditor V2 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_auditor_author`

Status: **READY FOR DIFFERENT-AGENT INDEPENDENT REVIEW; NOT AUTHORIZED**

This additive V2 closes every blocking finding against frozen auditor V1 while
retaining its byte, join, reviewed-V4 reconstruction, source replay, worker,
and immutable publication checks. V1 and its independent BLOCK are unchanged.
No exact audit was run. No canonical dataset, development source payload, RGB,
parent-label, G2, checkpoint, held-out, runtime, hardware, production, audit
report, audit failure, or accelerator payload was opened or produced.

## Candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v2.py` | `d57aacd4849ea3e79468618b73925418ad2035d47de636dc991afda777314b2a` |
| `scripts/audit_go2_shared_jepa_v5_raw_supervision_v2.py` | `4502ac44a451841af18e9f9eb545ef961bc81324ea84ce713e434c434e000ae9` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v2.py` | `45d60db1f1a7385b7941f8f52e01a923f056bb3f52cc85b7fec4097d54fa9399` |

Fixed exact success and failure leaves are:

- `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v2.json`
- `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v2.failed.json`

The exact PASS schema is
`lewm_go2_shared_jepa_v5_raw_supervision_audit_v2`. Callback-driven fixture
audits instead emit
`lewm_go2_shared_jepa_v5_raw_supervision_synthetic_audit_v2` with scope
`synthetic_non_authoritative_callback`; they cannot publish an exact report.

## Frozen lineage

The four V1 candidate artifacts rehash to the independent review's frozen
values. Its six-probe test is still
`9684b14c3a87825a1b0d9f4f5bfd17c98c67f92c198818fc441aec0d8b6776fc`.
The review Markdown is
`a61b64e337f5f6e9341db97665ea7a01818d9a74916f77d31cd9721453abdca8`.
The BLOCK JSON file is
`c427b927f863e587c25403ac00b9f06170844b5a936b492e9c213696bf378f5b`
and its canonical content hash remains
`4a8235ed6368f665cc17420dd93a810c7bc7b13963ac7ce79c278cf3bb8a6915`.

The frozen builder V2 closure consumed by the source-free cross-contract test
is:

| Artifact | SHA-256 |
| --- | --- |
| builder V2 source | `0ae5ddd836802ced1fcf7524b67970247dccace6787fd0acc7268cbae4d3e71c` |
| builder V2 CLI | `c11396874677c3cd3d0ef76353ea7de1449ef610d35f0b4256530a4f62b1d303` |
| builder V2 test | `6755044af535dc0c2de93f0f5bd79b01b140da33bc8ff2ec5b003ef592b50339` |
| builder V2 handoff | `7f278c5c24a8e9d89c6b0e3ecb9252acd0edec5729bd9fdde5d72231848bc04f` |

## V2 trust boundary

The callback API has no `exact` argument. The sealed exact API has no loader,
input-plan, source-inventory, or replay callback argument. It requires both an
externally frozen authorization file SHA-256 and manifest file SHA-256, fixes
repository/dataset/report paths, and hard-codes the frozen metadata loaders and
reviewed exact replayer.

Exact ordering is fail closed:

1. validate worker type/range and fixed lexical paths;
2. descriptor-read only the fixed authorization file at the externally supplied
   SHA-256;
3. complete a zero-target-open structural phase over its exact top-level keys,
   content hash, ordered nine-role source map, canonical role/path policy,
   source-map hash, candidate bindings, review bindings, implementation authors,
   distinct reviewers, and PASS flags;
4. reparse and revalidate the complete phase-one capsule to prevent fabricated
   dataclass state;
5. descriptor-read exactly nine fixed targets and duplicate-key-parse the two
   canonical machine PASS reviews, including candidate and false-authority
   cross-bindings;
6. only then open `manifest.json`, require its three authorization provenance
   hashes to match the accepted phase capsule, and inspect dataset/metadata/source
   bytes.

Absent and malformed authority tests instrument the manifest loader, both
metadata loaders, and the development-source hasher and observe zero calls.
A fabricated phase capsule is rejected before the first phase-two reader call.

## Population and schema closure

Before any frozen exact count comparison, V2 parses the committed pair,
endpoint, shard, and shard-index rows and derives:

- pair counts per role and 5,172 total pairs;
- endpoint-reference counts per role and 10,344 total references;
- unique-endpoint counts per role and 9,460 total unique endpoints;
- observed role population;
- mutually reconciled pair/endpoint/shard family populations, exactly eight per
  role for exact data; and
- 88 unique shard scenes with each endpoint count reconciled to its endpoint
  rows.

Only those derived populations are compared with the frozen constants. The
24-row false-population V1 exploit is represented by a one-row fixture with
forged frozen declarations: V2 rejects the declarations before any callback.

Every manifest, pair index, endpoint index, shard, shard index, access-ledger,
shape, byte-count, and population cardinality uses exact `type(value) is int`
validation. Booleans and floats are rejected without `int()` coercion. Tests
cover the three V1 coercion probes plus top-level population counts, manifest
array shapes/files/ledgers, pair `global_row`, shard-local shapes, and shard
index `shard_row`.

## Filesystem closure

The V2 preflight enumerates the complete tree and reads `manifest.json` plus
every declared dataset leaf through the retained descriptor-bound V1 reader.
Every leaf must be a regular file with `st_nlink == 1`; its full seven-field
device/inode/mode/link-count/size/mtime/ctime fingerprint must agree before
open, on the opened descriptor, after read, and through the retained ancestor
chain. The whole preflight is repeated after byte/raster/source replay. Tests
reject hard links to both the manifest and an array leaf.

## Retained V1 science

V2 calls the frozen V1 byte/join/raster engine only after its stronger
preflight. Exact mode supplies the fixed `_exact_sample_recomputer` in the
sealed entry itself. Retained checks include:

- complete manifest/file inventory and duplicate-key-safe canonical JSON/JSONL;
- exact pair and endpoint equality with metadata V5;
- missing/duplicate/orphan/cross-role/cross-scene/cross-family join rejection;
- all eight raw array classes, evidence hashes, reviewed-V4 rerasterization,
  and raster bytes for every endpoint;
- deterministic one-per-role/family sample and byte-exact original-geometry
  replay for 24 endpoints;
- two complete 354-file development-source passes and exact access ledger;
- six-worker cap, one native thread per worker, and empty CUDA/HIP/ROCr/GPU
  visibility; and
- retained-descriptor, true no-replace, inode-owned terminal report publication.

## Verification

Every command fixed OMP, OpenBLAS, MKL, and NumExpr threads to one, emptied all
accelerator visibility variables, and disabled external pytest plugins.

```text
Focused auditor V2 adversarial/synthetic suite:           25 passed
Auditor V2 + frozen V1 author/positive review controls:   63 passed, 6 deselected
Metadata V5 + reviewed V4 + builder/auditor V1/V2:       193 passed, 3 deselected
py_compile (V2 source, CLI, test):                         PASS
git diff --check (V2 source, CLI, test):                   PASS
```

The six deselections are the unchanged V1 BLOCK reproducers; all six adapted V2
regressions pass. The three broad-matrix deselections are stale predecessor V4
tests that expect its now-authorized implementation manifest/source receipt to
remain unauthorized. They are unrelated to this additive auditor; no failure
in the retained raw-supervision, metadata V5, or reviewed-V4 science path was
deselected.

## Required independent review

A reviewer other than `/root/raw_auditor_author` must, without exact execution:

1. rehash the candidate, V1/BLOCK lineage, frozen builder V2, metadata V5, and
   reviewed V4 closure;
2. independently adapt and pass all six V1 BLOCK probes against V2;
3. attack the synthetic/exact schema split, public signatures, internal helper
   reachability, fixed loader/replayer, and terminal publication paths;
4. instrument absent, malformed, reordered, duplicated, wrong-path,
   wrong-candidate, wrong-review, wrong-author, and fabricated-capsule authority
   cases and prove zero pre-authority dataset/metadata/development-source opens;
5. independently derive every actual pair/reference/endpoint/role/family/shard
   population before checking frozen declarations;
6. mutate bool/float cardinalities throughout manifest/index/shard/ledger/shape
   records and hard-link the manifest plus every dataset leaf class; and
7. issue the canonical machine PASS or BLOCK record at
   `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v2_independent_review_2026-07-13.json`.

Even independent PASS records for both builder and auditor grant no exact build
or audit. A later dual-review authorization must bind the final nine source
hashes, and a human must separately supply its frozen file hash to the builder
and auditor CLIs. An audit PASS remains evidence only; it grants no dataset use,
training, selection, calibration, G2, held-out, navigation, runtime, hardware,
production, or promotion authority.
