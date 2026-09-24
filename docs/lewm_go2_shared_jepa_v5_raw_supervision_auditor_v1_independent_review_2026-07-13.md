# Shared JEPA V5 raw-supervision auditor V1 independent review

Date: 2026-07-13

Reviewer: `/root/raw_auditor_v1_independent`

Verdict: **BLOCK**

The frozen V1 candidate was reviewed without modification. Its sealed exact
entry point, reviewed-V4 replay chain, whole-array reconstruction, deterministic
sample, worker cap, and report lifecycle have substantial passing evidence.
Four source-boundary and schema defects still contradict the claimed exact
audit contract. No exact dataset, development source payload, RGB, parent label,
G2, checkpoint, held-out, runtime, hardware, production, or accelerator payload
was opened.

## Frozen candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py` | `854d433084af4bda7dca1e39bed69bc76e9904546111e9289cbb4066660c798c` |
| `scripts/audit_go2_shared_jepa_v5_raw_supervision_v1.py` | `246a8de16a9645a0af8f0cf69e6241b16d68588d54ee9f8eb8b087519a9b908d` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v1.py` | `6dfe991e3f5abc7a5a7405ad1a9ad74382d05ba27e1beb5e6d087aed41351557` |
| author handoff | `7d693902bf4517bb19a87b6769af0c272403ba553daccb6e03d9cef88eec279d` |
| independent BLOCK reproducer | `9684b14c3a87825a1b0d9f4f5bfd17c98c67f92c198818fc441aec0d8b6776fc` |

The frozen builder, metadata V5 closure, and all nine reviewed V4 source hashes
also rehashed exactly. The auditor's literal schemas, paths, array layout, parent
maps, and worker bound match the frozen builder contract; `u1` and `|u1` are
equivalent NumPy dtype spellings.

## Blocking findings

### 1. The exported generic auditor permits caller-injected exact replay

`audit_dataset_v1()` is exported and accepts both `exact=True` and an arbitrary
`sample_recomputer`. It does not require the fixed authorization preflight,
metadata V5 loader, source inventory loader, or sealed `_exact_sample_recomputer`.
Its exact cardinality checks compare only manifest declarations to frozen
numbers; they do not reconcile those declarations with the actual pair,
endpoint, and shard populations.

The independent reproducer built a valid synthetic artifact with only 24 pairs,
24 unique endpoints, and 24 shards, one for every development role/family. It
then changed only the self-hashed manifest declarations to 5,172 pairs, 9,460
endpoints, 10,344 references, and 88 shards. Calling the exported function with
`exact=True` returned `PASS`, invoked the caller callback, and reported the
actual 24-row population under the exact audit schema.

The separate `audit_exact_dataset_v1()` signature correctly has no callback,
but that does not close the exported parallel exact route.

Relevant implementation locations:

- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:886`
- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:1476`
- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:1512`
- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:2380`

### 2. Structurally invalid authority opens a caller-selected source first

`_validate_exact_authorization()` validates and immediately opens each
`source_map` row while it is still accumulating the required role set. Only
after those opens does it require the exact nine roles.

The independent reproducer supplied a canonical self-hashed authorization with
both PASS flags but only one row: role `builder_source`, pointing at a synthetic
`arbitrary/referenced_frames.jsonl`. The candidate opened that caller-selected
file and only then rejected the eight missing roles. A malformed authority can
therefore cross the repository-source boundary before authority acceptance.

The validator also lacks a strict role-to-canonical-path map. Complete
structural validation, exact role/path binding, and cross-binding must precede
all source-map target opens.

Relevant implementation location:

- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:1787`

### 3. Dataset hard-link aliases pass the whole-tree audit

The descriptor-bound reviewed-source reader rejects any leaf whose link count
is not one. The dataset tree reader does not: `_resolve_regular_file()`,
`_read_bound_file()`, `_tree_file_inventory()`, and the manifest reader accept
regular files with `st_nlink > 1`.

The independent reproducer added an external hard link to a committed shard
array without changing a byte. The complete audit returned `PASS`. The external
name can subsequently mutate the supposedly immutable audited bytes, contrary
to the handoff's alias-rejection claim and the source reader's own trust model.

Relevant implementation locations:

- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:769`
- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:790`
- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:829`

### 4. JSON cardinalities are not strict integers

Pair-index and endpoint-index `row_count` values are passed through `int()`;
manifest shard `endpoint_count` is compared with ordinary Python numeric
equality. Consequently `1.0`, and even boolean `true`, are accepted as the
integer one after the manifest is canonically rehashed.

All three independent cases returned `PASS`. Exact top-level count dictionaries
use the same Python equality behavior, so strict integer validation must cover
every count before any comparison or coercion.

Relevant implementation locations:

- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:928`
- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:982`
- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py:1135`

## Passing evidence

The following behavior passed and can be retained by an additive successor:

- all candidate, frozen builder, metadata V5, and reviewed V4 hashes match;
- the sealed exact CLI fixes repository, dataset, report, and failure paths and
  exposes no caller replay callback;
- authorization preflight precedes metadata V5 and development source inventory
  opening on the sealed entry point;
- the inventory expands exactly 88 scenes into 352 allowlisted source files,
  before adding the two fixed parent contracts;
- the deterministic independent selection reproduces exactly one minimum hash
  for each of eight families in all three roles;
- one-worker and six-worker synthetic audits return byte-identical results, the
  cap rejects zero, seven, and booleans, and worker environments set four native
  thread variables to one while emptying CUDA, HIP, ROCr, and GPU ordinal;
- mutations of every one of the eight committed array classes are rejected;
- independently rebound pair, top-endpoint, and shard-index join mutations
  reach semantic rejection rather than only an outer file hash;
- the exact replay source visibly includes frame matching, camera mount and
  attitude checks, scene semantic hash, full-RPY object parity, yaw-body box
  conversion, reviewed V4 raycast, and reviewed V4 rasterization;
- source and publication ancestor aliases/swaps are rejected, owned temporary
  files are cleaned through retained descriptors, and late destinations are
  preserved by true `renameat2(RENAME_NOREPLACE)`; and
- an exact execution failure publishes one false-authority terminal receipt and
  a second attempt is refused.

## Verification

Every command disabled external pytest plugins, fixed OMP, OpenBLAS, MKL, and
NumExpr threads to one, and hid CUDA, HIP, ROCr, and HSA devices. Synthetic
builders used at most one worker; auditor checks used at most six.

```text
author focused auditor suite:          12 passed in 0.54s
independent positive controls:         26 passed
independent BLOCK reproducers:          6 failed
independent total:                     26 passed, 6 failed in 2.48s
py_compile (independent test):         PASS
git diff --check:                      PASS
```

The canonical exact dataset, audit report, audit failure, and dual-review build
authorization were all absent when checked. Exact execution was not run. This
BLOCK grants no exact audit/build, dataset use, training, selection,
calibration, G2, held-out, runtime, hardware, navigation, production,
promotion, or retry authority.

## Required successor

An additive successor must preserve the passing closure while:

1. making the callback-based synthetic auditor permanently non-exact and
   unreachable as an authoritative report path, with exact mode available only
   through the fixed loader and fixed replay implementation;
2. deriving and reconciling pair, endpoint-reference, unique-endpoint, role,
   family, and shard counts from the audited rows before comparing frozen exact
   constants;
3. validating the complete authorization and exact role-to-path map in a
   zero-open phase, then rehashing those targets only after acceptance;
4. rejecting multiply linked manifest and dataset leaves with stable
   before/open/after link-count and fingerprint checks; and
5. applying exact-integer validation to every manifest, index, shard, ledger,
   shape, and population cardinality, with passing regressions for all six
   independent failures.

