# Go2 N32 dynamic cell-square geometry implementation review

Date: 2026-07-11

Status: independently reviewed source revision authorized for metadata-only
machine-manifest preparation. Label-shard byte access, candidate execution, and
finalization remain unauthorized until the generated machine manifest is
independently reviewed and its exact file SHA-256 is supplied to the frozen
commands.

## Frozen Contract

- execution binding:
  `docs/lewm_go2_n32_dynamic_cell_square_geometry_binding_2026-07-11.md`
- binding SHA-256:
  `211043ee3c3200d1fc93febbae73059341aea19560c83f53f3b3bb231bf06e66`
- diagnostic scope: 320 registered train-role frames, 1,310,720 categorical
  target cells, all 129,021 known label occurrences, no learning
- immutable candidate:
  `.generated/go2_dynamic_cell_square_projection_diagnostic/v1/candidate.json`
- immutable final result:
  `.generated/go2_dynamic_cell_square_projection_diagnostic/v1/result.json`

No label shard, RGB image, source geometry, model output, selection,
calibration, G2, runtime, held-out, or sealed payload was opened during source
implementation or review.

## Reviewed Source Map

| Role | Path | SHA-256 |
|---|---|---|
| dynamic geometry | `lewm/benchmarks/go2_dynamic_cell_square_projection.py` | `ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf` |
| diagnostic core | `lewm/benchmarks/go2_dynamic_cell_square_projection_diagnostic.py` | `7f2405c8fef18fae718cb0442f341c5739e17fdb761ba94e0da17a5da9c807a5` |
| preparation | `scripts/prepare_go2_dynamic_cell_square_projection.py` | `a1fb765441620ff89549b78b429822f74f5f39ba6dde6cae83732eebda4555db` |
| runner | `scripts/diagnose_go2_dynamic_cell_square_projection.py` | `32255d10727430dd49151440728ba865c55046ee97d63952f85bd2dd4260698f` |
| finalizer | `scripts/finalize_go2_dynamic_cell_square_projection.py` | `20f816d2802216d700d2374635790e82120a3cc187d1a16d6ab2168ffd07a04b` |
| geometry test | `lewm/tests/test_go2_dynamic_cell_square_projection.py` | `98b4893cdf108fe35a0fe5b77f89a3b44fcad2bb3d178028bb00a476ffdd6026` |
| diagnostic test | `lewm/tests/test_go2_dynamic_cell_square_projection_diagnostic.py` | `aeb00d4119ee0bb38239ec47c3f1628c2bc32e59798fcc96c25401aa59d8ae84` |
| preparation test | `lewm/tests/test_prepare_go2_dynamic_cell_square_projection.py` | `5e461faeb254d1c8b34b1031448d6f4a6d5d13405d5e5944d93f9c120532cfe7` |
| finalizer test | `lewm/tests/test_finalize_go2_dynamic_cell_square_projection.py` | `494e8a7e29d894b2eae4f6d64e978bc9cc73f7cfa8bc8083dc9ea5dd98bcede6` |

## Independent Review

The first frozen-source review confirmed the full-quaternion/yaw-aligned
camera composition and exhaustive cell loop, then reproduced four blocking
integrity defects: an unbound preparation read graph, accepted lexical path
aliases, bool/int/float type confusion, and selected row copies retained across
shards. It also found incomplete denial telemetry and missing persisted output
absence evidence.

The reviewed revision now:

- derives and independently checks the exact preparation, runner, and
  finalizer read/write graphs, including all 20 committed shard identities;
- rejects outside paths, symlinks, raw aliases, substitutions, wrong roles,
  wrong modalities, and hash mismatches before unauthorized access;
- validates timestamps, runtime environment, nested records, and every numeric
  leaf with exact JSON types;
- writes selected rows directly into one canonical 320 x 4096 byte buffer and
  records release-before-next-shard events;
- records live denial precedence and the three exclusive output-absence facts;
- validates the complete finalizer phase contract in the runner before any
  candidate work.

The final independent recursive mutation sweep accepted zero integer-to-float
and zero integer-to-boolean substitutions. `/etc/passwd`, `/etc/shadow`,
`docs//...`, `docs/./...`, source-map substitutions, shard reorderings,
duplicate/out-of-order frame identities, alternate writes, and candidate
canonical/type mutations all reject. The final independent review reported no
remaining finding.

## Verification Commands

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /usr/bin/python3 -m pytest -q lewm/tests/test_go2_dynamic_cell_square_projection.py lewm/tests/test_go2_dynamic_cell_square_projection_diagnostic.py lewm/tests/test_prepare_go2_dynamic_cell_square_projection.py lewm/tests/test_finalize_go2_dynamic_cell_square_projection.py
```

Result: `216 passed in 1.30s`, no skips.

```text
/usr/bin/python3 -m py_compile lewm/benchmarks/go2_dynamic_cell_square_projection.py lewm/benchmarks/go2_dynamic_cell_square_projection_diagnostic.py scripts/prepare_go2_dynamic_cell_square_projection.py scripts/diagnose_go2_dynamic_cell_square_projection.py scripts/finalize_go2_dynamic_cell_square_projection.py lewm/tests/test_go2_dynamic_cell_square_projection.py lewm/tests/test_go2_dynamic_cell_square_projection_diagnostic.py lewm/tests/test_prepare_go2_dynamic_cell_square_projection.py lewm/tests/test_finalize_go2_dynamic_cell_square_projection.py
```

Result: exit 0 for all nine files.

The tests ran on bounded CPU threads. No GPU was used; the 2 GB integrated GPU
is not an authorized training or inference device.

## Authorization Boundary

This review authorizes only the fixed preparation command with this document's
exact SHA-256. Preparation may parse the frozen predecessor metadata and stat
the 20 committed shard paths, but may not open shard bytes. The generated
machine manifest must reproduce every source/input/phase hash and remain a
separate review gate before the runner or finalizer is invoked.
