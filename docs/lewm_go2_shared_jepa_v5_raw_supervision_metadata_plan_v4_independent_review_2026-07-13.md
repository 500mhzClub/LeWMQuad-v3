# Shared JEPA V5 raw-supervision metadata plan V4 independent review

Date: 2026-07-13

Verdict: **BLOCK**

This is a neutral different-agent quality-assurance review of the exact V4
candidate. V4 closes both frozen V3 findings and preserves the scientific and
metadata-access result, but it does not provide complete post-read fingerprint
continuity for intermediate directories in the descriptor chain.

## Frozen candidate

The review did not edit the candidate source, author tests, or author handoff.

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v4.py` | `d6282a6ee561d34fbe20542f31acd8c7bee82badfa74d1d640930148a9951de2` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4.py` | `724f1c93023256015fe0d468c56591fab35512de79c1e0b0822e78bccdb4a0e0` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_author_handoff_2026-07-13.md` | `4753d83517a41d2e70e8f25d7cb03ad3709f2d798d1f9f39eea358a527c91415` |

Independent QA:

- path:
  `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4_independent_qa.py`
- SHA-256:
  `5e079be910f5633c01df6d9afc2967715515b27293cc09f279eb71f373c40f78`

## Blocking mismatch

**Intermediate directory fingerprints are not bound through the read.**

V4 captures each root-chain and source-index directory's full seven-field
fingerprint and compares it with the opened descriptor immediately after open.
However, the retained chain rows at source lines 140-144, 267-269, and 311
store only `(parent_fd, component, child_fd, identity)`. The pre-read and
post-read `_revalidate_open_chain` loop at lines 160-179 therefore compares only
device/inode and directory type for those intermediate components.

Two deterministic temporary-tree cases change a directory's full fingerprint
during the first `os.read` while preserving its path, device/inode, directory
type, descriptor, and the source-index leaf bytes:

1. the immediate source-index parent directory; and
2. an ancestor between the filesystem root and the repository root.

Both calls return the frozen payload instead of rejecting. The changed fields
are observable in V4's own `_file_fingerprint`; no identity substitution,
symlink, hard link, nonregular file, or content-hash mismatch is needed.

The contract under review requires the original complete fingerprint for every
walked component to survive both pre-read and post-read validation. A successor
must retain each component's original full fingerprint and compare both its
descriptor `fstat` and descriptor-relative path entry against that fingerprint
in every chain revalidation.

## Passing checks

The independent QA confirmed that V4 otherwise:

- walks from the filesystem root component-by-component with descriptor-relative
  `O_DIRECTORY | O_NOFOLLOW` opens;
- rejects the two published V3 cases before reading: transient root-ancestor
  aliasing at open and same-inode leaf relinking with a changed fingerprint;
- rejects directory, FIFO, symlink, and hard-linked leaf entries before the
  first byte read;
- preserves 5,172 pairs, 10,344 endpoint uses, 9,460 unique endpoints, 88
  scenes, all frozen plan/inventory identities, and every false license flag;
- performs exactly ten regular-file opens across the seven permitted metadata
  files;
- performs exactly 704 metadata-only validations of the 352 selected source
  references; and
- performs zero excluded G2 inspections, zero excluded sidecar opens, and zero
  referenced source-frame/manifest/plan/summary or protected payload opens.

The referenced paths were metadata-validated only. No referenced source
payload, G2 payload, held-out payload, checkpoint, model, runtime, navigation,
hardware, production, or promotion payload was opened.

## Verification

All runs fixed OMP, OpenBLAS, MKL, and NumExpr threads to one, hid accelerator
devices, and disabled external pytest plugins.

```text
Independent V4 QA:       9 passed, 2 failed in 1.59s
V4 author suite:         26 passed in 2.52s
V1 + V2 + V3 suites:     50 passed in 6.89s
Frozen V3 QA replay:     14 passed, 2 failed in 2.49s
py_compile:              PASS
```

The two independent failures are the two manifestations of the same missing
post-read directory-fingerprint invariant. The machine-readable record is:

`docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_independent_review_block_2026-07-13.json`

## Authority

V4 remains useful frozen BLOCK evidence. It does not authorize raw-builder
source work, source payload access, dataset construction, training, selection,
calibration, G2 access, held-out access, runtime, hardware, production, or
promotion. An additive successor with new hashes requires separate review.
