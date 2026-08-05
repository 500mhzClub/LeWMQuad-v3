# Shared JEPA V5 raw-supervision metadata plan V3 independent review

Date: 2026-07-13

Verdict: **BLOCK**

V3 closes both published V2 source-index replacement failures and preserves the
frozen scientific and metadata-access result. It does not yet meet its complete
root/leaf continuity contract, so it cannot license raw-builder source work.

## Frozen inputs

The review did not edit the candidate source, author tests, author handoff, V1,
V2, or either predecessor BLOCK record.

| Artifact | SHA-256 |
|---|---|
| V3 source | `0adc6bfa0ea76484f9491a2bbde68f072fd4b908ca8cda7b112c4a32fe481247` |
| V3 author tests | `f1f0bff99f997e94677b542eb35c76332c8f79cf0ff88d5474011ed007f6aa78` |
| V3 author handoff | `66f55b3489c35cbbd5413f4f223942463fad08607564f08994d3f1887bac8160` |
| V2 source | `44641fb97d6172342a3129262c6a0047cae14048c5174a5ed9418420080e1def` |
| V2 author tests | `263b91d04029df172c9c43c79f4d81c4dd887672ee88b9e80555c2f240dc3cd7` |
| V2 independent QA | `3d9a8203b4cfc7aa208b6b319932aaa5d912ac337d02fa436663855bbd090b0c` |
| V2 independent BLOCK | `376a8a761b28502a8495b51554960c046e30f4dd1c7f7d7697d7e3c1b407c65a` |

Independent V3 QA:

- path:
  `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v3_independent_qa.py`
- SHA-256:
  `af32942fc4862b8734cc482b06abe19cd8217b274a32c742aaf46cab231663b0`

## Contract mismatches

### Repository-root descriptor continuity

V3 canonicalizes the absolute repository root, captures its identity, and then
opens that absolute path at source lines 194-200. `O_NOFOLLOW` applies to the
final root component, not to its absolute-path ancestors.

The independent synthetic case changes one root ancestor to a directory
symlink during the actual root `os.open`, delegates to the real `os.open`, and
restores the canonical ancestor before the call returns. The opened descriptor
has the expected device/inode, and later pathname revalidation sees the restored
canonical tree. V3 accepts the index and makes two descriptor reads.

Expected conformance is rejection before any read. A successor needs an atomic
no-follow root acquisition, such as a descriptor-relative component walk from a
stable filesystem anchor, rather than an absolute reopen after pathname
validation.

### Pre-open leaf fingerprint continuity

V3 records the leaf's pre-open device/inode and `st_nlink == 1`, then compares
the opened descriptor to that identity at source lines 247-274. It does not
compare the full pre-open fingerprint to the opened fingerprint.

The independent synthetic case moves the validated leaf, links the same inode
back at the canonical entry, and removes the moved name before `os.open`. The
opened leaf is regular, has the same device/inode, has `st_nlink == 1`, and has
the frozen bytes, but its full fingerprint changed because the relink changed
file metadata. V3 accepts it and makes two descriptor reads.

Expected conformance is rejection before any read. A successor must bind the
full pre-open leaf fingerprint to the opened descriptor and the pre-read entry,
not only device/inode and the current link count.

These are deterministic temporary-tree continuity cases. They do not open a
project source payload or protected role.

## Passing boundary checks

Independent QA confirmed all of the following:

- V2 still reproduces both frozen failures: parent-directory symlink replacement
  and leaf hard-link replacement;
- V3 rejects both published V2 cases before the first `os.read`;
- a different-inode directory-component replacement rejects before reading;
- a hard link added after leaf open rejects on the opened descriptor's link
  count before reading;
- directory, FIFO, symlink, and hard-linked leaf entries reject before reading;
- all exact V1/V2/V3 candidate and review hashes reproduce; and
- all license flags remain false.

## Scientific reconstruction

Independent reconstruction from the frozen paired-row metadata reproduced:

| Role | Pairs | Endpoint uses | Unique endpoints | Scenes |
|---|---:|---:|---:|---:|
| train | 4,262 | 8,524 | 7,777 | 72 |
| checkpoint selection | 495 | 990 | 924 | 8 |
| probability calibration | 415 | 830 | 759 | 8 |
| total | 5,172 | 10,344 | 9,460 | 88 |

Plan identities reproduced exactly:

- content:
  `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`;
- ordered pairs:
  `76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`;
- ordered endpoints:
  `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`.

Inventory identities reproduced exactly:

- scene/role:
  `f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b`;
- frames:
  `7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d`;
- manifests:
  `2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5`;
- plans:
  `0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4`;
- summaries:
  `bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a`.

## Access reconstruction

The exact-open trace reproduced ten regular-file opens across seven allowed
metadata files: paired manifest and row index once each, attitude manifest once,
three development sidecars twice each, and the source index once.

The retained-reference trace reproduced exactly 704 metadata validations:
`88 scenes * 4 references * 2 validation passes`. It observed zero validations
of the eight excluded G2 records.

Observed counts remained:

- excluded G2 sidecar opens: `0`;
- excluded G2 referenced-path inspections: `0`;
- selected source-frame/manifest/plan/summary payload opens: `0`;
- checkpoint/model/runtime/navigation/held-out/hardware/production opens: `0`;
- inventory payload-open ledger values: all `0`.

## Verification

All runs fixed OMP, OpenBLAS, MKL, and NumExpr threads to one; hid CUDA, HIP,
ROCr, and HSA devices; and disabled external pytest plugins.

```text
Independent V3 QA:       14 passed, 2 failed in 2.61s
V3 author suite:          24 passed in 2.58s
V1 + V2 author suites:    26 passed in 4.64s
Frozen V2 review replay:  24 passed, 2 failed in 2.60s
py_compile:               PASS
```

The two V3 failures are exactly the continuity mismatches above. The
machine-readable verdict is:

`docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v3_independent_review_block_2026-07-13.json`

Its SHA-256 is:
`f22ed2cb904005604333768258caf0718257ee49a664ef644ea22055f0c1c058`.

## Authority

V3 remains frozen BLOCK evidence. It licenses neither raw-builder source work,
source payload access, dataset construction, audit acceptance, training,
selection, calibration, G2 access, runtime, hardware, production, nor promotion.
An additive successor with new hashes requires different-agent review.
