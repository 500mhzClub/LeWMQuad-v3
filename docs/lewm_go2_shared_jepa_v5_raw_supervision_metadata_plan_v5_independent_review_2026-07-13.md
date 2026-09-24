# Shared JEPA V5 raw-supervision metadata plan V5 independent review

Date: 2026-07-13

Verdict: **PASS**

This is a neutral different-agent quality-assurance review of the exact V5
candidate. No blocking mismatch was found. The candidate closes the frozen V4
directory-fingerprint finding, retains the V1-V4 path and file-kind protections,
and preserves the frozen scientific result and metadata-only access boundary.

## Frozen candidate

The review did not edit the candidate source, author tests, or author handoff.
Their identities matched before and after the review.

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v5.py` | `67c4d325ddab3ac3405e231b78681f4b9ef17b4833ca199395f24ed7a8b82921` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5.py` | `384af6e2b254ea98d32fd7f4798beafe429a4cd83fee6e2903d0d1e8c84f9636` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_author_handoff_2026-07-13.md` | `b362d26372f01e670a477dda5e7abb5e55370cc1d8d89052545afa229e7bba66` |

Independent QA:

- path:
  `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5_independent_qa.py`
- SHA-256:
  `8a50bcf5275d243f06b92264e017f355fd54faaca8f8e73aab1e3cc45dc51298`

## Complete continuity result

V5 retains one original seven-field fingerprint `(device, inode, mode, link
count, size, mtime_ns, ctime_ns)` for each object in the open chain:

- the filesystem root fingerprint is captured at source lines 170-182;
- every component from the filesystem root through the repository root retains
  its complete fingerprint at lines 185-217;
- every source-index parent component retains its complete fingerprint at
  lines 228-260; and
- the source-index leaf retains its complete fingerprint at lines 263-295.

The common revalidator at lines 53-135 compares both the descriptor-relative
named entry and the open descriptor with the original fingerprint. It covers
the filesystem root, repository root, every repository ancestor, every source
parent, and the leaf. The same frozen validation mapping is invoked immediately
before the read loop at line 311 and immediately after it at line 319.

Independent temporary-tree tests captured both calls and confirmed that the
same complete original fingerprints were supplied on each side of every byte
read. They then changed the fingerprint of each owned test component in turn:

- two distinct repository ancestors;
- the repository root;
- both source-index parent levels; and
- the source-index leaf.

Every post-read change was rejected. Representative ancestor, repository-root,
source-parent, and leaf changes at the pre-read boundary were also rejected with
zero calls to `os.read`. A direct named-entry/descriptor probe confirmed the
same full-fingerprint behavior for the filesystem-root slot.

## Predecessor closure

The V5 QA applies the published predecessor cases to V5 rather than treating a
green author suite as sufficient evidence.

- V1: parent and leaf symlink escapes from the referenced-path boundary reject.
- V2: transient source-parent aliases and leaf hard-link replacement reject
  before reading.
- V3: transient repository-ancestor aliases, directory identity replacement,
  same-inode leaf relinking, and a hard link added to the opened leaf reject
  before reading.
- V4: complete-fingerprint changes to repository ancestors, repository root,
  source parents, and leaf reject at both validation boundaries.
- File kinds: directory, FIFO, symlink, and hard-linked source-index leaves all
  reject before the first byte read.

The frozen independent predecessor suites were also replayed unchanged. Their
intentional failures reproduce the prior BLOCK evidence exactly: V1 has one,
V2 has two, V3 has two, and V4 has two. Those are failures of the frozen
predecessors. All corresponding V5 transplant cases pass.

## Scientific and access result

The independently traced real reconstruction produced the unchanged result:

| Quantity | Result |
|---|---:|
| development pairs | 5,172 |
| endpoint uses | 10,344 |
| unique endpoints | 9,460 |
| retained source scenes | 88 |
| referenced-path validations | 704 |
| distinct selected references | 352 |
| regular metadata-file opens | 10 |
| permitted regular files opened | 7 |
| excluded/protected regular-file opens | 0 |

Frozen plan identities:

- content:
  `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`
- ordered pairs:
  `76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`
- ordered endpoints:
  `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`

Frozen inventory identities:

- scene-role:
  `f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b`
- frames:
  `7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d`
- manifests:
  `2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5`
- plans:
  `0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4`
- summaries:
  `bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a`

Every license flag remained false. The ten regular opens were exactly the
frozen dataset manifest, dataset rows, sidecar manifest, two reads of each of
the three development-role sidecars, and one source-index read. The access
ledger remained zero for every referenced-source open category. No excluded
G2 reference was validated or opened.

## Verification

All pytest runs fixed OMP, OpenBLAS, MKL, and NumExpr threads to one, hid
accelerator devices, and disabled external pytest plugins.

```text
Independent V5 QA:       26 passed in 0.84s
V5 author suite:         19 passed in 1.63s
V1-V5 author suites:     95 passed in 10.79s
V5 author + independent: 45 passed in 2.44s
Frozen V1 QA replay:     12 passed, 1 expected predecessor failure
Frozen V2 QA replay:     24 passed, 2 expected predecessor failures
Frozen V3 QA replay:     14 passed, 2 expected predecessor failures
Frozen V4 QA replay:      9 passed, 2 expected predecessor failures
py_compile:              PASS
git diff --check:        PASS
ASCII check:             PASS
```

## Access statement

The review opened only the seven frozen allowlisted metadata files and
temporary synthetic fixtures. It did not open any referenced source frame,
scene manifest, render plan, render summary, excluded G2 payload, held-out
payload, dataset payload, checkpoint, model, training, calibration, runtime,
navigation, hardware, production, or promotion payload.

## Authority

PASS clears the exact V5 metadata plan for its reviewed metadata-only planning
purpose. It does not itself authorize raw source construction, referenced
payload access, training, selection, calibration, G2 access, held-out access,
runtime, hardware, production, or promotion. Those operations require their
own separately frozen authority and execution gate.
