# Shared JEPA V5 raw-supervision metadata plan V1 independent review

Date: 2026-07-13

Verdict: **BLOCK**

The frozen metadata plan reconstructs the preregistered development population
and keeps its observed byte opens inside the metadata allowlist. It cannot yet
license separate builder source work because its source-inventory path boundary
accepts an in-repository symlink whose referent escapes the repository.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan.py` | `e7ab8727b0d93d3fd8f9e2a3ab5cfdc4f9199e18b8d0a7f5a1f7dc0b5dc0c18e` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan.py` | `e2b49e660292ff99a7794ff3c761f9563a9e182b2889aec3a4e94b835c4be56c` |
| author handoff | `557e6877f02ab61cf300177131735d0831304995dfe0b0f2482b0b5c91fc85fa` |
| development preregistration | `07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb` |
| source-inventory amendment | `39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_independent_access_review.py` | `b7180e901852e34cd412806aa9e8889c0da544b8ea83d3ba51f8efc663018bc6` |

The author source, author tests, handoff, preregistration, and amendment were not
edited by this review.

## Independent reconstruction

The independent test parses only the hash-bound paired manifest and row index,
the attitude-sidecar manifest, the three development sidecar role files, and the
96-row source index. It implements its own canonical JSON, row identity,
sidecar-join, full endpoint identity, and source-inventory reduction.

| Role | Pairs | Endpoint uses | Unique full identities | Scenes |
|---|---:|---:|---:|---:|
| train | 4,262 | 8,524 | 7,777 | 72 |
| checkpoint_selection | 495 | 990 | 924 | 8 |
| probability_calibration | 415 | 830 | 759 | 8 |
| total | 5,172 | 10,344 | 9,460 | 88 |

All 5,172 development rows joined one-to-one to the same-role sidecar global row
with exact scene, source-frame, environment, frame, timestamp, row-identity, and
sidecar-content hashes. The complete row index contained the expected 469 G2
metadata rows for exclusion. The independent 88-of-96 scene reduction excluded
exactly eight source-index metadata records and reproduced all five commitments:

| Inventory | SHA-256 |
|---|---|
| scene/role | `f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b` |
| frames | `7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d` |
| manifests | `2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5` |
| plans | `0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4` |
| summaries | `bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a` |

The candidate also reproduced plan, ordered-pair, and ordered-endpoint SHA-256
values `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`,
`76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`,
and `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`.
Every emitted license flag remained false.

## Blocking finding

### An in-repository symlink can name an outside source

`metadata_path()` at source lines 631-643 checks that a path is absolute,
lexically normalized, contains no `..`, and is lexically relative to the
repository root. It does not resolve the path, reject a symlink in any component,
or bind a regular-file identity.

The independent probe creates `repo/aliased-source` as a directory symlink to an
outside directory and supplies the normalized absolute metadata path
`repo/aliased-source/frames.jsonl`. The plan accepts and emits it. Resolving the
emitted path yields `outside/frames.jsonl`.

This test opens no referenced source file. The defect is nevertheless blocking:
the inventory hashes bind only the lexical path string, so a symlink substitution
does not change any frozen metadata or inventory hash. A later builder could
dereference an outside file while appearing to consume an authorized record.

The lexical outside-root probe passes, as do image-hash-only collision isolation,
cross-role and cross-scene sidecar rejection, family/source-split rejection,
missing/extra/repeated source-row rejection, and sidecar orphan/content-mutation
rejection.

## Access evidence

The controlled-open tracer observed exactly ten byte opens across seven allowed
metadata files:

- paired manifest once and paired row index once;
- attitude-sidecar manifest once;
- train, checkpoint-selection, and probability-calibration sidecars twice each;
- source index once.

No G2 sidecar or payload, source frames/scene manifest/render plan/render summary,
RGB, label shard, checkpoint/model output, runtime/navigation result,
held-out/sealed input, hardware, or production artifact was opened. G2 paired-row
and source-index metadata was read only for exclusion.

## Verification

All pytest commands disabled external plugins, set OMP/OpenBLAS/MKL/NumExpr
threads to one, and hid CUDA, HIP, ROCr, and HSA devices.

```text
author suite:       9 passed in 1.42s
independent suite:  12 passed, 1 failed in 3.80s
combined:           21 passed, 1 failed in 5.21s
py_compile:         passed
```

The one failure is
`test_inventory_rejects_in_repository_symlink_escape`; it is the blocking
access-boundary finding above.

## Required closure

A successor must reject missing or symlinked source paths without reading their
payload bytes, prove that every resolved path remains inside the repository, and
require no-symlink directory traversal plus `O_NOFOLLOW`/regular-file identity
checks again at builder dereference time. It must preserve every frozen count,
join, hash, substitution rejection, and zero-forbidden-open result above.

Until a different-agent successor review passes, this plan licenses neither
builder source work, payload access, dataset construction, nor training.
