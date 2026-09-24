# Shared JEPA V5 raw-supervision metadata plan V2 independent review

Date: 2026-07-13

Verdict: **BLOCK**

V2 closes the frozen V1 static symlink escape and preserves every scientific
identity, count, inventory hash, and observed payload-access prohibition.  It
cannot license raw-builder source work because its one allowed source-index
byte open is separated from its component validation.  Two deterministic
replacement probes cross that check/open gap while preserving the validated
inode and expected bytes.

## Frozen review inputs

The review did not edit any author source, author test, handoff, predecessor,
preregistration, or source-inventory amendment.

| Artifact | SHA-256 |
|---|---|
| V1 source | `e7ab8727b0d93d3fd8f9e2a3ab5cfdc4f9199e18b8d0a7f5a1f7dc0b5dc0c18e` |
| V1 author tests | `e2b49e660292ff99a7794ff3c761f9563a9e182b2889aec3a4e94b835c4be56c` |
| V1 author handoff | `557e6877f02ab61cf300177131735d0831304995dfe0b0f2482b0b5c91fc85fa` |
| V1 independent tests | `b7180e901852e34cd412806aa9e8889c0da544b8ea83d3ba51f8efc663018bc6` |
| V1 independent BLOCK | `fcedb1efaffe4ca07141f7750188409c9f8f474231d2bb4f2db750f36f3f07b5` |
| V2 source | `44641fb97d6172342a3129262c6a0047cae14048c5174a5ed9418420080e1def` |
| V2 author tests | `263b91d04029df172c9c43c79f4d81c4dd887672ee88b9e80555c2f240dc3cd7` |
| V2 author handoff | `a6629b32f7ee2266a57d9f705d52d6b2136c391ef5eb135937c4195cbe9f24fb` |
| independent V2 QA | `3d9a8203b4cfc7aa208b6b319932aaa5d912ac337d02fa436663855bbd090b0c` |

The V1 BLOCK was reproduced directly: V1 accepted a canonical-looking path
through an in-repository directory symlink to an outside file.  V2 rejected the
same input before any referenced byte open.  The V1 verdict remains frozen and
is not retroactively changed.

## Preserved scientific result

The independent reconstruction parsed the hash-bound paired row metadata and
rebuilt full endpoint identities from role, scene, episode, environment,
episode step, frame index, timestamp, and image SHA-256.

| Role | Pairs | Endpoint uses | Unique endpoints | Scenes |
|---|---:|---:|---:|---:|
| train | 4,262 | 8,524 | 7,777 | 72 |
| checkpoint selection | 495 | 990 | 924 | 8 |
| probability calibration | 415 | 830 | 759 | 8 |
| total | 5,172 | 10,344 | 9,460 | 88 |

V2 exactly retained the frozen plan, ordered-pair, and ordered-endpoint
identities:

- `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`;
- `76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`;
- `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`.

The independent 88-of-96 source-index reduction reproduced all five inventory
commitments:

| Inventory | SHA-256 |
|---|---|
| scene/role | `f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b` |
| frames | `7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d` |
| manifests | `2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5` |
| plans | `0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4` |
| summaries | `bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a` |

Every license flag remained false.

## Passing boundary evidence

Independent synthetic probes confirmed that V2 rejects:

- relative and noncanonical absolute spellings, including dot, dot-dot,
  duplicate-separator, trailing-separator, double-root, and NUL variants;
- lexical and resolved repository escapes;
- symlinks at either tested parent depth or the leaf, whether the target is
  inside or outside the repository;
- missing leaves, directories, FIFOs, hard links, repeated paths, and repeated
  device/inode identities; and
- relative, noncanonical, symlinked, missing, and non-directory repository
  roots, plus a different-inode source-index substitution.

Instrumentation observed exactly two metadata-only validations of each of the
four retained paths for each of 88 development records.  It observed no
validation or resolution of any path from the eight excluded G2 records.

The controlled-open trace observed exactly ten opens across the seven allowed
metadata files: the paired manifest and row index once each, the attitude
manifest once, each of the three development sidecars twice, and the source
index once.  It observed zero G2-sidecar, selected-source payload, RGB,
label-shard, checkpoint/model, runtime/navigation, held-out/sealed, hardware,
or production opens.

## Blocking findings

### Parent replacement crosses the source-index boundary

`_read_frozen_source_index` first calls `_validate_referenced_path` on the
absolute source-index pathname.  It later calls absolute `os.open(path,
O_NOFOLLOW)`.  `O_NOFOLLOW` protects only the final component; it does not bind
the parent directories inspected earlier.

The deterministic probe waits until validation returns, renames the validated
`index` directory outside the repository, replaces the original directory
entry with a symlink to the moved directory, and lets the absolute open
continue.  V2 follows the new parent symlink, opens the same validated inode
outside the repository, and accepts it because the inode and expected content
hash still match.

### Leaf hard-link replacement crosses the same gap

A second probe waits until validation returns, renames the validated leaf
outside the repository, and installs a hard link to that inode at the original
leaf pathname.  The open again sees the expected device/inode and bytes.  V2
accepts it because the post-open `fstat` does not recheck `st_nlink == 1`.

These are not arbitrary same-user reflection claims.  They are deterministic
filesystem input substitutions at the exact protected validation/open
boundary.  Both contradict the stated no-alias source-index access guarantee
and the governing rule that a source may not change between review and open.

## Verification

All commands fixed OMP, OpenBLAS, MKL, and NumExpr threads to one, hid CUDA,
HIP, ROCr, and HSA devices, and disabled external pytest plugins.

```text
V1 author + frozen V1 review + V2 author: 38 passed, 1 expected V1 BLOCK failure
independent V2 QA:                       24 passed, 2 failed in 2.48s
py_compile:                              PASS
git diff --check (independent QA):       PASS
```

The two V2 QA failures are the parent-symlink and leaf-hardlink replacement
probes above.  All other independent V2 QA cases pass.

## Required successor

An additive successor must preserve V1 and V2 unchanged and open the source
index through a descriptor-relative, no-follow walk anchored at an already
opened canonical repository-root directory descriptor.  Each directory
component and the leaf must be opened and `fstat`-validated through its parent
descriptor.  The opened leaf must remain regular, have `st_nlink == 1`, match
the expected identity, remain unchanged through the read, and match the frozen
content hash.  No absolute pathname may be reopened after validation.

Until that successor passes different-agent review, V2 licenses neither raw
builder source work, source payload access, dataset construction, nor training.
