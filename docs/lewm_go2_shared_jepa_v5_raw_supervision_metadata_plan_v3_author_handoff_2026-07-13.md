# Shared JEPA V5 raw-supervision metadata plan V3 author handoff

Date: 2026-07-13

Status: **author complete; different-agent review required**

## Scope

V3 is an additive execution-boundary successor to V2.  It preserves V1, V2,
their tests, both independent BLOCK records, and every scientific plan and
inventory identity.  It replaces only the one allowed frozen source-index byte
open.  It does not open a referenced source payload, grant raw-builder or
training authority, inspect G2 payloads, or authorize runtime or production
use.

## Frozen predecessor evidence

| Artifact | SHA-256 |
|---|---|
| V2 source | `44641fb97d6172342a3129262c6a0047cae14048c5174a5ed9418420080e1def` |
| V2 author tests | `263b91d04029df172c9c43c79f4d81c4dd887672ee88b9e80555c2f240dc3cd7` |
| V2 author handoff | `a6629b32f7ee2266a57d9f705d52d6b2136c391ef5eb135937c4195cbe9f24fb` |
| independent V2 QA | `3d9a8203b4cfc7aa208b6b319932aaa5d912ac337d02fa436663855bbd090b0c` |
| independent V2 BLOCK | `376a8a761b28502a8495b51554960c046e30f4dd1c7f7d7697d7e3c1b407c65a` |

The two V2 failures remain valid for V2.  V3 neither edits them nor
retroactively changes their verdict.

## V3 implementation

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v3.py` | `0adc6bfa0ea76484f9491a2bbde68f072fd4b908ca8cda7b112c4a32fe481247` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v3.py` | `f1f0bff99f997e94677b542eb35c76332c8f79cf0ff88d5474011ed007f6aa78` |

The source-index reader now:

1. requires a canonical, absolute, unaliased repository root and a canonical
   relative source-index path;
2. captures the repository-root identity and opens the root once with
   `O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`;
3. walks each source-index directory component relative to the already-open
   parent descriptor, opening it with `O_DIRECTORY | O_NOFOLLOW` and matching
   pre-open metadata to the opened descriptor identity;
4. prechecks the leaf, opens it relative to the final parent descriptor with
   `O_NOFOLLOW | O_NONBLOCK`, and requires an exact identity match, regular-file
   mode, and `st_nlink == 1` before any byte read;
5. revalidates the canonical root, every directory entry/open descriptor pair,
   and the leaf entry before reading;
6. reads only from the opened leaf descriptor; and
7. revalidates the opened fingerprint and entire descriptor chain after the
   read, then verifies the frozen content SHA-256.

There is no absolute source-index reopen after validation.  All directory and
leaf descriptors close on success or failure.

## Closed V2 failures

The V3 tests deterministically inject both V2 replacements at the leaf-open
boundary:

- moving the validated parent outside the repository and installing a symlink
  at its old entry; and
- moving the validated leaf outside and installing a hard link to the same
  inode at the old entry.

V3 rejects each before the first `os.read`.  It also rejects a different-inode
leaf replacement, a parent replacement before directory open, and a regular
leaf changed to a FIFO between `lstat` and `open`.  The FIFO case is opened
nonblocking and rejected before reading, so it cannot hang the process.

Static missing, directory, FIFO, symlink, hard-link, noncanonical relative
path, referenced-parent/leaf alias, and repository-root defenses remain in
place.  A mutation during the descriptor read is rejected by the opened-file
fingerprint and post-read chain checks.

## Preserved scientific result

V3 returns values exactly equal to V2:

| Role | Pairs | Endpoint uses | Unique endpoints | Scenes |
|---|---:|---:|---:|---:|
| train | 4,262 | 8,524 | 7,777 | 72 |
| checkpoint selection | 495 | 990 | 924 | 8 |
| probability calibration | 415 | 830 | 759 | 8 |
| total | 5,172 | 10,344 | 9,460 | 88 |

Preserved plan identities:

- plan content: `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`;
- ordered pairs: `76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`;
- ordered endpoints: `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`.

Preserved inventory identities:

- scene/role: `f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b`;
- frames: `7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d`;
- manifests: `2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5`;
- plans: `0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4`;
- summaries: `bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a`.

Every license flag remains false.

## Access evidence

The V3 exact-open tracer observed the same ten regular-file byte opens across
the same seven allowed metadata files: paired manifest and row index once each,
attitude manifest once, three development sidecars twice each, and source index
once.  Additional opens are directory descriptors only and carry no payload
bytes.

It observed no G2 sidecar, G2 referenced path, selected source frames, selected
scene manifest, render plan, render summary, RGB, label shard, checkpoint/model,
runtime/navigation, held-out/sealed, hardware, or production payload open.

## Verification

All commands fixed OMP, OpenBLAS, MKL, and NumExpr threads to one, hid CUDA,
HIP, ROCr, and HSA devices, and disabled external pytest plugins.

```text
V1 author + V2 author + V3 author suites: 50 passed in 6.93s
V3 author suite alone:                   24 passed in 2.42s
py_compile:                              PASS
git diff --check (V3 source/tests):      PASS
```

## Remaining authority

This is author evidence only.  A different agent must independently rehash the
frozen predecessor and V3 inputs, replay both V2 replacement exploits against
V2 and V3, reconstruct every count and identity, and reproduce the exact open
boundary.  A PASS may license only separate raw-builder source work.  Source
payload access, dataset construction, audit acceptance, training, selection,
calibration, G2, runtime, hardware, production, and promotion remain separately
gated.
