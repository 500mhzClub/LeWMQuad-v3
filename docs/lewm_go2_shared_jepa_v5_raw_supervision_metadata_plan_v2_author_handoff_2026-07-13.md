# Shared JEPA V5 raw-supervision metadata plan V2 author handoff

Date: 2026-07-13

Status: **author complete; different-agent review required**

## Scope

V2 is an additive path-boundary successor to the frozen V1 metadata plan.  It
closes the sole V1 review blocker without changing V1 or any scientific pair,
endpoint, source-record, count, ordering, or hash.  It does not license a raw
builder, open referenced source payloads, train a model, inspect G2 payloads, or
grant runtime or production authority.

## Frozen V1 and BLOCK evidence

These inputs were rehashed and were not edited:

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan.py` | `e7ab8727b0d93d3fd8f9e2a3ab5cfdc4f9199e18b8d0a7f5a1f7dc0b5dc0c18e` |
| V1 author tests | `e2b49e660292ff99a7794ff3c761f9563a9e182b2889aec3a4e94b835c4be56c` |
| V1 author handoff | `557e6877f02ab61cf300177131735d0831304995dfe0b0f2482b0b5c91fc85fa` |
| V1 independent adversarial tests | `b7180e901852e34cd412806aa9e8889c0da544b8ea83d3ba51f8efc663018bc6` |
| V1 independent BLOCK review | `fcedb1efaffe4ca07141f7750188409c9f8f474231d2bb4f2db750f36f3f07b5` |
| development raw-supervision preregistration | `07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb` |
| builder source-inventory amendment | `39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3` |

The V1 BLOCK remains valid for V1.  V2 does not overwrite, reinterpret, or
retroactively pass it.

## V2 implementation

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v2.py` | `44641fb97d6172342a3129262c6a0047cae14048c5174a5ed9418420080e1def` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v2.py` | `263b91d04029df172c9c43c79f4d81c4dd887672ee88b9e80555c2f240dc3cd7` |

V2 delegates the already-frozen scientific joins to V1 and adds a boundary for
the four retained metadata references per development scene.  Before returning
an inventory, it:

- requires a canonical absolute repository root with no symlink alias;
- rejects relative paths, lexical `.` or `..`, redundant separators, trailing
  separators, and any other noncanonical absolute spelling;
- proves lexical and resolved containment inside the repository;
- walks every retained component with `lstat` semantics and rejects a symlink
  at the leaf or in any parent directory;
- requires every leaf to exist and be a regular file;
- rejects hard links, repeated paths, and repeated device/inode identities;
- validates original source-index strings as well as emitted V1 records;
- validates only the 88 retained development records, never an excluded G2
  reference; and
- repeats no-follow identity validation while opening the one allowed frozen
  source-index metadata file.

The path checks read filesystem metadata only.  They do not read bytes from
frames, scene manifests, render plans, or render summaries.  This author stage
cannot remove the later builder's time-of-check/time-of-use obligation: the
builder must repeat component-safe traversal and use `O_NOFOLLOW`, regular-file
and expected-identity checks on the descriptor that it actually reads.

## Preserved scientific result

V2 exactly equals V1 for all frozen plan and inventory values:

| Role | Pairs | Endpoint uses | Unique endpoints | Scenes |
|---|---:|---:|---:|---:|
| train | 4,262 | 8,524 | 7,777 | 72 |
| checkpoint selection | 495 | 990 | 924 | 8 |
| probability calibration | 415 | 830 | 759 | 8 |
| total | 5,172 | 10,344 | 9,460 | 88 |

Preserved identities:

- plan content: `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`;
- ordered pairs: `76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`;
- ordered endpoints: `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`;
- scene/role inventory: `f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b`;
- frames inventory: `7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d`;
- manifests inventory: `2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5`;
- plans inventory: `0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4`; and
- summaries inventory: `bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a`.

All V1 license flags remain false.

## Access evidence

The V2 controlled-open test observed the exact ten allowed metadata-file opens:
the paired manifest and row index once each, attitude-sidecar manifest once,
each of the three development sidecars twice, and source index once.  It
observed no G2 sidecar or referenced G2 source, source frame, source scene
manifest, render-plan, render-summary, RGB, parent label shard, checkpoint or
model output, runtime/navigation, held-out/sealed, hardware, or production
payload open.  The source inventory ledger retains zero for every payload-open
category.

## Verification

Every command set OMP, OpenBLAS, MKL, and NumExpr threads to one, disabled
external pytest plugins, and hid HIP, CUDA, and ROCr devices.

```text
V1 author suite:       9 passed in 1.48s
frozen V1 review:     12 passed, 1 failed in 4.03s
V2 author suite:      17 passed in 3.23s
py_compile:           PASS
git diff --check:     PASS
```

The frozen V1 review's one failure is its pre-existing symlink-escape BLOCK and
is expected because that suite intentionally imports frozen V1.  The V2 suite
replays that exploit against V2 and rejects it, alongside internal parent and
leaf symlinks, resolved and lexical escapes, missing files, directories, FIFOs,
hard links, duplicate aliases, path normalization variants, aliased repository
roots, and source-index substitution.

## Remaining authority

This is author evidence only.  A different agent must independently hash the
frozen parents, reconstruct the exact V1/V2 identities, adversarially verify the
path and open boundaries, and issue an explicit PASS before any builder source
or payload access is licensed.  Builder implementation and the produced raw
dataset will still require their own separate review and audit.
