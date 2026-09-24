# Shared JEPA V5 raw-supervision auditor V1 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_auditor_author`

Status: **AUTHOR COMPLETE; DIFFERENT-AGENT REVIEW REQUIRED; EXACT AUDIT NOT RUN**

## Scope

V1 is an independently implemented auditor for the frozen Shared-JEPA V5
development raw-supervision artifact. It does not import the new V1 builder.
The dataset tree, JSON schemas, byte inventory, pair/endpoint joins, array
layout, role boundaries, access ledger, provenance, and deterministic sample
contract are implemented literally in the auditor. Only the older reviewed V4
camera composition, source semantics, geometry, raycast, and raster functions
are reused for the exact source replay.

The source and tests opened no exact raw-supervision artifact, development
frames, scene manifests, render plans, render summaries, RGB, label shard, G2,
held-out, checkpoint, model output, runtime result, hardware, or production
payload. Exact execution and exact report publication were not attempted.

## Candidate identities

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py` | `854d433084af4bda7dca1e39bed69bc76e9904546111e9289cbb4066660c798c` |
| `scripts/audit_go2_shared_jepa_v5_raw_supervision_v1.py` | `246a8de16a9645a0af8f0cf69e6241b16d68588d54ee9f8eb8b087519a9b908d` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v1.py` | `6dfe991e3f5abc7a5a7405ad1a9ad74382d05ba27e1beb5e6d087aed41351557` |

## Frozen builder cross-contract

The auditor independently matches the frozen builder V1 identities:

| Artifact | SHA-256 |
|---|---|
| builder source | `3bc1559776e2f8471bb6a7a1ddd8808b1f1224687dedf280fd2300820afe25ec` |
| builder CLI | `df5fd60b50ba852d44fd6fe0034c7e763fc08030875488be3850e774906ceeb3` |
| builder test | `15767446ba45851a7f5774560db8e8f6f87d831a51fde7585acffa028f3ba2e4` |
| builder author handoff | `9d9aee5f636069d8beef2362bcc43b9be0063207d9ffe17d9045f99e3c30d28c` |

An executable source-free check reproduced exact equality for the dataset,
shard, endpoint-index, eight-array, access-ledger, frozen-parent, and reviewed
V4 source contracts. A cross-contract test then built a synthetic artifact with
the frozen builder and passed it through the independently implemented auditor.
That test exposed and caused correction of the builder's initial ground-plane
scalar shape before either candidate was frozen.

## Complete artifact audit

The auditor:

1. requires the externally supplied file SHA-256 of canonical `manifest.json`;
2. rejects aliases, special files, extra/missing files, duplicate JSON keys,
   noncanonical JSON/JSONL, malformed content hashes, and any uncommitted byte;
3. reproduces metadata-plan V5 and requires `pairs.jsonl` to equal all 5,172
   exact plan pairs, in order;
4. joins all 10,344 pair endpoint uses to exactly 9,460 unique endpoint rows,
   rejecting cross-role, cross-scene, cross-family, missing, duplicate, and
   orphan endpoints;
5. validates all 88 scene-digest shard paths and every root/shard-local file
   commitment;
6. reconstructs every stored V4 evidence object from the seven raw arrays,
   rerasterizes every endpoint, and compares evidence hashes, raster hashes,
   and all stored `raster_labels.u1` bytes;
7. independently reconstructs the minimum
   `SHA256(role || NUL || family || NUL || endpoint identity)` sample and
   requires exactly eight families in each of the three roles; and
8. compares every raw array byte for all 24 sampled endpoints against original
   development-geometry replay.

The 88 shard validations and 24 source replays use at most six spawned CPU
workers. Every worker fixes the four native thread variables to one and empties
CUDA, HIP, ROCr, and ordinal visibility. No GPU is used.

## Exact source replay

Exact mode is structurally unreachable until the canonical dataset manifest
binds a PASS/PASS builder-and-auditor machine authorization. Before any metadata
or development payload can open, the auditor:

- validates the authorization file/content/source-map hashes;
- requires the nine exact builder/auditor source roles;
- reopens each authorized source through a filesystem-root-anchored,
  component-wise `O_NOFOLLOW` descriptor chain; and
- reproduces every frozen parent and reviewed V4 source hash.

After that gate, metadata V5 is rerun. The auditor independently opens and
hashes all 352 allowed per-scene source files plus the geometry and render-audit
contracts, then repeats the complete 354-file pass after the audit. Each read
retains and rechecks all seven fingerprint fields for every ancestor and leaf.
The exact builder's 708-open access ledger and complete 354-record provenance
inventory are reconstructed, including the independently counted source-frame
JSONL rows. All forbidden access counters must remain zero.

For each selected endpoint the auditor independently joins a parent pair,
validates the source frame, camera mount, attitude, FOV/near/far, render-plan and
summary binding, scene semantic hash, and full-RPY rendered-object parity. It
then calls the reviewed V4 box transform, raycast, and raster implementation
and compares the resulting eight raw arrays byte-for-byte with the artifact.

## Report lifecycle

The exact CLI exposes only `--manifest-sha256` and `--workers`; repository,
dataset, report, and failure paths are fixed. Exact success and failure leaves
are published relative to a retained canonical parent descriptor reached from
filesystem `/`. Publication uses an owned temporary file, fsync, descriptor-
relative `renameat2(RENAME_NOREPLACE)`, and parent-fd fsync. Occupied result or
failure leaves are never replaced. Cleanup removes only an auditor-owned
matching inode and preserves a swapped parent, late destination, replaced
temporary, or other foreign artifact.

An audit PASS remains evidence only. It does not itself authorize dataset use,
training, selection, calibration, G2, runtime, hardware, production, or
promotion.

## Verification

All commands fixed native threads to one, hid all accelerators, and disabled
external pytest plugins.

```text
Focused auditor V1 synthetic/cross-contract suite:   12 passed in 0.58s
Auditor + builder + metadata V5 + reviewed V4:      115 passed, 2 deselected
Literal frozen builder-contract equality check:      PASS
py_compile (source, CLI, test):                       PASS
git diff --check (source, CLI, test):                 PASS
ASCII check (source, CLI, test):                      PASS
```

The two deselections are the already documented legacy V4 tests that assert
its implementation manifest is still unauthorized. The current repository has
since authorized that unchanged implementation. They are unrelated to this
auditor; an unfiltered adjacent run produced 115 passes and those same two
stale failures.

## Required different-agent review

A reviewer other than `/root/raw_auditor_author` must:

1. rehash the candidate, frozen builder, metadata V5, and reviewed V4 closure;
2. prove missing/stale authorization reaches no metadata or development-source
   opener;
3. independently reconstruct the complete manifest, byte, pair, endpoint,
   shard, role/family, access-ledger, and provenance contracts;
4. mutate every array class and join boundary and confirm fail-closed behavior;
5. reproduce the deterministic 24-endpoint selection and independently inspect
   the source replay's frame/camera/full-RPY/raycast/raster path;
6. repeat one-worker/six-worker artifact compatibility and CPU/GPU isolation;
7. adversarially test parent aliases/swaps, occupied result/failure leaves,
   late destinations, true no-replace publication, and inode-owned cleanup; and
8. issue PASS or BLOCK without running the exact audit or opening exact source
   payloads.

Even a reviewer PASS grants no exact audit authority until one reviewed machine
authorization binds both frozen source closures.
