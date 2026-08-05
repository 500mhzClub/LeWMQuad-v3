# Shared JEPA V5 raw-supervision metadata plan V4 author handoff

Date: 2026-07-13

Status: **author complete; different-agent review required**

## Scope

V4 is an additive execution-boundary successor to V3. It preserves V1-V3,
their tests and review evidence, and every scientific plan and inventory
identity. It changes only the allowed frozen source-index byte-open boundary.
It does not open a referenced source payload, grant raw-builder or training
authority, inspect G2 payloads, or authorize runtime or production use.

## Frozen V3 evidence

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v3.py` | `0adc6bfa0ea76484f9491a2bbde68f072fd4b908ca8cda7b112c4a32fe481247` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v3.py` | `f1f0bff99f997e94677b542eb35c76332c8f79cf0ff88d5474011ed007f6aa78` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v3_author_handoff_2026-07-13.md` | `66f55b3489c35cbbd5413f4f223942463fad08607564f08994d3f1887bac8160` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v3_independent_qa.py` | `af32942fc4862b8734cc482b06abe19cd8217b274a32c742aaf46cab231663b0` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v3_independent_review_2026-07-13.md` | `95b20b533f579cf37ee4b895af0033f432aa9b206258947f9aaaa0af6b96a824` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v3_independent_review_block_2026-07-13.json` | `f22ed2cb904005604333768258caf0718257ee49a664ef644ea22055f0c1c058` |

The frozen V3 independent suite still produces its recorded `14 passed, 2
failed`. V4 does not edit V3 or retroactively change that verdict.

## V4 candidate

| Artifact | SHA-256 |
|---|---|
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v4.py` | `d6282a6ee561d34fbe20542f31acd8c7bee82badfa74d1d640930148a9951de2` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4.py` | `724f1c93023256015fe0d468c56591fab35512de79c1e0b0822e78bccdb4a0e0` |

The source-index reader now:

1. validates a canonical absolute repository root and canonical relative
   source-index path;
2. opens the filesystem root with `O_DIRECTORY | O_NOFOLLOW`, then opens every
   repository-root component descriptor-relative with the same constraints;
3. matches each component's pre-open identity and full fingerprint to the
   opened descriptor and requires the final repository descriptor to exactly
   match the validated root fingerprint;
4. retains V3's descriptor-relative no-follow walk beneath that repository
   descriptor;
5. captures the leaf's pre-open `(dev, ino, mode, nlink, size, mtime_ns,
   ctime_ns)` fingerprint, opens it relative with `O_NOFOLLOW | O_NONBLOCK`,
   and requires exact equality before the first `os.read`;
6. revalidates the repository path, descriptor chain, leaf name, opened leaf
   descriptor, and the same original leaf fingerprint after reading; and
7. closes every descriptor on success or failure, then verifies the frozen
   source-index content SHA-256.

## Closed V3 findings

The V4 author suite includes the two frozen V3 mismatch cases as explicit
regressions:

- a repository-root ancestor is transiently replaced by a symlink during the
  actual root-open boundary and restored before return; and
- the validated leaf is moved, hard-linked back to the same inode, and reduced
  to link count one, changing its full fingerprint while preserving identity.

V4 rejects both before the first byte read. The complete V4 suite also retains
V3's parent, leaf, FIFO, symlink, hard-link, mutation, canonical-path, hash,
count, license, and metadata-only access checks.

## Preserved result and access boundary

V4 is exactly equal to V3 for 5,172 pairs, 10,344 endpoint uses, 9,460 unique
endpoints, 88 source records, all frozen plan and inventory hashes, and every
false license flag.

The exact-open tracer observes ten regular-file byte opens over the same seven
allowlisted metadata files. Repository and source-index traversal adds only
directory descriptors. No referenced frames, scene manifest, render plan,
render summary, G2, held-out, model, runtime, hardware, or production payload
is opened.

## Verification

All commands fixed OMP, OpenBLAS, MKL, and NumExpr threads to one; hid CUDA,
HIP, ROCr, and HSA devices; and disabled external pytest plugins.

```text
V4 author suite:                    26 passed in 2.40s
V1 + V2 + V3 author suites:         50 passed in 6.96s
V3 frozen independent replay:       14 passed, 2 failed in 2.38s (recorded V3 BLOCK)
Focused V4 continuity regressions:  3 passed in 0.04s
py_compile:                         PASS
git diff --check (V4 source/tests): PASS
```

## Remaining authority

This is author evidence only. A different agent must independently rehash the
frozen inputs and V4 candidate, replay both V3 mismatches against V3 and V4,
reconstruct counts and identities, and reproduce the exact open boundary.
Only that separate review may issue a verdict. Source payload access, dataset
construction, training, selection, calibration, G2, runtime, hardware,
production, and promotion remain separately gated.
