# Go2 N32 pose-projection audit implementation manifest

Date: 2026-07-11

Status: frozen after implementation and tests, before the authoritative audit
result.

## Governing commitments

- audit binding SHA-256:
  `c959c45737b9242ef667772af4c7b72effcbb39ae687f5ee28226e38cd63854a`
- fit-panel amendment SHA-256:
  `56f29c4f2eb05c726b0b4461352fe89da2639b86bf9341ec3072958720cf7c6d`
- fit-only panel file SHA-256:
  `77d84e242d75b81fd2b96f086e9cf5df72f0a907e1fe7ce24fc48bbc5d514037`
- fit-only panel canonical content SHA-256:
  `8e44dd0238077120e97fd06b4550d6504627066c7e8ddfdfbd138fd7504ee7a8`

The immutable result path
`.generated/go2_n32_pose_projection_audit/v1/result.json` did not exist when
this manifest was written.

## Source freeze

- pure geometry module
  `lewm/benchmarks/go2_n32_pose_projection_audit.py`:
  `8835fbecc798c1cc3dd7a17b07821677a893854e1a1af0c3073c1bded9a07ac6`
- authoritative runner `scripts/audit_go2_n32_pose_projection.py`:
  `69b449f25d02f230bfef0dba2e5a707cebd7897af72b3ab01e8b7397cf7ff3a2`
- adversarial tests `lewm/tests/test_go2_n32_pose_projection_audit.py`:
  `d45df5146ea66a15c1630a9fd13fa7c64be2ff560a4c2e87bbc02e0017dfdb04`
- fit-only extractor `scripts/extract_go2_n32_pose_fit_panel.py`:
  `f9f4a15f37deff8571dff800fb21c4d50f12cdaa76d68416d9a6b22b8cf4b4bb`

The runner's complete local executable source closure contains only the runner
and the pure geometry module. It hashes both before and after metadata access.

## Access hardening

The runner:

- requires the exact audit-binding authorization before metadata access;
- hashes the original binding and fit-panel amendment before and after use;
- reads only the separately committed fit-only panel and explicitly records
  zero original-monolithic-panel byte opens;
- validates the fit artifact's schema, canonical content, source lineage,
  amendment, 160-row hash, family balance, train role, and 320 unique records;
- accepts only the frozen 20-summary path/hash allowlist and source files named
  exactly `frames.jsonl` under the train metadata root;
- verifies every source and summary hash before parsing and after parsing;
- requires every requested frame exactly once and reconciles exact ledger
  bucket, file, hash, parse, request, match, and forbidden-access counts;
- opens no RGB, label shard, checkpoint, model output, G2, non-train role, or
  sealed payload; and
- creates the result once through exclusive atomic publication.

## Verification

The following focused suite passed before the result existed:

```text
lewm/tests/test_go2_n32_pose_projection_audit.py
lewm/tests/test_categorical_radial_perception.py
lewm/tests/test_go2_categorical_radial_factorization.py

59 passed in 1.01s
```

The suite covers the registered level pose, translated and tilted cameras,
fit-only schema and mutation rejection, source/summary allowlisting,
missing/duplicate records, exact ordering-boundary inequalities, authorization,
pre/post summary and source mutation, access-ledger reconciliation, exclusive
output, and the fit-panel extractor.

This metadata audit cannot pass N32, G2, or a runtime gate. Its sole licensed
effect is to order the next registered N32 representation experiment.
