# Shared JEPA V5 staged lifecycle successor candidate

Date: 2026-07-13

Status: **successor candidate; different-agent review required; every production
identity remains unset**

## Superseded candidate

The first staged candidate removed the future-output dependency cycle, but its
independent review returned `BLOCK` for two authority defects:

1. G3 compared its passed-G2 predecessor only by checkpoint. A second role
   manifest could reassign the exact G2 scenes to G3, and full promotion would
   accept the two internally valid but mutually overlapping manifests.
2. The live one-shot core and three wrappers were not captured by a fixed
   source launcher or represented in downstream evidence. Copied or changed
   decision source was therefore outside the reconstructed evidence chain.

The review also found that a prebound output rejected only after the runner had
consumed its role-global attempt and performed input/inference work. The exact
blocked core/test identities were `b965b2e7...` / `c834d502...`; they must not
be bound.

## Successor authority graph

All affected schemas and lifecycle revisions are additive successors:

- stage authority `v3`;
- raw outcome and runner ledger `v8`;
- final report `v9`;
- publication `v3`;
- `runner_{g2,g3}_inputs_v2`;
- `finalizer_{g2,g3}_evidence_v2`;
- `publisher_{g2_candidate,full_promotion}_v2`.

The staged order remains G2 runner -> G2 finalizer -> G2 candidate publisher ->
G3 runner -> G3 finalizer -> full publisher. The successor adds these exact
cross-stage invariants:

1. Runner ledgers bind the role-manifest path, file SHA-256, canonical content
   SHA-256, and protocol generation. Final reports and publications reproduce
   the same four-field identity.
2. G3 requires its role-manifest path/file/protocol identity to equal the one
   reconstructed from the G2 candidate before captured G3 source, checkpoint,
   or scene input access. The file identity transitively fixes the content
   identity, which is revalidated when G3 opens the manifest.
3. The G3 ledger/report binds the complete G2 candidate predecessor: candidate
   publisher authority, publisher execution source, publication file/content,
   manifest, checkpoint, and exact G2 final report.
4. Full promotion requires identical G2/G3 manifest and checkpoint identities,
   then requires its explicit G2 report to be byte-identical to the G2 report
   inside G3's predecessor chain.
5. Every runner outcome and ledger path, final-report path, and publication
   path is checked absent through no-follow traversal before reservation,
   predecessor/artifact access, or inference. Writes remain exclusive-create.

## Non-circular source capture

Production authority hashes moved out of the shared core into the three fixed
entrypoint wrappers. All six remain `None`. Each wrapper:

- requires an already isolated Python process (`-I`, no user site);
- rejects execution outside its exact canonical path;
- no-follow opens and hashes the fixed launcher;
- executes the launcher from those captured bytes.

The launcher reopens and hashes its own canonical bytes, no-follow opens the
fixed core, requires the frozen core hash, and executes the core from captured
bytes under a private module identity. This is non-circular: wrappers bind the
launcher, the launcher binds the authority-independent core, and the core does
not hash back into either source.

Runner ledgers, final reports, and publications carry the exact wrapper,
launcher, and core path/hash identities. Each downstream authority binds the
predecessor execution identity, and independent reconstruction rejects any
identity substitution.

## Candidate identities

- staged core:
  `62a19f3028e9152120af990528752431b996f56b4bc9b62db32eba47ae235a1f`
- fixed captured launcher:
  `7f273649fa6c8b4256c552359927fc20bb59d1bfbd5b47194a3f5a941c5b8958`
- G2/G3 runner wrapper:
  `37402f0f75a7a4f475539e269e77aeae072ce80b0af0bcb4147e2ec1b33ff57a`
- G2/G3 finalizer wrapper:
  `f0426201f5344d0eb1d43e183e4755ac8fd7aecdc9af6e5b7c19076af3f5dc34`
- candidate/full publisher wrapper:
  `4e045365dadb28bd37cdbb49808bef7528d4e5cb0c3e77ff5aae678559174fab`
- staged adversarial tests:
  `0d531e1147615db70935c3539c1698edf06028de3f6e59e5a23cb0d2a5d55d26`
- independently passed V5 model source/test, unchanged:
  `b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9` /
  `848aa8be369b89c973a4da916f9c7abeff47eca12aceb4304cf612ed4d53227b`

## Author verification

- focused successor lifecycle: `30/30` passed;
- combined V5 model and successor lifecycle: `70/70` passed in `10.80 s`;
- bytecode compilation and diff whitespace checks: passed;
- native numerical threads: capped at one;
- CUDA, HIP, and ROCr visibility: disabled.

The regression set includes empty-tree execution, every lifecycle transition,
cross-manifest G2-to-G3 scene reassignment, wrong G2 predecessor at full
promotion, checkpoint mismatch, wrong revision/reuse, prebound runner/finalizer/
publisher outputs, execution-identity substitution, copied wrappers, changed
launcher/core bytes, isolated-Python enforcement, skeletal reports, and
permanent synthetic ineligibility.

Only temporary synthetic fixtures were opened. No repository dataset, real
checkpoint, G2/G3 role, held-out/sealed artifact, production authority, Torch
inference/training workload, or GPU was accessed.

## Review rule

A different agent must review these exact bytes and independently reproduce the
cross-manifest, exact-predecessor, copied/changed-source, prebinding, checkpoint,
wrong-revision, and complete empty-tree probes. No lifecycle stage may be bound
until that review passes. A PASS permits preparation of only the new G2 runner
authority first; it is not a G2 result or a license for any later stage.
