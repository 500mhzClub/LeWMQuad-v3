# Independent review: Shared JEPA V5 raw-supervision Auditor V10

Date: 2026-07-14

Reviewer: `/root/raw_v10_independent_review`

Implementation author: `/root/raw_v9_auth_hash_witness`

Governing amendment SHA-256: `02100ee004a572209866a3eb9356441600944b2da2d9b1010282ab992ad02a81`

Verdict: **PASS**

## Review boundary

This is a different-agent review of the frozen audit-only Auditor V10 source,
CLI, author test, and author handoff. The reviewer differs from `/root`, the V10
implementation author, both V9 implementation authors, and both V9 reviewers.
None of the four candidate files was modified.

Review work used source inspection, AST comparisons, and CPU-only tests in
temporary roots. It did not open canonical `.generated`, the built dataset,
the V9 terminal receipt, corpus or RGB payloads, checkpoints, G2, held-out,
runtime, navigation, hardware, production, or accelerator paths. It did not
run a build, exact audit, trainer, or accelerator workload.

## Frozen candidate

| Role | Path | SHA-256 |
|---|---|---|
| `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v10.py` | `3c87dc7878f2e0ae9c54e9b05f1183339b9839568832c0e2fcb6ce75dda984d9` |
| `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v10.py` | `695653257d4aeccef162f3e8f30fd0eba88a090f29cb811481898b4680fe3866` |
| `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v10.py` | `af084a3d097ae66db14f68db7c700843f1bc4007515eb07e444a5018036f177d` |
| `auditor_handoff` | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v10_author_handoff_2026-07-14.md` | `9635d4fa891e9734a6245b8cde3d6eaf8934bb8b9b7c90db9365aa6207e2c959` |

All four hashes and the amendment hash reproduced before and after review.

## Representation finding

The V9 terminal exception reproduces from an actual parsed `SceneManifest`:
the decoded JSON has list-valued boxes, while `SceneManifest.to_dict()` retains
tuples and the raw validator correctly rejects those tuples.

V10 keeps the decoded mapping and parsed manifest as distinct values. The
decoded value must have exact top-level type `dict`; its canonical fingerprint
is checked before and after parsing, raw validation, and render validation.
The exact same object returned by `_read_exact_source_json` is passed directly
to `_validate_raw_scene_object_records`. The parsed manifest remains the value
used for semantic hashing, rendered geometry, ray evidence, rasterization, and
array comparison. Tuple, generator, mapping-proxy, parsed/to-dict, and mutated
representations were independently rejected.

After erasing only the raw parameter and integrity guards, both changed replay
functions reduce to the frozen V9 AST. The unchanged science region, exact
sample recomputer, shard auditor, worker initializer/tasks, and complete closed
publication transaction are V9-identical after the mechanical V10 authority
rename. A full changed-definition inventory found only the two boundary
functions plus the expected V10 authorization, failure binding, and transaction
source-binding functions.

## Authority and lifecycle findings

Phase one accepts exactly eleven ordered, unique role/path/hash rows without
opening a mapped target. It binds the amendment, V9 build authorization,
Builder V9 source/review, V9 dataset manifest, V9 terminal failure, and the four
V10 review-candidate roles plus this review. It rejects `/root`, the V10 author,
both V9 authors, and both V9 reviewers as the V10 reviewer.

The frozen V9 build authorization was independently parsed outside canonical
data paths. Its file hash is
`7878c807fda5f7057addc302afeca77cf6163683f0b2c5b6d4e4ac2701f86792`,
its recomputed content hash is
`d16d0237694e8ed3b9741f6051390de0f759226b49dafa3d8a2ebe95ae7d7df9`,
and its nine-row source-map hash is
`a57d8ee6685627aaca1d9c52bbbf35c13a73ecadf852c788fc210566727c57da`.
V10 phase two deep-validates that capsule, reopens every nested V9 source-map
row, and binds the immutable V9 manifest and terminal-receipt hashes.

Temporary-root publication tests executed the real V10 stage, retained
descriptor chain, inotify journal, `RENAME_NOREPLACE`, post-rename checks,
parent fsync, and final linearization drain. Clean publication preserved the V9
receipt and dataset manifest byte-for-byte. Modify-then-restore attacks on
either artifact permanently poisoned publication. The V10 failure namespace is
distinct, canonical, no-replace, terminal, downstream-false, and no-retry.

The only production entry is keyword-only
`execute_exact_audit_v10(*, authorization_sha256, workers)`. Workers are exact
non-boolean integers in `[1,6]`, use `spawn`, reauthorize in initializer and
task, cap native math threads to one, and clear the inherited CUDA/ROCm device
visibility controls. Source and CLI expose no legacy auditor/builder import,
alternate target, test hook, skip, retry, fallback, trainer, G2, held-out,
runtime, hardware, or production entry.

## Concurrency note

An initial parallel test invocation produced one retained V9 fixture failure.
Both pytest processes were creating and deleting numbered directories directly
under the same retained `/tmp` ancestor. That changed `/tmp.st_nlink`, which the
anti-substitution chain intentionally freezes, so staging failed closed before
publication. The case passed immediately in isolation.

Separate basetemp names directly under `/tmp` reproduced the same shared-parent
condition. After two stable parent directories were created first and each
pytest basetemp was placed beneath its own parent, the suites ran concurrently
and passed `39/39` and `68/68`. This is a fixture-path collision, not a
scientific or publication defect. The exact V10 command is required to be
serialized with every `.generated` mutator. If that requirement were violated,
protected or retained-ancestor directory churn would fail closed and could not
produce a false PASS or modify a foreign V9 leaf. Reviewer QA also proves that
unrelated sibling-file churn is filtered while bound-source churn is rejected.

## Verification

| Check | Result |
|---|---|
| Frozen Auditor V10 author suite | `25 passed` |
| Reviewer-owned Auditor V10 QA | `14 passed` |
| Frozen Auditor V9 author plus independent QA | `68 passed` |
| Combined serialized V10/V9 gate | `107 passed` |
| Parallel V10 and retained gates under stable separate parents | `39 passed` and `68 passed` |
| Candidate source, CLI, tests, and reviewer QA compilation | PASS |
| Candidate and review whitespace scan | PASS |

Reviewer QA:
`lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v10_independent_qa.py`,
SHA-256
`ffde70d56bb06549877540872734456fb7adcecbd25f937d87ced1f47c3f41ea`.

The canonical machine review has content SHA-256
`725e4bd70c7eca9b0542ee0d51c8cf2d4897db3014b3dc9f2ceda4ea8c4acf6a`.

## Authority conclusion

The frozen Auditor V10 candidate satisfies the amendment and is approved as
audit source. This review grants only `auditor_source_approved=true`. It grants
no exact audit, rebuild, retry, dataset use, training, selection, calibration,
G2, held-out, runtime, navigation, hardware, production, promotion, or
deployment authority. A separate canonical V10 audit authorization and an
independent reproduction of that authorization file's SHA-256 are still
required before the sole exact V10 audit attempt.
