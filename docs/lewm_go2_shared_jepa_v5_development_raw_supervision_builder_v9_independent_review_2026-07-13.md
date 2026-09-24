# Shared JEPA V5 Raw Builder V9 independent review

Date: 2026-07-13

Reviewer: `/root/raw_v8_auditor_reviewer`

Implementation author: `/root/raw_v7_successor_author/auditor_v7_author`

Verdict: **PASS for Builder V9 source only**

## Authority boundary

This review covers only the frozen Builder V9 source, CLI, author test, and
handoff under the V9 linearization amendment. It grants only
`builder_source_approved=true`.

Exact build, exact audit, dataset use, training, selection, calibration, G2,
held-out, navigation, runtime, hardware, production, promotion, deployment,
and retry remain unauthorized. No V9 authorization was created or validated,
no exact entry was called, and no canonical authorization, `.generated`, data,
source payload, RGB, checkpoint, exact output, accelerator, G2, held-out,
runtime, hardware, or production path was opened.

## Frozen candidate

| Role | Path | SHA-256 |
| --- | --- | --- |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v9.py` | `2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v9.py` | `f239a4ef7c067a71f991b30e14bd5c8632c31be3173780fc25b3d9801fff79ee` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v9.py` | `541d1957df0a3da18c2b529cd2d7ca721d7e657c8ebcced2a37931d502cab7bc` |
| `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_author_handoff_2026-07-13.md` | `b6cdf34fa933214e1bb603681f4638f2226e093dad42705445fd8084d6442efd` |

All four hashes matched before and after review. The governing amendment hash
is `6fba5de8d7f04d85bd87e084096ae269c3d3dd6368a6db0b0f8f149c1c5cf773`.
The reviewer-owned QA hash is
`eee77121a80a1dd64fe924e12d4e0a1a369ae096780440828867ca35a71848cc`.

## Closure and preservation

The 83-row `FROZEN_PARENT_HASHES` map rehashed without mismatch, contains the
passing Builder V8 69-row closure as an exact subset, and has canonical content
SHA-256
`76823317704cb35ad3342cb27c03c218816da89b56c294897a7eddd651cdd83e`.
All nine reviewed V4 source rows also rehashed without mismatch.

An independent AST comparison normalized only `V8`/`v8` identifiers. Builder
V8 and V9 each contain 80 uniquely named top-level definitions, and all 80
class/function bodies are exactly equal after normalization. The only changed
normalized non-definition bindings are the predecessor map and replacement of
the two V8 amendment-path constants with the V9 linearization-amendment path.
The CLI changes only its V9 description, import, and exact-entry name.

Therefore the passing Builder V8 science, cardinalities, arrays, geometry,
source closure, worker behavior, deterministic ordering, staging, event
filtering, no-replace publication, cleanup, and resource limits are preserved.

## Authority and process findings

The phase-one gate validates exactly six authorization fields and the ordered,
unique nine-role V9 map before any mapped target opener. Both review bindings
require the exact schemas, fixed implementation authors, distinct `/root/...`
reviewers, exact candidates, canonical hashes, and `PASS`. Phase two accepts
only its immutable V9 capsule, rereads all nine roles in order, validates both
duplicate-key-free canonical review records, and rehashes the fixed predecessor
and V4 source closure.

The only production exact entry is keyword-only
`execute_exact_build_v9(*, authorization_sha256, workers)`. Worker counts are
exact non-boolean integers in `[1,6]`; every process pool uses `spawn`, the
fixed initializer, one native math thread, and empty CUDA, HIP, ROCr, and GPU
ordinal visibility. Static inspection found no legacy builder or auditor
import, legacy exact entry, test-module import, dynamic import, `eval`, `exec`,
mutable authority registry, callback, alternate opener/path, skip, retry, or
accelerator-visible assignment.

## Publication linearization

Builder V9 preserves Builder V8's complete transaction. It validates and
rehashes all retained sources and staged/published leaves, checks canonical
ancestry and destination identity, verifies the exact owned rename events, and
drains protected events after both terminal validation passes.

`require_final_quiet` ends with the literal
`_require_no_events("final publication identity")`. On the successful caller
path, the manifest is already in memory; control then performs only transaction,
staging, and retained-descriptor closes. The failure-only owned cleanup is not
reachable on success. The independent synthetic probe marked the final drain
and made every hash, stat/identity, fsync, rename, read, inventory, and cleanup
helper fail if called afterward. Publication returned successfully and no such
helper ran.

The V9 suite additionally proves one/six-worker identical manifest and complete
tree bytes, intermediate-symlink rejection, source/staging mutation during the
last source pass, complete terminal rehashing, mutation during rehash, ancestor
move between terminal boundaries, modify/restore poisoning, strict versus
ancestry-only event filtering, exact rename events, foreign-destination
preservation, and inode-owned cleanup.

## Verification evidence

All dynamic tests used synthetic temporary roots, one native math thread,
empty accelerator visibility, and disabled pytest plugin autoload.

| Check | Result |
| --- | --- |
| Frozen Builder V9 suite | `72 passed` |
| Retained Builder V8 suite | `69 passed` |
| Reviewer-owned AST and post-linearization QA | `2 passed` |
| Candidate and reviewer QA compile check | PASS |
| 83 predecessor and 9 V4 source rehash | PASS |
| Normalized definition AST equality | `80/80` |
| Exact/data/GPU operation | NONE |

## Decision

Builder V9 satisfies the frozen amendment and receives
`builder_source_approved=true` only. The canonical machine review content
SHA-256 is
`49d8024ae48211cc4fc7d7c2fb674c7ddc7adb38abccace1eb8c6bbc4f10b0df`.
Auditor V9 still requires its separate distinct reviewer PASS, dual PASS still
requires a separate nine-row authorization, and exact work remains forbidden
until the later human-supplied authorization hash step.
