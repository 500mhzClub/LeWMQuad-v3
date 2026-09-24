# Shared JEPA V5 raw-supervision Builder/Auditor V9 linearization successor amendment

Date: 2026-07-13

Amendment author: `/root`

Status: **source construction and different-agent review only; no exact authority**

## Trigger

Builder V8 passed different-agent review. Auditor V8 received a canonical
different-agent `BLOCK`: after its second event drain and final retained-
ancestry check, a repository ancestor could be moved and recreated as an alias
at entry to the final report-identity helper. Descriptor-relative and absolute
report lookups then reached the same retained inode, no later event drain read
the queued ancestor events, and the operation returned success.

V8 is terminal and may not be repaired, retried, authorized, or reinterpreted.
This amendment creates an additive V9 source/review namespace. It grants no
exact build, exact audit, dataset use, training, selection, calibration, G2,
held-out, runtime, navigation, hardware, production, promotion, deployment, or
retry authority.

No Builder V9 or Auditor V9 source, CLI, test, handoff, review,
authorization, exact output, dataset, or audit artifact existed when these
bytes were frozen.

## Frozen evidence

| Role | Path | SHA-256 |
| --- | --- | --- |
| Governing execution threat model | `docs/lewm_go2_scientific_execution_authority_threat_model_2026-07-13.md` | `3fa8954455f88756f975ffa9e91f51bfd76b8c6461d77a171e145b0f5e43dee3` |
| V8 terminal-quiet amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v8_terminal_quiet_successor_amendment_2026-07-13.md` | `054de82d8648cd6be7edff01b82d549ec916700ebffad51698d4c2041edc6c88` |
| V8 identity rebinding amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_v8_existing_agent_identity_rebinding_amendment_2026-07-13.md` | `392745c80ca2c6e7a103cca4a55c3614cd2c988de9a379fba950b0087df41698` |
| Builder V8 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v8.py` | `f45533354c8b45b88f8eadb2126ec5eaf96fe1f57c21a9bfcd95a8855bfaaa35` |
| Builder V8 CLI | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v8.py` | `f6471f1fa0ca7a13976f752a41ee9ddacbc76636e4d5fb0eee1ebf75bdaee72d` |
| Builder V8 test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v8.py` | `fc1f0cf3fc18bdbd1393be6a514bc04459f943f39b438ced78ebee30e7c57d9a` |
| Builder V8 handoff | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v8_author_handoff_2026-07-13.md` | `9f4898e3620ac87c9a0145be103c4fdf397f727fe37d9f6ca306a0f50916156b` |
| Builder V8 PASS review | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v8_independent_review_2026-07-13.json` | `74b39df6f9f3b0bd85ea45ad921c31abff43b58aba2e9c1ef1547ab22cb2dd27` |
| Auditor V8 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v8.py` | `fb585b4ee9c860eb6a2c2814ff84000a07f8cb070496e530bfb75905e67e1d87` |
| Auditor V8 CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v8.py` | `13c1ebedc6864db21951e0545133664a70a24f1aa02b6082764f0426737f6fc2` |
| Auditor V8 test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v8.py` | `4270c1a1350b8a7a0ef32daec5366cd719965e10776309ea299cd0e8172c8006` |
| Auditor V8 handoff | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v8_author_handoff_2026-07-13.md` | `ed3fdf3d2c9314e64b230997174936f9fedc282a224b0920ff05140d45f418d2` |
| Auditor V8 reviewer QA | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v8_independent_qa.py` | `5fe390c3c3ca94bc6e3bce7d153aa86a475bf35e521e1259294b85e588bd229b` |
| Auditor V8 review report | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v8_independent_review_2026-07-13.md` | `0d253de250cad30d01682ace9563e44ee649bbaa077a4b3d82d1e3ecdc0b6489` |
| Auditor V8 canonical BLOCK | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v8_independent_review_2026-07-13.json` | `63aa9f07c4c7d7603d6fd220d6cda5262cfea0dd8d459848a2a138ccff493a8f` |

The Builder V8 review content SHA-256 is
`c12c037bd2d6c3cbe099c1b0b94cc13924ef134c0c025161722df0c0f5944122`.
The Auditor V8 `BLOCK` content SHA-256 is
`fdc52fe997c7e83df0971b1b37626d771ffaa7b2dd9676b4868db4ee24844f15`.

## Threat boundary and publication linearization

The governing threat model states that scientific authority prevents
accidental privilege leakage, stale or substituted artifacts, repeated
selection, and unreviewed scientific execution. It is not a cryptographic or
kernel isolation boundary against arbitrary code already running as the same
operating-system user.

A finite userspace sequence of `stat`, hashing, and inotify reads cannot prove
that an uncooperative equal-privilege process will never mutate a pathname
after the last observation. V8's literal perpetual-path reading was therefore
stronger than the governing threat model and had no finite implementation.
V9 replaces it with an explicit linearizable contract:

1. all exact `.generated` mutators are externally serialized for the entire
   exact build or audit process;
2. retained no-follow descriptors, complete source/content hashes, canonical
   ancestry checks, and inotify still fail closed on any protected mutation
   observed before commit;
3. the final successful inotify drain is the publication linearization point;
4. no filesystem read, stat, hash, fsync, rename, unlink, chmod, write, or
   other namespace/content operation may occur between that successful final
   drain and returning the already prepared in-memory result; only descriptor
   closes and ordinary in-memory cleanup are allowed; and
5. every later consumer must reopen the canonical path with no-follow ancestry
   checks and independently verify the complete frozen manifest/report and
   file hashes before use. A post-linearization mutation therefore cannot
   create dataset or training authority.

Events caused by mutation during either validation pass are queued and must be
rejected by the final drain. Mutation after the final drain is, by definition,
a post-publication event rather than a mutation inside the completed
transaction; it is rejected at the next mandatory consumer validation. This
does not license arbitrary same-user code inside an authoritative process.

## V9 authorization closure

The only eligible V9 authorization source map is an ordered list of exactly
nine objects, each with exactly `role`, `path`, and lower-case `sha256`:

| Order | Role | Literal repository-relative POSIX path |
| ---: | --- | --- |
| 1 | `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v9.py` |
| 2 | `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v9.py` |
| 3 | `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v9.py` |
| 4 | `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_author_handoff_2026-07-13.md` |
| 5 | `builder_review` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_independent_review_2026-07-13.json` |
| 6 | `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v9.py` |
| 7 | `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v9.py` |
| 8 | `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v9.py` |
| 9 | `auditor_review` | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v9_independent_review_2026-07-13.json` |

The authorization schema is
`lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v9`. It has
exactly `schema`, `exact_build_authorized_after_independent_reviews`,
`builder_review`, `auditor_review`, `source_map`, and `content_sha256`.

Both nested review bindings use
`lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v9`.
Builder and Auditor review schemas are respectively
`lewm_go2_shared_jepa_v5_raw_supervision_builder_v9_independent_review_v1`
and
`lewm_go2_shared_jepa_v5_raw_supervision_auditor_v9_independent_review_v1`.
Canonical review records retain the V8 seven-field shape. Only `PASS` may
enter authorization; Builder authority has only
`builder_source_approved=true`, Auditor authority has only
`auditor_source_approved=true`, and every downstream/exact/retry field is
false.

Fixed implementation authors are:

- Builder V9: `/root/raw_v7_successor_author/auditor_v7_author`;
- Auditor V9:
  `/root/camera_v5_independent/camera_v7_pre_freeze_review/v7_review_artifact_schema`.

The two reviewers must start with `/root/`, differ from `/root`, both
implementation authors, and each other.

## Builder V9 mechanical successor

Builder V9 is a standalone mechanical successor to passing Builder V8. It
changes only V9 authority paths, roles, schemas, authors/reviewers,
provenance labels, accepted immutable capsule types, exact entry name, and CLI
import. It preserves all Builder V8 science, resource limits, source closure,
and the passing directory publication transaction. It may not import or call a
legacy builder or any auditor. Its only exact entry is:

    execute_exact_build_v9(*, authorization_sha256: str, workers: int)

Workers remain exact non-boolean integers in `[1,6]`, use `spawn`, expose no
accelerator, and use one native math thread. Author and reviewer tests must
mechanically compare the V8/V9 science and transaction ASTs after only the
permitted V9 authority renames and rerun every applicable retained Builder V8
test.

## Auditor V9 terminal successor

Auditor V9 is standalone and preserves Auditor V8 science, source closure,
worker policy, no-replace report publication, cleanup, and every passing
adversarial behavior. It changes only V9 authority identities and the terminal
ordering below. It may not import or call a legacy auditor, Builder V9, or a
legacy exact entry. Its only exact entry is:

    execute_exact_audit_v9(*, authorization_sha256: str, workers: int)

After owned report rename, post-rename validation, and parent fsync,
`require_final_quiet` must perform exactly:

1. drain and reject pending protected events;
2. revalidate every retained source/directory fingerprint and hash, the
   complete published audit inventory, retained canonical ancestry, and the
   canonical report name/fingerprint/hash against the attempt-owned report;
3. drain and reject protected events queued during that full pass;
4. repeat retained canonical ancestry plus complete report
   inventory/hash/destination identity validation; and
5. drain and reject protected events queued during the final identity pass.

Step 5 is the publication linearization point and must be the last filesystem
observation. Success may then only copy/return the already prepared in-memory
result and close retained descriptors. Any exception or protected event before
or at step 5 permanently poisons the attempt.

Absolute path validation must reject symlinks in every intermediate canonical
component; following an intermediate alias to the retained inode is not proof
of canonical identity. Descriptor-relative identity and hash validation remain
mandatory.

## Required synthetic proof

Author and different-agent reviewer tests use only temporary roots and must
prove:

- the frozen V8 ancestor-alias reproducer passes incorrectly under V8 and is
  rejected by V9 because the final drain observes its queued events;
- mutations injected before, during, and after each of the first two drains
  and during every source, ancestry, inventory/hash, and destination check are
  rejected no later than the final drain;
- the final drain is structurally the last filesystem observation and only
  in-memory return/descriptor-close operations follow it;
- every intermediate component of every absolute canonical lookup rejects a
  symlink or identity substitution;
- move/recreation and modify/restore during the transaction never repair
  poison;
- unrelated-name churn remains filtered while protected-name churn rejects;
- the successful owned rename emits exactly the reviewed event sequence;
- foreign destinations remain preserved and cleanup remains inode-owned;
- one-worker and six-worker science bytes remain identical; and
- production sources expose no test hook, mutable authority registry, dynamic
  import, legacy module object, alternate opener, skip, retry, or unrestricted
  path.

The tests must not open the canonical authorization, `.generated`, source
payload, dataset, RGB, checkpoint, exact output, G2, held-out, runtime,
hardware, or production paths and must not run an accelerator.

## Retained science

V9 preserves all V8/V7 science: 5,172 pairs, 10,344 endpoint references,
9,460 unique endpoints, 88 development scenes, three scene-disjoint roles,
one schedule/raycast per unique endpoint, full-RPY reviewed camera geometry,
eight fixed arrays, 64 x 64 three-state raster labels, strict joins, 354
provenance records, zero forbidden opens, deterministic one/six-worker bytes,
private staging, retained descriptors, fsync, and one
`renameat2(RENAME_NOREPLACE)`. Any scientific, schema, cardinality, geometry,
array, provenance, output, or resource change requires another additive
amendment.

## Review and authorization sequence

1. Freeze this amendment before any V9 source.
2. The two fixed non-root authors independently construct and freeze Builder
   V9 and Auditor V9 without exact/data work.
3. Two other distinct agents review the frozen candidates and publish the
   fixed canonical `PASS` or `BLOCK` JSON files.
4. Only dual `PASS` permits `/root` to publish the separate canonical nine-row
   V9 authorization.
5. Only after a human separately supplies that authorization file's frozen
   SHA-256 may either fixed V9 CLI consider exact work.
6. Exact build and audit are serialized with every other `.generated` mutator;
   the audit may run only after the one immutable build succeeds.

Any changed byte, `BLOCK`, source-map mismatch, publication anomaly, or failed
test requires another additive successor. This amendment is not an exact
authorization.

## Explicit non-authority

This amendment grants only source construction and different-agent review. It
does not authorize canonical authorization publication, exact build/audit,
opening `.generated` or any payload, training, selection, calibration, G2,
held-out, runtime, navigation, hardware, production, promotion, retry, or
deployment.
