# Shared JEPA V5 Raw-Supervision Auditor V13 Self-Resolving CLI Successor Amendment

Date: 2026-07-14

Amendment author: `/root`

Status: **source construction and different-agent review only; no exact authority**

## Trigger and terminal V12 invocation

Auditor V12 source passed independent review and its exact authorization was
independently fingerprinted. The frozen bindings are:

| Role | Path | SHA-256 |
|---|---|---|
| V12 amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_hsa_worker_isolation_successor_amendment_2026-07-14.md` | `f4892405cf0fd97f9096f99d840b5590810fd8640822ed2e8c4c254c0c3e6adf` |
| V12 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v12.py` | `f435406c7ff8d42a549cd678a65584bc88ac49f96b590247b811c6bb4b934943` |
| V12 CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v12.py` | `45f93534b02afe99722144509fc9b7dde72e735daa8bed1dc58951d3c0bb8471` |
| V12 test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v12.py` | `dbefb4dc455b45873e14256d5fa647e22fcf1eff1a43ba249e7b9fe7f5ed5dd7` |
| V12 handoff | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_author_handoff_2026-07-14.md` | `d1955fde4106cf54f1adb75fcbd84abb00b24597384c5ad05c51abf73b22e4ef` |
| V12 review | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_independent_review_2026-07-14.json` | `d7ae190f1971befbc26ae2e7b6a36955a614bf9c94f85860a4d4d26922d91d30` |
| V12 authorization | `docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_authorization_2026-07-14.json` | `6b5f317119a00308390b8a32f1057f34455313eb80ec190aa9d8d27052a81575` |
| V12 authorization witness | `docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_authorization_fingerprint_2026-07-14.json` | `662e6c2f6386b8822b3bd968a4faf0bf3e2e222ff4aac9df8a99cc680c254327` |
| V12 launch failure | `docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_launch_failure_2026-07-14.json` | `cc6313b1d6e56022204ba82dc57efc6b7cc85a715f078cd865883b61cee88eb3` |

The V12 authorization canonical content and source-map SHA-256 values are
`8db4611a321309a76a0dd81e3af0148fce788422e2008ccaef1039e3c5ae493a`
and `1fc3374101fca166fe74b34b779cf995ec46a12fbb609f5de3bc5a428d225bc2`.
The witness canonical content SHA-256 is
`4845826d1caeedc58d01b580a8681a71730eb0ba17205bde36d3673c9052741b`.
The launch-failure canonical content SHA-256 is
`b9775ef4705d7505931b64c7ceaad57fb8d18da72429bb877245fb534197b2ee`.

The one authorized V12 command was invoked once by
`/root/raw_v12_exact_execution`. It exited `1` after `0.2s` at CLI import line
29 with `ModuleNotFoundError: No module named 'lewm'`. Executing a file under
`scripts/` placed that directory, not the repository root, on `sys.path`; the
reviewed exact command declared no `PYTHONPATH`. The failure occurred before
`execute_exact_audit_v12` entered. Zero workers spawned, zero dataset or mapped
targets opened, both V12 report leaves remained absent, the immutable manifest
file SHA-256 remained
`e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360`,
and both GPUs remained idle. No retry occurred or is authorized.

This is a terminal launcher/source-contract defect, not a dataset or science
result. V12 must not be invoked again.

## Preserved V12 contract

V13 is a standalone audit-only successor. It preserves every accepted V12
behavior byte-semantically, including:

- the immutable Builder V9 dataset, manifest, inventory, pair/endpoint order,
  counts, sample commitments, and all eight arrays;
- the complete V9/V10/V11/V12 authority, review, terminal-failure, BLOCK, and
  authorization lineage;
- Builder V9 endpoint-context, content-provenance, frame-key, geometry,
  raycast, evidence, raster, dtype, shape, byte-order, and comparison rules;
- the five-selector worker isolation correction, native-thread-one contract,
  spawn initializer/task ordering, and one/six-worker parity;
- atomic no-replace publication, predecessor preservation, source monitoring,
  fsync, terminality, and success/failure arithmetic;
- exactly six workers, one fresh attempt, no retry, no fallback, no rebuild,
  no predecessor exact entry, and no downstream authority.

No V13 change may affect dataset access, audit samples, recomputation, numeric
operations, report arithmetic, success criteria, inventory, array bytes,
publication, or access-ledger semantics.

## Sole operational correction

The V13 CLI must remain usable by this exact script-form invocation with no
caller `PYTHONPATH`:

```text
/home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/audit_go2_shared_jepa_v5_raw_supervision_v13.py \
  --authorization-sha256 <V13_AUTHORIZATION_FILE_SHA256> \
  --workers 6
```

Before importing NumPy, `lewm`, the V13 auditor, or any third-party module, the
CLI must:

1. set the four frozen native math-thread variables to `1`;
2. set exactly the five frozen device selectors to the empty string;
3. remove `HSA_OVERRIDE_GFX_VERSION`;
4. resolve its own reviewed path and derive the literal repository root as its
   parent-of-`scripts`;
5. require the resolved current working directory to equal that root;
6. reject a non-directory root or a root missing the literal `lewm` package;
7. remove any existing occurrences of that exact root from `sys.path` and
   insert the exact resolved root once at index zero; and
8. only then import argparse, JSON, the V13 auditor, or any project module.

It may not consume `PYTHONPATH`, `PYTHONHOME`, user-site state, a caller path,
an alternate repository, a symlink alias, or an installed `lewm` package. The
authorized outer environment must unset `PYTHONPATH`, `PYTHONHOME`,
`PYTHONSTARTUP`, `PYTHONUSERBASE`, and `HSA_OVERRIDE_GFX_VERSION`, set
`PYTHONNOUSERSITE=1`, set all five selectors empty, and set all four native
thread variables to one.

The CLI change is package resolution only. It grants no alternate exact entry,
test hook, callback, retry, or fallback.

## V13 source and proof namespace

Fixed implementation author: `/root/raw_v11_builder_auditor_diff`.

Production closure:

1. `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v13.py`;
2. `scripts/audit_go2_shared_jepa_v5_raw_supervision_v13.py`.

Proof closure:

1. `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v13.py`;
2. `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_author_handoff_2026-07-14.md`.

Canonical review:
`docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_independent_review_2026-07-14.json`.

Canonical authorization:
`docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v13_authorization_2026-07-14.json`.

The only exact production entry is:

```python
execute_exact_audit_v13(*, authorization_sha256: str, workers: int)
```

The only possible output leaves are the immutable V9 dataset path plus
`.audit_v13.json` or `.audit_v13.failed.json`.

## Required proof

All author/reviewer tests are CPU-only, source-only, temporary-root,
accelerator-hidden, and native-thread-one. They must not open canonical data,
`.generated` payloads, RGB, checkpoints, GPUs, G2, held-out, or navigation.

They must:

1. independently rehash and deep-validate the complete V12 source/review/
   authorization/witness/launch-failure chain and both absent V12 report paths;
2. run the exact-shaped V13 script command from the repository root with
   `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, and `PYTHONUSERBASE` absent and
   require `--help` to exit `0` after importing the real V13 auditor;
3. run that command from another current directory, through a symlinked script,
   with a duplicate/foreign root on `sys.path`, and with an installed-package
   decoy, requiring fail-closed behavior before auditor entry or mapped opens;
4. use an import spy to prove environment sanitation and the one exact root
   insertion occur before NumPy/project import;
5. prove the exact CLI exposes only authorization hash and worker arguments,
   strict workers `[1,6]`, and no smoke/exact callback or alternate entry;
6. mechanically compare V13 and V12 science/transaction/worker ASTs and permit
   only namespace, transitive authority, V12 failure bindings, and CLI root
   bootstrap differences;
7. rerun all `289` V12/retained tests unchanged;
8. rerun hostile real-spawn initializer/all-task proofs and one/six-worker
   canonical science plus eight-array parity;
9. prove complete authorization validation happens before mapped target or
   dataset access and deep-validates the exact V12 authorization and its 25
   bound targets, canonical witness, launch failure, and V12 success/failure
   absence;
10. prove success/failure transactions protect and preserve the V9/V10/V11/V12
    predecessor evidence and absence commitments; and
11. run `py_compile`, whitespace checks, focused V13 tests, and every applicable
    V12/V11/V10/V9 Builder/Auditor suite.

## V13 authorization closure

The future V13 authorization uses an ordered unique source map with exactly:

1. `amendment`: this V13 amendment;
2. `v12_audit_authorization`: frozen V12 authorization;
3. `v12_authorization_witness`: frozen V12 authorization fingerprint witness;
4. `v12_launch_failure`: frozen V12 launch-failure record;
5. `v12_auditor_source`: frozen V12 source;
6. `v12_auditor_cli`: frozen V12 CLI;
7. `v12_auditor_test`: frozen V12 test;
8. `v12_auditor_handoff`: frozen V12 handoff;
9. `v12_auditor_review`: frozen passing V12 review;
10. `auditor_source`: frozen V13 source;
11. `auditor_cli`: frozen V13 CLI;
12. `auditor_test`: frozen V13 test;
13. `auditor_handoff`: frozen V13 handoff;
14. `auditor_review`: passing different-agent V13 review.

Phase one must validate the complete 14-role structure without opening a mapped
target. Phase two must open only those exact targets, deep-validate the V12
authorization and its exact 25-role source map and nested V9/V10/V11 closure,
validate the non-authorizing witness and launch failure, require both V12 report
leaves absent, and bind the immutable V9 manifest file/content/inventory.

The authorization schema sets only `exact_audit_v13_authorized=true`. All V9,
V10, V11, and V12 audit authority plus build, rebuild, retry, RGB decode,
dataset use, training, selection, calibration, G2, held-out, runtime,
navigation, hardware, production, promotion, and deployment authority are
false.

The future reviewer must start with `/root/` and differ from amendment author,
implementation author, every frozen V9/V10/V11/V12 implementation author and
reviewer, the V12 authorization witness, the future V13 authorization
publisher, and the future V13 fingerprint witness. `/root` is the fixed V13
authorization publisher. The fingerprint witness must differ from publisher,
amendment author, implementation author, reviewer, and every V13 candidate
author/reviewer.

## Sequence and non-authority

1. Freeze this source-free amendment before any V13 source exists.
2. The fixed author constructs only V13 source, CLI, tests, and handoff without
   canonical-data, `.generated`, RGB, exact, or GPU access.
3. An eligible different agent publishes one canonical `PASS` or `BLOCK` review.
4. Only `PASS` permits `/root` to publish the V13 authorization.
5. A distinct eligible agent independently reproduces the authorization hash.
6. Only then may one V13 six-worker exact audit run, serialized with every
   `.generated` mutator and using the exact environment and script invocation
   above.
7. A terminal failure grants no retry and requires another source-free
   successor.
8. A PASS report still does not authorize dataset use or training; that
   requires a separate later authorization.

This amendment grants only source construction and different-agent review. It
grants no exact execution, retry, rebuild, dataset use, training, selection,
calibration, G2, held-out, runtime, navigation, hardware, production,
promotion, or deployment authority.
