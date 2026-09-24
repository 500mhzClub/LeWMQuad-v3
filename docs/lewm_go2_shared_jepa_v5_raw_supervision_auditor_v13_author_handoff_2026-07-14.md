# Shared JEPA V5 raw-supervision Auditor V13 author handoff

Date: 2026-07-14

Implementation author: `/root/raw_v11_builder_auditor_diff`

Status: **source-only candidate complete; different-agent review required; no exact authority**

## Frozen governing amendment

This candidate implements only
`docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_self_resolving_cli_successor_amendment_2026-07-14.md`,
file SHA-256
`094072a8289e69a894310a1a327327ee92e7af5e448c39a8d2f6c9e0b3c008ed`.

The implementation did not open the canonical dataset, any `.generated`
payload, RGB, checkpoint, G2, held-out, runtime, navigation, hardware, or
production artifact. It did not run an exact audit, retry, rebuild, training
job, navigation job, or GPU job. All executable proof was CPU-only and used
source files or temporary synthetic roots with native math threads fixed to
one and accelerators hidden.

## Candidate artifacts

| Role | Path | File SHA-256 |
| --- | --- | --- |
| Auditor source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v13.py` | `fddc678187f082a0a245ff5868ca5d944cba4adc2703d3b97088d57451deb4b7` |
| Auditor CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v13.py` | `c7b2018f9296d92ab0abf3745a8afa5108a7404496fa382a7f75bd3b7307ba4b` |
| Auditor test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v13.py` | `7fb40f59be369ec35852cc10604a2bd8c0a08f083d19403ef1eb7b9c759d4c7e` |

The handoff intentionally does not self-hash. The independent reviewer must
compute its file hash after these bytes are frozen.

## Implemented correction

V13 preserves V12 science, workers, comparison, report arithmetic, access
ledger, and atomic publication behavior. Its only operational correction is a
self-resolving CLI bootstrap that runs from the reviewed repository root with
no caller `PYTHONPATH`. Before importing argparse, NumPy, `lewm`, or any project
module, the CLI:

- fixes all four native math-thread variables to `1`;
- clears exactly five accelerator selectors and removes
  `HSA_OVERRIDE_GFX_VERSION`;
- rejects caller Python path and user-site state;
- validates the exact regular script, literal repository root, current working
  directory, and literal non-aliased `lewm` package; and
- removes script/root duplicates and installs the one exact repository root at
  `sys.path[0]`, rejecting aliases and foreign or installed `lewm` packages.

V13 adds a 14-role outer authority closure. Phase one validates its complete
structure with zero mapped-target opens. Phase two deeply validates the frozen
V12 authorization and exact 25-role nested map, V12 PASS review, independent
authorization witness, terminal launch-failure record, the nested V9/V10/V11
lineage, and absence of both V12 report leaves. Success and terminal-failure
publication preserve and protect all V9/V10/V11/V12 predecessor evidence.

## Author proof

The focused V13 suite passes `73 passed in 1.83s`. It includes:

- exact script-form `--help` success from the repository root with
  `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, and `PYTHONUSERBASE` absent;
- import-spy proof that sanitation and one-root installation precede project
  import;
- fail-closed foreign-working-directory, symlinked-script, root-alias,
  duplicate-root, and foreign/installed-package-decoy cases;
- exact CLI surface and strict worker-range proof;
- independent rehash of the nine frozen V12 outer artifacts, semantic
  validation of the exact 25-role V12 map, and witness/launch tamper rejection
  before nested target opens;
- zero-open phase-one, alias rejection, closed V12-to-V13 AST deltas, and exact
  science, worker, transaction, and raw-manifest parity;
- hostile real-spawn initializer/all-task isolation and identical one/six-worker
  canonical science plus all-eight-array bytes; and
- real success/failure publication proving V12 authorization, witness, launch
  record, success absence, and failure absence are preserved and protected.

The unchanged V12, V11, and retained V10/V9 Builder/Auditor suites pass exactly
`289 passed in 5.03s`. The combined V13 and retained proof passes
`362 passed in 6.34s`. Source, CLI, and test pass `py_compile`; whitespace
checking found no trailing whitespace. The proof environment hid all five
accelerator selectors, unset `HSA_OVERRIDE_GFX_VERSION`, fixed all four native
thread variables to one, and disabled user-site and automatic pytest plugin
loading.

## Reviewer boundary

The next action is an independent source review by an eligible different
agent. The reviewer must independently hash all four candidate artifacts,
rehash and deep-validate the complete V12 authorization/review/witness/launch
chain, reproduce the source-only CPU proof, and publish one canonical `PASS` or
`BLOCK` review at the amendment's fixed path.

Only a PASS review permits `/root` to construct a V13 authorization, followed
by an independent authorization fingerprint witness. This handoff grants no
exact audit attempt, retry, rebuild, RGB decode, dataset use, training,
selection, calibration, G2, held-out, runtime, navigation, hardware,
production, promotion, or deployment authority.
