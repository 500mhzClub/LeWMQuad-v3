# Separate audit worker prepared

Implemented `scripts/independent_round_trip_separate_audit_development.py` with
parent-confirmed collection admission, a fresh audit-child requirement, the
fixed original multiarm raw auditor, failure preservation, distinct execution
receipts, and an ended-worker saved-evidence reader. Existing collection,
handoff, lifecycle, case-evidence and serial-runtime sources were unchanged.

The worker closes its audit log before case persistence. Collection and audit
logs are both bound in the complete saved-evidence roster. It never writes the
serial runtime's same-process execution receipt. Raw-audit failure, post-audit
data changes and persistence failure retain the original artifacts and a
worker-failure marker; no retry or population acceptance follows.

**112 focused tests passed in 7.35 seconds**, session 5637, exit 0. The suites
cover the new split audit, existing handoff and case evidence. Raw audit,
process admission and collection lifecycle receipts are synthetic in the new
suite; actual hashing, original case persistence and saved readout checks run.
The predecessor's real spawn process tests are separate earlier evidence.
The final change additionally checks audit ownership again after all saved
evidence admission and tests a change during that interval.

Preparation verified **2,021 source bindings**, session 15127, exit 0:
`docs/go2_independent_round_trip_separate_audit_preparation_2026-09-11.json`,
SHA-256 `d99d32d2591657fa2c9dd427784138819b7966212a61c8d745c1aeacc0e974c5`.

No actual separate raw audit, independent-layout collection, simulator scene,
or trained-model execution was started. At closing, the original case-5 audit
worker 2743870 (creation 1789071424.56) and scoped controller replay 2754886
(creation 1789077365.71) remained running. The replay's latest complete record
was frame 1365, with exact normalized decisions.

Next requirements before overlap are parent-owned audit zero-exit confirmation,
a bounded coordinator and role-aware single-scene admission, plus CPU-only
execution qualification and measured overlap resources. The saved reader
explicitly leaves parent exit verification and population acceptance false.
Final policy review and joined queue/input admission still precede the
independent study. Navigation reliability, real-time execution and hardware
qualification remain unproven.
