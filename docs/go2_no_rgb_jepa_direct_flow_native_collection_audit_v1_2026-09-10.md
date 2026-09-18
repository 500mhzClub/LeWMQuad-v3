# Tracking-recovery native collection and full audit source

The prospective no-RGB JEPA maze-02 collector and auditor use exactly the
DirectFlowResidualAnchoredController used by the full-controller prefix. They
derive from the bound original residual-anchored collector and auditor at
SHA-256 identities `4e98a79ea5ad16c1d2021deae52abd2aec0a38d4ace1df80d2ca3994d990e949`
and `6ae658b91a0859b8d4b7ae99ba9da22dc2541183c4a52409301bd857a24675f3`.

The only source changes are the controller import/construction, two collection
status strings, and explicit fallback identity fields plus their audit check.
Whole-module AST comparison verifies that the physical loop, sensor capture,
renderer witnesses, command endpoints, storage stops, terminal zero drain,
failure persistence, raw reconstruction, model-state checks, contact evaluation,
timing records and strict visibility/round-trip success calculations remain the
same. Tests also ensure that this comparison cannot hide a shortened physical
episode or a relaxed visibility success gate.

The collector and auditor remain parameterized internal functions, as in the
original implementation. Their names alone do not bind model/case identity or
grant execution authority. A future launcher must enforce the exact maze-02
no-RGB JEPA assignment and model digest, authenticate a positive completed
full-controller prefix using the prepared native-prefix checker, wait for the
existing six-case/frontier/hold/contact queue, verify all original model inputs,
refresh resource admission, and own exactly one fresh scene. It must retain
collection artifacts if later auditing or prefix comparison fails.

The full audit uses a fresh model/controller and reconstructs every recorded
decision from raw public sensors. Native pose remains evaluator-only. Prefix
comparison separately checks the 43,700 shared pre-intervention physics samples,
the complete 860-observation decision prefix and actual execution of command
859. A measured hold remains a hold, and subsequent physical outcomes require
the full evaluator. No partial replay or prefix result is navigation success.

This is source preparation only. No native launcher, simulator, generated
sensor input, new checkpoint or physical outcome is created by preparing or
testing these modules. The goal remains active pending end-to-end evidence.
