# Accounting-only correction to the scan orientation diagnostic

V1 terminated before integrating any trajectory: its raw-array grouping called
`array_binding(raw)` as a dictionary key, but that helper returns a per-array
binding dictionary, not a single hash. Its failure is retained unchanged at
`.generated/go2_active_scan_orientation_diagnostic_development_v1_attempt_001`.

V2 hashes the canonical sorted per-array binding dictionary. Array names, shape,
dtype and all bytes therefore remain bound; dictionary insertion order does not
affect group identity. The three numerical methods, two data rates, all sixteen
physical traces, original decision times and error endpoints are unchanged.
The V1 source and launch/failure hashes are also bound. No original simulation,
RGB capture, live controller, audit or task result is rerun or revised.

Read with the unchanged [V1 diagnostic design](go2_active_scan_orientation_diagnostic_development_v1_2026-09-05.md).
Regression tests exercise dictionary-key use, array ordering, dtype/shape/value
sensitivity and an actual recorded scan's exact live midpoint replay before
running the complete corrected panel. That preflight is an integrity check, not
selection among numerical results.

Exact fresh root: `.generated/go2_active_scan_orientation_diagnostic_development_v2_attempt_001`.
