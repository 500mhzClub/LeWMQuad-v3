# Recorded-sample check: input-root preflight failure

The first invocation of `scripts/check_go2_compact_depth_recorded_sample_v1.py`,
tool session 29519, exited 1 before output creation or archive conversion.
Its source SHA-256 was
`e90fc5cbfdc1953b75956c716772f5cc56c235654011f13e0e4a2ab2070e5c94`.
The prospective protocol SHA was
`217b33980a5cf3382def78469d3c602a7f958154e8c7e52d2e2b7a702bb88cda`.

The error was `ValueError: exact nonsymlink development attempt root required`
from `navigation_artifact_root_development.verify_artifacts`, called by
`input_identities`. The checker passed the nested episode directory to a
verifier that requires an immediate development-attempt root. The input files
themselves were not reported as changed or invalid.

A direct filesystem check confirmed that the exclusive
`go2_compact_depth_recorded_sample_v1_attempt_001` output did not exist and was
not a symlink. Thus there is no launched sample attempt, partial conversion,
sample result or failure file to overwrite. The checker source is outside the
2,656 paths bound by the active controller-prefix replay. That live experiment
continues unchanged.

The correction is to call the existing verifier on the original native attempt
root and prefix each of the same 26 selected artifact paths with its episode
directory. This preserves the original expected file hashes and selected-frame
scope. Revalidate that admission before any conversion. The source will be bound
at its corrected identity only when the exclusive launch is actually created;
the preflight failure above remains part of the record.

The corrected read-only input preflight passed all 26 selected bindings in
session 64630 and confirmed that the output was still absent. This witness
document is included in the subsequently prepared source roster.
