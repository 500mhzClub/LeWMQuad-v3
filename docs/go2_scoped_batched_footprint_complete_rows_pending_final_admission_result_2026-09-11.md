# Complete paired rows verified; final replay admission pending

All 1,428 saved comparison rows are present, ordered and authenticated against
the completed scoped-only reference. Every original public-input and decision
binding matches that reference. Every new baseline decision matches the prior
scoped candidate, and every new row reports exact normalized candidate
decisions and unchanged public arrays. The comparison file hash was stable
before and after this verification; the 2,153 frozen source bindings and
original launch were also checked.

These are provisional timing results from the complete saved row population:

| Window | Total controller time reduction | Scoped median | Scoped + batched median |
| --- | ---: | ---: | ---: |
| All 1,425 planning observations | 27.92% | 1.170 s | 0.795 s |
| Early frames 3–12 | 2.15% | 0.592 s | 0.586 s |
| Repeated hold frames 395–404 | 21.85% | 0.943 s | 0.749 s |
| Late frames 1418–1427 | 45.49% | 2.078 s | 1.074 s |

All 1,425 candidate planning calls still exceeded 100 ms. Measurements cover
controller observation calls, use alternating execution order on a shared
host and exclude sensor acquisition. They do not establish real-time control
or an isolated benchmark. Do not add or multiply these percentages with
earlier experiments measured under different shared-host conditions.

The original owner PID 2766980, creation time 1789083663.49, remained live.
No terminal result or failure existed at verification. The original final
input admission and terminal model/state report have not yet been accepted.
This row-level check does not substitute for them and does not authorize
adopting the candidate as a verified completed replay. No model or controller
was reexecuted by this check, and no new physical command was issued.

Verification record:
`docs/go2_scoped_batched_footprint_complete_rows_pending_final_admission_2026-09-11.json`,
SHA-256 `91a7eeb0fda653c798fc1b39e01861c2cc9ed73de0f4e625593a37b0135f261d`,
session 64221, exit 0. Once the original process publishes a terminal result,
authenticate that result, its final model/state checks and all output bindings
before treating the replay as complete. Further timing work should profile
the combined controller before choosing another optimization.
