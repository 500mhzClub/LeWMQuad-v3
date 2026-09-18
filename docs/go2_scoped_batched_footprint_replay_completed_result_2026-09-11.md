# Combined optimization replay completed and authenticated

The original combined replay completed its final input admission and exited
0 in session 94047. Completion authentication checked all 2,153 original
source bindings, both output bindings, all 1,428 ordered comparisons, all
1,425 forecasts, unchanged model identity and seven retained-state checkpoints
against the completed scoped-only reference. The original owner PID 2766980,
creation time 1789083663.49, ended.

Result SHA-256:
`5ebc45e317e217a39a144eacbb0029e3a57e59cf9713c27a1a0cf26c8d0cb5d0`.
Launch SHA-256:
`b6452eac3ed27de8df47ca336e7be906727b1e98198616e172d4f9261cd95f63`.
Verification record:
`docs/go2_scoped_batched_footprint_replay_completion_verification_2026-09-11.json`,
SHA-256 `2e7532d5b0c64b113cbb0de1d438dac9c245c5ee214579eb035012d0c29ae060`,
session 17660, exit 0. The checker uses the new profiler's 2,157-source union.
It did not repeat controller inference or the original full training ancestry.

| Window | Total controller time reduction | Scoped median | Scoped + batched median |
| --- | ---: | ---: | ---: |
| All 1,425 planning observations | 27.92% | 1.170 s | 0.795 s |
| Early frames 3–12 | 2.15% | 0.592 s | 0.586 s |
| Repeated hold frames 395–404 | 21.85% | 0.943 s | 0.749 s |
| Late frames 1418–1427 | 45.49% | 2.078 s | 1.074 s |

All candidate planning calls still exceeded 100 ms. Measurements alternate
controller order on a shared host and exclude acquisition; no isolated
benchmark or real-time qualification is claimed. The earlier scoped-reuse
speedup was measured in a different run and must not be added to this result.
The failed original round trip and visibility failure at frame 1173 remain
unchanged. This replay executed no new physical command.

The focused combined-controller profiler is prepared to identify remaining
costs in the same three fixed windows while reconstructing all 1,428 frames.
It must match every profiled decision to this completed replay. Preparation
passed 31 tests in 3.07 seconds (session 63671, exit 0) and source preflight
(session 2568, exit 0). Its frozen preparation record is
`docs/go2_scoped_batched_footprint_late_history_profile_preparation_2026-09-11.json`,
SHA-256 `7ff61c631305a8975517738766bd95413db4ff0f3a897a00fe4cc7c592416c30`.
The new diagnostic rehashes the actual raw episode and bound model inputs
before and after, reusing this completed replay's full training provenance
without falsely claiming to rerun that ancestry.
