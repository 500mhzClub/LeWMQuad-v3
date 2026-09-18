# Completed footprint receipt-copy comparison

The candidate completed the original 1,428-observation paired replay. All 1,425
forecasts and complete normalized decisions matched, and all seven retained
memory/map/residual/history checkpoints matched the completed packed baseline.
The exact original owner ended without a failure artifact.

| Window | Baseline total seconds | Candidate total seconds | Baseline median ms | Candidate median ms |
| --- | ---: | ---: | ---: | ---: |
| All 1,425 planning observations | 992.266 | 968.217 | 667.7 | 647.9 |
| Early navigation, 10 observations | 5.346 | 4.778 | 513.1 | 474.3 |
| Repeated hold, 10 observations | 7.155 | 7.212 | 654.6 | 656.9 |
| Late navigation, 10 observations | 8.991 | 8.258 | 833.7 | 779.4 |

The total planning-time reduction was **2.423571054%**. The hold window was
slightly slower. Every one of the 1,425 planning calls exceeded 100 ms in both
controllers. This is an incremental, alternating-order, shared-host,
controller-only comparison without profiling. It is not a real-time result;
its percentage must not be added to or multiplied by percentages from earlier
runs with different timing populations.

The verifier checked all 2,197 original source bindings within a 2,199-path
checker closure, authenticated both outputs, reauthenticated the completed
packed reference and actual raw/model bindings, reconstructed all 1,428 row
links and timing windows, and matched the complete report and seven prior state
identities. It did not rerun neural inference or full training ancestry.

- Result SHA-256:
  `3e82c140151b4e0976b65bd4b439c519f87a89e4f4df3cb9b2975df7324fe56d`.
- Launch SHA-256:
  `dc3d4c80ef00d1cf183414459ede26f7f24d1879c5829472bfeb44faa4697388`.
- [Completion verification](go2_receipt_copied_footprint_completion_verification_2026-09-11.json):
  `3426bf9cab4431bd87b6451ce25d28d9904f348f467cc63e602ba1fabd0dd951`.
- Checker: `scripts/verify_go2_receipt_copied_footprint_completion_v1.py`,
  session 50975, exit zero.

The original sensing failure at frame 1173 and failed round trip remain in the
evidence. No independent navigation, JEPA advantage, memory benefit, native
controller adoption or hardware qualification follows from this optimization.
The already frozen native diagnostics remain unchanged.
