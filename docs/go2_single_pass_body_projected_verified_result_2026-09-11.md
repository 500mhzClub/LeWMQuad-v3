# Single-pass body-projected replay result

The full paired replay and its ended-owner completion verification passed.
All 1,428 decision/input comparisons, 1,425 learned forecasts, and seven
observed-state comparisons agree under the predeclared normalization. The
original sensing failure at frame 1173 is retained. The completion verifier
reauthenticated the original raw/model inputs and all 2,390 bound sources.

| Timing window | Baseline total | Candidate total |
| --- | ---: | ---: |
| All 1,425 navigation decisions | 780.584101 s | 771.637477 s |
| First 10 navigation decisions | 4.236036 s | 4.343583 s |
| 10 repeated-hold decisions | 6.060395 s | 5.863949 s |
| Last 10 navigation decisions | 6.846202 s | 7.011704 s |

Total navigation time is 1.146145% lower in this paired run. Median decision
time changes from 517.523390 ms to 502.139332 ms. All 1,425 decisions in each
arm exceed 100 ms. Early and late window totals regress. This is a small, mixed
timing result, not a resolution of the real-time gap or a statistically
replicated speed claim. Do not add its percentage to earlier separate runs.

The candidate changes only the eight measured-sample bounds indices to the
existing single-pass query implementation. Packed insertion, body projection,
receipt handling, model, observer, and decision logic retain their declared
semantics. No profiler was attached. Native experiments have not adopted this
candidate, and the replay establishes no navigation or sensing qualification.

Artifact root:
`go2_single_pass_body_projected_late_history_v1_attempt_001` under the fixed
navigation development artifact root.

- Launch SHA-256: `86e325f68d0f7d5f389e911009c9ccdf1b6c3d24291e9a2c974af838ed1307d8`.
- Result SHA-256: `6144645f568d99ccf89b31a76e586277a315f840e2da6db835c06d0519150c44`.
- Complete comparison SHA-256: `b31a5b2d9aa89d3c23a07905c2f6b1de24267ab59910c93ad0b849caea520d54`.
- Completion receipt: `go2_single_pass_body_projected_completion_verification_2026-09-11.json`,
  SHA-256 `f9833329610096f9d5776ccc208d0de33988d8ebd4e6cdff4895004aa5190f1e`.

The original runner session 51355 and verification session 43837 exited zero.
No second full replay, automatic retry, or replacement completion receipt was
created. The full CPU replay slot is free; the native worker remains governed
by its existing queue and completion checks.
