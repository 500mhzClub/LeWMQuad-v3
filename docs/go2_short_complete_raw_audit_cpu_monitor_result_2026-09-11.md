# Complete short raw audit passed CPU monitoring

The full original 14-observation development audit completed under CPU
instrumentation. Its complete report matched the saved original report
exactly, including the unsuccessful navigation outcome. All 1,400 physics
samples and 13 command intervals were included by the original auditor.
No new command or independent-layout observation was collected.

Result SHA-256:
`f7118d416212d5a4758f15274e8c8596f68382b07ea8ab40592ca5922b87bacb`.
Launch SHA-256:
`66ed12f6245c5840dc4cde0dcdc5c3eba8668924bd1f96f4280645167a2edc12`.
Output root: `go2_short_complete_raw_audit_cpu_monitor_v1_attempt_001`.
Execution session 50510 exited 0; original owner PID 2771738, creation time
1789086434.6, ended. No retry or replacement was started.

The scope included original model loading, robot geometry construction, raw
sensor reconstruction, controller replay, command checks, physical stop and
contact checks, primary and auxiliary visibility, renderer witnesses and
round-trip evaluation. The original interface failure was deliberately
preserved. This episode contains no successful learned action selection.

The monitor recorded 4,037 CPU tensor operations, 3,935,571 Python calls and
6,303,459 native calls exposed through Python, with no recorded violations.
OpenCL was disabled before interpreter import. No monitored Genesis runtime
entry or accelerator initialization occurred. The bounded monitored work
and report comparison took 9.825805173953995 seconds; this is neither an
isolated latency benchmark nor a measured collection/audit speedup.

The worker authenticated the original source, completed reference, bound
model inputs and complete original raw artifact bindings before and after.
Completion verification separately checked all 2,045 source bindings, four
outputs, exact saved-report equality and the monitor record. It did not
rerun inference or repeat the complete original raw-input admission.

Verification record:
`docs/go2_short_complete_raw_audit_cpu_monitor_completion_verification_2026-09-11.json`,
SHA-256 `346f5886a856bb1d902b30059b147e8c01023526816d4ae0d4868b62f7881c96`,
session 94908, exit 0.
Preparation record:
`docs/go2_short_complete_raw_audit_cpu_monitor_preparation_2026-09-11.json`,
SHA-256 `3ce009f25a7555cef2770fcd0759aef039010e0ccf9ce7497c41a08e48c2c0b2`.
The preparation passed 27 focused tests in 4.68 seconds (session 85743,
exit 0) and source preflight (session 32545, exit 0).

This result and the completed four-arm factory startup provide separate
bounded evidence for common raw-audit routines and actual controller/model
paths. They do not constitute execution of the complete independent multiarm
auditor. Its reactive command audit, independent-layout evaluator and long
successful-planning branches were not executed by this probe. The inspected
reactive command checker and independent evaluator use NumPy and local
geometry/contract functions, but source inspection is not runtime coverage.

Next, integrate the frozen monitor around the actual independent-study audit
call in a source-bound successor worker, retain monitor evidence on failures,
and require it at parent acceptance. Review the remaining source differences
and make overlap admission depend on both completed evidence and runtime
enforcement. The current staged driver and its frozen predecessors remain
unchanged. Final population policy and queue/input review are still required.
No navigation, real-time, hardware or independent multiarm qualification is
claimed, and overlap is not yet enabled. Instrumentation is not an OS sandbox.
