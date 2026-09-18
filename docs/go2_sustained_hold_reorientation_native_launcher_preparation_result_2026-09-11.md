# Sustained-turn native launcher preparation

At 2026-09-11 08:57 UTC, the native launcher was implemented and its source
preflight passed. The focused launcher, input-admission, native-prefix and
collector/auditor source tests passed: 92 tests in 4.17 seconds, session 54654,
exit 0. Source preflight session 87190 exited 0 with 2174 bound source paths,
78,031,974,400 bytes available RAM and 608,536,981,504 artifact bytes free.

Preparation record:
`docs/go2_sustained_hold_reorientation_native_launcher_preparation_2026-09-11.json`
SHA-256 `a1bd44e864f660a5a2134599690fd3abacf627ac4eccfcdd631ed18d4527fb52`.
All source paths in that record are now frozen for this preparation.

An actual invocation with placeholder completion identities rejected the live
original raw replay before runtime input admission, model construction or
output creation: `original sustained raw replay is still live`. The planned
native output remains absent. This verifies the live-owner gate; it does not
constitute completed input admission or native evidence.

The next execution step is to prepare a one-shot waiter that admits the
completed original raw replay and completed original extended-budget waiter,
authenticates their full existing completion chains, and invokes this launcher
once. Preserve current owners and the original queue. Do not restart or replace
any current experiment. The fresh sustained-turn pilot cannot start before the
raw replay and original five-stage queue complete.

At the status check, the sustained raw replay had emitted frame 200 of the
407-observation prefix. The contact-scoring native worker had completed
collection and raw audit, while its parent was still performing final input
verification. Its worker readout reported 572 observations, five distinct
outbound edges, no arrival, no round trip, no native contacts, strict visibility
pass, and SENSOR_OR_MODEL_FAILURE. Median full iteration was 1127.958 ms and
all 572 samples exceeded 100 ms. These are worker results pending final run
completion; no final contact result existed at the check. Tracking and
extended-budget native outputs were still absent and their waiters remained
live.

The six-case comparison and completed frontier/hold follow-ups have no verified
round trip. One comparison baseline reached the outbound goal without returning.
Independent-layout comparative navigation, useful planning/memory effects,
real-time operation and bounded hardware evidence remain outstanding. No
independent-study policy was selected and the overall goal remains active.
