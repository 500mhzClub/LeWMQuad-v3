# Deferred-memo replay completion verification

The exact full paired replay launch is
`f7b6d7a7e49373f236019731454622fb2f8a473b29c0aeb826e0e0bdcbcdd8fb`,
with owner PID 2910940, creation 1789159905.04, on the original recorded boot.
The automatic waiter polls that exact owner every 30 seconds. It does not infer
completion from time elapsed and never restarts or replaces the original job.

After the owner ends, preserve any failure. Otherwise record the actual result
SHA-256 and invoke `scripts/verify_go2_deferred_memo_single_pass_completion_v1.py`
once in the waiter process. Reconstruct all 1,428 comparison rows, 1,425 forecast
count, seven state-hash witnesses and all timing windows from the original
authenticated baseline. Preserve the original sensing-failure scope. Recheck
raw/model input admission and all source/artifact identities. No observer or
model inference is rerun by this completion checker.

The checker passed 13 synthetic tests, including full row populations,
truncation, changed input/decision identities, altered state witnesses, bad
timing or execution order, incorrect scope and live/different-boot owners.

Bind the checker, test, waiter, protocol and original source closure before
registration. The waiter root is
`go2_deferred_memo_completion_wait_v1_attempt_001` under the fixed development
artifact root. The output receipt is
`docs/go2_deferred_memo_single_pass_completion_verification_2026-09-11.json`.
No retry, overwrite, native execution, navigation qualification or real-time
claim belongs to this completion workflow. Do not run a second checker while
this waiter owns the verification stage.
