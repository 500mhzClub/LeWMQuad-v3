# Optimized-controller profile: execution started

One CPU profiling run started under PID 2721383, creation epoch
1789059710.11, tool session 88157. It has reported
`FROZEN_FOOTPRINT_CONTROLLER_PROFILE_INPUT_ADMISSION_STARTED`.
Its exclusive root is
`go2_frozen_footprint_controller_windows_profile_v1_attempt_001` under
recovery-storage navigation artifacts. The output root is created only after
the complete original input admission; absence of that root during admission
is not a stopped process or permission to retry.

The exact command is:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/profile_go2_frozen_footprint_controller_windows_v1.py
```

Execution record: `go2_frozen_footprint_controller_profile_execution_2026-09-10.json`,
SHA-256 `9f3a2434c3cfd3423e81b4003acd72a222c222b1644b1127e813f0e3f00f12f0`.
It binds the exact owner, all 2,008 prepared sources, resource inspection,
completed paired replay and independent verification, and test/preflight
results. Preserve these sources and all predecessor evidence while it runs.

Sixteen focused tests passed in 2.17 seconds (session 32360, exit zero).
The initial fixture had five failures because it used sys.getprofile() as a
cProfile activity indicator. The corrected fixture checks actual profile
records: exactly ten observe calls in each fixed window and no normalization
calls within profiling. The initial failed test session 33303 is retained in
the execution record; it was not a native or real-data attempt.
Final source preflight passed with 2,008 bindings (session 32990, exit zero).

Before launch, available RAM was 70.89 GiB, artifact headroom was 594.81 GiB,
and 16 physical CPUs were available. Original direct-model native worker
2709978 remained live under batch parent 2659758. All 1,988 bindings in the
original contact-native waiter closure were rechecked unchanged. The existing
single-scene native queue retains its order.

On completion, authenticate the final result, source and artifact bindings;
reconstruct saved original/candidate decision and input hashes against the
completed paired replay; and recompute the guarded summaries from both raw
profile files. Use exclusive times for additive accounting and keep overlapping
cumulative times separate. The profile is diagnostic and includes overhead.
It does not replace the completed unprofiled timing result or add navigation,
independent-layout, real-time or hardware evidence.
