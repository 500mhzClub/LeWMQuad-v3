# Native scheduling recovery and perception/performance integration

The complete deferred-memo paired replay finished and its automatic checker
verified all 1,428 observations, 1,425 forecasts and seven retained-state
comparisons. Original replay and checker owners have ended.

- Replay result: `c7435317af221a6f7f85339ac8717765b79d9417f105e869a1500476c59507cf`.
- Completion receipt: `docs/go2_deferred_memo_single_pass_completion_verification_2026-09-11.json`,
  SHA-256 `3a75cfb02aeea139c6b019287ecff01d32512f3bd418b2b33aadb7535cdda41f`.
- Completion waiter result: `86f4737d663de9a84e285b0cba13217e24629028d0e84e18b2a441d1345af421`.

Controller total time was 751.931519 s for the baseline and 750.743682 s for
the candidate, about 0.16% improvement. Medians were 493.460 and 491.266 ms;
all 1,425 post-warmup observations exceeded 100 ms in both arms. Early and
repeated-hold windows regressed. The component copier improvement did not
translate into a useful full-controller speedup. Preserve this result and do
not adopt that change based on the component benchmark. No navigation or
real-time improvement is established.

The completed extended-budget native waiter has result
`1e2da182445446b05de79c09825979a0e4547212d3bd1db5ce999fca4928fc70`.
Its completed native result remains
`c92f0bdf5cc8ebb9e513470492578ec3457d02196dfc7238e2f6b12b9c621b27`.
It verifies outbound arrival and the failed return, not a round trip.

All four later native waiters terminated before scene dispatch. The
sustained-turn guard classified the still-live read-only performance runner
as a competing native runner by filename. The remaining waiters stopped on
missing predecessor completions. Their original roots, failures and event
streams are preserved. No original native child root was created. See
`scripts/native_waiter_dispatch_abort_development.py` for the exact bindings.
The sustained-turn, contact/flow and chained-anchor physical diagnostics
remain unexecuted; their hypotheses are not counted as results.

Prepared a separate first measured-plane native execution in
`scripts/run_go2_measured_plane_dispatch_recovery_v1.py`, under
`docs/go2_measured_plane_dispatch_recovery_v1_2026-09-11.md`. It keeps the
original scientific worker code, controller, model, scene, sensors, full raw
audit, physical prefix and success criteria. It changes scheduling admission
and uses a new output root. It waits on occupied slots before creating output
and never relabels the failed queue as complete.

Recovery tests: **10 passed in 2.21 s**, session 61504, exit 0. The exact
four-failure admission also passed against actual artifacts. Source preflight:
session 69370, exit 0, 2,562 source bindings, 81.80 GB available RAM, 593.37 GB
artifact free space, CPU 3.5% busy and GPUs idle. The preceding CPU-slot
completion check passed, session 15731, exit 0.

The new native invocation is **session 4300**, PID **2916106**, creation time
**1789162140.42**, original environment interpreter with `-B` and
`scripts/run_go2_measured_plane_dispatch_recovery_v1.py`. It was confirmed
running in input admission before output creation. Admission then completed
and the new native attempt launched. Do not restart it because admission or
collection is quiet or takes time. New root:
`go2_measured_plane_dispatch_recovery_v1_attempt_001` on the artifact volume.
Launch SHA-256:
`93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb`.
It binds 2,562 sources. The exact parent owner was rechecked live, and scene
worker PID **2916239**, creation time **1789162190.15**, was confirmed running
under the original environment's multiprocessing spawn command. The resource
tracker is PID 2916238. Both share boot
`1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Sensor collection was subsequently
confirmed at tick **355** with the exact parent and worker still live, worker
RSS about 3.47 GB, and no worker terminal, parent result or failure file. No
terminal outcome or completed physical-prefix audit is claimed.

Separately, implemented `MeasuredPlaneSinglePassController` to combine the
revised estimator with the already completed single-pass performance changes.
It excludes the ineffective deferred-memo change. Actual synthetic image
integration tests passed for full-input JEPA and no-RGB direct heads, with
complete decision and retained-state equality: **3 passed in 26.46 s**, session
58076, exit 0.

The trained comparison is prepared in
`scripts/replay_go2_measured_plane_single_pass_prefix_v1.py`, under
`docs/go2_measured_plane_single_pass_prefix_v1_2026-09-11.md`. It compares all
123 verified observations and 120 forecasts, stops before the next observation
following the changed physical command, and checks four complete retained-state
snapshots. Its 18 accounting/loop tests passed in 2.34 s, session 66331. Source
preflight passed in session 71597 with 2,492 sources. The trained prefix replay
has now launched in **session 37088**, PID **2916360**, creation time
**1789162249.92**, original interpreter with `-B` and the replay source path.
Root: `go2_measured_plane_single_pass_prefix_v1_attempt_001`; launch SHA-256
`7b0b3bf417344ee90d3f74ecc6c2c613623119f98a4ee8273b8f6c3f61e4dc42`.
It binds 2,492 sources. The replay completed, session 37088 exited 0 and its
exact original owner ended. Result SHA-256:
`5d0e1b47ca80b8c19474dd7a371957867c71ed9835cc97bc342b19752a22f087`.
All 123 complete decisions, 120 forecasts and four complete retained-state
checks matched. Both assigned models remained unchanged. The frozen runner
rechecked every compact output against the full recorded decisions and
reauthenticated original raw inputs before and after execution.

Across all 120 post-warmup observations:

| Measurement | Original measured-plane controller | Combined controller |
| --- | ---: | ---: |
| Median controller time | 751.164 ms | 481.906 ms |
| Total controller time | 104.269993 s | 61.754157 s |
| Observations over 100 ms | 120 | 120 |

This is a **40.8% total-time reduction on the fixed short prefix**, not an
isolated or late-history benchmark. Sensor acquisition was outside the timing
interval. All decisions still missed the 100-ms target. No observation after
the changed physical command was consumed and no navigation outcome was
inferred. The fresh native experiment uses the original measured-plane
controller, independently of this replay.

A read-only ended-owner check revalidated the exact launch/result identities,
all 2,492 source bindings and three artifact bindings, the 123-row population,
four state receipts and complete timing-accounting reconstruction. Session
68793 exited 0. It did not rerun a model or observer. Do not change sources,
tests or protocols bound by the still-live native launch, which includes these
prepared replay sources through its CPU-slot admission dependency.

The broad goal remains active: no verified round trip with the revised
perception, no independent-maze reliability, no matched advantage claim,
no real-time operation and no real-platform qualification yet.
