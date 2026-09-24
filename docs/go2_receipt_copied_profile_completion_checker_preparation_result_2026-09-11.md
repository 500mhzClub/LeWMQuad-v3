# Receipt-copy profile completion checker prepared

Prepared `scripts/verify_go2_receipt_copied_profile_completion_v1.py` while the
original profile continued. It passed 32 focused tests in 2.66 seconds and
verified its 2,206 source bindings. The actual live-owner check rejected
completion before reading any result or profiler output.

The checker accepts an explicit terminal result SHA-256 only after the recorded
owner ends on its original boot. It binds that result to the exact original
launch and source closure, rejects retained failure artifacts, and requires
the exact eight output artifacts. It reauthenticates the completed receipt-copy
reference and actual raw/model inputs, compares all 1,428 row identities, and
reconstructs all three full profiler JSON summaries from their pstats files.

The report check requires the exact model and 1,425 forecast comparisons,
three ten-frame windows, ordered observations, positive finite timings,
unchanged hold-window actions, matching recorded window timings, function and
module tables, and honest scope flags. It rejects unsupported speedup,
real-time, navigation or retained-state claims. Descriptive state-size
snapshots remain outside independent reconstruction.

At the latest check, profile PID 2828895 remained live and had reached frame
407. The original tracking child PID 2824049 also remained live, with advancing
CPU and read counters and no physics output yet. The sustained-turn and
contact-plus-flow waiters remained live. The profile occupies the single full
CPU replay slot; no second replay was started.

After the exact profile owner ends, obtain the hash of its completed result
and invoke the checker with `--result-sha256`. Only then inspect the complete
timing summaries to select the next optimization. A prepared checker is not
a completed profile or evidence of a new speedup.
