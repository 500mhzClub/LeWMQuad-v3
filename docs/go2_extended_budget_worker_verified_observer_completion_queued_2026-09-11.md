# Completed native worker and automatic observer verification

The original extended-budget native worker has ended. Its exact terminal
record and all 23,124 worker artifacts are authenticated, and its physical,
contact and timing readout reconstructs from the persisted collection/audit and
complete physical contact trace. All 2,144 admission source bindings remained
unchanged before and after verification.

- Worker terminal SHA-256:
  `74520a1e3fd9a92af486454a7d788a90659138a8b60fc93212debe8d109c75f2`.
- Admission receipt: `go2_extended_budget_completed_worker_admission_2026-09-11.json`.
- Receipt SHA-256:
  `2e748a012979225e1871d79a7240fe9bf1a14da54a2aeca3ea7d6db9b1fbeedf`.
- Admission session 55075 exited zero.

The worker result remains negative for a complete round trip. It records one
physically confirmed outbound arrival at frame 2935, 15 outbound crossings
over 10 distinct open edges, then three return crossings over three distinct
edges. There are zero contacts across 193,100 physics samples and the strict
physical visibility checks pass. The return stops at the previously
reconstructed floor-height rejection. Median observation/control time is
1537.235129 ms; all 3,848 samples exceed 100 ms. Physics remained paused during
computation. There is no real-time or hardware qualification.

This is completed-worker admission. The original parent and queue waiter have
their own final checks; their completion is not granted by this receipt.
Raw/model inference, physical-prefix comparison and training ancestry were not
rerun by this admission script. Original worker outputs and failures remain
unchanged. The admission rehashes all worker artifacts and reexecutes the
existing worker contract and physical readout checks.

Automatic completion verification for the separate measured-plane observer
history is now active:

- Waiter root: `go2_measured_plane_observer_completion_wait_v1_attempt_001` under
  the fixed navigation development artifact root.
- Waiter launch SHA-256:
  `188bbb617aad2f4cbd8c312b164d9e47f3ff5e7c5bdaf890f89777f62ce3c93c`.
- PID `2904406`, creation time `1789156891.38`, session `19672`.
- Exact command:
  `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/await_go2_measured_plane_observer_completion_v1.py`.
- Boot `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`; 2,431 bound sources.
- Preflight session 70381 exited zero before launch.

The waiter checks the exact original observer PID/create-time/command every
30 seconds. It will invoke the prepared checker once, using the actual result
SHA-256, only after that owner ends. It preserves execution failures and never
restarts the original job. Do not run the checker manually while the waiter is
live. The CPU replay/verification slot remains occupied until both stages end.

The observer was live and had passed frame 2550 without candidate failure at
the most recent progress check. This is not a completed observer result or a
navigation recovery claim. Its goal is to test a new causal estimation history
on the fixed executed trajectory before any new controller or native outcome
can be claimed.
