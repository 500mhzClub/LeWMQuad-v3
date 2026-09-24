# Receipt-copy controller profiling is running

Started the exact completed ReceiptCopiedFootprintController on the original
1,428-observation causal history. The purpose is to identify remaining runtime
cost after the paired replay's 2.4% incremental reduction. Every planning call
in that completed replay still exceeded 100 ms.

The profiling harness passed 17 focused tests in 2.20 seconds. Source preflight
verified 2,203 paths. Hardware assessment found about 75 GiB available RAM,
567 GiB free artifact storage and no competing full CPU replay/profile. The
original native queue was left unchanged.

The new process is PID **2828895**, creation time **1789121733.58**, command
`.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B
scripts/profile_go2_receipt_copied_footprint_late_history_v1.py`, boot
`1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`, tool session **42270**.

- Preparation: `go2_receipt_copied_footprint_profile_preparation_2026-09-11.json`,
  SHA-256 `18b03497f5ab48cb6ae5ad9a450bf180d04421573a675e9facfcad07fac23d5a`.
- Process-start receipt: `go2_receipt_copied_footprint_profile_process_start_2026-09-11.json`,
  SHA-256 `000fd233019fba4a9b64b65c82f1b47110b684e15a3168f4e7e83b4954710949`.
- Runtime launch SHA-256:
  `a202f29ad40951338b3016d6805131fae17ca3d5ffae5b26ec00d499896685e7`.
- Launch and confirmed-row verification:
  `go2_receipt_copied_footprint_profile_execution_2026-09-11.json`.

The original receipt-copy completion verifier reexecuted successfully without
overwriting its original output. Its complete receipt matched apart from the
new timestamp. Raw/model admission passed, the runtime launch was written,
and frame 0 was confirmed. The job now occupies the one full CPU replay slot.
Do not start another full replay until this exact owner ends.

Next, authenticate terminal result/artifact identities, reconstruct all three
pstats summaries and compare all 1,428 saved row identities with the completed
candidate. Inspect the measured dominant functions before choosing another
optimization. Keep the original sensing failure at frame 1173 and failed round
trip. No new speedup, real-time qualification or navigation success is yet
established by this running profile.
