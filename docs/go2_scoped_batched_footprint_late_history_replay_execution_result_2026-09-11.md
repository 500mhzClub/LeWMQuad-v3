# Combined scoped and batched replay started

Started the prepared `replay_go2_scoped_batched_footprint_late_history_v1.py`
against scoped result
`b7d80d3d98f9c4d0be92ab3a91d5d92f86d1ec3a5b78add2e078cb71a630670a`.
Original scoped completion and its ended owner were authenticated first.
Resource admission measured 74,381,205,504 available RAM bytes,
627,405,533,184 free artifact bytes and 16 physical CPUs. No other full
controller replay was observed before dispatch.

Original new owner: **PID 2766980**, creation **1789083663.49**,
tool session **94047**, boot `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.
The command uses the fixed single-thread environment and explicit completed
scoped result SHA. No retry or resume is permitted.

Execution record:
`docs/go2_scoped_batched_footprint_late_history_replay_execution_2026-09-11.json`,
SHA-256 `3332fa35ced3b113d7fa98910e7cdc2a38a9ec55fc3139c06d58103c0888af33`.
All 2,153 prepared source bindings were verified, session 46984, exit 0.

At registration the original process was live in full input admission, with
`SCOPED_BATCHED_FOOTPRINT_FULL_INPUT_ADMISSION_STARTED` observed. The attempt
root had not yet been created and controller comparisons had not started.
That admission normally takes substantial time; quiet output is not failure.
Continue observing this exact owner rather than starting a replacement.

The planned comparison covers 1,428 observations and 1,425 forecasts, with
ScopedFootprintAnchoredController as baseline and ScopedBatchedFootprintController
as candidate. It must retain complete normalized decisions and seven observed
state checks, with only the two documented patch-store type names normalized.
Original sensing failure and failed navigation are retained. This CPU replay
starts no native scene; the existing case-5 audit and native queue continue.
