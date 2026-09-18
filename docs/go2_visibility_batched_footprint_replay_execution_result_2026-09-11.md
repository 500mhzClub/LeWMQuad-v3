# Empty-visibility optimization: paired replay running

Implemented a separate retained-patch query that skips visible-index enumeration
for empty rows of the already-computed batch projection. It retains original
prefix-key access, chronological witnesses, early stopping and arithmetic-error
fallback. The original controller and all frozen native runs are unchanged.

The coverage implementation passed 48 focused tests. Controller integration
passed 10 tests, including actual public sensor packets, map state, both robot
footprint histories and scoped-cache ownership. The paired replay harness
passed 16 tests. Preparation failures and corrections are retained in the
preparation JSON: one inherited test instrumented the old class, and the new
replay initially referenced a nonexistent profile import. Both were corrected
before runtime launch; no original scientific attempt was restarted.

The 30-repeat alternating-order synthetic microbenchmark preserved every
complete result. Its total-time reductions were 29.09% for invisible histories
and 26.46% for sparse visibility. Fully visible/uncovered and immediate-witness
patterns were 0.49% and 1.41% slower. These are component measurements on
synthetic histories, not a full-controller speedup.

- Microbenchmark result: `go2_visibility_batched_retained_patch_microbenchmark_2026-09-11.json`,
  SHA-256 `4b3bfb228073d51b10794e27ecdd4f34988094fe5d9cd9f698335f9c9af7cc74`.
- Replay preparation: `go2_visibility_batched_footprint_replay_preparation_2026-09-11.json`,
  SHA-256 `b1e6879a539f1b3d50881a2484895c50ae64c197f171119e9ebd09b1d7fb86d7`.
- Process-start receipt: `go2_visibility_batched_footprint_replay_process_start_2026-09-11.json`,
  SHA-256 `37b9c4544d765f6885b40f6b82d5809c380ff299d20d685b08cb8201d5f4ddc5`.
- Launch SHA-256:
  `6945b551f09de3153378df50fb4e5b018ae695595c5788ccd194123571b7f02a`.
- Bound launch and confirmed-row receipt:
  `go2_visibility_batched_footprint_replay_execution_2026-09-11.json`.

Source/resource preflight passed with 2,217 source bindings, about 73 GiB
available RAM and 564 GiB free artifact storage. No competing full CPU replay
was live. The new owner is PID **2834244**, creation time **1789123877.95**,
command `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B
scripts/replay_go2_visibility_batched_footprint_late_history_v1.py`, boot
`1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`, tool session **48376**.

Input admission passed and the paired replay began at frame 0. It occupies the
one full CPU replay slot. It must complete all 1,428 observations, 1,425 forecast
comparisons and seven original retained-state checks, then reauthenticate its
inputs. Normalize only declared controller metadata and exact patch type tags;
all witness, pixel, map, residual and model values remain part of comparison.

Next, monitor this exact owner and the original tracking collection, preserve
any failure, and verify the complete paired timing report after termination.
The original sensing failure at frame 1173 and failed round trip remain. No
full-controller speedup, native adoption, independent navigation, real-time
qualification or hardware qualification has yet been established.
