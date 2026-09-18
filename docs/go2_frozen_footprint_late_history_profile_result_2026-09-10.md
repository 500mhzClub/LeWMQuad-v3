# Late-history controller profile completed and verified

The frozen-footprint controller reconstructed all 1,428 original decisions
through frame 1427, including 1,425 forecasts. Normalized complete decisions
matched the original recording and public inputs remained unchanged. The
assigned model state remained unchanged with no gradients. The original full
input admission completed before and after replay; the profiler owner exited.

The following are totals for each fixed ten-observation window. Inclusive
times overlap and must not be added to one another. Profiling overhead is
included, acquisition is excluded, and the shared-host run is not an isolated
real-time benchmark.

| Window | Total profiled time | Footprint calls | Footprint inclusive time | Retained-patch coverage exclusive time |
| --- | ---: | ---: | ---: | ---: |
| Frames 3–12 | 9.619 s | 60 | 2.459 s | 0.024 s |
| Frames 395–404 | 20.987 s | 180 | 14.086 s | 2.910 s |
| Frames 1418–1427 | 36.852 s | 168 | 29.307 s | 12.028 s |

The late window spends approximately 79.5% of profiled time within footprint
queries. Retained-patch coverage scans are the largest individual exclusive
cost. Footprint call count falls slightly from the hold to late window while
its inclusive time more than doubles; this supports investigating retained
history cost as well as duplicate queries. These observations do not measure
the speedup of the prepared scoped-reuse candidate.

The independent verification authenticated 2,130 source bindings and all eight
output artifacts, recomputed all 1,428 original decision hashes, checked the
fixed windows, and regenerated all three JSON summaries from their raw `.prof`
files. It did not independently rerun neural inference or the complete original
training/native input admission.

- Result: `07c096a8e6d83ac7676337993643e30dc35ebdec65758674f9ffb71c58b23c84`.
- Verification: `9528b0ea89bc3f62b6728ac34ffdb73da0907694e9ee54249d8ab7caa6bdccdb`.
- [Verification and extracted costs](go2_frozen_footprint_late_history_profile_verification_2026-09-10.json).

The original visibility failure at frame 1173 and zero-round-trip outcome are
preserved. Reproducing this history establishes neither qualified sensing nor
successful navigation.

The completed result satisfies the prepared scoped-reuse replay's profile
dependency. That paired timing replay is still unexecuted. First finish the
[full-controller tracking prefix](go2_no_rgb_jepa_direct_flow_controller_prefix_execution_2026-09-10.json)
now running as PID 2749113, creation time 1789073975.55, on boot
`1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Its exact command is recorded there.
Confirm that owner has ended and review its result before starting another
full-controller replay; quiet input hashing does not mean it stopped. The
existing sixth-case/frontier/hold/contact native ordering remains unchanged.
