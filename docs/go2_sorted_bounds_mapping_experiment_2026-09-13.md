# Sorted measured-voxel insertion

Mapping averaged 203 ms in the combined controller's early paired replay.
This experiment replaces repeated indexed min/max updates with stable sorting
and contiguous reductions. Cell order, outward rounding, first witnesses,
sample counts, latest frames, retained bounds and all query methods remain.
Three focused tests pass (0.34 s), including accumulated bounds, signed zeros,
negative voxel boundaries, sample enclosure, independent storage and queries.

The paired complete-controller replay changes all eight retained sample-bound
indexes in an otherwise identical combined controller. Both arms reproduce
all 13 recorded decisions apart from the existing proposal-work count; ten
calls are timed after three warmup observations. The model is unchanged.

Total time falls from 4.241 to 4.157 s (1.97%). Median decision time is
428.53 versus 415.86 ms; mean mapping time is 203.56 versus 197.17 ms. These
are a short shared-host comparison, not a robust whole-journey speed claim.
The gain is small relative to the approximately fourfold gap to the 100 ms
decision period. This candidate is not adopted and does not justify a new
native navigation run or a large replay on its own.

Result: `docs/go2_sorted_bounds_early_controller_2026-09-13.json` (session
22062, exit zero). The console print retains the reused driver's older status
label; the saved report has the correct `SORTED_BOUNDS_EARLY_CONTROLLER_COMPLETE`
status and explicit baseline, candidate and additional source identities.

Further timing work should address the larger data flow through mapping and
planning, with actual observation and command timing preserved, rather than
treating this small insertion optimization as the real-time solution.
