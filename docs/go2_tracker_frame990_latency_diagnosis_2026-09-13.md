# Recurrent frame-990 tracking latency

The complete sampled-plane and full-consensus tracker traces both have their
slowest call at recorded frame 990: 1,033.21 ms and 983.21 ms respectively.
This motivated a bounded raw-sensor diagnostic on the reused-maze recording.

Isolated plane-candidate extraction at that frame took approximately 16.4 ms
per camera under cProfile, with 6,370 primary and 18,856 auxiliary candidates.
It did not invoke the dense extractor. Plane fallback does not explain this
particular spike; `go2_tracker_frame990_plane_profile_2026-09-13.json` retains
the measured profile.

A fresh tracker consumed the original 991 consecutive sensor observations.
Only frame 990 was profiled. The profiled call took 1.393 s, including profiler
overhead. Six retained-image chains accounted for 0.973 s cumulative time:
109 direct optical-flow associations, 254 LK calls and 15,611 patch-agreement
tests. Patch agreement accounted for 0.525 s, LK for 0.266 s, and all rigid
registration for 0.245 s. These cumulative costs overlap and are not additive.

The original corner/depth/rigid gates were retained; no native pose, learned
model, map, planner or simulator was used. The full profile and selected
reference evidence are in `go2_tracker_frame990_full_profile_2026-09-13.json`.

The next targeted optimization is batching the small photometric patch
calculations, preserving the original OpenCV patch samples and thresholds.
Repeated scalar means/standard deviations currently dominate that operation.
This is a proposed implementation target, not an established speedup. Any
reuse of chain computations must also preserve each original corner identity,
measured endpoint and loss history.
