# Existing tracking fallback recovers the recorded failed observation

The full observer-history replay completed 860 observations. It reproduced
every original raw visual receipt, including the failure at frame 859. The
candidate matched the original through all 859 preceding observations. At
frame 859, the existing direct-flow fallback admitted a current auxiliary-camera
pose under the unchanged rigid fit, actual gyro, displacement, temporal and
bridge checks. The replay stopped there and consumed no frame 860.

The front camera's thirteen associations still failed rigid consensus. The
auxiliary camera had 45 associations, of which 31 were inliers (68.89%), covering
eight reference and seven current image grid cells. Fit RMS was 0.41245 mm and
gyro disagreement was 0.00070025 radians. These are fit diagnostics, not a
calibrated physical pose-error bound. This was a measured incremental bridge:
bridge count seven of the unchanged ten-frame budget, with no anchor promotion
or history reset. Continued tracking and anchor reacquisition remain unproven.

An independent verification authenticated all 1,918 source bindings, the three
output artifacts and 5,260 original input bindings. It compared all 860 saved
baseline receipts against the original episode and all 859 unchanged candidate
receipts. It reran the boundary rigid fit using the saved associations and the
gyro witness from the full observer replay; the fitted body-frame rotation and
composed position reproduced exactly. It did not independently reintegrate the
gyro sensor history or rerun all observer frames.

Evidence:

- Observer result: `55b60a844d230f7122ac0838402f78b22ef634aba64555ca4dbae044694d9131`.
- Independent verification: `3a26c805fa531243acc28b19c28fe4779789ec51fca86f03e577e78538ac41a9`.
- [Recorded verification](go2_no_rgb_jepa_direct_flow_observer_prefix_verification_2026-09-10.json).
- [Replay protocol](go2_no_rgb_jepa_direct_flow_observer_prefix_v1_2026-09-10.md).

No floor registration, mapping, learned-model inference, command selection or
native execution was performed by this observer diagnostic. The original
episode remains a failed development episode with five open edges and no
verified round trip.

The next [full-controller prefix](go2_no_rgb_jepa_direct_flow_controller_prefix_v1_2026-09-10.md)
keeps the original anchored planner and floor-registration implementation while
substituting this observer. Thirteen focused boundary and integration tests
passed, and the 1,947-path source preflight passed. Its owner started after the
late-history profiler completed and exited; the timestamped
[execution record](go2_no_rgb_jepa_direct_flow_controller_prefix_execution_2026-09-10.json)
identifies the live input-admission process. The separate paired performance
replay has not been launched. The full-controller result and any fresh native
navigation outcome remain pending.
