# Camera-gap association diagnosis

The actual-interval chained-flow successor passes six focused tests (1.95 s):
moving-image endpoint equivalence, retained real timestamps, track loss through
occlusion or invalid depth, and bounded interval/elapsed lifetime checks.
It preserves the original image and rigid-fit limits and uses only acquired
images, with at most 32 links and 3.2 seconds of elapsed history.

The full 100/200/300 ms recorded schedule still fails at frame 690, after 345
accepted camera observations and 691 gyro packets (54.47 s wall time).
All 28 chained pair attempts fail. Primary chains retain at most eight endpoint
depth pairs; auxiliary chains retain up to 37 but fail rigid consensus or its
minimum support requirement. Accepted poses match the direct-flow predecessor.
Maximum accepted position error is 4.93 mm; no full-journey success is claimed.

A four-packet, read-only reconstruction of frames 687–690 narrows the direct
auxiliary rejection: 19 of 34 matches are consistent (55.88%, below the 60%
requirement), despite support across eight reference and nine current grid
cells. The rejected fit has 0.553 mm residual RMS and 11.20 mm translation in
the auxiliary reference frame. These are diagnostics of a rejected fit, not
an accepted pose or proof that all of its inliers are correct. Primary direct
flow has nine matches, below the minimum of twelve.

A separate association-only check initializes LK from the integrated gyro
and each image's own measured depth, including the fixed auxiliary camera
lever arm under a zero-body-translation hypothesis. Pixel, photometric, depth
and rigid-fit limits are unchanged. This increases primary matches from nine
to eleven but does not recover either camera's fit. The gyro-seeded helper is
not integrated into the tracker. No native pose is loaded by either pair
diagnostic.

The next recorded experiment uses the unchanged actual-interval chained
tracker at a fixed 200 ms camera period (5 Hz), while consuming all 100 ms gyro
packets. It runs through source frame 3439, with the final scheduled camera
at 3438. This operating rate was selected after the recorded 300 ms failure.
It tests tracking under subsampling; it does not simulate processing delay,
qualify real-time execution, or execute new navigation.

Evidence:

- `go2_gapped_chained_flow_journey_v1_attempt_001` under the navigation artifact root.
- `docs/go2_gapped_camera_frame690_fit_diagnosis_2026-09-13.json`.
- `docs/go2_gyro_seeded_frame690_pair_2026-09-13.json`.
- `go2_gapped_camera_5hz_journey_v1_attempt_001` under the navigation artifact root.

## Fixed 5-Hz result

The 5-Hz experiment failed at source frame 2072 after 1,036 accepted camera
observations (last frame 2070) and 2,073 gyro packets. It passed the earlier
frame-690 problem and tracked past the source run's outbound-arrival frame
2061, but did not track the complete return journey. This is not a new arrival
or navigation result. Maximum accepted position error was 6.66 mm and maximum
rotation error 0.00343 rad; wall time was 170.41 s.

At the new failure, primary direct flow supplies only seven depth pairs.
Auxiliary direct flow supplies eighteen but fails the consensus/grid/displacement
check. All retained-reference chains also fail. Merely reducing camera frequency
therefore does not support full-journey operation with the current association.

Accepted-prefix tracker calls average 97.28 ms (median 95.45 ms, p95 186.58 ms,
maximum 296.44 ms); 48 exceed the 200 ms camera period. Reusing these durations
in a one-worker FIFO calculation gives median output age 96.52 ms and maximum
296.44 ms, without sustained backlog. This calculation excludes the failed call,
acquisition, mapping and planning. It is not a complete timing qualification.
The paired `go2_gapped_camera_5hz_timing_2026-09-13.{json,png,svg}` files preserve
the calculation and plot, explicitly labeled as the accepted prefix.

The next useful direction is reducing tracker computation at the original
10-Hz camera rate, whose complete recorded tracking has already worked. In
particular, rigid registration currently evaluates all 128 random proposals
even when a proposal already explains every matched point. An early exit in
that exact case could reduce work without reducing fit support; it still needs
implementation and measured comparison. No such optimization is adopted here.
