# Short camera-gap tracking failure and direct-flow successor

The recorded independent-layout-00 JEPA journey failed at source frame 684
when visual observations used a repeating 100/200/300 ms interval. All 685
gyro packets were consumed; 342 visual observations were accepted, last at
frame 681. The maximum accepted position error was 4.93 mm. This is a partial
tracking result, not continuous navigation.

The unchanged diagnostic reproduced the failure and retained its inner cause:
neither a retained anchor nor the previous frame supported a current pose.
Primary image matching reported insufficient rigid-pose matches for all eight
retained references. The auxiliary fit rejected consensus, grid support or
displacement. No plane conflict was recorded. Direct-flow fallback made no
pair attempts: its original guard required a 100 ms interval, while this pair
was 300 ms apart. Chained flow also retained its consecutive-100-ms requirement.

Evaluator-only native pose shows 0.13847 rad rotation and 5.73 mm translation
over frames 681–684. Unlike the earlier 500 ms failure at frame 285, this
rotation is below the unchanged 0.20 rad increment limit. Native pose was not
used by either tracker.

The separate `GappedDirectFlowTracker` successor permits direct optical flow
between the immediately preceding and current processed images using their
actual bounded interval. It retains the original photometric, reverse-flow,
depth, rigid-fit, disagreement and displacement limits. It does not construct
missing images, reset history, alter the chained fallback or integrate the
result into the fixed-cadence controller.

Three focused tests passed (3.42 s), exercising real optical-flow and depth
fits with missing descriptor-pair support at 100, 300 and 500 ms. These use
static images; the recorded moving journey is the next test. The successor
uses the same 100/200/300 ms schedule through frame 3439 and is explicitly
selected after the predecessor failure.

Artifacts under the navigation development artifact root:

- `go2_gapped_camera_short_gap_journey_v1_attempt_001`: original failure.
- `go2_gapped_camera_short_gap_failure_diagnostic_v1_attempt_001`: unchanged
  reproduction, exception chain and tracker state.
- `go2_gapped_direct_flow_journey_v1_attempt_001`: separate successor test.

No new native navigation, delayed control, real-time qualification or hardware
validation is established by these recorded-sensor tests.

## Successor result

The actual-interval direct-flow successor recovered frame 684 and accepted
frames 685 and 687, using the auxiliary camera. It then failed at frame 690
after 691 gyro packets and 345 visual observations (53.21 s wall time).
Maximum accepted position error remained 4.93 mm and rotation error 0.001342 rad.
Median tracking time was 84.83 ms; maximum was 273.25 ms on the shared host.

At frame 690, primary direct flow retained only nine depth pairs from 49
corners, below the rigid-fit support requirement. Auxiliary direct flow retained
34 pairs from 58 corners but failed the rigid consensus/grid/displacement test.
The chained fallback made no pair attempts because its image-clock restriction
still requires consecutive 100 ms acquisitions. Enabling direct flow therefore
fixes the earlier missing fallback but does not establish journey-wide tracking.
The next unresolved issue is robust measured image association across gaps;
the displacement/fit limits have not been weakened.
