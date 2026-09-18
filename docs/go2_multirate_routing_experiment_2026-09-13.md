# Routing updates independent of the perception clock

`lewm/multirate_routing_map_development.py` separates routing cells from the
old mapper's retained robot-contact histories. It uses the same measured
floor-coverage and sampled obstacle-height calculations, accepts strictly
newer public pose/depth pairs with their actual acquisition timestamps, and
publishes immutable routing snapshots. It neither supplies a robot-volume
contact check nor authorizes motion. An old map is explicitly old; publishing
or reading it does not change its measurement time.

The recorded prefix compares updates on every image with updates every fourth
image (10 Hz versus 2.5 Hz). At 10 Hz, all 61 original map count/reference
receipts match and all 58 available recorded route paths and statuses match.
Complete retained cell identity against the original recording is not proven.
At 2.5 Hz, 16 updates retain a subset of the 10 Hz cells. Only 2 of 58 route
paths match the full-rate map. Up to 140 floor cells and 11 occupied cells
present in the full-rate map are absent from the slower map. At the final
frame, floor cells are 1,455 versus 1,503 and occupied cells 91 versus 94.

Mean update time is 96.53 ms for the full-rate map and 105.02 ms for the
quarter-rate map. This is routing-only work with already-recorded public
poses. It excludes acquisition, pose estimation, robot-contact checks,
planning and actuation. The 300 ms maximum age in the prefix report measures
the sampling gap only; it does not include map computation or delivery delay.
An ideal independent worker using the measured service times leaves three
initial 10 Hz queries without a completed map and reaches 500 ms map age at
the remaining query times. The maximum measured map call is 239.11 ms.
This calculation assumes no acquisition or transfer overhead. There is no
continuous execution or navigation result.

The first prefix attempt failed before frame 0 because JSON converted typed
identity tuples to lists. Its failure is retained in
`docs/go2_multirate_routing_prefix_2026-09-13.failure.json`. The corrected
decoder restores identity tuples only and preserves numeric evidence and
timestamps. The corrected result is
`docs/go2_multirate_routing_prefix_v2_2026-09-13.json` (session 79807, exit zero).

Next integration work must use a current measured robot pose for planning
against the older routing map, keep sensing and command expiry independent
of mapping, and validate candidate motion against fresh sensor evidence.
Routing snapshots alone must not inherit the old controller's contact or
clearance claims. The measured sparse-map route changes require a new
prospective simulation once that complete path is implemented.
