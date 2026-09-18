# Reactive measured-floor connector prefix: completed, startup veto identified

Root `go2_reactive_observed_route_prefix_v1_attempt_001` completed with result
SHA-256 `2ceee2441c3600eff6b2c1176e0c70d93e8d535b419957db8cbcd3326bcc3cdc`.
Launch SHA-256 `27bb563289dbab18a780bcf9c1dd7ff58fb4e0b7f61d2eb5822b4a635ef0d126`
binds 1,429 sources; decision stream SHA-256
`8fc5f332b88f7a61ab26865e61251d0f5dc84d3158582601b51ff424588c8fe1`.
Session 38519 completed. Four actual public observations reproduced the same
observed pose, persistent map, auxiliary partition and mission as the original.
At tick 3, the baseline requested zero versus original forward. It stopped before
reading the outcome of that alternative request. No native baseline was run.

The baseline's current nominal clearance was 1.900060275 m; the target connector
minimum clearance was 1.425 m and current footprint had no possible intersection.
Heading error was 0.050495605 rad, within the 0.1 rad forward threshold. The sole
startup veto was requiring the entire current-to-waypoint connector to be observed
floor: cells [-1,0] through [8,0] were unknown. The original route proposal permits
an explicitly unknown start connector, subsequently checked by learned predicted
surface/path gates. Treating the baseline's stronger connector observation rule
as a matched non-predictive condition would confound the comparison at startup.

This prefix is preserved as compatibility/diagnostic evidence, not a failed native
navigation episode or proof of baseline inability. A separately named nominal-route
variant may permit the same explicit unknown connector while retaining current
footprint and known-obstacle connector checks. Its future-motion gates still differ
from learned prediction and must be reported as part of the policy comparison.

Eleven source tests passed in 2.07 s. Replay and final authentication took
13.963292 s after launch. Post-replay hardware recorded 77.304 GB RAM available,
52.773 GB artifact space free, CPU 3.3% busy and the separately owned waypoint
native worker still running. No baseline model inference, training, native scene,
hardware execution, navigation success or planning/memory advantage occurred.
