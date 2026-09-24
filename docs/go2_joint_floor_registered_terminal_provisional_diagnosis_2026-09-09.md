# Eighth native collection: provisional terminal diagnosis

The native collection ended; its final raw audit and actual-prefix comparison
are still pending. Do not describe the whole native attempt as completed yet.
Root `go2_joint_floor_registered_maze_pilot_v1_attempt_001`, launch SHA-256
`7478a094256d99aa3c25958806707730efb07eb3277dd83e411437b7d9ee98a5`.
Collection `full_jepa_novel_maze_00/result.json` SHA-256:
`a07f53205344d5e6fc58f754d3180268eacae8e2e8ea684e5850837124caf261`.
Closed decision stream SHA-256:
`e92d311fed32b3886f26eefc4f14c999cd5b37a7bfde6c6b54532cd0524fa64a`.

1067 commands,1068 paired observations,54100 physics samples. Controller terminal
at1057: NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS, then10
zero drain commands. No physical/acquisition stop, observed arrival or return.
Last mission phase OUTBOUND, observed goal distance2.887777272235155m. This is
observed distance; native trajectory/pose-error readout is still outstanding.

At1057 common-plane registration remains admitted:26503 candidates,7303primary
and19200auxiliary; second covariance eigenvalue0.1484669621m². Maximum point
residuals20.0251/12.4121micrometres. Correction -3.254638mm along the initial
floor normal and0.002838711radians tilt. The current original primary-floor fit
also remains available with7303seeds. All19200 current auxiliary samples are
classified as original floor, with zero additional confirmations. The terminal
problem is therefore different from losing all current primary floor seeds.

Hold/forward intersect a retained auxiliary FR-foot ambiguous enclosure in
cell[86,33,-11], first frame362, last sample364. Left arc/turn intersect retained
primary FL-foot cell[99,40,-10], first123,last143. Other turn hits include retained
FR-foot cells[87,33,-11], [86,34,-11] and auxiliary[85,34,-11], with last samples
142,363,361,357. These are original possible intersections, not confirmed physical
collisions. Five actions fail surface checks. Right arc passes surface checks
and its first seven nominal segments, but its final700–800ms segment fails the
unchanged0.45m nominal radius. Thus no action passes both full filters.

A read-only current-view probe3405 completed. Transforming all eight corners of
the six distinct first-hit enclosures into the saved gravity-aligned map gives
height intervals within approximately -1.60 to+0.24mm of the fixed floor hypothesis.
However, neither current camera covers every5cm floor square touched by any
enclosure. Height proximity alone cannot resolve these ambiguous classifications.
The six queries cover cells[43,16], [49,20], and combinations of[42,16], [42,17],
[43,16], [43,17]. All current primary/auxiliary coverage tests returned false.
This floating-point probe is not calibrated uncertainty or ground support.

Prepared separate diagnosis
`scripts/diagnose_go2_retained_floor_resolution_v1.py` checks whether a strictly
later executed observation, no later than1057, completely covered each enclosure's
grid squares under the unchanged pose/floor hypotheses. It changes no historical
classification or command. Results must be reviewed before proposing any
evidence-based memory revision. The native audit continues in session90647.
