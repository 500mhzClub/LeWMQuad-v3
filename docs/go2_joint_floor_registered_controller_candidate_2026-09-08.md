# Common measured floor plane for consistent pose registration

This candidate replaces the failed requirement that each camera independently
identify a full plane. It fits one plane to all currently measured candidate
points in the shared body coordinate system. Plane observability is checked on
the combined geometry. The predecessor failure remains separately preserved in
`go2_floor_registered_prefix_failure_2026-09-08.md`.

`lewm/joint_measured_floor_plane_development.py` combines camera counts, means
and covariance matrices with equal weight per measured point. Empty candidate
sets contribute zero points and no geometric moments. Both public camera
packets remain required; unavailable image rays are not fabricated. The combined
fit requires at least 100 points, second covariance eigenvalue at least
0.0025 m², and up alignment at least 0.97. Every contributing point in every
camera must have residual at most 3 mm. A dense view cannot dilute a sparse
contradicting surface past that maximum-residual gate. No trimming, absolute
initial-height seed band, command integration or native geometry is introduced.

The sufficient statistics and per-camera residual counts are retained in a new
pose-evidence schema in `lewm/joint_floor_registered_evidence_development.py`.
Its accessor reconstructs the combined moments, checks every residual gate,
and independently validates the original visual witness and the unchanged
height/tilt correction composition. Fresh raw replay must also reconstruct
statistics and residuals from the bound camera packets; summary composition
alone does not prove pixel identity.

The initial reference remains frozen. Raw visual in-plane position and heading
are preserved; normal correction gates remain 5 cm and 0.10 radians. Failure of
combined observability, coherence, sensor timing or original visual admission
latches a stop. Static shared floor identity remains an uncalibrated hypothesis.

`lewm/joint_floor_registered_controller_development.py` uses the new pose for
both camera maps/partitions, contact geometry, residuals and mission tracking.
It retains the complete original visual evidence and unchanged visual tracker,
model, action set, mission, nominal constraints and contact policy. The three
consumer methods are again exact narrow derivatives except for the explicitly
typed pose accessor. No historical memory is rewritten.

Focused tests cover a narrow camera with a well-spread companion, combined rank
failure, absent candidates, sparse conflicting surfaces, exact covariance
composition, preserved all-point residual limits, witness tampering, actual
synthetic public packets through all consumers, and latched sensor failures.
These are development tests, not successful navigation or calibrated sensing.

Before launch, 22 combined geometry/controller tests passed in 4.05 seconds and
five prefix-comparison tests passed in 1.88 seconds. Read-only inspection 29747
also completed on public frames 0, 329, 330 and 331 of the completed confirmed-
floor recording. All four combined fits were admitted. Frame 329 combines
14482 candidates; second covariance eigenvalue is 0.024598045301285162 m²,
maximum residuals primary/auxiliary are 0.00000534242433702925 /
0.000005416510983791145 m. Frames 330 and 331 have combined second eigenvalues
0.02579347785568593 and 0.02365910450150216 m², with all maxima below 6 micrometres.
These sparse geometry checks do not replay a candidate trajectory or demonstrate
that a different action will reach the goal.

Next is one fresh candidate replay against the immutable completed confirmed-
floor predecessor. Its complete raw sensor/model/command audit is verified and
reused, instead of executing that already completed predecessor audit again.
The candidate must preserve the original visual witness and raw forecasts when
both controllers plan, and stop at the first different command or terminal.
The failed independent-plane prefix is separately hash-bound as identity and
failure evidence; no state or corrected trajectory is resumed from it.
