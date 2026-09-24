# Raw completion and physical-prefix helper prepared

The new helper authenticates the completed prospective raw replay and compares a
later native successor with the original hold run through observation 406. It
requires all 407 original and candidate observations, 404 model forecasts, exact
complete normalized decisions, and current public-packet fingerprints. Its packet
ordering matches the sustained replay rather than a predecessor's different order.

For a native comparison, all 21,050 preintervention samples must match for every
recorded physics field. Both command tapes must contain the fully completed
changed command, and both traces must contain its 50 additional samples. The
post-intervention values may differ. The helper thus requires at least 21,100
recorded samples while comparing only the shared 21,050-sample trajectory.
This does not substitute for the complete successor raw audit.

All 34 synthetic completion, causal-boundary and physical-prefix tests passed in
3.34 seconds, session 18520, exit zero. They reject incomplete streams, changed
forecasts/observed state, wrong model or launch identities, changed scope flags,
wrong public packets, incomplete commands, altered earlier physics and missing
intervention samples. A source preflight verified 1,979 paths in session 33248,
exit zero. No actual completed replay or native successor was admitted by these
synthetic tests.

The source is `scripts/sustained_hold_reorientation_native_prefix_development.py`.
The preparation record is
`docs/go2_sustained_hold_reorientation_native_prefix_preparation_2026-09-11.json`,
SHA-256 `a50a56836c91d882408a3ddb8f64b17bae718ba7052438e46dea9eac02b91d84`.
Preparation completed in session 81096, exit zero.

The original raw replay, PID 2813368 / creation 1789115323.91 / session 87754,
remains live and last reported observation 100 reconstructed. Its exact launch is
`6f68017be0f68198ab08af97fb4d90f33ebbdc2b1b56f8da123e8b34b4dd89df`.
Do not restart it. The contact native parent remains 2808232, with tracking and
extended budget queued behind it.

Next prepare a native launcher that calls this real completion helper and the
original queue/input admission before allocating a scene. It must wait for the
completed replay and existing queue, retain the fixed model and original raw
audit, and execute the prospective first-command comparison on collected physics.
No sustained native attempt, policy selection, navigation success, real-time
qualification or hardware qualification has been established.
