# Recorded fused navigation interface V2: serialization correction only

The exact protocol is V1's
`docs/go2_rgbd_fused_navigation_interface_development_v1_2026-09-06.md`, except
the new output is
`.generated/go2_rgbd_fused_navigation_interface_development_v2_attempt_001`.
V1 terminated with AssertionError on its first anchor comparison before saving
any arm result. Independent read-only diagnosis verified that the only differing
top-level fusion field was `initial_velocity_prior`: Python dataclass tuples
versus JSON lists. JSON-normalized values are exactly equal.

V2 normalizes in-memory state through JSON serialization before exact equality.
No numeric tolerance, ignored field, prior change, estimator change, controller
change, altered input, action execution, physical replay or uncertainty relaxation.
The original V1 launch, failure and all456source bindings remain immutable and
are verified. V2 binds those two failure-witness files, adds its own script,
this protocol and five focused serialization tests to the inherited closure,
and performs the same before/after input verification. Keep any scientific or
subsequent infrastructure failure. This is not a navigation mission or retry
of a physical experiment; it is a separately preserved interface diagnostic.
