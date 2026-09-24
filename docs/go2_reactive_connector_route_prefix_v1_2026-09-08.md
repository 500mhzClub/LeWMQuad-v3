# Nearer observed-route connector baseline prefix V1

The completed reactive nominal baseline stopped with a clear current nominal
radius and current articulated surface, but a blocked connector to its selected
lookahead waypoint. The final measured connector clearance was 0.448060239 m,
below the original 0.45 m radius. Its heading error was inside the turn threshold,
so it neither turned nor advanced. The observed route itself remained available.

This separately named non-predictive controller retains the original rule unless
an intermediate WAYPOINT selection has no action because its measured connector
fails the nominal radius, while current nominal clearance and surface checks
pass. It then considers only earlier cell centres on that same observed route,
in reverse route order, and chooses the latest with a clear 0.45 m connector and
distance greater than the existing 0.04 m local-target radius. It does not look
beyond the original target. If none exists, the original infeasibility remains.
It turns toward the chosen measured target using the same 0.1 rad threshold or
requests the same forward primitive when aligned. No command outcome is forecast.

Current observed pose, original current clearance and original target connector
must reconstruct exactly. Every replacement centre requires observed floor. The
route and current footprint receipt remain unchanged. Unknown connector cells
are still explicitly reported under the predecessor policy and are never written
into the map as free space. The selected target and checked alternatives are
recorded along with the entire original selection. A valid original action,
current geometry failure, exhausted view budget or selected exact mission goal
does not activate fallback. Exact goal coordinates and arrival tolerances remain
unchanged. Observer, persistent map/history, mission, shared budget, command
dispatch and latched stops are inherited.

The prefix binds the completed reactive native result
`1a3bf1e0f796d8fe5ae7b0b11feb837b7299064f83f6b9f44457d0040a229dd6`
and readout result
`4aeea872d44ec90f93d5ff29ce6e144af4b08add3394db0c94bd006c3593f2af`,
their artifacts and frozen sources. The fresh controller consumes only their
public primary/auxiliary RGB-D and body packets. Causal observation/map/mission
receipts, original routes and current geometry checks must match through the
first changed command; each new selection must equal the pure fallback transform
of its recorded predecessor selection and current observed map. Every other
pre-intervention decision field must match, apart from the controller identity
and explicit policy-enabled flag.

Stop at the first changed command or terminal decision, before consuming its
unexecuted outcome. This is observation-scope verification, not navigation or
success evidence. No high-level model, native pose, command-integrated pose or
future-action geometry is used. The original baseline failure is retained.
This is further development on reused maze 0, not a new independent layout or
evidence isolating persistent-memory benefit.

Launch after focused real-geometry corner, unchanged-gate and controller tests
pass. One CPU replay requires 8 GiB available RAM and 256 MiB artifact headroom
above the 40 GiB reserve. It may run beside the separately owned native recovery
scene when measured resources admit both; it creates no new scene or checkpoint.
Sources and completed inputs are checked before and after replay. Exclusive
output is `go2_reactive_connector_route_prefix_v1_attempt_001`. Preserve partial
failures; no retry or old-future-outcome continuation is implied. A new full
native run and independent raw audit remain necessary for any changed command.
