# Nearer observed-route connector prefix result

Completed root: `go2_reactive_connector_route_prefix_v1_attempt_001`.
Result SHA-256: `99386e0bdc125cf250b70c0925d5d38b5bb01a4cdc7949b646fecc139b35d236`.
Launch SHA-256: `bb5c9009a44eb94ca80dfea7bd66f70becb3739d42fe1e32a001cd3539b437c1`.
Decision stream SHA-256: `d8ecfeaea370c3d3ad2206b257cb0bc597015a834a6160746a5ef257694722f1`.
The launch freezes 1,471 sources. Replay wall time was 55.329449318 seconds.

All 91 public observation frames through tick 90 reproduced the original
reactive baseline's observation, map, mission, original route and current
geometry receipts. All commands before tick 90 and all other pre-intervention
decision fields matched exactly, apart from the declared controller identity.
The complete changed selection equals the independently reconstructed pure
nearer-route transform. No learned model or future command outcome was used.

At tick 90 the original zero request becomes forward `[0.2, 0, 0]`.
The original target `[1.175, -0.42500000000000004]` at route index 12 becomes
the preceding observed target `[1.175, -0.375]` at index 11. Current nominal
clearance is 0.4679907278469982 m and the current articulated surface is clear.
The original connector clearance is 0.4465830575483517 m; the replacement is
0.4590458885075443 m, with the original 0.45 m radius unchanged. Heading error
is 0.020081282225894538 rad and no replacement connector cell is unknown.

Replay stops before consuming the outcome of that changed command. This proves
the causal scope of the intervention, not its physical effectiveness. No arrival,
round trip, independent-layout result or navigation success has been added.
The original reactive failure remains intact. Eight focused policy tests and
two native collector/audit scope checks passed before native admission.

The next experiment is a fresh full-length execution using
`scripts/run_go2_reactive_connector_maze_pilot_v1.py`, after the existing native
view-recovery job completes and resource admission passes. It must reproduce
the physical and public prefix through the observation at tick 90, execute the
new command, and independently replay/audit the full resulting trajectory.
