# Reactive nominal-route baseline prefix V1

The measured-floor connector baseline reproduced four public observations but
vetoed startup solely because the robot-to-route connector lay outside observed
floor. Preserve that completed source and prefix. This separately named variant
changes only the full-floor-coverage veto on the already nominally clear
connector. Unknown connector cells remain recorded and are never inserted into
the floor map or called free space. The original learned controller likewise
permits explicitly unknown initial route connectors, though its predictive
motion checks differ from this baseline's current-geometry checks.

`ReactiveNominalRouteSelector` retains the original reactive rule, route/target
selection, heading threshold, scan, current footprint and current/connector
0.45 m known-obstacle gates. It may request forward with unknown connector cells;
the receipt records that fact and states unobserved space is not certified.
There is still no model, candidate future outcome, command-integrated pose,
predictive surface/path filter or future articulated-motion certificate.
`ReactiveNominalRoundTripController` inherits the full observed mission, sensor
admission, map, history and terminal handling unchanged, replacing only selector
and truthful result identity. Five focused tests cover explicit unknown handling,
three original known-geometry vetoes and exact source-change scope.

Run `scripts/replay_go2_reactive_nominal_route_prefix_v1.py` once at exclusive
`go2_reactive_nominal_route_prefix_v1_attempt_001`. Bind the original completed
recovery native/readout plus measured-floor baseline prefix result
`2ceee2441c3600eff6b2c1176e0c70d93e8d535b419957db8cbcd3326bcc3cdc`, their
artifacts and frozen source/native dependencies before/after replay. Compare
actual public observer/map/mission/command outputs through the first changed
command or terminal policy, then stop before its unexecuted outcome. No model
state is deserialized for inference. This is compatibility evidence, with zero
native baseline executions and no navigation or causal model/memory claim.

As in the predecessor, assess hardware; allow one ordered CPU replay alongside
the separately owned live waypoint native scene only with 8 GiB available RAM
and 256 MiB above the unchanged 40 GiB reserve. Record actual competing work.
No second scene, GPU or training runs. The live native attempt and every prior
failure remain immutable. Future native baseline comparison must explicitly
retain these geometry-gate differences and obtain actual raw-audited outcomes.
