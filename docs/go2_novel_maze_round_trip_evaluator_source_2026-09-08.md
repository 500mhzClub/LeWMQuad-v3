# Native round-trip evaluator source

`lewm/novel_maze_round_trip_evaluation_development.py` supplies evaluator-only
native dwell and physical path checks for the four prospective maze layouts.
It is not connected to native collection yet and is not imported by the
round-trip controller or its packet adapters.

The evaluator admits complete finite native pose, twist, requested-command,
contact and timestamp arrays. It requires the original 2 ms physics clock,
normalized quaternions, the settled initial pose at sample 749, and an explicit
maximum population consistent with the mission component's 4,000-tick source
bound. The timestamp origin was checked against the completed native recording:
samples begin at 0.002 s and sample 749 is 1.5 s.

Each of at most two ordered observed arrival claims binds its phase, instructed
coordinate, frame, clock and ten-interval dwell. At the corresponding actual
physics endpoint, the evaluator requires all 501 samples to lie within the
original native 6 cm region and below the original 5 cm/s speed threshold, and
the intervening 500 requests to be zero. A separate terminal home quiet window,
ten terminal drain commands, absence of contacts/stops, and the observed
round-trip terminal identity are also required for its numerical candidate gate.

Physical path evidence uses native world coordinates and the evaluator-only
maze graph. Every crossed cell boundary must be a declared open maze edge;
positions must remain inside maze cells; and each 2 ms XY displacement must be
at most 0.3 m/s times 2 ms plus 0.1 micrometre numerical tolerance. This rejects
teleports even when their cell sequence follows valid edges. The actual outbound
cell sequence with revisit loops removed must equal the source-defined unique
tree route, and the corresponding return sequence must be its reverse. Full
crossing records and revisits remain available, rather than reporting only an
endpoint. This establishes a candidate for physically retracing the route, not
that every exploration branch was retraced or that memory caused improvement.

Even when all numerical checks pass, `verified_round_trip` remains false and
`requires_raw_sensor_command_and_visibility_audit` remains true. The caller must
independently verify the actual collection, command dispatch and sensor/render
contracts. Centre-cell traversal is not a robot clearance certificate and cannot
replace contact or geometry audits.

Eight synthetic tests passed in 1.84 s: a smooth complete two-dwell round trip;
separate failures for outbound nonzero command, excessive quiet speed, physical
contact, budget terminal, and missing drain; teleport/closed-wall crossings;
and bad clock, arrival order or incomplete window. No native maze has been
executed by this source. It remains unfrozen pending collector/auditor integration.
