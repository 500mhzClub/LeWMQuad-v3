# Preliminary physical round-trip pass

The fresh stop-conditioned controller run completed collection and passes the
unchanged physical round-trip evaluator on reused development maze 02. The full
raw sensor/controller/visibility audit is still running; this is not its final
verified result.

| Arrival | Observation | Maximum distance during the required second | Maximum speed during that second | Physical dwell |
| --- | ---: | ---: | ---: | --- |
| Outbound | 3062 | 0.04377756 m | 0.04416197 m/s | Pass |
| Return | 4739 | 0.02199548 m | 0.03225391 m/s | Pass |

The unchanged limits are 0.06 m and 0.05 m/s throughout the one-second window,
with zero command requests. The separate terminal quiet window also passes.
The robot physically retraced its outbound route: 15 outbound and 11 return
crossings, zero invalid crossings, and reversed loop-erased cell routes.
The 238,200-sample physical trace contains zero recorded contact flags.
There was no physical or acquisition stop.

Collection took 4,207.51 seconds (70.13 minutes) and recorded 4,750 decisions.
The prior run declared return arrival at frame 4738 and failed its speed window
at 0.06086774 m/s. The prospective rule requires a measured quiet interval under
a zero request before starting the dwell. In this fresh run it declared return
arrival at frame 4739 and passed without changing physical criteria. Outbound
arrival timing remained at frame 3062. The prior negative outcome is preserved.

The accompanying JSON records the closed collection, physical-trace and launch
identities and the evaluator output. Evaluation used the existing
`lewm.novel_maze_round_trip_evaluation_development.evaluate`, with the already
declared 8,000-step budget. No recorded arrival was relabeled or shifted during
evaluation. The rule was applied by the live controller in a fresh simulation.

This is one reused-layout development result with the fixed no-RGB direct
predictor and RGB-D controller perception. It does not establish independent
maze reliability, JEPA/planning/memory advantage, real-time execution or hardware
validation. The full audit must finish, and the queued independent layout-0
full-RGB JEPA experiment is the next native case.
