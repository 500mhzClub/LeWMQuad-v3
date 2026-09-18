# Observation-cadence replanning goal probe V1

The augmented fixed-model probe failed after a second five-command right arc:
20-mm direct and 59-mm JEPA endpoint errors exceeded roughly 11–12 mm of
predicted nominal clearance. The conflicting obstacle cell was already seen.
The recorded run and diagnostic remain unchanged.

Prospectively test one implementation change: on every new 100-ms public
observation, discard the previous candidate remainder and perform the existing
selection again. Execute only the first command of that candidate. The model
still predicts its declared 500-ms first horizon and the same full candidate
plan; no interpolation or 100-ms prediction is fabricated. This is receding
execution with 500-ms lookahead, not evidence that unexecuted candidate
endpoints occurred. More frequent feedback may help, but does not bound model
error, certify intermediate motion or eliminate action chattering.

Keep the existing model weights, RGB/body/control contracts, RGB-D observer,
retained floor memory, route and view selector, scoring, six action plans,
surface checks, 0.45-m nominal radius, 240 navigation ticks, 0.04-m policy
arrival radius and quiet/dwell rules. The native 0.06-m arrival gate, physical
stops, timing assumptions and ten-command terminal drain remain unchanged.
The collector changes only its explicit controller type and experiment label;
the raw auditor replays that same new type and imports the original command
auditor and native goal function unchanged.

Use exactly the same two fixed augmented first-seed models and randomized
order as the previous two-case probe: full JEPA and full direct, seed
2026091001 on `family_episode_039`. Admit all eighteen completed fits again,
bind both previous outcomes and the clearance diagnostic, and verify exact
physical/public warmup prefixes. No model, seed or checkpoint selection.

Freeze the new source closure before the exclusive
`go2_observation_replan_goal_probe_v1_attempt_001` root. Run two fresh processes
with one numerical thread each, as in the completed two-process probe and
within the passing four-process native/fitting benchmark capacity. Recheck
32 GiB available RAM and 8 GiB output allowance above the unchanged 40-GiB
reserve. Monitor resources, preserve both launched cases and every failure,
and perform full raw replay and source/input verification afterward. No retry
or resume is part of this attempt.

Report both outcomes and measured timing. Native physics remains paused
during compute; the new cadence may cost more. The layout is a development
integration case, with zero independent novel mazes. No result here completes
the broader navigation, matched-baseline, backtracking or hardware goal.
