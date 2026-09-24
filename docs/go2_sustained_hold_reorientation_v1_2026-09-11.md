# Prospective sustained hold reorientation

The completed hold-reorientation experiment executed 138 left-turn interventions.
Native physics confirmed that a short turn followed by ten holds usually lost
almost all of its heading change. The original raw 800 ms nominal-path checks
rejected every translation after the first intervention. This candidate tests
whether a bounded turn, checked again against fresh observations and predictions
at every 100 ms boundary, can leave that repeated hold/turn state.

The original ten-discretionary-hold trigger, first chosen turn, current raw model
forecasts, scores, sensor/map state, and every sampled-surface and nominal-path
veto are retained. At the trigger, the candidate records the observed map heading
and the selected action's predicted yaw at 800 ms. If that predicted yaw is
defined and points in the commanded turn direction, it becomes a measured-angle
target. An opposite or zero signed target preserves the original single turn.

While the original planner continues to choose discretionary hold, the candidate
may continue the same turn. Each subsequent command requires a newly admitted
observation, freshly computed complete forecast and score bank, and that turn's
current phase, surface and full 800 ms path checks. It stops when the measured
heading change reaches the original predicted target, after at most eight total
turn commands, when the turn becomes inadmissible, when the original planner
chooses another action or special recovery, or on a mission-goal change or missing
observation. A stopped recovery must accrue fresh hold credit before retriggering.
The eight-command cap is the original eight-interval forecast-bank length; it is
not an open-loop action tape. Only one 100 ms command is issued per observation.

The target is a prediction, not a guarantee of achieved rotation or safety.
Heading comes from the admitted observed map pose, not command integration or
simulator state. No observed-map reset, correction-limit change, contact
exemption, smaller nominal radius, retraining, modified success criterion or
queued-controller mutation is part of this candidate.

Implementation: `lewm/sustained_hold_reorientation_development.py` and
`lewm/sustained_hold_reorientation_controller_development.py`. Focused tests are
`lewm/tests/test_sustained_hold_reorientation_development.py`.

Before any native execution, reconstruct the existing helper on the completed
original saved history and locate the first changed requested command. Then run
a raw complete-controller prospective replay through that boundary with the
original model, observation and physical-input bindings. Do not consume later
original observations as evidence of successor outcomes. A fresh native
diagnostic requires separate queue/resource admission and the unchanged raw
audit. This source candidate does not select an independent-study policy or
establish navigation, timing or hardware qualification.
