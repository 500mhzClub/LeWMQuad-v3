# Prospective longer-budget development pipeline

The completed no-RGB direct maze-02 collection reached the outbound goal at
frame 2935, with at most 6.8 seconds left before its original cutoff. Native
geometry and the existing speed bound imply a straight-line return lower
bound of 15.3515 seconds, excluding walls, turns and quiet dwell. The physical
readout and budget diagnosis are recorded separately; the original full audit
is still pending when this source is prepared.

This successor prepares a 4,000-navigation-tick global budget (400 simulated
seconds), using the existing mission maximum. It allows 4,013 command intervals
including the three warmup and ten terminal drain intervals, 4,014 observations,
and at most 201,400 physics samples. If outbound arrival remained at frame
2935, there would be 106.8 seconds before the new cutoff. Neither that arrival
time nor successful return is assumed. The global clock and memory do not reset
at outbound arrival.

`scripts/extended_budget_anchored_maze_development.py` composes the original
anchored collector and full raw auditor with isolated function globals. It
retains original function code, defaults and class closures. It extends all
identified acquisition, auxiliary RGB/depth packet, renderer witness, compressed
decision stream, RGB-D replay population and command audit bounds together.
The paired/dual/renderer classes use cooperative inheritance to preserve the
original capture order and superclass behavior. No imported globals change.

The episode collection allowance rises from 10 to 14 GiB for the longer frame
population. The original 40 GiB reserve and 1 GiB persistence headroom remain.
The existing 4,096-frame memory bound is sufficient. The independent native
evaluator already admits the 4,000-tick mission maximum; its success, contact,
arrival, speed, route and quiet-dwell criteria are reused exactly. Sensor
reconstruction, raw model replay and strict visibility checks are retained.

The prospective scientific intervention is budget only: same controller,
assigned no-RGB direct predictive model, layout-02 specification, appearance,
physics seed, sensing, gait and all safety checks as the original sixth case.
RGB remains in tracking and mapping. The model's RGB ablation does not make
this a non-predictive baseline. No JEPA advantage is implied.

This module has no runtime launcher. Before any execution, complete the original
six-case audit and review, preserve its failures, and admit a separate exclusive
attempt after the existing frontier, hold, commitment-contact and tracking queue.
Bind the original model, raw inputs, seeds and complete source identities;
verify the original pre-cutoff command/observation prefix for budget-only
interpretation, permitting only the declared mission-budget receipt field to
differ before the original terminal boundary. A divergence is evidence to
report, not grounds to replace an attempt. A launcher must recheck resources
and native ownership and preserve terminal outcomes without automatic retry.

Synthetic tests establish recording/audit bounds and code composition only.
No scene, new sensor episode, model replay, hardware command, independent-layout
qualification or navigation success follows from source preparation.
