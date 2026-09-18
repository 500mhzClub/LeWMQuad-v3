# Prospective response to discretionary waypoint holds

The first 1,000 observed decisions of the active adapter case contain 595
discretionary holds, each outranking raw-feasible movement. This distinct
development candidate changes that preference after ten consecutive selected
waypoint holds. Ten is the existing reobservation wait allowance, fixed before
testing this candidate; it is not fitted to a particular favorable frame.

On the next observation, choose the highest original model utility among the
raw-feasible left/right in-place turns, even if its utility is below holding.
Retain the exact phase, measured articulated-surface and all-eight-step nominal
path checks, forecasts, scores and residual targets. No corrected path exception
is introduced. If neither turn passes, keep the original hold. Accept no hold
whose original raw-feasible alternative already has higher utility.

Issue only one original 100-ms turn command, reset the hold count, then obtain
a fresh observation and run the entire original selector again. This is not an
open-loop turn tape. It introduces no forward, backward or lateral command.
Nonhold selections, view acquisition, mission-goal targets, existing recovery
interventions, mission target changes and gaps in selector calls reset the
counter. Repeated or backwards frame calls are rejected. Mission settling and
warmup never call this selector and cannot earn intervention credit.

The original 3,000-command mission budget remains the global execution bound.
This candidate does not guarantee useful views, escape, progress or absence of
cycles, and does not bound holds when both turns remain vetoed. In particular,
the currently raw-feasible left turn can initially turn away from the waypoint.
The experiment tests whether reobservation can resolve that local stagnation;
turning, increased visible area and model scores are not navigation success.

The implementation is a separately named subclass of the exact active adapter
case's residual anchored controller. It changes neither the active six-case
batch nor the separately queued reached-frontier controller. It is not yet the
final independent-comparison policy. Do not render or run independent layouts
to tune this candidate.

Before native execution, complete a saved-decision boundary check and then a
raw-sensor, model-inference prefix replay against the original completed case.
Stop each prefix at its first changed command: later original observations are
not outcomes of the candidate. Authenticate the same expanded JEPA checkpoint,
map/contact/mission state and full original decisions before that boundary.
The queued frontier experiment owns the next native slot; this document adds
no concurrent native launch or automatic replacement of that experiment.
