# Execution-horizon intermediate-waypoint prefix V1

The completed first maze and nominal-reentry repeat both failed. The recovery
restored the observed nominal gate but ordinary planning lost it again without
another maze edge. A diagnostic example at tick 300 of the immutable repeat
has body waypoint [0.0628302845, 0.0034153669] m. The selected hold has 800 ms
utility 0.002185113 m; forward's 800 ms endpoint [0.1215952113, 0.0077293273] m
overshoots that local waypoint, giving utility -0.145565311 m despite its first
100 ms endpoint [0.0061378200, 0.0038659982] m. This motivates a prospective
execution-horizon utility experiment, without claiming what an unexecuted
alternative would have achieved.

`lewm/executed_waypoint_score_development.py` changes only intermediate WAYPOINT
utility to existing distance/bearing potential at the actually committed 100 ms
endpoint. Subtract the existing strictly past, public-pose executed XY residual
mean for scoring, as already done at the exact final goal. Retain unmodified
corrected-model forecasts, all original eight nominal path segments, first-step
surface vetoes, phase allowance, and 800 ms contact penalty. Rerank only those
originally feasible actions. Preserve the old candidate scores and action in the
new receipt. Corrected scoring XY is not a new certified path or error bound.
Exact final-goal behavior, view acquisition and explicit nominal reentry remain
unchanged. Model parameters, observer, memory, mission, backtracking obligation,
3,000-tick shared budget and latched failures remain inherited.

Eight focused source tests pass, including a real calculated full-horizon
overshoot example, causal residual admission, original surface/path/phase vetoes,
full-horizon contact cost, unchanged final/view/reentry paths and latched failure.
An initial synthetic fixture omitted the required native-state provenance field;
it was corrected before launch, with the production check retained.

Run `scripts/replay_go2_executed_waypoint_maze_prefix_v1.py` once at exclusive
`go2_executed_waypoint_maze_prefix_v1_attempt_001`. Bind completed recovery native
result `b48f3c79d19889c67438073a3a7305d0eafcbc5f8b7286ea014e2eb1fd711739`
and readout `0ff22595c9a49fb855212952a11f02915a6269c22db3011180c03dbfda0a20f7`,
all artifacts and frozen recursive source/native/model dependencies before and
after replay. One fresh CPU model/controller consumes only original public paired
RGB-D and body packets, stopping immediately at the first changed command or
terminal policy. Compare observer, map, mission, causal residuals, raw forecasts
and original constraints exactly through that decision. Compare the new selection
with the exact pure score transformation of the old selection and its causal
pre-command residual receipt. Other original decision fields must remain exact
before the first intervention; score metadata is intentionally different.

Require 8 GiB RAM and 256 MiB above the unchanged 40 GiB artifact reserve; record
hardware before/after. One ordered prefix offers no independent process unit;
no GPU, native scene or training is used. Stop before reading any outcome of the
changed command. This is prospective compatibility evidence on a reused maze,
not navigation success or matched planning/memory attribution. A later native
attempt requires its own explicit source, resource envelope and raw audit.
