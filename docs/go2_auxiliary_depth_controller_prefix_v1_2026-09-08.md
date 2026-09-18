# Integrated auxiliary-depth controller prefix replay V1

Use the complete audited 20-frame robot-visible primary/auxiliary capture and
the passing unchanged-primary compatibility result. Keep the preselected
corrected seed-2026091001 full-JEPA/full-direct models and original observer,
100-ms execution semantics, route/view selector, eight-step forecasts and
arrival/collision gates. Add the distinct causal auxiliary depth packet and
observed-pose auxiliary map integration only. Retain auxiliary obstacles and
unknowns along with floor evidence; never route native poses or segmentation
into the online map/model. The learned models still consume primary RGB/body/
control histories only.

Replay both integrated controllers twice from fresh model/state instances and
require exact new records and unchanged model state. All 20 sensor observations
come from the original fixed 19-command prefix. Bound causal comparisons by
the first integrated-controller command or terminal difference, including its
preceding observation. Later records are shadow observations on the recorded
trajectory; infer no counterfactual action outcome or closed-loop success.
Report all map/observer failures and auxiliary return-classification counts.

Freeze source, focused packet/map/controller/boundary tests, the primary
compatibility report and complete source/input chain before
`go2_auxiliary_depth_controller_prefix_v1_attempt_001`. Use one CPU process and
numerical thread, 8 GiB available RAM and 256 MiB output allowance above the
40-GiB reserve. This is a bounded replay, not a large training/native workload.
Reverify input/source/model identities and preserve every failure. No native
mission, optimizer, independent-maze, realistic timing or hardware qualification
is performed; a passing replay only precedes a prospective mission.
