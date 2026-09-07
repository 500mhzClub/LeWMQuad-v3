# Actuator gain pair development V1

Specified before collection on 5 September 2026. The preceding native identity
readback measured 100/10 joint gains where the pinned checkpoint configuration
specifies 20/0.5. This study tests that single configuration intervention.
It does not retroactively repair previous results or assert complete training
environment parity.

Sixteen fresh trials: the same eight observed development geometries, each with
two arms in fixed order, `default` then `checkpoint`. Case i uses seed
2026090800+i in both arms. Both use the factorial V1 **baseline** controller,
its exact 85-tick-per-edge budget, unchanged contact/arrival thresholds, 1.5 s
settling, camera mount, gait checkpoint and 2-ms/20-ms/100-ms timing. The second
boundary is the same within-corridor continuation. No state is loaded from a
previous run. All geometry widths/motifs remain development-only.

Read the native joint gains and initial measured pose, twist, joint positions
and velocities before the intervention. Require the original gains to be
100/10 and bind the checkpoint's expected 20/0.5. `default` leaves them untouched;
`checkpoint` explicitly sets only the 12 actuated joints to the configuration's
values. Read back every joint and require the intended gains before any physics
step and after the last step. Match pre-intervention physical state across arms;
settling trajectories are expected to differ because gains are the intervention.
Record all native contacts, physical states, commands, causal decisions and RGB
boundaries exactly as in the preceding study.

Primary endpoint remains two sustained, directed, contact-free crossings and
a usable instantaneous final arrival. Report sustained arrival separately;
do not infer stable stopping from an instantaneous pass. Report contacts,
first-arrival checks, total active duration and the paired case table. Numerical
thresholds are unchanged. Infer no independent-maze uncertainty from this panel.

Physical and integrity stopping rules are unchanged. A physical failure ends
that trial; an infrastructure/integrity failure ends the study. No retries,
extra seeds, adaptive controllers or parameter searches. Fixed completion is an
interpretable result, including a negative effect. If gains help, the next gate
is broader local-control/command characterization and sensor-backed execution,
not immediate JEPA or hardware promotion. If they do not help, retain the null
and investigate other specific plant/adapter differences before more model work.
