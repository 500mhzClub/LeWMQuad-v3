# Fresh native planning-map persistence comparison on maze0

Execute CurrentObservationPlanningController with the same assigned full-JEPA
model, sensors, physics, renderer witnesses, gait, mission and budget as the
current completed learned maze0 baseline. Change only accumulated planning-cell
queries to the separately defined current paired-observation view. Keep visual
tracking/floor anchors, contact history, prediction/residuals, mission/settling
and scan state. This is not a fully memoryless controller or runtime shortcut.

Exclusive output:go2_current_observation_planning_maze_pilot_v1_attempt_001.
Run scripts/run_go2_current_observation_planning_maze_pilot_v1.py with the actual
completed --learned-result-sha256, first with --preflight-only. Require that
baseline's full raw audit, model identity and floor-transport intervention
comparison, even when its physical/strict visibility outcome is negative.
Also require the exact completed planning-map prefix result
8d0e1391f95c71cd71394356c902a9367a1d37fad75571197456dd0b0a6ca90a
and all eleven saved prospective decisions. Never read incomplete baseline
outcomes or alter the running/frozen source to admit this experiment.

Assess hardware/competition/storage before launch. Require32GiBavailableRAM,
40GiBartifact reserve plus10GiBcollection and1GiBpersistence headroom. One CPU
native scene/process, one OpenCV/PyTorch/BLAS thread and one spawned task. These
are admission checks, not OS quotas. Keep3000navigation ticks shared outbound/
return, three warmup observations, ten terminal zero commands and all existing
physical/acquisition/measurement failure rules. Physics remains paused during
computation. Do not launch beside another native scene.

The collector preserves the original physics and acquisition implementation.
The full raw audit reconstructs every public observation, entire controller
decision, actual command, native arrival window, path retracing, quiet boundary,
camera visibility and hard measurement result using a freshly loaded identical
model. Preserve every failure and distinguish physical candidates from strict
verified round trips. Keep collected files and completed audits even if a later
prefix or artifact check fails; no overwrite, automatic retry or deletion.

Compare the fresh run directly with the completed current learned baseline
through observation10: all1250physical samples, eleven paired public packets,
ten earlier actual requests and all eleven complete saved prospective decisions.
Shared observed/executed residual state and all eight raw forecast banks must
match. The original command at10 is[0,0,.45], candidate[.16,0,.45]. A physical
stop while executing the new request remains an actual outcome. Do not demand
or infer equality of any following trajectory. The prior prefix's observation
source was the completed eleventh episode; current-baseline physical equality
must be established anew by this native comparison.

Preserve the established execution order: finish current learned baseline and
readout, fixed independent learned layouts1–3, reactive maze0 and paired readout,
then fixed independent reactive layouts1–3. This memory pilot follows those
comparisons unless an explicit later goal decision changes that schedule.
No queued method or source is modified by preparation. This experiment alone
cannot establish statistical reliability, unseen-layout memory advantage,
JEPA advantage,100ms timing, deployment or hardware qualification.
