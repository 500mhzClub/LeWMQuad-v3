# Fresh physical maze1 tracking fallback pilot V1

Run DirectFlowFloorTransportController on the same development maze1, assigned
full-JEPA model, training-only correction, paired camera observations, gyro,
gait, dynamics and mission as the completed independent learned maze1 trial.
Only the explicitly defined missing-pose association fallback changes. Preserve
original rigid/gyro/temporal/reference-conflict checks and the ten-frame measured
bridge budget. No reference, gyro or pose history reset, no calibration change,
no residual-feasibility intervention, and no model training. The repeated layout
is development-only; this is not a new independent generalization trial.

Require completed learned cohort result
`a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`,
and completed corrected tracking prefix V2 result
`345b54f1b3c647be516040f2b8bf03b4ab3c709b9737dc0bdf7c78bedcacdbe3`.
The prefix admission checks all215saved decisions,214prior actual requests,
211preceding raw prediction banks, exact original decisions before214, accepted
current full-controller recovery at214 and unchanged weights/inputs. V1's
terminal serialization-validation failure remains preserved and authenticated;
it does not serve as a successful prefix. No original artifact is replaced.

Fresh physical collection begins at episode start with the unchanged public
mission and3000navigation ticks shared outbound/return, three warmup
observations and ten terminal zero commands. One CPU native scene/process,
one OpenCV/PyTorch/BLAS thread, one spawned task. Physics pauses for computation.
The collector and audit preserve original physical acquisition and evaluation
calculations except the controller class and explicit status/enable metadata.
Native truth remains available to the evaluator only, never to the controller.

The independent raw audit loads a fresh identical assigned model, reconstructs
all paired observations and complete controller decisions, audits every actual
command, and evaluates native settled arrivals, route retracing, terminal quiet,
strict visibility, hard measurement failures and physical/acquisition stops.
Retain failures and distinguish a physical candidate from a strict verified
round trip. Authenticate source/input/model/artifact identities before and after.

Compare the fresh physical prefix with original learned maze1 and the completed
V2 replay:11,450physics samples through observation214, all215paired public
packets, every earlier actual command and all215complete prospective decisions.
The original failure's zero command at214 becomes left-turn[0,0,0.45]. Do not
compare or borrow following physical outcomes. A partial changed command due
to a new physical stop remains an actual negative outcome. Preserve already
collected files and completed audits if a later prefix or identity check fails.

Output is exclusive `go2_direct_flow_maze01_pilot_v1_attempt_001` under the
existing navigation development artifact root. Runner:
`scripts/run_go2_direct_flow_maze01_pilot_v1.py --learned-cohort-result-sha256`
with the exact cohort hash above. First run `--preflight-only`; that path must
return before output creation or worker/model/scene execution. Require32GiB
available RAM and40GiBartifact reserve plus10GiBcollection/1GiBpersistence
headroom. These are admission checks, not enforced OS resource quotas. Refresh
CPU topology/affinity/load, RAM, GPU/VRAM, storage and competing jobs before
substantial work. CPU-only preflight may overlap the existing native pilot.

Queue order remains the active planning-memory maze0 pilot, then the already
prepared residual-feasibility maze2 pilot, then this tracking maze1 pilot.
Only one native scene at a time. Preserve all running and frozen attempts.
No automatic retry, deletion, sealed access, model modification, real-robot
motion, production/promotion or deployment claim follows from this protocol.
Even a native pass here would not by itself establish reliability, independent
layout generalization, JEPA/memory advantage,100ms timing or hardware readiness.
