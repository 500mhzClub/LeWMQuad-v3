# Reactive comparator shares current observed state and changes the first action

Implemented ReactiveFloorTransportController as a separate comparator. It uses
the current paired-camera motion observer, measured floor registration/transport,
map/contact memory and settled-boundary mission. Its unchanged reactive connector
selector uses observed route geometry and heading, with no learned model,
forecast residual or predicted candidate outcomes. The older reactive native
episodes used older perception/map/mission implementations and therefore cannot
serve as this comparator's execution evidence.

Controller tests52285 CLOSED:4passed3.86s. Realistic synthetic public packets
produce identical learned/reactive warmup evidence, registered/transported poses,
maps and settling receipts; missing auxiliary RGB stops before another map update.
Mission/command integration preserves both settled arrival transitions and map
objects, the geometry-wait budget and terminal latch. AST comparison limits the
inherited reactive advance changes to the current pose accessor and full3D
settling position. These are component checks, not native navigation evidence.

Prefix tests14185 CLOSED:11passed2.06s. Shared-state comparison accepts only the
validated mission pose-source wording change; pose, map, partition, distance,
settling and raw evidence differences fail. Command/terminal disagreement and
the fixed64frame limit prevent requests for the next old observation/decision.
After testing, one receipt field was renamed to accurately describe the current
command match; no algorithm changed.

Preflight44746 CLOSED:1666source/input bindings verified, no output/scene created.
76,508,983,296bytes available RAM;101,523,349,504bytes artifact free;CPU3.3%busy,
bothGPUs0%busy. The existing native parent/worker were the only substantial
competing Python job. One CPU replay used an8GiB admission and256MiB allowance
above40GiBreserve, beside the unchanged single native scene.

Execution74127 CLOSED exit0,17.509519994s after launch. Root:
go2_reactive_floor_transport_prefix_v1_attempt_001.

- Result:71a5ecd8486d6d8354762c5dc249307cc7d1bf7dc57fe6d1f2df76372d5aaba9
- Launch:b4dc1fb36ed790ad66094c6458e212fd2496e462c75e2e34184bbd608f845d57
- Complete decision stream:2f34b5424addae309150ff76c8cf9f87ba3de79632d20da5b082b9f62aee9fc4

All1666source bindings, completed eleventh native artifact bindings and the
current native source-identity launch passed verification before and after.
The result hash was checked again when reading these scalars. No native scene,
learned model or current running trajectory observation was loaded by the replay.

Four original native observations0..3 were reconstructed with primary RGBD,
public body/fast-gyro data and paired auxiliary RGB/depth. Raw and admitted pose,
map receipts, auxiliary partition, observed goal distance and settling mission
match exactly, apart from the explicitly validated pose-source wording. All
three earlier actual commands match. At frame3, reactive requests[.2,0,0]
(forward) instead of learned[.16,0,.45](left arc); both remain nonterminal.
The reactive waypoint is[.475,.025]m in the observed map and measured heading
error.0504942877767rad. The current-geometry rule records unknown connector
cells and performs no predicted future-path check. No observation after that
changed command was consumed or interpreted as its outcome.

This completes actual-packet integration on the common pre-command prefix. It
does not establish the reactive command's physical outcome, a native matched
comparison, isolated prediction-ranking effect, memory advantage, JEPA advantage,
real-time performance or qualification. Predicted feasibility and current
geometry are different method-level rules; do not describe their gates as equal.

Future reactive native execution needs its own new collector/audit definition,
fresh scene and physical/public/common-decision comparison through observation3
(900physics samples before that command), followed only by its newly executed
trajectory. No such native launch is prepared or started by this result. Keep
the current learned maze0 attempt and fixed independent-layout cohort unchanged.
Current native session19976/PID2447506 remains live:1052timingrows through1051,
21m33s elapsed/21m13s CPU,98.5%CPU,RSS5,679,376KiB;no result/failure file.
Full goal active, eleven completed maze episodes and zero verified round trips.
