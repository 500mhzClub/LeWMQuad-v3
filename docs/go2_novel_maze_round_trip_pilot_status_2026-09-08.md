# First maze pilot working status

This mutable working note is not a substitute for the completed result report:
`docs/go2_novel_maze_round_trip_pilot_result_2026-09-08.md`.
The active full navigation/comparison goal remains unachieved. No blocker is asserted.

## Completed first maze

Artifact base:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.

- Native root: `go2_novel_maze_round_trip_pilot_v1_attempt_001`.
- Launch: `98f656a5bd8c4d7afdde6cda6a094afcb2c2b33c0cccb8efa30f6585947e7985`, 1,401 bound sources.
- Native result: `5874c1fea08b40d90e676b69acb7d3a103d96a6281e0911801bf6bd6e1ab1570`.
- Readout root: `go2_novel_maze_round_trip_readout_v1_attempt_001`.
- Readout result: `fe2b0fbdadba8f1ae29e67d8897d2ea3bdfe36ec5caa8e4a4a4200a349087bc4`.
- Tool sessions 57614 (native) and 75341 (readout) completed.

Collection: 424 commands, 425 paired observations, 21,950 physics samples; terminal
at decision 414 with no phase-admissible surface/nominal candidate, then ten zero
drain commands. No physical/acquisition stop or arrival. Raw sensor/model/command,
model-state and physical visibility audits passed. Native cell traversal:
`[-1,0] -> [0,0] -> [0,-1]`; two valid edges. Native path length 2.407213306 m,
minimum outbound-goal distance 2.768958301 m, terminal distance 3.010469432 m.
There is now one completed independently raw-audited new development-maze execution,
and zero verified arrivals or round trips. All launched sources/results stay frozen.

## Next explicit recovery check

New source:
`lewm/nominal_clearance_reentry_development.py` and
`lewm/nominal_reentry_round_trip_controller_development.py`.
Eight tests passed in 1.71 s. The original 0.45 m veto remains recorded; an
experimental nonzero recovery may be selected only from an already violated
current nominal radius, within the existing phase and surface restrictions,
with all eight predicted segments no closer than current observed clearance and
strict first-step predicted clearance gain. Its native outcome is recorded below.

Completed prefix script:
`scripts/replay_go2_nominal_reentry_maze_prefix_v1.py`.
Protocol: `docs/go2_nominal_reentry_maze_prefix_v1_2026-09-08.md`.
Output: `go2_nominal_reentry_maze_prefix_v1_attempt_001`.
It completed with 408 observations and the first changed command at tick 407:
left turn instead of zero wait. All original decisions before intervention,
public observations, map, mission, raw forecasts and original constraints matched.
No new-action outcome was read. Result:
`8ece4c85f9226265b5c6a425ff589071024738c815513cb0f0d34063e02bb77b`.
Its 1,410 sources and bound first-maze report are frozen. Tool session 36732 completed.

## Completed native recovery attempt

Script: `scripts/run_go2_nominal_reentry_maze_pilot_v1.py`.
Output: `go2_nominal_reentry_maze_pilot_v1_attempt_001`.
Launch: `676c13ed551a986ee5669503bcac961de63d564dc932be9dbaabe5993ee37c59`,
1,419 bound sources. Tool session **58272** completed; parent PID **2342374** and
worker PID **2342432** terminated. Result SHA-256:
`b48f3c79d19889c67438073a3a7305d0eafcbc5f8b7286ea014e2eb1fd711739`.
Navigation failed at 432; all raw audits and exact prefix comparison passed.

The new collector and audit are narrow separate successors, with eight reentry
tests, two native source-scope tests and six physical/public prefix-comparison
tests passing. They preserve the same scene, model, sensors, 3,000-tick budget,
arrival/return and raw measurement checks. The native result additionally must
match all 408 observations and 21,100 physics samples before the bound intervention.
This is a reused development maze, not a new independent layout.

The native protocol and source files are frozen by launch. Dedicated readout
`scripts/read_go2_nominal_reentry_maze_pilot_v1.py` completed in session 90071,
with result `0ff22595c9a49fb855212952a11f02915a6269c22db3011180c03dbfda0a20f7`.
Full report: `docs/go2_nominal_reentry_maze_pilot_result_2026-09-08.md`.
Three recovery turns restored the observed nominal gate at 410, but ordinary
planning lost it again at 422. No additional maze edge or arrival was reached.
The successor investigates execution-horizon scoring for intermediate waypoints;
the completed recovery result and its sources remain immutable.

Execution-horizon waypoint policy and prefix are now complete:
`docs/go2_executed_waypoint_maze_prefix_result_2026-09-08.md`.
Prefix result `5ab4f64ba8e528d55236f8ca6ab995c338e2272be95efebd4ef12192f517e37f`
first differs at command 3, left arc instead of forward, with identical prior
causal observations and constraints. Session 5357 completed. Eight policy
tests and seven native-scope/prefix checks passed. Prepared native script:
`scripts/run_go2_executed_waypoint_maze_pilot_v1.py`, exclusive
`go2_executed_waypoint_maze_pilot_v1_attempt_001`. Explicit resource allowance
is now 10 GiB collection plus 1 GiB persistence above the unchanged 40 GiB reserve.

## Executed-waypoint native and readout complete

Native launch `f7fec194358f3821c036df413c360f355cd758ec7d1ca5c1b7466b99ceea9f4e`
binds 1,436 sources. Tool session **58173**, parent PID **2345062** and worker
PID **2345120** are the actual current handles; they were confirmed live after
launch. Revalidate these handles before waiting and never restart from silence.
The completed collection ended at decision 1155 for no
phase-admissible surface/nominal candidate, then ten zero drain commands. There
were 1,165 complete commands, 1,166 paired observations and 59,000 physics samples.
No physical or acquisition stop and no arrival were recorded. Final observed
outbound-goal distance was 2.851210920 m. All raw audits and physical/public
prefix comparisons passed; session 58173 completed and its worker terminated.
Top-level result: `d5ba1136c969cd6591028f435a93f0e35a4dae7b9cbf217524887b459271ace8`.
Readout session 41463 also completed, with result
`1bb9ae412452fd68a99fa827ad304d9201f5dfdc2b7bb24f57e003390ba7799c`.
Full report: `docs/go2_executed_waypoint_maze_pilot_result_2026-09-08.md`.
Actual traversal crossed three open edges through [0,0], [0,1], [1,1], with
4.909109277 m path length and no arrival. Eleven recovery turns failed to restore
the nominal gate; final observed clearance was 0.431087914 m. No navigation success.
Hardware admission: 82.811 GB RAM available, 55.261 GB artifact space free,
CPU 0.2% busy and GPUs idle. One CPU scene, no training. The declared 10+1 GiB
envelope is a censoring constraint; 40 GiB reserve remains unchanged.

Completed dedicated readout:
`scripts/read_go2_executed_waypoint_maze_pilot_v1.py` and
`docs/go2_executed_waypoint_maze_readout_v1_2026-09-08.md`.
The helper's selected-action native/error/censoring test passed in 1.78 s.
It retains failed outcomes and existing recovery evidence, and adds actual
100 ms errors for original forecasts versus causal scoring XY. A changed local
ranking is not treated as an alternative executed trajectory. Across 1,120
complete selected waypoint intervals, mean original first-step XY error was
7.719 mm and causal scoring XY error 6.154 mm. Median full iteration was 861.631 ms;
all cycles exceeded the 100 ms simulated interval while physics paused.

## Reactive baseline source and completed prefixes

Added an independent reactive controller with the same public observer, persistent
map and round-trip mission, without a world model, forecast residual or candidate
future-outcome evaluation. Its geometry gates differ explicitly from predicted
future surface/path gates; it is not yet a pure ranking-only ablation.

The initial measured-floor connector variant passed 11 tests and completed a
four-observation prefix, but vetoed startup solely on unseen floor beneath the
robot. Preserved result:
`2ceee2441c3600eff6b2c1176e0c70d93e8d535b419957db8cbcd3326bcc3cdc`.
Report: `docs/go2_reactive_observed_route_prefix_result_2026-09-08.md`.

The separately named nominal-route variant explicitly permits the original
unknown start connector while retaining current-footprint and known-obstacle
connector gates. Five additional tests passed. Its eight-observation prefix first
differs at command 7, forward instead of learned left arc; observed pose/map/mission
and earlier command outputs match exactly. Result:
`0653ca664495f8113296e0958bd2a1d2a3b2b6e3da536516e7efa70f17f8394a`.
Report: `docs/go2_reactive_nominal_route_prefix_result_2026-09-08.md`.
Sessions 38519 and 29906 both completed. All launched baseline sources are frozen.
No native baseline episode has been executed. Next baseline work needs its own
collector and independent raw audit; do not feed old future outcomes after a
changed command or silently reuse learned prediction gates.

The native baseline collector, fresh-controller raw audit, command audit and
two physical/public prefix comparisons are now prepared. Nine focused native
checks passed in 1.73 s. Script:
`scripts/run_go2_reactive_nominal_maze_pilot_v1.py`.
Protocol: `docs/go2_reactive_nominal_maze_pilot_v1_2026-09-08.md`.
Exclusive future output: `go2_reactive_nominal_maze_pilot_v1_attempt_001`.
Preflight-only session 90735 completed: all completed inputs and 1,462 source
bindings verified; no output root or scene created. RAM admission passed, but
storage admission failed: 51,645,128,704 bytes available versus 54,760,833,024
required for the same full-length 10+1 GiB envelope over the 40 GiB reserve.
Do not shrink that envelope. Real launch command after resources permit:
`scripts/run_go2_reactive_nominal_maze_pilot_v1.py --prefix-result-sha256 0653ca664495f8113296e0958bd2a1d2a3b2b6e3da536516e7efa70f17f8394a`.

## Exact cache proposal; explicit user approval pending

Zero-step current-scene cache inspection completed in session 18648, result
`f409c5a6211b7443e06b1045be5ae6983332922670cfb2b6ce6738a82e9876ce` at
`go2_maze_geometry_cache_identity_v1_attempt_001`. Its 1,440-source launch binds
53 current geometries and 13 distinct derived cache keys. None of those keys
exists among the 8,304 historical GSD leaves; all keys remain explicitly protected,
including future creation. No SDF payload was deserialized by the inspector,
no physics step ran and no preexisting cache metadata changed.

Exact proposed retirement metadata:
`docs/go2_gsd_cache_retirement_proposal_2026-09-08.json`, SHA-256
`bc405303fb8f7d227c71bf7e965d2262ca6ad8f8e817e78512427bbacf8eaa77`.
Review: `docs/go2_gsd_cache_retirement_review_2026-09-08.md`.
It proposes 8,304 ordinary cache files, 81.782 GiB allocated, excluding current
geometry keys and all experiment artifacts. Historical geometry reuse may require
regeneration; exact regenerated bytes are not established. Two identity tests and
two retirement-helper tests passed. No deletion has occurred.

An asynchronous user approval question is pending; **no approval response has
been received**. Do not infer approval from the preselected option or elapsed time.
The prepared executor is `scripts/retire_go2_reviewed_geometry_cache_v1.py` and
requires the exact proposal hash plus a separately recorded, genuine user approval
message and authorization-file hash. No authorization file has been created.
Only if the user approves, record that approval, recheck original PIDs terminated,
other native tasks/open cache users and all candidate metadata, then execute the
exact reviewed deletion with its journal. Otherwise continue safe source/readout
work; do not delete cache, lower the reserve or mark the full goal achieved.

Storage follow-up: `docs/go2_novel_maze_storage_followup_2026-09-08.md`.
The old TinyQuadJEPA candidate is absent. The 81.782 GiB Genesis GSD cache contains
collision geometry SDFs, not just compiler products; no cache payload was opened
or removed. Preserve current scene cache identities before any concrete retirement
proposal. No cleanup authorization is assumed, and no genuine blocker is asserted
while the native attempt and source/audit work remain available.

Hardware before prefix: 83.207 GB available RAM, 56.734 GB free artifact space,
CPU approximately 0.3% busy and no competing experiment. Before the recovery
native launch there were 83.226 GB available RAM and 56.702 GB free artifact space.
The new explicit envelope permits 11 GiB collection plus 1 GiB persistence above
the unchanged 40 GiB reserve, instead of the predecessor's 12+1 GiB allowance.
This smaller resource allowance may censor a long attempt and must be reported;
it does not alter mission timing or silently lower the reserve. The original
collector remains unchanged. Do not delete dependencies or mutate launched sources.

Matched direct, remaining layouts, reactive/non-predictive controls and
RGB/planning/memory attribution remain outstanding. Any new recovery attempt
must be independently collected/audited and does not turn this reused maze into
new independent evaluation evidence.


## Translating view-recovery successor: prefix complete; native storage unavailable

This goal turn made progress: a new recovery policy, verified public-sensor
prefix, native collector/audit/readout and completed resource preflight now
exist. The full goal remains active and unachieved. Three native policy attempts
on one maze still provide zero verified arrivals or round trips; none of the
other maze indices has been executed. Cache approval has not arrived and no
cleanup, authorization file or native successor output was created.

The new view-recovery policy reuses the frozen nominal reentry function but
explicitly expands the recovery action allowance during VIEW_ACQUISITION. It
retains the original phase receipt, raw predictions, surface and all-eight-path
checks and separates original-phase eligible counts from recovery eligible
counts. Translations compete with turns under the same first-endpoint gain minus
full-plan contact utility. Hold, worsened paths, surface conflicts and exhausted
view budgets cannot activate recovery. Other phases and mission/observer state
remain inherited. Original launched sources are unchanged.

Eleven policy and exact predecessor-transformation tests passed in 1.79 s.
`scripts/replay_go2_view_reentry_maze_prefix_v1.py` completed in session 65832,
output `go2_view_reentry_maze_prefix_v1_attempt_001`, result SHA-256
`6d550308c908eca596149df684f639a032c1cef706a36e19aa31cff7a5cfd3f4`.
Launch SHA-256 is
`d99dc111015ffd3ae1859b0d6cf554349e5667cbcd854ea5e909b3525ea38bd6`,
binding 1,447 sources; decision stream SHA-256 is
`17c45e505d372bcca54e47541b673dee9604649798b8bbd5b6faa8497b7f5e64`.
Post-launch wall time was 387.114 s. No process from this prefix remains active.

All 596 causal observations/maps/mission/residual receipts, raw forecasts,
original constraints and preceding commands matched. First change was command
595, right arc [0.16,0,-0.45] instead of zero, with no terminal/model failure.
Current clearance was 0.449071776928 m. Right arc and forward passed the explicit
nonworsening recovery test, but both were excluded by the original view phase.
Right arc's first-endpoint predicted gain was 1.070005498 mm, contact score
0.0364070985845 and utility -0.0426185128031 m. Replay stopped before consuming
any observation after that unexecuted command. This intervention is earlier than
the predecessor terminal stop at 1155; the old continuation is not a new-policy
outcome. No physical clearance or navigation improvement is proved.
Report: `docs/go2_view_reentry_maze_prefix_result_2026-09-08.md`.
The completed prefix's launched sources are frozen.

The successor native pipeline is prepared but not launched:
`scripts/run_go2_view_reentry_maze_pilot_v1.py`, with the same 10+1 GiB allowance
above 40 GiB reserve and 32 GiB memory admission. Collector/raw-audit scope test
passed in 0.11 s. It admits the completed prefix result above. Preflight-only
session 60822 completed and verified all inputs and 1,453 sources. Available RAM
82,770,055,168 bytes passed; artifact free space 51,606,925,312 bytes was below
54,760,833,024 required (3,153,907,712 bytes short). CPU was 0.2% busy, GPUs idle,
no competing experiment. No output root or scene was created. Do not lower the
envelope to force launch. Reactive baseline remains first in the native queue.

`scripts/read_go2_view_reentry_maze_pilot_v1.py` is prepared; its actual executed
translation/censored-endpoint test passed in 2.01 s. The reactive readout now
checks same-layout pairing and preserves each audited outcome without claiming
planning/memory attribution; both tests passed in 2.08 s. Thirteen explicit new
or revised Python sources passed syntax/whitespace checks. No native baseline,
new recovery episode or cleanup output root exists.

Next: actual user approval of the already reviewed cache proposal is still
pending. Do not treat a goal continuation or status question as approval. If
approved, use the prepared exact retirement executor after all metadata/process
rechecks, then reassess resources and run the reactive baseline before the new
recovery native attempt. Otherwise retain both prepared native pipelines and
all failure evidence. Neither prefix replay nor passing tests achieves the
navigation objective.


## Resource impasse audit: consecutive no-progress check 1

The preceding goal turn was progress (completed recovery implementation, prefix
and native preflight). This continuation rechecked external state: artifact free
space 51,606,192,128 bytes versus 54,760,833,024 required, shortfall 3,154,640,896
bytes. RAM was 83,611,611,136 bytes. No run/replay/inspection/readout/retirement
experiment process was live; both prepared native roots and the cleanup output
root were absent. This is not a verified wait on a live job.

No user cache-retirement approval has arrived. Necessary collectors, audits,
readouts and prefix checks are complete for the next experiments; further
retrospective replay would not test the changed command or supply the missing
matched native baseline. No next native launch can pass the unchanged full
resource admission. This check is no progress, not a new scientific result.
Leave the goal active at this first consecutive no-progress impasse check;
revalidate external state on continuation. Do not fabricate approval, shrink the
resource envelope, delete cache or mark navigation achieved.


## Resource impasse audit: consecutive no-progress check 2

The previous turn was no progress, not a live-job wait. This continuation
revalidated the same condition: 51,605,803,008 artifact bytes free versus
54,760,833,024 required, shortfall 3,155,030,016 bytes. No experiment process was
live; both native output roots and the reviewed-cleanup output root remain
absent. Cache-retirement approval has still not arrived. No new scientific work,
launch or deletion occurred. The next necessary native experiment cannot pass
resource admission. Leave the goal active at the second consecutive impasse
check; if the same condition persists on the third and no meaningful next action
becomes available, mark the full goal blocked rather than continue status loops.


## Resource impasse audit: third consecutive check; goal marked blocked

The previous turn was no progress. Third consecutive revalidation found
51,605,467,136 artifact bytes free versus 54,760,833,024 required, a shortfall of
3,155,365,888 bytes. No experiment process was live. Reactive native, view-recovery
native and reviewed-cleanup output roots remained absent. No cache-retirement
approval arrived; no deletion occurred. All necessary preparation for the next
native experiments is complete, and no further retrospective replay can supply
the missing fresh execution evidence. The full navigation goal was marked
blocked, not complete, after the required repeated-impasse audit.

Resume on an actual external-state change: enough additional artifact space to
pass the full native allowance, or explicit approval of the existing reviewed
cache proposal followed by its exact guarded execution and resource recheck.
Then launch the reactive baseline first, independently audit/read it, and proceed
with the prepared translating view-recovery experiment when resources permit.
Preserve the full objective, frozen sources, failures and pending comparisons.
No verified arrival/round trip, independent-layout success, real-time or hardware
qualification has been established. A resumed blocked goal starts a fresh
three-turn impasse audit if the same condition remains.


## Cache cleanup approved; privileged inspection required

The user approved the recommended Genesis cleanup with "do it". Authorization
is recorded in docs/go2_gsd_cache_retirement_authorization_2026-09-08.json,
SHA-256 c55dfbe858b7d6e17107189a38c6bc3187afc9c9d2b72b05959a52daa705ca5e.
This authorizes the exact reviewed 8,304 GSD files; pip cleanup is not included.
Do not request this approval again.

The unchanged executor session 80367 terminated before artifact creation or
unlink: Linux denied psutil.open_files for protected user-systemd descriptors.
Read-only follow-up identified protected OS-service descriptor lists; none was
exempted from the guard. Noninteractive sudo failed because a password is
required. Nothing was deleted and no retirement output root exists.

Prepared scripts/retire_go2_approved_cache_2026-09-08.sh for the user's terminal.
It binds the original proposal and approval plus the new privileged-inspection
executor c9ecbfae608bafd55f28c78bcf8be3b7baa5156526185cc1ab55cdc07d5306d1.
The executor retains every original check, inspects UID-1000 processes as root,
and permanently drops all UID/GID privileges to 1000 before owner-bound artifact
verification, metadata checks, output creation or unlink. The focused scope/order
test passed in 0.11 s; shell syntax passed. No privileged execution was possible
from this session. The current dependency is terminal administrator access,
not missing deletion approval. After successful execution verify the journal,
retained keys and reclaimed space, then launch the prepared reactive baseline.
See docs/go2_gsd_cache_retirement_privileged_inspection_2026-09-08.md.


## Approved cleanup verified; native baseline launch requested

The user reported the prepared administrator-inspection command complete.
Verified cleanup result SHA-256
fd5068f10140821ee8a4f712e060b56996625f38e1c305fc79a65b61349bcdd3,
status APPROVED_GEOMETRY_CACHE_RETIREMENT_COMPLETE. The journal contains each
of the exact 8,304 approved names once, all are now absent, and all 13 retained
current-maze keys are excluded. None of those retained keys currently exists,
consistent with the original inspection. The cache directory remains intact.
Cleanup records are owned by UID 1000. Allocated bytes retired: 87,812,689,920;
recorded free-space gain: 87,812,009,984 bytes. Current experiment-volume free
space: 139,417,485,312 bytes. No scientific artifact deletion is recorded.
The previous storage and privileged-inspection blockers are resolved.

Requested the prepared reactive nominal native baseline using completed prefix
0653ca664495f8113296e0958bd2a1d2a3b2b6e3da536516e7efa70f17f8394a.
Its launcher must verify all frozen inputs/sources and full RAM/storage admission
before creating the scene. Follow its returned live session, do not restart on
silence, and independently audit/read the completed native outcome. The same
maze is reused; no successful arrival, round trip or independent-layout result
has been added by cache cleanup. The goal tool still reports the old blocked
status; its exposed update API has no active/resume status. Do not recreate or
mark the full goal complete to change bookkeeping; continue the authorized work.


## Reactive native baseline confirmed running after cleanup

Session 2205 launched go2_reactive_nominal_maze_pilot_v1_attempt_001, launch
SHA-256 a2098de91926d68d3ab37cd7ec963f0a483001dce02baec5a159aa8f71f71520,
1,462 frozen sources. Native preflight: 82,664,718,336 bytes available RAM,
139,417,112,576 artifact bytes free, CPU 0.3% busy, idle GPUs. At the most recent
stream observation tick 86 had completed, terminal=None, request [0,0,-0.45],
observed goal distance 3.212801041640529 m. The worker automatically performs
fresh raw sensor/controller/command audits and both physical/public prefix
comparisons after collection. No completed native outcome is claimed yet.

Poll live session 2205; never restart on an observation timeout. All launched
baseline sources are frozen. On completion read/audit the result, then run
scripts/read_go2_reactive_nominal_maze_pilot_v1.py with its exact native-result
hash and record the matched-scene comparison. The prepared view-recovery native
experiment follows when this scene/audit is finished and resources pass again.
Cleanup verification is recorded in docs/go2_gsd_cache_retirement_result_2026-09-08.md.


## Goal resumed through its client; reactive result complete; recovery running

The user explicitly requested goal resumption. The OpenAI Docs skill and official
https://learn.chatgpt.com/use-cases/follow-goals documentation identify the normal
client control `/goal resume`. The tool API itself exposes no active-status
mutation; a read-only app-server proxy attempt found no control socket for this
standalone TUI. The current TUI was positively identified by its parent process
chain in tmux pane %0, with an empty input field and "Goal stalled (/goal resume)"
footer. Applied the documented `/goal resume` control to that same session.
get_goal then verified status active, the exact existing objective and preserved
accounting (6,912,970 tokens and 82,336 seconds at resume). No goal was recreated,
marked complete or edited through storage. This resolves the earlier bookkeeping
note: automatic goal continuation is now active. No subagent was spawned.

The reactive native session 2205 completed, result
1a3bf1e0f796d8fe5ae7b0b11feb837b7299064f83f6b9f44457d0040a229dd6.
It stopped at tick 141 for NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY,
151 complete commands, 152 paired observations, 8,300 physics samples, one valid
open edge [-1,0] -> [0,0], no physical/acquisition stop, no arrivals/return.
All raw sensor/controller/command and visibility gates and both physical/public
prefix comparisons passed. Readout session 61061 completed, result
4aeea872d44ec90f93d5ff29ce6e144af4b08add3394db0c94bd006c3593f2af,
launch c68bf6a2fdae96cd917c35f42418bf2cd04268a50550d6fb9727d7e50ab69edb,
1,465 sources. Native path 1.131608657 m; minimum/terminal goal distance
3.101834808 m. Final current clearance .485307661 m was clear but the measured
waypoint connector clearance .448060239 m failed .45 m, with heading inside the
turn threshold. This identifies a local-target/connector limitation, not a
physical outcome for an alternative command. Matched-scene report is
`docs/go2_reactive_nominal_maze_pilot_result_2026-09-08.md`. Both controllers
failed; differing geometry gates and shared persistent memory prevent ranking-
only or memory-attribution claims. Four native policy attempts on one maze now
exist; verified arrivals/round trips remain zero. Other maze indices unexecuted.

The prepared view-recovery native attempt is now running in session 77381:
`go2_view_reentry_maze_pilot_v1_attempt_001`, launch SHA-256
cac5f6a64a1a50913e3437133f188089f8e6f7500082ffb9273528300c277e4e,
1,453 frozen sources. Preflight: 82,679,767,040 bytes available RAM,
138,894,635,008 artifact bytes free, CPU .3% busy, idle GPUs and no competing
experiment. Full 10+1 GiB allowance above 40 GiB reserve unchanged; one CPU scene.
Latest complete decision observed was tick 239, terminal=None, request
[0,0,-.45], observed goal distance 3.347608108 m. It has not yet reached the
prospective first changed command at 595. All launched recovery sources are
frozen. Poll session 77381 without restarting on silence. The worker performs
its raw audit and physical/public prefix comparison after collection. On a
completed native result, run scripts/read_go2_view_reentry_maze_pilot_v1.py with
that exact result hash; preserve every failure and the actual recovery outcome.
The goal is active and unachieved. This resumed run starts a fresh blocked audit
if a genuine new impasse later recurs; current work is progress with a live job.

## Nearer-route reactive prefix complete; native successor admitted by preflight

The status-only preceding turn was a verified wait: session 77381 was polled
and remained live. This continuation made progress by completing the new
reactive connector prefix and preparing its native collector/audit/launcher.

Session 43325 completed `go2_reactive_connector_route_prefix_v1_attempt_001`.
Result SHA-256 99386e0bdc125cf250b70c0925d5d38b5bb01a4cdc7949b646fecc139b35d236;
launch bb5c9009a44eb94ca80dfea7bd66f70becb3739d42fe1e32a001cd3539b437c1;
decision stream d8ecfeaea370c3d3ad2206b257cb0bc597015a834a6160746a5ef257694722f1;
1,471 frozen sources, 55.329449318 s replay. All 91 public frames through tick
90 preserved the original observation/map/mission, original route and current
geometry. All earlier commands and other pre-intervention decision fields were
exact. First changed command: forward [0.2,0,0] instead of zero. The preceding
observed route target raises measured connector clearance from .446583058 to
.459045889 m, with .45 m radius unchanged and no unknown connector cells.
No changed-command outcome was consumed. Report:
`docs/go2_reactive_connector_route_prefix_result_2026-09-08.md`.

New native collector and auditor differ only by controller and collector status
identity from the completed reactive nominal baseline. Two focused AST scope
checks passed; existing eight policy tests had passed. Launcher:
`scripts/run_go2_reactive_connector_maze_pilot_v1.py`, protocol:
`docs/go2_reactive_connector_maze_pilot_v1_2026-09-08.md`.
Preflight session 67913 completed: 1,477 sources and input/source union verified,
77,002,461,184 bytes RAM available, 134,825,349,120 artifact bytes free, CPU 3.3%
busy, GPUs idle. Full 10+1 GiB allowance over 40 GiB reserve admitted. No output
or scene created. Do not launch a second native scene before session 77381 and
its audit finish. Recheck hardware at actual launch; use the exact prefix SHA
above with --prefix-result-sha256. This is a new attempt, not a restart.

Recovery session 77381 remains live. Latest complete stream checked at tick
1357, terminal=None, OUTBOUND, observed goal distance 1.5498057813314834 m,
arrivals=[]; one translating recovery at tick 595 (right_arc). No completed
result exists. These live observed values are not independently audited native
outcomes. Poll the same session, then run the already prepared
`scripts/read_go2_view_reentry_maze_pilot_v1.py` with its completed native result
hash. Keep the complete goal active: four completed failures on one maze and
zero verified arrivals/round trips remain the established native evidence.

## Reactive execution readout ready; recovery collection stopped, audit live

This continuation made progress. Added
`lewm/reactive_connector_execution_readout_development.py` and
`scripts/read_go2_reactive_connector_maze_pilot_v1.py`, with protocol
`docs/go2_reactive_connector_maze_readout_v1_2026-09-08.md`. Seven focused checks
passed in 2.08 seconds (session 55498 complete). The readout binds actual
nonterminal fallback commands and unchanged original .45 m checks to the native
tape, reads only complete 100 ms endpoints, censors partial intervals, and
preserves failed audit outcomes and comparison limitations. It is not launched.

Recovery session 77381 is still live, most recently polled successfully after
collection finished. The controller stopped at tick 1507 with
NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS, followed by drain
through tick 1517. Last observed distance 1.2742285923386731 m, no arrival.
Neither root result, worker terminal nor failure file exists at the latest
check: raw audit is pending, so do not restart or launch the next native scene.

The final selection's six nominal paths and primary surface checks all pass;
auxiliary front-left foot intersection vetoes every action. The provisional
public-only diagnostic session 44162 completed. It reconstructed the cited
latest frames 1420/1424 and exact full-frame classification counts. Their seven
cited samples pass neighboring ground mesh and validity checks but lie
10.53–10.73 mm above the retained plane, outside the unchanged 10 mm band.
Decision stream SHA-256 is
d05990c76efe733399cd3a7049ddec33db1bec65fc93299c23b8e29c9c639b7a,
checked unchanged before/after. No native pose, segmentation, model or altered
classification was used. Helper:
`scripts/view_reentry_auxiliary_floor_diagnosis_development.py`; scope and exact
findings: `docs/go2_view_reentry_auxiliary_contact_provisional_diagnosis_2026-09-08.md`.
This is a registration hypothesis, not proof of physical floor or a contact
waiver. Investigate initial plane tilt versus later pose error after binding
the completed native result; do not widen a threshold based on these samples.

Next: poll 77381; finish the exact-hash recovery readout and record its audited
native outcome. Then launch the preflighted reactive connector attempt with
prefix result 99386e0bdc125cf250b70c0925d5d38b5bb01a4cdc7949b646fecc139b35d236
after fresh resource admission. Keep all launched sources/failures intact and
the full goal active. Other maze indices and verified round trips remain absent.

## Plane-registration hypothesis narrowed; completed readout extended before launch

This continuation made progress through new public geometric evidence and a
reproducible diagnosis. Session 77381 remains live; its worker PID 2359642 was
confirmed running at about 98.5% of one CPU, with collection complete and audit
pending. Do not restart. The worker log confirms 1,517 completed commands,
1,518 paired frames, 76,600 physics samples, no physical/acquisition stop,
controller stop at 1507 plus ten zero drain commands, no arrival.

Initial primary and auxiliary measured plane fits do not resolve the late
height-band mismatch: cited samples remain roughly 11–12 mm away. Contemporary
primary planes, fitted without auxiliary samples, put the same cited points
within 13–22 micrometres. Contemporary auxiliary fits agree. This supports
later observation/plane registration drift over an initial-plane-only fix;
it is not physical floor certification or permission to ignore contacts.

Added `scripts/view_reentry_floor_plane_registration_diagnosis_development.py`.
Two focused checks passed in 1.78 s (session 36596), and consolidated public
reconstruction session 81042 completed with all six plane populations/RMS values
matching the exploratory diagnostic. The complete decision stream SHA remains
d05990c76efe733399cd3a7049ddec33db1bec65fc93299c23b8e29c9c639b7a.
Detailed fit values and scope are in the provisional auxiliary diagnosis doc.

Confirmed `scripts/read_go2_view_reentry_maze_pilot_v1.py` and its protocol are
absent from the frozen recovery native and reactive prefix/native/readout source
closures. Extended this still-unlaunched readout to include and bind the new
public-only registration diagnosis and focused test/provisional-evidence paths.
Its final native input/source verification occurs after both outcome readout and
diagnosis. No controller, mapper, band, classification or command was changed.

Next actions remain: poll 77381; exact-hash completed recovery readout; report
audited outcome; launch prepared reactive connector with fresh resources after
the current audit completes. In subsequent learned-policy work, investigate
continuous public floor-plane registration instead of merely widening the band;
native pose must remain evaluator-only. The full goal remains active.

## Recovery audit/readout complete; reactive connector native running

This continuation made progress. Recovery session 77381 completed in
2935.967778723 s, result
0f40eb01e5d5feaf004d0c0e98a9b6d712791dcd676b6013ac965fbb1603ffb8.
Readout session 71629 completed, result
7f3105a1864b21f99f24726350d7ffa1636f2ed8ef0f8ce72d849fc29c13fe76,
launch 1b2d3e5c16aae4671d2a690b985baf718ffe24c2aa142a25cdc532a7629cf133,
1,460 sources. These handles are terminal; do not poll or restart them.

The actual trajectory travelled 7.080279728 m, reached minimum goal distance
1.271093459 m and ended 1.276852922 m away. Nine valid crossings include boundary
oscillation; the loop-erased route has five edges and ends at cell [2,0]. No
arrival or return. The tick-595 translating recovery had 3.491 mm raw forecast
XY error and ordinary nominal clearance resumed at 597. All 596 pre-intervention
observations and physical/public evidence match the bound prefix; physical
prefix SHA 593d2984f886977249a90d3a7c44bf2e1003381ce7cc3aece9a4dc18f3496990.

IMPORTANT: do not summarize all audit gates as passed. The hard-measurement
failure list is empty and all auxiliary frames pass, but original strict
physical visibility is FALSE at primary frame 909. One sampled pixel [260,428]
measures 1.075383186 m against analytic 2.544169196 m. The adjacent 3x3 native
and expected grids switch between foreground/background walls; the single
pixel differs at the projected edge. This is an edge-precision hypothesis,
not an approved exception, corrected pixel or passing visibility result.
Complete findings: `docs/go2_view_reentry_maze_pilot_result_2026-09-08.md`.
Five native attempts on this one maze have completed (four learned, one reactive),
all navigation failures; latest also fails strict visibility. Zero verified
arrivals/round trips and no other layout execution remain.

Reactive connector native is now running in session 32876, root
go2_reactive_connector_maze_pilot_v1_attempt_001, launch
e9d885a0540f4179b9966cb69429cf7d9b036ceae524243129fc1b90500449ab,
1,477 frozen sources, graphical worker PID 2366487. Fresh resources admitted:
84,328,321,024 bytes RAM available, 134,184,128,512 artifact bytes free, CPU 0.3%,
GPUs idle and no competing experiment. It was launched after the recovery native
and audit finished; the bounded recovery readout finished during launcher
admission. Full 10+1 GiB allowance above 40 GiB reserve unchanged. Poll this
session without restarting; fresh raw audit and physical/public comparison to
the original reactive prefix through tick 90 run automatically. The separately
prepared `scripts/read_go2_reactive_connector_maze_pilot_v1.py` awaits the exact
completed result hash.

Added an UNINSTALLED floor-confirmation candidate and focused tests:
`lewm/current_primary_floor_plane_development.py`,
`lewm/tests/test_current_primary_floor_plane_development.py`,
`scripts/current_primary_floor_confirmation_probe_development.py`.
Three tests pass in .88 s. Public probe session 49952 completed: all seven cited
samples confirm against a plane fitted only from original accepted current
primary floor seeds; total added auxiliary patches are 13,148/14,018. No old
class or command was changed. The first extra XY-rectangle-restricted candidate
confirmed none of those seven; that uninstalled diagnostic outcome is retained
in `docs/go2_current_primary_floor_confirmation_candidate_2026-09-08.md`.
Full historical integration and first-intervention replay remain required; do
not infer the terminal constraint is solved or change the running native source.

Next: follow session 32876 and audit/read its outcome. Independently integrate
the observed-floor candidate with original raw/partition witnesses intact,
verify all non-foot/unknown/path/mission checks, and prepare prospective replay.
Investigate the separate visibility edge discrepancy before qualified successor
claims. Keep the complete goal active and preserve all failed evidence.

## Reactive connector complete; confirmed-floor causal prefix running

This continuation made progress. Session 32876 completed the reactive connector
native/audit with result
d17f634e17f049caf40a0bbcbb65e0300594c1a234cc07f32838df30ecec0d70.
Readout session 99413 completed with result
5b625ed72d1b419e489cb50f57c10ae0e3d06e6c8a3de6b88971db4d791f7dc8,
launch 9e04a34b09a1a18615846783cf0f7d1e6ef914ad1fece2cac363a3834154e43b,
1,481 sources. These handles are terminal. Native 283 commands/284 frames,
14,900 physics samples, terminal 273 NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY
plus ten zero drain commands. No physical/acquisition stop or arrival/return.
All raw and strict visibility checks pass; hard failures empty. Path 1.913348914 m,
minimum/terminal goal distance 2.808442892/2.816749751 m, two valid route edges.
77 fallback intervals executed; final current clearance .446064455 m fails .45 m.
All 91 physical/public prefix observations match; hash
97f34f60a64ed90bc02d1e6171ad2a0cad4579c85e63eb5253d801b692dd8445.
Full report: `docs/go2_reactive_connector_maze_pilot_result_2026-09-08.md`.
Six completed policy attempts (four learned, two reactive), all navigation
failures on the same layout; view-recovery also has strict visibility failure.

Integrated the new auxiliary classification in separately named sources:
`lewm/confirmed_auxiliary_floor_memory_development.py` and
`lewm/confirmed_floor_round_trip_controller_development.py`.
Original point indices, original primary/auxiliary partitions and full original
surface checks remain intact. One additional complete auxiliary partition uses
the current-primary plane confirmation; only the four nominal auxiliary foot
queries use it. Other/unknown returns still block; primary and all non-foot
checks are unchanged. No pose, raw floor grid or mission change. Six focused
memory/controller tests passed in 2.10 s (session 44102 complete), in addition
to the three prior plane/patch checks.

Launched `scripts/replay_go2_confirmed_floor_maze_prefix_v1.py`, protocol
`docs/go2_confirmed_floor_maze_prefix_v1_2026-09-08.md`.
Live session **61119**, root `go2_confirmed_floor_maze_prefix_v1_attempt_001`,
launch e58b99512f217b409bfd165bf6e6065e01e6199551c5afffc62d5daf2a58a00c,
1,470 frozen sources. Available RAM 84,112,850,944 bytes; artifact free
133,240,823,808 bytes. One ordered CPU replay, 12 GiB RAM admission and 256 MiB
headroom above 40 GiB reserve. No native scene or training is running.
Latest stdout reports frame 100 with no failure or completed prefix result.
Poll 61119 without restarting on silence. All its sources are now frozen.

The replay uses a diagnostic frozen predecessor selector with its own sequential
view state. Original complete surface receipts embedded in revised checks must
equal recorded old receipts. A private residual copy clears only the current
pending forecast to reconstruct selection-time state; complete predecessor
selection, raw map/evidence/mission/forecast/nominal paths and other earlier
decision fields must match. Stop at the first changed command/terminal before
consuming its outcome. Helper:
`scripts/confirmed_floor_prefix_selection_development.py`.

Next: follow 61119 and inspect its actual first intervention. No native successor
is prepared yet. The prior primary frame-909 visibility discrepancy remains an
unresolved strict failure: captured raster precision is 8 subpixel bits and 24
depth bits; expected/native 3x3 depths switch foreground/background around the
single failing pixel. Existing pixel-footprint diagnostics already label such
projected boundaries ambiguous but explicitly do not certify those pixels or
authorize salvaging old data. Do not silently promote that diagnostic to a
passing criterion or repair policy pixels from native geometry. Investigate
prospective sensor/evaluator precision explicitly while keeping the full goal
active and every failed outcome intact.

## Floor native successor prepared; edge diagnosis complete

The preceding status turn was a verified wait: live session 61119 advanced to
frame 900. This continuation adds tested native successor/readout sources and
completed edge-diagnosis evidence. Session 61119 is still live; latest frame
1300 reports no changed command or reconstruction failure. Never restart it
on silence. There is no new native execution yet.

Edge diagnosis session 78500 completed with result
73d6077b81430ebe78518634568a2fe4f5295966d551ca258c0c53f0a1f0b9a1,
launch bcb05b54b2e5159105a966c96875e8ec0afa39188c0892ffe77fab0ffb6f568e,
1,464 source bindings. Frame 909 pixel [260,428] lies 0.00003849127 pixels
from the foreground edge. Hypothetical nearest 1/256-pixel endpoint rounding
changes its signed side. This supports but does not prove native rounding or
certify boundary depths. Strict visibility and navigation remain failed.
Report: docs/go2_view_reentry_raster_edge_diagnosis_result_2026-09-08.md.
The prior nine-view native edge assay already supports finite-raster boundary
behavior (docs/go2_terminal_event_coverage_and_edge_assay_progress_2026-09-06.md);
do not repeat that assay as if it were missing. Policy-side uncertainty and
thin-obstacle/near-plane counterexamples remain a separate unresolved task.

Prepared scripts/run_go2_confirmed_floor_maze_pilot_v1.py, matching episode/audit
modules and scripts/confirmed_floor_native_prefix_comparison_development.py.
The only native collector/auditor changes are controller and result identities.
The prefix comparison normalizes only the added auxiliary confirmation receipt,
requires complete original surface witnesses and retains exact raw physics,
public sensor, observer, original map/partition, mission, forecast and nominal
path evidence. Fourteen focused scope/comparison tests passed (98398 complete).
Protocol: docs/go2_confirmed_floor_maze_pilot_v1_2026-09-08.md.

Prepared scripts/read_go2_confirmed_floor_maze_pilot_v1.py and
lewm/confirmed_floor_execution_readout_development.py. Six execution tests pass
(20514 complete), including actual tape/endpoint matching, partial interval
censoring and forbidden constraint changes. Readout protocol:
docs/go2_confirmed_floor_maze_readout_v1_2026-09-08.md.

Next: complete 61119, inspect and record the first changed command under its
actual result hash, then run native resource/source preflight and the exclusive
prospective attempt. If its unchanged physical prefix includes frame 909,
strict visibility is expected to fail again; this run tests the floor policy's
physical outcome without qualifying that sensor failure. No prior failed result
or frozen source was edited. Full independent-maze, matched comparison, timing
and hardware goal remains active and unachieved.

## Confirmed-floor prefix complete; fresh native attempt running

Session 61119 is terminal, exit 0. Prefix result
dfc4ed16525eb3c5474d4a436b92664f700d7bd91c9ff607571493b0adbe94bc,
1,483 observations, first changed command at frame 1482: original left_turn
[0,0,0.45] becomes left_arc [0.16,0,0.45], no terminal difference.
Decision stream 12c9e4fa949394322acc9d3b14aee4e878bfadfe93c47d1359d80b46a474e30b.
All recorded causal/original constraints and model/source/input checks pass.
Replay wall time 1554.3127937079407 s. Full report:
docs/go2_confirmed_floor_maze_prefix_result_2026-09-08.md.

The current plane at 1482 is unavailable (172 seeds, insufficient two-axis
extent), adding zero current confirmations. Earlier per-return classifications
remain in persistent memory. Those retained confirmations remove five original
FL_foot:0 candidate conflicts; the left_turn candidate was already clear.
All six nominal paths remain clear. This identifies an actual prospective
command intervention, not its outcome, a support certificate or a new arrival.

Prepared imports/input scope check 63298 completed without output creation.
Native preflight 40543 completed: 1,478 sources, memory/storage pass. After that
preflight the prefix result note added the current-plane-unavailable distinction;
actual launch freshly verifies and binds this final note.

**Live native session 53952**, root
go2_confirmed_floor_maze_pilot_v1_attempt_001, launch
f0486f8eabaf8c0ac857d5c85cd9b9c40f72cb47de815696736bbb2f5cd13f7a,
1,478 sources, graphical worker PID 2374869. Available RAM 82,313,560,064 bytes;
artifact free 133,102,104,576 bytes, CPU busy 0.8%, no competing experiment.
One CPU scene, unchanged 10+1 GiB storage allowance above 40 GiB reserve.
All launched sources are frozen. Follow this handle without restarting on
silence; the worker automatically collects, performs full fresh raw audit, and
compares actual native/public evidence through the first changed command.

Next: poll 53952. After its completed result, use the separately prepared
scripts/read_go2_confirmed_floor_maze_pilot_v1.py with the exact native result
hash, inspect navigation and floor-enabled actual motion, and preserve every
failure. Frame-909 strict visibility is expected to recur in the exact prefix
and remains disqualifying. Six prior completed maze attempts remain navigation
failures; the seventh is in progress. No verified arrival/return or independent
layout execution was added in this continuation. The full goal remains active.

## Conditional boundary model and counterexamples completed; native still live

The preceding turn was progress (completed prefix, tested native/readout sources,
and launched the prospective scene). This turn adds two completed diagnoses and
14 passing tests, while polling the same live native session 53952. Latest
persisted timing receipt is tick 875 (about 1107 ms wall time for that 100 ms
physics interval); no terminal stdout yet. Follow 53952 without restarting.
All native sources remain frozen and strict visibility gates remain unchanged.

Synthetic counterexample session 28211 completed. Root
go2_depth_boundary_counterexamples_v1_attempt_001, launch
fe7038843d217e4ec56beb6c62595e9d62714a6596d2b8be7b7c291d2b3f65a4,
result d928e84da2e947076eb40e6c770d26e3e788766511d6d20cdbf5ddd82b634189,
257 source bindings. Three tests pass. Missing 1 mm post and fabricated 1.5 m
empty-gap return both pass the old strict and stable-interior metrics: all 60
foreground rays are excluded by the 2 cm face-interior condition. Those rays
also lie outside the footprint diagnostic's compared domain, so its ambiguous
count is zero. A separate near-plane occluder correctly fails with one clipped
and one falsely public-valid ray. Report:
docs/go2_depth_boundary_counterexamples_result_2026-09-08.md.
No native recording or policy packet was used in those synthetic cases.

Implemented lewm/pixel_visible_surface_intervals_development.py: evaluator-only
front-facing box/floor projection, pairwise nearer-face region subtraction, and
separate visible depth intervals over a supplied pixel rectangle. No empty
foreground/background gap is filled. Eleven tests pass, including 392 independent
ray traces across eight rotated/overlapping scenes. A broad-occluder test first
exposed a numerical clipping sliver; explicitly solving the clipping-plane
coordinate fixed it before launch. Source now frozen by the completed diagnosis.
Floating-point bounds, zero-area ties and the supplied angular radius remain
uncertified; coincident faces may duplicate reported area.

Interval diagnosis session 62601 completed, root
go2_visible_surface_intervals_v1_attempt_001, launch
59cfbd66934617440a55a8c772d3aed82c5cf8cc373049c0ecde0d8f217381fd,
result 7131f396b608007b07d767982a7eb3f7e36e5e8d55a66bb4b3fd7d9b3ad8c632,
1,470 sources. Original frame-909 strict score reconstructs and stays false.
At [260,428], supplied radii 1/256 and 0.5 pixel both support the actual native
foreground return near 1.0754 m and reject fabricated 1.5 m empty-gap depth.
The small region rejects the synthetic missed post; the large region permits
visible background around it, demonstrating why the angular bound matters.
Both retain/reject the near-plane occluder. Report:
docs/go2_visible_surface_intervals_result_2026-09-08.md.

Source research/readout located the prior core precision result
fc2d51f3011294573247cfb1782f9c0631dca8a1daa1a8ae390af2e70db60819,
which reports llvmpipe LLVM 20.1.2, 256 bits; Mesa 25.2.8-0ubuntu0.24.04.2,
OpenGL 4.5 core. The official Mesa documentation explains software rasterization
and half-pixel centres, but does not by itself bind the exact vertex snapping
error assumed above. Do not treat the captured eight subpixel bits as that proof.

Next: follow native 53952 through collection, fresh raw audit and actual prefix
comparison, then run the prepared completed readout under the exact result hash.
Independently establish a bounded prospective angular/numerical sensor model and
policy-side uncertainty before adopting the new evaluator for qualification;
the original strict failure cannot be retrospectively cleared. No new arrival,
return, independent-layout result, real-time or hardware claim was established.

## Actual mesh coverage and upstream raster source checked

Previous turn was progress (two completed diagnoses and tested interval helper).
This turn adds a completed actual-mesh diagnosis and seven tests. Native session
53952 remains live; latest checked timing receipt is tick 1282, about 1379 ms
for its 100 ms physics interval. No terminal result has been reported. Follow
the same handle and do not restart on observation silence. Its sources remain
frozen; the full fresh raw audit and prefix comparison still await collection.

Upstream Mesa source download 27977 completed. Official 25.2.8 archive is
43,813,260 bytes, SHA-256
097842f3e49d996868b38688db87b006f7d4541e93ce86d2f341d8b3e7be7c93,
matching the official release notes. Only lp_setup_tri.c, lp_rast.h and
lp_setup.h were extracted under /tmp/lewm_mesa25_precision_mw3x54ji; no build,
installation or runtime change. Source defines eight fractional bits and rounds
scaled triangle positions to integers, with implementation-dependent ties.
Installed packages are 25.2.8-0ubuntu0.24.04.2; exact binary/source equivalence
has not been proved. Review and individual file hashes:
docs/go2_llvmpipe_subpixel_source_review_2026-09-08.md.

Implemented lewm/mesh_subpixel_coverage_diagnosis_development.py and seven tests
(all passed in 0.14 s). Completed probe session 10218, root
go2_mesh_subpixel_coverage_v1_attempt_001, launch
4bda1e1e91c6347408b03a2b44816a82a899ea60a22155024d19a9d7e5c1a2c0,
result 8f0eaccafbdde8406665fd7758ee4ccbd0ab22ea1fe78112f076bf778e49788a,
1,469 sources. Actual PLY float32 positions match the raster readback hash
5977284f74911d62a9ac1560d38a6e0e56fe49c82fb94abd52a7663a0b768fe8.
28,600 vertices/14,300 triangles; 6,901 triangles tested without clipping,
7,399 excluded explicitly. At frame 909, exact projection selects nearest plane
triangle 5774 at 2.5441692391 m; hypothetical 1/256-pixel snapping adds foreground
triangle 10277 at extrapolated plane depth 1.0753915405 m. Actual native depth
1.0753831863 m differs by 8.354 micrometres. Original strict score reconstructs
and remains false. Full report:
docs/go2_mesh_subpixel_coverage_result_2026-09-08.md.

This strengthens the mechanism diagnosis using actual subdivided mesh inputs.
It still does not reproduce shader transforms, clipping/culling, native rounding
mode or framebuffer interpolation, and gives no complete angular/numerical
bound. Do not promote it to a passing gate or modify running sources.

Next remains native 53952, followed by prepared completed native readout under
its exact result hash. The independent sensor work should now address the
remaining transform/numerical bounds and policy-side representation, rather than
repeat the established edge-snap examples. Six prior completed native attempts
remain failures; the seventh is running. Zero new arrival, return, independent
layout, timing qualification or hardware result was added. Goal stays active.

## Confirmed-floor collection failed; fresh audit still running

Previous turn was progress (upstream review and completed mesh diagnosis).
Native session 53952 remains live in its raw-audit/final-comparison phase.
Collection has completed: 1547 complete commands, 1548 paired observations,
78100 physics samples, terminal 1537 NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS,
then ten zero commands. No physical/acquisition stop or observed arrival.
Observed goal distance 1.1449574973376988 m. This is provisional collection
evidence; no top-level completed result or audited native metric exists yet.
Do not restart the worker or launch a successor before its completed audit.

Read-only inspection 97938 verified collection result
8241e3753fe952701bee956632257eeca95fd40b4b0b50126ec18325c665829a and
decision stream cfbd2ef3396eff6a3c548a32a55dd4b6b52ee9b49eed000b26cfe06538c352a4,
before/after inspection. All final nominal paths and primary checks pass;
all six candidates fail auxiliary FL_foot:0 bounds. New cited cells are
[152,-10,-10], [152,-11,-10], [151,-10,-10], latest frames 1469/1474/1466.
Current plane has zero original primary seeds; original and added current
auxiliary floor counts are both zero. The first native command intervention
at 1482 is the prescribed left arc, but actual prefix proof awaits final audit.

Plane inspection 71161 completed on the same fixed collection/stream identities.
At 1537, 57442 primary and 254370 auxiliary measured ground-quad points fit
mutually consistent planes with RMS 10.186/4.926 micrometres. Candidate heights
are 10.628–16.185 mm and 10.402–16.067 mm above the retained initial floor,
respectively: every current candidate lies outside the 10 mm original seed band.
Earlier cited frames also have coherent planes. This exposes the remaining
initial-height seed dependency, not an absence of measured floor geometry.
Provisional report with limits:
docs/go2_confirmed_floor_terminal_provisional_diagnosis_2026-09-08.md.

Added an UNINSTALLED public point-box helper,
lewm/pixel_support_point_bounds_development.py, with eight tests passing in
0.16 s. Exact rational corner checks cover both public camera mounts, outward
float64 interval arithmetic, public-range boundary preservation, monotonicity
and invalid inputs. This does not calibrate the supplied pixel/depth bounds,
include observer pose error, or change any controller/index. Candidate scope:
docs/go2_pixel_support_point_bounds_candidate_2026-09-08.md.

Next: follow native 53952 to completed result, then the prepared readout using
its exact hash. Diagnose the now-observed floor/pose registration dependency
and prepare a separately named observation-grounded correction with explicit
pose-consumer/witness consistency if it changes poses. Do not merely overwrite
joint-pose witnesses or waive unknown contacts. Sensor angular/numerical bounds
and policy-side uncertainty also remain incomplete. Full goal stays active;
no verified arrival, return, independent layout or hardware evidence was added.


## Seventh native result completed; floor-registered controller prepared

Previous goal turn was a verified wait on live native session 53952. This turn
made progress: completed its audit/readout and implemented a consistent new
pose-estimator candidate. Session 53952 is now CLOSED, total 3763.926885 s.
Native result 5ee4ef051e1a506f205aae51610deece18f755fb5440c40b1113e22e5ba317ee;
audit 810cd40922c821c3a0ba523048dfa442ef8ba4b0070b7e6bb8ff1a1f57dab283.
All fresh sensor/model/command replay passes; strict primary frame909 still
fails; hard measurement failures empty; zero arrivals/returns. Native path
7.256963716639283 m, min goal distance 1.1474975121459456 m, terminal
1.1482243239084386 m. Actual intervention prefix 1483 frames through1482 exact,
physics SHA42f715936bc7144adbd5525cce80676ebe332a80795903eaef858fd79e459973.
Seven native maze0 attempts have completed, all failures. Other mazes unexecuted.

Readout 89285 CLOSED, result
fd632bdca173ec2ea82be2b2d11f364c90746f01dfabd0d8d93c2be92a449bc8,
launch d240b42c6e50edf412bd6b7f42d9c6c5f1b37703ce4aac787fa88740965a1b72,
1482 sources. Forty actually executed intervals used newly allowed auxiliary
floor-contact candidates, zero censored. First1482 actual100msXY
[0.004806358067816313,0.0035872758027524453] m, forecast error .0033289164630069257 m.
Receipt-inclusive median1137.0062295 ms/100ms paused physics. Full report:
docs/go2_confirmed_floor_maze_pilot_result_2026-09-08.md.

New candidate lewm/floor_pose_registration_development.py fits two contemporary
measured floor planes without initial absolute-height seeds or trimmed points.
lewm/floor_registered_evidence_development.py retains original visual evidence
and admits a separate registered pose. lewm/floor_registered_controller_development.py
uses that pose consistently in surface/map/contact, residual and mission
consumers. Original tracker/model/raw witnesses remain intact. Static shared
floor identity and uncertainty remain uncalibrated; this is no floor/support
certificate. Initial reference frozen, all failures latch. See
 docs/go2_floor_registered_controller_candidate_2026-09-08.md.
11 geometry +11 controller/evidence tests passed (combined22 in4.55s), six
prefix tests2.06s, one native scope test.11s, ten native-prefix tests1.98s.
Source-only preflight49312 CLOSED:1489 sources,11new, all verified, no artifact,
model or native scene created by that preflight.

Read-only public-plane inspection90294 CLOSED; same collection/stream hashes
checked before/after, all nine frames0,595,909,1420,1466,1469,1474,1482,1537 admit
compatible planes. Terminal primary/aux3611/15940 samples, max residual
9.210/11.007 micrometres. Proposed terminal correction -12.6429027882 mm along
initial floor normal and .00326729291987 radians tilt. No candidate trajectory
or contact outcome was inferred from that sparse inspection.

LIVE prefix session43443, rootgo2_floor_registered_maze_prefix_v1_attempt_001,
launch bfcf3260e39a15ec7d94858b492099e85fde2349fbb11eeddc2aafd348a5b92e.
It uses completed seventh native/readout hashes, two fresh controllers in one
CPU process, complete predecessor equality, original visual-witness equality,
and same raw forecasts where both plan. Stops at first changed command/terminal
before an unexecuted outcome. Its source bindings are frozen now; do not edit
candidate/evidence/geometry/prefix sources. Follow handle, never restart on silence.
Separately prepared (not launched) native episode/audit/launcher/prefix-check
and protocol forgo2_floor_registered_maze_pilot_v1_attempt_001. Native launch
still requires completed prefix result, result review, and hardware preflight.
No independent layout, arrival, return, real-time or hardware evidence added.

Prefix43443 latest confirmed progress: frame100 (101 complete comparisons),
handle still live. Launch binds1491 sources, availableRAM82,139,643,904 bytes,
artifactfree128,188,612,608 bytes. No intervention/result/failure reported yet.
All other handles from this turn are closed. Next continue43443; on completed
nonterminal intervention write the result review, run prepared native preflight,
then the fresh single-scene pilot. If prefix admission fails, preserve it and
diagnose that actual failure before any successor. Full objective stays active.

## Common-plane successor running — latest update 2026-09-09

The preceding goal turn made progress. This turn completed the failed prefix
diagnosis, implemented a different combined-camera estimator, tested it, and
started its fresh prospective replay. Full goal remains active and unachieved.

Session43443 is CLOSED, exit1: candidate registration failed at frame329 because
the primary camera's second covariance eigenvalue .0020401083831666457 m² was
below the independent-plane threshold .0025 m². It still had1179 points and
4.4866 micrometre maximum residual. Auxiliary had13303 points, eigenvalue
.022449931065491948 m² and maximum5.4817 micrometres. Diagnostic98438 CLOSED,
public frames0,327,328,329,330,331 only; later frames are descriptive predecessor
geometry, never an unexecuted candidate trajectory.
Failure562acbc1463ef16864be4e6cf2128e47289d2441258585b4e521799b6fe53c81;
savedstream542e85af02532711c6777ffb8ef7fd0f20360274459de35d793cc07b6f006889.
No successful prefix result exists, and the prepared independent-plane native
pilot was NOT launched. Its frozen sources/failure are retained. Report:
docs/go2_floor_registered_prefix_failure_2026-09-08.md.

The separate common-plane candidate fits all current measured camera candidates
in one body frame, checks combined two-axis extent with the unchanged .0025 m²
gate, and retains the unchanged3mm maximum residual gate for EVERY contributing
point in either camera. Narrow or empty candidate sets add no invented geometry;
they need not independently identify a plane. Both current public packets remain
required. Sparse conflicting surfaces and combined rank deficiency still reject.
No points are trimmed. This changes the measurement model prospectively rather
than revising the failed independent-camera admission result.
Sources: lewm/joint_measured_floor_plane_development.py,
lewm/joint_floor_registered_evidence_development.py,
lewm/joint_floor_registered_controller_development.py.
Original visual evidence/observer, correction limits, model, action set,
contact policy and consistent pose consumers remain intact. New schema validates
combined sufficient statistics, per-camera residual accounting and raw witness.
Static shared floor identity/uncertainty remain uncalibrated.

22 combined-plane/controller tests passed4.05s (97645 CLOSED), five prefix tests
passed1.88s (62454 CLOSED). Public inspection29747 CLOSED: frames0,329,330,331
all admit combined fits. At329 total14482 samples, second eigenvalue
.024598045301285162 m², primary/auxmax5.3424/5.4165 micrometres. No trajectory
or command outcome was inferred. Candidate/protocol documents are separately
named go2_joint_floor_registered_*_2026-09-08.md and are now frozen by replay.

ONLY LIVE handle: session66629, PID2386768,
rootgo2_joint_floor_registered_maze_prefix_v1_attempt_001,
launch e42805bdb7edc73939aeef89b00978c60a0d36b5a06e947204ef19e71d24fa02,
1501 sources. Latest verified progress frame400 (401 comparisons); no command
intervention or admission failure yet. It passed the previous329 failure point.
One fresh candidate reuses the completed, hash-verified predecessor full raw
audit rather than rerunning that already completed controller audit. Original
visual witness and raw forecasts are compared on every applicable frame; stop
before the first changed command's unexecuted outcome. Failed predecessor prefix
launch/failure/stream are also explicitly bound and verified. Do not restart on
silence or modify launched source bindings. Launch resources:16physical/32logical
CPUs, affinity0..31, CPU.5%, availableRAM81,851,617,280 bytes, artifactfree
128,146,870,272 bytes, workspacefree21,362,622,464 bytes. DiscreteGPU8%,
1,398,738,944/34,208,743,424 bytes VRAM. No competing experiment.

Prepared, NOT LAUNCHED: scripts/run_go2_joint_floor_registered_maze_pilot_v1.py,
joint_floor_registered_maze_episode_development.py and audit, plus protocol.
Reuse the tested generic floor_registered_native_prefix_comparison helper for
exact physics/public and complete prospective-candidate decisions. One native
scope test passed; helpers' ten negative/positive prefix tests remain applicable.
Also prepared scripts/read_go2_joint_floor_registered_maze_pilot_v1.py and
protocol. The unbound pose-readout helper now dispatches only the two reviewed
registration schemas;18 paired-schema tests plus native scope passed5.82s
(90695 CLOSED). Its earlier nine-test independent-schema run93294 also CLOSED.
Prepared source/import closure99245 CLOSED:1512 sources,11new verified, no
runtime admission or artifact creation. This is not a native preflight pass.

Next: follow66629. If it yields a completed nonterminal first-command change,
write the result review at the launcher's explicit expected path, run the new
native --preflight-only under that result hash, then execute the single fresh
scene. Preserve any failure and diagnose before further changes. Seven native
maze0 attempts remain failures, zero arrivals/returns, mazes1–3 unexecuted,
no real-time/hardware qualification. Existing strict visibility failure remains.

Latest same-handle poll66629 confirmed frame600 (601 comparisons), still live,
no different command or terminal reported. This verified wait is not a stopped
job. Preserve the same handle and follow to completion before native preflight.

## Common-plane native launched — 2026-09-09

The common-plane prefix is COMPLETE, not still running. Result SHA-256
`c1a2c347d5aea7a1e0db352a451daf4add20c9c42905886712778cbdd53166ea`;
960 frames through first changed command959: hold `[0,0,0]` instead of left
turn `[0,0,.45]`. No terminal difference, original visual evidence exact,
same raw predictions where both plan, weights unchanged. No changed-command
outcome was inferred. Review:
`docs/go2_joint_floor_registered_maze_prefix_result_2026-09-08.md`.

Native preflight89727 CLOSED exit0:1509 bound sources, RAM81,804,304,384 bytes,
artifactfree128,061,353,984 bytes, both admissions passed, no competing scene.
Native session90647 is LIVE; parent2389194, worker2389334. Exclusive root
`go2_joint_floor_registered_maze_pilot_v1_attempt_001`, launch SHA-256
`7478a094256d99aa3c25958806707730efb07eb3277dd83e411437b7d9ee98a5`.
One CPU scene, unchanged numerical settings and source bindings frozen.
Latest actual timing row221, no completed result, failure or worker terminal.
Launch RAM81,808,650,240 bytes, artifactfree128,060,035,072, workspacefree
21,362,540,544; CPU.6%,16physical/32logical, affinity0..31, discreteGPU8%,
1,398,738,944/34,208,743,424 VRAM bytes. Monitor the existing handle; do not
restart on silence. Its immutable prefix already contains strict visibility
failure909; no visibility fix or qualification is claimed.

A separate small timing check completed (36681 CLOSED) using saved visual fits
at100,600,1500, matching each recorded rotation, position, inlier count and RMS
exactly. Plain registration11–13ms; two feature frames17–35ms, matching4–11ms.
Temporary report `/tmp/go2_visual_fit_timing_20260909_result.json`, SHA-256
`8bc73214ea0e3ed24d299db79493a75f951a8b2cadaa5288e86166b55b48f121`.
This is component timing only. Previous temporary diagnostic30067 is closed;
its output was unavailable, so no finding depends on that lost output.
Completed seventh trajectory's1548 saved timing rows show median acquisition
213.7222075ms and controller884.892166ms. Full-loop profiling is warranted.

Bounded full-controller timing diagnostic49066 is LIVE, separate immutable
predecessor replay, no native scene or policy modification. Protocol/source:
`docs/go2_controller_timing_diagnosis_v1_2026-09-09.md` and
`scripts/diagnose_go2_controller_timing_v1.py`. Root
`go2_controller_timing_diagnosis_v1_attempt_001`, launch SHA-256
`7d4867e6a916179031a4ff004264f72167d07e00c684c323f0f36e77f54580ea`,
1480 sources. First101 observations, profile only20/60/100; every complete
decision must equal its saved predecessor. Latest completed profiled frame60;
profiling overhead is not a plain runtime estimate. Hardware before: RAM
79,001,513,984 bytes, artifactfree127,598,493,696, CPU3.6%; one diagnostic
process beside the existing single native scene. No current scientific source
is modified. Next review the completed profile, follow90647 through actual
collection/audit/prefix comparison, then run its prepared completed readout.
Seven completed native failures and zero verified arrivals/returns still stand.

Timing diagnostic49066 is now CLOSED exit0. Completed result
`b7dfb633bbaa585d2346c87fee9889be2787d8a97cfa01b67c638699bbafafab`.
All101 complete decisions exact; weights unchanged and bindings reverified.
95 non-warmup/unprofiled calls median798.479455ms. Three profiled calls show
deepcopy0.456–0.465s cumulative and nine floor-index computations0.291–0.328s
cumulative each step (instrumented, nested costs; not additive or a speedup).
Report `docs/go2_controller_timing_diagnosis_result_2026-09-09.md` records
scope and next concrete performance targets. No optimization is installed.
Only native90647 remains an active experiment handle.

Latest verified native90647 poll: timing row335 completed, no result/failure/
worker terminal. Resource sample384.389s: CPU3.6%, RAM79,382,872,064 bytes,
artifactfree127,021,735,936 bytes. Continue this handle; collection is live.

## Eighth collection stopped; cache equivalence and memory diagnosis complete

The preceding goal turn made progress. This turn implemented and verified a
separate per-observation floor cache, benchmarked a separate receipt copier,
and diagnosed the eighth collection's actual terminal constraints. Full goal
remains active and unachieved. No blocker is asserted.

Native90647 remains LIVE, worker2389334. Collection is now closed:1067commands,
1068paired frames,54100physics samples. Controller terminal1057 then10zero drain,
NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS. No physical/
acquisition stop or observed arrival/return; observed goal distance2.887777272m.
Raw audit/prefix comparison are pending: no top-level result/failure/worker
terminal/audit JSON yet at last check. Do not restart. Collection SHA
`a07f53205344d5e6fc58f754d3180268eacae8e2e8ea684e5850837124caf261`,
decision stream `e92d311fed32b3886f26eefc4f14c999cd5b37a7bfde6c6b54532cd0524fa64a`.
Provisional report:
`docs/go2_joint_floor_registered_terminal_provisional_diagnosis_2026-09-09.md`.
Current floor planes remain available; the remaining conflicts include old
ambiguous nominal-foot returns. Right arc alone passes surface checks but fails
the final700–800ms nominal segment. Original constraints stay intact.

Floor input diagnosis82830 CLOSED: result
`49de94b98890885d0c47bf78313d2383e62c0d3fc2583795bd05cc1a81291c60`,
101exact decisions, nine calls/two exact input groups at20/60/100.
Frame-cache source/tests and component-copy details:
`docs/go2_frame_cache_and_copy_progress_2026-09-09.md`.
24cache tests passed5.92s; nine receipt-copy tests passed.14s. Original current
native source bindings remain unchanged. Both candidate source sets are frozen
by their own separate launches; do not edit them.

Cache replay88913 CLOSED: all960 full candidate decisions exact, weights
unchanged, result `633a16730506480011d9a00a4a76c63daa9c62c19eaf39695a3376abfc97e14b`.
Launch `6e16b5963432fc4f0ed4ab131c1e8d21d0e5ae8e93f3557a9cb66b44fe61db6a`,
1508sources,762.990808s; median non-warmup controller734.115640ms (unpaired).
All960 had2misses/0fallback;946 had7hits,13 had6hits, initial8hits.
Report `docs/go2_frame_cached_floor_prefix_result_2026-09-09.md`.

Copy benchmark18249 CLOSED: result
`fc17e49bb2ded6236ff8adb95051284f5d99d849a68793fe5f678237703001c5`,
launch `9e49a80b95bce98709281e4a6beff29386fff2373d0784848fce910742f7bebc`,
1505sources. Paired eight-repeat saved-receipt copying1.73–1.95x component
ratio; no copier is installed in a controller and no full-loop speedup claimed.

Retained-floor diagnosis56870 CLOSED:915paired observations examined, four of
six terminal first-hit enclosures gain a strictly later complete floor-grid
coverage witness; two remain unresolved. Result
`3d9a5d3109eff7fb9f8e38acf347ac13a631ef0e643679883eeed3db468b9912`,
launch `79bcbbcc2b060f35c363c3dc39342b0254c5bc483c02819b84f54d1a54746da7`,
1511sources/6441collection artifacts verified before/after. Primary[99,40,-10]
covered by auxiliary733; primary[87,33,-11] and auxiliary[86,33,-11]/[87,33,-11]
covered by auxiliary850. Primary[86,34,-11] and auxiliary[85,34,-11] unresolved.
See `docs/go2_retained_floor_resolution_diagnosis_result_2026-09-09.md` for
the next concrete source implementation: preserve original indices/checks,
resolve only every explicitly bounded ambiguous foot hit using later single-view
complete coverage and original floor-height band, retain unresolved/non-foot
hits, invalidate on newer contributing samples. No such policy exists yet.

Next: follow native90647 through final raw audit/physical-prefix comparison,
then run the prepared readout under the actual result hash and record native
trajectory/registered-pose errors. In parallel, implement/test the separately
named later-evidence contact-resolution candidate above. It must retain the
complete original action/model/nominal horizon and pass a prospective prefix
before its own native attempt. Seven fully completed native failures plus this
provisional eighth failure; zero arrivals/returns, independent mazes1–3 still
unexecuted. All other handles in this turn are closed.

## Status reconciliation: eighth audit complete and successor tests pass

The eighth native attempt and readout are now complete, superseding the pending
audit status above. See `go2_joint_floor_registered_maze_pilot_result_2026-09-09.md`
for the bound results. Raw sensor/model/command audits and the 960-frame
prospective prefix comparison passed; strict physical visibility remains failed.
Native path length was 4.669673281m and terminal goal distance 2.886041376m.
There was no arrival or return. Eight completed native maze-0 attempts have now
failed. Pose registration improved measured pose accuracy on this trajectory;
it did not establish a navigation advantage. Median receipt-inclusive loop time
was 1163.238353ms for each 100ms paused-physics step.

The separately named later-floor-evidence memory/controller successor and its
prospective prefix runner now exist. They retain original contact records and
resolve ambiguous nominal-foot hits only with strictly later complete floor
coverage from one measured view; unresolved and non-foot contacts stay blocking.
Focused verification in session56651 completed: 25 tests passed in 3.77s across
`test_later_floor_evidence_development.py` and
`test_later_floor_resolution_prefix_development.py`.
The new prefix has not been launched. No matching native, readout, prefix, or
pytest process was running at this status check after that verification closed.

Next: complete source/resource preflight and run the later-floor-resolution
prospective prefix to its first command change. Review the result before a
separately named native attempt. Independent mazes1–3, verified arrival and
physical return, matched baseline/ablation evidence, strict sensor qualification,
real-time execution, and bounded hardware validation remain outstanding.
The full navigation goal remains active and unachieved.

## Later-floor-resolution prospective replay launched

The next goal turn moved from status verification to execution. Preflight56855
closed successfully:1531 sources,81,964,556,288 available RAM bytes,
124,715,778,048 artifact-free bytes,16physical/32logical CPUs,0.4% aggregate
CPU activity. One fresh candidate process, one numerical thread, no scene.

Replay session58083, process2395945, is live at the last check, through frame400
with no reported invariant failure or command difference. Exclusive root
go2_later_floor_resolution_maze_prefix_v1_attempt_001; launch SHA-256
1a33d6a74ed8813c2cccb206e40d1e502b157c267c5b66bfb88766b6b36765c9.
Its sources are now frozen. Do not restart this attempt or change bound files.

Separately named later-floor-resolution native collector/audit, prefix comparator,
launcher, protocol, execution readout and tests are prepared but unlaunched.
The combined native-scope/prefix/readout tests passed24 in2.22s (session9567).
The native wrapper's review caught and corrected a local artifact-binding
variable shadow before launch. The successor readout reports actually executed
later-floor-cleared commands, preserving censored intervals and rejecting
future/same-frame witnesses, missing hits, changed non-foot queries and wrong
tape requests. Native launch still requires completed prospective evidence,
its review and a fresh resource/source preflight. Eight native failures and
zero arrivals/returns remain the completed evidence.

## Later-floor prefix complete; ninth native attempt launched

Replay58083 is CLOSED exit0. Result
a4e34e4ca8b72422f8fc6c1ca54b33a80c9821ff076fbc431d0575f0a1701fbe,
835.671191s,960 frames. First command difference959: left_turn[0,0,0.45]
instead of hold. No terminal difference. First hypothetical contact resolution946.
At959 a primary FL_foot:0 enclosure[94,48,-10], latest sample343, is cleared by
complete primary floor coverage at344 of map square[46,24]. Exactly one hit is
resolved; selected contact and full nominal horizon checks pass. Original
pose/map/mission, raw forecasts and original contact queries remain exact;
weights unchanged. No unexecuted successor outcome was consumed. Review:
`go2_later_floor_resolution_maze_prefix_result_2026-09-09.md`.

Initial native preflight49584 failed before output creation because the new
wrapper expected the wrong prefix status string. The unlaunched wrapper was
corrected; frozen replay/controller sources were untouched. Preflight73946 then
passed:1539 sources,81,820,778,496 RAM bytes and124,568,416,256 artifact-free bytes,
1.7% aggregate CPU, no competing experiment. Required32GiB RAM and51GiB total
artifact reserve/allowances passed.

Ninth native attempt session59231 is now LIVE, parent2398198, worker2398270.
Exclusive root go2_later_floor_resolution_maze_pilot_v1_attempt_001;
launch SHA-256 c3d035abcc69b3b42ecb160021203e7d6d685a176e860e5c3044afa2035cefa4,
1539 bound sources. One CPU scene, one numerical thread. Its source is frozen;
do not edit, restart or overwrite it. Follow this same handle through collection,
full raw audit and physical/public/prospective prefix comparison. Known strict
visibility failure909 lies before intervention959 and remains part of the audit.

Prepared but unlaunched completed-native readout:
`scripts/read_go2_later_floor_resolution_maze_pilot_v1.py`, protocol
`go2_later_floor_resolution_maze_readout_v1_2026-09-09.md`. It requires the actual
completed native result hash. Collector/comparator/readout tests passed24 in2.22s.
No ninth native outcome is available yet. Eight completed failures, zero verified
arrivals/returns and the full independent-layout/baseline/timing/hardware work
remain. The goal is active and unachieved; no blocker is asserted.

## Ninth collection continues; renderer provenance corrected

The next goal turn verified native59231 still live and completed two separate
bounded diagnoses while it collected. Latest provisional saved-decision read
(session44586 closed) reached tick907: OUTBOUND, observed goal distance
3.223391216m, right_turn, no failure/terminal. This is live observer evidence,
not a completed native trajectory or arrival audit. Do not restart native59231.

Renderer identity helper added separately, not installed in frozen collectors:
`lewm_genesis/lewm_genesis/camera_renderer_identity_development.py`.
Seven tests passed0.16s, covering actual-target requirement, bounded queries
and context release without framebuffer mutation. Its source is now frozen by
the provenance diagnostic.

Provenance47326 CLOSED: result
76a592ec2bfa31927acb07cfdc6bb056de78df1693ab4066477f20a15159150a,
launch bcef998cccc5acccf0fe0c7fe4537fe3b4aee6210b6438719f35826de1048cac,
1544 sources. Worker2398270 progressed579–582 timing rows across3.1772595s;
one deduplicated amdgpu client at0000:7b:00.0 performed52,437,270ns of additional
graphics work. A separately queried zero-scene EGL context identified AMD
radeonsi on that same PCI device, OpenGL4.6 Mesa25.2.8. The earlier empty
context had reported llvmpipe/OpenGL4.5. Neither is a direct query of the live
worker's camera context; do not infer historical renderer identities. Report:
`go2_live_maze_renderer_provenance_result_2026-09-09.md`. No live worker change,
new scene, physics step, draw call or image. Installed runtime library hashes
and EGL selection source were recorded and verified.

Radeonsi source review found viewport-adaptive8/10/12-bit quantization despite
reported8-bit subpixel capabilities. Fixed hypothesis diagnosis6675 CLOSED:
result a3b158cfc9312a0c5c02b8d5f218561dcc03cd31597bc7cafe1748be4b6a7522,
launch6065242364de203e51b54f7318a65c59e4f59815092fd4ed006636794b6c39aa,
1555 sources. All three modes retained.8- and12-bit rounding introduce the
same foreground triangle whose extrapolated plane is8.354um from native;
10-bit retains the background triangle1.468786m away.8-bit predecessor report
exact. No chosen mode, changed pixel or revised strict score. Report:
`go2_radeonsi_mesh_quantization_result_2026-09-09.md`.

This corrects the provenance assumptions for further sensor diagnosis. Actual
camera-context/render-state capture, shader/clipping/interpolation bounds and
consistent policy-side point/pose uncertainty remain unresolved. The ninth
native result/audit/readout remains the immediate navigation task. No new
navigation outcome or qualification is claimed; full goal stays active.

Later live check: native59231 is still running, parent2398198 and worker2398270
confirmed by process state. Worker elapsed23m01s,98.3% one-core CPU, RSS6,047,268KiB.
Bounded provisional stream read56722 CLOSED: at tick1178, OUTBOUND, observed
goal distance2.237466916m, right_turn[0,0,-0.45], no failure or terminal.
Collection has continued beyond the eighth terminal1057. This is observer/live
execution evidence only; final native sensor/command/trajectory audits and the
960-frame physical/public/prospective comparison remain pending. Keep the same
native handle. Both renderer diagnoses and temporary stream readouts are closed.
The prepared completed-native readout still requires the actual terminal result
hash; no ninth arrival or round trip has been verified.

## Ninth collection ended; transition diagnoses and full phase timing complete

Current goal turn made progress: it established the arrival-window braking
cause, reproduced all terminal correspondence losses, tested a fixed subpixel
candidate and preserved its negative result. No blocker or completion claim.

Native59231 remains LIVE in full audit/comparison, with parent2398198 and
worker2398270 confirmed by process state. Collection is complete:1880commands,
1881paired observations,94750physics samples,10terminal zero ticks. First
observed arrival1866 switches RETURN; visual failure1870 ends collection.
The ground-truth arrival window stays within3.761944cm but reaches15.247449cm/s
versus the unchanged5cm/s speed limit. Zero verified arrivals/round trips.
The peak is at the beginning of braking; current speed at1866 is0.209434cm/s.
All31 late one-second windows fail at least one of distance/speed/zero-request.

Arrival dynamics92614 CLOSED: result
c0cfd9ab1d40c7497735f499dfb8d9ea5e8717f028c165e20882ce835ed88227.
Match diagnosis34677 CLOSED: result
29565cd5f2d87ba6c9d657780fc9ca75f908ab7cd28d0713f4da28d2b758e081.
Frame1870 has22 actual selected corners; previous-frame12 mutual/LK-consistent
matches become8 after the unchanged one-pixel location gate. Depth removes0.
All8retained refs also fall below12. Original correspondence arrays exact.

Separate subpixel candidate tests4passed0.17s. Fixed pair comparison22220
CLOSED: resulte1989bb27999ff04100700a121c6dea3bef81ed22a041afc05474e00eb38ca23.
It still fails all9terminal pairs, including10instead of8incremental matches.
Candidate not adopted. All18frames/25pairs and both implementations' failures
are retained; sources are frozen. Full evidence and exact artifact identities:
`go2_ninth_maze_transition_diagnosis_result_2026-09-09.md`.

Phase timing8699 CLOSED: result
434397c65391d235d6708a3e24675118463aea6ed89e9061721e71e643a0ce0a.
All960 complete decisions exact, unchanged model, removed hooks,892.887630s.
Median controller839.349784ms; model7.316164ms. Geometry/map/selection dominate.
No real-time or controlled speedup claim. Report:
`go2_controller_phase_timing_result_2026-09-09.md`.

Immediate next actions: finish the same native59231 audit and prefix comparison;
run the prepared readout only after the actual completed native result exists;
implement measured settling and address terminal feature availability using
separately named successors. Do not restart native59231, alter failed/frozen
sources, relax original qualification, or adopt the failed subpixel candidate.
Independent-layout runs, baseline/ablation comparisons, physical backtracking,
strict visibility/uncertainty, real-time operation and hardware evidence remain.

## Ninth native audit complete; measured-settling controller prefix running

This goal turn made progress through implementation, focused tests, a completed
saved-observation comparison and completion of the ninth native audit. No
blocker is asserted; the full goal remains active and unachieved.

Native59231 CLOSED exit0. Result
3745c0b7c45265a2fcc38caf732b6f3f4b228487f344d23502dc7395077d5755,
wall4510.6477477950975s. Audit
26dad9cb08152eed49d3e518a5653af264b5dbe51091a64aca2031e30b4bb9af:
raw sensor reconstruction, full model/controller replay, command audit and
model-state checks all pass. Prefix comparison
444ad02042b58ea96a6029243d335c78ab218b7b81c57297ef6fc98dee833dc0:
960 complete physical/public/candidate decisions exact through intervention959.
Strict primary visibility still fails only909; all auxiliary visibility checks
pass and hard-measurement-failure list is empty. Arrival1866 fails the same
native1s speed requirement; no physical return. Mean/max registered observed
XY error4.235033/8.577553mm. Nine completed native outcomes, zero verified
arrivals/round trips. Do not restart the closed native handle.

Prepared ninth execution readout16797 is now LIVE, invoked with the actual
completed native result hash above. Root
go2_later_floor_resolution_maze_readout_v1_attempt_001. Follow to completion;
its full readout result and pose/trajectory/contact summaries are still pending.

Measured-settling mission/controller11 tests passed3.70s. Original saved-mission
adapter32534 failed before frame0 because JSON identities needed the existing
explicit list-to-tuple conversion; original failure/source retained. New JSON
adapter6 tests passed2.50s. Comparison84035 CLOSED:result
bf875306754c620d393d7a05496081c318c17d624726d2ae8bf348bab9caa053.
All1867 original mission receipts and registered-pose witness checks exact;
first counter difference1857 rejects observed10.811438cm/s braking. At1866
the new mission has9 quiet intervals and remains OUTBOUND, while the original
switches RETURN. It stopped at that first behavior difference.

A separately named boundary successor requires a measured quiet start before
counting ten complete subsequent quiet intervals.6 tests passed1.83s; complete
decision comparison7 tests passed0.10s. It retains original tracking and all
later-floor/model/planner/memory behavior; failed subpixel candidate excluded.
Full raw-controller prefix56186 is LIVE:
go2_settled_boundary_controller_prefix_v1_attempt_001,
launchcf320b97b50fd5e9d6369608d690147cdb1cce9bb15a19d54cfb89bc126f9f1e,
1555 frozen sources. Preflight45491 passed; actual launch RAM81,271,074,816
and artifact-free118,629,277,696 bytes. One CPU replay, one numerical thread.
Frame0 reported; follow the same handle through1866 and final revalidation.
The native audit was pending at launch and has since completed successfully.

Detailed implementation, failed-adapter custody, hashes and validation:
`go2_measured_settling_mission_prefix_result_2026-09-09.md`.
Next: finish readout16797 and prefix56186; review actual evidence before a
fresh native settling successor. No tenth native attempt is launched or
prepared yet. Return tracking, strict visibility, independent layouts,
baselines/ablations, real-time execution and hardware evidence remain unfinished.

Ninth readout16797 is now CLOSED exit0: result
7635c951522895fc03059a491be4f1539a11f85205e406780bb529ef43b70e5c,
launch858f8f3c91e05482e33fca56e6182c8a458fd1b2c615ad5e30e23a8a97e0e31a,
1543 sources. Actual path8.946266774399417m, minimum goal distance
0.021496480659927094m, terminal distance0.039020034366980254m.86completed
selected later-floor-resolved100ms intervals,0censored;1765completed waypoint
intervals,1714local reranks. Registered3D mean/max4.236899/8.581741mm;
median receipt-inclusive iteration1184.193229ms. No original outcome changed.
Full native/readout report: `go2_later_floor_resolution_maze_pilot_result_2026-09-09.md`.

Only controller prefix56186 remains live from this turn, last reported frame300.
PID2407049 confirmed running; source validation again passed all1555bindings.
Readout PID2407234 subsequently exited successfully. Follow56186 through1866
and final source/input/model checks; never restart on quiet output alone.

## Settled-boundary native successor prepared while replay continues

Current goal turn made concrete preparation progress; no blocker or completion
claim. Replay56186 is confirmed LIVE, last reported700, with no mismatch or
terminal failure reported. It has not reached its1866 stopping observation.
No tenth native output exists and no new native episode has been launched.

Prepared scripts:
- `scripts/run_go2_settled_boundary_maze_pilot_v1.py`
- `scripts/settled_boundary_maze_episode_development.py`
- `scripts/settled_boundary_maze_audit_development.py`
- `scripts/settled_boundary_native_prefix_comparison_development.py`
- `scripts/read_go2_settled_boundary_maze_pilot_v1.py`

The collector/audit are exact ninth-source derivatives changing only the
controller import/class and status labels. Native prefix comparison requires
identical physics/public inputs and all requested commands through1866,
including the hold at the changed mission state. It separately requires every
complete new decision to equal the prospective replay and rejects any
undeclared observer/map/model/planner difference. Mission-state change1866
is not mislabeled as a command change. The native launcher requires the actual
completed controller-prefix result SHA, the completed ninth raw-audited native
result3745c0b7c45265a2fcc38caf732b6f3f4b228487f344d23502dc7395077d5755,
and readout7635c951522895fc03059a491be4f1539a11f85205e406780bb529ef43b70e5c.

Scope/prefix tests14passed1.91s in79133; admission tests16passed2.05s in35268;
readout scope test1passed0.11s.31 preparation tests total. The prepared readout
retains ninth execution, pose accuracy, timing, failed/censored interval and
qualification accounting, changing only native input/output/status identities.

Protocols: `go2_settled_boundary_maze_pilot_v1_2026-09-09.md` and
`go2_settled_boundary_maze_readout_v1_2026-09-09.md`. Neither is a launched run.
After56186 completes, review its actual terminal result, run the new native
launcher --preflight-only with that exact hash, then launch the same prepared
native experiment if its full source/input/resource checks pass. One native
CPU scene/one numerical thread;32GiB RAM and existing10GiB collection plus1GiB
persistence above40GiB reserve. No parallel native scenes or automatic retry.
Keep the strict visibility failure909 and all nine previous outcomes intact.
The failed subpixel feature candidate is excluded. Measured settling does not
certify continuous speed or resolve return tracking, visibility or uncertainty.

Final check43761 CLOSED: all1555 running-prefix sources unchanged;1567 explicitly
discovered prepared native/readout sources validated. Both future output roots
are absent. Replay56186 subsequently reported frame800 and remains LIVE.

## Packed-owned mapping optimization: exact component result, full replay live

This goal turn made progress through a separate measured performance change,
tests and a completed paired real-observation benchmark. It did not change the
settled-boundary controller or prepared native experiment.

Found and reused the earlier batched insertion work (completed311-decision
benchmark577a13a3331b00428e6ac75f2f4da47a3cb128095d8e5a9350b1a20aaa663a49).
The new packed-owned candidate groups frozen25mm,+/-50m coordinates into
ordered12-bit fields and ensures each stored2x3 bound owns its data.10 tests
passed1.23s; full-stack empty-index wiring/public-packet tests2passed3.80s.
No existing index/controller source was modified.

Bounded benchmark8906 CLOSED:result
e3f603f7baed193e23b53116fe14f46a94bc6a98ad4b5eebca0af604dbe9a872,
launchfdbeffa26ed3cbefc3655eb00be372c312f8dea5508c93fba775cb508cb809ff,
1546 sources. All128 public-cloud insertions and sampled queries exact in
two opposite-order passes. Original/packed insertion-total ratios3.057791 and
2.962971. Final bound-array backing bytes1002912(original/packed) versus
42277536(previous batched),20894cells; excludes object overhead/other memory.
All cloud/input/source bindings rechecked. No whole-controller speedup claim.
Report: `go2_packed_owned_maze_bounds_benchmark_result_2026-09-09.md`.

Full packed-owned ninth-controller replay94114 is now LIVE:
go2_packed_owned_maze_controller_replay_v1_attempt_001,
launch90dc683b552a22f90cf2c920e2a1e041d2ae9a6ae1ff0a7c438ea08bbfb97b54,
1550 sources. Preflight1055 passed:RAM76,662,919,168 andartifact118,392,938,496
bytes; one CPU replay/one numerical thread. Last reported100. Follow through
all1881 original decisions, including original visual failure1870 and drain,
then final source/input/model checks. No native adoption is declared.

Settled-boundary prefix56186 remains LIVE, last reported1500 with no mismatch.
This is still the immediate predecessor for the prepared tenth native pilot.
Once it completes, inspect the result, run the exact-hash native preflight,
and launch if checks pass. The packed-owned replay is independent and may
continue alongside the single native scene; it does not change that scene's
frozen scientific definition. Full goal remains active; no blocker or completion.

Later confirmed progress: settled-boundary56186 at1600; packed-owned94114 at300.
Both remain LIVE with no reported mismatch. Source revalidation61919 checks
their respective1555/1550 frozen source manifests; neither output has a
terminal result or failure yet. The prepared native tenth launch remains
dependent on the settled-boundary prefix's actual completed result hash.

## Latest verified status: settling replay failed; optimization replay continues

Status inspection recovered closed session56186, exit1, after progress1800.
The complete-decision comparator rejected the candidate with
`ValueError('complete decision changed outside declared mission settling')`.
The retained failure.json SHA-256 is
f7dedb15dabba46667624a63434b4ba0bcf01d7b294c501dc19a095bf5f138ef.
There is no completed prefix result. The precise differing fields and failure
frame have not yet been diagnosed in this inspection. Preserve this attempt;
diagnose the mismatch before preparing any corrected successor. The tenth
native output root remains absent and its prerequisite has not passed.

Packed-owned replay94114 remains live, confirmed PID2410144, with progress800
and no terminal result or failure at inspection. Its full equivalence result
remains pending. The nine native outcomes and zero verified arrivals/round
trips are unchanged; the full goal remains active and unachieved.

## Settling comparison diagnosed and corrected; fresh replay launched

This goal turn is progress: it diagnosed the actual failed decision, implemented
and tested a narrow comparison successor, checked the saved failure, and
launched a fresh full controller replay. No blocker or completion is asserted.

V1 mismatch is1866: original planner_mode NEW, candidate WAYPOINT. Original
arrival changes the goal to home and the existing target setter resets mode;
candidate retains its preceding outbound goal/mode while held. All other
differences are the already declared settling/mission/label fields; both
commands remain zero. New comparison requires the immediately preceding
candidate witness and the exact held target-transition conditions.22 tests
passed1.80s; saved-mismatch check79563 completed and all1555 original sources
were revalidated. Original failed source/output is preserved. Detailed report:
go2_settled_boundary_prefix_target_reset_diagnosis_2026-09-09.md.

Fresh replay17684 LIVE, PID2412337:
go2_settled_boundary_controller_prefix_v2_attempt_001,
launch e36fa13e0c1bfb58868e9d3e5d2d5e4bfac2b5cb9a8808fd45023f790488c7d5,
1559 frozen sources, last reported0. Controller implementation is unchanged.
Preflight8792 passed; actual launch RAM77,569,933,312 and artifact-free
118,114,914,304 bytes. Follow the same handle through1866 and final checks.

Prepared native/readout V2 scripts and protocols now use this revised prefix
admission/comparison. Collector, raw audit, worker and readout computation are
unchanged.38 preparation tests passed. Source verification69911 completed,
checking1559 live-prefix and1571 prepared-source bindings. Once V2 replay
completes, run scripts/run_go2_settled_boundary_maze_pilot_v2.py with its actual
result hash and --preflight-only, then launch if admitted. V1 native remains
unlaunched and must not be used for the revised prefix. No tenth native output
exists. Use the V2 readout after native raw audit/prefix completion.

Packed-owned replay94114 remains LIVE, PID2410144, last reported1300, no
reported mismatch. Complete all1881 original decisions and final validation
before any adoption claim. It stays separate from the settling native pilot.
The full goal remains active and unachieved.

## Packed-owned full replay completed; paired controller timing launched

This goal turn is progress: the full performance equivalence replay completed,
its actual result was reviewed, and a tested controlled timing experiment was
prepared and launched after preflight. No blocker or goal completion is asserted.

Replay94114 CLOSED exit0. Result
4de3195f7976768c7840189e8c5c77227f5842bd090163c5975f2d8d8063b125,
compressed decision stream
ad92a19a38cdb9c4336030fc739b7af0ef61fe444178d0d2e214122e01606c55.
All1881 complete original decisions exact, first terminal1870 unchanged,
model state4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6
unchanged,1550 source/input bindings rechecked. Wall1781.3632316070143s is not
a controlled speed comparison. No native adoption. Full report:
go2_packed_owned_controller_replay_result_2026-09-09.md.

Paired timing63970 LIVE, PID2414108:
go2_packed_owned_controller_pair_v1_attempt_001,
launch3f83ac410219c5848d567a4931c9b5477ab51a2ca2259270613ce6096c825439,
1553 sources.11 focused tests passed2.14s. Prepared source check24180 passed;
preflight49027 passed. Actual launch RAM79,032,418,304 and artifact-free
117,912,555,520 bytes,3.6% CPU. One CPU process/two independent controller/model
states, one numerical thread beside settling replay. Each of the first256
observations is run by both original and packed-owned controllers, alternating
first position. Complete decisions, private input immutability and final model/
source/input checks are required. Full observe and production receipt writer
timed; decoding/sensing/physics/comparison excluded. Last reported64. Follow
through255 and final validation; no full-loop or later-trajectory speed claim.
Protocol:go2_packed_owned_controller_pair_v1_2026-09-09.md.

Settling V2 replay17684 LIVE, PID2412337, last reported800 without mismatch.
Its1559 sources revalidated by24180. It remains the prerequisite to native V2
preflight and launch; no tenth native has launched. Do not restart closed94114
or either original failed settling attempt.

Baseline context rechecked: the nine native policy attempts include two earlier
reactive runs. Nominal result1a3bf1e0f796d8fe5ae7b0b11feb837b7299064f83f6b9f44457d0040a229dd6
and connector resultd17f634e17f049caf40a0bbcbb65e0300594c1a234cc07f32838df30ecec0d70
match their exact recorded hashes. Both have zero round-trip successes and
zero independent layouts. Their distinct geometry gates and shared persistent
memory prevent matched attribution to prediction, JEPA training or memory.
Do not repeat or describe these completed reactive pilots as unexecuted;
the outstanding requirement is properly matched contribution/generalization
evidence. All goal-level arrival/return, visibility, real-time and hardware
limitations remain unchanged.

## Paired timing complete; actual camera identity integration complete

This goal turn is progress: controlled timing completed and a bounded actual
camera integration probe was implemented, preflighted, run and verified. The
full goal remains active and unachieved; no blocker is asserted.

Paired timing63970 CLOSED exit0. Result
7a91c2a5588002989f607da26db5ad0e6cedd789500a27c6f6f957a2f952a8c5.
All256 paired decisions exact; compressed streams both
ae76078064eed0096398866bb5c9ee7ffc00cd418e00ee3991dc359ae6d70144,
input immutability and model/source/input final checks pass.1553 sources;
wall404.3786761770025s. Median controller+receipt716.389810 ->621.678939ms;
paired ratio1.152069. Controller alone699.123943 ->603.449410ms. All256
observations in both arms still exceed100ms. This early-prefix/shared-machine
result excludes sensing/decoding/physics and does not establish a full-loop
speedup or real-time operation. Report:
go2_packed_owned_controller_pair_result_2026-09-09.md. No native adoption or
repeat benchmark is needed without a changed implementation/question.

Camera probe36972 CLOSED exit0, after timing finished. Preflight18011 passed;
one new CPU scene with a plane/fixed box and no robot or physics steps. Result
c1636416aa33295641358bec601a6bb6202e664410104f6afb9dbcd35103a8f5,
launch214974735107a19abc495985048fd0a7fa658d47aa2e5dc4392530c563ac1983,
1546 sources. All3 RGB/depth captures byte-identical across actual-camera
identity readback. Archive SHA
2061c24611a7065a33c9e2b97c7f8b72569baf4dc9006c7b8ec69640ab406291.
Actual context AMD/radeonsi, OpenGL4.6 core Mesa25.2.8, EGLcard0, softwareFalse;
single-sample24-bit depth, four-sample RGB, queried subpixel bits8. These do
not establish an arithmetic error bound or historical maze-renderer identity.
Report:go2_camera_renderer_identity_result_2026-09-09.md. Helper unchanged;
no frozen collector modified, no visibility failure relabeled.

Only settling V2 replay17684 remains LIVE, PID2412337, last reported1200 with
no mismatch. Follow through1866 and final source/input/model validation. Then
review the actual result and preflight scripts/run_go2_settled_boundary_maze_pilot_v2.py
with its exact hash before launching the same prepared native experiment.
Native/readout V2 remain unlaunched. Packed-owned mapping and the camera helper
remain excluded from that scientific definition; later explicit integrations
can use their completed evidence. Zero verified arrivals/round trips and all
independent-layout, matched-contribution, sensing/timing and hardware gaps
remain unchanged.

## Settling prefix passed; tenth native policy attempt launched

This goal turn is progress: the full settling replay completed, its actual
result was reviewed, native preflight passed and the new maze episode launched.
The prior turn was also progress (completed paired timing and camera integration).
No blocker or full-goal completion is asserted.

Replay17684 CLOSED exit0. Result
146bcb82547e28e24aad784ed69f05cdf4596dcdee7ebf996fd3fe0a14c53a58;
streamf76b730368d565f5eac6a8c320458183e018f393696b8a25ddf1564605df9475.
All1867 observations through1866 complete; first quiet difference1857, first
mission behavior difference1866. Final mission OUTBOUND, eight quiet intervals,
no arrival, held zero command. Complete decisions are exact outside declared
mission/settling and witnessed target-reset fields; all actual requested
commands exact. Model/source/input checks pass,1559 sources; wall1990.667471s.
Report:go2_settled_boundary_controller_prefix_v2_result_2026-09-09.md.
Do not restart this completed replay or the failed V1 attempt.

Native preflight40779 CLOSED exit0:1573 sources and completed inputs verified,
RAM81,940,430,848 and artifact-free117,647,486,976 bytes. One native scene and
one numerical thread admitted,32GiB RAM/10GiB collection+1GiB persistence above
40GiB reserve. No competing numerical job remained at native launch.

Native61184 is now LIVE:
go2_settled_boundary_maze_pilot_v2_attempt_001,
launch eee7ddac0d0a5410806c7e16f6d5abc71ee2cf9ffb3772ed4f910333868de188,
parent PID2417210, worker PID2417266. Actual launch RAM81,822,412,800 and
artifact-free117,646,929,920 bytes,0.6% CPU.1573 frozen sources. The worker
has built the scene and completed command/receipt tick0,571.968647ms including
the command and receipt. No terminal/failure artifact was present at inspection.
Follow the same live handle and bounded current worker/timing logs through
collection, full raw audit, physical/public/prospective prefix comparison and
final source/input/model validation. Never restart on quiet output alone.

The controller/mission change is observed settling before arrival. Original
tracking, model, mapping/contact logic, sensing, scene, gains, friction, seed,
budget and safety/visibility thresholds remain fixed. The failed subpixel
candidate, packed-owned mapping, camera identity helper and new single-pass
query candidate are excluded. Visibility909 precedes the changed mission
behavior and remains a preserved failing qualification condition. Physical
arrival/return must satisfy the unchanged native2ms-sample criteria; a
successful prefix or collection alone is not verified arrival/return.

After native completion, run scripts/read_go2_settled_boundary_maze_pilot_v2.py
with the actual completed --native-result-sha256. That readout is prepared but
unlaunched. Nine earlier native policy outcomes remain preserved and zero
verified arrivals/round trips remain the completed evidence until the new
audit proves otherwise.

Separate pending performance source:
lewm/single_pass_sample_bounds_development.py removes duplicate voxel query
enumeration, batches exact box predicates and retains scalar sphere arithmetic.
Eleven tests passed0.76s. Not timed, integrated, replay-qualified or adopted;
do not claim a speedup from tests. It can be benchmarked independently later
without changing the frozen native sources. Independent layouts, properly
matched prediction/JEPA/memory comparisons, realistic sensing/timing and
bounded hardware evidence remain unfinished; the full goal stays active.

## Single-pass query benchmark complete; full controller replay launched

This goal turn is progress: a recorded-cloud query benchmark completed, a
separate controller integration passed tests, full equivalence replay launched,
and a gated paired timing successor was prepared. No blocker or completion.

Query benchmark89167 CLOSED exit0. Result
570a3978c98b26d8c47d25a31f646c9278eae8221a86af2cd8ca6538e8991d79;
launchbd063784b959d499e2da79fb402fdea79e316846372657aa03d167ea20e562db,
1550 sources.36,864 timed queries per implementation across two opposite-order
passes; every complete receipt and evolving index state exact. Paired batch
ratios2.003319 and2.049575; final20894cells. Reference-receipt SHA
b8107c99187f05961d0fbf89c94be305e7d95ad24a179810abc8a3d93398a245
identical across passes. Wall29.499662s. Preflight62196 and nine-query-definition
check96745 passed. Component result is not an actual controller-query workload
or whole-loop speed claim. Report:go2_single_pass_maze_queries_result_2026-09-09.md.

SinglePassLaterFloorController replaces only eight empty persistent bound indices
and preserves original complete decision labels. Two wiring/public-packet tests
passed3.83s. Full replay70411 is LIVE, PID2418569:
go2_single_pass_maze_controller_replay_v1_attempt_001,
launchf3bfad77ee427563b1d94a1090308929486248bcbd63e3ab67bd857b9f2611b9,
1554 frozen sources. Preflight58821 passed; actual launch RAM78,088,425,472
and artifact-free116,067,303,424 bytes,3.5% CPU. One CPU replay/one numerical
thread beside the single native settling scene. Last reported100 without
mismatch. Follow all1881 ninth original decisions, including original terminal
failure1870 and drain, then final source/input/model checks. No native adoption.

Prepared scripts/benchmark_go2_single_pass_controller_pair_v1.py and protocol
go2_single_pass_controller_pair_v1_2026-09-09.md require the actual completed
single-pass full-replay hash before preflight/launch. Eleven tests passed2.21s.
The same256-frame alternating-order original-versus-candidate comparison times
observe and the production receipt writer, retaining complete decisions and
deadline misses. It measures combined packed insertion plus single-pass query
changes against the original controller, not isolated query improvement.
This benchmark is not launched.

Native61184 remains LIVE, parent2417210/worker2417266, latest completed receipt
tick631. No collection result, root failure or worker terminal exists at that
inspection. Follow its collection, raw audit, prefix comparison and final
validation; then invoke the already prepared V2 readout with the actual native
result hash. Do not alter any of its1573 frozen sources or settings. All query
optimizations and paired timing remain independent of that native experiment.
Zero verified arrivals/round trips remain the completed evidence. The full goal
and all outstanding contribution/generalization, sensing/timing and hardware
requirements remain intact and active.

Final source check46746 CLOSED exit0: all1573 live-native sources and1554
live-replay sources unchanged;1557 prepared paired-benchmark sources validated.
Replay70411 subsequently reported200 without mismatch. Both numerical jobs
remain live; no new native outcome or benchmark adoption is claimed.

## Auxiliary tracking diagnosis and public RGB packet integration completed

The user-status turn verified both live jobs. This continuation makes progress:
the completed auxiliary diagnosis was inspected and recorded, a distinct causal
RGB packet was implemented/tested and its actual-capture integration passed.
The full goal remains active and unachieved; no blocker is asserted.

Auxiliary diagnosis79343 is CLOSED, result
4980170b5c02e9ab8de55f331583f7b0070decdc53aa753dbe9466821cf855da;
launch b8d90d03570043029353d6bac02fd4b89fac827bca26402c2bab1a688c214544.
The missing old session handle is resolved by its completed result, not a
reason to restart.1548 sources; wall1.238170434s. All17 consecutive auxiliary
pairs fit (front15/17);4/8 auxiliary retained pairs fit (front0/8). At1869→1870,
auxiliary22 lifted matches fit with0.196mm translation error; front8 fails.
All4 retained auxiliary failures preserved. Native errors are postfit evaluation
only, not uncertainty bounds. Original feature/count witnesses reproduce
exactly. V1 failed before fitting because artifact_path received a case folder
instead of the attempt root; its launch00438ea2df4021ae4c08343dea2aa5a78fe1848065a46e93712a90306f2f3e1e
and failurefc7b88723e3fda3a2dfa302283dd4a6df6c81c898523b71781985b197b7a24e3
remain preserved. Report:go2_auxiliary_return_pair_diagnosis_result_2026-09-09.md.

New causal_auxiliary_rgb_observation_development and
novel_maze_auxiliary_rgb_packet_development modules keep auxiliary RGB distinct
from primary RGB and bind it to the current public auxiliary depth, acquisition
time, fixed calibration, episode and primary pixels. Seventeen tests passed
2.22s (4725 CLOSED). No primary packet is relabeled. Native pose/segmentation
are excluded; ideal simulated zero latency remains an explicit assumption.

Actual18-frame packet audit49037 CLOSED exit0. Result
122da7713c522ca06e8c6881597d42cc2b6f49a35bae691703aaf94f20a2b5c0;
launch52a6efe4ebd2bcfb5add19f7bfe27ccfbbbadaabbdffdf7eea235c7a5dbad908.
All pixels/public depth/feature witnesses exact;1553 source bindings and final
input checks pass; wall0.791064653s. Preflight18901 passed with RAM67,803,103,232
and artifact-free111,932,260,352 bytes,6.9% CPU. One CPU/numerical thread,2GiB
RAM/64MiB output above40GiB reserve beside the existing native and replay jobs.
Report:go2_auxiliary_rgb_packet_audit_result_2026-09-09.md.

Next auxiliary step is continuous causal observer replay from episode start,
with per-camera reference/retention state and fixed-extrinsic body conversion.
Pair fits and packet validation do not establish online continuity. Do not
install into or alter the already frozen tenth native attempt. No controller
integration, online tracking recovery or new navigation success is claimed.

Native61184 remains LIVE, last inspected receipt1760; no case collection result
or root terminal/failure artifact then. Full replay70411 remains LIVE and last
reported1700 without mismatch. Continue both existing handles. After70411
completion use its actual result hash to preflight/launch the already prepared
single-pass paired benchmark. After61184 collection, finish full raw audit and
prefix/final validation, then invoke the prepared V2 native readout. Zero
verified arrivals/round trips, independent-layout and matched-contribution
gaps, visibility failure and timing/hardware limits remain unchanged.

## Single-pass full replay passed; tenth collection ended, audit still live

Replay70411 CLOSED exit0. Result
4687d67fbb53fce3b29a122e379b51805fa685b4c39e3e817e3a5862374d342e;
stream649b85ece2777a66931465e4d65d8dc9a09c9152c70ccb9182d7fdaf0666e07c.
All1881 complete decisions exact, first terminal1870 preserved, model unchanged;
1554 sources and final input/source checks pass. Wall1784.984079356s is not a
controlled timing claim. Report:go2_single_pass_controller_replay_result_2026-09-09.md.
Prepared paired benchmark preflight43704 launched with this actual result hash.

Native61184 collection has ended with1883 observations,1882 commands and94850
physics samples. Case status SETTLED_BOUNDARY_MAZE_TERMINAL_AUDIT_REQUIRED;
schedule terminal SENSOR_OR_MODEL_FAILURE;10 terminal zero commands;
physical_stop/acquisition_stop None. Observed arrival is now1868, with10quiet
intervals, then last admitted mission receipt1871 in RETURN. This is collection
evidence only: native settling verification, tracking diagnosis, raw replay,
prefix/final checks and the prepared readout remain pending. Root result and
worker terminal absent at inspection; worker2417266 confirmed live at98.5% CPU
after collection. Continue61184, never restart the completed episode.

Source check63762 CLOSED pass: all1573 native,1554 replay and1553 packet-audit
source bindings unchanged after the new packet work and reports.

Paired benchmark preflight43704 CLOSED pass:1557 sources, RAM74,000,007,168
and artifact-free111,473,516,544 bytes,3.7% CPU. One numerical process/thread
beside the existing native worker, now performing its raw audit. The prepared
concurrency text says native scene; at actual launch the collection had ended
and the same worker was auditing. Resource allowance remains16GiB available
RAM and1GiB output above40GiB reserve; no new scene is launched.

Benchmark81883 is LIVE, PID2422742:
go2_single_pass_controller_pair_v1_attempt_001,
launch1976e3b5d93cdef6c8fa2c379416dfa1ebf6e66d7de2708dc5b458a4e84602a4.
Frame0 reported. Continue this handle through256 alternating-order pairs and
final full decision/model/input/source checks. Do not infer timing improvement
until the actual paired result. Native61184/worker2417266 remains confirmed
live; its collection is complete but full audit/prefix/readout is still pending.

## Combined timing complete; continuous dual-camera observer implemented

This continuation is progress: paired timing completed, the dual-camera
observer and continuous replay were implemented, tests/preflight passed and
the actual replay process started. The preceding turn also made progress
(packet integration and completed full optimization replay). No blocker or
full-goal completion is asserted.

Benchmark81883 CLOSED exit0. Result
85bcc450c6286c624c976b49cbc468d366275c87a80deec175439a967211def9;
both compressed streams ae76078064eed0096398866bb5c9ee7ffc00cd418e00ee3991dc359ae6d70144.
All256 complete decisions exact; model/source/input checks pass.1557 sources;
wall404.301770772s. Median controller+receipt726.092713→624.6901975ms;
paired median ratio1.177638. All256 frames in both arms exceed100ms. This
combined packed insertion/single-pass query result does not isolate additional
gain over the earlier packed-only candidate. No native adoption or whole-loop
timing qualification. Report:go2_single_pass_controller_pair_result_2026-09-09.md.

New lewm/dual_camera_anchor_pose_development.py stores both captured views at
each accepted body pose. Original primary anchor/increment path runs first;
only missing-current-pose or exhausted-bridge outcomes try the auxiliary path.
Qualified primary conflicts remain terminal; primary qualified increments at
exhaustion must agree with the auxiliary pose. Fixed camera extrinsics convert
gyro, fitted poses and point witnesses. Eight paired anchors and the previous
view share the existing10-frame bridge cap; no bridge promotion/reset. Half-
overlap retention uses the selected camera's feature population. This observer
is not installed in a controller or either native episode.

Ten tests2076 CLOSED passed3.32s: complete primary pose/continuity equivalence,
real synthetic front feature loss/downward fallback/front return, primary and
cross-camera qualified conflicts, shared bridge exhaustion and fault latching.
Source syntax valid. Prepared continuous replay explicitly processes all1881
ninth observations from frame0; compares primary pose/continuity/reference/
overlap evidence until first auxiliary attempt; preserves terminal failure and
opens native poses only after observation finishes for error evaluation.

Preflight36694 CLOSED pass:1558 sources, RAM74,326,736,896 and artifact-free
111,407,603,712 bytes,3.6% CPU. The paired benchmark process has finished.
One CPU observer replay/numerical thread is admitted beside the existing tenth
native worker, now auditing;4GiB available RAM and1GiB output above40GiB reserve.
Replay73161 started via scripts/replay_go2_dual_camera_observer_v1.py; inspect
its existing handle/root go2_dual_camera_observer_replay_v1_attempt_001 for
launch/result/failure. Never restart merely because startup verification is
quiet. Native61184/worker2417266 confirmed live98.8% CPU, root result/audit/
failure absent at last inspection. Its full raw audit, prefix/final validation
and prepared readout remain pending. All full-goal limitations remain active.

Replay73161 confirmed LIVE, PID2424066, launch
f0ea9c76b46e2c0724159deed6d97f821827a27415d91b007dca99b06f1a56d0.
Reported frames0,100,200,300,400,500 with no auxiliary attempt, terminal failure
or primary-prefix mismatch. Continue to1881 and postfit/source/input checks;
do not infer recovery from the unchanged early prefix. If tracking completes,
inspect full errors and failure witnesses before prospective controller
integration. Native61184 remains the existing audit handle.

## Continuous observer recovered all recorded frames; integration replay starting

This continuation is progress: the continuous observer completed successfully,
motion-evidence/controller integration was implemented and tested, and the
full registered-pose replay passed preflight. The previous turn also made
progress (observer implementation, tests and launch). Goal remains active and
unachieved, no blocker asserted.

Observer73161 CLOSED exit0. Result
540f63243f74b63e90dadeaa8e7aebde936bb7880d49f0c77ffc5e8affc59eea;
stream80debed2accb3361e445a9ca44896a591703455bbdb8b944e2c393f9682562b5;
postfit59ca2616cc6f23465c439eefceed9f5fcf36682f3bb57669195132a328907551.
All1881 poses accepted, no terminal failure/reset. Exactly1870 original primary
pose/continuity/reference/overlap frames before auxiliary firstuse1870.
Auxiliaryselected1870(ref1854,inliers17),1871(ref1870,inliers27),1876(ref1875,
inliers28); all anchor measurements following primary missing translation.
1877primary-selected frames plus initialization;8retainedrefs,331keyframes,
1measured bridge. Final1558 source/input checks pass. Wall187.523493808s.
Postfit XYZ max15.468mm/mean7.552mm, XY max8.986mm/mean4.379mm, rotation
max0.011267rad/mean0.005182rad. Observermedian54.525784ms/max200.110765ms,
18/1881over100ms. These exclude decoding/mapping/planning/physics and are not
uncertainty bounds. Report:go2_dual_camera_observer_replay_result_2026-09-09.md.
Original failed navigation outcome unchanged: this follows the old stop/drain
trajectory, not new commands or a prospective return.

New DualCameraVisualMotion provides explicitly bound auxiliary pixel/depth
hashes, calibration and current/historical camera-choice witnesses while
retaining existing joint-pose composition checks. current_dual_camera_pose
checks these additional bindings before floor registration. New
DualCameraSettledController wires it into the original settled mission,
registration, map, residual and selector; no performance optimizations added.
This controller has only a synthetic initial-packet/stop test so far, not a
full controller replay or native adoption.

Tests55307 initially failed7/7 on a snapshot timestamp variable typo (now vs
now_ns) in the new unfrozen adapter. Fixed before any attempt launch. Tests6262
CLOSED passed7 in3.67s: primary evidence equivalence outside declared added
modality metadata, actual synthetic auxiliary fallback through existing joint
rotation-witness validation, tamper rejection, stale pose semantics, initial
public floor/map/residual pipeline and latched zero command on missing RGB.

Prepared scripts/replay_go2_dual_camera_registered_v1.py uses one continuous
motion wrapper and unchanged JointFloorRegistration on all1881 ninth frames.
It requires raw evidence equality to the completed observer, exact original
registered-pose/floor receipts through1869, and postfit-only native error
evaluation. Full controller commands are not executed. Preflight52906 CLOSED
pass:1566sources, RAM74,190,471,168/artifact-free111,379,759,104 bytes,3.7%CPU.
One CPU/numerical thread beside the continuing tenth episode audit,4GiB RAM
and1GiB output above40GiBreserve. Native61184 remains LIVE with no new output;
continue its audit/prefix/final checks and prepared readout. Registered replay
execution has been started; follow its existing handle, never restart on quiet
source/input verification. All full-goal limits remain unchanged.

Registered replay39917 confirmed LIVE, PID2425545, launch
85842f515db20635c0f7df9807c3d10030510e3b63bdca85561cbc4b7d604c24,
root go2_dual_camera_registered_replay_v1_attempt_001. Frame0 admitted without
failure. Continue all1881 raw/floor comparisons and final postfit/input/source
checks. Worker2417266 (native61184) also confirmed LIVE at99.0% CPU after one
hour total elapsed; collection ended earlier and the worker is still auditing.
No new native scene, qualification, retry or outcome relabeling occurred.

## Registered replay passed; complete controller prefix prepared on closed collection

This continuation is progress: full registered replay completed, the complete
controller comparator/protocol/launcher were implemented and tested, and
preflight started. The preceding turn also made progress (motion integration
and registered replay launch). Goal remains active and unachieved, no blocker.

Registered39917 CLOSED exit0. Result
2310908d24aee137884d6e4fcfded92a2b31f03b93edfae7608ba48f2a0b4190;
streamd35e8b9fade8ebb987581031becdbd7c3c189463c0e3a946d0db5e3d9532c303;
postfitcd87c21c947be50c9768679f297d00cfd746fbd288701766361b87dc3c5f98f1.
All1881 raw poses match the completed observer and pass explicit dual-camera
admission; all1881 register to the floor and pass the existing accessor.1870
complete original registered-pose/floor receipts exact before auxiliary use.
No failure. RegisteredXYZ mean4.229/max8.582mm, XYmean4.227/max8.578mm,
rotationmean0.004236/max0.010556rad; postfit-only native evaluation. Combined
replay processingmedian140.789853/max302.732054ms, all1881over100ms; includes
extra replay admission/comparison and is not native-loop timing.1566source/
input final checks pass; wall343.070163273s. Report:
go2_dual_camera_registered_replay_result_2026-09-09.md.

The tenth collection's exact case result is
2875fbdee5d5069de2ec3ef872f942e5a52b53d34eafccd88108cab45a3cd914.
It has1883observations,1882completedcommands,94850physics samples,10terminal
zero commands. Frozen worker source confirms persistence/scene destruction
and complete collection hashing precede the read-only native audit. Worker
2417266 remains live; observed one open case descriptor and zero case-write
descriptors. The full native audit is still separate and pending.

New lewm/dual_camera_controller_prefix_comparison_development.py validates
dual-camera and registered witnesses and then normalizes only explicit added
metadata paths. Every remaining complete decision and command must equal the
original settled controller; primary selection with no auxiliary attempt is
required. Tests58907 first found aliasing between current_pose/last_visual in
in-memory snapshots (4fail/3pass); fixed by independent copies at explicit
normalization paths before any attempt launch. Tests63907 CLOSED passed7 in
7.87s, including command/mission/unrecognized-field/binding/camera tampering.

Prepared scripts/replay_go2_dual_camera_controller_prefix_v1.py and protocol
go2_dual_camera_controller_prefix_v1_2026-09-09.md bind the closed tenth
collection's explicit full artifact inventory plus the completed registered
replay. A fresh assigned model/controller starts at frame0 and stops after
the first auxiliary-camera decision, without consuming later recorded frames.
This permits useful prefix work beside the native audit. Before any new native
run, the tenth audited root result and its final artifact map must match these
same bound collection bytes; this prefix never replaces or bypasses that audit.
No mapping-performance candidate is included. Preflight55998 started with the
actual registered result hash; inspect its existing handle before execution.
Native61184 remains the audit handle. All whole-goal limitations remain active.

Preflight55998 CLOSED pass:1599sources, RAM73,770,557,440 and artifact-free
111,371,751,424 bytes,3.7%CPU. Controller-prefix execution9221 started and its
PID2426566 is confirmed live during initial verification. One CPU/numerical
thread beside the existing native audit,8GiBavailableRAM and1GiBoutput above
40GiBreserve; no new scene or model training. Follow9221 to its launch and
first camera intervention. Nativeworker2417266 remains confirmed live99.1%CPU;
its case-specific worker terminal, root result/failure and case audit were
absent at inspection. Do not restart either running job on quiet output.

## Preserve V1 JSON adapter failure; V2 restores explicit identities only

Prefix9221 is CLOSED exit1, not live. V1 launched as
8b01122596274745c0d2e0de89cf8f51956ebafc316b58977d7339223f571efe
then failed at comparison frame0 before admitting any prefix row. Failure
25421b3950be88c00340b586399ec151324bb771509782de6c809c7567637ca2;
mismatchf3e1d3d3894547374ff4dfb4aa4fb4f16d219cca707c9e4cffb2202a7488f79a.
Cause: the launcher serialized the controller decision to JSON lists before
calling strict pose admission, which requires tuple episode identities. This
is the same representation boundary already handled by the existing
registered_json_pose adapter; it is not a navigation failure or model change.
Preserve all V1 source/artifacts and do not retry its root.

New dual_camera_json_prefix_comparison_development restores only the three
explicit raw/registered identity paths in both decoded decisions using existing
strict json_identity, then delegates to the frozen V1 comparison unchanged.
New V2 script/protocol/root inherit and bind failed V1 source/launch/failure/
mismatch identities; controller, model, sensor/scene, thresholds, collection
inputs and first-intervention stopping rule remain unchanged. Tests6340 CLOSED
passed6 in7.10s, covering real serialized decisions, malformed/wrong episodes,
command differences and no mutation. Saved actual mismatch frame0 also passes
the V2 comparator with exact original decision and commands (check96425);
source/input bindings revalidated. V2 preflight started with the same actual
completed registered replay result. Native61184 remains LIVE and unmodified;
its root result/audit/worker terminal/failure still absent at latest inspection.

Saved actual frame0 check96425 CLOSED pass including final source/input checks.
V2 preflight12117 CLOSED pass:1603sources, RAM73,550,147,584 and artifact-free
111,369,089,024 bytes,3.7%CPU. One CPU/numerical thread beside native audit;
8GiBavailableRAM and1GiBoutput above40GiBreserve. V2 execution has now started
with the same registered-result hash. Follow the new V2 handle/root, preserving
failed V1. Nativeworker2417266 remains live; no source or old outcome changed.

V2 prefix39133 confirmed LIVE, PID2427374, launch
28aec506d225093a8f0b8d022d019c9d5e616c504430837020442532b5b68c8a,
root go2_dual_camera_controller_prefix_v2_attempt_001. Actual frame0 passes the
full comparison. Continue to first auxiliary intervention and final source/
input/model checks. Native61184 also remains live with quiet audit output.
Do not use failed V1 or the preparation-only V1 result status for admission;
the expected successor status is DUAL_CAMERA_CONTROLLER_PREFIX_V2_COMPLETE.

## Tenth native audit complete: first physically settled arrival, return still failed

This continuation is progress: the tenth native audit finished, next-experiment
collection/audit/session/prefix-admission components were implemented and
tested, and the prepared tenth readout launched. The preceding turn also made
progress (registered replay completion and corrected controller-prefix launch).
Full goal remains active and unachieved, no blocker asserted.

Native61184 CLOSED exit0. Root result
a7a02db120b4b662cd66efee01f10784edb6f2ed6984bdc420007773a1b1b6fb,
status SETTLED_BOUNDARY_MAZE_PILOT_V2_COMPLETE; wall4828.639275411s. All raw
sensor reconstruction/model-controller replay/actual command/model-state checks
pass. Full physical/public/prospective prefix exact for1867 observations;
first mission-state difference1866, all compared commands exact, physical
prefix9b6e3058da959b29264be1f7def259b8a2d0c79028a121ddab4810251479c4b5.
Audit0c5debfa4eb4c8986c5b661f7902ddcf36158e71131e13ced02536ad3f5c76f3.

The independent native one-second arrival window now PASSES at frame1868:
maximum goal distance0.037619440722m, maximum3D speed0.039015776111m/s against
0.05m/s. This is the first physically settled outbound arrival among completed
episodes. It is not a qualified navigation/round-trip success: return has no
edge crossings, physical route not retraced, terminal native quiet false,
strict visibility false, zero verified round trips. Hard-measurement failures
remain empty. Only development maze0 has executed; all broader requirements
remain unchanged. Do not keep reporting zero physically verified arrivals,
and do not promote this arrival into independent-layout or full qualification.

Prepared readout scripts/read_go2_settled_boundary_maze_pilot_v2.py launched as
8542 with the actual root hash above. Follow to completion, inspect full failure,
route/pose/timing evidence and record the result. No new native scene is live.
Controller-prefix39133 remains LIVE and last reported400 without mismatch.

Prepared separate dual_camera_novel_maze_session_development,
dual_camera_settled_maze_episode_development, dual_camera_settled_maze_audit_development,
dual_camera_native_prefix_comparison_development and dual_camera_native_admission_development.
Session reads already captured auxiliary RGB after unchanged paired capture;
no extra render/physics or privileged packet fields. Collector/auditor pass
the new modality through the same controller, retaining all other gates.
Native comparison requires exact physical/public history and earlier commands,
plus the actual bound candidate command at the first auxiliary intervention.
Admission requires the complete V2 prefix, full tenth native audit and exact
matching collection hashes. Nineteen tests20342 CLOSED passed1.92s. Preparation
report:go2_dual_camera_native_preparation_v1_2026-09-09.md. Next native launcher/
readout are not yet prepared or launched; no existing source was modified.

Actual final-binding check95912 CLOSED pass: all11331 closed-collection
bindings in the live V2 controller prefix match the completed tenth native
artifact map, including its launch and case result. Predecessor qualification
failure remains preserved. This resolves the pending collection-versus-final-
audit identity condition; controller-prefix completion is still required.
Tenth physical/public prefix report SHA
9673a7e60427c7a2153a544b343f4f56ec56f4e9574fe00958570334fab946a1.

Readout8542 confirmed LIVE, PID2428382, processing the completed tenth root;
prefix39133 confirmed LIVE, PID2427374, last reported400 with no mismatch.
No native scene/audit worker remains live after61184 completion. Continue
these same two handles. After readout completes, record full execution metrics;
prepare the next native launcher/readout around the actual audited predecessor
and the still-required completed V2 controller prefix. Do not bypass that
prefix or infer a prospective return from the ninth stop/drain replay.

Readout8542 CLOSED exit0. Result
a5523b729310d39a81baf9ccf9cf94fa6fb8a82077e322d4fbe394a45748a70d;
launchcaa8777e40aeddd78b70bda38ab64b9a67a628c55a3a171894658f7b1e7de2ab,
1575sources, original outcome unchanged. Path8.946839280m/188.2simulateds,
minimum goal distance0.021496481m/terminal0.038714644m. First visual admission
failure1872, last admitted1871. RegisteredXYZ mean4.235/max8.582mm; XYmean
4.233/max8.578mm.86completed later-floor intervals,1765waypoint intervals,
1714localreranks. Receipt-inclusive median1231.596967/max1936.498306ms,
all1883over100ms. Complete report:
go2_settled_boundary_maze_pilot_result_2026-09-09.md.

Only prefix39133 remains LIVE, latest reported600 without mismatch. Continue
to its first auxiliary intervention and final checks. The next native launcher
and readout can now be prepared with the actual completed tenth native/readout
hashes above, but execution still requires the completed V2 controller prefix.
All new native-preparation modules remain unlaunched. Goal remains active:
one physically settled outbound arrival, zero verified round trips, no newly
executed independent layouts or matched contribution/hardware qualification.

## Dual-camera native launcher and readout prepared

This continuation made implementation/test progress; the preceding status turn
was a verified wait (live PID2427374/session39133, then frame1100). Goal remains
active and unachieved; no blocker asserted. The first physically settled
outbound arrival and failed return/strict visibility remain unchanged.

New scripts/run_go2_dual_camera_settled_maze_pilot_v1.py and
scripts/read_go2_dual_camera_settled_maze_pilot_v1.py are ready for completed-
prefix admission. New scripts/dual_camera_intervention_witness_development.py
requires the saved intervention/stream/summary to match, no earlier/later
intervention, active controller and raw/registered current poses at its exact
frame. The native launcher pins the actual tenth native and readout hashes,
checks full predecessor/prefix bindings, uses predecessor native binary/data
identity and the unchanged model admission, and reruns input/source checks in
the worker before and after execution. The readout adds actual camera-use
frames while preserving physical/error/route/timing calculations.

Focused tests24546 CLOSED34passed1.97s. One added stale-frame case and tighter
current-pose check then passed all16 execution-scope tests76324 in1.82s;
the19 unchanged preparation tests had passed in24546. Source68936 CLOSED pass:
all1603 live frozen sources unchanged, compatible tenth readout inheritance,
1618 combined prepared source paths, both new output roots fresh. CLI77586
CLOSED pass. Collector/auditor/readout exact diff scope reviewed. No frozen
source was edited; no new native execution or performance candidate adopted.

Prefix39133 remains LIVE, latest frame1300 without mismatch. Continue the same
handle until its first auxiliary intervention and final checks; do not restart.
Expected completed status DUAL_CAMERA_CONTROLLER_PREFIX_V2_COMPLETE. If it
passes active-controller/current-pose admission, run the new launcher with the
actual --prefix-result-sha256 and --preflight-only, inspect current hardware/
competition and admission results, then launch one fresh prospective episode.
After completion, run the prepared readout with its actual native result hash.
New roots go2_dual_camera_settled_maze_pilot_v1_attempt_001 and
go2_dual_camera_settled_maze_readout_v1_attempt_001. Preparation details:
docs/go2_dual_camera_native_preparation_v1_2026-09-09.md.

Latest same-handle observation: prefix39133 reached1500 without mismatch;
PID2427374 confirmed Rsl,28m31s elapsed/28m29sCPU,99.8%CPU,RSS5,730,012KiB.
No terminal result or new native launch was observed. Continue this process
and admit its actual completion before the prepared prospective native run.

## Dual-camera controller prefix completed; native preflight started

Prefix39133 CLOSED exit0, result
62037c72cc85546a5379299226aabf3d61f4715d92ce64dfb503a0ebd98909c1,
status DUAL_CAMERA_CONTROLLER_PREFIX_V2_COMPLETE,2164.020888841s. Exactly1873
observations:1872 complete prior decisions exact outside validated modality
metadata, all earlier commands exact. First auxiliary intervention1872 obtains
a registered pose after primary missingness, remains active in RETURN with
command[0,0,-0.45], no terminal/failure. No later recorded observations consumed;
assigned model4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6
and all frozen source/input checks pass. Stream
fdd7dcc6a0943e3820f484658017bba267f051f48e27fd63358ee4011d9c14f6,
interventiond1e4e4a21c0a3fcfee471f345906d0c93509897cb31e839193c397cfba3fab6f.
Report docs/go2_dual_camera_controller_prefix_result_2026-09-09.md.

Preflight3614 is now running the prepared native launcher with this actual
--prefix-result-sha256 and --preflight-only. It must validate the saved stream/
intervention and all predecessor bindings, inspect hardware and pass memory/
storage admission before a fresh native launch. No new native scene yet.
Initial hardware check99662 CLOSED:16physical/32logicalCPU,3.5%CPUbusy,
76,164,587,520availableRAM,111,109,492,736artifact-free bytes; only the then-live
prefix and the pre-existing small python3 process competed. Final preflight
must refresh these measurements. Continue3614; preserve the completed prefix.
This continuation completes evidence that changes the next action; goal stays
active and unachieved with one physical settled arrival and zero round trips.

Native preflight3614 CLOSED pass:1616 source bindings; all completed inputs/
sources and saved intervention admitted. Current hardware81,595,289,600bytes
availableRAM,111,048,318,976artifact-free,0.5%CPUbusy; only pre-existing tiny
python3PID6681 competes, no native scene. Memory32GiB and storage10GiBcollection
plus1GiBpersist above40GiBreserve both pass. Execution of the prepared native
launcher has now been requested with the same completed prefix result hash;
follow its returned handle through admission/launch, collection, full raw audit
and prospective physical/public prefix comparison. Do not start a second scene.

Native session67399 now reports LAUNCHED, launch
afb8485c377a12989004cdc7827476948d5308b3b89387a7327b385c2739cba6,
root go2_dual_camera_settled_maze_pilot_v1_attempt_001. All1616 launch sources
are now frozen: do not edit them. Actual launch admission confirms all11331
predecessor collection bindings matched, the1873-frame saved stream/intervention
matches its summary, current registered pose available at1872 and no following
recorded observations consumed. No terminal result or failure is present.
Follow67399 through its worker startup, collection/full raw audit/prefix check.
This is the eleventh requested native episode, not an eleventh completed result.

Native67399 startup confirmed: parent2432607, worker2432832 (resource tracker
2432831). Worker is Rl/98%CPU; first paired raw captures, setup checks and
compressed decisions are present, timing stream reached tick92. No root
failure, worker terminal or case result exists yet. Monitor elapsed128.12s:
RAM79,938,166,784bytes, artifact-free110,804,099,072bytes, CPU3.6%, native
workerRSS2,757,992,448bytes. Collection is live, not stalled or complete.
The worker log contains the existing Genesis neutral-qpos joint-limit warning;
no terminal exception observed. Continue same handle and preserve all1616
frozen sources. New readout remains unlaunched until actual completed native
result/full raw audit/prefix identity are available.

## Exact map-performance integration for the dual-camera controller

This continuation adds and tests a separate performance candidate while the
native episode continues. The previous turn completed the dual-camera prefix
and launched fresh collection, so it was progress. Goal remains active and
unachieved; no blocker asserted.

New lewm/single_pass_dual_camera_controller_development.py subclasses the
unchanged DualCameraSettledController and installs the existing empty-state
SinglePassLaterFloorMap. Its eight bounds indices use the already tested
packed-owned/single-pass representation. No observer, settling mission,
registered pose, model, selector, memory semantics or decision labels change.
The live native scene has not adopted this candidate.

Tests24224 CLOSED2passed3.00s: distinct empty indices with original dual-camera
and mission types, complete actual public RGB/depth/gyro/floor pipeline
decision and stored bounds equality, independent owned bound storage, and
the unchanged terminal missing-RGB stop without map insertion. Source43148
CLOSED pass confirms all1616 live native launch sources unchanged.

New scripts/replay_go2_single_pass_dual_camera_prefix_v1.py binds the completed
1873-frame dual-camera prefix, its fully audited tenth native collection and
the earlier single-pass complete-controller equality result4687d67fbb53fce3b29a122e379b51805fa685b4c39e3e817e3a5862374d342e.
It requires every complete decision to match, including the first auxiliary
intervention1872, all earlier actual commands and unchanged public input arrays.
It consumes no later recorded observations. Telemetry is not a controlled
speed comparison or native-loop deadline evidence. Protocol:
docs/go2_single_pass_dual_camera_prefix_v1_2026-09-09.md.

Preflight91936 CLOSED pass:1624 source bindings,77,764,628,480availableRAM,
109,749,084,160artifact-free bytes,3.7%CPUbusy; only the one live native parent/
worker and its resource tracker plus old small python3 compete. Admission
reserves8GiB replay plus32GiB native RAM headroom and1GiB replay plus11GiB native
output headroom above40GiBreserve. One CPU/numerical/OpenCV thread, no new scene.
Execution requested with the actual prefix hash62037c72cc85546a5379299226aabf3d61f4715d92ce64dfb503a0ebd98909c1,
exclusive root go2_single_pass_dual_camera_prefix_v1_attempt_001. Follow its
returned handle through exact replay and final checks; preserve failures.

Native67399 remains live: timing stream last reached tick472, no terminal
output reported. Continue parent2432607/worker2432832 on the same handle.
The prior settled arrival and zero verified round trips remain unchanged.

Optimization replay2602 now LAUNCHED, PID2434078, launch
1ffeb5d5028a5092a7ed5f58da06556e860ec6231cd1ae102e25174986e87f97,
root go2_single_pass_dual_camera_prefix_v1_attempt_001,1624 frozen sources.
Frame0 complete-decision/input-array comparison passed. Process confirmed
running, RSS1,677,852,672bytes. Follow2602 to all1873 frames and final checks;
do not edit any of its bound sources or adopt it in the live native scene.
Native67399 reached tick591 and continues on parent2432607/worker2432832.
The only two numerical jobs are this single native scene and the independent
one-CPU equality replay; no new native outcome is available yet.

## Future actual maze renderer-provenance integration prepared

This continuation implemented/tested an acquisition provenance recorder and
its bounded actual-maze integration launcher. The preceding turn implemented
and launched the map-equivalence replay, so it was progress. Goal remains
active and unachieved; no blocker asserted. No frozen live source was edited.

New maze_renderer_witness_development and
renderer_witness_dual_camera_maze_session_development query the original
camera context after primary capture and after paired capture/restored primary
pose, with no extra render/physics calls or public packet changes. Record
identity/sampling/precision plus raw pixel hashes and exact acquisition
boundaries, reject query-side pose/clock changes or context drift, and preserve
latched failure evidence. Saved witnesses must match separately audited raw
capture hashes/sample/time and optical pose. Endpoint provenance does not
reconstruct shader arithmetic or prove a raster error bound.

Installed renderer source check34378 CLOSED pass against completed actual-
Camera probe:1dc7c47b17a82c8aad7c2801bbfad7632dcaeeab8ecc8a382ebe06dee21a0b84.
The inspected segmentation path uses the single-sample draw target and disables
multisampling, supporting the post-pair query point. Old visibility failure909
and historical camera identity remain unchanged. Tests90857 CLOSED13passed2.01s
for capture/query/audit/failure/persistence invariants.

Prepared scripts/probe_go2_maze_renderer_witness_v1.py with separate bounded
collector and comparison modules. It requires the actual completed current
native result/full raw audit/prefix identity and the existing actual-Camera
helper probe, then executes only three unchanged zero-command warmup
observations/900 physical samples. Explicit acquisition cutoff, no fabricated
mission terminal. Reuse full raw sensor/model/command audit and compare every
startup physical sample, complete decision, public packet and full raw native
primary/auxiliary capture. Six context endpoints must match raw acquisitions.
Tests98697 CLOSED15passed2.29s; collector diff reviewed; CLI9968 CLOSED pass.
Source39474 CLOSED pass:1631 prepared probe sources, all1616 native and1624
optimization frozen bindings unchanged, probe root absent. No new scene or
probe preflight launched; current native completion is still required.

Protocol docs/go2_maze_renderer_witness_probe_v1_2026-09-09.md; preparation
report docs/go2_maze_renderer_witness_preparation_v1_2026-09-09.md. Future root
go2_maze_renderer_witness_probe_v1_attempt_001. After current67399 full audit
completes, use its actual --native-result-sha256 and --preflight-only, review
actual hardware/competition and run this single bounded scene if admission
passes. No parallel native scene and no change to the current experiment.

Latest same-handle updates: native67399 timing tick1597; optimization2602
frame1400 without reported mismatch. Both remain live. Continue67399 toward
the physical return/terminal/full audit; continue2602 to all1873 prefix decisions
and final identity checks. No new completed round-trip result is available.

## New native floor-extent stop; map-equivalence replay complete

Optimization2602 CLOSED exit0, result
18b79c77516ef738366298ee3e093952a250cfd083938818004d232e71a7ca96,
stream550606ee52254fa3e6a57cafa961eee9aa3d367a59e4552e927ec2d974aff201,
wall1841.509078746s. All1873 complete decisions and input arrays exact,
all1872 earlier actual commands exact, final auxiliary intervention1872 remains
active with turn[0,0,-0.45], model/source/input checks pass. No following old
observation consumed; no speed/native-adoption claim. It does not cover the
new native trajectory after1872. Report:
docs/go2_single_pass_dual_camera_prefix_result_2026-09-09.md.

Native67399 collection has now stopped on a new controller failure1904, with
last admitted mission/registered pose1903. Closed case result
dc5d54e1372b67ae88c783cfb06034777c3e50eccdba40f2aeb1dc5bb3f4fdc5:
1915 observations,1914 completed commands,96450 physics samples, ten terminal
zeros, no physical/acquisition stop. Case status
DUAL_CAMERA_SETTLED_MAZE_TERMINAL_AUDIT_REQUIRED; root audit/result pending.
Worker2432832 remains active on full raw audit; do not restart or edit sources.
Observed arrival1868/return turn1872 occurred, but no new independent physical
arrival or round-trip result is established until full audit finishes.

Preliminary read-only reconstruction15541 CLOSED pass, all1616 sources and19
explicit selected input bindings checked before/after. Frames0 and1903 floor
fits reconstruct exactly. Frame1904 still has valid current auxiliary visual
pose, primary floor candidates0/auxiliary6514, and the original joint fit rejects
insufficient_combined_two_axis_extent. Compose/fit have no differing moment
fields: the generic “admitted combined measured moments must reconstruct
exactly” error denotes unavailable extent here. At1903 auxiliary6780 points
passed, maximum residual4.058999262e-6m/RMS1.110891883e-6m. No native pose was
used in fitting, no point trimming or threshold change. Full raw visual replay
is still separate and pending. Report:
docs/go2_dual_camera_floor_extent_preliminary_diagnosis_2026-09-09.md.

Immediate work: continue67399 full audit and then the already prepared native
readout. Bind a fuller extent/geometry diagnosis to the final audited collection.
Investigate explicit current-visual-measurement continuation from an admitted
floor reference during current plane unavailability, without inventing a
current plane, weakening existing extent/coherence gates or hiding conflicts.
No continuation candidate is implemented. The prepared renderer witness probe
still requires this native completion and its own hardware preflight; no new
scene was launched. Goal remains active and unachieved; this turn made progress.

## Floor-extent diagnosis completed; measured visual transport implemented

The preceding user-status turn was a verified wait: parent2432607 and
worker2432832 were confirmed live. This continuation makes implementation and
diagnostic progress. Goal remains active/unachieved: one independently verified
settled outbound arrival, zero round trips, only development maze0 exercised.

Diagnosis preflight42731 CLOSED pass; numerical25318 CLOSED exit0. Root
go2_dual_camera_floor_extent_diagnosis_v1_attempt_001, launch
edd9d0d591af674629abf36b52c44a6451fe7c4f85bdd332bdccc0047cf6f9c9,
result1677bd7be89e5dd85c2e467693575aa5144f3f72fcef09ab16e8afa38b513a35.
1618 sources/65 selected closed inputs verified before/after; all14selected
frames reconstructed, every earlier admitted plane exact, first failure1904.
The second eigenvalue falls from0.002648481588m² at1903 to0.002327368543m² at1904
across the unchanged0.0025m² gate (51.4634mm ->48.2428mm spatial standard
deviation).1904has6514auxiliary/0primary candidates; algebraic maximum residual
3.291055885e-6m, no postfit robot candidates. No threshold change, trimming or
native-pose fitting. Report docs/go2_dual_camera_floor_extent_diagnosis_result_2026-09-09.md.

New measured_floor_transport modules now implement an explicit current visual
pose composed from the most recent admitted floor anchor while current count/
extent is insufficient. Original full-plane registration/reacquisition stays
exact. All current candidates retain3mm conflict checks against the transported
reference; raw visual gates remain, no new uncertainty bound or current-plane
claim, no unadmitted anchor promotion. Mapping/residual/mission/contact share
explicit schema dispatch. This is an unexecuted development hypothesis.

Tests57641 CLOSED31passed10.56s; CLI2482/source33055 CLOSED pass,1626 prepared
prefix source paths. Original1616native/1618diagnosis/1624optimization source
bindings unchanged. Full preparation:
docs/go2_measured_floor_transport_preparation_2026-09-09.md.
Prepared scripts/replay_go2_measured_floor_transport_prefix_v1.py requires the
actual completed eleventh --native-result-sha256 and --preflight-only; it binds
all65diagnosis inputs to the final native artifact map. Fresh controller replay
must exactly match1904prior complete decisions/actual commands outside validated
labels, then admit the same raw current visual pose and explicit transport
from recorded1903anchor at1904 in active RETURN. Stop before1905. No replay
root/preflight or changed-action native outcome exists yet.

Independent read-only geometric check86294 is running on the bound1904 candidate
points, using saved1903/1904 visual witnesses; no controller/nativepose/future
observations. Native67399 remains live, last confirmed worker2432832 Rl/99.3%CPU,
84m00s elapsed/83m26s CPU. No root result/failure/audit observed. Continue the same
native handle; after completion run its prepared readout, admit the new replay
and the separately prepared bounded renderer witness probe after resource checks.

Native67399 CLOSED exit0, result
44710966178a57b31f7da3bec10ad4f21710bcff3701b0a1038750f1ef6d747c,
wall5138.616231464s. Full raw sensor/additional RGB/model/actual-command/state
audits pass; physical/public/prospective1873frame prefix exact. The settled
outbound arrival1868 is now independently audited in this episode as well:
maximum goal distance0.037619440722m and speed0.039015776111m/s during the
one-second window. This repeats the same physical prefix on the same layout;
it is not independent-layout evidence. Two completed episodes now verify that
settled outbound arrival, zero complete returns. No return edge crossed,
strict visibility stillFalse, hard measurement failed frames[]; no navigation
qualification. Eleven native episodes are complete.

Readout90363 is running with this native result hash. Floor-transport full
prefix preflight30809 is also running; admit all65diagnosis bindings against
the final native artifact map before launching the fresh controller replay.
Independent geometry86294 CLOSED pass: all6514current auxiliary candidates fit
the transported reference within4.629786670e-5m(max)/1.803456171e-5m(RMS), below
3mm. Correction norm0.014757592402m/angle0.004207392595rad. No controller replay,
nativepose or following old observation used. Full prefix intervention remains
unproven. The bounded renderer probe preflight has also been requested after
native completion; record its returned handle and inspect actual resources.

Floor-transport preflight30809 CLOSED pass:1626sources and all65diagnosis
bindings matched the final native artifact map.80,483,913,728bytes availableRAM,
104,753,135,616artifact-free,3.7%CPUbusy. Replay90487 now running, PID2441811
confirmed Rsl/96.7%CPU, still in final launch admission at last check. Follow
the same handle through launch and0..1904; do not edit candidate sources.
Readout90363/PID2441592 remains live Rsl/99.1%CPU; no final output yet.

Renderer probe preflight4298 CLOSED pass:1631sources,80,723,902,464bytes RAM,
104,740,368,384artifact-free,3.6%CPUbusy, no live native scene. One bounded
32GiB scene plus independent8GiB replay and8GiB readout fits the observed
resource headroom. No scientific parallelism within an episode. The actual
three-observation renderer probe execution has been requested with completed
native44710966178a57b31f7da3bec10ad4f21710bcff3701b0a1038750f1ef6d747c;
follow its returned handle and preserve any failure. It grants no new maze
navigation result or historical raster error bound.

Replay90487 LAUNCHED, launch
3c2cb776dca96f31c80077799b6b397597d73554fee5a76d31b56a34bf433699,
root go2_measured_floor_transport_prefix_v1_attempt_001. All1626 bound sources
are frozen. Frame0 complete-decision/public-array comparison passed. Continue
samePID2441811/session90487, no restart from quiet output. Native renderer
probe session58428 is live in its own admission/startup; record its launch
identity when returned. Readout90363 still live. No new verified round trip.

Readout90363 CLOSED exit0, result
6fbbf9a63ca770317fc929c49796993ff3cf9b268cb4572d2d39a7bfd660d5db,
launch42a54ce75e58d37bea443f26ca87756ba92cb6c2ad5f6d0eb9a2e602ca7fa8cf,
1618frozen sources. Path9.027919650m/191.4simulated seconds, mean/max admitted
registeredXYZerror0.004210428/0.008581741m, median receipt-inclusive iteration
1263.296772ms with all1915over100ms. Auxiliary selected at1872,1873,1877,1878,
1879,1903,1904; seven current measured uses, no latched snapshots counted.
Native and readout report:
docs/go2_dual_camera_settled_maze_pilot_result_2026-09-09.md.
Only replay90487/PID2441811 and bounded renderer58428/PID2441952 remain live;
do not poll closed native67399 or readout90363.

Renderer58428 CLOSED exit0, launch
1158e27563615dafa02d3830a11c0bb8124cf05c473801fa50e6a7e6783950c4,
1631frozen sources; result
5107b9a48c9767149b0f1a217875065f6e98489fe97885626cca9a0ace547f88,
wall63.929525689s. Three original zero-command startup observations/900physics
samples, full raw sensor/model/actual-command audit and original physical/
public/raw-capture/complete-decision comparison pass. All six actual camera
context endpoints match their raw acquisition witnesses, no query failures.
This proves startup recorder integration, not historical context or raster
error bounds. No new navigation episode. Report:
docs/go2_maze_renderer_witness_probe_result_2026-09-09.md.

Only floor-transport replay90487/PID2441811 remains LIVE, latest reported
frame100 without mismatch; PID confirmed Rsl/99.5%CPU,4m09s elapsed/4m08sCPU,
RSS2,145,864KiB. Native/readout/renderer jobs are closed. Continue the same
replay through1904 and final model/input/source checks. While it runs, prepare
the separate prospective floor-transport collector, full raw audit, physical/
public/complete-prefix comparison, launcher and readout; these native successor
files have not been written. The renderer recorder is now independently tested
for optional explicit integration in that future collector. No running or
frozen sources may be edited. Still zero verified round trips; goal active.

## Prospective floor-transport native collector, audit and readout prepared

This continuation makes implementation/test progress; preceding turn completed
the eleventh audit/readout, floor diagnosis, renderer probe and launched the
causal replay, so it also made progress. Full goal remains active/unachieved.
Replay90487/PID2441811 was confirmed live at turn start, later reached1000
without mismatch. Continue the same handle through1904 and final checks.

Prepared separate native collector/audit, saved intervention admission,
physical/public/prospective prefix comparison, launcher and typed readout.
The collector uses the same full episode with MeasuredFloorTransportController
and the tested RendererWitnessDualCameraMazeSession. Original physics, sensing,
map/contact/selector algorithms, gains/friction, budget, raw audit and strict
visibility/outcome logic remain. Full acquisition provenance is audited in
addition; no historical context/error-bound claim. Native admission requires
the actual completed current prefix and prior native/readout/renderer hashes.

Tests59046 CLOSED18passed5.25s;58117 CLOSED19passed6.11s.37new checks cover
exact saved intervention, old1903anchor/current1904pose, causal boundedstream,
renderer prerequisite, original collector/raw audit scope, all95950physical
prefix samples through intervention, paired public/complete decisions, actual
commands, and typed transport readout/missingness without forged currentplanes.
CLI5672/22681 CLOSED pass; source24961 CLOSED pass,1658combined preparedpaths,
all1626live-prefix/1631renderer/1618readout bindings unchanged. No new native
output root or preflight created. Preparation:
docs/go2_measured_floor_transport_native_preparation_2026-09-09.md.

After90487completes, run scripts/run_go2_measured_floor_transport_maze_pilot_v1.py
with its actual --prefix-result-sha256 and --preflight-only. Inspect current
hardware/competition and retain32GiB RAM,10GiB collection+1GiB persistence
above40GiBreserve. Then one fresh scene if admission passes; future root
go2_measured_floor_transport_maze_pilot_v1_attempt_001. Dedicated readout:
scripts/read_go2_measured_floor_transport_maze_pilot_v1.py with actual completed
native --native-result-sha256. Future root
go2_measured_floor_transport_maze_readout_v1_attempt_001. No model training or
new independent layout; still zero verified round trips and no qualification.

Verified wait continuation: replay90487/PID2441811 remains Rsl/99.8%CPU,
25m17s elapsed/25m15s CPU, RSS5,562,200KiB, latestframe1400without mismatch.
No result/failure/mismatch artifact was present at the preceding check. No
restart, source edit, native preflight or new scene. The native preparation
turn was progress; this turn is a verified wait on the same active process.
Follow90487through1904and final checks, then use the actual completed result
SHA for the prepared native launcher's --preflight-only admission. Goal active.

## Floor-transport complete controller prefix passed; native preflight running

Prior goal turn was a verified wait on live90487/PID2441811. This continuation
completed that replay and changes the next action, so it makes progress.
90487 CLOSED exit0, result
5e0d7d578203f9f0069804dbec422f8b74284b8a73809a54478b204a997fd454,
wall2054.033175562s.1626frozen sources. All1905observations processed; every
complete pre1904decision exact outside validated labels and all1904prioractual
commands exact. Current raw visual evidence at1904is exactly original; explicit
transport from saved1903anchor admits currentpose, remains active RETURN with
command[0,0,-.45], no terminal/failure. Model/input/source checks pass and all65
diagnosis bindings match final eleventh native artifacts. No later old
observation consumed. Stream16031895df0e3586205aa8cb3e88d17dd24986195355ea6e0492c89fc70ec1d5,
intervention4a2b632fb62488bf835f88ff0752a7d1f8d27cd1fa10c56e763e648da65c9eac.
Report docs/go2_measured_floor_transport_prefix_result_2026-09-09.md.

Native preflight27530 is now running the prepared launcher with this actual
completed --prefix-result-sha256 and --preflight-only. It must admit the entire
saved stream/summary/intervention, completed eleventh/native/readout and actual
renderer probe, model/runtime/source bindings and refreshed hardware before a
fresh scene. No new native output yet. If all checks pass, execute the same
launcher without --preflight-only, preserve its returned handle, and follow
collection/fullrawaudit/physical-public-prospective prefix to completion.
Goal remains active with zero verified round trips; no blocker asserted.

## Native floor-transport preflight passed; execution command started

Preflight27530 CLOSED exit0: 1654 source bindings and completed inputs verified,
memory/storage admission passed. Available RAM81,482,600,448bytes; artifact
free104,373,059,584bytes; workspace free21,360,701,440bytes. CPU0.5%busy,
16physical/32logical CPUs; only competing Python PID6681 RSS2,674,688bytes.
GPU card0/card1 busy0%/8%, VRAM used72,568,832/1,392,443,392bytes.
No output or native execution was created by preflight.

Started the same command without --preflight-only using completed prefix SHA
5e0d7d578203f9f0069804dbec422f8b74284b8a73809a54478b204a997fd454.
Execution session19976 is live, initially quiet during repeated admission;
native launch marker and collection have not yet been observed. Continue this
handle; do not restart or edit bound sources. One CPU scene/thread requested.
Eleven completed navigation episodes, two settled outbound arrivals on reused
maze0, zero verified round trips. Remaining: complete this native attempt and
readout, resolve strict visibility failure909 and return failures as evidence
requires, execute independent layouts and matched contribution comparisons,
and address receipt-inclusive control latency above100ms. Goal active.

## Floor-transport native collecting; fixed independent-layout cohort prepared

Previous turn made progress by passing preflight and starting19976. This turn
confirmed the actual native launch and collection and implemented/tested the
next three-layout execution, so it also makes progress. No restart occurred.
Native launch8dbd36ce4c300bef2b42b31cb6c04d5633624163684f1a5e88de5fcbb8002369,
1654frozen sources. Session19976 parent2446755 worker2447506; worker confirmed
live98.6%CPU and242completed timingrows through241. No result/failure yet.
Continue this exact handle through collection/fullrawaudit/prefix/final checks.

New scripts/run_go2_independent_floor_transport_mazes_v1.py fixes layouts1,2,3
in order with unchanged current controller/model and fresh process/memory per
case. Uses existing native collector/raw audit; only experiment-scope metadata
changes. Requires the actual completed current native result, including full
raw audit/prefix validity, but does not require scientific success. Scientific
failures stay in the three-case denominator; infrastructure/resource failure
stops and retains partial evidence without retry or skipping cases. Original
3000sharedticks/physics/gates/renderer witnesses retained. One native scene at
a time;32GiBRAM and40GiBreserve+11GiBper remaining case, refreshed admissions.

Tests12167 CLOSED30pass/2fail exposed exclusive progress filename reuse; fixed
only new unlaunched runner to immutable per-case progress snapshots.9277 CLOSED
32passed2.63s;CLI9048 CLOSEDpass;source29649 CLOSED1658preparedpaths with all
1654live native bindings unchanged. No independent output/preflight/execution.
Details and four new source identities:
docs/go2_independent_floor_transport_study_preparation_2026-09-09.md.
After current native completion and dedicated readout, preflight this new
runner with the actual --native-result-sha256, inspect hardware and absence of
other native scenes, then execute the fixed cohort if admitted. Do not require
the known unresolved strict visibility/return result to be positive before
gathering the fixed independent-layout evidence. Goal remains active/unachieved;
eleven completed navigation episodes, zero verified round trips, no blocker.

Verified wait continuation:19976remains live. Worker2447506 confirmed
Rl/98.4%CPU,10m12s elapsed/10m03s CPU,RSS3,848,040KiB;460completed timingrows
through459, no terminal result/failure file. Previous turn made progress by
launching collection and preparing the independent cohort; this turn is a
verified wait, not a new experiment or a completed native result. No source
changes, restart or second scene. Continue19976through1904intervention and
the resulting fresh trajectory, fullrawaudit and final input/source checks.

## Shared-current-state reactive comparator integrated on actual native packets

Previous turn was a verified wait on19976. This continuation makes progress:
implemented separate ReactiveFloorTransportController,15focused checks passed,
and completed an actual recorded-packet causal replay without a new scene.
The older reactive pilots used older observer/map/mission rules, so cannot
substitute for this comparator's future native evidence. New comparator shares
current paired-camera motion, measured floor transport, map/contact memory and
settled-boundary mission; retains existing ReactiveConnectorRouteSelector.
No learned model, forecast residual or candidate future outcomes. This is a
method-level comparator, not an isolated ranking ablation or equal future gates.

Tests52285 CLOSED4pass3.86s;14185 CLOSED11pass2.06s. Preflight44746 CLOSEDpass,
1666sources,76.5GBRAM/101.5GBartifactfree and CPU3.3%busy. Replay74127 CLOSED,
result71a5ecd8486d6d8354762c5dc249307cc7d1bf7dc57fe6d1f2df76372d5aaba9,
launchb4dc1fb36ed790ad66094c6458e212fd2496e462c75e2e34184bbd608f845d57,
stream2f34b5424addae309150ff76c8cf9f87ba3de79632d20da5b082b9f62aee9fc4.
17.509519994s,all1666source/input bindings verified before/after. Four native
observations0..3 match raw/admitted pose,map,partition,distance and mission
outside validated pose-source wording. Three earlier actualcommands match.
At3reactive requests[.2,0,0]vslearned[.16,0,.45], both active. Stops before
the following old observation; no unexecuted outcome inferred. New source,
protocol and tests are now frozen by the replay launch. Details:
docs/go2_reactive_floor_transport_prefix_result_2026-09-09.md.

No reactive native collector/audit/launcher prepared yet. It requires a fresh
scene and its own full raw audit/commonprefix through3(900physics samples),
then actual new command outcomes. Keep existing learned independent cohort
fixed; complete native19976and its readout first, then run the prepared
independent-layout preflight and admitted cohort as previously ordered.
Native19976 worker2447506confirmedlive98.5%CPU,21m33s elapsed/21m13s CPU,
RSS5,679,376KiB;1052timingrows through1051,no result/failure. No restart or
running/frozen source edit. Goal active/unachieved,zero verified round trips.

## Reactive current-state native pilot prepared; learned collection remains live

Previous turn made progress through completed actual-packet reactive prefix.
This turn also makes progress: prepared separate reactive collector, full raw
audit, exact physical/public/prospective prefix comparison and native launcher.
Full current learned sensing/physics/geometry/mission/gates/renderer witnesses
retained; reactive controller has no high-level model/residual or predicted
candidate gates. Actual prefix admission requires all four saved decisions.
Fresh native comparison requires all900physics samples/four paired packets,
exact current native shared state and every candidate decision. Later physical
outcomes may differ; no borrowed outcomes or fake completed intervention.

Tests7080 CLOSED25passed2.59s;CLI16307 CLOSEDpass;preparation80135 CLOSED
1675sourcebindings,all1654live native/1666reactive-prefix bindings unchanged,
actual completed four-decision prefix admitted. No reactive native output,
preflight or scene. Full preparation and six new source identities:
docs/go2_reactive_floor_transport_native_preparation_2026-09-09.md.
Runner:scripts/run_go2_reactive_floor_transport_maze_pilot_v1.py, actual
--learned-result-sha256 required. Order unchanged: finish19976and readout,
execute admitted fixed learned layouts1,2,3 cohort, then preflight/execute
this reactive maze0 pilot after current hardware/competition assessment.
Reactive results/readout and cross-method comparison still unexecuted.

Native19976worker2447506confirmedliveRl/98.5%CPU,34m04s elapsed/33m35s CPU,
RSS7,397,228KiB;1613completed timingrows through1612,no result/failure file.
Continue samehandle through1904intervention, actual subsequent trajectory,
rawaudit/prefix and finalbindings. Goal active/unachieved;eleven completed maze
episodes,zero verified round trips,no blocker and no restart.

## Live floor-transport intervention passed the earlier stop; floor reacquired

Previous turn made progress by completing reactive native preparation. This
continuation is a verified wait on live19976/PID2447506, with provisional closed
decision inspection. No new scene, model load, source change or restart.
Worker confirmed Rl/98.5%CPU,43m06s elapsed/42m30s CPU,RSS8,635,048KiB;
2018completed timingrows through2017. No case collection result, root result or
failure file. Native collection continues; fullrawaudit has not completed.

Read-only inspection34564 CLOSED used only completed live rows through1907.
At1868 the controller reports outbound arrival and RETURN, not yet independently
reverified on this new run. At1903 original floor registration is admitted.
At1904..1907 the current schema is measured_visual_floor_transport_evidence,
anchor ages1..4, all decisions active RETURN with requested[0,0,-.45], no
controller failure. At1904 position[3.877104248806246,-1.3015972181875413,
.01973057004560251] matches the prospective witness numerically, pending the
complete physical/public/decision prefix audit. No unexecuted outcome was used.

Read-only inspection72645 CLOSED used only completed live rows through2017.
Original full floor registration has resumed by1950;1950,2000,2017 remain
active RETURN without failure.1950/2000 request[.16,0,.45],2017requests[0,0,.45].
Observed2017position[3.780128967565861,-1.0054268209776296,.031525193717499254],
observed home distance3.9115544357418477m. These are provisional controller
records, not independently audited native displacement, route retracing,
settled arrival or a verified round trip. All original outcomes/visibility
failures remain preserved. Continue same19976through actual terminal collection,
fullrawaudit, exact prefix and finalbindings, then the previously ordered
readout/independent-layout cohort/reactive pilot. Goal active/unachieved.

## Paired reactive readout prepared; return remains incomplete

Prepared the current learned/reactive paired physical outcome readout with
exact launch matching, native/shared-prefix admission, preserved negative
outcomes and no isolated JEPA/planning/memory claims. Seventeen focused tests
passed in0.12s;CLI29504 and source preparation52270 closed successfully.
1697 prepared readout sources; all1666 frozen reactive-prefix sources,
including1654 live native sources, unchanged. No reactive native/readout
execution. Full source identities and scope are recorded in
docs/go2_reactive_floor_transport_readout_preparation_2026-09-09.md.

Status verification: native19976 worker2447506 remained active98.6%CPU after
60m19s, RSS10,931,828KiB. Collection subsequently had2801completed timingrows
through2800; no case result, root result or root failure. Continue unchanged.
At completed decision2410 tracking was valid, phaseRETURN, requestedhold,
observed home distance3.8194315376076102m. Waypoint was approximately0.385m
ahead. Provisional scoring inspection suggests a short executed-progress versus
full-plan contact-penalty issue; candidate masks remain to be checked and no
formal causal diagnosis is claimed. Do not change the running controller or
the fixed independent-layout cohort based on this provisional observation.

Goal remains active/unachieved. Eleven completed development-maze episodes,
two independently verified settled outbound arrivals, zero verified round
trips; twelfth episode still running. Independent layouts1–3 and the current
reactive native comparison remain unexecuted. Full raw audit, strict visibility,
return reliability, independent generalization, contribution comparisons,
100ms timing and bounded real-platform validation remain outstanding.

## Return hold selection masks inspected while collection continues

Previous goal turn was a verified wait with preparation bookkeeping. This turn
adds bounded decision evidence: inspection16462 closed successfully after
reading at most2891completed receipt lines and parsing2410/2890. At both ticks
RETURN is active, hold is selected, and there is no failure, mission hold or
infeasible-action wait. Five actions pass full-path admission. Left arc's
higher utility is excluded by later predicted segments crossing the0.45m
nominal radius; forward is admissible but scores below hold because the score
subtracts800ms contact cost from100ms executed waypoint benefit. Observed home
distance changes only0.00056417m between the two inspected endpoints48simulated
seconds apart. This explains those decisions, not unexecuted physical outcomes.
Detailed scalar evidence and limitations:
docs/go2_floor_transport_live_return_selection_diagnosis_2026-09-09.md.

Native19976/PID2447506 confirmedliveRl98.6%CPU at62m32s, CPU61m40s,
RSS11,243,688KiB. No restart, running-source modification or new scene. Full
collection/audit and prefix verification remain pending. Goal active;zero
verified round trips. Fixed independent-layout cohort and reactive comparison
remain next after the current completed result/readout.

## Current collection ended at mission budget; raw audit still pending

The same native19976 collection has now completed3014decision receipts,
3013completed commands,151400physics samples and3014paired camera frames.
Case resultstatusMEASURED_FLOOR_TRANSPORT_MAZE_TERMINAL_AUDIT_REQUIRED,
SHA2560153c24c6974fb5f8df736ca0a24874766b3ef93b552ad020af56242296bb8cc.
There was no reported physical stop or acquisition stop. Schedule terminal is
MISSION_TICK_BUDGET_EXHAUSTED with all10terminal zero commands completed.
Final mission receipt is frame3003, phaseRETURN, one observed arrival,
observed home distance3.8187576601079534m, no mission failure and no verified
round trip. This is a collection result, not an independently audited outcome.

Worker2447506remainsliveRl98.6%CPU at65m28s, CPU64m33s,RSS10,567,132KiB.
Parent2446755and session19976remainactive. No root result/failure, case audit
or worker terminal file exists at this check. Persistence/fullrawreplay and
exact physical/public/prospective prefix validation must finish on this same
handle; do not restart or launch a dependent experiment yet.

The provisional two-decision selection diagnosis has SHA256
326fb3cc911d6c53b62a8b5ca791ae840d070b1f9db17bef15cc11b9739bf860.
Previous11episodes remain completed/audited; this12thcollection is complete
but its audit is pending. Goal active/unachieved,zero verified round trips.

## Fixed reactive layouts1–3 cohort prepared before independent outcomes

Previous goal turn made progress by explaining the recorded hold ranking and
observing terminal collection. This turn prepared the matching reactive
independent-layout cohort without changing any existing attempt. Fixed layouts
1,2,3 and order; same current reactive collector/raw audit, sensing, gait,
settling mission and persistent state semantics, with fresh memory/process per
case. Requires completed fixed learned cohort and reactive maze0 pilot; admits
scientific negatives but rejects incomplete/raw-invalid evidence. Preserves
all negative/absent-return outcomes, pairs native evaluations and retains any
partial infrastructure failure without automatic retry or case omission.

Tests27103closed27pass2.69s;CLI59064closedpass;source preparation23527closedpass:
1682prepared independent reactive sources,all1654live native and1666reactive
prefix frozen sources unchanged. Independent learned, reactive maze0 and new
independent reactive output roots all absent. No new scene or model execution.
Source identities, coverage and protocol:
docs/go2_independent_reactive_floor_transport_study_preparation_2026-09-09.md.

Native19976worker2447506confirmedliveRl98.7%CPU,72m52s elapsed/71m57sCPU,
RSS10,646,680KiB. Collection remains complete at MISSION_TICK_BUDGET_EXHAUSTED;
full audit/prefix and final result are pending. No root failure, case audit or
worker terminal. Continue same handle unchanged. After current audit/readout,
retain order: learned layouts1–3, reactive maze0, paired maze0 readout, then new
reactive layouts1–3 cohort. Refresh hardware assessment before substantial jobs.
Goal active/unachieved,zero verified round trips;no blocker.

## Persistent-memory setting audited; no current memory ablation exists

Previous goal turn made progress by preparing/testing the fixed reactive
independent cohort. This turn traced the current controller memory contract
and verified the constructor behavior. Runtime inspection8078 and direct
constructor check8986 closed successfully: persistent=False is rejected by
the inherited MatchedModelGoalProbe guard, while persistent=True constructs
an empty current controller without a model or scene. The mission selector
also hardcodes persistent surface filtering. Planning floor/occupied cells,
primary/auxiliary indices/partitions, retained patches and later floor evidence
are separate accumulated state. A nominal flag toggle is not a memory ablation.

Source identities, concrete state consumers and the requirements for a
separately scoped planning-map comparison are recorded in
docs/go2_floor_transport_memory_ablation_scope_audit_2026-09-09.md.
No memory ablation is implemented/executed or counted as evidence. In particular,
first-witness frame cannot identify current-view cells because insertion uses
setdefault. Existing running/frozen methods and queued cohorts remain unchanged.

Native19976worker2447506confirmedliveRl98.8%CPU at77m51s/76m56sCPU,
RSS10,646,680KiB; full audit remains pending. Hardware51478closed:CPU3.4%busy,
72.108GBRAMavailable,95.329GBartifactfree,21.360GBworkspacefree,bothGPUs0%busy.
No restart or new native job. Continue current audit and the established
readout/learned-cohort/reactive-pilot/readout/reactive-cohort order. Goal active,
unachieved,zero verified round trips;no blocker.

## Current-observation planning-map component implemented and tested

Previous goal turn made progress by proving persistent=False is unavailable
and identifying separate memory consumers. This turn implements the separately
named CurrentObservationPlanningController/Map/Selector component. It supplies
current paired floor/obstacle cells to the existing complete planning selector
chain while retaining all accumulated contact evidence, tracking/floor anchor,
learned inputs/residuals, mission/settling and scan state. Original observation
receipts remain unchanged; first-witness-frame filtering is not used. This is
an ablation of accumulated planning-cell queries, not a memoryless controller.

Final focused tests74597closed12pass4.28s, including synthetic temporary floor
loss with unchanged complete baseline fields and articulated contact queries,
route sensitivity, current re-observed membership, no historical-floor fallback,
stale/failed rejection and complete selector-chain routing/contact dispatch.
Source preparation81669closed1658paths,all1654live sources unchanged.
Preparation, source identities and remaining prospective steps:
docs/go2_current_observation_planning_component_preparation_2026-09-09.md.
No recorded native prefix, new scene, checkpoint load or ablation outcome.

Native19976worker2447506confirmedliveRl98.9%CPU at87m27s/86m32sCPU,
RSS10,649,388KiB. Raw audit/prefix remains pending after budget-terminal
collection. No root result/failure, case audit or worker terminal. Continue
samehandle without restart; existing experiment/readout/cohort order unchanged.
Goal active/unachieved,zero verified round trips;no blocker.

## Actual-packet planning-map intervention passed at frame10

Previous goal turn implemented/tested the current-observation planning component.
This turn prepared/tested its causal replay and completed actual execution of
that replay. Tests58496closed15pass2.03s;hardware19269closed with72.200GBRAM,
95.332GBartifactfree,CPU3.5%busy,bothGPUs0%. Preflight35257closedpass1670sources,
71.629GBRAMavailable. Replay79952closedpass23.671889048069715s.

Eleven actual paired observations0–10; ten prior actual requests exact. At10
the candidate requestsleft_arc[.16,0,.45] versus originalleft_turn[0,0,.45].
Shared raw/admitted visual,map,auxiliary partition,mission/settling and executed
residual state match outside validated predecessor wording. All8compared raw
forecast banks match; public arrays and model weights unchanged. No following
old observation/decision consumed and no physical outcome inferred. Planning
uses1579floor/75occupied current cells versus2036/90retained cells; waypointX
changes.475to.525m. This is a verified decision intervention, not memory benefit.

Result8d0e1391f95c71cd71394356c902a9367a1d37fad75571197456dd0b0a6ca90a;
launchf2ccd1cc505c8b076f7006fdc96ee30e6fc2d997dfea97d05e911477aa039735;
streamd8a7e1b357e835a55a5855a1ac1ccc82b3c263d8cd98100e9d2a6a051666a735.
All1670source/input bindings verified before/after; component and replay sources
are now frozen. Full identities and actual scalar comparison:
docs/go2_current_observation_planning_prefix_result_2026-09-09.md.

Native19976worker2447506stillliveRl99.0%CPU at95m43s/94m48sCPU,
RSS10,664,380KiB. Full audit/prefix pending; no restart or new native scene.
Existing readout/independent learned/reactive execution order unchanged. This
memory comparator still needs separate native collection/audit/prefix support
and fresh physical evidence. Goal active/unachieved;zero verified round trips.

## Planning-map native comparison prepared; original audit remains active

Previous goal turn made progress by completing the actual-packet planning-map
prefix at10. This turn prepared its separate native collector, full raw audit,
1250sample/11packet/full-prospective-decision prefix checker and native launcher.
Original physics and evaluation calculations are unchanged outside the declared
controller/metadata substitution. Same assigned model for collection and a fresh
identical model for replay. Prior collection/audit evidence remains retained if
a later check fails. No new native launch or outcome.

Tests6876closed25pass2.57s;CLI85267closedpass;source preparation59039closed
1678prepared paths,all1670frozen planning-prefix sources including1654live
baseline sources unchanged. Actual completed11decision prefix admitted with
first changed command10. Output root absent. Full source identities and checks:
docs/go2_current_observation_planning_native_preparation_2026-09-09.md.

Native19976worker2447506confirmedliveRl99.1%CPU at106m25s/105m30sCPU,
RSS10,683,760KiB. No root result/failure, case audit or worker terminal; same
ongoing full audit after budget-terminal collection. Do not restart. Existing
readout/learned-cohort/reactive-pilot/readout/reactive-cohort sequence remains;
this memory pilot follows, with current hardware assessment before execution.
Goal active/unachieved,zero verified round trips;no blocker.

## Verified wait on the original raw audit

Previous goal turn made progress by completing native planning-memory
preparation and25focused tests. This turn is a verified wait on native19976,
parent2446755and worker2447506. The same session remains active with no new
terminal output. Worker CPU time advanced from109m10s to110m01s between
authoritative process checks; final elapsed110m57s,Rl99.1%CPU,RSS10,685,004KiB.
No saved case audit, worker terminal, root result or root failure exists.

The terminal collection result was rehashed and remains exactly
0153c24c6974fb5f8df736ca0a24874766b3ef93b552ad020af56242296bb8cc:
3014decisions,MISSION_TICK_BUDGET_EXHAUSTED,no physical/acquisition stop.
No source change, restart, new scene or dependent experiment. Keep the current
process and established experiment order. Goal remains active/unachieved,
zero verified round trips;this live prerequisite is not a blocker.

## Paired planning-memory physical outcome readout prepared

Previous goal turn was a verified wait. This turn prepared the paired native
planning-memory readout. Exact scene/mission/runtime/budget/model and full
1250sample/11packet/eight-forecast prefix admission; retained contact/localization/
prediction/mission/scan state explicit. Original actual-motion/evaluation
calculations preserved; candidate traces add current planning-cell counts.
Physical candidates, strict visibility failures, absent returns and all original
timings remain separate. No controller selection or memory-benefit claim.

20focused tests passed0.12s;CLI54515closedpass;source preparation18788closed
1688prepared paired readout paths,all1670frozen prefix sources including1654live
baseline sources unchanged. Planning native/readout outputs absent. No model,
scene or completed readout execution. Full identities and scope:
docs/go2_planning_memory_readout_preparation_2026-09-09.md.

Native19976worker2447506confirmedliveRl99.1%CPU at116m09s/115m13sCPU,
RSS10,686,252KiB. No case audit, worker terminal, root result or root failure.
Continue same full audit and established sequence; append this readout after
the eventual planning-memory native pilot. Goal active/unachieved,zero verified
round trips;no blocker.

## Audit wait and next-cohort admission review

The preceding status turn was a verified wait: process2447506 was confirmed
running, with no terminal audit or root result. This continuation re-polled
the same session19976 twice; both polls returned the same live session with
no new terminal output. Worker CPU time advanced from2h03m55s to2h04m34s;
final elapsed2h05m30s, Rl99.2%CPU, RSS10,963,528KiB. No case audit, worker
terminal, root result or root failure exists. Collection remains budget-terminal
with3014decisions,3013completed commands,151400physics samples and10zero drain.

Re-read the current readout, fixed-layout cohort launcher and admission helper.
The next cohort requires the completed predecessor's raw sensor/model/command
audit and exact prospective prefix, but correctly retains scientific negatives
without requiring a successful return or strict visibility pass. It fixes
layouts1,2,3, refreshes resource admission after validation and before each
case, and stops on infrastructure failure while preserving partial evidence.
No source change, native restart, new scene or dependent execution occurred.
Custody instructions hash remains unchanged. Continue the original audit,
then its readout and the established fixed cohort/comparison sequence with
a fresh hardware assessment. Goal active/unachieved; zero verified round trips.

## Twelfth episode raw audit saved; final prefix remains active

Previous turn was a verified wait. This turn observed and inspected the newly
saved full raw audit449d23341e1449754eaf698757be2b2553cefdcd4345d77e372ac03cb66ea368
(10,754,355bytes). Sensor reconstruction, model/controller/command replay,
command audit and model-state checks pass. Outbound native one-second arrival
at1868passes with the same3.761944cm maximum distance and3.901578cm/s maximum
full-3D speed as the two preceding maze0 runs. This is now a third audited
outbound arrival on that same prefix, not an independent-layout success.
There is no home arrival, physical retracing, terminal quiet or native
round-trip candidate pass. Strict primary visibility first fails at909.

Unlike the preceding run, new hard measurement failures are present at1924,
1925,1930: one stable-interior primary ray each exceeds the unchanged1mm gate
(maximum errors1.005780,1.057865,1.051541mm). Auxiliary visibility passes at
all three. Do not collapse these into the earlier boundary-only failure or
relax the metric gate. Full inspected values, artifact identity, scope and
fresh hardware assessment are recorded in
docs/go2_measured_floor_transport_saved_raw_audit_2026-09-09.md.

Same session19976remains active. Latest worker2447506elapsed2h09m23s,
CPU2h08m26s,Rl99.2%,RSS10,666,020KiB. Prefix comparison, worker terminal,
root result and root failure remain absent. The raw audit is saved; final
prospective prefix and bindings must finish before dependent admission.
No restart or new native execution. Continue the established sequence.
Goal remains active/unachieved with zero verified round trips.

## Native worker completed; parent final verification remains active

Previous goal turn produced the saved raw audit. This turn observed the
successful prefix comparison99b7ff1a75e5c3a87731c809cf9ce9ad111adfd9e20181eea37d584e706ec5ec:
95,950physical samples and1,905paired public observations exact through1904;
all earlier requests and complete prospective candidate decisions match.
Worker terminal493df96dd733283c1d09682c0cb503671440577201e1431016506cc1cc10b32d
reports MEASURED_FLOOR_TRANSPORT_MAZE_COLLECTED_AND_RAW_AUDITED, no failure,
7994.413019503001s wall and11,952,197,632bytes peakRSS. All scientific failures
remain unchanged. Full identities appended to the saved raw audit report.

Session19976 emitted the successful worker terminal message and remains live.
Worker2447506 has exited; parent2446755 is Rsl performing final verification,
elapsed2h18m09s/CPU4m55s at inspection. Root result/failure remain absent.
Do not confuse the exited worker with a failed or completed parent attempt;
continue polling the same session. Next action remains completed-result
readout followed by fixed independent learned-layout preflight/execution.
No new scene, source change or restart. Goal active/unachieved;zero round trips.

## Original native attempt complete; readout and cohort preflight launched

Previous turn completed the worker/prefix checks. This turn session19976
closedexit0 with final rootresult
1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374,
statusMEASURED_FLOOR_TRANSPORT_MAZE_PILOT_V1_COMPLETE, wall8139.471536834026s,
1654source bindings and18123artifacts. Twelve navigation episodes are now
completed/audited;three settled outbound arrivals on the same maze0prefix,
zero verified round trips. Strict909and new hard1924/1925/1930failures remain.
Final native report:docs/go2_measured_floor_transport_maze_pilot_result_2026-09-09.md.

Hardware2507closed:83,455,864,832bytes availableRAM,95,288,840,192artifactfree,
21,360,197,632workspacefree,all32logical affinity/16physical cores,0.3%CPU,
bothGPUs0%,no competing native job. Launched the prepared CPU readout in
session48066 and independent-layout source/input/resource preflight in
session52560, both using the actual final native SHA above. Preflight is
--preflight-only; no new scene or cohort output has been created. The two
independent CPU checks overlap with ample resource headroom. Both sessions
were polled live with no new terminal output. Require their successful
completion, refresh hardware, then launch the fixed learned layouts1,2,3
cohort. Preserve subsequent reactive and planning-memory comparison order.
Goal active/unachieved;no blocker. Do not poll closed native session19976 again.

## Readout complete; fixed independent cohort command started

Previous turn completed native result publication and launched readout/preflight.
This turn readout48066closedexit0 with result
a46e6051b125347804df4aab948e68a6466007952f7529b4f316ae39b114728c,
launch0b824ca244ac6fc2bd6164826e38b0a1489cc8a4501031ff98948b45e30d1c1c,
1658sources verified before/after. Actual path10.304553639084356m over301.3
simulated seconds;terminal native home distance3.820938777565301m. Eight
transported frames1904–1911,2996current-floor poses;mean registered3D error
4.262246mm,max8.581741mm on this executed trajectory. Median receipt-inclusive
iteration1237.033925ms;all3014iterations exceed100ms. Same failed return and
measurement failures;no model, memory or real-time advantage claimed.

Preflight52560closedexit0:1658sources/all completed inputs verified,
fixedlayouts1,2,3,outputnotcreated. Final refresh83,057,709,056bytesRAMavailable,
95,268,552,704artifactfree versus78,383,153,152requiredfree for all3cases;
CPU0.3%,all32logical affinity/16physical cores,bothGPUs0%,no competing native.
Then started actual cohort launcher session11941 with nativeSHA1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374.
CurrentPID2465301running,CPU63.10suser+10.88ssystem,RSS1,205,673,984bytes;
same handle polled live without terminal output. Exclusive cohort output is
still absent during source/input revalidation; no new maze outcome yet.

Detailed readout and launch admission:
docs/go2_measured_floor_transport_maze_readout_result_2026-09-09.md.
Next:continue11941,record launch identity and verify first native worker/startup
once created;monitor fixed1,2,3collection/audit progress,one scene at a time.
Do not repoll closed48066/52560or restart11941. Reactive maze0/readout,
independent reactive cohort and planning-memory pilot/readout remain queued.
Goal active/unachieved;12completed audited episodes,zero verified round trips.

## Independent cohort launch frozen and maze1 worker active

Previous turn completed readout/preflight and started the cohort command.
This turn the same session11941 published INDEPENDENT_FLOOR_TRANSPORT_LAUNCHED
3053ca602d8e45700550188a3da12e69c3b83314af5a74f32bc616c5425b91c9.
The exclusive root now exists with1658source bindings, fixed case order1,2,3,
unchanged model4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6
and explicit admission of the predecessor's unsuccessful round trip and
strict visibility result. Case1admission SHA
ac7b62fc5d0778638ba8805a1e2252172070f1002b880d368296c6403be84315.

Parent2465301 is sleeping with294.39sCPU, tracker2466224sleeping,
worker2466225running with43.10sCPU/RSS1,350,328,320bytes. Worker log exists
and is empty during its final input validation; case1output remains absent.
No new navigation outcome yet. Keep11941 and its same first worker.

Used spare CPU capacity for actual completed-baseline admission checks of
the already prepared reactive and planning-memory pilots. Hardware84326closed
with82,531,508,224bytesRAMavailable,95,272,263,680artifactfree,CPU3.3%,
all32logical affinity/16physical cores,bothGPUs0%,only original cohort launcher
and small background Python process competing. Reviewed both preflight paths:
they return before creating outputs or spawning native workers. Launched
reactive preflight92582/PID2465867 and planning-memory preflight87724/PID2465868
using --learned-result-sha2561597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374
and --preflight-only. Both polled live without terminal output; their native
output roots remain absent. No source change or additional scene.

The cohort's subsequent case1resource check records these competing CPU
preflights:81,085,353,984bytesRAMavailable,95,256,641,536artifactfree,
CPU6.5%,bothGPUs0%; required free78,383,153,152bytes for all3cases. Adequate
headroom remains; one native scene only. These early comparison preflights
check real final baseline bindings but do not move their native executions
ahead of the established cohort/comparison sequence. Reassess resources at
each later launch. Next:monitor11941for actual case1startup/collection, finish
92582and87724, preserve their admissions and all failures. Goal active,
12completed audited episodes,zero verified round trips;no blocker.

Final poll in this turn completed both comparison preflights with exit0:
92582reactive verified1675source bindings;87724planning-memory verified1678.
Both input/source, memory and storage admissions pass, output_createdFalse,
native_executionFalse. Their final refresh records80,101,212,160bytesRAM,
95,255,359,488artifactfree,21,360,177,152workspacefree,CPU3.4%,bothGPUs0%.
The above live-preflight observations describe the earlier checks; both handles
are now closed and must not be repolled. Native11941remains live without new
terminal output. Upcoming reactive and planning-memory launches have now
passed admission against the actual completed baseline, but neither native
comparison has executed. Refresh resources before their queued native runs.

## Maze1 collected a sensor/model stop; objective-pair admission implemented

Previous turn froze the independent launch and completed both comparison
preflights. This turn maze1started and collected225paired observations and
decisions,224completed commands,11,950physics samples and10terminal zero
commands. Collection result SHA
b403ca7dd2d15fced064d798c3cd204641289e7eb78c74607e51c51d89cfaf84,
schedule_terminalSENSOR_OR_MODEL_FAILURE,physical_stopNone,acquisition_stopNone.
Bounded decision inspection55519closed: first failure214,
"same-episode current visual evidence required", requested[0,0,0], evidenceNone.
At213the preceding active decision requestedleft_arc[.16,0,.45]. No arrival
is recorded. This receipt alone does not identify the underlying sensor cause.

The original worker remains in full raw audit. Latest2466225Rl99.0%,
elapsed9m40s/CPU9m34s,RSS2,728,148KiB;parent2465301sleeping. Case audit,
worker terminal, progress_after_01and rootfailure remain absent. Same11941
polled live; no restart. The thirteenth episode is collected, while12remain
completed/audited. If raw validity passes, retain this scientific negative and
continue fixedlayout2; infrastructure failure must stop the partial cohort.

Used independent source work to prepare the remaining JEPA training-objective
comparison. Added lewm/matched_rollout_objective_admission_development.py
and focused test: fixedfirst-seedfullJEPA versusfullsupervised_rollout,
same initialization/data/schedule/configuration and rollout head; each retains
its own identically specified training-only XY correction procedure. Exact
final snapshot identities required; reject changed heads, variants, schedules,
checkpoints, correction provenance/weights and forecast clocks. No checkpoint
loaded or native comparison executed.21tests passed0.11s and authenticated
actual baseline-launch/fit metadata passed the new helper, with input hashes
rechecked. All1658active cohort source bindings remain unchanged.

Implementation identities and reviewed training differences:
docs/go2_matched_rollout_objective_admission_preparation_2026-09-09.md.
This completes a comparison admission component, not a JEPA benefit or new
native outcome. Existing direct-comparator assignment and current cohort/
reactive/planning-memory order remain intact. Next:monitor11941through case1
audit and subsequent case2startup; prepare causal objective-pair replay using
the exact assigned evaluation-only loader. Goal active/unachieved;zero verified
round trips, no blocker.

## First independent case audited; maze2 active; correspondence failure reproduced

Previous turn implemented the objective-pair admission and observed maze1's
terminal collection. This turn maze1completed all raw audit checks and strict
visibility with no hard measurement failures, no arrival and no round trip.
Audit7719c771ef8b5ecdde828b3e70b0839167be6394b38eb8bbf3f16b0396763b05;
workerterminal97d2075639548de17e73f8857703975b57bebadef6497ea8e3fbe2f59496f1dc;
progress_after_01 966e102bbc20bd78e0d3c3ffb90c9e3de00082e42c58ee48ec2d2871ffd1615a.
Worker wall734.5053428560495s,peakRSS3,303,362,560bytes. The same cohort11941
retained the scientific failure and started maze2worker2468174 under admission
3aa9b4e8794e013918bf8b16ddd8f6b0c5555ecf5068b7e87d0682f75b368c39.
Last inspected workerRl98.9%,elapsed4m18s/CPU4m15s,RSS2,843,636KiB;
maze2output exists,collection result absent. Parent2465301remains live;
worker2466225completed and must not be polled as the active worker.

Inspected retained original_visual_evidence at214: primary and auxiliary both
fail all eight references206–213. Added and executed fixed-frame correspondence
diagnosis83838closedexit0,2.9185069389641285s,1659source/1386input bindings
verified before/after. Result15ec380fc03906fef61dbd5c9a8418856c63bdf531337152db7d8ad0ac685a56,
launch9d98467e4887832bbea9d06c158d9bed7e56792f6e6642a82c59fb513a1c3241.
All16recorded camera/reference rejection reasons reproduced exactly.
Primary match counts2,4,5,6,4,9,8,10;auxiliary0,0,1,1,2,3,8,12.
Auxiliary213best initial rigid consensus is11despite12lifted matches and
94validproposals,below required12before iterative pruning. Current214images
have62primary/89auxiliary selected liftable features: correspondence support
loss, not absent images/features. No threshold change, new pose or command.

Full identities, measured resources, actual diagnosis scope and remaining
association-stage questions:
docs/go2_independent_maze01_correspondence_diagnosis_result_2026-09-09.md.
The new diagnostic script is now frozen by its executed launch. Continue11941
through fixed2,3; diagnose matching stages before a prospective tracking change.
Goal active/unachieved;13completed audited episodes including1independent-layout
case,zero verified round trips. No blocker;other comparison order unchanged.

## Matching-stage diagnosis complete; maze2 collected a constraint stop

Previous turn audited maze1 and reproduced all16reference rejections. This
turn added/executed fixed matching-stage instrumentation63231closedexit0,
result79295cc24a0c4fa1f20794bd5f1c01c9be1f15de0699197da44d5e455ff82424,
launchfd93e739567139fe4cf7bec25e43a5d794aabe89c84bfa396433a7c7506d5462.
All64correspondence output arrays byte-exact to the frozen matcher;16final
counts equal the prior diagnosis;1661source/1386case input bindings plus
predecessor identities verified before/after. Work0.8026564470492303s,
oneCPUthread, no model/native execution/pose admission or gate change.

For213to214primary:18mutual→17flow-status→13forward/backward→10descriptor-
location agreement→10lifted. Auxiliary:31→29→25→12→12. Depth lifting removes
none after earlier gates for all16pairs. Main latest-pair loss is the1pixel
descriptor-location agreement gate, but rejected tracks have not been shown
physically correct. A different association method needs prospective geometric,
continuity, raw-prefix and physical validation; do not relax the gate on this
case. Full source identities/resources/results:
docs/go2_independent_maze01_match_stage_diagnosis_result_2026-09-09.md.

Meanwhile maze2collection completed with resultSHA
50f932d68cdc011aacf36de724329dc29c3a674a032827770fe6863b6ce3855c,
514paired observations/decisions,513completed commands,26,400physics samples,
10zero drain, physical/acquisition stopsNone. Terminal is
NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS; bounded scan45946
closed and found no decision failure string. This is a distinct constraint/
action-feasibility stop, not the maze1visual-registration failure. Full raw
audit pending;case2worker terminal/progress_after_02/rootfailure absent.

Same11941active;latest parent2465301sleeping and worker2468174Rl98.5%,
elapsed11m32s/CPU11m22s,RSS3,572,048KiB. Keep this worker until authoritative
terminal. Thirteen episodes completed/audited;fourteenth collected. No new
arrival/success claim,zero verified round trips. Next:complete case2audit,
preserve its feasibility evidence and allow the fixed cohort to proceed to3
if raw-valid; prepare any tracking successor separately. Goal active/no blocker.

## Maze2 raw audit saved; nominal-feasibility stop traced

Previous turn completed matcher stage diagnostics and observed maze2's
constraint stop. This turn inspected the recorded infeasible blocks463,
484–491and493–503. It recovered at492withright_turn before11consecutive
infeasible observations caused the503stop. All6sampled articulated-surface
checks pass at493/503; all6first100msnominal connectors fail against0.45m.
No controller exception string or observed arrival. Goal distance at503is
3.9472608257675637m,stillOUTBOUND.

Saved raw audit71e3aaaa27af0f95d26c3e3a7960094578ca5fdf0e7b0c0231bda1ee5c0a2798
passes raw sensor/model/command/model-state and strict visibility checks;
hard failures and native arrival windows empty. Worker terminal and
progress_after_02remain pending during final binding verification. Same11941
polled live; last worker2468174Rl99.1%,elapsed20m15s/CPU20m05s,
RSS3,574,560KiB. Parent2465301sleeping. No restart or case3launch yet.

The saved proposal gives current clearance0.4526949885562709m at493and
0.4519749401270365m at503, both above0.45. Therefore the existing reentry rule
for an already violated current radius does not activate. Forecast first-step
clearances instead range0.44590–0.44837m at493and0.44641–0.44936m at503.
Online residual correction affects first-step waypoint scoring but not these
original feasibility forecasts: at503hold's corrected scored displacement
is[0.00008232057310602735,-0.000031916234693115525]m versus forecast
[-0.006277943029999733,0.004246499389410019]m before online correction.
No corrected-action physical outcome or longer-horizon error bound is inferred.

Final inspection4194verified decision stream54,219,039bytes,
SHA d8ad4eefbfae066634932c5f3c95ff9420ee8b3db52fe8f9502291d67814f106,
unchanged before/after. Full receipts/source interpretation and requirements
for any separate prospective scoring/feasibility consistency change:
docs/go2_independent_maze02_feasibility_diagnosis_2026-09-09.md.
Continue11941through final case2checks and fixedcase3. Existing baseline
cohort/queue/source remains unchanged. Goal active,zero verified round trips.

## Independent cohort complete; feasibility component tested; reactive starting

Previous status turn was a verified wait: live parent2465301 and maze3worker
2470969 were directly inspected. This turn completed28 synthetic feasibility
tests and observed all three cases plus the parent finish normally. Session11941
closedexit0; do not repoll or restart it. Result
a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720,
1658sources/6200artifacts,3070.694777289871s. All1658sources independently
rehashed unchanged. All three independent cases raw-audited/strict-visible,
zero arrivals and zero round trips. Fifteen total completed/audited episodes.

Maze2terminal1c07b3e815a2c7aab45d87ef700c20fe22352e5651d0771959386ef46e316407;
progress_after02f9db6c70961982d55acf69bb762bf25d81890f155120d34cab4ccb31dd0dc237.
Maze3lost current visual evidence at observation264 after263leftturn;
275paired observations,274commands,14450physics samples,10zero drain.
Collection67d6a2f9e99b592497712ec53d5dff85a5626abb67e9a6cab2f451bb6ff48420;
audit47bc53f3141c65a480fcd2d8e1e01e477c0461920e4f6ba1e8f415401f7c08dd;
workerterminalca6553e0441dfffddc91430444c04591d4c0c506c81a28377e4b9fa7f0005311.
Full scope and identities:
docs/go2_independent_floor_transport_mazes_result_2026-09-09.md.

New separate ResidualFirstIntervalController component preserves raw forecasts
and original surface vetoes, rechecks corrected100ms surface plus all8nominal
segments, selects by original corrected100msutility and remembers raw residual
targets.28testsPASS1.88s, no actual prefix/native use. Full source identities,
test scope and prospective next step:
docs/go2_residual_first_interval_feasibility_preparation_2026-09-09.md.

Refreshed hardware:16physical32logical/all32affinity,CPU4.9%,RAM82.084GB,
GPUsboth0%,artifactfree91.870GB,workspace21.360GB. Only prior cohort final
verification competed at inspection; it then exited normally. Submitted
prepared reactive native runner in session18370 with exact learned result
1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374.
At submission no output yet: input admission may take time, so poll18370 and
confirm launch/worker before calling it native execution. One scene/process,
one OpenCV/BLAS thread; runner refreshes memory/storage before output creation.
Goal active, no blocker, no new training/hardware or deployment claim.

### Reactive launch confirmed; final feasibility test count

Session18370 admitted and printed REACTIVE_FLOOR_TRANSPORT_NATIVE_LAUNCHED.
LaunchSHA f3408677bddb3fdfe4e82be4503b1f67dc23d039bb0524a52b813a51b4c1fce3,
1675frozen sources. Parent2472753 sleeping, resource tracker2473156 and
worker2473157running,55.91CPU seconds/RSS1,353,924,608bytes at inspection.
Case directory absent then: worker input verification still precedes actual
scene collection. Keep this exact session/attempt; do not restart. Refreshed
launch hardware82.542GBavailableRAM,91.871GBartifactfree,21.360GBworkspace,
CPU0.2%,bothGPUs0%,all32CPUaffinity; no prior native job remained.

Final component review added an explicit prohibition on negative source ticks
before episode start and an empty-history no-op test. Final29testsPASS1.85s;
helperSHA41b9e6801d5d9388680284412b133af4c76bcaf02f86df953982dc6cb7d340fb,
controllerSHA316f25c5db917519d07195d0a8cfcdc9a208352dc224fc57ac8385efa069f911,
testsSHA00b13b43802b5b39fe52f2b9d6befcb98f723fd2a0e335218fccb497f0399280.
Preparation document updated to final identities. No frozen sources modified.
Next parallel CPU work may prepare the bound first-command-divergence replay
on completed maze2; script pattern is
scripts/replay_go2_current_observation_planning_prefix_v1.py and comparison
helper lewm/current_observation_planning_prefix_development.py, but use exact
current floor-transport labels and complete raw-selection preservation for
this intervention. Avoid consuming observations after command divergence.
Keep reactive paired-readout/cohort then planning-memory native queue unchanged.

## Feasibility prefix prepared and submitted; reactive collection stopped

Previous goal turn made progress: completed cohort11941,29componenttests and
launched reactive18370. This turn implemented a full actual-maze2 prefix
runner plus exact-state comparator and27testsPASS2.10s. Prepared1666sources
(1658inherited+8new). Submitted session97118,PID2474093; live at initial
inspection29.29CPU seconds/RSS1,350,176,768bytes, input verification underway.
No prefix result or physical improvement claimed. Protocol/source identities,
test coverage, hardware and command are recorded in
docs/go2_residual_first_interval_prefix_preparation_2026-09-09.md.
Treat submitted source paths as fixed; poll97118 and record actual launch,
first command divergence or retained failure. Do not follow changed commands
with the old episode's next observation.

Reactive18370 remains active. Worker2473157 completed scene collection:
284paired observations,283completedcommands,14900physics samples,10zero drain.
Firstterminal273NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY, no exception
string, goal distance2.8197429239327283m, zero request. Physical/acquisition
stopsNone. Collection result560870fbc325d00d113a95c2c62879148012f0c49489a241e01b6434e93accd9.
Decisionstream1d060370f34a9e2990b3d1e112a1b73a129204f68938fdbed5d022b1ee815fb3
rehashed unchanged around bounded terminal inspection. Audit/workerterminal/
rootresult remain pending; do not yet count this as a completed audited episode.
Fifteen completed/audited episodes plus this collected sixteenth; zero verified
round trips. After18370terminal, run prepared paired reactive readout and
independent reactive cohort with actual result identities. Goal active/no blocker.

## Maze3 match-stage diagnosis complete; both active jobs advancing

Previous turn made progress by implementing/testing/submitting prefix97118.
This turn added and executed a fixed maze3 failure diagnosis5722closedexit0,
result995d163abbea7bd63ee9607a6d60052421cbe49044095a52a88745709ad90134,
launch9fea82edfe278fd75c5ea7a2dec6ae8d3dfd266ff664da5f2c781670b6eb7e8d,
1660sources/1686inputs checked before/after,3.866693033836782s.
All64original correspondence arrays exact; all16rejections reproduced.
Latest263→264primary11mutual→8forward/backward→6location/depth;
auxiliary28→24flowstatus→18forward/backward→11location/depth. Depth removed
none after earlier gates. Allpairsbelow12, so no rigid fit/pose admission.
Full source identities/hardware/interpretation:
docs/go2_independent_maze03_match_stage_diagnosis_result_2026-09-09.md.
This supports a separately verified association successor, not weaker gates.

Prefix97118admitted launch64bdc2e18508cc6f1e1386e86ae478156f7a9a192a23398cf0080fa7b8c32283,
1666sources,PID2474093. Latest stdout frame160 with no mismatch/failure.
Admission79.563GBRAM90.926GBartifactfree,CPU3.3%,GPUs0%,all32CPUaffinity.
Same actual maze2/model from start, stop at first command/terminal divergence.

Reactive18370saved raw auditd458a25a710229d4ab12e9736e0dc5e5eb8b9f00859cbedc2c695cf2adb23b92:
rawsensor/controller-command/commandPASS,strictvisibilityTrue,hardfailures[],
noarrivals. Native traversed[-1,0]→[0,0]→[0,-1] via two declared-open edges.
Worker terminal now saved165391c926ba90540ee68b8f7da8a95eae0b10b116905cf1e0fed153ac4701d3,
statusREACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED,wall733.3451112201437s,
peakRSS3,289,071,616bytes. Root completion still pending final input/artifact
verification. Do not restart or launch another native scene until parent
terminal. Sixteenth episode now raw-audited, zero verified round trips.

### Reactive pilot complete; next comparison jobs submitted

18370closedexit0withfinal4d377b9ec202c96099615ca5e3679037d5cb81d30c319afdd7febb9b9f0c9837,
1675sources/1743artifacts,911.3410807489417s. Prefix verification passed4paired
frames/900physics samples and complete candidate decisions at intervention3;
actual changed command completed. Full result/scope/resources recorded in
docs/go2_reactive_floor_transport_maze_pilot_result_2026-09-09.md.
Do not repoll18370or oldworker2473157; both exited normally.

Submitted paired reactive maze0readout59988 with final reactiveSHAabove and
learnedreadouta46e6051b125347804df4aab948e68a6466007952f7529b4f316ae39b114728c.
Submitted fixed independent reactive cohort18182 with same reactivepilotSHA and
learnedcohorta0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720.
These input-verification jobs may overlap: the CPU readout is not a dependency
of cohort admission and cannot select/change its cases; only one native scene
may run. No source/configuration change or different case order. Both handles
currently live with no launch output yet; inspect PID/launchbeforeclaiming
native collection. Refreshed81.019GBRAM90.872GBartifactfree,CPU3.4%,GPUs0%,
all32affinity; only substantive competitor was prefix2474093at2.561GBRSS.
Memory native comparison remains next after the fixed reactive cohort.

Prefix97118continues unchanged; lateststdout416 with no mismatch. Preserve
its1666frozen sources and inspect firstdivergence/failure rather than selecting
another prefix. Goalactive,16completed/auditedepisodes,zeroverifiedroundtrips.

### Paired reactive readout complete

59988closedexit0,resultdb726052b74b4afe4220f86334c855a4f371e808aeeb06626520b17349bfe783,
launch5f754ff78a722fbfb7efe26fdc827ab9b88bd791d38eacb462a09f82ec660079,
1698sources.22matchedlaunchfields admitted, exact physically reproduced
intervention retained. Reactivepath1.9133489139592943m,closestoutboundgoal
2.808442892484211m,28.3simseconds,finalXY[1.1367318614971147,-0.7537140377218317].
Noarrival,strictvisibilityPASS; learnedoutbound1butstrictvisibilityFAIL.
Bothzero roundtrips. Reactive receiptinclusive median718.103454ms versus
learned1237.033925ms;differenttrajectories/lengths,notisolatedinferencecost.
Fullscopeandcomparison:
docs/go2_reactive_floor_transport_readout_result_2026-09-09.md.
Do not repoll59988.18182livePID2475541,89.41CPUseconds1.310GBRSSlastinspection,
inputverificationbeforelaunch.97118laststdout448,still no reported mismatch;
preserve both active attempts. Goal active/no blocker.

### Actual feasibility prefix complete at first changed command463

97118closedexit0; do not repoll. Result45b9b10e0dc89d4ba477499f817a08df02cc79392c3bf3df9b0363394f7b714b,
launch64bdc2e18508cc6f1e1386e86ae478156f7a9a192a23398cf0080fa7b8c32283,
1666frozen sources,573.5599385609385s.464actualpairedobservations0–463,
all463priorrequests exact,461complete rawforecastbanks exact, complete original
selections/observed/mission/residualstate exact. Firstfallback/selectedaction/
requestedcommanddifference463,originalzero→rightarc[.16,0,-.45],no terminal
change; no following observation consumed, unchanged assigned model/inputs.
Currentclearance.45751049799042826; rightarcrawfirstclearance.4497106959665005,
correctedall8segmentminimum.45062258240705394; bothsurfacechecksPASS; only
rightarceligible. This is prospective policy evidence, not a physical result.
Full result/numerical/source scope and next fresh-native requirements:
docs/go2_residual_first_interval_prefix_result_2026-09-09.md.
Future native prefix must cover464pairedendpoints/23900physics samples through
463and reproduce complete candidate decisions before its new command outcome.

18182remains the active fixed independentreactivecohort; PID2475541live during
prefixfinalhardware inspection, inputverificationpending, no launch output yet.
No other native scene active. Memory native still follows that cohort; prepare
residualnative source separately while it runs. Goalactive,16auditedepisodes,
zeroverifiedroundtrips; currentturnmadeprogress on matchingdiagnosis, completed
reactivepilot/readout, completedactualfeasibilityprefix and nextcohortsubmission.

## Residual native successor prepared; independent reactive case1 active

Previous turn made progress completing the actual feasibility prefix and paired
reactive readout. This turn added separate residual native collector/audit,
streaming464-decision admission,23900-sample physical-prefix comparison,
launcher/protocol and26testsPASS3.40s. AST tests preserve original physical/
evaluation calculations except controller/status/metadata. Fresh audit models,
raw failure retention and exclusion of post-intervention outcomes are tested.
No native execution of this successor yet. Source identities/resources/command:
docs/go2_residual_first_interval_native_preparation_2026-09-09.md.
Preflight-only60165submitted,PID2476981live at20.79CPU seconds1.205GBRSS,
inputverificationpending. Pollsamehandle; no retry or success assumption.

Independent reactive18182launched159e55ad6e6373b9a951cb5eb83830bcfce39cb5a5372ccd24c06ca29c29587a,
1682frozen sources,32GiBmemory admission,78,383,153,152required artifactbytes
for all3remaining cases. Parent2475541; tracker2476368; firstworker2476369.
Actualcase1directory collecting; no collectionresult/audit/workerterminal or
rootfailure at last inspection. Preserve fixed order1,2,3 and original method.
Latestpreflightresourcecheck80.335GBRAM90.586GBartifactfree,CPU3.3%,GPUs0%,
all32CPUaffinity; reactiveworker2.687GBRSS. One native scene plus CPU preflight.
Memory native pilot follows reactivecohort, then residualmaze2native; prepared
comparison definitions remain unchanged. Goalactive,16auditedepisodes,
zeroverifiedroundtrips, no blocker.

## Native preflight passed; direct-flow failed-pair probe completed

Previous turn made progress preparing/testing residualnative.60165nowclosedexit0:
1673sources,input/sourcebindingsverified,memory/storagePASS,outputFalse,
nativeexecutionFalse.79.843GBRAM90.214GBartifactfree,CPU3.3%,GPUs0%,all32affinity.
Do not repoll60165. Residualnative remains queued after reactivecohort and
planningmemorypilot; existing prepared/frozen sources unchanged.

Added directcornerflow association with explicit patch/flow checks,9testsPASS
0.21s, then ran fixed actual failed-pair diagnostic69292closedexit0.
Result6525b5e02975def2a296b64ca630bcd55a9da63592f4fbe8f0c828e8820070db,
launchcb1f721e935622a69794fd444bf5e8ae9fe0b8c6d95ce511885dff3d8ede7f5a,
1661sources3071inputscheckedbeforeafter,5.640770641854033s.
Original4rejections reproduced. Alternate original-rigid-gate pass3/4:
maze1aux31/42inliers,posthoctranslationerror.1342mm;
maze3primary12/13,.0644mm;maze3aux20/27,.5927mm.
Maze1primary22lifted stillfails fraction/grid/displacement gate. Full rigid/
gyro thresholds unchanged; association rule explicitly different. No pose
admitted, temporalcontinuity/controllerintegration/nativeexecution yet.
Full source IDs, scope and next guardedobserver integration requirements:
docs/go2_direct_corner_flow_failed_pairs_result_2026-09-09.md.

Reactivecase1completed raw-audited SENSOR_OR_MODEL_FAILURE:171pairedobs,
170commands9250physics10drain,physical/acquisitionNone,strictvisibilityPASS,
rawsensor/controller/commandPASS,verifiedroundtripFalse. Collection
052fdd1233deb9539dfb6fe15a0fccb02ee436f02d722e33b5f930f822f63cd8;
auditb82434126766eb5904f3d4851b1e857ad8e9a2aabe7039780d909c75f750feaa;
workerterminalc5fd185ae6329ae6f54e98980f785369415fc8539e1dcabf371d94e3ea829a76;
progress_after0154b4681584f5d2d00647752f7df840f7d0df69821fb8a646dd4a47385d125a10.
Workerwall896.9798205150291s,peakRSS2,956,836,864bytes. Same18182parent2475541
advanced fixedcase2; oldworker2476369exited, newworker2477889live at38.29CPU
seconds/RSS1,647,685,632bytes. Do not restart; keep2,3fixed.17episodes now
completed/audited,zeroverifiedroundtrips. Goalactive,no blocker.

## Full-observer direct-flow prefix submitted; reactive maze2 complete

The preceding status turn verified a live cohort process and newly saved maze2
raw audit (verified wait and new completion evidence). This continuation made
implementation progress: added the bounded full-controller maze1 replay,
19 causal-boundary tests and fixed protocol. Together with the 11 observer
integration tests,30testsPASS1.94s. Full identities, scope and resource evidence:
docs/go2_direct_flow_observer_prefix_preparation_2026-09-09.md.
Replay40411submitted,PID2479734live at30.31CPU seconds/RSS1,330,671,616bytes;
input authentication pending. Preserve this attempt and its sources. All
original complete decisions must match through213; stop at original failed
observation214 whether recovery succeeds or fails. No following tape consumed.

Reactive maze2 now worker-terminal complete,1227.1331745260395s,peakRSS3,722,727,424bytes:
collection2dfd7c72619a8bd5fd492b42a2888341b674844a3df74f5a79eda334b707722b,
audite8910c8a5e1d783b8a7dd0f09d8454b7976fe76eb18874ffff72a5e3f3a2af5f,
workerterminal2d7a8a5dd4f099cef69c9d98ecfd787632f84974e72baeaf36f6bf04debcea06,
progress_after0206212c085383d9dc190e82ea88c8a8aa328fc786e66ef379482877e0bf57ace0.
408pairedobservations407commands21100physics10drain,terminal
NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY,physical/acquisitionNone.
Rawsensor/controller/commandPASS,strictvisibilityPASS,noarrivalwindows,
verifiedroundtripFalse. Reactive maze1 also has zero nativearrivalwindows.
18episodes now completed/raw-audited,zeroverifiedroundtrips.

Same18182parent2475541advanced fixedmaze3worker2479656,verifiedrunning15.12CPU
seconds/RSS1,298,239,488bytes. Maze3collection/rootresult/rootfailure absent at
inspection. No restart; finish the fixed cohort before planningmemorynative,
then residualmaze2native. Fresh81.083GBRAM88.894GBartifactfree,CPU3.3%,GPUs0%,
all32affinity supports concurrentCPUreplay. Goalactive,no blocker.

40411admission passed: launch4d9900f40d39eb40d7aa1740b99838756b7dc48cb595fafacae5c8b01fc72d2f,
1668sources6201cohortbindingsplusupstreamverification. Lateststdoutframe32,
no reported mismatch. Launch80.471GBRAM88.893GBartifactfree,CPU3.4%,GPUs0%.
Continue same live replay through originalfailedobservation214 only; not yet
a completed prefix or native tracking result.

## V1 replay validation failure preserved; V2 submitted; reactive maze3 raw-audited

40411closedexit1 at its extra boundary pose checker: identity tuple became a
list during JSON serialization. All214preceding decisions exact; candidate214
advanced tick214,auxiliaryreference213ANCHOR_MEASUREMENTaccepted,selectedleftturn
[0,0,.45],no controllerfailure/terminal. The runner did not reach final input/
model checks. Preserve V1 as terminal validation failure, not a completed prefix.
Failure3eae1b519549f7762a3d353cf430ee9856863b9dabf6a233c186aea6a66c87e5,
streamd4d90b933d07f18ce86d3dac86763d002a6b0f1e3fccdc4bd79801fadc656521.
Full failure and interpretation:docs/go2_direct_flow_maze01_prefix_v1_failure_2026-09-09.md.

Separate V2 runner retains live evidence for live checks and requires exact
serialized receipt equality; observer/controller/model/comparator/rules unchanged.
FivefocusedtestsPASS0.17s. Submitted74784,PID2481351live58.61CPU seconds,
RSS1,353,830,400bytes,authenticationpending. All new source IDs/resources in
docs/go2_direct_flow_observer_prefix_preparation_2026-09-09.md. Do not restart
40411or edit either attempt's frozen sources. Goal progress includes implemented
actual replay, complete original prefix evidence, isolated runner diagnosis and
tested/submitted explicit correction; no scientific recovery claim yet.

Reactive maze3 collected117pairedobservations116commands6550physics10drain.
Firstterminal106NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY,no failure,
physical/acquisitionNone. Raw sensor/controller/commandPASS,strictvisibilityPASS,
zero nativearrivalwindows and no roundtrip. Collection
983bcc1f745af0157576d7344509973e70911ef2ce40660dbad442d902bab248;
auditff91d55a823c286b5fb9f99b374b7b0469e7467ca8cfea426c19f67afdb645e2.
Worker2479656andparent2475541stilllive; workerterminal/rootresult absent at
inspection.19episodesraw-audited,zeroverifiedroundtrips. Finalize same18182
before launching prepared planningmemorynative,then residualmaze2native.

Reactive terminal observations verified from streams:maze1=160(decision159,
visualfailure),maze2=397(currentgeometrynoaction),maze3=106(currentgeometrynoaction).
Maze1and3nativevisitedonlystart[-1,0]. Maze2nativevisited[-1,0],[0,0],[0,1],
[-1,1],[-1,2]; noinvalidcrossings. Allthreezeroarrivals. These are original
post-hoc evaluator outcomes, not controller inputs or successful navigation.

## Corrected tracking prefix complete; reactive cohort complete; memory pilot submitted

74784closedexit0,result345b54f1b3c647be516040f2b8bf03b4ab3c709b9737dc0bdf7c78bedcacdbe3,
launch9ec86d78eeb4ef9fa46e2d1031b8b1294cd665b9e413141e9b36cb54a4cf6151,
stream13b4b1017957396bfc7a1b1a8c6cb43fe4a2e65afe8ed680aec1489aea1fcfea.
1673sources,349.3820586500224s,allinput/source/modelcheckscomplete.215observations,
214complete original decisions/priorrequests exact,211priorrawforecastbanks exact.
At214fullcontroller recovers originalvisualfailure usingauxiliaryreference213
ANCHOR_MEASUREMENT,requestsleftturn[0,0,.45]instead ofzero,noterminal/failure.
BothliveposecontractsPASS,weights/inputsunchanged,no following observation.
All215candidate decisions also exactly reproduce preserved V1 candidates;
V1remains a terminal runner-validationfailure. No native recovery claim.
Fullresult:docs/go2_direct_flow_maze01_prefix_result_2026-09-09.md.

Newtrackingnativeprefixhelper22testsPASS1.88s; actual completedV2report and all
215decisions admitted after result/launch/streamauthentication. Requires11450
physics samples,215paired public observations,214priorrequests and all215new
decisions; no following physical outcomes compared. Sourcehashes/resultscope
in reportabove. Native trackingcollector/audit/launcher still need implementation.

18182closedexit0,result51fa637e8a70bbefc088067c548eb0e50149b490b02a97af04cb2fba720418bd,
1682sources4292artifacts3307.7784918081015s. Allfixed1/2/3complete,raw/strictPASS,
zeroarrivals/roundtrips. Finalworkerterminald3e8d92c3075d9e2ae16ce8d33f4e2262917de4aa02de26455714b3567ac8927,
progress0322655f11341ca653712ab12e8db25744f4ec67d5d7ba1e274bd478739edebbc5.
Reactive maze2 crossed4openedges versuslearned1; bothfailgoal. Neither method
leftstartcell in1/3. No learned-planning advantage established. Full outcome
and identities:docs/go2_independent_reactive_floor_transport_mazes_result_2026-09-09.md.
Oldparents/workers2475541/2479656/2481351confirmedabsent. Do not repoll18182,
74784or40411.19completed/raw-auditednativeepisodes,zeroverifiedroundtrips.

Submitted queuedplanning-memorynative70604,PID2483136live37.26CPU seconds,
RSS1,205,399,552bytes,inputauthenticationpending. Uses actualbaselineSHA
1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374,
unchangedpreparedrunner/source/protocol. Priorpreflight87724PASS1678sources
reconfirmed from recordedstatus; no redundant preflightrerun. Fresh82.216GBRAM,
88.404GBartifactfree,CPU3.4%,GPUs0%,all32affinity; no native competitor. One
scene only. Continue same70604; residualmaze2native follows, then separately
preparedtrackingnative. Goalactive/unachieved,no blocker; currentturnmade
implementation and verified experimental progress.

70604subsequently passedlaunchadmission,d8ea49c914abbc4f50f0ac0d973062cb98896015a28172eb937aefea2777c65e,
1678frozen sources. Parent2483136sleeping170.91CPU seconds1,207,996,416RSS;
tracker2483357,newworker2483358running4.13CPU seconds1,336,823,808RSS.
Rootresult/failureabsent. This is the sole active native pilot; preserve its
sources and samehandle. Collection/audit completion not yet claimed.

## Tracking native preflight passed; memory native complete; next jobs submitted

Previous goal turn made progress completing corrected tracking prefixes and
reactive cohort, then launching memorynative. This turn implemented separate
trackingnativecollector/audit/launcher/protocol and28testsPASS3.01s. SourceAST
preserves originalphysical/evaluation calculations outside explicit controller/
metadata changes.77113closedexit0preflightPASS1681sources,allinput/source,
memory/storagePASS,outputFalse/nativeFalse.81.435GBRAM88.033GBartifactfree,
CPU3.3%,GPUs0%,all32affinity. FullscopeandSHAidentities:
docs/go2_direct_flow_maze01_native_preparation_2026-09-09.md.
Actual trackingnative remains after residualmaze2; do not repoll77113.

70604closedexit0,resultdca1757aa1358ca48bf5d40240337aafb43391d0306b376309e1b3239da89780,
1678sources657artifacts694.1448035400826s.103pairedobservations102commands
5850physics10drain,firstterminalobs92(decision91)visualmissingnessfailure,
physical/acquisitionNone. Rawsensor/model/controller/commandPASS,exactnative
1250physics/11pairedprefix/8forecastbanks and allsaveddecisions through10;
changedleftarccommand completed. Noarrival,nocrossing,zeroverifiedroundtrips.
StrictvisibilityFAIL,hardframes18/84,primaryinteriorerrors1.023805/1.035692mm;
auxiliaryPASS. Fullresult/identities/scope:
docs/go2_current_observation_planning_maze_pilot_result_2026-09-09.md.
20completed/raw-auditednativeepisodes,zeroverifiedroundtrips. Do not repoll70604.

Added actual matched JEPA/supervised-rollout prefix runner for allfixed1/2/3,
21testsPASS1.94s. Samecontroller/rollouthead, own authenticatedtrainingbias,
freshmodels/historypercase; exactoriginalJEPAdecisions and sharedobservedstate;
stopbeforefollowingfirstchangedcommand/terminal. Preparation/sourceidentities:
docs/go2_matched_objective_prefixes_preparation_2026-09-09.md.

After completedmemoryparent exited, fresh82.783GBRAM88.064GBartifactfree,
CPU0.3%,GPUs0%,all32affinity supported sequential submissions:
-78591residualnative,PID2485335live39.33CPU seconds/RSS1,227,497,472bytes;
-49332memorypairedreadout,PID2485368live38.16CPU seconds/RSS1,198,223,360bytes;
-58952matchedobjectiveprefixes,PID2485387live37.0CPU seconds/RSS1,350,909,952bytes.
All initiallyinputauthenticationpending. Onlyresidualmaycreateanativescene;
otherjobsCPU-onlyandindependent. Capacity32+8+12GiBbelowavailable82.78GB,
notOSquotas;launchersrefreshbeforeexecution. Preserve all three handles/sources,
no restart or success assumption. Goalactive/unachieved,no blocker.

## Matched objective prefixes and memory readout complete; residual pilot active

58952closedexit0,resultedf680896e11ef22096cf56323094be2de24b712405466149cdeeb59bab2a6fd,
launch665fc0db005929ce763e132b2161d210cb32bdbc9c8f753cc162cc9f81897fc6,
1664sources182.68536946782842s. Allfixed1/2/3completed; eachconsumed0–3only,
allfouroriginalJEPAdecisions andthreeprioractualcommands exact,sharedcurrent
raw/registered/map/missionstate exact. At3JEPAleftarc[.16,0,.45]vs supervised
rightturn[0,0,-.45],noterminalchange; bothfull6x8x5rolloutbanks recorded.
Neverconsumed4. Bothbase/correctionassignments exact,modelstatesunchanged,
nograds,allinput/source/artifactcheckscompleted. Supervised correctedstate
171c8576d2c3fcfd0ce698351acf86a05f2ea29bdf829041e98641cdac778a73.
This is a prospective pipeline decision difference, not JEPA navigation advantage.
Fullresult/identities/next900physics4pairednativecomparison requirements:
docs/go2_matched_objective_prefixes_result_2026-09-09.md.
Supervised native all-three cohort and prefixchecker/launcher remain unimplemented.

49332closedexit0,result3d11666a31d229355bafbd50f622c73ef0dd7acf2882654eb07fe1b8e7747214,
launch9320c50cdcc2e3f936019ad429b8caa475f6fbbfbac5101cce5df70e62aa3171,
1688sources. Currentplanningnativepath.498157934m,10.2simseconds,closestgoal
3.783563618m,zeroarrival,strictFAIL. Receiptinclusivemedian945.811852ms,
max1221.033506ms,all103over100ms. Baselinepersistent1237.033925msmedian,
oneoutboundbutstrictFAIL,noreturn; notisolatedmemorytiming or memoryadvantage.
Full paired result:docs/go2_current_observation_planning_readout_result_2026-09-09.md.
Do not repoll58952or49332.20auditednativeepisodes,zeroverifiedroundtrips.

78591residualnative passedlaunchadmission:
78131563622b925a0eb06819534d09e4cec1dfeaeb6354df0cd8a8e5cb443b1b,
1673frozen sources. Parent2485335sleeping343.15CPU seconds/RSS1,229,705,216;
tracker2485787,worker2485788running118.22CPU seconds/RSS1,417,732,096.
Launch79.987GBRAM88.036GBartifactfree,CPU6.5%,GPUs0%,all32affinity.
No result/failure yet. This is sole active native pilot; keep samehandle and
sources. Trackingnative1681-sourcepreflightpassed and remains next, then prepare
the matchedsupervisednativecohort. Goalactive/no blocker; this turn completed
native memory evidence, model-objective intervention evidence and tracking
native preparation rather than reclassifying any negative as success.

## Supervised native cohort implemented and tested; residual pilot continues

The previous status turn was a verified wait: residual parent 2485335 and
worker 2485788 were confirmed live, with no terminal result. This turn makes
implementation progress: completed the fixed supervised-rollout cohort runner,
physical-prefix checker, paired-outcome helper, protocol and focused tests.
41 tests PASS in 2.89s, session29204 exited0. Original collection/controller/
evaluation functions are directly reused. Tests cover complete fixed prefixes,
altered physical/public/command/model evidence, future-observation exclusion,
fixed outcome order, fresh audit models and retained evidence after failures.
Full scope, six source hashes and measured resources are recorded in
docs/go2_supervised_rollout_native_preparation_2026-09-09.md.

Submitted CPU-only supervised preflight84246, PID2488200. Latest live check:
114.53 CPU seconds, RSS1,257,623,552bytes; no preflight result yet. Output remains
absent and no supervised native execution has started. Do not restart this
handle or modify its six sources. Preflight has enough current resources:
76.756GB available RAM,84.855GB artifact free,CPU6.4%,both GPUs idle,all32
affinity. These measurements are capacity evidence, not timing qualification.

Same residual native78591 remains live. Parent2485335sleeping344.41CPU seconds;
worker2485788running1485.07CPU seconds,RSS5,795,475,456bytes. Its complete
timing rows reach1117(lasttick1116), beyond the original maze2 terminal503.
Collection result, raw audit, worker terminal and root result/failure still
absent. This is live continuation evidence, not yet an audited recovery,
arrival, round trip or planning advantage.20completed/raw-audited episodes,
zeroverifiedroundtrips remains the verified count.

Continue the same residual run; after its parent terminates and its evidence
is inspected, refresh resources and launch the already preflighted tracking
maze1 pilot. Then execute the fixed supervised layouts1–3 after its preflight
passes and resources refresh. Goal active/unachieved, no blocker. Do not
repoll closed test29204 or hardware96375 sessions.

## Supervised preflight complete; residual actual-outcome readout prepared

Previous goal turn made implementation progress with the supervised cohort and
41passingtests. This turn completed84246exit0preflight:1,672sources,allthree
actual prefixes and upstream input/source bindings PASS, no output/native
execution. Final76.523GBRAM83.802GBartifactfree; required34.360GBRAM78.383GB
storage,CPU3.3%,GPUs0%,all32affinity. Its six sources remain unchanged. See
docs/go2_supervised_rollout_native_preparation_2026-09-09.md.

Prepared actual residual fallback outcome comparison against original maze2:
new helper, runner, protocol and23focusedtestsPASS. A source-preparation check
caught missing inherited verifier input bindings before execution; corrected
and covered by regression. Final30909exit0 verifies1,679source/input/environment
bindings. Actual original-data smoke20002exit0 authenticates four consumed
artifacts before/after; reproduces514observations,terminal503,1.503742253m
path,noarrival/roundtrip,strictPASS,no fallback,medianreceipt962.827517ms.
Full source IDs, original measurements and preserved preparation failure:
docs/go2_residual_first_interval_readout_preparation_2026-09-09.md.
No full paired readout output yet; await actual final residual native SHA.

Same78591residual native remains live. Parent2485335sleeping345.12CPU seconds;
worker2485788running2213.19CPU seconds,RSS7,734,571,008bytes. Timing stream
has1740complete rows(lasttick1739). Root result/failure, collection result,
raw audit and worker terminal remain absent. Continuing beyond original503
is live evidence only; final trajectory, prefix and audit are not yet verified.
20completed/raw-auditednativeepisodes,zeroverifiedroundtrips remains current.

Next: finish same78591 and inspect its exact final artifacts. After its parent
terminates, refresh resources and submit prepared trackingnative maze1. The
residual paired CPU readout may overlap that job with measured headroom. Then
run fixed supervisedlayouts1–3; their preflight is complete. Goal active and
unachieved, no blocker. Do not repoll closed84246,20407,30296,59028,20002,
77080,75826,30909 or10263 sessions. No launched source or prior result changed.

## Receipt-copy selector implemented; paired controller replay submitted

Previous goal turn made progress: supervised preflight completed and residual
outcome readout was implemented, tested and checked on original data. This turn
reviewed existing completed timing evidence rather than rerunning its diagnosis.
Current controller already inherits the frame cache; receipt-copy integration
remained outstanding. Implemented a separate selector using 15 original code
objects with explicit isolated copy/dependency bindings and unchanged state flow.
No original module global, live object or frozen source changed.

25 tests pass: 14 in 1.89s (59683 exit0) and 11 in 2.11s (60233 exit0).
Submitted paired complete-episode replay78762, PID2491693, initially live at
26.65 CPU seconds and RSS1,330,696,192bytes. Its launch/result/failure files
were absent while authenticating inputs. Preserve this handle and new sources.
Compare both controllers against all514 original maze2 decisions; alternate
execution order and report controller-only paired timing. No new native scene
or current speedup claim. Full source identities and scope:
docs/go2_receipt_copied_controller_preparation_2026-09-09.md.

Fresh hardware:74.616GB available RAM,80.925GB artifact free,CPU3.5%,GPU0/1%,
16physical/32logical CPUs,all32 affinity. One16GiB CPU replay allowance beside
the existing sole native scene fits available headroom; launch refreshes it.

Same residual78591 remains live, parent2485335sleeping345.68CPU seconds;
worker2485788running2811.33CPU seconds,RSS9,145,622,528bytes. Timing stream
has2225complete rows(lasttick2224). Collection result, raw audit, worker terminal
and root result/failure remain absent.20completed/raw-audited native episodes,
zeroverifiedroundtrips remains the verified result. Do not restart the pilot.

Next: follow same78591 to completion and inspect actual result; then tracking
native maze1 with refreshed resources, with prepared residual CPU readout able
to overlap. Supervised fixed1–3 cohort follows; preflight already passed.
Also follow78762 to full equivalence/timing or preserved failure. Goal active,
unachieved, no blocker. Closed59683,60233 and hardware52435 need no further polls.

## Paired timing replay admitted; unchanged tracking maze3 prefix submitted

The previous goal turn made implementation and submission progress. This turn
confirmed78762 launch admission and216 exact completed decision pairs. Launch
96f6a1b90454689d0fe8b6f602ba178535be58054d2401e45553f586e8ea569e,
1,671sources;73.319GBRAM80.380GBartifact free,CPU3.6%,GPUs0%,all32affinity.
PID2491693running490.93CPU seconds,RSS2,794,754,048bytes. Full514pair result
and final checks remain pending; no partial timing/speedup conclusion.

Prepared the unchanged direct-flow tracker for the other original visual-failure
case, maze3. Comparator changes only fixed boundary214 to264; replay calculations
match maze1V2 with the scene/output assignment changes.12testsPASS2.34s,91305
closedexit0. Submitted47935,PID2493328live54.51CPU seconds,RSS1,330,827,264,
input authentication pending with launch/result/failure absent. No native scene,
changed threshold, altered controller/model or following observation. Complete
scope, four source hashes and resource evidence:
docs/go2_direct_flow_maze03_prefix_preparation_2026-09-09.md.

Fresh71.175GBavailableRAM79.628GBartifactfree,CPU6.5%,GPUsidle,all32affinity;
planned32+16+8GiB for the sole native scene and two independent CPU replays
fits current headroom. No native concurrency increase. Source files frozen.

Same residual78591 remains live. Parent2485335sleeping346.17CPU seconds;
worker2485788running3334.47CPU seconds,RSS10,316,537,856bytes. Timing stream
has2612complete rows(lasttick2611); collection result, raw audit, worker terminal
and root result/failure remain absent.20completed/raw-audited native episodes,
zeroverifiedroundtrips remains the verified result. Continue same handle.

Next: finish residual and audit its actual result; refresh resources for prepared
trackingmaze1 native, with the residual outcome readout able to overlap. Then
supervisedfixed1–3 (preflight complete). Finish78762 paired benchmark and47935
maze3 prefix; preserve either result or failure. Goal active/unachieved, no blocker.
Closed91305 and hardware87496 need no further polls.

## Completed replays and stagnation diagnosis; hold successor prepared

The intervening status turn was a verified wait: same native78591 and its
parent2485335 were confirmed live; no terminal result/failure. This turn made
implementation progress and completed tests and source verification for a
separate hold-reconsideration candidate and prospective prefix replay.

Superseding the earlier partial reports:78762 paired benchmark exited0,
result60849db8aa7a2b09e5e6de977115cb953f6c1b893ed62b0533ccd19b31f1ff95.
All514 original/candidate decisions match exactly. Active500 observations:
median controller750.178271ms original versus695.7320085ms candidate,
median paired reduction54.313986ms. All500 each exceed100ms. Both execution
order subgroups improve; this is controller replay timing, no full-loop or
native qualification. Details:docs/go2_receipt_copied_controller_benchmark_result_2026-09-09.md.

47935 unchanged tracking maze3 prefix exited0,
result104edd70824e7d4e7b909be924eb68a918542efd3464ff3247f9570ddc28669e.
All264 prior decisions/commands and261 forecast banks exact. At264, primary
reference263 with13 valid depth pairs passes unchanged checks; controller
requests left turn instead of original terminal zero. Never observation265.
Details:docs/go2_direct_flow_maze03_prefix_result_2026-09-09.md. No native maze3
launcher or fresh continuation yet; tracking maze1 remains next native run.

Residual maze2 collection closed with3014paired observations,3013commands and
151400physics samples,10terminal zero intervals. Collection
f7a946687f108c354a7c6f4e5155530875ca0da7bcd7e57f9564ff89c23fa5b4;
terminal3003 MISSION_TICK_BUDGET_EXHAUSTED, no observed arrival, reported
goal distance3.847554391690519m. Raw audit, physical prefix comparison and
final result still pending, so20completed/raw-audited episodes remains the
verified count, with zero verified round trips. Preserve same78591/sources.

Closed-stream stagnation diagnosis96580 exited0,
result97ddebca3610af75f0f7a84bb8a74df2e7fb36dbdcfc4a58a1a94a122d0496b5.
2643holds among3000active selections; original no-action fallback attempted
only at463. Late sampled translating actions fail raw nominal-path gates while
hold and turns remain feasible. No rescoring or physical-success inference.
Details:docs/go2_residual_maze02_stagnation_result_2026-09-09.md.

Implemented ResidualHoldFeasibilityController and separate replay helper,
runner, protocol and tests. Keep original hold unless a strictly better
original utility passes corrected first-interval geometry, all8segments,
original/corrected surface checks and phase restrictions. Raw forecasts,
residual targets, later points, yaw/contact and original no-action recovery
remain unchanged. No forced translation, timeout or threshold change.
Replay stops at first changed command, comparing all prior complete decisions
and causal state; truncated input cannot be mistaken for a completed prefix.

19policy/controller testsPASS2.01s84891;28replay/admission testsPASS4.10s31370,
47total. Earlier68806 27testsPASS before truncated-input regression added.
1589 source-preparation checkPASS1684bindings. No model loaded or replay/native
output created; actual completed residual native result is still required.
Seven source hashes, exact scope and next CLI are recorded in
docs/go2_residual_hold_prefix_preparation_2026-09-09.md.

76601hardwareclosed:72.576GBavailableRAM78.143GBartifactfree21.359GBworkspace
free,CPU3.4%,GPUsidle,all32affinity. Supervised batch requires78.383GB;
resolve capacity before its launch, do not bypass frozen admission. Immediate
tracking pilot and bounded CPU analysis still fit. Latest native parent2485335
sleeping348.34CPU seconds, worker2485788running5689.52CPU seconds,
RSS10,758,172,672bytes. No new native scene, restart, deletion or frozen edit.

Next: finish same78591 raw audit and final comparison; refresh resources and
launch prepared tracking maze1. Prepared residual paired readout and newhold
prefix may overlap that single native scene with measured headroom after final
input admission. Supervised fixed1–3 follows once storage meets its requirement.
Goal remains active and unachieved; independent successes, prediction/JEPA/
memory advantages, physical return and practical timing remain unproven.

## Fresh maze3 tracking experiment implemented; preflight live

Previous goal turn made implementation progress: hold-reconsideration policy
and replay completed47tests and1684-source/environment preparation verification.
This turn adds a fresh physical test for the completed unchanged maze3 tracking
replay. It does not alter the active residual pilot or reorder the native queue.

Implemented separately named maze3 collector, native-prefix comparator, runner,
protocol and focused tests. Collector calculations match the original except
status labels; directly reuse the layout-parameterized maze1 raw audit. Same
frozen DirectFlowFloorTransportController, model, observed state, numerical
settings and physical evaluator. No additional tracking tuning, residual
feasibility or receipt-copy optimization. Fresh collection/audit model owners.

Native-prefix admission requires the actual completed maze3 replay
104edd70824e7d4e7b909be924eb68a918542efd3464ff3247f9570ddc28669e.
Compare13950physics samples,265paired observations,264prior actual commands,
261raw forecast banks and all265complete new decisions. First changed command
at264 is leftturn[0,0,.45]. Do not compare following original physical outcomes;
preserve partial new dispatch and scientific/infra failures as actual evidence.

28testsPASS2.99s56219exit0, covering unchanged physical calculations, direct
audit reuse, fixed assignments, fresh models, retained failure evidence and
exact physical/public/decision prefix boundaries. All six new source hashes
and the reused audit hash rechecked. Full scope and source identities:
docs/go2_direct_flow_maze03_native_preparation_2026-09-09.md.

Submitted CPU-only native preflight14145,PID2498090. Most recent check running
212.06CPU seconds,RSS1,326,587,904bytes. No terminal output yet, no maze3
native output created. Keep this handle/sources; do not restart. Hardware45523
closed:72.559GBavailableRAM78.117GBartifactfree21.359GBworkspacefree,CPU3.4%,
GPUsidle,all32affinity. One CPU preflight fits beside the sole native scene.

Same residual78591 remains live:parent2485335sleeping348.79CPU seconds;
worker2485788running6184.04CPU seconds,RSS10,758,172,672bytes. Raw audit,
final result and failure absent. Its closed collection remains provisional;
20completed/raw-audited episodes,zeroverifiedroundtrips is still the count.

Next: finish residual raw audit/prefix; refresh resources and launch prepared
trackingmaze1. Residual paired readout and hold prefix can overlap that native
scene after admitting the actual final residual result. Supervised fixed1–3
follows once78.383GBstorage requirement passes; trackingmaze3 then follows.
No deletion, attempt restart, frozen-source mutation, new native concurrency
or deployment claim. Goal remains active and unachieved, with meaningful
preparation progress and no overall blocker. Closed56219/45523 need no repoll.

## Maze3 preflight complete; current combined timing candidate submitted

Previous goal turn made implementation progress on maze3 native preparation.
This turn completed14145exit0preflight:1685source/input bindings and actual
265-decision tracking prefix verified, memory/storage admissionPASS, no output
or native execution. Final71.991GBRAM78.114GBartifactfree,CPU3.3%,GPUsidle,
all32affinity. Appended result to docs/go2_direct_flow_maze03_native_preparation_2026-09-09.md.

Reviewed earlier completed performance evidence before implementing another
optimization. Existing packed-owned/single-pass indices had preserved1873
dual-camera decisions, but were not adopted by the current measured-floor
controller. The completed receipt-copy benchmark still measured695.732ms
median. Compose those existing index and receipt-copy implementations in a
separate SinglePassReceiptCopiedController, preserving current observation,
registration, mission, residual, map semantics and original decision metadata.
Only eight independently owned empty indices are substituted. No native edit.

Prepared paired full514-observation maze2 replay: receipt-copy alone versus
combined candidate, both exact against every saved original decision. Alternate
arm order, stop before following a mismatched command, preserve input/model
states and report additional index timing benefit, including100ms misses.
Authenticate both completed optimization predecessors and original cohort.
Do not infer current speedup, whole-loop timing or navigation from prior results.

Initial24468 exited1 before output/model/replay: generic ordered-launch verifier
encountered KeyError('input_sha256') on the older single-pass launch schema.
Output root confirmed absent. Corrected preparation to call each predecessor's
exact original verifier. Preserved failed runner/test hashes and reason in
docs/go2_single_pass_receipt_copied_preparation_2026-09-09.md. No original
source/evidence or completed attempt changed. Three regression cases enforce
schema dispatch, source-conflict rejection and propagated verification failure.

17testsPASS:initial81533 14tests3.28s includes3controller and11benchmark;
final21689 14benchmark tests2.03s after3regressions added. Five final source
bindings checked against preparation. Resubmitted75686,PID2500165confirmed
running68.12CPU seconds,RSS1,401,135,104bytes. Launch/result/failure absent
while input authentication proceeds; preserve same handle and final sources.
No model/timing outcome yet. Closed24468,81533,21689,91653,95623 and14145
need no repoll.

Fresh95623hardware:72.388GBavailableRAM78.113GBartifactfree21.359GBworkspace
free,CPU3.3%,GPUsidle,all32affinity. One16GiB paired CPU replay beside the
single32GiB native allowance fits. Resources are checked again before admission.

Same residual78591 remains live,parent2485335sleeping349.26CPU seconds,
worker2485788running6701.29CPU seconds,RSS10,758,172,672bytes. Final result
and failure absent. Collection is closed but raw audit/comparison pending;
20completed/raw-audited episodes,zeroverifiedroundtrips remains verified.

Next: finish residual and admit its actual result; refresh resources for prepared
trackingmaze1. Residual paired readout/hold replay then become admissible.
Supervised fixed1–3 requires restoring78.383GBstorage headroom; trackingmaze3
follows. Finish75686 full equivalence/timing or preserve its failure. Goal active,
unachieved and not blocked. No deletion, restart of a live job, new native
concurrency, policy promotion or hardware claim.

## Exact pip-cache proposal prepared; existing audit and benchmark remain live

Previous goal turn made implementation progress and completed maze3 preflight.
This turn verified the same two running handles and prepared a concrete storage
option for the pending supervised gate. No native, model or frozen-source
change. The audit worker has the original compressed decision stream open;
its nonterminal state is confirmed by the live process, not inferred from files.

Read-only metadata inspection of /home/andrewknowles/.cache/pip found142 ordinary
UID1000 single-link files,1,528,332,288allocated bytes (about1.423GiB), among3573
directories. No symlink, protected name, special entry or traversal error; cache
payloads not opened. Accessible processes show no active package installer or
open cache file;45processes were inaccessible, so do not claim universal proof.
The active residual launch's binding/environment fields contain no literal
reference to this path. The proposal preserves all directories/unlisted files.

Exact file/metadata proposal:docs/go2_pip_cache_cleanup_proposal_2026-09-09.json,
SHA95d9acb80c802893f40f7cee236b80eec6f8d4845afe3fc981bb4c28147a6381.
Review/scope/tradeoff:docs/go2_pip_cache_cleanup_review_2026-09-09.md.
Artifactfree78,112,460,800bytes versus supervised requirement78,383,153,152;
potential1.53GBrecovery gives limited headroom and does not guarantee the queue
fits after trackingmaze1. Do not lower reserve or drop fixed comparison cases.

Asked asynchronously for explicit approval to remove only those142named files.
Question remains pending; no removal. Earlier GSD authorization explicitly says
pip_cache_cleanup_included:false. Do not infer approval from elapsed time,
current goal continuation or the earlier completed GSD cleanup. If approved,
recheck users and every path/leaf identity before first unlink, journal actual
removals, preserve all unlisted entries and refresh resource admission.

Same75686benchmark remains live,PID2500165running320.08CPU seconds,
RSS1,379,676,160bytes. Launch/result/failure absent during inherited-input
authentication; no timing/equivalence result yet. Same78591native remains live:
parent2485335sleeping349.49CPU seconds,worker2485788running6953.32CPU seconds,
RSS10,758,172,672bytes. Launch78131563622b925a0eb06819534d09e4cec1dfeaeb6354df0cd8a8e5cb443b1b
unchanged; final result/failure absent.20auditedepisodes,zeroverifiedroundtrips.

Continue those handles, then prepared trackingmaze1 with resources refreshed.
Residual readout and hold replay await its completed raw result. Supervised1–3
await storage admission; trackingmaze3 remains prepared/preflighted afterward.
Goal active/unachieved; no overall impasse while existing jobs and independent
work continue. Do not restart either live job or treat cache approval as given.

Subsequent same-handle poll:75686 admitted launch
5ae02642ad918e2b271b48e92e0f0adb9c392c7b2598412d03b36656e2b81e18,
1696sources. First61complete decision pairs exact; no complete timing/equivalence
claim. Launch71.706GBRAM78.116GBartifactfree,CPU3.4%,GPUsidle,all32affinity.
PID2500165confirmedrunning417.44CPU seconds; residualworker2485788running
7052.70CPU seconds, same78591 still nonterminal. Cache approval pending.

## Verified wait on the same audit and paired replay

Previous goal turn made progress by preparing the exact storage proposal and
observing successful combined-benchmark admission. This turn is a verified
wait:78591 and75686 were polled repeatedly and both remained live. No restarted
job, source edit, native submission, cleanup or inferred approval.

Residual worker2485788 advanced from7127.64 to7299.68CPU seconds; its compressed
decision read reached338,825,216of347,142,791bytes, and the decision stream was
no longer open at the last process check. Raw audit/prefix/final result/failure
were still absent, so do not claim completion from that observation. Latest
workerRSS10,919,538,688bytes; parent2485335sleeping349.81CPU seconds.
Benchmark2500165running661.42CPU seconds,RSS2,813,362,176bytes, with its
frame128 progress marker received and no reported mismatch or final result.

Keep the same handles. The queued trackingmaze1 launch awaits actual completed
audit/prefix/result and resource refresh. Supervised storage gate and exact
pip-cache approval remain pending independently; goal is active, not blocked.

## Residual raw audit and physical prefix pass; paired timing reaches514

Previous turn was a verified wait on the two live handles. This turn obtained
new completed evidence from the same residual worker: raw audit SHA
5928241136736e148bf64cdd91294774da1255074b6c10aea80309037f1c4fd8.
Primary/auxiliary reconstruction, complete model/command replay, command audit,
unchanged weights and strict physical visibility allPASS, zerohard measurement
failures. No observed/native arrival, no physical stop, no verified round trip.
Native traversal has only the open[-1,0]to[0,0]crossing at sample9966, no invalid
crossing, no return traversal, terminal native quietFAIL. Action counts match
the closed stagnation diagnosis:2643holds,147left turns,91right turns,75left arcs,
44right arcs. This is a scientifically negative but raw-audited trajectory.

Physical-prefix comparison SHA
c65b1d2ea83cd87b013054c8304341a3209b3129c0f7f35e760ddc8a6dc57321:
464observations,23900physics samples,461raw forecast banks exact. All prior
actual requests and complete candidate decisions match prospective replay.
Firstintervention463zero→rightarc[.16,0,-.45]completed. Native raw physics prefix
e35365ec72a863f7c8e5f0d085b740ea12aafcb3e15ee81201fd21664cb07d51.
No following original physical outcomes compared/inferred. Appended these
results to docs/go2_residual_maze02_stagnation_result_2026-09-09.md.

Same78591 remains nonterminal while worker/source/artifact/root checks finish;
worker terminal and root result/failure still absent. Latest confirmed live
worker2485788running7714.74CPU seconds, parent2485335sleeping350.20CPU seconds.
Do not admit downstream replay/readout from partial documents; require final
rootresult SHA. The previous20finished audited episodes plus this newly completed
raw audit still establish zero round trips; final attempt admission is pending.

Same75686 reached all514paired timing rows after its frame512marker; each row
records complete original/candidate equality. PID2500165confirmedrunning1071.51
CPU seconds before final checks. Root result/failure remain absent; no final
timing estimate or current speedup conclusion yet. Preserve both running jobs.

Next: finish final residual admission, refresh resources and launch prepared
trackingmaze1. Then admit residual paired outcome readout and hold prefix using
the actual final SHA. Finish75686 final checks. Supervised storage gate and
pip-cache approval remain pending; trackingmaze3 is prepared/preflighted.
Goal active and unachieved, no overall blocker and no deletion or live restart.

## Residual pilot and combined benchmark complete; next three jobs submitted

Previous goal turn completed raw-audit and prefix evidence. This turn completed
the same78591 parent,exit0, final native result
55a7d5071f39337b3c9ea329e5b48320f11c8a5a9ba6e34296926006768ce466.
1673sources,18123artifacts,8254.2632445s wall. Worker terminal
d85d98ffb06a028b62826f68b5b4a0255270d2d14798a72c826aa2077eb995ae;
all final raw/prefix/source/artifact checks pass. Strict visibilityPASS,
zerohard measurement failures, no arrival/return/roundtrip.21completed and
raw-audited native episodes,zeroverifiedroundtrips is now the aggregate.
Result/critical bindings checked again. Full report:
docs/go2_residual_first_interval_maze_pilot_result_2026-09-09.md.

75686 also exited0, final combined benchmark result
d688f2ed9d30177d2e55fb98e9c9f25d2b035f2258d86449ac8e86615cd13c72.
1696sources,1059.9303622s wall after admission;514complete decisions exact,
public inputs and both model states unchanged. Active500median receipt-copy
661.482598ms versuscombined567.2465515ms, medianpaired reduction97.1747875ms.
Both order subgroups improve (98.333615/97.0875525ms reductions), all500each
exceed100ms. Additional index benefit over receipt-copy on this episode only;
do not add separate-run speedups or claim whole-loop/native timing. Result and
all output hashes rechecked. Full report:
docs/go2_single_pass_receipt_copied_benchmark_result_2026-09-09.md.

Submitted guarded tracking77300/PID2503306 and paired readout26695/PID2504054
behind the exact residual parent incarnation. Both observed parent termination,
authenticated final result/critical bindings as applicable and refreshed hardware
before exec of the original prepared runners. No duplicated launcher, original
source edit or new admission bypass. Parent completion resources83.16/83.29GBRAM,
78.081GBartifactfree,GPUsidle,all32affinity; conservative48/56GiBcombined memory
checks passed. Tracking's own full source/model/prefix admission still applies.

Submitted hold-reconsideration prefix69725/PID2504565 with the exact final
residual SHA. Fresh68959hardware:81.529GBavailableRAM78.080GBartifactfree,
21.359GBworkspacefree,CPU6.5%,GPUsidle,all32affinity. Planned32GiBnative plus
8GiBreadout plus8GiBhold replay fits. No benchmark remains active. Each runner
checks resources again before creating output. Only tracking creates a scene.

Latest actual processes:tracking2503306running191.07CPU seconds/RSS1,301,164,032;
readout2504054running190.06CPU seconds/RSS1,226,493,952;
hold2504565running101.39CPU seconds/RSS1,413,750,784. All three launch/result/
failure files absent during input admission. Preserve handles77300/26695/69725
and their frozen sources. Do not resubmit while these jobs remain live.

Next: finish three admissions and inspect actual outcomes/failures. The hold
replay must stop at its first changed command; later physical continuation needs
a separate new experiment. Native queue remains trackingmaze1, supervised1–3
once storage passes, then trackingmaze3. Pip-cache approval still pending; no
deletion. Goal active/unachieved, no overall blocker. Closed78591,75686,68959
need no further polling.

## Tracking native and residual readout admitted; hold prefix still authenticating

Previous goal turn completed the residual native and paired benchmark and
submitted their next jobs. This turn confirmed actual tracking77300 admission,
launch6024e127fa1546b75c548b91cbbdcbd7e173b25dde24d2611c903ef06c784392.
Parent2503306sleeping351.34CPU seconds,RSS1,215,160,320bytes, tracker2505032,
worker2505033running. Preserve this sole native pilot and its source definitions.

Residual outcome readout26695 admitted launch
82137678a68de81bd1d46b084361aba98404db534c3b7169495553bf634f1a8c.
PID2504054running518.46CPU seconds,RSS1,340,239,872bytes. No final result/failure
yet. Hold prefix69725/PID2504565running430.78CPU seconds,RSS1,439,805,440bytes;
launch/result/failure still absent during inherited-input authentication. These
are live jobs; no observation timeout or missing terminal file caused a restart.

No new navigation or timing claim.21completed/raw-audited episodes andzero
verifiedroundtrips remains the aggregate. Continue77300/26695/69725. Full
readout and prospective hold outcome remain necessary before preparing the
corresponding changed-command native test. Supervised storage and exact pip-cache
approval remain pending; no deletion. Goal remains active with no overall blocker.

## Residual paired readout complete; hold replay admitted

The previous status turn yielded new evidence: paired readout result
aa1abf8ce0110dca271bc5f93fabf51a62bb41315fa5971afe16f87c25c59888.
Session 26695 is now confirmed closed, exit 0. Its launch and all 1,679 source
bindings were rechecked. Full analysis is recorded in
docs/go2_residual_first_interval_maze_readout_result_2026-09-09.md.
Closest native goal distance changed from 3.946789929 to 3.844697442 m, with
one open-edge crossing and no arrival in either run. The one executed fallback
reduced first-step XY error from 11.743 to 9.737 mm, without an error-bound or
physical-clearance claim. Successor full-iteration median is 1162.687231 ms;
all 3,014 measured iterations exceed 100 ms. The repeated holds remain the main
observed failure of this successor.

Same hold replay 69725 admitted launch
b344a274f4649943c24d0e6350c5d8cf00a2820a5cb6283ad9ba3e6dd222a725
and reported progress through frame 288. PID 2504565 is confirmed live. Same
tracking native 77300 has live parent 2503306 and worker 2505033. Neither has a
terminal result or failure at this check. Preserve both; no restart or duplicate
scene. Aggregate remains 21 completed/raw-audited episodes, zero round trips.
Supervised storage admission and the exact pip-cache proposal remain pending;
no deletion. Goal active, with meaningful work continuing and no overall blocker.

## Current optimized phase diagnosis submitted; tracking reaches a new audited failure

Previous goal turn was progress: confirmed completed paired residual readout,
checked its 1,679 source bindings and recorded the result. This turn added a
phase diagnosis for the verified combined controller, whose actual remaining
costs are not established by the older predecessor profile. 27 focused tests
passed in 4.98 seconds (89130, exit 0). The runner requires all 514 original
complete decisions, unchanged public arrays/model, actual commands, exact
exclusive-time partition and hook cleanup; truncation and any mismatch stop it.
Preparation, four frozen source hashes and fresh hardware measurements are in
docs/go2_single_pass_receipt_phases_preparation_2026-09-09.md.

Submitted session 10216, live PID 2507444, authenticating inputs before output
admission. One CPU timing replay plus the existing 32 GiB native and 8 GiB hold
allowances fit the measured 76,757,487,616 available RAM bytes. Artifact free
space 76,054,351,872 bytes remains below the fixed supervised cohort gate.
No deletion, native concurrency increase or running-source edit.

Tracking 77300 now has completed collection, full raw audit and exact physical
prefix comparison. 515 observations, 514 completed commands and 26,450 physics
samples. First terminal is frame 504, SENSOR_OR_MODEL_FAILURE: current measured
candidate conflicts with transported floor reference. Raw reconstruction,
complete replay, command audit and strict visibility pass. No cell crossing,
arrival or round trip. Prefix 215 observations/11,450 physics/211 forecast banks
exact; the changed frame-214 left turn completed. Identities and limits recorded
in docs/go2_direct_flow_maze01_native_result_2026-09-09.md. Worker/root final
verification remains pending; preserve the same parent and worker.

Same hold replay 69725 reached frame 832 without a reported changed command
or failure. Await its complete result; do not infer a negative full-prefix
finding from partial progress. Goal active and unachieved. Final completed
attempt aggregate remains 21 until tracking's final result is admitted; the
additional inspected raw audit also reports zero verified round trips.

Final live check this turn: tracking parent 2503306 sleeping, worker 2505033
running at 1,578.91 CPU seconds; neither root result nor failure yet. Hold
2504565 running at 1,850.77 CPU seconds, progress through frame 960, admitted
1,684 sources, no terminal result/failure. Phase diagnosis 2507444 running at
235.36 CPU seconds and 1,399,619,584 RSS bytes, still authenticating before
launch/output. Keep handles 77300, 69725 and 10216. No observed timeout was
treated as termination and no job was resubmitted.

## Tracking final, hold output failure retained, phase diagnosis complete

Previous goal turn was progress: implemented/tested/submitted current phase
diagnosis and identified the tracking pilot's raw-audited floor rejection.
This turn completed tracking session 77300, exit 0, final result
d6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de.
1,681 sources, 3,129 artifacts, 2,126.415617493 seconds after admission. Worker
terminal c2e9cd3a0e71a5210ddfdbe481d95ac13e1be4e5a404b58bb72335f870115ea6.
Launch/audit/prefix/worker bindings checked again. Both native processes ended.
Aggregate is now 22 completed, raw-audited episodes and zero verified round trips.

Hold V1 session 69725 exited 1 on its frozen compressed output ceiling. Exact
retained stream b7d386890457e4c29e91660ddf2b19c72f4ac151daf3c7713673786bd753b33d
has 1,185 unchanged decisions, frames 0–1184, 134,243,265 bytes. Failure SHA
548bad1eef69df9c2afd51accd3df9d2db17ef7aaf90939c3a45b990cd55e085.
It is an output-accounting failure, not a complete negative policy finding.
Preserved every artifact and frozen V1 source. V2 uses AST-identical replay,
the same policy/model/input/boundary, fresh initial state, 2 GiB total output
and 1 GiB compressed ceiling; memory/reserve unchanged. 15 tests pass. Submitted
13252, PID 2509963, authenticating. Details and frozen hashes:
docs/go2_residual_hold_prefix_v2_preparation_2026-09-09.md.

Implemented and tested the separate floor-conflict diagnostic. It reconstructs
the original failed registration from frame-503 admitted state and raw packet
504, without model/tracker execution or changing thresholds/decisions. It must
reproduce the same error and account for every camera candidate. 22 tests pass;
the initial two fixture serialization failures were corrected in test data and
documented. Actual completed-native admission passes. Submitted 14291, PID
2509927, authenticating. See
docs/go2_direct_flow_maze01_floor_conflict_preparation_2026-09-09.md.

Fresh hardware 78334: 81,062,461,440 RAM bytes available, 75,988,107,264 artifact
bytes free, 21,358,866,432 workspace bytes free, CPU 6.3%, all 32 affinity,
GPUs idle. Two new 8 GiB CPU analyses plus the finishing phase diagnosis fit;
no native scene is running. Supervised fixed-cohort storage gate remains unmet;
exact pip-cache approval remains pending and no deletion occurred.

Phase session 10216 then exited 0, result
6ad04101046cdcb9e855a100a446bb0206397354614608f89a03c38d80ecdf76,
launch e4132d9e76e7225924dee4208523c1f76adc38eed00668690309ae93590b54d7.
1,701 sources, all 514 complete decisions exact, unchanged public inputs/model,
model hooks removed, exact exclusive duration accounting. All source and output
bindings rechecked. Active-500 instrumented median 574.5553725 ms, all over
100 ms. Largest exclusive mean costs: selector 118.846 ms, contact 111.354 ms,
original auxiliary 81.396 ms, floor registration 73.500 ms, primary coverage
58.092 ms, motion 52.093 ms; model only 7.393 ms. Full report:
docs/go2_single_pass_receipt_phases_result_2026-09-09.md. No controlled additional
speedup or whole-loop claim. Goal active/unachieved with concrete new evidence.

## Scoped support reuse implemented and submitted for full paired verification

Previous goal turn was progress: finalized tracking, completed current phase
diagnosis, preserved the hold V1 output failure and submitted two new analyses.
This turn used the measured 111.354 ms contact cost to inspect repeated geometry
work. Added a cache restricted to one footprint, retaining original support
computations for exact posture/direction keys and returning independent receipt
copies. No cache survives candidate/observation boundaries or failures. All
existing geometry, contact rules, model, indices and controller science methods
remain unchanged; no frozen source was edited.

11 cache/controller tests pass, including actual dual-camera mapping and six
complete contact receipts. Each query makes at least five support requests with
one computation; every returned receipt equals the original. 15 benchmark tests
pass for original-decision fidelity, input/model ownership, alternating timing,
immediate mismatch stop and predecessor verification. Preparation and six frozen
hashes: docs/go2_scoped_support_cache_preparation_2026-09-09.md.

Submitted paired benchmark 98475 after fresh 54334 hardware (81,219,289,088 RAM
bytes available, 75,975,356,416 artifact bytes free, CPU 6.5%, all 32 affinity,
GPUs idle). Benchmark 16 GiB + floor 8 GiB + hold 8 GiB allowances fit. No native
scene runs. Submission is not a completed admission or speedup result.

Same floor diagnostic 14291 admitted launch
f9cb771321f9e056719cc4716ed5b8b9a3651ca2afed1f944a4c9b581206b70c.
Same hold V2 13252 remains live in inherited input authentication. Continue both
without restarting. Supervised cohort remains below its unchanged storage gate;
pip-cache approval is still pending, with no deletion. Goal active and unachieved:
22 completed/raw-audited native episodes, zero verified round trips.

## Completed floor diagnosis identifies a partial-observation hypothesis

Previous goal turn was progress: implemented/tested scoped support reuse and
submitted its full paired benchmark. This turn first verified the same three
live handles, then completed floor diagnostic 14291, exit 0, result
d2512d3241e38ac608aa2185cb62e5aafdd4fc13db962fffc68e45d82eaa992e.
All 1,686 source bindings and the output launch binding were rechecked. No
restart or inference from a missing result file occurred.

The exact frame-504 original rejection reproduces. Primary candidate count 0;
auxiliary 5,956, with 82 exceeding 3 mm. Maximum 3.392187 mm, RMS 1.491898 mm,
signed mean +1.059830 mm, signed range −1.175571 to +3.392187 mm. Full floor
anchor frame 490 is 14 frames old. Joint plane is unavailable because second
eigenvalue 0.002387664056787141 m² is below the original 0.0025 m² extent gate.
Current visual pose used auxiliary reference 503 without direct-flow fallback.
No model/tracker rerun, changed decision, native truth or later observation.

Current-packet algebra only: removing the mean signed offset with the same
transported normal gives residual extrema −2.235402 and +2.332357 mm, inside
the unchanged gate. This is a prospective height-only measurement hypothesis,
not an admitted pose or claimed physical outcome. Require original count,
all-point coherence and correction limits, keep normal fixed and never promote
partial observations into full anchors. Tests, controller integration and a
fresh first-change replay remain to be implemented before physical execution.
Full report: docs/go2_direct_flow_maze01_floor_conflict_result_2026-09-09.md.

Hold V2 13252/PID 2509963 and support benchmark 98475/PID 2511382 remain live.
Last measured CPU times 1,086.95 s and 334.65 s, respectively, during inherited
authentication; their launch/result/failure files were still absent. Kernel
logical read counters were about 2.165 TB and 665 GB, respectively; these are
logical reads, not physical disk-I/O measurements. Nested predecessor checks
are expensive, but their frozen verifiers were preserved. Goal remains active
and unachieved; supervised storage and exact cleanup approval remain pending.

## Partial-height controller tested; fresh command-boundary replay submitted

Previous status turn was a verified wait: both specific CPU processes were
confirmed live and both launch files existed, without final result/failure.
This turn completed the height-only implementation and prospective replay
harness. Lost earlier component-test output was not treated as a pass: no
pytest was still live, then fresh session 97394 exited 0 with 24 tests passing
in 10.20 s. Prefix/runner session 64762 exited 0 with 23 tests passing in 3.80 s.

PartialHeightDirectFlowController retains the original tracker and planner
methods. Only the missing-plane transport dependency and three typed pose
accessors differ. The scalar update is allowed only after the original exact
candidate conflict, with sufficient current points and insufficient two-axis
extent. It retains normal, rotation and full anchor, requires all points within
the unchanged 3 mm gate and the original total correction limits, and never
promotes partial observations into full anchors. Tests cover repeated partial
observations and original full-plane reacquisition, all-point rejection,
forged evidence, consumer routing and unchanged method code/dependencies.

The fresh replay requires complete original decisions through frame 503 and
unchanged raw tracker evidence at frame 504, validates the live partial pose
and retained full anchor, and stops at that original failure regardless of the
outcome. It does not consume observation 505. Comparison failures are retained;
tests cover changed state/input/model, truncation, anchor promotion and boundary
errors. No native execution or navigation improvement is inferred.

Actual completed tracking admission passes. Fresh hardware 86761: available
RAM 75,821,559,808 bytes, artifact free 75,850,260,480 bytes, workspace free
21,358,764,032 bytes, CPU 6.6%, 16 physical/32 logical/all affinity, both GPUs
idle. Existing hold 8 GiB + support benchmark 16 GiB + new replay 8 GiB fit.
Seven frozen source hashes and exact protocol are recorded in
docs/go2_partial_floor_height_prefix_preparation_2026-09-09.md.

Submitted session 20949, PID 2513816, fresh output
go2_partial_floor_height_prefix_v1_attempt_001. Last observed running in input
authentication, 11 CPU seconds, RSS 1,269,340 KiB. Preserve this same process;
submission is not completed admission or a positive replay result.

Same hold V2 13252/PID 2509963 now has launch
6ce4b8e6497461f8b36821674e4336db27bfe6e95155e6619b12cd7ba0691e11
and progress through frame 832. Same support benchmark 98475/PID 2511382 has
launch 2f1b337ed04947b649b7c6b4f06fc1579083c230cccf8bac63d0b7a72e8bb5f7
and progress through frame 512. Both are live; no completed result has been
admitted. No process restarted, no native scene, no deletion, no frozen source
changed. Goal remains active: 22 completed/raw-audited native episodes and zero
verified round trips. Supervised storage gate and exact cleanup approval remain
pending without blocking this scientific implementation and replay work.

## Fresh partial-height native comparison prepared behind completed replay

Previous goal turn was progress: completed47 focused tests and submitted the
fresh height-prefix replay. This turn confirmed all three same CPU handles
live, then implemented the native collector, independent raw audit, strict
prospective admission and physical-prefix comparison for the height candidate.
No native run was submitted and no replay outcome was assumed.

30 tests passed in3.85 s (28970 exit0), including unchanged original physical
and evaluation AST calculations, separate fresh models, retained worker
failure evidence, actual predecessor decision reconstruction, rejection of an
unchanged hold/negative replay, exact505-observation/501-bank native past and
exclusion of physical outcomes after command504. The comparison checks25,950
physics samples and504 preceding actual commands. New native files and protocol
are frozen in docs/go2_partial_floor_height_native_preparation_2026-09-09.md.

Source-only closure verification54421 exited0 with all1,700 current bindings
matching. Full native --preflight-only remains pending a completed positive
height replay. Hardware:73,263,136,768 RAM bytes available,75,768,889,344 artifact
bytes free,21,358,682,112 workspace bytes free, CPU9.6%,16 physical/32 logical,
all affinity, GPUs idle. Refresh before native launch. Protocol permits this
new corrective pilot in an idle native slot after positive replay/resource
admission; it does not change the supervised cohort or maze3 queue dependency.

Same hold13252/PID2509963 progressed through1152; same support98475/PID2511382
was last through512 and remains live during completion work; same height20949/
PID2513816 remains live in inherited input authentication. At the latest file
check neither result nor failure exists for any of the three; height launch is
still absent. Preserve all three processes. No native scene or deletion.
Goal active/unachieved:22 completed/raw-audited native episodes, zero round trips.

## Support cache complete; height boundary recovered pending final verification

Previous goal turn was progress: prepared and tested the native height pilot.
This turn completed support benchmark 98475, exit 0, result
93a7962cef275f170d51bcfd0387b6c195ceaf8e36f30376fd849179bf2082d9.
Independent check 38551 revalidated all 1,709 current source bindings and every
output binding. All 514 complete decisions exact, public inputs and both model
states unchanged. Active-500 medians 577.181104 ms baseline and 565.780799 ms
cached; median paired reduction 18.1989945 ms. Both order groups improve, but
all 500 observations still exceed 100 ms. Acquisition/receipt I/O excluded;
no end-to-end or native performance claim. Full recorded evidence:
docs/go2_scoped_support_cache_result_2026-09-09.md.

Static verification graph inspection 90604 (exit 0) found 26 unique functions
and 1,227 expanded direct-call paths; common ancestry is repeated extensively.
This is not runtime/disk-I/O counting. Implemented an isolated, single-call
digest scope: original first digest, guarded reuse, fresh final SHA-256 for
every cached file, final population metadata recheck and unconditional clearing.
Original verification conditions/code/defaults/closures remain; module globals
are not mutated. Initial two synthetic traversal failures were fixed, and nested
code traversal gained a regression test. Final helper20 and runner5 tests pass.
Frozen sources, failures, limits and hardware recorded in
docs/go2_scoped_verification_digest_preparation_2026-09-09.md.

Hardware 92285: RAM 74,854,051,840 bytes available, artifact free 75,698,769,920,
workspace free 21,358,641,152, CPU 6.5%, 16 physical/32 logical/all affinity,
GPUs idle. Hold8 + height8 + new verification8 GiB allowances fit. Submitted
paired verification benchmark 20770/PID2516206, launch
a6d3d860c00cc3116a67429b38ef420b298808953331d65d6812a0f760a78022.
It runs scoped then original verification on the exact completed tracking
native inputs, preserving both stages. One fixed-order pair cannot establish a
controlled speedup. No existing or prepared launcher has adopted the helper.

Same height replay 20949/PID2513816 admitted launch
9db1c429ee00d5b0e5d5c44117c57eb1a0b49ca1c6df57ff4205751357629df3.
Interim read 57623 (exit 0) found all 505 saved rows through frame504. Boundary
comparison records partial_height_admitted/controller_recovered/request_changed
all true; no terminal/failure, selected left arc [0.16,0,0.45] instead of zero.
It stopped at that observation. No result or failure file yet: final ancestry
verification remains running, so this is not completed replay admission and
does not permit the prepared native launcher to proceed yet. Keep the same job.

Same hold V2 13252/PID2509963 last progressed through1632, beyond the earlier
V1 output ceiling, without a completed result. No restart or new native scene.
Aggregate stays22 completed/raw-audited native episodes, zero round trips.
Goal active. Next: complete height verification, admit its exact changed
command and run the prepared native preflight; retain both other jobs and their
outcomes. No cleanup approval was inferred and no deletion occurred.

## Height replay complete; scoped native preflight passes and episode submitted

Previous goal turn was progress: completed support-cache evidence, implemented
and submitted paired verifier testing, and observed the height boundary without
prematurely admitting it. This turn completed both remaining verifications.

Verification benchmark 20770 exited 0, result
137867773bfe6c6eb05a125a288012ff6017aa3134f4687d7bccdc7f99c02071.
All 1,689 sources and outputs rechecked by 88182. Scoped138.461881753 s versus
original354.217828018 s on one fixed-order pair. Scoped1,673,231 requests,
122,830 unique files, 56,312,123,706 bytes hashed initially and freshly at the
end, unchanged context and no retained cache. This is not a controlled speedup
claim. Full evidence: docs/go2_scoped_verification_digest_result_2026-09-09.md.

Height replay 20949 exited 0, result
b8338e4954569251e6052356a6b74b3056a6190fc971eb5445125fe3d179a374.
All 1,693 sources and output bindings rechecked; native admission reconstructed
all505 saved decisions against the original tracking episode (49620 exit0).
504 exact preceding decisions/commands and501 forecast banks; raw tracker,
public inputs and model unchanged. At504 the 1.059830 mm height correction
retains full anchor490 and leaves all5,956 auxiliary points inside3 mm, max
2.332357 mm. Full controller recovers and selects left arc[0.16,0,0.45]; no
following observation. Full report: docs/go2_partial_floor_height_prefix_result_2026-09-09.md.

Prepared a separately named native launcher using the now-admitted scoped
verifier. Original prepared launcher remains unused and frozen. Collector,
audit, physical-prefix comparator and all scientific worker calculations are
identical; only verification execution and status labels differ. New18 tests
pass in2.04 s (48088 exit0), with frozen hashes and protocol in
docs/go2_partial_floor_height_scoped_native_preparation_2026-09-09.md.

Native preflight93182 exited0: all1,709 sources/inputs verified; 122,863 unique
files freshly rehashed after5,202,299 requests; memory/storage pass, no output
or native scene. Refresh81181: RAM74,227,920,896 bytes available, artifact
75,583,508,480 bytes free, workspace21,358,587,904 bytes free, CPU3.3%, all32
affinity, GPUs idle. Hold V2 RSS8,523,759,616 bytes; conservative16 GiB growth
reserve plus native32 GiB fits. Submitted the fresh scoped native launcher
without --preflight-only. Authentication precedes physical collection; no
completed new episode or outcome has been claimed.

Read-only duplicate-hash/metadata inventory of four completed native roots found
potential2,280,169,472 allocated bytes across4,414 duplicate groups. Payloads
were not rehashed and no files changed. This alone does not meet supervised
cohort storage requirements; no deduplication/deletion approval was inferred.

Same hold13252/PID2509963 last progressed through2496; keep its process and
artifacts. Goal active:22 completed/raw-audited native episodes, zero verified
round trips. Next action is follow the fresh native parent/worker through
collection, full raw audit and exact physical prefix, and retain the hold result.

New native parent handle65703/PID2518502 confirmed live at12 CPU seconds during
input authentication. Keep it and hold13252/PID2509963. Completed20949,20770,
93182,81181,48088,49620 and88182 handles are closed; do not repoll or restart them.

## Hold negative complete; anchored continuation submitted; height audit worker complete

Previous goal/status turn yielded new evidence: full hold V2 completed and the
height collector reached a later terminal. This turn independently authenticated
both hold results and implemented, tested, froze and submitted the next distinct
prospective candidate. No overall blocker and no navigation success.

Hold13252 exited0, result47f8ab41c12d91d9f46a1def1308218a5eead9ca6c18651683051477001db89a:
all3,004 decisions and3,000 forecast banks exact, zero changed commands, terminal
MISSION_TICK_BUDGET_EXHAUSTED at3003. Veto readout3476 exited0, result
14a6b2cc7bcc889823426ab9961444754a849cde6c3d8a60514480db82ee68c8:
all2,643 hold scores reconstructed;24 have no better allowed movement,2,619
have only better alternatives with first-point-invariant nominal path vetoes.
Independent40923 exits0 with all1,687 replay sources/two outputs and1,725
diagnosis sources/one output checked. Check4325 output was lost, so no result
is attributed to it. Full report: docs/go2_residual_hold_negative_and_veto_result_2026-09-09.md.

New anchored continuation preserves original first-point recovery, then anchors
later raw model displacements at the corrected first point. All original
short-horizon utilities, phase/current/surface gates and eight full-radius path
segments remain. Later corrections are explicitly uncalibrated; raw model
predictions remain residual targets.23 component tests44835 passed; initial
prefix test25265 found an incorrect predecessor status label, fixed before
freeze. Final45 prefix tests86237 pass4.15s, including missing segments, reduced
radius, hidden veto, disconnected path, complete state and prospective stopping.
Frozen7 sources and hardware10226 are in
docs/go2_residual_anchored_continuation_preparation_2026-09-09.md.

Submitted19984/PID2523206, fresh exclusive root
go2_residual_anchored_continuation_prefix_v1_attempt_001. Same model/native2
inputs, one CPU replay,16 GiB plus conservative native32 GiB allowance,2 GiB
output and40 GiB reserve. Actual RAM80,594,472,960 and artifact73,061,388,288
bytes pass. The runner rechecks after admission. Last observed live in input
authentication; no launch/result or positive intervention assumed. Keep process.

Native65703/PID2518502 remains live during final ancestry/output verification.
Its single worker2519242 has finished with
PARTIAL_FLOOR_HEIGHT_MAZE01_SCOPED_VERIFICATION_COLLECTED_AND_RAW_AUDITED.
Worker records strict visibility pass, unchanged model, exact505-observation
physical/public prefix and no round trip. Root result/failure absent at last
check, so aggregate remains22 finally admitted episodes plus this raw-audited
worker awaiting parent completion. Zero verified round trips.

Saved656-row stream81160 exits0: frame504 requests left arc[.16,0,.45]; first
terminal644 SENSOR_OR_MODEL_FAILURE, reason 'same-episode current visual evidence
required'. Frame644 inspection30364 exits0: outer evidence is current typed
partial-floor-height pose; inner original visual evidence is CURRENT_VISUAL_POSE,
current joint frame644 and terminal_failure=None. Full floor anchor633 retained.
This suggests an unhandled typed-evidence consumer, not established loss of
visual tracking; investigate the exact call path after completed result admission.
No frozen implementation was changed or native episode restarted. Next: finish
parent verification and native result report, follow19984 through its prospective
boundary, diagnose the644 consumer failure. Storage-limited supervised cohort
and unanswered exact pip-cache cleanup approval remain unchanged.

## Height result admitted; positive anchored prefix and verified storage duplicates

Previous goal turn made progress by submitting the read-only duplicate inventory.
Native65703 now exited0, result
32edbb748e04e18816e0b0fb265f465706ac8c8684849d73a091321f79a3b07f.
Independent89804 checked all1,709 sources and3,975 artifacts. Full raw replay,
strict visibility and505-observation/25,950-physics/501-bank exact common prefix
pass. Changed left arc504 completely dispatched.656 observations,655 completed
commands, no cell crossing, no goal arrival, no terminal quiet and no round trip.
Aggregate now23 completed/raw-audited native episodes and zero round trips.
Full report: docs/go2_partial_floor_height_maze01_native_result_2026-09-09.md.

Correction: the terminal retains accepted controller tick644 but is in saved
observation645. Successful644 typed pose validates after restoring serialized
identity tuples (32895). Actual645 (91145) has evidence=None, raw visual terminal
failure/current_pose=None, NO_QUALIFIED_REFERENCE and MEASURED_BRIDGE_BUDGET_EXHAUSTED.
Direct-flow failure: 'bounded measured bridge exhausted without anchor observation'.
Withdraw the earlier suspected typed-consumer integration fault; it came from
reading successful644 instead of failing645. No frozen source changed.

Anchored replay19984 exited0, result
3cbd24abad8a6c70565648977ce8482df37b4c90b8a2e28c728799910cf402b5.
181 observations/178 banks,180 unchanged prior commands; first change180 hold
to left arc[.16,0,.45], no following old observation, no terminal/failure,
unchanged model/public/state. Independent1482 checked1,739 sources/two outputs.
Positive prospective boundary only; fresh native/audit/physical-prefix remain.
Full report: docs/go2_residual_anchored_continuation_prefix_result_2026-09-09.md.

User asked about recreatable training-data cleanup and the exact disk blocker.
Measured free artifact capacity~68 GiB versus frozen supervised cohort73 GiB;
about5 GiB short. Maze3 tracking follows that cohort. Original training admission
still requires full family/switch raw collections and fits. Metadata98458 found
potential7,219,273,728 duplicate allocated bytes in their53,856 declared bindings.

Read-only inventory42175 exited0 after18.694867 s, result
732b0a95a98d598843e1d56f85612093b3d78e3e1ed145f4e27d24f2ceb6d577,
proposal0a0b273b47b4d0df3b0ce76104b0cb348574e339b6dfa0ee235ab3e15a890bb7.
Root go2_training_artifact_duplicate_inventory_v1_attempt_001. Fresh hashes of
32,733 files/9,611,435,777 bytes confirm5,523 groups and27,210 duplicate copies,
potential7,219,273,728 allocated-byte saving (6.72 GiB). Independent1482 checked
all inventory source/output bindings. No input mutation or regeneration claim.
Inventory grouping tests9 pass58339. All previous jobs are now closed.

Prepared an exact journaled hard-link consolidation preserving every original
path and byte. Synthetic replacement/failure tests initially10 pass83754; full
preflight27043 stopped because user-systemd3119 descriptors are kernel-protected.
Revised the unpublished operation to reject identified development interpreters
and visible artifact users, fail if ownership/command identity cannot be read,
and record descriptor gaps for identified unrelated processes without claiming
global quiescence. Expanded14 tests pass83383. New preflight86724 is running;
no consolidation output or authorization exists. Original proposal/inputs intact.

Operation sources currently: runner86f7074fedf1725458f7519187936afdea14dd2a76d7443e9a9480daf01e00d5,
testsdb65976d9924f42946d920582e5fb1f270dd6794f65f2af99e743a80a976e56b,
protocol3f4126ad5e45a75d54b7ea0aa84ef20bd54ea603c6156520990a6c6ca1178d2e.
Exact protocol: docs/go2_training_artifact_hardlink_consolidation_v1_2026-09-09.md.
Finish preflight and prepare concrete user approval; do not infer it from storage
questions. No pip/GSD cleanup is included. Native workers must remain finished
during conversion. Goal remains active and not blocked overall.

## User approved exact storage consolidation; transaction running

Final revised preflight86724 exited0 with every candidate hash/metadata matching.
No other known development worker or accessible training descriptor was present;
four identified unrelated services had descriptor visibility gaps, recorded
without a universal-quiescence claim. Free artifact72,989,745,152 bytes; verified
potential saving7,219,273,728 bytes would clear the fixed73 GiB cohort gate.
Exact ready review: docs/go2_training_artifact_hardlink_review_2026-09-09.md.

After that concrete proposal, user explicitly replied 'do it'. Recorded exact
approval, reviewed source hashes, inventory/proposal identities and excluded
scopes in docs/go2_training_artifact_hardlink_authorization_2026-09-09.json.
This authorizes only the27,210 listed hard-link replacements, retaining every
original path and byte. It does not authorize pip/GSD or other cleanup.

Submitted approved runner30695 with --approval-json naming that authorization.
Latest output TRAINING_CONSOLIDATION_GROUP0; same process is running. No native,
replay or dataset consumer may start before it finishes. Exclusive transaction
root: go2_training_artifact_hardlink_consolidation_v1_attempt_001. Preserve its
intent/completion journal and any failure; no automatic retry or scope expansion.

Next finish complete original-artifact verification and measure actual capacity.
If the unchanged73 GiB gate passes, start the already prepared supervised
layouts1,2,3 cohort in that order with original runner/protocol. Anchored-native
preparation is separate source-only future work; no such files or scene were
created yet. All original science failures remain preserved.

## Approved consolidation complete; fixed supervised comparison submitted

Consolidation30695/PID2528150 exited0, result
7f0b0ff8cc7ace44a2542738107f8dd3a7b6ef454b942e240c51ad8d80913297.
All27,210 approved replacements retain original paths/content. Original three
training result/source identities and all53,856 artifact bindings reverified.
Independent42200 verifies complete paired transaction journal and output/source
hashes;46398 checks every32,733 affected name's actual canonical inode, link
count/mode/ownership/size. Actual duplicate allocation recovered7,219,273,728
bytes (6.723472595 GiB). No unauthorized cleanup or data regeneration.

Volume free-space growth greatly exceeded this operation; do not attribute
the recorded507,672,141,824-byte filesystem delta to our6.72 GiB consolidation.
At42200 the volume reports709,855,936,512 free bytes; the original fixed cohort
gate78,383,153,152 bytes passes. RAM71,053,438,976 available, one native worker
allowance32 GiB, CPU3.3% with all32 affinity; no competing development worker.

Submitted original supervised runner19047/PID2534319 with exact learned result
a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720.
Last observed live at28 elapsed/27 CPU seconds during input authentication.
Exclusive root go2_supervised_rollout_mazes_v1_attempt_001; no launch or native
outcome assumed yet. Keep this process, fixed layouts1,2,3 order and all audits.
Storage blocker resolved. Full report:
docs/go2_training_artifact_hardlink_consolidation_result_2026-09-09.md.

During consolidation only source work was performed: drafted separate anchored
native collector, audit and181-observation/9,750-physics/178-bank prefix comparator.
Those three files remain untested/unfrozen, with no launcher or execution
protocol. Status recorded in
docs/go2_residual_anchored_continuation_native_preparation_2026-09-09.md.
No such native run started. Next continue supervised comparison and prepare
anchored native tests/launcher independently; keep later maze3 queue dependency.
All prior handles except19047 are closed. Goal remains active:23 completed native
episodes and zero verified round trips, with no overall blocker.

## Supervised worker live; anchored native implementation and tests complete

The fixed supervised comparison19047/PID2534319 has launched with launch SHA
49a182da3d795f1b31910fb2b6123732db66349dfc011d768bb72fc9732a09e3.
Its first native worker2535576 is live; the first case is
full_supervised_rollout_novel_maze_01. No completed cohort outcome is claimed.
Preserve this process and layouts1,2,3 order.

The anchored native collector, audit, exact-prefix comparator, separate launcher,
protocol and two focused test files are now frozen.25 native-prefix tests and29
launcher/admission/failure-retention tests pass. Full source identities and
resource assessment are recorded in
docs/go2_residual_anchored_continuation_native_preparation_2026-09-09.md.
Read-only preflight35981/PID2537733 is live, authenticating the fixed actual
prefix/prior/source/environment chain. It creates no native output or scene.
Queue remains supervised cohort, prepared direct-flow maze3, then anchored maze2.

Additional authenticated readout50395 narrowed the prior height-run failure:
last anchor-qualified frame634 was not retained (18/34 feature overlap exceeds
the half-overlap retention trigger);635 through644 consumed ten measured bridge
increments while retained references626 through633 failed in both cameras.
At645 both direct-flow increments qualify, but the bridge budget is exhausted.
The next tracking investigation concerns retaining already anchor-qualified
views and anchor matching, with original bridge and rigid gates preserved.
Details: docs/go2_partial_floor_height_maze01_anchor_loss_readout_2026-09-09.md.
No tracking policy change or new navigation success is claimed. Overall goal
remains active, with23 completed/raw-audited native episodes and zero round trips.

## Anchored native preflight passed; recent qualified reference replay submitted

Anchored native preflight35981/PID2537733 exited0. All1746 source bindings and
the original transitive verification conditions passed. Its per-call scope
freshly hashed137,894 unique files before and after,64,656,470,245 bytes each,
with no retained cache/global mutation. Memory and storage admissions pass;
no output or scene was created. The frozen anchored pilot is ready after the
supervised cohort and prepared direct-flow maze3.

A separate recent-qualified-reference observer/controller and causal prefix
replay are now frozen. It retains only the immediately preceding already
anchor-qualified view, alongside the original bank, and tries it only after
original reference missingness. It preserves original conflicts, raw fitting,
camera and continuity gates, and never retains a bridge/floor/native pose.
An explicit single-dispatch method copy is tested against original calculations
with the original joint witness wrapper retained.16 component/raw-image/public
pose tests,17 causal comparison tests and9 replay/verification tests pass.
The initial subclass-dispatch test failure was fixed before freezing/execution.

Full source identities, protocol, evidence and measured hardware allowances:
docs/go2_recent_qualified_anchor_preparation_2026-09-09.md. Actual completed
height-result/audit admission19432 passed. Submitted replay23895/PID2543628,
live in initial input authentication; preserve the same process and exclusive
go2_recent_qualified_anchor_prefix_v1_attempt_001 output. It will consume no
observation after the first changed command or either terminal, maximum646.
No replay result or physical recovery is claimed yet.

Supervised19047/PID2534319 and first native worker2535576 remain live. The first
case has a complete decision receipt1272; no completed audit or navigation
outcome is inferred. Only those two experiment handles are live; earlier
preflight, inspection and test handles are closed. Goal remains active:
23 completed/raw-audited native episodes, zero verified round trips, no overall
blocker. Next finish both live jobs, preserve every outcome, and advance the
recorded native queue when its current owner finishes.

## Recent reference replay launched; future native calculation drafts tested

Replay23895/PID2543628 passed initial input verification and launched with SHA
9c12a4d3683d6bc6f61e837bec7c9afb8441b95a6cf9ffc20086d44bf30c74f7.
All1718 sources are bound; the scope processed5,267,164 digest requests across
126,848 unique paths, fresh initial/final hashes each covering58,830,880,547 bytes,
with26 isolated functions and no retained cache/global mutation. Launch hardware
has63,643,029,504 bytes RAM available,704,831,873,024 artifact-volume bytes free,
full32-CPU affinity,16 physical cores and4.3% CPU busy. It runs alongside the
single supervised native worker with the recorded8+32GiB RAM allowances.

The live replay has reported frame288 without a comparator failure; no completed
outcome or intervention boundary is assumed. Supervised19047 remains live; it
advanced past decision receipt1516. Existing runner already records paired
three-layout outcomes against complete raw audits, so no duplicate comparison
readout was added. Retention native collector/audit drafts pass three comparisons
against the original physical/evaluation calculations. They are unfrozen and
have no launcher or completed positive replay admission yet. Details:
docs/go2_recent_qualified_anchor_native_preparation_2026-09-09.md.
Preserve both live attempts and the existing native queue;23 audited native
episodes and zero verified round trips remain the completed evidence.

## Retention replay verified positive; fresh native pilot frozen and preflighting

Replay23895 exited0, result
6e25c6c561473b60966a1a47388a01b48ab3e547da0c8ab13aa1e267b9fad302.
Independent61685 verified1718 sources/two outputs and reconstructed all646 saved
comparisons for native admission. First extra-reference/decision difference635;
two attempts and one qualified view;635 complete original decisions and642 raw
banks compare. Commands remain identical through644. At645 the new controller
requests left turn[0,0,.45] with no terminal/failure, while the original has
SENSOR_OR_MODEL_FAILURE and zero request. No following observation is consumed.
This is replay evidence only. Full result:
docs/go2_recent_qualified_anchor_prefix_result_2026-09-09.md.

Authenticated evaluator-only pose readout41285 used observations0 through645
within33000 prior physics samples. Candidate registered XY error at645 is
6.540261mm; maximum across646 observations7.669866mm. These measured simulation
errors are not calibrated bounds or post-command physical outcomes.

The complete native pilot and its eight new source/test/protocol files are now
frozen, pinned to the verified replay and exact645 boundary. Three physical/
evaluation calculation checks,23 native-prefix tests and28 final worker/admission
tests pass (54 total). All eight hashes, hardware and queue are in
docs/go2_recent_qualified_anchor_native_preparation_2026-09-09.md.
Read-only preflight71438/PID2547935 is live in input verification; no new native
output/scene is assumed. This is separate from the ready anchored-continuation
maze2 pilot. Queue remains supervised layouts1,2,3, direct-flow maze3, anchored
maze2, then retention maze1.

Supervised19047/PID2534319 and first worker2535576 remain live; decision receipt
2430 is complete. No new native audit or navigation outcome is claimed. Only
19047 and71438 are live experiment handles; replay23895, independent61685,
pose41285 and focused test handles are closed. Goal stays active with23 audited
native episodes and zero verified round trips; no overall blocker.

## Retention preflight passed; fixed native queue launched

Retention preflight71438 exited0:1726 source bindings, all original verification
conditions, both resource admissions pass. No output or native scene. Its
preparation document now records the completed scope and hardware evidence.

A separately named scheduler now owns the recorded order after the original
supervised cohort: direct-flow maze3, anchored-continuation maze2, then recent
qualified reference maze1. All original runners, models and scientific settings
remain frozen.37 focused scheduling/authentication tests pass; real-data source
coverage35090 verifies all1685/1746/1726 prospective bindings. Queue preflight79265
passed with1803 sources. Details and frozen hashes:
docs/go2_prepared_native_queue_preparation_2026-09-09.md.

Scheduler37343 is live; launch SHA
651a5815275ecfdd20c5ff6ff7f4d8cdbe4b5c52bbcaf3ad124a6ac97ffd2fb7.
Its initial WAITING event confirms the original supervised parent2534319 and
worker2535576 still own the only native scene. Original supervised19047 has no
completed first-case progress/audit/result yet. Do not separately launch another
native scene or restart the queue. The scheduler will authenticate the complete
original cohort before advancing, retain scientific negatives, and stop on any
process/integrity/raw-audit/prefix failure without automatic retry.

Latest artifact free space699,909,349,376 bytes and workspace free21,358,022,656
bytes. Storage is unblocked. Only19047 and37343 are live experiment handles;
71438 and all scheduler test/read-only verification handles have closed. Evidence
remains23 raw-audited native episodes, zero verified round trips; goal active.

## First supervised collection ended; original raw audit still pending

The next continuation directly polled both original handles19047 and37343;
both remain live. Native worker2535576 is running, parent2534319 is waiting,
and scheduler2551088 has continued WAITING events for those same owners. No
original failure, first-case audit, worker terminal, cohort progress or final
cohort result exists at this inspection. This is a verified live wait.

The first case's persisted collection result now exists at
go2_supervised_rollout_mazes_v1_attempt_001/
full_supervised_rollout_novel_maze_01/result.json, SHA
07f3fdd33410bf115e0612734cbd097d15e00ac1042761f58132e0bc1b55e082,
with bytes unchanged before/after the narrow read. It records3014 observations
and decisions,3013 completed commands,151400 physics samples,10 terminal zero
ticks and MISSION_TICK_BUDGET_EXHAUSTED. No physical/acquisition stop is recorded;
the observed mission remains OUTBOUND with no arrivals. Status is explicitly
MEASURED_FLOOR_TRANSPORT_MAZE_TERMINAL_AUDIT_REQUIRED. This is a provisional
collection receipt, not a completed raw audit or verified navigation outcome.
Do not add it to the23 audited episodes or infer a paired training advantage.

Current resource monitor reports58,497,798,144 bytes available RAM,
699,908,354,048 artifact bytes free, full32-CPU affinity and4.5% CPU busy.
Original audit and fixed queue continue; no source changes, retry, extra scene
or new policy variant is needed before their evidence completes.

## First supervised raw audit saved; final worker verification pending

Original19047 and scheduler37343 remain live. At8858.32s of the supervised
cohort, first-case audit and physical-prefix comparison have been written.
The original worker2535576 remains running in its subsequent verification;
worker terminal/cohort progress/final result are not yet present. Their absence
is not a process failure or retry reason. Earlier continuations polled these
same live handles and observed increasing native-worker CPU time throughout.

Saved first-case audit SHA
4bd54d5454e2665cb354817be2eef447897fa5d3ab4d65541d9368fdcf37de60:
raw sensor reconstruction, model-command replay, command audit and unchanged
model all true; strict physical visibility true, no hard measurement failures.
It records1501 right turns,1498 left turns and1 hold across3000 selections,
no observed/native arrivals, no cell crossings, and only start cell[-1,0].
No return traversal, terminal native quiet or verified round trip. This is a
saved audit receipt pending the original terminal artifact/source checks, so
the confirmed aggregate remains23 audited episodes until those complete.

Saved prefix comparison SHA
b585690d9af0400fab25bc6c5109ab1c0f2adce4634dfa6b1208c2d6402ef0bc:
900 exact prior physical samples,4 public observations, all prior commands and
observed state exact, one paired forecast bank, original JEPA decisions exact
and complete candidate decisions equal the prospective supervised replay.
At frame3, original left arc[.16,0,.45] becomes completed supervised right
turn[0,0,-.45]. No following original physical outcomes are compared or inferred.
Both saved JSON files were byte-identical before/after the narrow inspection.

Available RAM68,818,173,952 bytes and artifact free699,895,214,080 bytes.
After the first worker terminal authenticates, diagnose its repeated turning
while the remaining fixed cohort runs; preserve all three cases and the queue.
No training-objective advantage, reliable navigation or goal completion follows
from longer tracking survival without translational exploration or arrival.

## First supervised episode fully authenticated; all-selection score readout submitted

Original19047 reported first-case status
SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED, verified_round_trip false and no
worker failure. Completed terminal SHA
730b2d8a20d680066854427e6a660c58346d296f6073ee12ec3ae984574380e5;
first cohort-progress SHA
d55cafb5cab562a258a19e12840385653e3cbee3d298f0ddc1e0f5758aa1a30e,
with original ordered remaining layouts[2,3]. Original first worker2535576 has
exited; second worker2563586 is active and the resource monitor names maze02.
Keep original19047 and scheduler37343; no new native scene was independently
launched. Scheduler2551088 continues waiting for the complete original cohort.

Independent14939 exited0: all1672 original source and18119 first-case artifact
bindings verified before/after, exact launch/terminal/progress identities,
unchanged assigned supervised model, raw audit, physical prefix and worker log
confirmed. The original worker completed its transitive input verifier;14939
did not independently reexecute that expensive transitive verifier. Worker wall
9027.174647341017s, maximum RSS10,206,183,424 bytes. Aggregate now24 completed
raw-audited native episodes, zero verified round trips. Strict visibility passes,
no hard measurement failures, no arrivals/crossings,3000 turn/hold selections.

First-case decision stream SHA
8a00d0d75f0d43b5abe36c8b6e64568cf97f423b7d1d7eb2d28688ac42ff347d;
command tape SHA
4187da563d5ce49c385c500bd4ead5aac18f6d1e21f794b7abbcba0f4822f000;
timing stream SHA
4e506ed11453a01dc05fbccbba48e918f30d38d7a2aab66b3169420197ea40cb.
Read-only41024 reverified the decision stream before/after and reconstructed
the complete original score output exactly at frames3 through8. All translation
candidates there are feasible; contact penalties lower their utility despite
predicted displacement. Do not extrapolate this six-frame finding to all3000.

Prepared a separate bounded readout of all3014 observations and3000 selections,
using the original scorer to reconstruct every complete selection and to compare
saved translation vetoes and score components. No model/controller changes or
alternative physical-outcome inference. Protocol:
docs/go2_supervised_maze01_turn_scores_v1_2026-09-09.md,
SHA d45cefa182edb46baf1852207398a20368fce15f84cc3895b7b6f9ce19949e2d.
Runner scripts/read_go2_supervised_maze01_turn_scores_v1.py,
SHA 8c7119b3711bd27541275d58d746f48c8763073d187e04dc2c7e91b1f3941dc0.
Submitted with the original Genesis interpreter and full explicit numerical env;
keep its exclusive go2_supervised_maze01_turn_scores_v1_attempt_001 output.
The runner records fresh hardware before source/output authentication and its
8GiB CPU readout allowance fits alongside the single native scene. Initial
verification14939 hardware:77,543,718,912 bytes available RAM,
699,888,844,800 artifact free,4.1% CPU busy/full32 affinity, card1 GPU5%.
No score-readout result is assumed yet. Reliable navigation remains unproven.

Readout93282/PID2564405 passed initial input/source authentication and launched
with SHA49d9ec85580bb64ec2eb345381c4ce580c1ac1b38f171c75223b9e2b92570554,
1674 source bindings. Original first-case sources plus only the new runner and
protocol are bound. Launch RAM76,407,267,328 bytes, artifact free699,251,703,808
bytes,4.0% CPU busy/full32 affinity; card1 GPU6%. Subsequent direct inspection
finds the same readout live at173.2 CPU seconds and1,094,152,192 bytes RSS,
without result/failure. Native monitor confirms maze02 with75,930,021,888 bytes
available RAM,698,785,996,800 artifact bytes free and7.5% CPU busy. Keep all
three live handles19047,37343,93282;14939 and41024 are closed. Wait for the
complete aggregate before choosing a scoring change or preparing another pilot.

## Every supervised maze1 turning selection loses translation through contact cost

Readout93282 exited0, result SHA
e546494d75770c174453663d14cb5d38754a8f994bf89ba847ab425dbd3ddf8f,
wall224.61413529515266s. All3014 observations and3000 actual selections were
included, each complete score output reconstructed exactly with the original
scorer, and all source/input artifacts verified before/after. Every action was
feasible at every selection; there are no translation phase/surface/path vetoes.
All2999 selected turns had a feasible translation candidate with greater
predicted geometric potential progress but lower utility after the800ms contact
penalty. Forward alone meets that comparison at all3000 selections, including
the one hold. Actual consecutive turn direction changed1003 times.

Independent24281 exited0: result identity, all1674 sources, launch binding,
original terminal/audit identities and all population/aggregate identities pass.
It authenticates the completed all-selection readout; it does not claim a second
full raw-controller reconstruction or alternative executed outcomes. Details:
docs/go2_supervised_maze01_turn_scores_result_2026-09-09.md.

The inherited scorer values100ms pose progress but charges800ms contact risk.
The earlier executed-waypoint protocol deliberately retained that contact term
while changing pose utility; this is an explicit prospective variable to test
next, not a retroactive correction to the original study. Next prepare a
separately named causal replay using contact cost at the actually committed
100ms horizon while preserving all existing800ms path constraints, surface and
phase vetoes, model forecasts and observer/mission/memory state. Stop at the first
changed command; do not infer its physical outcome. Original native queue and
supervised cohort stay frozen. Only19047 and37343 remain live experiment handles.
Aggregate remains24 raw-audited episodes, zero verified round trips; goal active.

## Prospective100ms contact-cost replay prepared, tested and submitted

New component/controller/comparator/runner preserve the original ViewReentrySelector,
MeasuredFloorTransportController observe/advance, supervised model, causal pose
utility, coefficient1.2 and all800ms geometric/surface/phase vetoes. Only intermediate
waypoint contact cost uses predicted step1 instead of step8. Full old scores and
forecasts remain receipts. No change to final-goal/reentry behavior or actual queue.

All41 focused checks passed. Read-only preflight43825 exited0 with1721 sources,
137,864 unique files,944,482 digest requests,64,214,606,916 bytes hashed initially
and freshly again finally,23 isolated original verifier functions and no imported
globals changed. Full hardware admission passed alongside native maze2. Freeze
identities and protocol:docs/go2_supervised_commitment_contact_preparation_2026-09-09.md
and docs/go2_supervised_commitment_contact_prefix_v1_2026-09-09.md.

Submitted actual replay93799 with exclusive
go2_supervised_commitment_contact_prefix_v1_attempt_001 output. It authenticates
inputs again before launch, then stops at the first changed request or either
terminal and verifies again afterward. No completed replay or changed physical
outcome is assumed yet. Native19047/worker2563586 remains in maze2 and37343 waits
for the complete original cohort; no independent native scene was launched.

## Contact horizon changes the first supervised command; causal replay verified

Replay93799 exited0, result SHA
37b29828635e88fab77f81447f6a05b890911426fc3478d8aac451426a229de0,
launch SHA a412b59f4f865daf8920bd1a3f894a4fa3f685985c4d1130c1ea38cc19407e18.
Four observations, three identical prior commands, one matched full forecast
bank: frame3 right turn[0,0,-.45] becomes right arc[.16,0,-.45]. No terminal or
failure, no following recorded observation consumed. All original feasibility
checks and complete observed-state receipts match; model and public arrays
remain unchanged. Final original input/source verification passed. Reported
178.8003472359851s includes final verification and is not controller latency.

Independent49502 exited0: all1721 source and output bindings verified before/after,
original worker/stream/tape identities and all four saved complete comparisons
checked against their actual predecessor. No second full model execution or
transitive input verifier is claimed. Details and limits:
docs/go2_supervised_commitment_contact_prefix_result_2026-09-09.md.

Next prepare a separately named fixed supervised commitment-contact maze1 native
pilot, requiring the same first physical/public prefix and full raw audit. It
must run after the current frozen queue, which remains unchanged. Only19047 and
37343 are live experiment handles;43825/93799/16907/49502 are closed. Native
maze2 most recently reached tick1714.
Aggregate remains24 raw-audited episodes and zero verified round trips; no
navigation benefit follows from this causal replay alone. Storage stays unblocked.

## Supervised commitment-contact native pilot verified and waiting after the queue

Concrete progress this turn: prepared the fixed fresh maze1 native execution,
complete raw audit, physical/public causal-prefix comparison and completed-command
requirement. All97 focused checks passed. Read-only native preflight88313 exited0
with1822 sources and137,969 unique files, original transitive verification
conditions preserved, fresh final hashes, and hardware admission passed. Native
runner SHA9cf3a4d103dce745d1909548e99316ba7ea95c4c6aa8ed989df83f1a7de75439;
case full_supervised_commitment_contact_maze_01. It preserves the same supervised
model/correction and all original physics/sensing/audit calculations; only the
already verified contact-cost controller is changed. The full native episode
remains unexecuted, so no physical benefit is claimed.

Prepared/tested a separate one-shot waiter, without modifying the current queue.
All86 waiter/native integration checks passed; waiter preflight34358 exited0 and
independent74104 proved its1825-source union covers every original1822 native
binding. Submitted14895/PID2571800, launch SHA
4631acdc04f710c93f84d161ee428b7c8003ff7d663e4e75f0880697401f529b.
It is authoritatively live and WAITING for original queue PID2551088/start
tick130406252; no new native scene was launched. It will authenticate the complete
existing queue and run exactly one frozen contact-horizon native pilot afterward,
with full source/input/output and raw-audit verification and no retries.

Preparation and freeze details:
docs/go2_supervised_commitment_contact_native_preparation_2026-09-09.md.
Keep live handles19047 (supervised, worker2563586 on maze2),37343 (original
queue),14895 (new waiting launcher). Closed this turn:65774,67417,88313,40962,
84344,34358,74104. Most recent maze2 timing2520; complete second-worker result
was not yet present. Aggregate remains24 completed raw-audited native episodes
and zero verified round trips. Goal stays active; storage is not a blocker.

Independent15486 subsequently exited0: exact waiter launch and1825 sources
verified; waiter2571800 and original queue2551088 both authoritatively live,
no contact-cost native output yet. Original queue's later event correctly
excludes the waiting observer from competing native owners. Keep14895 live;
15486 is closed. The added waiter does not block the original queue's progression.

## Supervised maze2 collection finished; raw audit still pending

The preceding short turns were verified waits on the same original worker.
Read-only monitor86769 tracked PID2563586 from tick2754 to3013 with increasing
CPU time and unchanged process identity. It exited0 when the collection result
appeared; the native worker, original queue2551088 and waiter2571800 stayed live.
No queue/waiter/cohort terminal failure file appeared.

Narrow inspection83307 exited0: original supervised launch SHA
49a182da3d795f1b31910fb2b6123732db66349dfc011d768bb72fc9732a09e3,
unchanged assigned maze2 supervised model and the new collection JSON verified
before/after. Collection result SHA
f4c0f49e84ce0cbce230fa128e473688283f32b491778ccef78d794287b7b1c8,
at go2_supervised_rollout_mazes_v1_attempt_001/
full_supervised_rollout_novel_maze_02/result.json. Status remains
MEASURED_FLOOR_TRANSPORT_MAZE_TERMINAL_AUDIT_REQUIRED.

Reported3014 observations/decisions,3013 completed commands,151400 physics
samples,10 terminal-zero ticks and MISSION_TICK_BUDGET_EXHAUSTED. Physical and
acquisition stop fields are null. Final observed mission receipt is OUTBOUND,
no arrivals, observed goal distance4.669717084121626m, budget terminal at3003.
This is a collection summary, not an independently raw-audited navigation outcome.
At inspection, maze2 audit, physical-prefix comparison and worker terminal were
all absent. Do not increase the audited episode count or infer physical clearance,
pose accuracy, cell crossings or any round trip from this intermediate report.
The original worker is still performing post-collection work; preserve it.

The collection's explicit storage fields also exposed a prose error in the new
contact-native protocol: actual original/frozen code uses40GiB reserve +10GiB
collection +1GiB persistence, not the paragraph's8GiB+3GiB split. The51GiB total
and all runtime parameters are unchanged. An explicit correction is recorded in
docs/go2_supervised_commitment_contact_native_preparation_2026-09-09.md;
no frozen source/protocol bytes or running definition were changed.

Keep19047/worker2563586,37343 and14895 live.86769 and83307 are closed. Aggregate
remains24 completed raw-audited native episodes, zero verified round trips.
Next authenticate the completed maze2 worker when available, then retain the
original ordered maze3 cohort and both existing queues. No overall storage blocker.

## Complete saved maze2 tape contains only in-place turn requests

Read-only15241 exited0 after persistence made the complete command tape
available. Original launch and collection identities were verified before/after;
the original action-menu source binding was checked. Tape SHA
6a3b9f370ddccc8854a08191c539581be010c48765bf5cd4b4f7178ff43e1ccc.
All3013 saved requests are ordered, marked completed, and have the exact original
pre/post sample indices749+50i and799+50i. Roles/counts:3 warmup zeros,3000
navigation requests,10 terminal-zero drain requests. The navigation population
is exactly1500 right-turn requests and1500 left-turn requests; no forward, arc
or hold navigation request is present. First navigation request at3 is
[0,0,-.45]. This is a full saved-tape readout, not a physical-command audit or
fresh model replay. It does not establish absence of physical drift or identify
every maze2 score's causal mechanism. The already queued contact-horizon native
test remains fixed; no outcome-adaptive definition change is made.

Worker2563586 remains the same live process. Over the current monitoring interval
its CPU time advanced from1:15:59 to1:18:17 and read-character count from
386,440,991,075 to389,272,748,052, with no further collection writes. This supports
active post-collection processing rather than a stalled worker; it is not an
audit completion claim. At the last check maze2 audit, physical-prefix and worker
terminal files were still absent, as were cohort/queue/waiter failure files.
Keep19047,37343 and14895 live.15241 is closed. Count remains24 raw-audited native
episodes and zero verified round trips; wait for the original worker terminal.

Subsequent verified waits confirm the same worker start tick130800823. A narrow
read-only check of its /proc descriptors, matching only the exact known maze2
decision-stream path and excluding protected names, observed compressed-stream
read position advance13,897,728 to22,548,480 bytes of240,900,186 while CPU ticks
advanced499,027 to507,566. This confirms ongoing input processing without
modifying the worker or reading arbitrary artifact trees. Read-ahead/compression
mean these positions are not completed-frame counts or an audit percentage.
Audit, physical-prefix and worker-terminal outputs are still absent; no completed
audit or additional verified episode is claimed. Both waiters remain live.

## Supervised maze2 audit saved on 2026-09-10; worker final checks pending

The intervening turns were verified waits on original worker2563586/start
tick130800823. Its exact decision-stream read cursor advanced to240,900,186
bytes and the descriptor then closed while CPU time continued increasing.
The original audit and physical-prefix comparison have now been saved under
go2_supervised_rollout_mazes_v1_attempt_001. Audit SHA
c001cde994dd2d1ac8011fcdc9485a5a03d6cac69a2499d9e7e13cf7434e725c;
prefix comparison SHA
b585690d9af0400fab25bc6c5109ab1c0f2adce4634dfa6b1208c2d6402ef0bc.
Narrow read-only inspection checked these identities and original launch SHA
49a182da3d795f1b31910fb2b6123732db66349dfc011d768bb72fc9732a09e3
before/after. A separate read-only check verified all1672 launch source bindings.
Neither check independently reran the raw controller or transitive runtime
verifier; the original worker is completing its own final checks.

Saved raw sensor reconstruction, model/command replay, command audit and
unchanged-model flags all pass. The selected-action population is1500 right
turns and1500 left turns. Native evaluation reports no arrivals or cell crossings,
only start cell[-1,0], no round trip, and terminal native quiet pass. Physical
stop is null. The four-observation/900-physics-sample prefix passes with one
paired forecast bank and the expected frame3 JEPA left arc to supervised right
turn intervention; the candidate intervention command is completed.

Strict physical visibility FAILS:3 of3014 primary footprint checks, at frames
943,1889,2319. Each has one boundary bad ray and a failed original1mm score;
maximum original errors are1.2788373837807865m,1.3322491903769988m and
1.397061712255538m respectively. All three have zero stable-interior bad rays,
stable-interior metric pass and no near-occlusion failure. No auxiliary frame
fails visibility and hard_measurement_failed_frames is empty. These diagnostic
distinctions do not waive the strict failures or grant visibility qualification.

At this inspection the maze2 worker terminal and cohort progress-after02 files
are not yet authenticated; do not increment the completed-worker count. Keep
19047/worker2563586,37343 and14895 live with unchanged definitions and order.
Aggregate remains24 completed raw-audited episodes and zero verified round trips.
Next authenticate the complete worker record and its output bindings, then retain
the original maze3 and subsequent queued experiments. Storage remains unblocked.

## Supervised maze2 worker authenticated; original maze3 worker started

The original supervisor19047 reported maze2
SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED with verified_round_trip=False
and no worker failure. Worker2563586 exited; no restart was performed.
Completed worker SHA
7a836a87a8c03a668ce5b7528bda536e905c0ea67d27d2a75523afa9f36fad38;
cohort_progress_after_02.json SHA
a110ec00ee205b070cda97c415f71724c00bf5a704eb26ec5f408ade2b6933d1.
Progress contains exactly completed cases1,2 and remaining[3], original order[1,2,3].

Independent read-only16522 exited0: all18119 worker output bindings verified,
all1672 source bindings checked before/after, exact launch/collection/audit/prefix/
terminal/progress identities checked before/after, terminal outcome fields matched
the saved audit, complete collection and prefix receipts matched their files,
worker log matched its binding, and preceding case1 terminal identity matched.
Total hashed9,134,899,441 bytes. This was an independent binding/receipt check;
it did not rerun the full raw controller or original transitive runtime verifier.
Those were performed by the completed original worker. Worker wall time
8824.880337469978s and maximum RSS10,478,325,760 bytes are whole-worker measures,
not controller latency. The negative navigation and strict visibility outcomes
in the preceding section are unchanged.

Aggregate is now25 completed raw-audited native episodes and zero verified round
trips. These counts include failed scientific outcomes and reused development
layouts; they do not establish reliability or independent-layout qualification.
The original supervisor2534319 remains live and has advanced to fresh worker
2585353/start tick131683685 for maze3. Keep original handles19047,37343,14895
live and unchanged. Independent16522 and the maze2 monitors are closed.
Next authenticate maze3 when complete, then the full paired cohort and the
already queued controller experiments. Goal remains active and storage unblocked.

## Full maze2 score analysis completed alongside original maze3

Concrete progress: independently authenticated the completed maze2 population,
then reused the frozen maze1 score-analysis functions for all3014 observations
and3000 selections. Analysis56970/PID2586362 exited0 with result SHA
9e2e78192de73340e9dbed6f1b2a47d9432176b89b1d572f9056800ea04a956d,
launch SHAa459d8c001f10dccb69f7fe44759322047a6240fa7929f399a5d59dc97f9182b,
at go2_supervised_maze02_turn_scores_v1_attempt_001. All3000 complete score outputs
reconstructed exactly. All six actions were feasible throughout; no translation
phase, surface or segment veto occurred. Every selected turn had a feasible
forward candidate with greater geometric progress but lower utility after the
original800ms contact penalty. There were1500 right turns,1500 left turns and995
consecutive direction changes. This repeats the maze1 mechanism without changing
any policy or inferring an alternative physical outcome.

Independent44216 exited0:1675 sources before/after,18124 input bindings, output
identities and complete-population aggregates verified. Detailed preparation,
preflight correction before freezing, hardware/concurrency, numeric results and
verification limits are in
docs/go2_supervised_maze02_turn_scores_result_2026-09-10.md.
The already frozen contact-horizon native pilot remains unchanged and queued.
Original maze3 worker2585353 remains live and was observed at tick461. Keep19047,
37343 and14895 live;51785/2417/56970/44216 are closed. Count remains25 raw-audited
native episodes and zero verified round trips. Goal remains active.

## Supervised maze3 collection finished; original raw audit pending

The intervening turns were verified waits on original maze3 worker2585353/start
tick131683685, with increasing saved ticks and CPU time. Read-only monitor13136
observed tick3013 and the collection result, then exited0. Original supervisor
2534319, queue2551088 and contact-native waiter2571800 remain live. No experiment
was restarted or changed.

Collection result SHA
6e65e53656eba51d2d3d2c201187d603aa94adf97b69711ddc53fc72517eac7c,
at go2_supervised_rollout_mazes_v1_attempt_001/
full_supervised_rollout_novel_maze_03/result.json. Its status is
MEASURED_FLOOR_TRANSPORT_MAZE_TERMINAL_AUDIT_REQUIRED. It reports3014 decisions,
3014 primary/auxiliary frames,3013 completed commands,151400 physics samples,
10 terminal-zero ticks and MISSION_TICK_BUDGET_EXHAUSTED. Physical and acquisition
stop fields are null. Final observed mission receipt is frame3003, OUTBOUND,
no arrivals, active goal[0,-1.3], observed goal distance1.367013726468755m.
These are collection receipts, not raw-audited native navigation outcomes.

Complete saved command-tape SHA
7707a8c139457257e6526b3090561c8177da2cde8272c00ece327e8aaf5d88cd.
Read-only inspection verified original launch, collection and tape identities
before/after, exact assigned maze3 supervised model in the launch, and the
original action-menu source binding. All3013 requests are ordered, marked
completed, and have exact pre/post sample indices749+50i and799+50i. Roles/counts:
3 warmup zeros,3000 online navigation requests,10 terminal-drain zeros. Navigation
requests are1500 left turns,1499 right turns and1 zero command, with no forward
or arc translation. First navigation request at3 is[0,0,-.45]. This tape readout
does not independently reexecute the physical-command audit or determine the
score mechanism for every maze3 selection. It does not establish absence of
physical drift, native cell crossings or visibility qualification.

At inspection, maze3 raw-audit, physical-prefix comparison and worker terminal
were absent. Keep original handles19047,37343 and14895 live. The same worker
2585353 is performing post-collection work. Aggregate remains25 completed
raw-audited native episodes and zero verified round trips. Next authenticate
the completed maze3 worker, then the full paired supervised cohort, and preserve
the original downstream queue and contact-horizon native test.

## Supervised maze3 raw audit and completed worker authenticated

The original maze3 worker completed with
SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED, no worker failure and no round trip.
Worker2585353 exited normally; no restart was performed. Exact identities:

- Raw audit:7a3396c404a66c861bc301a56b3c43d7921555d3899dbed6784ac59c6c538c22.
- Physical prefix:b585690d9af0400fab25bc6c5109ab1c0f2adce4634dfa6b1208c2d6402ef0bc.
- Worker terminal:1a7133781f638930cff9a73231e72da1a749ce513665b9ee83f27fae90a43dc9.
- Progress-after03:b80ab2aa1b5b30bed9f66bfd75edb1e87a613f686da5b80ac216a84209236a2b.
- Physics trace:2f4b6e6edf700cacb464339316b9efec7a942aa62ac1a9527b54d1eaca5455e4.

Raw sensor reconstruction, complete model/command replay, command audit,
unchanged model and strict physical visibility all PASS. There are no hard
measurement failed frames, primary strict failures or auxiliary visibility
failures. Selected actions are1499 right turns,1500 left turns and1 hold. Native
evaluation reports no arrivals or cell crossings and only start cell[-1,0].
Physical stop is null. The expected four-observation/900-sample prefix and one
paired forecast bank pass; the frame3 supervised right turn was completed.

Independent52815 exited0: all18119 worker output bindings verified, all1672
source bindings checked before/after, exact launch/collection/audit/prefix/
terminal/progress identities checked before/after, complete saved receipts and
outcomes matched their original files, and both preceding worker identities
matched. Total hashed9,923,912,037 bytes. Progress contains exactly completed
cases1,2,3, remaining[], original order[1,2,3]. Worker wall time8714.0026679039s
and maximum RSS11,605,090,304 bytes are whole-worker measures, not controller
latency. This independent binding check did not rerun the raw controller or
the original transitive runtime verifier; the completed worker performed those.

The terminal_native_quiet_pass field is FALSE. Independent evaluator-only41792
exited0 after verifying all1672 sources before/after and original launch,
collection, worker, audit and bound physics-trace identities before/after.
It reexecuted the entire original native evaluator and required exact equality
with both saved audit and worker outcomes. The final501 samples are all outside
the6cm starting-position radius: distances range0.06435060529043939 to
0.07069786553408591m. Maximum speed is0.04554695285649247m/s, no sample exceeds
0.05m/s, and all last500 requests are zero. Thus starting-position distance,
not final speed or nonzero requests, fails this conjunction. Final native local
XY is[0.023810430654640356,0.06656764666950646]m. These are evaluator-only native
measurements; they grant no controller-native-state access or pose-error bound.
The raw sensor/model/command audit was not independently reexecuted by41792.

Aggregate is now26 completed raw-audited native episodes and zero verified round
trips. Reused development layouts and negative outcomes are included in that
count; it is not a reliability result. Original supervisor2534319/handle19047
is performing the complete cohort's final verification. The cohort result has
not yet been authenticated. Keep19047,37343 and14895 live with their original
definitions;52815,41792 and the maze3 worker/monitors are closed. Next authenticate
the complete paired cohort, then retain the already queued controller experiments.

## Complete supervised cohort authenticated; original queue advanced

Original supervisor2534319/handle19047 exited0 with
SUPERVISED_ROLLOUT_MAZES_V1_COMPLETE. Completed result SHA256:
16e9fccb6aead934d3f4835e7d75a0532d6b99a1c5f5cbc790f12c621d933033.
All three fixed cases ran in order[1,2,3], with three reused development-layout
executions and zero new independent-layout executions. All three supervised
workers selected only turns or holds, exhausted the shared navigation budget,
and recorded no native arrivals, cell crossings or verified round trips.

Independent16696 exited0. It checked all54371 cohort output bindings and all1672
source bindings before/after, exact cohort/launch/learned-result identities
before/after, exact complete worker-terminal equality with the three conditions,
worker-output membership, all16 original matched launch fields, and exact
recomputation of the three fixed pairs using the original paired_outcomes
function. It hashed28,809,919,420 bytes in57,722 operations. This check did not
rerun the raw controller or the transitive runtime verifier. The original
supervisor and downstream queue performed their original verification paths.

| Layout | JEPA / supervised arrivals | JEPA / supervised cell crossings | JEPA / supervised strict visibility | Verified round trips |
| --- | --- | --- | --- | --- |
| 1 | 0 / 0 | 0 / 0 | PASS / PASS | 0 / 0 |
| 2 | 0 / 0 | 1 / 0 | PASS / FAIL | 0 / 0 |
| 3 | 0 / 0 | 0 / 0 | PASS / PASS | 0 / 0 |

These fixed development pairs establish neither training-objective advantage
nor reliable navigation. Supervised maze2's three strict visibility failures
remain failures. Longer budget execution alone is not navigation improvement.
Aggregate remains26 completed raw-audited native episodes and zero verified
round trips; cohort completion adds no episode beyond its three workers.

Original queue2551088/handle37343 emitted SUPERVISED_AUTHENTICATED with the exact
result/launch identities,1672 sources,54371 outputs, all raw audits and physical
prefixes passing, and original_verifier_reexecuted=true. It then launched the
unchanged direct-flow maze3 runner as PID2609045/starttick132610582, using the
frozen learned-cohort SHA a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720.
At this observation, that original child is live in input verification and its
exclusive attempt output has not yet been created; native collection has not
been confirmed. Its creation is not another completed episode.

Queue launch admission recorded82,205,077,504 available RAM bytes and
680,921,919,488 free artifact bytes; the latest direct observation is about
634.16GiB free artifact storage. Storage is not blocking this sequence.
Retain the original queue and contact waiter2571800/handle14895. Fixed remaining
order: direct-flow maze3, residual-anchored continuation maze2, recent-qualified
reference maze1, then the separately waiting supervised contact-horizon maze1
pilot. No parallel native scene, retry, skipped test, source modification or
training-data deletion was introduced.

## Direct-flow maze3 launch and original worker authenticated

The original queued runner created its exclusive attempt launch:
48493379a21f2691a82195873b62ed101e927d3667d36493e4e6aa90d5aa3f28.
All1685 source bindings independently match current source bytes and the
original queue's frozen source union. The planned case is exactly
full_jepa_direct_flow_maze_03, layout3, full/jepa,
seed_2026091001_full_jepa, using DirectFlowFloorTransportController and one
native scene worker. The complete saved prospective prefix report exactly
matches the launch report, bound to prefix result
104edd70824e7d4e7b909be924eb68a918542efd3464ff3247f9570ddc28669e.
Its model state remains
4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.

Runner2609045/starttick132610582 spawned worker2610136/starttick132650460.
Child2610135 is the multiprocessing resource tracker, not another native scene.
The original worker is live with increasing CPU time, performing its own
precollection input verification. No collection result or completed worker
receipt exists yet. Native outcomes and actual execution of the prefix's
frame264 changed command remain unverified. This launch binding inspection
did not independently rerun the transitive verifier or controller.

The pilot's terminal failure path is failure.json; observe that exact path,
its result.json, and full_jepa_direct_flow_maze_03_worker_terminal.json.
Preserve the original runner, worker, queue37343 and contact waiter14895.
The previous goal turn made progress by independently authenticating the
complete supervised cohort and recording the original queue's advance.

The same worker subsequently entered native collection. Its completed timing
receipt stream advanced from its first row to tick24, while the original
runner/worker process identities and launch SHA remained unchanged. This is
live collection progress, not a raw-audited prefix or navigation outcome.
At supervisor elapsed384.349s, available RAM was79,528,460,288 bytes, artifact
free space680,870,514,688 bytes, and worker RSS2,424,684,544 bytes. No collection
result, worker terminal, pilot result or failure.json existed at that check.
The frame264 intervention is still ahead. Bounded collection monitors25432,
11280 and58624 exited0; original queue37343 and contact waiter14895 remain live.

## Direct-flow maze3 streamed intervention observed

The original native worker continued beyond the original frame264 tracking
failure. At live timing tick309, independent read-only inspection67350 exited0:
all first265 complete streamed decisions exactly matched the frozen prospective
prefix decisions. The inspection authenticated the original launch, prefix
result104edd70824e7d4e7b909be924eb68a918542efd3464ff3247f9570ddc28669e,
and that result's compressed decision-stream binding before comparing bounded,
consecutive rows. At264, the new controller requested[0,0,.45] with terminal
and failure both null. The actual following observation265 also requested
[0,0,.45] with terminal and failure null; that following observation was not
part of the frozen prospective replay or its exact-decision claim.

This is a live receipt inspection only. The growing current stream has no final
output binding yet. Complete physics/public-packet prefix identity, actual
command-tape completion, raw model/sensor replay, visibility and navigation
outcomes remain pending the original worker's final audits. Continuing beyond
the former failure is not an arrival, round trip or reliability result.
Bounded monitor51073 exited0; no failure, completed collection or worker terminal
existed at its final tick298 observation. Preserve worker2610136 and its original
runner2609045, queue37343 and contact waiter14895. Count remains26 completed
raw-audited native episodes with zero verified round trips.

## Direct-flow maze3 provisional mission snapshot through766

Read-only stream inspection68109 exited0 after checking the original launch
SHA before/after and reading exactly consecutive bounded rows0..766 from the
growing decision stream. Requested commands:14 zeros,131 left arcs,1 right arc,
547 left turns and74 right turns. No terminal receipt appeared in that window.
At766 the visual status is CURRENT_VISUAL_POSE, the mission is OUTBOUND with
arrivals[], and observed goal distance is1.303241672842386m to the declared
initial-body goal[0,-1.3]. The receipt explicitly has native_state_used=false
and uncalibrated pose uncertainty. These are observation-based live receipts,
not native goal-distance measurements or a physical completion audit.

The same original worker was confirmed live with increasing CPU time and a
timing stream reaching816 at the following monitor check. No collection result,
raw audit, prefix comparison, worker terminal, pilot result or failure.json
existed then. No completed-episode count or scientific claim changes from this
provisional snapshot; authenticate the original final outputs when available.

## Direct-flow maze3 collection ended; observation1206 tracking failure

Original collection completed with DIRECT_FLOW_MAZE03_TERMINAL_AUDIT_REQUIRED:
1217 primary/auxiliary observations and decisions,1216 completed command ticks,
61550 physics samples and10 terminal zero-drain commands. Physical and
acquisition stops are null; schedule terminal is SENSOR_OR_MODEL_FAILURE.
The last mission receipt is frame1205, OUTBOUND, arrivals[], observed goal
distance1.3238731265148527m. This is not an audited native goal-distance result.

Fixed saved identities:

- Collection result:659f3aa6a75cb29ce72847308e89abf691e703e81739ce07fd6448d9b2e4cd23.
- Closed decision stream:fd55017ee0e2463aa4d443508047c586e27d253c9561a8d5811b9bab1042ea36.
- Command tape:23f07bd793d166e1ff8e6ad1863b3c7cfe9939196865b1466d9aa364ce2f4afd.

Read-only51059 exited0: all1685 launch sources checked before/after; exact
launch, collection, closed stream and tape identities checked before/after;
all1216 tape entries ordered, marked completed, and matched to their decision
requests and exact sample indices749+50i to799+50i. The frame264 changed left
turn is marked completed. This is receipt verification, not independent physics,
raw sensor/model, visibility or transitive-verifier reexecution. Across all1217
decision rows, requested commands comprise192 left arcs,1 right arc,76 right
turns,921 left turns and27 zeros, including the final observation without an
additional command. The raw command tape contains26 zeros and1216 total requests.

Focused60175 exited0 with the closed stream's identity verified before/after.
The first terminal row is OBSERVATION1206. Its controller tick and retained
mission frame are1205 because the failed visual observation never advanced
the controller. Do not misreport1205 as the failed observation. The failure is
same-episode current visual evidence required; the visual cause is bounded
measured bridge exhausted without anchor observation, on both cameras.

Frame1190 was anchor-qualified and retained:14/48 overlap, retain=true.
Frames1191 and1192 were anchor-qualified against1190 but not retained.
Frames1193..1194 used measured increment bridges. Frame1195 again qualified
against1190, resetting the bridge sequence, but was not retained:12/17 overlap
exceeds the unchanged0.5 retention threshold. Frames1196..1205 then used exactly
ten measured increment bridges. At1206, retained references1190..1183 all failed;
both cameras report MEASURED_BRIDGE_BUDGET_EXHAUSTED. The direct-flow fallback
recorded pair_attempts[] and accepted=false. A qualifying adjacent measurement
alone does not override the existing bridge budget or provide a pose-error bound.

This supplies a specific reference-retention diagnosis to assess alongside the
already queued recent-qualified-reference pilot. It does not prove that an
additional retained reference would qualify or improve native navigation in
maze3. No new controller variant, relaxed gate, restart or extra native scene
was introduced. Worker2610136 remains live for the original raw audit; its final
prefix, worker terminal and complete pilot result are pending. Keep queue37343
and contact waiter14895. Aggregate remains26 completed raw-audited episodes and
zero verified round trips until the new worker is completely authenticated.

Source review of the existing recent-qualified-reference implementation narrows
that follow-up. It retains only an immediately preceding anchor-qualified view,
requires object identity with the previous observation, and clears that extra
reference after a bridge. It cannot directly reuse1195 at failed observation1206.
Within the inspected late window,1193 and1196 are the missing-anchor observations
immediately following qualified-but-unretained1192 and1195. These are local
candidate opportunities, not established first interventions or qualified fits.
A new maze3 application must replay causally from frame0 and stop at its first
changed command; it cannot consume the old later trajectory after that boundary.

The existing RecentQualifiedAnchorController also includes partial-floor-height
registration, whereas this native input uses DirectFlowFloorTransportController.
Using that controller unchanged would combine two interventions. Any isolated
maze3 reference-retention replay should instead retain the direct-flow native
base and reuse only RecentQualifiedAnchorVisualMotion/RecentQualifiedAnchorPose,
with complete original/new decision comparisons and explicit field accounting.
No such variant has been implemented or run here. The original worker, audit,
queue order and already frozen recent-reference maze1 pilot remain unchanged.

## Isolated reference-retention replay prepared while original audit runs

Implemented RecentQualifiedDirectFlowController directly over the unchanged
direct-flow controller, with the existing recent-qualified visual-motion owner
only. Prepared a bounded causal comparator and maze3 replay; runtime admission
requires the completed original native result, raw audits, exact closed
collection bindings and completed frame264 physical-prefix command. It cannot
run on the provisional collection result. Its candidate input observations are
0..1206, stopping earlier at the first changed command or either terminal.

Focused56253 exited0:63 tests passed4.58s. Source-only95625 exited0:1719 source
bindings verified, native result and new replay output absent, no actual model
or native execution. Seven prepared source hashes, resource evidence and the
remaining runtime preflight/execution steps are recorded in
docs/go2_recent_qualified_direct_flow_maze03_preparation_2026-09-10.md.
Hardware71618 found76,105,031,680 available RAM bytes and676,326,555,648 free
artifact bytes, sufficient for the declared8GiB replay plus32GiB native
allowances at that observation. Actual replay admission must recheck resources.

Original direct-flow worker2610136 remains live in its raw audit; queue37343
and contact waiter14895 retain their original definitions. The source preparation
does not add a native episode or establish a navigation outcome. After the
complete native input is authenticated, preflight and execute this CPU replay
without changing the original native queue.

## Direct-flow maze3 raw audit and physical prefix saved

The original worker saved raw audit
2b8e6c1c42f3b9473a77b35aa6cef259fc10c105199ab094398312b65ae2e5b6
and physical-prefix comparison
ec02a7c66647e8e7c9011df59c41a25f4c8482922c36a102bf40190494c5f121.
Raw sensor reconstruction, raw model/command replay, command audit, unchanged
model and strict physical visibility all PASS. Hard measurement failures are[].
The1217-frame renderer audit has2434 capture endpoints, paired readbacks equal
and all witnesses matching their raw acquisitions.

Native evaluation has no arrival windows, no cell crossings, only start
cell[-1,0], no return traversal and verified_round_trip=false. Terminal native
quiet is false. Selected actions are192 left arcs,1 right arc,76 right turns,
921 left turns and13 holds. Passing raw/visibility audits and extending the
tracking lifetime did not produce navigation success on this development layout.

The complete265-observation physical/public prefix passes with13950 physics
samples,261 matching raw forecast banks and all preceding observed decisions
and commands exact. The changed frame264 command[0,0,.45] is completed; the
original command was[0,0,0]. Raw prefix fingerprint:
27ff764ed8fce7d779b5930b8bbcbff2a90899f8621dbff8f58f4a39f87d90c6.

Independent read-only receipt check exited0 after all1685 source bindings and
the launch, collection, closed stream, command tape, audit and physical-prefix
identities were checked before/after. It required the saved raw flags and every
declared prefix assertion exactly. It did not independently reexecute the raw
controller, physical-prefix function or transitive verifier, and did not yet
check the full worker output binding map, which is pending its terminal receipt.

Original worker2610136 remains live in final input verification; its terminal
and the complete pilot result are not saved yet. Preserve runner2609045,
queue37343 and contact waiter14895. Authenticate the complete worker/pilot when
available, then perform the prepared isolated-reference replay's runtime
preflight and execution. No new completed-worker count is claimed at this stage.

## Direct-flow maze3 completed worker independently authenticated

Worker2610136 exited normally with
DIRECT_FLOW_MAZE03_COLLECTED_AND_RAW_AUDITED and no worker failure. Terminal SHA:
aadd9a9b5f71d0c2074e2b235e2feac265281050c8a74f775e3392328699a0dd.
Independent61678 exited0: all7337 declared worker output bindings checked;
all1685 launch sources and the exact launch/collection/closed-stream/tape/audit/
prefix/terminal identities checked before/after; complete collection and prefix
receipts matched the terminal; all recorded outcome fields matched the raw
audit; the closed worker log matched its declared hash. The check hashed
4,880,303,359 bytes in10,723 operations. It did not independently reexecute
the raw controller or original transitive verifier; the original worker did.

Worker wall time3396.1033406509086s and maximum RSS6,057,406,464 bytes are whole
worker measures, not controller or full-loop latency. Worker-log SHA:
d2a06be8dfcfae3b521abbe8723758b6d1f9c4187825a8973fd5651e6e7ac305.
The authenticated outcome remains zero native arrivals, cell crossings and
verified round trips, despite a correct executed tracking intervention and
passing raw/strict-visibility audits. This is one reused development layout.

Aggregate is now27 completed raw-audited native episodes and zero verified
round trips. Original pilot supervisor2609045 remains live in final verification;
complete pilot result.json is still pending. Worker and temporary monitors
89777,44582,79461,58984,42242,61678 are closed. Keep original queue37343 and
contact waiter14895; do not restart the finished worker. Next authenticate the
complete pilot result, let the original queue advance, then preflight/execute
the already prepared isolated-reference CPU replay using that exact result SHA.

## Complete direct-flow pilot verified; isolated replay preflight running

Original pilot supervisor2609045 exited0 and saved
DIRECT_FLOW_MAZE03_PILOT_V1_COMPLETE, SHA256:
6be6aa6e60be4b1a9e3d9b79aa3b55e4becc3265fe208ce900c8e1962c3db3ea.
The result contains the exact independently verified worker,1685 sources and
7341 output bindings. Final result/launch/worker identities and all source
bindings were independently checked before/after. The complete output map is
exactly the already verified7337 worker bindings plus launch, closed resource
monitor, closed worker log and worker terminal; all four additional bindings
were verified. Previously checked image hashes were not redundantly repeated
in this final-map check. The original supervisor performed its full verifier.
Pilot wall time3776.6211977240164s is not controller or full-loop latency.

Queue37343 recorded NATIVE_CHILD_EXITED returncode0 and is performing its
original completed-pilot authentication. Native stdout SHA:
fefe1405cb11358648c35e748b1ce064d3b49c7b675f3bb4be12c32a9a15bea7.
Original queue2551088 and contact waiter2571800 remain live with their original
start identities. Finished worker2610136 and pilot2609045 must not be restarted.
Bounded completion monitors29859,97781 and7217 are closed.

Started isolated-reference runtime preflight1143 with the exact complete pilot
SHA above and --preflight-only. All seven prepared source hashes were rechecked
unchanged. This is the original prepared runner, not a replacement attempt.
Preflight is still running, with no replay output or actual model replay claimed.
Wait for1143; if it passes, execute that prepared CPU replay once using the same
complete native SHA. Preserve the native queue order and contact waiter.

Isolated-reference preflight1143 then exited0:1719 sources,127083 unique files,
1,799,188 digest requests,59,020,986,394 bytes freshly hashed initially and finally,
all original verification conditions executed across23 isolated functions,
no retained cache or imported-global mutation. Its fresh hardware admission
passed with81,299,656,704 available RAM bytes and675,447,046,144 free artifact
bytes. Submitted the actual prepared replay as13420/PID2620352/starttick133059248,
using the same completed pilot SHA. It remains in initial input verification;
no replay result or intervention outcome is claimed. The original queue is
authenticating its completed direct-flow child before the next native job.
Keep13420,37343 and14895;1143 and the completed native pilot/worker are closed.

The original queue then authenticated the complete direct-flow pilot with
all7341 outputs,1685 sources and its original verifier reexecuted. It launched
the fixed residual-anchored-continuation runner as PID2620473/starttick133066785,
using prefix SHA3cbd24abad8a6c70565648977ce8482df37b4c90b8a2e28c728799910cf402b5.
That original child is live in input verification; its exclusive output root
was absent at the latest check. Queue launch admission recorded81,056,141,312
available RAM bytes and675,446,956,032 free artifact bytes alongside the CPU
replay. No duplicate native scene or queue change was introduced.

Actual isolated-reference replay13420 admitted launch
f9c6c834f1032f0bf6c8600205ab65758781451305987263fb444b1f923f9a34
and reported frame0. Independent launch verification checked all1719 source
bindings, the exact completed native input/model identity, one CPU replay,
zero native scene workers,1207-frame bound and absence of a floor-height
intervention. Its initial original-context verifier again passed127083 unique
files and fresh initial/final hashing. The model replay is now running, with
no completed prefix result claimed. Preserve13420/PID2620352 and the original
queue37343/contact waiter14895; the next native job remains owned by that queue.

## Residual-anchored-continuation native launch verified

The original queued runner2620473 admitted launch
c4681e31baaf5dc1c8fa368854ddbcf19866090787741b946c4d37b9a3a477b3
for full_jepa_residual_anchored_continuation_maze_02, layout2, full/jepa,
seed_2026091001_full_jepa, ResidualAnchoredContinuationController. All1746 source
bindings independently match current files and the original queue source union.
The complete prospective prefix report matches the fixed result
3cbd24abad8a6c70565648977ce8482df37b4c90b8a2e28c728799910cf402b5,
with181 observations, first command difference180 and178 matching raw banks.
Launch and prefix identities were checked before/after that comparison.

Original native worker2620852/starttick133090201 is live in its input
verification. Child2620851/starttick133090200 is the resource tracker, not an
additional scene. Runner2620473 retains starttick133066785. Launch admission
recorded80,554,541,056 available RAM bytes and675,428,139,008 free artifact bytes
alongside the CPU reference replay. No decision timing stream, collection result
or terminal receipt existed at the latest worker check; native collection is
not yet confirmed. One native scene remains the configured limit.

The independent reference replay13420/PID2620352 reported progress through352
without a failure or completed result. This is replay progress, not a native
outcome. Preserve both original tasks, queue37343 and contact waiter14895.

The same residual-continuation worker subsequently began native collection and
reached timing tick88. Its worker identity remains2620852/starttick133090201;
no collection result or terminal receipt existed at that check. The declared
first changed-command boundary180 is still ahead. Reference replay13420 reported
progress through512 without a failure or completed result. These are live
execution/replay progress observations, not additional completed episodes.

## Residual-continuation streamed intervention matches the frozen prefix

Native collection passed frame180. Independent bounded stream comparison39678
exited0: all181 complete decisions0..180 exactly match the fixed prospective
prefix. The original launch and frozen prefix-result/decision-stream bindings
were authenticated. At180 the native controller requested[.16,0,.45], terminal
and failure null. The following native observation181 requested[0,0,0], also
with terminal and failure null; that following observation is outside the
prospective prefix's exact-decision claim. Collection monitor18391 exited0.

This is live decision-receipt evidence only. The complete physical/public
prefix, completed command tape, raw model/sensor replay, visibility and native
navigation outcome await the original worker's final audits. The reference
replay13420 has separately reported progress through704 without a completed
result or failure. Preserve both original running tasks and queue order.

## Residual-continuation complete; 28 audited native episodes, zero round trips

The original residual-continuation pilot completed with result SHA
818a598ca6336866cf5f4768c11edaf67c8c1ca60896c305f93f69fd0ed5230c.
Collection contains266 observations,265 completed commands and14000 physics
samples, including10 terminal drain commands. Terminal observation255 is
NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS. Physical and
acquisition stops are null. There are no observed or native arrivals and no
verified round trip. The native evaluator records three crossings across the
same first-cell edge, with loop-erased route[-1,0]→[0,0], and terminal quiet
fails. These are not three newly explored cells.

Raw sensor reconstruction, model replay, command audit, unchanged model and
strict physical visibility all pass; hard measurement failures are empty.
All266 renderer frames and532 capture endpoints pass their recorded witness
checks. The physical/public prefix matches exactly through181 observations
and9750 physics samples, including178 exact raw forecast banks. Changed command
180 is[.16,0,.45] and actually completed. Prefix physics fingerprint:
fa9b9ab27d58a421e86486757dba98be8ca48cb56dfe320c09e83bc93229df0a.

Exact completed receipts:
- collection:4bfe9ca2ab590f89029cb5c9f09adb46747d105e3fde2c95e97f70c398e7d767
- decisions:0e81124252ccdda87c252a8b2ac55613c6bf7ab505110af12fa1e30903add118
- tape:aea133d89ad7a18e930245871e326c2f274883e4c1a2a14583aa4b7d9c96a4c7
- audit:0d1707cbce6f6b145eab4fa1479787918bbbe5960f25d569100b2daaaa7af6d9
- prefix:5721b4c4583445ee985dbf7cb68380a0597484362a298aa92600494090e7eefc
- terminal:1903f04562b21398154b6a40d1301009b46cce572655f299ca7132f28486a2ac
- closed worker log:4a9a12f5eed4125f3ee9bf187dcb052cb0d84336890a28c63b268828d283383b
- closed native stdout:22918be2d7c613e1f526bd373ff72a816465f8f9e436e783e646d31eba552e01

Independent check49414 exited0: all1746 pilot sources before/after, all1635
pilot output bindings, launch/source agreement, exact collection/audit/prefix
agreement with terminal and worker-log identity. The same check authenticated
the completed1719-source reference replay below. Combined8574 hash operations,
1,017,397,569 bytes. This check verifies source/output/receipt bindings; it does
not claim a second raw-controller or transitive input-verifier execution.
The original worker and supervisor executed their complete verifiers.
Worker wall1016.1367116731126s and maximum RSS2,993,999,872 bytes are whole-worker
measurements, not control-loop timing. Runner2620473 and worker2620852 are closed.
Queue37343 recorded the child exit0 and is authenticating it before its final
fixed recent-reference maze1 job. Contact waiter14895 remains behind that queue.

The terminal decision diagnosis was recovered from the closed, SHA-bound
stream, checked before/after. Frame244 selected hold; frame245 first became
infeasible in WAYPOINT; frames246..255 remained infeasible in VIEW_ACQUISITION.
At255 the scan phase permits only hold/left_turn/right_turn. All six candidates
pass the surface-conflict filter, three survive the phase filter before nominal
constraints, and zero survive nominal constraints. Current mapped nominal
clearance is0.44410933877138264m against radius0.45m, nearest observed cell[13,11].
Every candidate already fails its first100ms nominal segment: clearance ranges
0.4382756206176447..0.4420986018974154m. Thus this terminal rejection is not solely
a later-horizon forecast veto. Current visual pose remains valid. The anchored
hold continuation is restricted to feasible WAYPOINT holds and preserves the
current-clearance gate; it cannot resolve this scan-phase nominal overlap.
These are observed-map nominal constraints, not proof of physical collision or
an authorization to relax clearance.

## Isolated recent-reference maze3 replay completed positively

Replay13420 exited0 with result
16c6917d2e4c2b728bd08a330290e141a93a500f9f28b7a3abd27e9d4f51926a
and decision-stream SHA
f3312bc922c01dcde8f030b2ca8549d28611de25b6ceb9626065f4ce5ab6785f.
It consumed1207 observations0..1206 and compared1203 exact raw forecast banks.
Complete original decisions match for1193 observations before the first extra
reference attempt at1193. That single attempt qualified. The candidate then
retains its actual changed visual and downstream state, rather than resetting
to the original. All executed requests through1205 still match the original.
At1206 the original reports SENSOR_OR_MODEL_FAILURE with zero request, while
the candidate is nonterminal and requests[0,0,.45]. Replay stopped there without
consuming any following recorded observation or inferring an unexecuted outcome.
The JEPA model remains4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.

The final original-context verifier again freshly hashed127083 unique files
initially and finally,59,020,986,394 bytes per pass,1,799,188 requests and23
isolated functions; no retained cache or imported-global mutation. Check49414
also independently verified all1719 current sources before/after, both output
bindings and the exact result identity. This is causal CPU replay evidence,
not a29th native episode or navigation success.

Next preparation: a fresh maze3 pilot using RecentQualifiedDirectFlowController
and the complete frozen replay above, after the existing recent-reference maze1
and contact-horizon maze1 executions. Preserve those original processes/order.
Require1207 exact prospective native decisions,61050 shared physics samples,
1193 normalized original decisions,1203 exact raw banks, actual changed-command
completion at1206 and the unchanged complete raw/visibility/native outcome audit.
The new pilot must not introduce the partial-floor-height intervention.
No new maze3 native runner, native launch or native outcome is claimed yet.

Storage rechecked:629GiB available on the artifact volume,20GiB on the workspace
volume. Storage is not blocking this queue; no additional deletion was performed.

The isolated-reference maze3 native preparation is now implemented in separate
files and documented in
docs/go2_recent_qualified_direct_flow_maze03_native_preparation_2026-09-10.md.
Its96 focused tests passed. Runtime preflight54205 is running; no new maze3
native scene has launched. Actual execution requires the complete original
contact-waiter result SHA, its original native verification and queue receipts,
fresh resource admission and no competing native worker. Preserve54205 until
it exits, then record its actual result before scheduling later work.

The original queue has launched its final recent-reference maze1 child2623245
(process create_time1789011306.56), worker2624197(create_time1789011673.38).
Launch4634b6766427fbad908e460513fa01d9f7035f6e541fdacaf4cb6c5193c4f579
and all1726 sources independently verify against the unchanged1825-source live
queue/contact union. Casefull_jepa_recent_qualified_anchor_maze_01 remains the
original fixed layout1/full/jepa intervention. Its outcome is pending; the
completed audited episode count remains28 with zero verified round trips.

Preflight54205 then exited0:1843 sources,127210 unique input files,1,867,547
digest requests and27 isolated verifier functions. Initial and final passes
each freshly hashed59,145,158,231 bytes; no retained cache or imported-global
mutation. The complete saved prospective comparison stream was admitted.
Both memory/storage capacity checks passed; no native output was created.
The prepared isolated maze3 native runner is ready for later execution after
the original contact-waiter completion gate. Its exact command and source
identities are in the native-preparation document above. No separate new
waiter or native process was started for this prepared pilot.

Original queue37343/PID2551088 and contact waiter14895/PID2571800 remain the live
scheduling owners. Current original maze1 worker2624197/starttick133270638 was
still actively verifying inputs at the latest check; native collection had
not yet produced a timing stream. Keep these original processes. Closed
reference replay13420, preflight54205 and observer41486 must not be restarted.

## Actual zero-command drift and expanded training targets

The next goal turn made progress, not merely a wait. Current maze1 worker2624197
was revalidated live and began native collection. At timing tick770 no collection
or terminal existed. Bounded comparison93670 exited0: all646 complete native
decisions0..645 equal the frozen recent-reference prospective replay exactly.
The launch,1726 sources and frozen replay bindings were checked before/after.
At645 the native request is[0,0,.45], nonterminal; following646 requests
[.16,0,.45], also nonterminal. Physical/public prefix, command completion and
raw audit remain pending. This does not add a completed episode to the28 count.

Posthoc residual-continuation diagnostic93998 confirmed about11.5mm actual
drift during the11 completed zero requests244..254. The observed/native
displacement disagreement is about0.09mm over this interval, with absolute XY
pose error about3.2mm. The existing translating view-reentry policy cannot
admit any raw candidate because every predicted path reduces current clearance.
See docs/go2_residual_continuation_zero_command_drift_2026-09-10.md for exact
bindings and limits. This motivates stopping/transient prediction work; it
does not authorize fitting navigation outcomes or relaxing clearance.

A training-only census of existing recordings then identified4010 available
all-phase contexts, against408 in the current fits. Completed target derivation
result4d300f77849d174cc9d7bd2a35d276e0996d1b5ac8795bee4f419291ada6b328
is in go2_all_phase_training_targets_v1_attempt_001. It retains4800 planned
slots, all456 old training slots exactly,3974 valid first-step motion targets
and3140 complete eight-step motion sequences. All recorded command switches
still occur in phase3; context expansion does not repair that collection gap.
Fourteen tests and an independent source/output/consumed-leaf check passed.
The preparatory metadata failure48685 created no output; the corrected exclusive
derivation33328 exited0. Both are closed. Full accounting and the next input
materialization task are in docs/go2_all_phase_training_targets_result_2026-09-10.md.

No model training, future RGB materialization or new native scene occurred in
this target derivation. Continue the original queue37343/contact waiter14895.
The isolated direct-flow maze3 pilot remains prepared and preflight-passed,
awaiting its authenticated contact-waiter completion prerequisite.

Final live check for this turn: original maze1 worker2624197 remains running
with create_time1789011673.38 and has reached timing tick1005. Its collection,
worker terminal, pilot result and failure files are absent. Queue2551088,
contact waiter2571800 and native parent2623245 retain their original process
identities. Preserve the original live work; do not interpret this observation
point as a terminal result. The completed audited count remains28, zero round trips.

## Expanded causal input validation running; maze1 collection ended

The next goal turn implemented and tested a separate all-phase training-only
input stream. Forty-five tests passed, plus a corrected six-sample real-data
check of all original untrained model conditions without parameter updates.
The full4800-slot/4010-available input validation is live as80757/PID2628968,
starttick133510189, launch
b8eb7241b54842dd10d9a55d97752a6ad8f4e4514e8bcf5e32b4551e3da1e090,
1123 sources. It reported2688 materialized contexts. Preserve this original
process and source closure. Preparation, the harmless preparatory condition-name
error, and an inherited metadata discrepancy requiring a bound scope correction
before fitting are documented in
docs/go2_all_phase_training_inputs_preparation_2026-09-10.md.

Original recent-reference maze1 collection ended with1556 observations,
1555 completed commands,78500 physics samples and10 zero-drain commands.
Physical and acquisition stops are null. First terminal observation1545 has
decision/mission frame1544 and SENSOR_OR_MODEL_FAILURE:
floor registration exceeds fixed development correction gates. Original visual
evidence remains CURRENT_VISUAL_POSE at that failure. No observed arrival;
last observed goal distance1.2906426854727853m, OUTBOUND. Do not confuse this
failure with exhaustion of the visual bridge budget.

Closed collection bindings:
- result.json:5ccc20a4d1056c29af8f3efb066e4523954b8351b17c971a548c81712f2b528f
- context_decisions.jsonl.gz:707aa3818e9ab38d0127ffef5acd3d3796af1d64287aa544294d252db9be99da
- command_tape.json:3c089ffff7690f6da9cde49288a5c30aca3a37e57b4beb6a560815af28556e5e

Receipt check96068 closed0:1726 sources and launch/collection/stream/tape
bindings before/after,1555 ordered complete tape entries at exact sample
endpoints, all corresponding decisions equal the tape,1556 stream rows.
Only frame635 has extra reference attempts:two attempts,one qualified,using634.
Across the1556 observations there are27 zero requests,221 left arcs,64 right
turns and1244 left turns (the final observation has no following command).
This is receipt verification, not a second raw controller/native evaluation.
Worker2624197/starttick133270638 remains live in raw audit with its original
parent2623245, queue37343/PID2551088 and contact waiter14895/PID2571800.
The audited episode count remains28 until the original worker/pilot completes
and its full audits and prefix are authenticated.

The all-phase input check80757 then completed successfully with result
ef48950b7987eaf9310ba8124a00c2e6e13c9b84b7cda6a362ac7ddb6ecd63fb.
All4010 causal/training inputs match, all408 old input witnesses are exact,
and every4800 planned slot is retained. The independently verified source/
artifact/scope readout confirms32080 past-packet reads,28444 training-only
future reads, zero inference future reads and4694 consumed policy leaves.
The inherited false future-image flags remain preserved and are corrected by
the bound scope record8fcfdc678b721a84ed53c2545ec55284e2185d7b1e6bc5ca0dc26da4832e2dca
at docs/go2_all_phase_training_inputs_scope_correction_2026-09-10.json.
Require that record alongside the result before fitting.

Input-check wall314.06464697094634s,1123 sources; closed80757/PID2628968.
All4010 cached samples would contain7,359,601,120 tensor bytes; actual checker
cache usage was zero. No optimizer, parameter update or trained model resulted.
The next combined-view, transfer isolation, schedule and fitting tasks are
specified in docs/go2_all_phase_training_inputs_result_2026-09-10.md.
Original queue37343/contact waiter14895 and native worker2624197 remain owned
by their original processes. Preserve their execution and all current negatives.

## Combined expanded study validated; fitting helpers prepared

The combined5256-slot study now preserves4800 training and456 original transfer
slots with exact index translation and reader isolation. Full original evidence
authentication before and after the actual check passed. All420 available
transfer inputs and14 representative expanded training contexts reproduce their
bound input witnesses; synthetic transfer score accounting is unchanged.
Result docs/go2_all_phase_study_stream_preparation_result_2026-09-10.json,
SHA c2d632f79588e057d7dbf0777ed26fe4c17f89f13f3a9a9efd8585d12f01a9dd,
binds1130 sources. Original live queue/contact union1825 sources remains exact.
Checker15692/PID2631005 exited0; materialization9.902825506869704s excluding
before/after authentication. Eighty-five focused tests passed across study,
original input/fit, new schedules and new fit guards. No optimizer update or
fitted checkpoint resulted.

Separate expanded schedule/fit helpers remove the old fixed408-context and
five-tick-offset assumptions while retaining the original learner and1200x6
budget. All three original seed schedules cover every4010 available context
and preserve75 draws per family trial/50 per switch trial,600 batches per source,
with no transfer draw. Exact source/schedule bindings and next benchmark/launcher
work are in docs/go2_all_phase_study_and_fit_preparation_2026-09-10.md and
docs/go2_all_phase_training_schedules_preparation_2026-09-10.json.
The full prospective matched fit/native roster and resource benchmark still
need completion before fitting. Do not infer trained improvements from this
preparation or treat reindexed contexts as independent recordings.

Last live check retains original owners2551088/2571800/2623245/2624197. Recent
maze1 remains in raw audit with no final pilot result; audited count28, zero
verified round trips. Artifact free space approximately623GiB, workspace20GiB.

Immediately after that check, recent-reference maze1 wrote raw audit
5cb5c958dfb9ed6c181a157ab2282d9d44173fd5285f9c72a7816c7d66daa47e and prefix
6f9bd7b85f0198c56ddf0ea1dc71b1a0b3d65c14559be08eacb982d7d550f1e8.
Raw sensor/model/command/model-unchanged and strict visibility all pass;
hard failed frames[],1556 frames3112 captures witnessed; no arrivals, crossings
or round trip. Physical/public prefix33000 samples646 frames642 banks passes;
first reference635, original decisions exact through634, first changed command
645 left turn completed and candidate decisions match prospective replay.
Worker-terminal and final pilot result remain absent. Preserve original owners
and await full pilot/queue authentication before counting this as episode29.

## Episode29 and original prepared queue complete; fitting benchmark launched

Recent-reference maze1 final pilot4e1768c2da812f52c41a446efbe8bca9568292bc59bffb85d7e5ef4f41300e5d
is complete and authenticated by the original queue. Queue37343/PID2551088
closed0 with result1473b2f801991698b41ddd8d97af627e2b1d34621e1cdc6c48886afb03141111.
All original four job verifiers reexecuted and all source/output identities
passed. The recent-reference child has1726 sources9375 outputs; independent
before/after binding verification84998 closed0 and queue-output verification
63175 closed0. Episode29 has no arrival/crossing/round trip despite valid raw
sensor/model/command and strict-visibility evidence. Worker wall4278.480098410044s,
peak RSS6,467,809,280 bytes. See
docs/go2_recent_qualified_anchor_maze01_result_2026-09-10.md.

Original contact waiter14895/PID2571800 launched native parent2633510 using the
completed queue identity. Contact launchf61bbd19f01bde603945de9e2feff6c787bd2c62d87942a7a41967e498052096.
Preserve it; the isolated recent-reference direct-flow maze3 successor stays
behind contact-waiter completion. Completed raw-audited count29, zero round trips.

The expanded-data full-cache fitting benchmark is actually live15290/PID2633175,
launch961f7fadf5f955f3fecc26c1a9494e71a31abe81575a1c330ebaf4762a53be8e,
1144 sources. Full-cache serial_0 completed4010 contexts and two fresh20-update
benchmark fits; warm212.4372640750371s, peak RSS9,176,956,928 bytes, exact expected
cache7,359,601,120 bytes. Parent advanced to serial_1/PID2634114. No full
scientific fit has completed and no parallel decision is yet available.
Protocol, full18-fit roster, runtime ownership and ordered next actions are in
docs/go2_all_phase_matched_fits_progress_2026-09-10.md. Preserve all live benchmark
sources and output ownership; never replace it merely because observation waits.

## Contact episode30 complete; expanded full fits and isolated maze3 are live

The contact-native result90116b248154b9d4731501d5c6cc6a317345101b9f39c6ed5ce712c9baa32986
completed with1822 sources945 outputs. Original contact waiter14895 closed0,
resultdc57fe4cd5fb8083ec6ac43ae11d85e6d3e048db62d9b03f06361eb621378536.
Collection151 observations150 completed commands8250 physics samples; terminal
140 retains decision139 and reports same-episode current visual evidence required.
Raw sensor/model/command/model-unchanged/strict visibility pass; no arrival,
crossing or round trip. Prefix4 observations900 samplesone bank exact; completed
right-arc intervention at3. Completed audited count30,zero verified round trips.

The already prepared isolated recent-reference direct-flow maze3 pilot is live
25801/PID2636286,worker2637549,launch25c34387e44b035f05a4f372ecf70cd337e875de508ba3a9b037ccf741da0eb7,
1843 sources. Exact original queue/contact prerequisite checks completed before
launch. Preserve this single native owner. Full contact evidence is in
docs/go2_supervised_commitment_contact_maze01_result_2026-09-10.md.

Expanded fitting benchmark15290 completed0:2e74b02b76038f92a8b74256c083cf8ecfbe28d5a5c1bc07c5cbafda214b054f.
All six serial/parallel model/ledger pairs match exactly;measured full-phase
speedup2.6259688480137835,selected workers3. Original one-owner waiter6946/PID2635725
authenticated it and launched full-fit parent2636352. Full-fit launch
de9fb37f6cd51c609c21fb6ba03b2470167dedad2ecb9166907874bd351d1542,1144 sources,
workers2636575/2636576/2636577. Each has warmed all4010 training contexts and
started its first full1200-update fit. Do not start another fit run.
Benchmark result and future admission actions are documented in
docs/go2_all_phase_fit_benchmark_result_2026-09-10.md and
docs/go2_all_phase_matched_fits_progress_2026-09-10.md.

Prepared new translation correction keeps all36 available motionless contexts
in exact draw accounting,with zero XY contribution. Seven tests and actual
three-schedule/4010-context synthetic-prediction accounting passed. Complete
fitted-model admission and actual correction fitting still remain. See
docs/go2_all_phase_training_translation_bias_preparation_2026-09-10.md.
