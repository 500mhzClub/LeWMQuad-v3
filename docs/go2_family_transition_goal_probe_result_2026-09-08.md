# Family-transition native goal probe result — 2026-09-08

The preregistered full-JEPA checkpoint with the corner-support observer reached
zero of two goals. Both native cases completed collection and exact raw replay
auditing. One stopped on visual-pose loss; the other contacted the front panel.
The first case also contains a hard depth-measurement failure during its terminal
zero-command drain. No navigation, depth or real-time qualification is granted.

This is a known-family integration probe, not independent-maze evaluation. The
two fixed mirrored cluster-00 cases, 1.2 m goal, six action plans, five-tick
commit, 1.2 m contact penalty, 240-tick navigation budget and arrival/quiet checks
were inherited unchanged. Only the newly admitted final model and separately
tested corner observer replaced their historical counterparts. The model was
designated before [fit results](go2_family_transition_fits_result_2026-09-08.md),
and was not selected by downstream performance. Physics remained paused during
sensor acquisition and computation.

| Case | Terminal outcome | Native terminal goal distance | Closest goal distance | Replans |
|---|---|---:|---:|---:|
| `family_episode_052` | Visual-pose loss, then ten zero intervals | 0.975310 m | 0.972100 m | 6 |
| `family_episode_039` | Disallowed panel contact | 0.892337 m | 0.890668 m | 9 |

Case 052 selected two left arcs, one right arc and three right turns. At frame 29
all seven retained references (28, 27, 26, 23, 15, 7, 0) failed with zero surviving
lifted correspondences. The current corner frame had 48 detected, 40 liftable
features, occupying only three cells. The receipt-bound readout reconstructed
the same rejection for every reference using the unchanged detector, matching
and rigid-registration code. The maximum position error among accepted poses
was 1.329 mm; that does not provide a pose after rejection or a physical error
bound. No contact occurred. All ten zero intervals completed.

At frame 36, during that drain, the strict depth audit found one false public-valid
near ray. Stable-interior metric checks still passed (3,197 rays, maximum error
0.133 mm), but `near_occlusion_failure=true` makes this a hard measurement failure.
It must remain in the result; a stable-interior pass does not erase it. The failure
occurred after the controller had already latched its visual stop. Its precise
rendering/visibility cause has not yet been isolated.

Case 039 selected four right turns, four left arcs and one right arc. At tick 43
it selected a left arc with 0.343371 m predicted terminal progress, 0.086090 full
four-second contact score, and 0.000234965 first-half-second contact score. After
40 physics steps (80 ms), the front-left hip contacted `front_partial_panel` at
sample 2939, with recorded force magnitude 262.772 N. The external native guard
stopped the episode. These are uncalibrated model scores; the observation shows
a serious missed contact on the selected action, without establishing outcomes
for actions that were not executed. All recorded hard depth-measurement checks
passed in this case. Maximum accepted-pose position error was 1.924 mm.

Both raw sensor reconstruction and exact model/observer/command replay passed;
the checkpoint state remained unchanged. There were 84 camera frames, 5,640
physics samples and 15 online model selections. The native attempt took 203.192
seconds wall time and bound 402 artifacts totaling 139,617,246 bytes to 888 frozen
source paths. The separately frozen readout bound 892 sources and ran no native
simulation, optimizer or replacement controller.

| Recorded stage | Case 052 median / max (ms) | Case 039 median / max (ms) |
|---|---:|---:|
| Acquisition | 107.289 / 126.894 | 113.218 / 125.937 |
| Controller | 47.261 / 71.645 | 55.711 / 69.676 |
| Full iteration with command | 186.108 / 224.597 | 195.169 / 218.387 |

Full iterations exceeded 100 ms in 38/39 and 43/44 records respectively. These
distributions include warmup and, for 052, terminal drain; controller time is not
an isolated model or observer benchmark. No controller computation exceeded
100 ms, but acquisition plus control plus command execution did. Faster component
execution alone would not establish latency-aware physical control.

Artifacts reside under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/`.
Native root: `go2_family_transition_goal_probe_v1_attempt_001`.
Readout root: `go2_family_transition_goal_readout_v1_attempt_001`.

| Identity | SHA-256 |
|---|---|
| Native launch | `5d708568e8e1fb8d518d741d471172d0f718d7d158d130e98fe1fd6af0a55608` |
| Native result | `eb70bb999f78f063031575b2773e2871651f095cb583e0bd6fb4b94ec84bfa4e` |
| Readout launch | `7cb2cc1f15293e6ac682e7fcbf5662a884c7fff2c2ec96561584f0679404fd53` |
| Readout result | `39e697ea85ea582e7ecba7f36a853fcda5592d0531f2bf12ee89689bbebdfbe2` |
| Final full-JEPA snapshot | `bcb8874e2adf89053463206267a4ccb90380909c324e303734a59b038f5b1821` |

The source protocols are [native probe](go2_family_transition_goal_probe_v1_2026-09-08.md)
and [readout](go2_family_transition_goal_readout_v1_2026-09-08.md). All sources and
bound artifacts were verified before and after execution/readout. Preserve both
attempts; neither is a retry target.

The next development step is to address observation-grounded clearance and route
memory before another goal attempt. The current controller has a local terminal
distance score and an uncalibrated learned contact penalty, but no persistent
observed free-space map, frontier route or physical backtracking policy. Its
training windows contain remaining portions of fixed actions; arbitrary fresh
action switches during motion require separate support and validation. A new
planner must account for unknown/near-range space and the robot footprint, select
waypoints from observed evidence, and compare learned action prediction against
matched nonpredictive choices. Do not infer safe clearance from low contact scores,
weaken the visual gates, or fill missing poses by command integration.

In parallel with future source development, isolate the recorded frame-36 near-ray
failure without changing its original score, and profile acquisition stages before
redesigning packet delivery. Rendering and synchronous file serialization are
possible timing contributors based on source inspection, not yet measured causes.
Any new observer/planner or sensing implementation needs a separately frozen
protocol and complete-trace tests before fresh native execution. Independent maze
layouts, exploration, useful memory, physical backtracking, matched ablations and
realistic timing remain necessary for the active overall goal.
