# Qualified-overlap observer permits a near-goal approach, not verified arrival

The one fresh full-RGB direct-model case 039 reached minimum native distance
0.062001 m and ended 0.076555 m from the goal. Terminal initial-body XY was
[1.227707, 0.071365] m. Zero goals passed the unchanged arrival/quiet test.
This known development case does not establish independent-maze reliability.

The observer changed numeric pose at frame 13 and the first command at tick 38.
All 39 RGB frames, all six forecasts and physics/public body/gyro array prefixes
through that command decision exactly matched the predecessor. Observer/map
evidence differed as expected. The old tick-136 anchor-loss failure was avoided
on the new executed trajectory. All 228 subsequent accepted poses used anchors;
none used a bridge. There were 72 promotions: 69 half-overlap, two support-margin
and one qualified alternative-reference promotion.

At tick 229, both the retained-anchor fits and previous-frame fit failed because
too few rigid-pose matches remained. This is a different failure from exhausted
bridging with working incremental tracking. Accepted reference match counts fell
to 21, 19, 15, 19, 14 and 12 at frames 223–228. The last frame classified no floor
returns. Public depth at frames 228 and 229 was valid at all 307,200 pixels, with
ranges 0.204617–0.372717 m and 0.206779–0.373599 m, respectively. Missing depth
coverage is therefore not established as the cause of the feature loss.

Final waypoint scoring also targets [1.175, 0.025] m, the centre of the goal's
5-cm map cell, rather than the exact transformed mission point. At tick 223 the
observed robot was [1.240125, 0.052196] m in the initial body frame, 0.065836 m
from the actual goal. At tick 228 the newly occupied map cell changed the route
from the goal cell to a frontier. These facts motivate a separate exact-terminal
target diagnostic; they do not prove an unexecuted corrected controller arrives.

All 240 RGB-D frames passed strict visibility and hard measurement checks.
Fresh-model raw sensor-to-command replay passed, model states were unchanged,
and no physical stop or disallowed contact occurred. Maximum accepted observed
XY error was 3.290 mm. Every selected forecast passed the existing surface and
nominal constraints. There were 46 selections: eight right turns, one hold, two
right arcs, thirteen forwards, fourteen left arcs and eight left turns.

The run retained 239 commands and 12,700 native physics samples. Every complete
iteration exceeded 100 ms: median 447.798 ms, maximum 662.631 ms. Physics paused
during computation. Final available classification retained 4,312,588 returns,
including 1,222,293 floor and 3,090,295 other returns. Full post-launch work took
238.247 seconds. These remain simulation diagnostics with uncalibrated pose,
contact and future-gait uncertainty, not real-time or hardware evidence.

Twenty-five focused tests passed in 2.82 seconds. Preflight recorded 82.08 GB
available RAM, 85.33 GB artifact storage free, 0.2% CPU utilization and idle GPUs.
One CPU worker ran the one affected causal case. The native result binds 1,021
sources and 994 artifacts totaling 490,333,545 bytes; the readout binds 1,023
sources. All predecessor failures and artifacts remain unchanged.

| Root / artifact | SHA-256 |
|---|---|
| `go2_overlap_retention_goal_probe_v1_attempt_001/launch.json` | `4d2672407324c591be22bf2f57c73888a64d30b275a2fbf5bcf4fe2d5e562349` |
| `go2_overlap_retention_goal_probe_v1_attempt_001/result.json` | `fa67b645ae0439b803adce6d1a73c90e29862daeb8f8cd9b037183acd1477765` |
| `go2_overlap_retention_goal_readout_v1_attempt_001/launch.json` | `e8fac7554aebfd0212a8c8fd6ad7a11fe351327cc3aa0beade5d23fa4c1a4c75` |
| `go2_overlap_retention_goal_readout_v1_attempt_001/result.json` | `4f38b4fc8cca26808eb248db5d6484c5b68f29c73c2e14ea2f2a5dd82bc76799` |

Next, diagnose scoring against the exact mission point when the selected
waypoint is the terminal observed goal cell. Preserve every model prediction,
surface and nominal veto, observed route and arrival criterion. Continue the
separate investigation of near-wall feature loss. The full goal remains active.
