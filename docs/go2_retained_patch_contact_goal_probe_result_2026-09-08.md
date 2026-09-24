# Retained patch contact probe: movement followed by anchor loss

The prospectively selected full-RGB direct-model case 039 reached zero verified
goals. It ended 1.250938 m from the goal, with minimum distance 1.115987 m and
terminal initial-body displacement [0.449036, -1.000449] m. This is one affected
development case, not a new matched cohort or independent-maze evaluation.

The first changed command was tick 78: a complete frame-45 depth patch covered
the forward candidate's left-front foot where the coarse floor grid did not.
All 79 observations through that decision, including RGB, observer/map evidence,
all six model forecasts, physics and public body/gyro array prefixes, exactly
match the predecessor. Seven candidate conflicts were removed on the actually
executed new trajectory. No unknown/non-floor or non-foot veto was removed.

The controller entered waypoint mode at tick 23 and continued moving after the
former hold region. Its 27 selections were six right turns, one hold, four right
arcs, three forwards, twelve left arcs and one left turn. At tick 136 it stopped
with SENSOR_OR_MODEL_FAILURE. The underlying cause was exhausted measured
visual bridging: frames 126–135 had frame-to-frame RGB-D estimates but no
qualified retained anchor. At 136 all eight anchors failed registration, while
the previous-frame fit still had 287 inliers across all twelve image cells.
The existing ten-frame bridge limit correctly prevented continued use.

The last qualified anchor observation was frame 125 against reference 118:
91 inliers, fraction 0.938144, eight reference and nine current image cells,
2.693-mm residual RMS. It did not trigger existing motion or absolute grid
support promotion. This evidence motivates examining loss of feature overlap
before an anchor becomes unusable. It does not justify relaxing registration,
promoting bridge-only poses, extending the bridge budget or resuming this run.

All 147 retained RGB-D frames passed the strict visibility and hard measurement
checks. Fresh-model raw sensor-to-command replay passed; model states remained
unchanged. There were no physical stops or disallowed contacts. Maximum accepted
observed XY error was 5.035 mm. All selected forecasts passed the existing
surface and nominal constraints. These checks do not certify future gait,
support, pose uncertainty or execution-time clearance.

There were 146 command intervals and 8,050 native physics samples. Every full
iteration exceeded 100 ms: median 442.903 ms, maximum 682.111 ms. Physics paused
for computation, so this remains slower-than-real-time simulation evidence.
The final available classification retained 2,558,185 returns: 736,721 floor and
1,821,464 other. At the last forecast, 134 patch prefixes occupied 164,659,200
bytes. Full post-launch work took 161.329 seconds.

Preflight found 82.10 GB available RAM, 85.65 GB free artifact storage and idle
GPUs. One CPU worker ran the single causal case. The native result binds 1,007
sources and 622 artifacts totaling 292,346,459 bytes; the readout binds 1,009
sources. Fifty focused tests passed together, followed by three integration
tests after adding the unknown-contact-veto case: 51 distinct tests in total.

| Root / artifact | SHA-256 |
|---|---|
| `go2_retained_patch_contact_goal_probe_v1_attempt_001/launch.json` | `9a17318bde6998f7f0c74ad1a123c79862255994f5d382ab62ae0840d4e3207a` |
| `go2_retained_patch_contact_goal_probe_v1_attempt_001/result.json` | `9842921b16b0fb2dffc501c6d958fbfd6ffb72e57a9c56855b2d5f9b54e61685` |
| `go2_retained_patch_contact_goal_readout_v1_attempt_001/launch.json` | `cda4112f4677237eef634c0f73fbe61d2f8cfbc595c201a0d7947cf851cc0b26` |
| `go2_retained_patch_contact_goal_readout_v1_attempt_001/result.json` | `dd2c39c3f791f19ab146e6d1a521d06546f0d72376771b4cb7e7cc7f6664c451` |

The full navigation goal remains active. The next bounded step is a separately
named observer replay that tests earlier retention of already anchor-qualified
views when their measured feature overlap falls, with unchanged acceptance and
bridge limits. Any changed observer needs fresh native execution afterward.
