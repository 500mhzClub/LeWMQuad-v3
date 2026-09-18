# Sixth case: movement without observed arrival through frame 1999

The fixed first 2,000 recorded observations were read twice identically while
the original sixth-case worker remained live. The first 800 rows also match
the previously saved canonical prefix identity. All 1,908 original source
bindings, the original launch identity and the prior prefix record were
verified. Execution session 34824 exited 0.

| Observation window | First goal distance | Last goal distance | Minimum goal distance |
| --- | ---: | ---: | ---: |
| 0–799 | 4.687 m | 3.989 m | 3.294 m |
| 800–1199 | 3.992 m | 3.491 m | 3.484 m |
| 1200–1599 | 3.492 m | 4.006 m | 3.220 m |
| 1600–1999 | 4.005 m | 3.324 m | 3.324 m |

There are no observed arrivals in these windows. The final registered visual
position is approximately `[2.450, -0.391]` m in the initial body reference.
Selected actions comprise 682 left arcs, 107 right arcs, 758 right turns,
139 left turns, 305 holds and three forward actions. Six observations have no
selected action, including the three warmup observations. Requested commands
and selected-action counts are distinct quantities.

This confirms sustained changes in observed position, with substantial
variation in distance to the goal. A maze can require detours, so these
distances alone do not prove an incorrect route. The movement is not yet a
verified physical traversal or navigation success. Collection, raw physics,
contact and sensor/model audit are incomplete; the completed audit must assess
physical progress and localization accuracy before the case informs the
independent-study controller choice. No additional native experiment is
launched or selected from this live prefix.

Record:
`docs/go2_no_rgb_direct_maze02_live_2000_observation_prefix_2026-09-11.json`,
SHA-256 `1d995486505b58272082773a446427ecb365668762f22976461bfb86f5270362`.
The record includes full prefix identities, bounded-window counts, snapshots
and the final saved selection. Reading stopped after frame 1999 without
attempting to read the growing gzip stream to its end. This analysis ran no
model inference and issued no physical command.
