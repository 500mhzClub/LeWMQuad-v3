# Prospective novel-maze round-trip scenes: source candidate

Four fixed seed-generated 4-by-4 spanning-tree mazes are implemented in
`lewm/novel_maze_round_trip_scene_development.py`. They contain 16 cells, 15
open graph edges and 25 physical wall boxes each. Cell pitch is 1.3 m, wall
thickness 0.08 m and height 1.4 m. Spawn is the centre of a single-exit cell
at world [-1.3,0], facing its opening. The instructed goal is the farthest
cell by graph distance; the mission then requires a return to the starting
coordinate with memory retained by a future controller integration.

Only goal and return coordinates plus the return requirement appear in the
public mission. Graph, wall geometry and shortest route are construction/
evaluation data. They are not free-space observations, commands or teacher
routes for the controller. No native simulator has executed these scenes.

| Index | Shortest outbound length | Graph turns | Topology SHA-256, invariant to grid rotations/reflections |
| --- | ---: | ---: | --- |
| 0 | 7.8 m | 5 | `fe9de58f9b859cd8732fd6da53aaab1ca07ede710f2f5906bb4bf0aa2bc4378a` |
| 1 | 16.9 m | 6 | `62c021dfd88e64c0b937dcff16bf7e4bf62eda52f5ca99008c2de8d9b52a0656` |
| 2 | 9.1 m | 4 | `8cd69ca89101f05418d64bebbfbaa997ab6a9e38fca4be3283573d1f3c370edf` |
| 3 | 9.1 m | 4 | `7b349ce7d6cd7554a36e02c6e17420cc3b58cd4167609ac8acf1a8c8694f2ead` |

Seven tests passed in 1.84 s: connected deterministic trees, topology uniqueness
and disjointness from the 24 earlier counterfactual plus eight online-choice
source-generated layouts, exact wall openings/nonedges, nominal radius-0.45-m
clearance along every graph-centre segment, bounded mission coordinates,
public/private field separation, scene-pack parameters and invalid indices.
The reviewed predecessor generators construct dictionaries without reading
runtime artifacts or protected material. This is source topology comparison,
not a blanket claim of disjointness from every historical scene in the repo.

The current native controller is still fixed to [1.2,0] and 240 navigation
ticks; its camera/replay sessions cap at 254 observations. Those limits are
not silently changed here. Before native use, explicitly integrate variable
mission coordinates, two arrival states, observation-budget bounds, physical
return evaluation, source/input custody and storage/hardware allowances.
Use identical scene/mission conditions across learned and nonpredictive
baselines, and freeze prospective evaluation choices before executing them.
No resampling based on controller results or qualification from these tests.
