# Fine stored obstacle geometry

At the first missing route in the continuous-connector/optional-plane native
experiment (pose frame 108, map frame 104), reconstruction of all 328,994
stored above-floor depth points exactly reproduced the 150 coarse occupied
cells. The nearest point was 0.477291 m from the estimated position. Its 5 cm
cell extended to 0.440396 m; a 1 cm representation of the same points extended
to 0.470396 m. The nominal footprint radius remains 0.45 m. The nearest point
originated in the primary camera at frame 76. No native state was used.
Evidence: `go2_continuous_connector_optional_plane_native_layout00_v1_attempt_001/stored_obstacle_resolution_diagnostic.json`.

The map now optionally retains a parallel 1 cm occupied-cell set built from
the exact same registered depth points and height band. FineStoredMap enables
this for the development runtime. The start connector uses continuous disk
distance against those fine squares. The coarse floor graph, its conservative
interior obstacle inflation, unknown connector reporting, footprint and current
depth dispatch checks remain unchanged. Fine/coarse coverage must agree before
route construction. Global fine-cell bounds represent the same 10 m square as
the original map; this does not enlarge the physical map.

Two focused tests passed in 1.93 s, covering clear fine-cell entry, actual disk
contact rejection, incomplete fine evidence and the earlier immediate
start-collision rejection. Full replay of all 450 planning observations found
no missing routes, versus 424 previously. Median/max route computation was
8.02/25.73 ms in this replay. These are retrospective geometry and computation
results, not prospective navigation or host real-time evidence.
Evidence: `fine_stored_route_switch_diagnostic.json` in the same directory.

Prospective native experiment:
`go2_fine_stored_obstacle_native_layout00_v1_attempt_001`, session 98416.
The experiment keeps the model, learned alignment scoring, action bank,
optional plane refinement, partial-height observations and 1,800-tick mission
budget. The experiment stopped with a visual-tracking failure after 346
acquired frames and 1,730 requested policy steps. All 346 camera pairs were
persisted; the process exited with failure. There was no physical contact stop.

There were 85 route-to-frontier plans, with no missing-route or view-recovery
plans; 84 met the deadline. Actions were 21 forward, 15 left arc, four right
arc, 25 left turn, ten right turn and ten hold. Final/minimum observed goal
distance was 1.559 m, down from the initial 2.6 m. Maximum native XY
displacement was 1.450 m. Post-estimation comparison of the 341 admitted poses
gave median/max position error 4.45/9.63 mm. There was no goal arrival.

This is prospective movement beyond the preceding map stall, not successful
navigation. Late in the run, route waypoint distance shrank from 6 cm at frame
244 to about 1.5 mm at frame 324; the planner continued selecting turns toward
that nearby frontier. Handling arrival at an exploration frontier remains a
separate navigation issue. The saved sensor trajectory is being replayed to
identify the new visual failure at the farther position. Replay completed 341
poses exactly matching the native estimator output, then reproduced the visual
failure at frame 341. The optional refinement was selected at frames 49, 112,
195, 196, 293 and 338; it is not the new failure category.

Exception-local inspection of the original robust registration at frame 341
found the best rejected fit retained 21/26 correspondences (80.77%), with
9.09 mm relative translation. Its reference image covered six grid cells and
current image five, below the six-cell rule. The accepted current pixels span
386.1 by 274.0 pixels; their covariance eigenvalues are 4,662 and 18,311 pixel
squared. Thus this particular candidate fails the discrete image-grid rule,
not match count, inlier fraction or translation. Other direct/chain attempts
had only four current grid cells. The frame's RGB images show the checker
wall close to the primary camera and floor/wall in the auxiliary view.
Evidence: `optional_plane_refinement_replay.json` and
`consensus_failure_diagnostic.json` in the fine-stored native directory.
No spatial-support gate has been changed. Next perception work should evaluate
the spatial-support rule or joint camera evidence; exploration also needs
explicit handling of a reached frontier rather than repeated turns toward a
millimetre-scale waypoint. No job remains running.
