# Observed floor registration controller candidate

This separately named development controller addresses pose/floor registration
drift throughout the online pipeline. It is not a completed navigation result.
No launched predecessor source, observation, classification, or outcome is edited.

## Evidence and correction

`lewm/floor_pose_registration_development.py` extracts stride-four candidates
only where all four surrounding measured ground quads and all nine pixels pass
the existing geometric/validity tests. It does not use the initial absolute
10 mm floor-height band. Each camera independently needs at least 100 candidates,
second covariance eigenvalue at least 0.0025 m², up alignment at least 0.97, and
maximum residual at most 3 mm. No points are trimmed to make a fit pass.
The two planes, expressed in the same body coordinates, must agree within
3 mm in offset and 0.01 radians in normal direction. These are development
admission thresholds, not calibrated physical error bounds.

The initial reference is frozen from the first paired observation. Initial up
comes from the existing quiet public specific-force history. Later candidate
extraction uses that initial up transformed by the independently validated raw
visual rotation. No native pose, segmentation, geometry, or later outcome enters
the correction. A static, shared flat-floor identity remains a hypothesis.

The correction aligns the current measured normal with the initial normal,
restores the raw visual forward azimuth within that plane, and adjusts position
only along the initial normal. Raw in-plane position and heading are preserved.
Corrections above 5 cm or 0.10 radians stop admission; they are not uncertainty
bounds. Lack of either measured plane, disagreement, missing packets, and
chronology failures latch a stop rather than extrapolate a floor observation.

`lewm/floor_registered_evidence_development.py` retains the entire original
visual evidence under its own unchanged contract. The corrected pose has a new
schema and `floor_registered_joint` mode. A separate accessor checks both the
original joint-fit witness and the correction composition. Fresh raw replay must
also reconstruct the image-derived fits and frozen reference chronology.

## All pose consumers

`lewm/floor_registered_controller_development.py` uses that accessor for surface
insertion, the executed residual estimator, and mission advancement. Both camera
partitions, retained patches, map coverage, obstacle cells, footprint checks and
waypoint selection consume the same surface pose. The visual tracker itself
remains unchanged, with its complete raw witness retained. Historical maps are
accumulated from each contemporary corrected pose; no later map rewrite occurs.
The confirmed-floor policy, nominal constraints, model, input history and action
set remain inherited. This is a pose-estimator intervention, not just a change
to one auxiliary floor mask.

The three narrow consumer methods derive from these launch-bound predecessor
files; AST comparison allows only the pose accessor substitution:

- `lewm/joint_visual_surface_memory_development.py`:
  `32761d47ea6308bb48dd5594743bfc0e523eb680613c2a9b7ae9bfc73bcb9fdb`
- `lewm/online_executed_residual_development.py`:
  `3a358b2bbf399349f4e8e29d8ff7ca6e5b767d43e518439d0f8cc4000cee4896`
- `lewm/observed_round_trip_controller_development.py`:
  `4e98af43ae963737b639f3cb566e3e3f490c700c73c9c45f470e6eafc4ce3fc6`

## Initial checks and limitations

Eleven geometry tests passed. Eleven further consumer/evidence tests passed,
including actual two-frame synthetic public packets through the controller/map,
shared corrected pose, unchanged raw evidence, complete return accounting,
duplicate/missing packet stops, tamper rejection and narrow method comparison.
These are synthetic integration checks, not a native robot or model run.

Read-only inspection session 90294 completed on the fixed collection and stream
of `go2_confirmed_floor_maze_pilot_v1_attempt_001/full_jepa_novel_maze_00`:
collection SHA-256
`8241e3753fe952701bee956632257eeca95fd40b4b0b50126ec18325c665829a`,
stream SHA-256
`cfbd2ef3396eff6a3c548a32a55dd4b6b52ee9b49eed000b26cfe06538c352a4`.
Both hashes were verified before and after. This inspection used public packets
and recorded visual evidence only, at frames 0, 595, 909, 1420, 1466, 1469,
1474, 1482 and 1537. Both cameras admitted compatible planes at all nine frames.
Initial primary/auxiliary candidates numbered 6783/19145. At terminal frame
1537 they numbered 3611/15940, maximum fit residuals 9.210/11.007 micrometres,
intercamera angle 0.00000544318 radians and offset difference 3.144 micrometres.
The terminal proposed normal translation correction was -0.0126429027882 m,
normal alignment 0.00326729291987 radians. No commands or classifications were
replayed or changed by that inspection. The collection's final audit was still
running when this candidate was prepared, so it remains provisional evidence.

Sensor angular/depth bounds, pose uncertainty and floor identity are still
uncalibrated. Plane consistency does not certify support or free space and
cannot clear the predecessor's strict visibility failure. No verified arrival,
return, independent-layout, realtime or hardware evidence is added here.

Next: complete the existing native audit/readout, run the separately bound
first-intervention replay, then review that result before a fresh native run.
