# Current controller memory scope audit

The current controller has no working persistent=False ablation. A direct
constructor check in session8986 completed successfully using model=None and
geometry=None, without loading a model, observation or native scene:

- persistent=False fails with ValueError:
  "active-view probe requires persistent surface evidence".
- persistent=True constructs MeasuredFloorTransportController at tick-1,
  with zero route entries/floor cells/occupied cells and no loaded model.

Runtime method-resolution inspection8078 located this guard in
lewm/matched_model_goal_probe_development.py:32. It remains inherited by the
current controller. Do not call persistent=False an implemented comparator,
an executed experiment, or evidence for a memory contribution.

The source trace establishes several distinct uses of retained observations:

| State | Current use | Consequence for a comparison |
| --- | --- | --- |
| Observed floor and occupied cells | Accumulate via setdefault in FrameCachedFloorMap; used for route proposals and nominal path checks | Changing only a surface query flag leaves planning-map memory present |
| Primary and auxiliary returns and partitions | Accumulate in all-return and classified indices | Switching one primary query would not remove auxiliary or typed contact history |
| Retained floor patches and later floor evidence | Support contact classification and resolving earlier foot ambiguity using later observations | Clearing these changes contact evidence and cannot be silently presented as only a route-memory change |
| Paired-camera visual tracking and measured floor anchor | Maintain the deployment-visible reference pose across observations | Removing this history changes state estimation, beyond a planning-memory comparison |
| Four-frame learned history and recent executed residuals | Supply learned forecasts and causal correction | These are temporal prediction state, distinct from persistent spatial planning memory |
| Mission and settling state | Preserve outbound/return phase and measured quiet intervals | Resetting these would change the task and arrival rule |

MissionTargetWaypointSelector.choose explicitly calls the surface filter with
persistent=True at line68. Current FrameCachedFloorMap retains floor/occupied
cells at lines167/171, and classifications at lines36/72/105. LaterResolvedFloorMap
records paired floor evidence at line99. MeasuredFloorTransportController.advance
keeps the map, tracker, learned history and residuals across the return
transition, changing only target-specific selector state. These sources do not
establish a memory advantage; they identify the implementation that must be
controlled in an experiment.

A separately named planning-map persistence ablation can isolate retained
floor/obstacle routing information while preserving the same tracker,
registration, learned history, residuals, contact evidence, mission and budget.
It must build current-view floor and obstacle cells from the actual paired
observations. Filtering accumulated cells by first-witness frame is incorrect:
setdefault retains the earliest frame even when a cell is observed again.
Both waypoint proposals and subsequent nominal/reentry checks must use the
same explicitly scoped map view. Its report must say that contact and
localization histories remain; it is not a fully memoryless controller.

A full spatial-evidence ablation would additionally require a separately
defined treatment of all primary/auxiliary indices, typed partitions, patches
and later-contact evidence, with contemporaneous classification and accounting
still consistent. It must not claim an isolated route-memory effect or erase
state needed to verify the observation clock. Neither ablation is implemented
or executed by this audit. Any successor needs causal prefix verification and
fresh outcomes after its first changed command. Fixed queued comparisons and
all existing frozen/running sources remain unchanged.

Inspected source SHA-256 identities:

- lewm/matched_model_goal_probe_development.py:
  e4492fbecc2b71b9b6c720bf436c5b27efae2cbeee9d2923d12fd1bdc46db57f
- lewm/mission_target_waypoint_selection_development.py:
  8fbb466da3b8cdb86d4f8387491fde59452b69ea97b6b7ebd1ac4465577bbef1
- lewm/frame_cached_floor_map_development.py:
  b3cf5d43aab4840ac120dfb6daf9291444b9595359acc2f0456014f85f61734c
- lewm/later_floor_resolution_controller_development.py:
  cb9954bd24735761135338e8f813e645e8bb595470414aa5f4b710c03c8eb79c
- lewm/measured_floor_transport_controller_development.py:
  40fcbb64318ae5887ca23cfbc870636b0eec68ecd9f21918413bb1671dd9b2ce

Hardware inspection51478 completed:16physical/32logical CPUs with full affinity,
3.4%CPU busy,72,108,052,480bytes availableRAM,95,329,415,168bytes artifact free,
21,360,402,432bytes workspace free; both GPUs0%busy. Current native audit worker
2447506 remained active at98.8%CPU after77m51s,CPU76m56s,RSS10,646,680KiB.
This is available capacity, not a reason to restart or change that attempt.
Refresh resources before the next substantial job. No new job was launched.
