# Sixth case physically reaches the outbound goal; return budget exhausted

The no-RGB direct-prediction case completed collection with 3,014 observations,
3,013 executed command intervals and 151,400 physics samples. It recorded one
outbound arrival at frame 2935 and switched to RETURN. It exhausted the global
navigation budget at frame 3003, then completed the ten zero-command drain
intervals. It did not complete a round trip.

The independently reconstructed native evaluator confirms the outbound dwell:
over the required one-second window, maximum goal distance was 0.034598 m and
maximum speed was 0.025618 m/s. There were zero contact samples in the complete
physics trace, 15 outbound cell crossings over ten distinct declared open
edges, no invalid crossing, and no recorded position outside the maze.
The loop-erased outbound path matches the evaluator's expected route. The
return phase traversed no additional cell edge.

The trace includes a physical excursion from `[-1, 1]` to `[-1, 2]` and back,
followed by a return through earlier cells and a different route to the goal.
The excursion cell has two declared neighbors, so it must not be called a
known dead end. This behavior alone does not prove a memory advantage over a
matched control.

At the 14 saved prefix snapshot frames, maximum XY difference between the
recorded visual position and native position was 0.019171 m. At the terminal
mission frame it was 0.008657 m. These are selected-frame comparisons, not the
maximum over the complete episode. The full original sensor/model, visibility
and command audit is still running under the original worker. In particular,
physical outbound arrival does not establish strict sensing validity or an
independent-layout navigation result. The audited episode count remains 42
pending the completed worker and parent receipts.

Collection SHA-256:
`c323046813ca032fc2012a5f2774aef2bebd5c09937da7d0feb70c5ce577109b`.
Physical readout:
`docs/go2_no_rgb_direct_maze02_collection_physical_readout_2026-09-11.json`,
SHA-256 `2bb7e7830dddfc60b665b7be320e102b2403379f9b6499bbc2de74b9d8cea07c`.
Session 86564 exited 0. It checked 1,908 sources and the completed collection,
physics, command, timing, specification and persistence-file bindings before
and after; reconstructed the native evaluator; and checked every command
endpoint, request, slew-limited application and phase against the physics
trace. It did not rerun model inference or the full recorded decision stream.

The return budget is an explicit limitation. At arrival the robot was 4.6662 m
from home, with at most 6.8 s until the budget cutoff. Under the existing native
step criterion (including tolerance), merely reaching the 6 cm home region
along a straight line would take at least 15.3515 s. This lower bound excludes
walls, turns, gait limitations and the additional required quiet dwell. A
qualifying return was therefore impossible within the remaining budget.

Budget diagnosis:
`docs/go2_no_rgb_direct_maze02_return_budget_diagnosis_2026-09-11.json`,
SHA-256 `329a40f728f28c0f91d9c904416c3cccfaee960a0cea8c3b0a146158bc33f75e`.
Session 33937 exited 0. The preceding readout attempt 95295 exited 1 when its
incorrect dead-end assumption was rejected; it wrote no artifact. The
corrected record retains that failure and the actual two-neighbor topology.

After the full audit and fixed comparison review, prospective follow-up should
test faster outbound planning or a budget that permits the observed return
task. A longer budget would not itself prove return success. This case uses
the learned direct-outcome head with its RGB-input ablation; RGB remains in
the tracking/mapping frontend. It establishes no JEPA or learned-RGB benefit.
The original case, deadlines, outcome and queued experiments remain unchanged.
