# Fully nonpredictive measured-plane maze-02 development comparison

Execute one fresh reactive episode after the exact existing learned episode and
the registered nominal predictive comparison finish with complete raw audits.
Scientific success is not an admission requirement for either predecessor.
Execution or audit failure is retained and does not authorize automatic bypass,
retry or source changes. The separate reactive prefix completed at frame 3:
learned left arc `[0.16, 0, 0.45]`, reactive forward `[0.2, 0, 0]`, with identical
full measured observation, map and mission receipts and zero reactive model calls.

The reactive controller uses the same measured-plane perception, floor
registration, persistent observation map, public mission, robot, renderer,
sensor contracts, scene 02, command cadence, 4,000-tick budget, settle and drain
intervals, environment and resource limits. It loads no high-level world model,
does not use learned residual correction, and evaluates no candidate future
outcomes. The pretrained low-level locomotion policy remains the common physical
actuation system. This is a whole-method comparison, not an isolated test of
predictive ranking: current-geometry feasibility and reactive route selection
replace predictive gates and scoring.

The existing reactive collector, complete raw controller replay and command
audit retain their original function code with private substitutions for the
extended budget, bounded sensor dependencies and measured-plane constructor.
There is no modification of any live or completed predecessor source. The first
900 physics samples, all four initial public packets, both complete controller
decisions, and the first three requested commands must match the prospective
prefix. The changed fourth command must actually complete its 50 physics steps.
Later physical outcomes belong to the reactive episode and are not inferred from
the learned recording.

One scene worker runs after the original nominal waiter ends and its complete
result, child worker, artifacts and learned predecessor are authenticated.
Available RAM must be at least 32 GiB and artifact space at least 55 GiB before
dispatch. Collection retains the 14 GiB allowance and all original reserve and
persistence checks. CPU OpenCV/BLAS remain single-threaded. Physics pauses during
controller computation, so this study does not establish realtime execution.

Every collected command must reproduce from raw RGB/depth/gyro evidence. Retain
the full physical contact, sensor visibility, requested/applied/slewed command,
renderer witness, timing, observed progress, return and backtracking evidence.
Round-trip success requires the same physical outcome, strict visibility and
absence of hard measurement failures as the learned and nominal episodes. An
early failure keeps its full audit without claiming an executed intervention.

Output: `go2_reactive_measured_plane_maze02_v1_attempt_001` under the existing
development artifact root. This single reused development layout does not prove
independent-maze generalization, JEPA advantage, isolated planning or memory
contributions, realistic latency, hardware readiness or completion of the goal.
