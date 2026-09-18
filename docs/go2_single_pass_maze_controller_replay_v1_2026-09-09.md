# Complete recorded controller replay with single-pass measured-bound queries

The component query benchmark570a3978c98b26d8c47d25a31f646c9278eae8221a86af2cd8ca6538e8991d79
completed36,864 queries per implementation with every receipt exact. Median
paired batch speed ratios were2.003319 and2.049575 in opposite execution orders.
This does not yet prove complete controller equivalence or controller speed.

Run a fresh complete replay of all1881 ninth-maze public observation pairs with
SinglePassLaterFloorController. Replace only the eight empty persistent
measured-bound indices with SinglePassMeasuredSampleBoundsIndex; insertion uses
the already tested packed-owned path, and queries use the new single enumeration
and batched Boolean box predicates while preserving scalar sphere arithmetic.
Current-frame SurfaceIndex, model, observer, map/mission, planner, contact policy,
residual memory, output labels and numeric settings remain unchanged.

Require every complete decision to equal the recorded ninth original, including
first terminal1870 and ten drain observations. Do not stop at an early successful
prefix. Save all candidate decisions and uninstrumented observe times, but make
no controlled speed claim from this replay. Revalidate original native artifacts,
completed component benchmark, source identities and model state before/after.
No training, simulator, original-attempt resume, native adoption or failure
relabeling. The controller uses the ninth original mission, not settling V2.

Exclusive output:go2_single_pass_maze_controller_replay_v1_attempt_001.
Run scripts/replay_go2_single_pass_maze_controller_v1.py --preflight-only first.
Assess topology/affinity, CPU/GPU/VRAM, RAM, competing work and both volumes.
One CPU replay and one numerical thread beside the single native settling scene.
Require8GiB available RAM and1GiB output above40GiB reserve, recorded as capacity
admission rather than enforced OS limits. Preserve any failure. No change to
the running native source, renderer or execution settings is authorized by this
performance replay. Navigation, timing, independent layouts, matched baseline/
ablation and hardware claims remain unproved.
