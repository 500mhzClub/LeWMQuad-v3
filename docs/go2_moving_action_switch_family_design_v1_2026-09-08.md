# Moving-action family collection design V1

This is the implementation design for the next development data collection.
Freeze its executable specification, collector, raw auditor, tests and launch
bindings before scientific collection. The full navigation objective remains
unchanged; collection is a prerequisite for testing better learned control.

Use the existing six candidate actions: hold, forward, left arc, right arc,
left turn and right turn. Cross all six one-second prefix actions with all six
four-second suffix plans on four canonical existing geometry clusters. This
defines 144 prospective cells, including 24 repeat-action and 120 changed-action
cells. Hold suffixes measure braking; hold prefixes measure departure from rest.
These are actual alternative executions from matched moving contexts, not labels
invented for unexecuted actions in a recorded policy trajectory.

The existing episode roster is shuffled, so numerical episode spacing does not
identify clusters. Source inspection gives these exact canonical geometry
witnesses, all `left_open`, appearance seed 2026090940 and original action `hold`:

| Geometry cluster | Existing episode identity | New data role |
|---|---|---|
| `cluster_00` | `family_episode_013` | train |
| `cluster_01` | `family_episode_083` | train |
| `cluster_02` | `family_episode_026` | geometry_transfer |
| `cluster_03` | `family_episode_003` | geometry_transfer |

The existing specification supplies geometry/initialization identity only; new
explicit branch specifications and command tapes define all prefix/suffix actions.
Retain clusters 0/1 for training and 2/3 for development geometry transfer.
All six suffix siblings of a prefix/geometry cell share one role. These reused
parameter clusters are not new independent mazes or final evaluation. Assert
these source assignments in the new executable specification before freezing.

Each complete case has the existing 1.5-second native settle, three zero-command
history warmup ticks, ten ticks of the selected prefix's initial command, forty
ticks of the selected suffix's exact existing candidate plan, and ten zero-command
drain ticks. Expected full lengths are 63 high-level commands, 64 RGB-D frames
and 3,900 physics samples. The branch observation is frame 13. Future targets
at eight 500-ms horizons derive from the executed suffix, with the unchanged
current-body XY/relative-yaw convention and contact-censored motion/images.
Past model input is the actual four-packet RGB/body/control history at the branch.

Use fresh CPU simulator processes, the strict existing family scene/render/depth/
gyro/actuator/geometry setup and 2-ms/50-Hz/100-ms timing. Preserve complete physics,
contact attribution, public sensor histories, RGB/depth, geometry/setup/friction
witnesses, command tapes and stopped outcomes. A prefix stop leaves that cell
unavailable; it does not authorize replacement, continuation or invented suffix
targets. Every planned cell remains in accounting.

Verify exact physical/public-history arrays, camera geometry and RGB/depth prefix
identities across all six suffix siblings through frame 13. No simulator-state
copy or checkpoint resume is needed: each prefix is executed anew from the same
frozen initial conditions. Do not substitute a tolerance after observing a
failed exact match. Keep benchmark copies distinct from scientific data.

Before the 144-cell collection, run a separately frozen serial-versus-four-worker
benchmark on four representative complete cases with exact paired outputs and
all raw checks. Select useful concurrency from measured throughput and equality;
the prior collection benchmark is supporting evidence, not a new measurement.
Assess CPU/RAM/GPU/storage and competing work before benchmarking and collection.
Plan a conservative 24-GiB scientific output allowance plus the 40-GiB artifact
reserve, and recheck it against measured benchmark bytes before launch. Preserve
all prior failures and source bindings; use new exclusive output roots.

Do not train from a partial or unaudited population. After complete collection,
derive typed causal branch/suffix inputs and freeze a separate matched learning
study with direct, supervised-rollout and JEPA objectives, full/no-predictor-RGB
inputs and multiple optimization seeds. Keep transfer observations past-only for
inference and transfer labels out of optimization. No checkpoint choice follows
from the preceding one-trajectory error ranking.

Subsequent progress still requires real closed-loop arrival, independent-maze
exploration, memory and physical backtracking, matched reactive/non-predictive
baselines, realistic sensing, computation during motion and bounded hardware
evidence. This data design grants none of those scientific conclusions.
