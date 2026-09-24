# Matched training controls queued on independent layout 0

The existing matched-fit study already names full-RGB supervised-rollout and
direct models as training controls for the primary full-RGB JEPA model. Those
two controls are now queued on the same fixed independent layout 0. The JEPA
case is still collecting; no completed independent navigation outcome has been
used to select these controls.

Execution order:

1. `seed_2026091001_full_jepa`, `frozen_reference` — currently executing.
2. `reactive`, no high-level model — existing queue.
3. `seed_2026091001_full_supervised_rollout`, `frozen_reference` — newly queued.
4. `seed_2026091001_full_direct`, `frozen_reference` — newly queued.

The learned arms share the first training seed, full-RGB input variant, existing
matched training data/schedule and 1,200-update fit budget. Their existing
training-only translation corrections are reused. This step neither trains new
models nor selects new checkpoints. The same current navigation runner retains
the scene/appearance seeds, sensor interfaces, low-level gait, planning and
memory implementation, 8,000-tick budget, stopping rule and physical evaluator.
Each controller generates its own closed-loop trajectory.

The comparison between learned arms addresses their training objectives within
this model family. The reactive comparison contrasts complete methods. Direct
training still produces an action-conditioned predictive model: it is not the
non-predictive baseline. This roster alone does not isolate online prediction
or persistent memory. Those ablations remain separate work.

All four cases are runs on **one maze**, not four independent maze replicates.
Further fixed layouts and training seeds remain necessary to assess reliability
and whether an observed advantage persists. Physics still pauses during
computation; no real-time or hardware qualification follows.

The queue processes only wait for exact predecessor process identities, require
operational completion, and then execute the existing one-case runner. They do
not require positive scientific results. Operational failures stop the queue
and preserve evidence. There are no automatic retries or existing-root writes.
Every native launch retains the runner's hardware assessment. Collection and
audit jobs stay serial to avoid interference with their filesystem-wide space
accounting.

| Case | Queue PID | Creation time | Tool session |
| --- | ---: | ---: | ---: |
| Supervised rollout | 3149828 | 1789261694.22 | 74097 |
| Direct | 3149894 | 1789261729.43 | 7463 |

Expected corrected model identities, to compare with actual launch records:

- Supervised rollout:
  `755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.
- Direct:
  `8cf81bec6c67261df9d25b56d2f6546abd887735ab8977021bb54a3c1963f1df`.

Exact process/assignment receipts are
`go2_independent_layout00_supervised_queue_2026-09-13.json` and
`go2_independent_layout00_direct_queue_2026-09-13.json`. Original matched model
assignments are in `scripts/all_phase_fit_execution_development.py`.
