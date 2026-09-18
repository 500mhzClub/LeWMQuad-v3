# Second independent layout: unchanged JEPA and supervised comparison

The next fixed development inventory layout is index 1. Run these two cases
after the already queued layout-0 commitment-contact supervised case:

1. `frozen_reference`, `seed_2026091001_full_jepa`.
2. `frozen_reference`, `seed_2026091001_full_supervised_rollout`.

Use the existing `scripts/run_go2_stop_conditioned_independent_case_v1.py`.
Both retain the original 100 ms progress / 800 ms contact score, all original
geometry filters, action bank, persistent observed memory, sensors, gait,
8,000-decision allowance and stop-conditioned physical evaluator. Their model
identities are respectively
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`
and `755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.
Use the saved matched fits; do not retrain for this pair.

Layout 1 is chosen by the existing inventory order. At assignment, layout 0
has a verified JEPA round trip, a negative reactive result with a strict sensor
visibility failure, and an ongoing supervised run with near-start observations.
The direct, nominal and paired contact-horizon cases remain pending. This pair
is a development replication decided with that knowledge, not a blinded final
benchmark. Its settings do not depend on the remaining layout-0 outcomes.

Both new cases wait for operational completion of their exact predecessors,
including negative scientific outcomes. An operational failure stops the queue;
there is no automatic retry or replacement layout. Collection and audit remain
serial. The existing runner assesses CPU, RAM and disk at actual launch. At
queue preparation the active run reported about 62.5 GiB available RAM and
479.2 GiB artifact disk free; those observations are not launch admission for
the future cases. No additional preflight suite is introduced.

Exact queue owners and output paths are recorded in
`go2_independent_layout01_jepa_queue_2026-09-13.json` and
`go2_independent_layout01_supervised_rollout_queue_2026-09-13.json`.
Neither new native case has started at preparation. These cases will add one
maze replicate, taking the completed inventory to two layouts if they finish.
Two layouts alone do not establish reliability. Further replication, prediction
and memory ablations, continuous timing, realistic sensing and bounded hardware
evidence remain necessary. Physics still pauses during controller computation.
