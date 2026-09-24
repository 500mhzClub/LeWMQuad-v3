# Fixed controller on the three remaining development layouts

This prospective study executes layouts 1, 2 and 3 from the existing four-layout
maze definition, in that order, with the exact measured-floor-transport controller
and assigned full-JEPA model used in the current maze0 attempt. Prepare this
definition before seeing any of these three execution outcomes. They are
previously unexecuted development layouts according to the current experiment
record, not sealed benchmarks or a statistical guarantee of generalization.
Their geometry remains evaluator-only; only the existing coordinate mission and
actual sensor packets reach each fresh controller. Each case starts with empty
online memory and the same frozen checkpoint. No checkpoint training, selection,
controller adaptation, seed replacement or case omission between layouts.

Run scripts/run_go2_independent_floor_transport_mazes_v1.py using the actual
completed --native-result-sha256 from
go2_measured_floor_transport_maze_pilot_v1_attempt_001, first with
--preflight-only. The native predecessor must have completed its full raw
sensor/model/command replay and exact prospective intervention comparison.
A failed physical return or strict visibility result is preserved and does not
prevent testing the fixed controller on the other layouts. Infrastructure
failure or an incomplete predecessor does prevent admission. Do not launch this
cohort until that attempt has finished and there is no other live native scene.

Exclusive output: go2_independent_floor_transport_mazes_v1_attempt_001.
One worker and native scene at a time, a fresh spawned process per layout,
one OpenCV/PyTorch/BLAS thread. Keep the original 3000 navigation ticks shared
between outbound and return, three warmup observations and ten terminal zero
commands. Each episode uses the unchanged native collector and raw audit,
renderer acquisition witnesses, tracking/plane/transport gates, model, mapping,
action selection, geometry, gait, gains, friction and physical stop rules.
No retrospective maze0 physical-prefix equality requirement applies to different
layouts. The full new sensor-to-command history must replay exactly for each.

Assess hardware and competing processes before execution. Require 32 GiB
available RAM and, above the original 40 GiB reserve, the original 10 GiB
collection plus 1 GiB persistence allowance for every remaining case. Refresh
these measurements after input validation and before each case. These are
admission checks, not OS resource quotas. Monitor CPU/GPU, memory and storage
throughout. Completed cases release their RAM but their files remain. Resource
failure stops the partial cohort without deleting, overwriting or retrying it.

After each case persist its collection, raw audit, actual native outcome,
resource admission, logs and artifact hashes, plus cohort progress. Scientific
navigation failures remain in the denominator and the next fixed case executes.
Collection, persistence or raw-replay infrastructure failures stop the cohort
and retain the partial evidence. Do not automatically retry or skip a failed
layout. The aggregate result is complete only after all three fixed cases have
completed their raw audits, regardless of their scientific success count.

Use the original independent physical evaluation: settled outbound arrival,
settled return, physical route retracing, command and guard audits and strict
camera visibility. Preserve both physical candidate results and the stricter
verified-round-trip result. The wrapper changes only experiment-scope metadata
on the returned audit; every physical calculation and gate remains unchanged.
Zero successful cases is a valid negative scientific result.

This study supplies independent-layout execution evidence for one fixed model
and controller. It supplies no matched baseline, JEPA advantage, predictive
planning advantage, memory advantage, real-time qualification, calibrated pose
uncertainty or hardware validation. Physics remains paused during computation.
The full navigation goal remains active after this study unless every separate
goal requirement has subsequently been demonstrated.
